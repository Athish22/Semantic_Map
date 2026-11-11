"""Persona model: read from RDF, convert to/from YAML, write Turtle, validate.

The RDF pattern follows PIIRRITE modelets 4-5 as instantiated in ABox.ttl:

    :Anna a piirrite:UserProfile ;
        piirrite:hasRequirement [
            piirrite:hasImportance  [ piirrite:hasValue 0.6 ] ;
            piirrite:hasFlexibility [ piirrite:hasValue 0.4 ] ;
            piirrite:hasCondition [ a piirrite:ComposedCondition ;
                piirrite:usesOperator piirrite:AND ;
                piirrite:hasOperand [ a piirrite:AtomicCondition ;
                    piirrite:hasTargetProperty pcv:P_Noise ;
                    piirrite:hasRequiredValue 50 ;
                    piirrite:usesOperator piirrite:LESSER_THAN ] , ... ] ] .
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from . import kg
from .ns import (COMPARISON_OPS, DATA, DERIVED, LOGICAL_OPS, PCV, PIIRRITE, SAREF, SKOS,
                 local_name, norm_op, norm_prop, resolve_prop)

Value = Union[str, float, int, bool]


# --------------------------------------------------------------------------- model
@dataclass
class DistanceFactor:
    op: str            # comparison operator on the distance
    metres: float
    kind: str = "From bird's flight"


@dataclass
class Atomic:
    prop: str          # property local name as written, e.g. "P_Noise"
    op: str            # EQUALS | LESSER_THAN | GREATER_THAN
    values: list[Value]
    distance: DistanceFactor | None = None

    @property
    def key(self) -> str:   # canonical key used to look the property up in the data
        return resolve_prop(self.prop)

    def describe(self) -> str:
        sym = {"EQUALS": "=", "LESSER_THAN": "<", "GREATER_THAN": ">"}[self.op]
        val = self.values[0] if len(self.values) == 1 else "{" + ", ".join(map(str, self.values)) + "}"
        s = f"{self.prop} {sym} {val}"
        if self.distance:
            dsym = {"EQUALS": "=", "LESSER_THAN": "<", "GREATER_THAN": ">"}[self.distance.op]
            s = f"[{s}] within {dsym}{self.distance.metres:g} m"
        return s


@dataclass
class Composed:
    op: str            # AND | OR | NOT
    operands: list["Condition"]

    def describe(self) -> str:
        if self.op == "NOT":
            return f"NOT({self.operands[0].describe()})"
        return "(" + f" {self.op} ".join(o.describe() for o in self.operands) + ")"


Condition = Union[Atomic, Composed]


@dataclass
class Requirement:
    importance: float  # may be math.inf ("INF" in the data = must-have)
    flexibility: float
    conditions: list[Condition]   # several top-level conditions = implicit AND
    label: str = ""

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.conditions)

    def is_hard(self, imp_threshold: float = 0.9, flex_threshold: float = 0.1) -> bool:
        return math.isinf(self.importance) or (
            self.importance >= imp_threshold and self.flexibility <= flex_threshold)

    @property
    def weight(self) -> float:   # importance used in the weighted sum
        return 1.0 if math.isinf(self.importance) else self.importance


@dataclass
class Persona:
    id: str
    requirements: list[Requirement]
    label: str = ""
    category: str = ""
    source: str = ""
    notes: list[str] = field(default_factory=list)


def iter_atomics(cond: Condition):
    if isinstance(cond, Atomic):
        yield cond
    else:
        for o in cond.operands:
            yield from iter_atomics(o)


def depth(cond: Condition) -> int:
    """Nesting depth of composed conditions (paper's VCQ10: 'A and B' -> 1)."""
    if isinstance(cond, Atomic):
        return 0
    return 1 + max(depth(o) for o in cond.operands)


# --------------------------------------------------------------------------- RDF -> model
def _py(lit) -> Value:
    if isinstance(lit, Literal):
        dt = lit.datatype
        text = str(lit)
        if dt == XSD.boolean:
            return text.lower() == "true"
        if dt in (XSD.integer, XSD.int, XSD.double, XSD.float, XSD.decimal):
            if text.upper() in ("INF", "+INF"):
                return math.inf
            try:
                f = float(text)
            except ValueError:
                return text
            return int(f) if f.is_integer() and dt in (XSD.integer, XSD.int) else f
        return text
    return local_name(lit)


def _num(g: Graph, node, pred) -> float:
    wrapper = g.value(node, pred)
    if wrapper is None:
        raise ValueError(f"requirement without {local_name(pred)}")
    v = _py(g.value(wrapper, PIIRRITE.hasValue))
    return float(v)


def _parse_condition(g: Graph, node) -> Condition:
    ops = [norm_op(o) for o in g.objects(node, PIIRRITE.usesOperator)]
    operands = list(g.objects(node, PIIRRITE.hasOperand))
    if operands or (ops and ops[0] in LOGICAL_OPS) or (node, RDF.type, PIIRRITE.ComposedCondition) in g:
        if not ops:
            raise ValueError("composed condition without operator")
        kids = sorted((_parse_condition(g, o) for o in operands), key=lambda c: c.describe())
        return Composed(ops[0], kids)
    prop = g.value(node, PIIRRITE.hasTargetProperty)
    if prop is None:
        raise ValueError("atomic condition without target property")
    values = sorted((_py(v) for v in g.objects(node, PIIRRITE.hasRequiredValue)), key=str)
    df = None
    dfn = g.value(node, PIIRRITE.hasDistanceFactor)
    if dfn is not None:
        df = DistanceFactor(norm_op(g.value(dfn, PIIRRITE.usesOperator)),
                            float(_py(g.value(dfn, PIIRRITE.hasRequiredValue))),
                            str(g.value(dfn, PIIRRITE.distanceType) or "From bird's flight"))
    return Atomic(local_name(prop), ops[0] if ops else "EQUALS", values, df)


def _comment_labels(path: Path) -> dict[str, list[tuple[str, set[str]]]]:
    """Recover the '# comment' labels of requirements from the Turtle source.

    rdflib drops comments, so match each commented block to a requirement by
    the set of target properties it mentions."""
    out: dict[str, list[tuple[str, set[str]]]] = {}
    current = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\n(?=:\w+ a piirrite:UserProfile)", text):
        m = re.match(r":(\w+) a piirrite:UserProfile", block)
        if not m:
            continue
        current = m.group(1)
        block = block.split("\n\n")[0]
        chunks = re.split(r"#\s*", block)[1:]
        out[current] = [(c.splitlines()[0].strip(),
                         {norm_prop(p) for p in re.findall(r"hasTargetProperty \w+:(\w+)", c)})
                        for c in chunks]
    return out


def parse_personas(g: Graph, source: str = "", labels=None) -> list[Persona]:
    personas = []
    for s in sorted(g.subjects(RDF.type, PIIRRITE.UserProfile), key=str):
        pid = local_name(s)
        reqs, notes = [], []
        for r in g.objects(s, PIIRRITE.hasRequirement):
            try:
                conds = [_parse_condition(g, c) for c in
                         list(g.objects(r, PIIRRITE.hasCondition)) + list(g.objects(r, PIIRRITE.hasAtomicCondition))]
                req = Requirement(_num(g, r, PIIRRITE.hasImportance), _num(g, r, PIIRRITE.hasFlexibility),
                                  sorted(conds, key=lambda c: c.describe()), str(g.value(r, RDFS.label) or ""))
            except ValueError as exc:
                notes.append(f"skipped a requirement: {exc}")
                continue
            if not req.label and labels and pid in labels:
                props = {norm_prop(a.prop) for c in req.conditions for a in iter_atomics(c)}
                for text, lprops in labels[pid]:
                    if lprops == props:
                        req.label = text
                        break
            reqs.append(req)
        reqs.sort(key=lambda r: (r.label, r.describe()))
        personas.append(Persona(pid, reqs, str(g.value(s, RDFS.label) or ""),
                                str(g.value(s, RDFS.comment) or ""),
                                source, notes))
    return personas


def load_all() -> list[Persona]:
    out: list[Persona] = []
    seen = set()
    for source, g in kg.persona_sources():
        labels = _comment_labels(kg.DATA_DIR / "ABox.ttl") if source == "ABox.ttl" else None
        for p in parse_personas(g, source, labels):
            if p.id in seen:
                p.notes.append(f"duplicate id {p.id!r}; earlier definition wins")
                continue
            seen.add(p.id)
            out.append(p)
    return out


def get(pid: str) -> Persona:
    for p in load_all():
        if p.id.lower() == pid.lower():
            return p
    raise KeyError(f"no persona {pid!r}; known: {', '.join(p.id for p in load_all())}")


# --------------------------------------------------------------------------- YAML <-> model
def _imp_out(x: float):
    return "INF" if math.isinf(x) else x


def condition_to_dict(c: Condition) -> dict:
    if isinstance(c, Composed):
        ops = [condition_to_dict(o) for o in c.operands]
        return {c.op: ops[0] if c.op == "NOT" else ops}
    d: dict = {"property": c.prop, "op": c.op,
               "value": c.values[0] if len(c.values) == 1 else list(c.values)}
    if c.distance:
        d["distance"] = {"op": c.distance.op, "metres": c.distance.metres}
    return d


def to_dict(p: Persona) -> dict:
    reqs = []
    for r in p.requirements:
        conds = [condition_to_dict(c) for c in r.conditions]
        reqs.append({"label": r.label, "importance": _imp_out(r.importance), "flexibility": r.flexibility,
                     "condition": conds[0] if len(conds) == 1 else {"AND": conds}})
    return {"id": p.id, "label": p.label, "category": p.category, "requirements": reqs}


def condition_from_dict(d) -> Condition:
    if not isinstance(d, dict):
        raise ValueError(f"condition must be a mapping, got {d!r}")
    logical = [k for k in d if str(k).upper() in LOGICAL_OPS]
    if logical:
        if len(d) != 1:
            raise ValueError(f"a logical condition must have exactly one key, got {list(d)}")
        op = str(logical[0]).upper()
        raw = d[logical[0]]
        raw = raw if isinstance(raw, list) else [raw]
        return Composed(op, [condition_from_dict(x) for x in raw])
    if "property" not in d:
        raise ValueError(f"atomic condition needs 'property': {d!r}")
    vals = d.get("value", d.get("values"))
    if vals is None:
        raise ValueError(f"condition on {d['property']} needs a 'value'")
    vals = vals if isinstance(vals, list) else [vals]
    df = None
    if "distance" in d:
        dd = d["distance"]
        df = DistanceFactor(norm_op(dd.get("op", "LESSER_THAN")), float(dd.get("metres", dd.get("value"))))
    return Atomic(str(d["property"]), norm_op(d.get("op", "EQUALS")), vals, df)


def _imp_in(x) -> float:
    if isinstance(x, str) and x.strip().upper() in ("INF", "MUST", "HARD"):
        return math.inf
    return float(x)


def from_dict(d: dict, source: str = "") -> Persona:
    for key in ("id", "requirements"):
        if key not in d:
            raise ValueError(f"persona is missing {key!r}")
    reqs = []
    for i, r in enumerate(d["requirements"] or []):
        if "condition" not in r and "conditions" not in r:
            raise ValueError(f"requirement #{i + 1} ({r.get('label', '')}) has no condition")
        raw = r.get("condition", r.get("conditions"))
        raw = raw if isinstance(raw, list) else [raw]
        reqs.append(Requirement(_imp_in(r.get("importance", 0.5)), float(r.get("flexibility", 0.5)),
                                [condition_from_dict(c) for c in raw], str(r.get("label", ""))))
    return Persona(str(d["id"]), reqs, str(d.get("label", "")), str(d.get("category", "")), source)


# --------------------------------------------------------------------------- model -> Turtle
def _lit(v: Value) -> Literal:
    if isinstance(v, bool):
        return Literal(v, datatype=XSD.boolean)
    if isinstance(v, int):
        return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):
        if math.isinf(v):
            return Literal("INF", datatype=XSD.double)
        return Literal(repr(v), datatype=XSD.decimal)    # serialises as 0.6, not 6e-01
    return Literal(str(v), datatype=XSD.string)


def _prop_iri(name: str) -> URIRef:
    return PCV[local_name(name)]


def _emit_condition(g: Graph, c: Condition):
    n = BNode()
    if isinstance(c, Composed):
        g.add((n, RDF.type, PIIRRITE.ComposedCondition))
        g.add((n, PIIRRITE.usesOperator, PIIRRITE[c.op]))
        for o in c.operands:
            g.add((n, PIIRRITE.hasOperand, _emit_condition(g, o)))
        return n
    g.add((n, RDF.type, PIIRRITE.AtomicCondition))
    g.add((n, PIIRRITE.hasTargetProperty, _prop_iri(c.prop)))
    g.add((n, PIIRRITE.usesOperator, PIIRRITE[c.op]))
    for v in c.values:
        g.add((n, PIIRRITE.hasRequiredValue, _lit(v)))
    if c.distance:
        dn = BNode()
        g.add((n, PIIRRITE.hasDistanceFactor, dn))
        g.add((dn, RDF.type, PIIRRITE.DistanceFactor))
        g.add((dn, PIIRRITE.distanceType, Literal(c.distance.kind)))
        g.add((dn, PIIRRITE.hasRequiredValue, _lit(c.distance.metres)))
        g.add((dn, PIIRRITE.usesOperator, PIIRRITE[c.distance.op]))
    return n


def to_graph(p: Persona) -> Graph:
    g = Graph()
    g.bind("", DATA)
    g.bind("piirrite", PIIRRITE)
    g.bind("pcv", PCV)
    g.bind("saref", SAREF)
    s = DATA[p.id]
    g.add((s, RDF.type, PIIRRITE.UserProfile))
    if p.label:
        g.add((s, RDFS.label, Literal(p.label)))
    if p.category:
        g.add((s, RDFS.comment, Literal(p.category)))
    for r in p.requirements:
        rn = BNode()
        g.add((s, PIIRRITE.hasRequirement, rn))
        if r.label:
            g.add((rn, RDFS.label, Literal(r.label)))
        for pred, val in ((PIIRRITE.hasImportance, r.importance), (PIIRRITE.hasFlexibility, r.flexibility)):
            w = BNode()
            g.add((rn, pred, w))
            g.add((w, PIIRRITE.hasValue, _lit(float(val))))
        for c in r.conditions:
            g.add((rn, PIIRRITE.hasCondition, _emit_condition(g, c)))
    return g


def to_turtle(p: Persona) -> str:
    return to_graph(p).serialize(format="turtle")


# --------------------------------------------------------------------------- validation
def vocabulary() -> set[str]:
    """Canonical keys of every property defined in GoT.ttl, plus derived ones."""
    g = kg.got()
    keys = {norm_prop(s) for s in g.subjects(RDF.type, SAREF.Property)}
    keys |= {norm_prop(s) for s in g.subjects(SKOS.inScheme, PCV.Properties)}
    return keys | set(DERIVED)


def validate(p: Persona, vocab: set[str] | None = None, coverage: dict | None = None,
             existing_ids: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). `coverage` maps property key -> {value: count} in the data."""
    errors, warnings = [], []
    vocab = vocab if vocab is not None else vocabulary()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", p.id):
        errors.append(f"id {p.id!r} must start with a letter and contain only letters, digits, _")
    if existing_ids and p.id.lower() in {i.lower() for i in existing_ids}:
        errors.append(f"a persona called {p.id!r} already exists")
    if not p.requirements:
        errors.append("persona has no requirements")

    def check(c: Condition, where: str):
        if isinstance(c, Composed):
            if c.op == "NOT" and len(c.operands) != 1:
                errors.append(f"{where}: NOT takes exactly one operand, got {len(c.operands)}")
            if c.op in ("AND", "OR"):
                if not c.operands:
                    errors.append(f"{where}: {c.op} without operands")
                elif len(c.operands) == 1:
                    warnings.append(f"{where}: {c.op} with a single operand")
            for o in c.operands:
                check(o, where)
            return
        if c.op not in COMPARISON_OPS:
            errors.append(f"{where}: {c.op} is not a comparison operator")
        if not c.values:
            errors.append(f"{where}: {c.prop} has no required value")
        if c.op in ("LESSER_THAN", "GREATER_THAN"):
            if len(c.values) != 1 or isinstance(c.values[0], bool) or not _is_number(c.values[0]):
                errors.append(f"{where}: {c.op} on {c.prop} needs one numeric value, got {c.values}")
        if c.key not in vocab:
            errors.append(f"{where}: unknown property {c.prop!r} (not in GoT.ttl; see `cli.py properties`)")
        elif coverage is not None:
            seen = coverage.get(c.key, {})
            if not seen:
                warnings.append(f"{where}: no entity in the data has {c.prop}; it will always be 'unknown'")
            elif c.op == "EQUALS" and not all(_is_number(v) and not isinstance(v, bool) for v in c.values):
                missing = [v for v in c.values if str(v).lower() not in {str(k).lower() for k in seen}]
                if missing:
                    warnings.append(f"{where}: value(s) {missing} never occur for {c.prop} "
                                    f"(seen: {', '.join(sorted(map(str, seen))[:8])})")
        if c.distance and (c.distance.metres <= 0 or c.distance.op not in COMPARISON_OPS):
            errors.append(f"{where}: distance factor needs a comparison operator and a positive distance")

    for i, r in enumerate(p.requirements, 1):
        where = f"requirement {i} ({r.label or r.describe()})"
        if not (0 <= r.importance <= 1 or math.isinf(r.importance)):
            errors.append(f"{where}: importance {r.importance} outside [0,1] (use INF for must-have)")
        if not 0 <= r.flexibility <= 1:
            errors.append(f"{where}: flexibility {r.flexibility} outside [0,1]")
        if not r.conditions:
            errors.append(f"{where}: needs at least one condition")
        for c in r.conditions:
            check(c, where)
    return errors, warnings


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def save(p: Persona, directory: Path | None = None) -> Path:
    directory = directory or kg.PERSONA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{p.id}.ttl"
    path.write_text(to_turtle(p), encoding="utf-8")
    return path
