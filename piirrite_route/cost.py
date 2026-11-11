"""Matching persona requirements against edge context -> inadequacy in [0, 1].

Inadequacy functions satisfy the paper's four properties:
 (i) values in [0,1]; (ii) 0 when the condition holds; (iii) grows as the
 condition is less satisfied; (iv) grows faster when flexibility is lower.

* numeric  LESSER_THAN / GREATER_THAN: smoothstep over a tolerance band of width
  w = 0.4 * f * max(|R|, 1)   (noise R=50 dB, f=0.25 -> 5 dB, as in the paper)
* numeric  EQUALS: linear over the same band (paper's temperature function)
* ordinal  (smoothness, wheelchair): EQUALS means "at least", band = 3f levels
* categorical / boolean: crisp 0/1, then scaled by (1 - f/2) at requirement level
* logic: AND = max, OR = min, NOT = 1 - x; unknown operands are skipped
* distance factor: distance d* to the nearest matching entity; for "within < D"
  the atom holds when d* < D and fades out over a band of 0.4 * f * D beyond.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from shapely import STRtree

from .network import Feature, Network
from .ns import WAYPOINT_PROPS
from .personas import Atomic, Composed, Condition, Persona, Requirement, iter_atomics

ORDINAL = {
    "P_SMOOTHNESS": {"impassable": 0, "very_horrible": 1, "horrible": 2, "very_bad": 3, "bad": 4,
                     "intermediate": 5, "good": 6, "excellent": 7},
    "P_WHEELCHAIR": {"no": 0, "limited": 1, "yes": 2, "designated": 3},
}
TRUE = {"true", "yes", "1", "designated"}
FALSE = {"false", "no", "0", "none"}
ROUTE_LEVEL = {"P_PATH_LENGTH"}      # properties of a whole route, not of an edge


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        return float(str(v).strip().rstrip("%"))
    except ValueError:
        return None


def _bool(v):
    s = str(v).strip().lower()
    return True if s in TRUE else False if s in FALSE else None


def compare(key: str, op: str, required: list, actual, flex: float) -> tuple[float, bool]:
    """Violation of one atomic comparison for a single actual value -> (degree, graded?)."""
    scale = ORDINAL.get(key)
    if scale and str(actual).lower() in scale and all(str(r).lower() in scale for r in required):
        a = scale[str(actual).lower()]
        w = 3 * flex
        best = 1.0
        for r in required:
            r = scale[str(r).lower()]
            excess = {"EQUALS": max(0, r - a), "GREATER_THAN": max(0, r + 1 - a),
                      "LESSER_THAN": max(0, a - r + 1)}[op]
            best = min(best, 0.0 if excess == 0 else (1.0 if w == 0 else min(1.0, excess / w)))
        return best, True
    a = _num(actual)
    if a is not None and all(_num(r) is not None for r in required):
        best = 1.0
        for r in map(_num, required):
            w = 0.4 * flex * max(abs(r), 1.0)
            if op == "EQUALS":
                e = abs(a - r)
                v = 0.0 if e == 0 else (1.0 if w == 0 else min(1.0, e / w))
            else:
                e = max(0.0, a - r) if op == "LESSER_THAN" else max(0.0, r - a)
                if op == "LESSER_THAN" and a >= r and e == 0:   # strict inequality
                    e = 1e-9
                v = 0.0 if e == 0 else (1.0 if w == 0 else smoothstep(e / w))
            best = min(best, v)
        return best, True
    if op != "EQUALS":
        return 1.0, False
    ab = _bool(actual)
    for r in required:
        rb = _bool(r)
        if (ab is not None and rb is not None and ab == rb) or str(r).strip().lower() == str(actual).strip().lower():
            return 0.0, False
    return 1.0, False


def feature_matches(f: Feature, a: Atomic) -> bool:
    for v in f.props.get(a.key, []) + [x for *_, x in f.timed.get(a.key, [])]:
        if compare(a.key, a.op, a.values, v, 0.0)[0] == 0.0:
            return True
    return False


@dataclass
class Context:
    """Evaluates conditions on network edges for one persona at one time."""
    net: Network
    when: datetime | None
    _df: dict = field(default_factory=dict)        # id(atomic) -> (STRtree, n matches)
    _area: dict = field(default_factory=dict)      # (edge path, key) -> values

    # -- value lookup ---------------------------------------------------------
    def values(self, edge: dict, key: str):
        vals = edge["props"].get(key)
        if vals is None and key in WAYPOINT_PROPS:
            vals = edge["near"].get(key)
        if vals is None:
            vals = self._area_values(edge, key)
        return vals

    def _area_values(self, edge: dict, key: str):
        if self.net is None or "geom" not in edge:
            return None
        ck = (edge.get("path"), id(edge["geom"]), key)
        if ck in self._area:
            return self._area[ck]
        tree, areas = self.net.areas()
        mid = edge["geom"].interpolate(0.5, normalized=True)
        out, series = [], []
        for i in tree.query(mid, predicate="intersects"):
            ar = areas[i]
            if key in ar.props:
                out += ar.props[key]
            series += ar.timed.get(key, [])
        if series:            # time-dependent: one value valid at `when`, pooled over overlapping areas
            out.append(self._at_time(series))
        self._area[ck] = out or None
        return self._area[ck]

    def _at_time(self, series):
        if self.when is None:
            return sorted(series, key=lambda s: s[0] or datetime.min)[0][2]
        inside = [(s, v) for s, e, v in series if s and e and s <= self.when < e]   # half-open: hourly
        if inside:                                                                  # intervals share ends
            return max(inside, key=lambda x: x[0])[1]
        return min(series, key=lambda s: abs(((s[0] or self.when) - self.when).total_seconds()))[2]

    # -- distance factors -------------------------------------------------------
    def nearest_match(self, edge: dict, a: Atomic) -> float | None:
        if id(a) not in self._df:
            hits = [f.geom for f in self.net.features if feature_matches(f, a)]
            self._df[id(a)] = (STRtree(hits) if hits else None, len(hits))
        tree, n = self._df[id(a)]
        if tree is None:
            return None
        limit = a.distance.metres * 3 + 1
        idx, dist = tree.query_nearest(edge["geom"], max_distance=limit, return_distance=True)
        return float(dist.min()) if len(dist) else math.inf

    # -- evaluation -------------------------------------------------------------
    def evaluate(self, c: Condition, edge: dict, flex: float):
        """-> (violation in [0,1], graded) or None if unknown."""
        if isinstance(c, Composed):
            kids = [self.evaluate(o, edge, flex) for o in c.operands]
            known = [k for k in kids if k is not None]
            if not known:
                return None
            graded = any(g for _, g in known)
            if c.op == "NOT":
                return 1.0 - known[0][0], graded
            vals = [v for v, _ in known]
            return (max(vals) if c.op == "AND" else min(vals)), graded
        if c.distance:
            d = self.nearest_match(edge, c)
            if d is None:
                return None
            D, w = c.distance.metres, 0.4 * flex * c.distance.metres
            if c.distance.op == "LESSER_THAN":
                v = 0.0 if d < D else (1.0 if w == 0 else smoothstep((d - D) / w))
            elif c.distance.op == "GREATER_THAN":
                v = 0.0 if d > D else (1.0 if w == 0 else smoothstep((D - d) / w))
            else:
                v = 0.0 if abs(d - D) < 1 else (1.0 if w == 0 else min(1.0, abs(d - D) / w))
            return v, w > 0
        vals = self.values(edge, c.key)
        if not vals:
            return None
        results = [compare(c.key, c.op, c.values, v, flex) for v in vals]
        return min(results, key=lambda r: r[0])

    def requirement(self, r: Requirement, edge: dict):
        """Inadequacy of one requirement on one edge, or None if unknown."""
        parts = [self.evaluate(c, edge, r.flexibility) for c in r.conditions]
        known = [p for p in parts if p is not None]
        if not known:
            return None
        v = max(p[0] for p in known)
        graded = any(p[1] for p in known)
        return v if graded else v * (1 - 0.5 * r.flexibility)


# ------------------------------------------------------------------------- classification
def is_waypoint(r: Requirement) -> bool:
    """'Go via X' requirements: un-negated EQUALS on place-like properties."""
    return bool(r.conditions) and all(
        isinstance(c, Atomic) and c.key in WAYPOINT_PROPS and c.op == "EQUALS" and not c.distance
        for c in r.conditions)


def classify(p: Persona, net: Network) -> dict[str, list]:
    """Split requirements into edge / waypoint / unsupported (with reasons)."""
    out = {"edge": [], "waypoint": [], "unsupported": []}
    for r in p.requirements:
        atoms = [a for c in r.conditions for a in iter_atomics(c)]
        if any(a.key in ROUTE_LEVEL for a in atoms):
            out["unsupported"].append((r, "constrains the whole route length, not individual edges"))
            continue
        missing = [a.prop for a in atoms if not net.coverage.get(a.key) and a.key != "P_WC_ACCESS"]
        if len(missing) == len(atoms):
            out["unsupported"].append((r, f"no entity in the data carries {', '.join(sorted(set(missing)))}"))
            continue
        (out["waypoint"] if is_waypoint(r) else out["edge"]).append(r)
    return out
