"""Namespaces and vocabulary normalisation.

The published data is not fully consistent: persona properties appear under both
``pcv:`` and ``saref:``, and operators are spelled EQUAL/EQUALS/In/LESSER_THAN.
Everything is normalised here so the rest of the code sees one vocabulary.
"""
from rdflib import Namespace

DATA = Namespace("http://piirrite.univ-lyon1.fr/data#")
PCV = Namespace("http://piirrite.univ-lyon1.fr/vocabulary#")
PIIRRITE = Namespace("http://piirrite.univ-lyon1.fr/ontology/core#")
SAREF = Namespace("https://saref.etsi.org/core/")
GEO = Namespace("http://www.opengis.net/ont/geosparql#")
TIME = Namespace("http://www.w3.org/2006/time#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

COMPARISON_OPS = {"EQUALS", "LESSER_THAN", "GREATER_THAN"}
LOGICAL_OPS = {"AND", "OR", "NOT"}

_OP_ALIASES = {
    "EQUAL": "EQUALS", "EQUALS": "EQUALS", "EQ": "EQUALS", "IN": "EQUALS", "==": "EQUALS", "=": "EQUALS",
    "LESSER_THAN": "LESSER_THAN", "LESS_THAN": "LESSER_THAN", "LT": "LESSER_THAN", "<": "LESSER_THAN",
    "GREATER_THAN": "GREATER_THAN", "GT": "GREATER_THAN", ">": "GREATER_THAN",
    "AND": "AND", "OR": "OR", "NOT": "NOT",
}


# Properties computed from the data rather than stored as saref:PropertyValue.
DERIVED = {
    "D_STAIRS": "true if the path is a staircase (handrail, step_count or incline=up/down)",
    "D_INDOOR": "true if the path is indoors (feature kind K_Indoor) or covered",
    "D_ELEVATOR": "true if the path is an elevator",
    "D_TRAM": "'yes' on tram tracks / tram stops (use with a distance factor)",
    "D_SLOPE": "absolute incline in percent, parsed from P_Incline",
}

# Persona properties found in ABox.ttl that have no data under that name,
# mapped onto something the data does contain.
ALIASES = {
    "P_STAIRS_PRESENT": "D_STAIRS",
    "P_INDOOR": "D_INDOOR",
    "P_TRAMWAY": "D_TRAM",
    "P_SLOPE": "D_SLOPE",
}

# Properties that describe places to visit rather than path segments: a
# requirement on them (not negated) becomes a waypoint ("go via a bin").
WAYPOINT_PROPS = {"P_AMENITY", "P_BIN", "P_WC_ACCESS", "P_NAME", "P_VENDING", "P_CHARGING_POINT", "P_LEISURE"}


def resolve_prop(name: str) -> str:
    key = norm_prop(name)
    return ALIASES.get(key, key)


def local_name(iri) -> str:
    s = str(iri)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[1]
    return s


def norm_op(name: str) -> str:
    key = local_name(name).upper()
    if key not in _OP_ALIASES:
        raise ValueError(f"unknown operator {name!r}")
    return _OP_ALIASES[key]


def norm_prop(name: str) -> str:
    """Canonical property key: local name, case-insensitive, with the P_ prefix."""
    n = local_name(name)
    if not n.upper().startswith(("P_", "D_")):
        n = "P_" + n
    return n.upper()
