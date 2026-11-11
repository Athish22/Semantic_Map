"""Fast unit tests: inadequacy functions, condition logic, persona I/O and validation."""
import math

import pytest
from rdflib import Graph

from piirrite_route import personas as P
from piirrite_route.cost import Context, compare
from piirrite_route.personas import Atomic, Composed, Requirement

VOCAB = {"P_NOISE", "P_LIT", "P_SMOOTHNESS", "D_STAIRS", "D_TRAM", "P_WEATHER_CONDITION", "D_INDOOR"}


# --------------------------------------------------------------- paper properties (i)-(iv)
@pytest.mark.parametrize("op,req,vals", [
    ("LESSER_THAN", 50, [40, 50.5, 52, 55, 60, 80]),
    ("GREATER_THAN", 7, [9, 6.5, 6, 5, 2]),
    ("EQUALS", 24, [24, 25, 27, 30, 40]),
])
def test_numeric_inadequacy_properties(op, req, vals):
    for f in (0.0, 0.25, 0.5, 1.0):
        scores = [compare("P_X", op, [req], v, f)[0] for v in vals]
        assert all(0.0 <= s <= 1.0 for s in scores)                      # (i)
        assert scores[0] == 0.0                                           # (ii) condition met
        assert scores == sorted(scores)                                   # (iii) monotone in violation
    for v in vals[1:]:                                                    # (iv) less flexible -> higher
        s = [compare("P_X", op, [req], v, f)[0] for f in (1.0, 0.5, 0.25, 0.0)]
        assert s == sorted(s)


def test_paper_noise_example_is_fully_inadequate():
    # Anna: noise < 50 dB, f = 0.25, path at 57.3 dB -> 1.0 (paper, section 4.2)
    assert compare("P_NOISE", "LESSER_THAN", [50], 57.3, 0.25)[0] == 1.0


def test_ordinal_smoothness_means_at_least():
    assert compare("P_SMOOTHNESS", "EQUALS", ["good"], "excellent", 0.0)[0] == 0.0
    assert compare("P_SMOOTHNESS", "EQUALS", ["good"], "intermediate", 0.0)[0] == 1.0
    assert compare("P_SMOOTHNESS", "EQUALS", ["good"], "intermediate", 1.0)[0] == pytest.approx(1 / 3)


def test_categorical_and_boolean():
    assert compare("P_LIT", "EQUALS", ["yes"], "yes", 0.5)[0] == 0.0
    assert compare("P_LIT", "EQUALS", ["yes"], "no", 0.5)[0] == 1.0
    assert compare("D_STAIRS", "EQUALS", [False], False, 0)[0] == 0.0
    assert compare("D_STAIRS", "EQUALS", [False], "no", 0)[0] == 0.0
    assert compare("D_STAIRS", "EQUALS", [False], True, 0)[0] == 1.0


# --------------------------------------------------------------- logic
def edge(**props):
    return {"props": {k: v if isinstance(v, list) else [v] for k, v in props.items()}, "near": {}}


def test_composed_logic():
    ctx = Context(net=None, when=None)
    lit = Atomic("P_Lit", "EQUALS", ["yes"])
    indoor = Atomic("D_INDOOR", "EQUALS", [True])
    rain = Atomic("P_weather_condition", "EQUALS", ["rain", "thunderstorm"])
    shelter = Composed("OR", [indoor, Composed("NOT", [rain])])
    assert ctx.evaluate(shelter, edge(D_INDOOR=False, P_WEATHER_CONDITION="rain"), 0)[0] == 1.0
    assert ctx.evaluate(shelter, edge(D_INDOOR=True, P_WEATHER_CONDITION="rain"), 0)[0] == 0.0
    assert ctx.evaluate(shelter, edge(D_INDOOR=False, P_WEATHER_CONDITION="clear"), 0)[0] == 0.0
    both = Composed("AND", [lit, indoor])
    assert ctx.evaluate(both, edge(P_LIT="yes", D_INDOOR=False), 0)[0] == 1.0
    # unknown operands are skipped; all-unknown -> None
    assert ctx.evaluate(both, edge(P_LIT="yes", D_INDOOR=True, P_WEATHER_CONDITION="x"), 0)[0] == 0.0
    assert ctx.evaluate(Atomic("P_Lit", "EQUALS", ["yes"]), edge(D_INDOOR=True), 0) is None


def test_categorical_requirement_scaled_by_flexibility():
    ctx = Context(net=None, when=None)
    strict = Requirement(0.8, 0.0, [Atomic("P_Lit", "EQUALS", ["yes"])])
    loose = Requirement(0.8, 1.0, [Atomic("P_Lit", "EQUALS", ["yes"])])
    e = edge(P_LIT="no")
    assert ctx.requirement(strict, e) == 1.0
    assert ctx.requirement(loose, e) == 0.5


# --------------------------------------------------------------- persona I/O
LEA = {
    "id": "Lea", "label": "Wheelchair user", "category": "mobility",
    "requirements": [
        {"label": "No stairs", "importance": "INF", "flexibility": 0.0,
         "condition": {"property": "D_STAIRS", "op": "EQUALS", "value": False}},
        {"label": "Away from tram", "importance": 0.6, "flexibility": 0.4,
         "condition": {"NOT": {"property": "D_TRAM", "op": "EQUALS", "value": "yes",
                               "distance": {"op": "LESSER_THAN", "metres": 30.0}}}},
        {"label": "Smooth", "importance": 0.7, "flexibility": 0.4,
         "condition": {"property": "P_Smoothness", "op": "EQUALS", "value": "good"}},
    ],
}


def test_yaml_turtle_roundtrip():
    p = P.from_dict(LEA)
    g = Graph()
    g.parse(data=P.to_turtle(p), format="turtle")
    (back,) = P.parse_personas(g)
    def canon(d):   # RDF has no requirement order
        return {**d, "requirements": sorted(d["requirements"], key=lambda r: r["label"])}
    assert canon(P.to_dict(back)) == canon(P.to_dict(p))
    assert any(math.isinf(r.importance) for r in back.requirements)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["requirements"][2]["condition"].update(property="P_Nope"), "unknown property"),
    (lambda d: d["requirements"][1].update(flexibility=1.5), "flexibility 1.5"),
    (lambda d: d["requirements"][1].update(importance=2), "importance 2"),
    (lambda d: d["requirements"][1].update(condition={"NOT": [
        {"property": "P_Lit", "op": "EQUALS", "value": "yes"},
        {"property": "P_Lit", "op": "EQUALS", "value": "no"}]}), "exactly one operand"),
    (lambda d: d["requirements"][2]["condition"].update(op="LESSER_THAN"), "numeric value"),
    (lambda d: d.update(id="9lives"), "must start with a letter"),
])
def test_validation_errors(mutate, needle):
    import copy
    d = copy.deepcopy(LEA)
    mutate(d)
    errors, _ = P.validate(P.from_dict(d), vocab=VOCAB)
    assert any(needle in e for e in errors), errors


def test_validation_ok_and_duplicate_id():
    p = P.from_dict(LEA)
    assert P.validate(p, vocab=VOCAB) == ([], [])
    errors, _ = P.validate(p, vocab=VOCAB, existing_ids={"lea"})
    assert any("already exists" in e for e in errors)


def test_missing_condition_is_rejected():
    d = {"id": "X", "requirements": [{"label": "oops", "importance": 0.5, "flexibility": 0.5}]}
    with pytest.raises(ValueError, match="no condition"):
        P.from_dict(d)


def test_depth_matches_paper_vcq10_definition():
    a, b, c = (Atomic("P_Lit", "EQUALS", [v]) for v in "abc")
    assert P.depth(Composed("AND", [a, b])) == 1
    assert P.depth(Composed("AND", [Composed("NOT", [Composed("OR", [a, b])]), c])) == 3
