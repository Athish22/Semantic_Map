"""Integration tests against the real PIIRRITE knowledge graph (slower; uses cached parses)."""
import logging

import pytest
from rdflib import Graph
from rdflib.compare import isomorphic

from piirrite_route import kg, personas as P

logging.getLogger("rdflib").setLevel(logging.ERROR)

A, B = (45.7800, 4.8650), (45.7845, 4.8790)


@pytest.fixture(scope="module")
def router():
    from piirrite_route.route import Router
    return Router()


def test_existing_personas_load():
    ids = {p.id for p in P.load_all()}
    assert {"Anna", "Dan", "Sacha", "Colin", "Meriem", "Francois"} <= ids
    anna = P.get("Anna")
    noise = next(r for r in anna.requirements if r.label == "Peu de bruit")
    assert (noise.importance, noise.flexibility) == (0.6, 0.4)
    assert "NOT([P_Tramway = yes] within <100 m)" in noise.describe()


def test_anna_roundtrip_is_isomorphic():
    """export -> YAML dict -> Turtle -> parse reproduces Anna's requirement structure."""
    anna = P.get("Anna")
    rebuilt = P.from_dict(P.to_dict(anna))
    g1, g2 = P.to_graph(anna), Graph()
    g2.parse(data=P.to_turtle(rebuilt), format="turtle")
    assert isomorphic(g1, g2)


def _vcq_graph():
    g = Graph()
    for f in kg.persona_files():
        g.parse(f, format="turtle")
    return g


def test_paper_vcq9_finds_distance_factor_requirements():
    """Paper listing 2 (namespaces as in the published data) over the added personas."""
    q = """PREFIX piirrite: <http://piirrite.univ-lyon1.fr/ontology/core#>
    SELECT DISTINCT ?user ?requirement WHERE {
      ?user a piirrite:UserProfile ; piirrite:hasRequirement ?requirement .
      ?requirement piirrite:hasCondition ?condition .
      ?condition (piirrite:hasCondition|piirrite:hasOperand)* ?atomic .
      ?atomic a piirrite:AtomicCondition ; piirrite:hasDistanceFactor ?df .
      ?df a piirrite:DistanceFactor . }"""
    if not kg.persona_files():
        pytest.skip("no added personas")
    users = {str(r.user).rsplit("#", 1)[1] for r in _vcq_graph().query(q)}
    expected = {p.id for p in P.load_all() if p.source.startswith("personas/") and any(
        a.distance for r in p.requirements for c in r.conditions for a in P.iter_atomics(c))}
    assert users == expected


def test_vcq10_depth_query_runs():
    q = """PREFIX piirrite: <http://piirrite.univ-lyon1.fr/ontology/core#>
    SELECT ?root (COUNT(DISTINCT ?any) AS ?n) WHERE {
      ?u a piirrite:UserProfile ; piirrite:hasRequirement/piirrite:hasCondition ?root .
      ?root piirrite:hasOperand* ?any } GROUP BY ?root"""
    if not kg.persona_files():
        pytest.skip("no added personas")
    assert len(list(_vcq_graph().query(q))) > 0


def test_hard_constraint_prunes_stairs(router):
    lea = P.from_dict({"id": "T", "requirements": [
        {"importance": "INF", "flexibility": 0, "condition": {"property": "D_STAIRS", "op": "EQUALS", "value": False}}]})
    s, e = router.snap(*A)[0], router.snap(*B)[0]
    rt = router.route(lea, s, e)
    assert not any(router.G[u][v]["props"].get("D_STAIRS") == [True] for u, v in zip(rt.nodes, rt.nodes[1:]))
    assert rt.stats[0].pruned > 0
    assert rt.length >= router.evaluate_path(lea, router.shortest(s, e)).length


def test_distance_factor_moves_route_away_from_tram(router):
    p = P.from_dict({"id": "T", "requirements": [{"importance": 1.0, "flexibility": 0.2, "condition": {
        "NOT": {"property": "D_TRAM", "op": "EQUALS", "value": "yes", "distance": {"op": "LESSER_THAN", "metres": 50}}}}]})
    s, e = router.snap(*A)[0], router.snap(*B)[0]
    near_tram_route = router.route(p, s, e).stats[0].bad_m
    near_tram_shortest = router.evaluate_path(p, router.shortest(s, e)).stats[0].bad_m
    assert near_tram_route < near_tram_shortest


def test_time_dependent_weather_changes_cost(router):
    from piirrite_route import network as N
    series = sorted((x for f in N.load().features for x in f.timed.get("P_WEATHER_CONDITION", [])),
                    key=lambda x: x[0])
    rain = next(t for t, _, v in series if v == "rain")
    clear = next(t for t, _, v in series if v == "clear")
    p = P.from_dict({"id": "T", "requirements": [{"importance": 0.8, "flexibility": 0.3, "condition": {"OR": [
        {"property": "D_INDOOR", "op": "EQUALS", "value": True},
        {"NOT": {"property": "P_weather_condition", "op": "EQUALS", "value": "rain"}}]}}]})
    s, e = router.snap(*A)[0], router.snap(*B)[0]
    assert router.route(p, s, e, rain).stats[0].bad_m > 0
    assert router.route(p, s, e, clear).stats[0].bad_m == 0
