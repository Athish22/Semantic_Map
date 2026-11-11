"""Walkable network extracted from the ABox, plus spatial indexes for context lookups.

* Nodes: path vertices shared by >1 path (or path ends), keyed by (x, y, level).
  Keying on the level stops indoor corridors on different floors from merging.
  Multi-level paths (stairs/elevators, ``level=0;1``) are copied onto every
  level they serve and joined by vertical links that carry the path's attributes.
* Edges: segments between nodes, each carrying the source path's properties.
* Coordinates are projected to local metres (equirectangular; fine at campus scale).

Extraction iterates triples directly; rdflib's SPARQL engine is far too slow for
bulk scans of the 144k-triple ABox.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

import networkx as nx
from rdflib import RDF
from shapely import STRtree, wkt
from shapely.geometry import LineString, Point
from shapely.ops import transform

from . import kg
from .ns import GEO, PCV, PIIRRITE, SAREF, TIME, local_name, norm_prop

LAT0, LON0 = 45.781, 4.870          # projection origin (La Doua campus)
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110540
SNAP = 0.5                           # metres, vertex snapping tolerance
CACHE_VERSION = 5


@dataclass(frozen=True)
class Proj:
    """Local equirectangular projection (metres) around an origin; fine over a few km."""
    lat0: float
    lon0: float

    @property
    def kx(self) -> float:
        return 111320 * math.cos(math.radians(self.lat0))

    def to_m(self, lon, lat):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * KY

    def to_ll(self, x, y):
        return x / self.kx + self.lon0, y / KY + self.lat0


CAMPUS = Proj(LAT0, LON0)


def to_m(lon, lat):
    return CAMPUS.to_m(lon, lat)


def to_ll(x, y):
    return CAMPUS.to_ll(x, y)


def proj_of(G) -> Proj:
    """Projection of a graph's coordinates (campus graphs cached before this existed have none)."""
    return G.graph.get("proj", CAMPUS)


@dataclass
class Feature:
    """Any spatial entity (POI, polyline, area, path) with its property values."""
    iri: str
    cls: str
    geom: object                                     # shapely geometry in metres
    kinds: set[str] = field(default_factory=set)
    props: dict[str, list] = field(default_factory=dict)        # key -> static values
    timed: dict[str, list] = field(default_factory=dict)        # key -> [(start, end, value)]


@dataclass
class Network:
    graph: nx.Graph
    features: list[Feature]
    paths: dict[str, Feature]
    coverage: dict[str, Counter]                     # property key -> value counts (all features)
    time_range: tuple[datetime, datetime] | None
    proj: Proj = CAMPUS
    kind: str = "campus"
    _area_tree: STRtree | None = None
    _areas: list[Feature] | None = None

    # -- context lookups -------------------------------------------------------------
    def areas(self) -> tuple[STRtree, list[Feature]]:
        if self._area_tree is None:
            self._areas = [f for f in self.features if f.cls == "AreaOfInterest" and (f.timed or f.props)
                           and f.geom.geom_type in ("Polygon", "MultiPolygon")]
            self._area_tree = STRtree([a.geom for a in self._areas])
        return self._area_tree, self._areas

    def add_features(self, feats: list[Feature]) -> None:
        """Add non-path features (e.g. weather areas) after construction."""
        self.features.extend(feats)
        for f in feats:
            for k, vs in list(f.props.items()) + [(k, [v for *_, v in tv]) for k, tv in f.timed.items()]:
                for v in vs:
                    self.coverage.setdefault(k, Counter())[str(v)] += 1
        self._area_tree = self._areas = None

    def matching(self, predicate) -> list[Feature]:
        return [f for f in self.features if predicate(f)]


# ------------------------------------------------------------------------- extraction
def _parse_dt(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text)[:26])
    except ValueError:
        return None


def _interval(g, iv):
    def instant(pred):
        node = g.value(iv, pred)
        if node is None:
            return None
        v = g.value(node, TIME.inXSDDateTimeStamp) or g.value(node, TIME.inXSDDate)
        return _parse_dt(v) if v is not None else None
    start, end = instant(TIME.hasBeginning), instant(TIME.hasEnd)
    if start and end and start == end:          # daily values (inXSDDate): valid all day
        end = start.replace(hour=23, minute=59, second=59)
    return start, end


def _levels(values) -> list[float]:
    out = []
    for v in values or ["0"]:
        for part in re.split(r"[;,]", str(v)):
            part = part.strip()
            m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", part)
            try:
                if m:
                    a, b = float(m.group(1)), float(m.group(2))
                    out += [float(x) for x in range(int(math.floor(a)), int(math.ceil(b)) + 1)]
                else:
                    out.append(float(part))
            except ValueError:
                continue
    return sorted(set(out)) or [0.0]


def _derive(f: Feature) -> None:
    """Add D_* properties (see ns.DERIVED)."""
    p, k = f.props, f.kinds
    incline = [str(v).lower() for v in p.get("P_INCLINE", [])]
    stairs = ("K_HANDRAIL" in k or "P_STEPCOUNT" in p or any(v in ("up", "down") for v in incline)
              or any(str(v).lower() == "steps" for v in p.get("P_HIGHWAY", [])))
    elevator = "K_ELEVATOR" in k or any(str(v).lower() == "elevator" for v in p.get("P_AMENITY", []))
    # a multi-level navigable polygon that is not an elevator is a stairwell
    if (f.cls == "NavigablePath" and f.geom.geom_type != "LineString" and not elevator
            and len(_levels(p.get("P_LEVEL"))) > 1):
        stairs = True
    if f.cls == "NavigablePath":
        p["D_STAIRS"] = [stairs and not elevator]
        p["D_ELEVATOR"] = [elevator]
        p["D_INDOOR"] = ["K_INDOOR" in k or bool(p.get("P_COVERED"))]
    if elevator:
        p.setdefault("P_AMENITY", []).append("elevator")
    if "K_TRAM" in k or any(str(v).lower() == "tram" for v in p.get("P_RAILWAY", [])):
        p["D_TRAM"] = ["yes"]
    slopes = []
    for v in incline:
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", v)
        if m:
            slopes.append(abs(float(m.group(1))))
    if slopes:
        p["D_SLOPE"] = [max(slopes)]


def _extract() -> tuple[list[Feature], tuple | None]:
    g = kg.abox()
    classes = {"PointOfInterest", "PolylineOfInterest", "AreaOfInterest", "NavigablePath"}
    feats, intervals = [], {}
    tmin, tmax = None, None
    for cls in classes:
        for s in g.subjects(RDF.type, PIIRRITE[cls]):
            gn = g.value(s, GEO.hasGeometry)
            w = g.value(gn, GEO.asWKT) if gn is not None else None
            if w is None:
                continue
            try:
                geom = transform(to_m, wkt.loads(str(w)))
            except Exception:
                continue
            f = Feature(str(s), cls, geom, {local_name(k).upper() for k in g.objects(s, SAREF.hasFeatureKind)})
            for pv in g.objects(s, SAREF.hasPropertyValue):
                prop, val = g.value(pv, SAREF.isValueOfProperty), g.value(pv, SAREF.hasValue)
                if prop is None or val is None:
                    continue
                key, val = norm_prop(prop), val.toPython() if hasattr(val, "toPython") else str(val)
                iv = g.value(pv, PIIRRITE.hasTemporalValidity)
                if iv is not None:
                    if iv not in intervals:
                        intervals[iv] = _interval(g, iv)
                    start, end = intervals[iv]
                    f.timed.setdefault(key, []).append((start, end, val))
                    if start:
                        tmin = start if tmin is None or start < tmin else tmin
                    if end:
                        tmax = end if tmax is None or end > tmax else tmax
                else:
                    f.props.setdefault(key, []).append(val)
            # passesBy: POIs along a path contribute their place-like properties
            f.passes = [str(o) for o in g.objects(s, PIIRRITE.passesBy)]
            _derive(f)
            feats.append(f)
    return feats, (tmin, tmax) if tmin else None


def _build_graph(feats: list[Feature]) -> nx.Graph:
    by_iri = {f.iri: f for f in feats}
    paths = [f for f in feats if f.cls == "NavigablePath" and f.geom.geom_type == "LineString"]

    def key(x, y, lvl):
        return (round(x / SNAP) * SNAP, round(y / SNAP) * SNAP, lvl)

    use = Counter()
    plevels = {}
    for f in paths:
        plevels[f.iri] = _levels(f.props.get("P_LEVEL"))
        for lvl in plevels[f.iri]:
            seen = set()
            for x, y in f.geom.coords:
                k = key(x, y, lvl)
                if k not in seen:
                    use[k] += 1
                    seen.add(k)
    G = nx.Graph()
    for f in paths:
        # place-like properties of passed-by POIs (amenity, bin, toilets...) for waypoint/avoid checks
        near = defaultdict(list)
        for poi in getattr(f, "passes", []):
            pf = by_iri.get(poi)
            if pf:
                for k, vs in pf.props.items():
                    near[k].extend(vs)
        attrs = {"path": f.iri, "props": f.props, "near": dict(near), "kinds": f.kinds}
        levels = plevels[f.iri]
        coords = list(f.geom.coords)
        for lvl in levels:
            seg = [coords[0]]
            for i in range(1, len(coords)):
                seg.append(coords[i])
                k = key(*coords[i], lvl)
                if i == len(coords) - 1 or use[k] > 1:
                    a, b = key(*seg[0], lvl), k
                    if a != b:
                        line = LineString(seg)
                        if not G.has_edge(a, b) or G[a][b]["length"] > line.length:
                            G.add_edge(a, b, length=line.length, geom=line, level=lvl, **attrs)
                    seg = [coords[i]]
        if len(levels) > 1:                      # vertical links between the floors it serves
            vlen = 3.0 * (2.0 if f.props.get("D_STAIRS") == [True] else 1.0)
            for end in (coords[0], coords[-1]):
                for l1, l2 in zip(levels, levels[1:]):
                    a, b = key(*end, l1), key(*end, l2)
                    G.add_edge(a, b, length=vlen * (l2 - l1), geom=LineString([end, end]),
                               level=l1, vertical=True, **attrs)
    _link_vertical_polygons(G, [f for f in feats if f.cls == "NavigablePath"
                                and f.geom.geom_type in ("Polygon", "MultiPolygon")])
    _bridge_gaps(G)
    for n in G.nodes:
        G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["level"] = n
    return G


def _link_vertical_polygons(G: nx.Graph, polys: list[Feature], reach: float = 4.0) -> None:
    """Elevator shafts / stairwells are multi-level polygons: on every floor they
    serve, take the graph node closest to the shaft and chain the floors together."""
    by_level = defaultdict(list)
    for n in G.nodes:
        by_level[n[2]].append(n)
    trees = {l: STRtree([Point(n[0], n[1]) for n in ns]) for l, ns in by_level.items()}
    for f in polys:
        levels = _levels(f.props.get("P_LEVEL"))
        if len(levels) < 2:
            continue
        zone, c = f.geom.buffer(reach), f.geom.centroid
        stops = []
        for lvl in levels:
            if lvl not in trees:
                continue
            cands = [by_level[lvl][j] for j in trees[lvl].query(zone)]
            cands = [n for n in cands if zone.contains(Point(n[0], n[1]))]
            if cands:
                stops.append(min(cands, key=lambda n: (n[0] - c.x) ** 2 + (n[1] - c.y) ** 2))
        per_floor = 4.0 if f.props.get("D_ELEVATOR") == [True] else 6.0   # metres-equivalent
        for a, b in zip(stops, stops[1:]):
            G.add_edge(a, b, length=per_floor * abs(b[2] - a[2]), geom=LineString([a[:2], b[:2]]),
                       level=a[2], vertical=True, path=f.iri, props=f.props, near={}, kinds=f.kinds)


BRIDGE = 3.0   # metres


def _bridge_gaps(G: nx.Graph) -> None:
    """Join components whose nodes lie within BRIDGE metres on the same level.

    OSM indoor corridors and outdoor footways often stop a metre or two apart at
    doorways instead of sharing a vertex. Bridges are unattributed short links."""
    comp_of = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_of[n] = i
    by_level = defaultdict(list)
    for n in G.nodes:
        by_level[n[2]].append(n)
    added = 0
    for lvl, nodes in by_level.items():
        tree = STRtree([Point(n[0], n[1]) for n in nodes])
        for i, n in enumerate(nodes):
            for j in tree.query(Point(n[0], n[1]).buffer(BRIDGE)):
                m = nodes[j]
                if comp_of[m] != comp_of[n] and not G.has_edge(n, m):
                    d = math.hypot(n[0] - m[0], n[1] - m[1])
                    if d <= BRIDGE:
                        G.add_edge(n, m, length=max(d, 0.1), geom=LineString([n[:2], m[:2]]), level=lvl,
                                   path=None, props={}, near={}, kinds=set(), bridge=True)
                        added += 1
    G.graph["bridges"] = added


@lru_cache(maxsize=None)
def load() -> Network:
    src = kg.DATA_DIR / "ABox.ttl"
    cache = kg.DATA_DIR / f".network.v{CACHE_VERSION}.{src.stat().st_size}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    feats, trange = _extract()
    G = _build_graph(feats)
    coverage: dict[str, Counter] = defaultdict(Counter)
    for f in feats:
        for k, vs in list(f.props.items()) + [(k, [v for *_, v in tv]) for k, tv in f.timed.items()]:
            for v in vs:
                coverage[k][str(v)] += 1
    net = Network(G, feats, {f.iri: f for f in feats if f.cls == "NavigablePath"}, dict(coverage), trange)
    with cache.open("wb") as fh:
        pickle.dump(net, fh)
    return net


def main_component(G: nx.Graph) -> set:
    return max(nx.connected_components(G), key=len)


def nearest_node(net: Network, lat: float, lon: float, level: float = 0.0, nodes=None):
    x, y = net.proj.to_m(lon, lat)
    cand = nodes if nodes is not None else net.graph.nodes
    best, bd = None, math.inf
    for n in cand:
        if n[2] != level:
            continue
        d = (n[0] - x) ** 2 + (n[1] - y) ** 2
        if d < bd:
            best, bd = n, d
    return best, math.sqrt(bd)
