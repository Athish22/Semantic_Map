"""Personalised routing: per-persona edge weights + Dijkstra + waypoints + explanations."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
from shapely import STRtree
from shapely.geometry import Point

from . import network as N
from .cost import Context, classify, feature_matches
from .network import Network
from .personas import Atomic, Persona, Requirement


@dataclass
class Settings:
    lam: float = 3.0                 # weight of inadequacy relative to length
    unknown_penalty: float = 0.25    # inadequacy assumed where a property is missing
    hard_importance: float = 0.9     # importance >= this ...
    hard_flexibility: float = 0.1    # ... and flexibility <= this => hard constraint
    detour_factor: float = 2.0       # soft waypoint allowance (x importance x (1-flex) x base cost)


@dataclass
class ReqStats:
    req: Requirement
    hard: bool
    pruned: int = 0
    ok_m: float = 0.0
    bad_m: float = 0.0
    unknown_m: float = 0.0
    bad_score: float = 0.0           # length-weighted inadequacy


@dataclass
class Route:
    persona: str
    nodes: list
    length: float
    cost: float
    stats: list[ReqStats] = field(default_factory=list)
    waypoints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def coords_ll(self, G: nx.Graph):
        """Polyline (lat, lon) list following edge geometries."""
        out, proj = [], N.proj_of(G)
        for u, v in zip(self.nodes, self.nodes[1:]):
            line = list(G[u][v]["geom"].coords)
            if math.dist(line[0], u[:2]) > math.dist(line[-1], u[:2]):
                line.reverse()
            for x, y in line:
                lon, lat = proj.to_ll(x, y)
                out.append((lat, lon))
        return out


def ekey(u, v):
    return (u, v) if u <= v else (v, u)


class Router:
    def __init__(self, net: Network | None = None, settings: Settings | None = None):
        self.net = net or N.load()
        self.s = settings or Settings()
        self.main = N.main_component(self.net.graph)
        self.G = self.net.graph.subgraph(self.main)
        self._node_tree = {}

    # -- snapping ----------------------------------------------------------------
    def snap(self, lat: float, lon: float, level: float = 0.0):
        node, dist = N.nearest_node(self.net, lat, lon, level, self.main)
        if node is None:
            raise ValueError(f"no walkable node on level {level}")
        return node, dist

    def _nearest_main_node(self, geom, level: float):
        if level not in self._node_tree:
            nodes = [n for n in self.main if n[2] == level]
            self._node_tree[level] = (nodes, STRtree([Point(n[0], n[1]) for n in nodes]) if nodes else None)
        nodes, tree = self._node_tree[level]
        if tree is None:
            return None
        i, d = tree.query_nearest(geom.centroid, return_distance=True)
        return nodes[int(i[0])] if len(i) and d[0] < 40 else None

    # -- weights -------------------------------------------------------------------
    def weights(self, p: Persona, when: datetime | None):
        """Per-edge weights for p. The last result is reused when route() and evaluate_path()
        score the same Persona object with the same time and settings (fresh stats each call)."""
        key = (id(p), when, self.s.lam, self.s.unknown_penalty, self.s.hard_importance, self.s.hard_flexibility)
        cached = getattr(self, "_wcache", None)
        if cached and cached[0] == key and cached[1] is p:
            _, _, W, inad, parts, pruned, ctx = cached
            stats = {id(r): ReqStats(r, r.is_hard(self.s.hard_importance, self.s.hard_flexibility),
                                     pruned=pruned[id(r)]) for r in parts["edge"]}
            return W, inad, parts, stats, ctx
        W, inad, parts, stats, ctx = self._weights(p, when)
        self._wcache = (key, p, W, inad, parts, {k: s.pruned for k, s in stats.items()}, ctx)
        return W, inad, parts, stats, ctx

    def _weights(self, p: Persona, when: datetime | None):
        ctx = Context(self.net, when)
        parts = classify(p, self.net)
        stats = {id(r): ReqStats(r, r.is_hard(self.s.hard_importance, self.s.hard_flexibility))
                 for r in parts["edge"]}
        W, inad = {}, {}
        for u, v, d in self.G.edges(data=True):
            total, pruned, per = 0.0, False, {}
            for r in parts["edge"]:
                a = ctx.requirement(r, d)
                per[id(r)] = a
                st = stats[id(r)]
                if a is None:
                    total += r.weight * self.s.unknown_penalty
                    continue
                if st.hard and a >= 0.999:
                    st.pruned += 1
                    pruned = True
                total += r.weight * a
            k = ekey(u, v)
            inad[k] = per
            W[k] = None if pruned else d["length"] * (1 + self.s.lam * total)
        return W, inad, parts, stats, ctx

    # -- waypoints -----------------------------------------------------------------
    def _stop_groups(self, r: Requirement):
        groups = []
        for c in r.conditions:
            assert isinstance(c, Atomic)
            probe = c
            if c.key == "P_WC_ACCESS":            # persona vocabulary -> OSM amenity=toilets
                probe = Atomic("P_Amenity", "EQUALS", ["toilets"])
            nodes = set()
            for f in self.net.features:
                if f.cls != "NavigablePath" and feature_matches(f, probe):
                    lvl = N._levels(f.props.get("P_LEVEL"))[0]
                    n = self._nearest_main_node(f.geom, lvl) or self._nearest_main_node(f.geom, 0.0)
                    if n:
                        nodes.add(n)
            groups.append((r, c, nodes))
        return groups

    # -- main entry point ------------------------------------------------------------
    def route(self, p: Persona, start, end, when: datetime | None = None) -> Route:
        W, inad, parts, stats, _ = self.weights(p, when)
        wf = lambda u, v, d: W.get(ekey(u, v))      # None hides pruned edges
        notes = [f"ignored '{r.label or r.describe()}': {why}" for r, why in parts["unsupported"]]
        groups = [g for r in parts["waypoint"] for g in self._stop_groups(r)]
        try:
            to_end = nx.single_source_dijkstra_path_length(self.G, end, weight=wf)
        except nx.NodeNotFound:
            to_end = {}
        cur, nodes, visited = start, [start], []
        while groups:
            dist, paths = nx.single_source_dijkstra(self.G, cur, weight=wf)
            base = dist.get(end, math.inf)
            best = None
            for gi, (r, c, cands) in enumerate(groups):
                reach = [n for n in cands if n in dist and n in to_end]
                if not reach:
                    continue
                n = min(reach, key=lambda n: dist[n] + to_end[n])
                if best is None or dist[n] < best[0]:
                    best = (dist[n], gi, n)
            if best is None:
                for r, c, cands in groups:
                    notes.append(f"waypoint '{c.describe()}' unreachable"
                                 + ("" if cands else " (no matching place in the data)"))
                break
            _, gi, n = best
            r, c, _ = groups.pop(gi)
            hard = r.is_hard(self.s.hard_importance, self.s.hard_flexibility)
            extra = dist[n] + to_end[n] - base
            allowance = self.s.detour_factor * r.weight * (1 - r.flexibility) * base
            if not hard and extra > allowance:
                notes.append(f"skipped soft waypoint '{c.describe()}': detour +{extra:.0f} > allowance {allowance:.0f}")
                continue
            nodes += paths[n][1:]
            visited.append(c.describe())
            cur = n
        try:
            nodes += nx.dijkstra_path(self.G, cur, end, weight=wf)[1:]
        except nx.NetworkXNoPath:
            raise RuntimeError(f"{p.id}: no route satisfies the hard constraints between these points") from None
        rt = Route(p.id, nodes, 0.0, 0.0, list(stats.values()), visited, notes)
        self._measure(rt, W, inad, stats)
        return rt

    def shortest(self, start, end) -> list:
        return nx.dijkstra_path(self.G, start, end, weight="length")

    def _measure(self, rt: Route, W, inad, stats):
        for u, v in zip(rt.nodes, rt.nodes[1:]):
            k, L = ekey(u, v), self.G[u][v]["length"]
            rt.length += L
            rt.cost += W[k] or 0.0
            for rid, a in inad[k].items():
                st = stats[rid]
                if a is None:
                    st.unknown_m += L
                elif a > 0:
                    st.bad_m += L
                    st.bad_score += a * L
                else:
                    st.ok_m += L

    def evaluate_path(self, p: Persona, nodes: list, when=None) -> Route:
        """Score an arbitrary node path (e.g. the plain shortest path) for persona p."""
        W, inad, parts, stats, _ = self.weights(p, when)
        rt = Route(p.id, nodes, 0.0, 0.0, list(stats.values()))
        for st in stats.values():
            st.pruned = 0
        self._measure(rt, W, inad, stats)
        return rt


def explain(rt: Route, baseline: Route | None = None) -> list[str]:
    lines = [f"{rt.persona}: {rt.length:.0f} m"
             + (f" (shortest path {baseline.length:.0f} m, +{rt.length - baseline.length:.0f} m)" if baseline else "")]
    base = {id(s.req): s for s in baseline.stats} if baseline else {}
    for st in rt.stats:
        r = st.req
        tag = "HARD" if st.hard else f"imp {r.weight:.2f}, flex {r.flexibility:.2f}"
        line = f"  - {r.label or r.describe()} [{tag}]: unmet on {st.bad_m:.0f} m, unknown on {st.unknown_m:.0f} m"
        b = base.get(id(r))
        if b:
            line += f"  | shortest path: unmet on {b.bad_m:.0f} m, unknown on {b.unknown_m:.0f} m"
        if st.pruned:
            line += f"  ({st.pruned} edges excluded)"
        lines.append(line)
    for w in rt.waypoints:
        lines.append(f"  - via {w}")
    lines += [f"  ! {n}" for n in rt.notes]
    return lines
