"""Local web UI for persona-aware routing.

    python server.py                 # http://127.0.0.1:8765, opens a browser tab
    python server.py --port 9000 --no-browser

Standard library only (http.server); the page itself is web/index.html.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml

sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("rdflib").setLevel(logging.ERROR)

from piirrite_route import kg, network as N, osm, personas as P, route as R  # noqa: E402
from piirrite_route.cost import classify  # noqa: E402
from piirrite_route.ns import DERIVED  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
TEMPLATES = kg.PERSONA_DIR / "new"
LOCK = threading.RLock()          # the router and persona cache are not thread-safe
STATE: dict = {}


# ----------------------------------------------------------------------------- state
def router() -> R.Router:
    if "router" not in STATE:
        STATE["router"] = R.Router()
    return STATE["router"]


CAMPUS_SNAP_M = 75     # auto mode: use the campus graph when A and B are this close to it
OSM_ROUTERS: list = []  # most recently used first: (bbox, Router)


def osm_router(a, b) -> R.Router:
    """Router on an OpenStreetMap network covering a and b (reuses a cached area when it fits)."""
    for i, (bb, rt) in enumerate(OSM_ROUTERS):
        s, w, n, e = bb
        inner = (s + 0.002, w + 0.003, n - 0.002, e - 0.003)        # ~200 m away from the edge
        if osm.contains(inner, a) and osm.contains(inner, b):
            OSM_ROUTERS.insert(0, OSM_ROUTERS.pop(i))
            return rt
    bb = osm.bbox_for(a, b)
    rt = R.Router(osm.build(bb))
    OSM_ROUTERS.insert(0, (bb, rt))
    del OSM_ROUTERS[3:]
    return rt


def pick_router(body: dict) -> R.Router:
    mode = body.get("mode", "auto")
    a, b = tuple(body["from"]), tuple(body["to"])
    if mode == "campus":
        return router()
    if mode == "auto":
        c = router()
        if max(c.snap(*a)[1], c.snap(*b)[1]) <= CAMPUS_SNAP_M:
            return c
    return osm_router(a, b)


def personas() -> list[P.Persona]:
    if "personas" not in STATE:
        STATE["personas"] = P.load_all()
    return STATE["personas"]


def persona(pid: str) -> P.Persona:
    for p in personas():
        if p.id.lower() == pid.lower():
            return p
    raise KeyError(f"no persona {pid!r}")


def invalidate_personas():
    STATE.pop("personas", None)


# ----------------------------------------------------------------------------- serialisers
def _num(x):
    return "INF" if isinstance(x, float) and math.isinf(x) else x


def persona_json(p: P.Persona) -> dict:
    parts = classify(p, router().net)
    kind = {id(r): "edge" for r in parts["edge"]}
    kind |= {id(r): "waypoint" for r in parts["waypoint"]}
    why = {id(r): reason for r, reason in parts["unsupported"]}
    s = router().s
    return {
        "id": p.id, "label": p.label, "category": p.category, "source": p.source,
        "editable": p.source.startswith("personas/"), "notes": p.notes,
        "requirements": [{
            "label": r.label, "describe": r.describe(), "importance": _num(r.importance),
            "flexibility": r.flexibility, "hard": r.is_hard(s.hard_importance, s.hard_flexibility),
            "kind": kind.get(id(r), "ignored"), "ignored_reason": why.get(id(r)),
        } for r in p.requirements],
    }


def _ll(node, G):
    lon, lat = N.proj_of(G).to_ll(*node[:2])
    return [round(lat, 6), round(lon, 6)]


def segments(G, nodes) -> list[dict]:
    """Route split into runs on the same floor: [{level, coords}], vertical moves as level 'v'."""
    out, proj = [], N.proj_of(G)
    for u, v in zip(nodes, nodes[1:]):
        line = list(G[u][v]["geom"].coords)
        if math.dist(line[0], u[:2]) > math.dist(line[-1], u[:2]):
            line.reverse()
        lvl = u[2] if u[2] == v[2] else "v"
        pts = [[round(la, 6), round(lo, 6)] for lo, la in (proj.to_ll(x, y) for x, y in line)]
        if out and out[-1]["level"] == lvl:
            out[-1]["coords"] += pts[1:]
        else:
            out.append({"level": lvl, "coords": pts})
    return out


def stats_json(rt: R.Route, base: R.Route | None) -> list[dict]:
    b = {id(st.req): st for st in base.stats} if base else {}
    out = []
    for st in rt.stats:
        r, bs = st.req, b.get(id(st.req))
        out.append({
            "label": r.label, "describe": r.describe(), "hard": st.hard,
            "importance": _num(r.importance), "flexibility": r.flexibility,
            "ok_m": round(st.ok_m, 1), "bad_m": round(st.bad_m, 1), "unknown_m": round(st.unknown_m, 1),
            "pruned": st.pruned,
            "base": {"ok_m": round(bs.ok_m, 1), "bad_m": round(bs.bad_m, 1),
                     "unknown_m": round(bs.unknown_m, 1)} if bs else None,
        })
    return out


def _settings(body: dict) -> R.Settings:
    return R.Settings(lam=float(body.get("lam", 3.0)), unknown_penalty=float(body.get("unknown_penalty", 0.25)))


def _when(body: dict, rt: R.Router):
    t = body.get("time")
    when = datetime.fromisoformat(t) if t else None
    if rt.net.kind == "osm":            # live weather: default to now, fetch the day on demand
        when = when or datetime.now().replace(minute=0, second=0, microsecond=0)
        osm.ensure_weather(rt.net, when)
    return when


def _endpoints(body: dict):
    rt = pick_router(body)
    (s, ds), (e, de) = rt.snap(*body["from"]), rt.snap(*body["to"])
    return rt, s, e, ds, de


def network_info(rt: R.Router, when) -> dict:
    net = rt.net
    info = {"kind": net.kind, "edges": rt.G.number_of_edges(), "time": when.isoformat(timespec="minutes") if when else None}
    if net.kind == "osm":
        info["bbox"] = net.bbox
        info["notes"] = getattr(net, "weather_notes", {}).get(when.date(), []) if when else []
    return info


# ----------------------------------------------------------------------------- API
def api_meta(_q, _b):
    net = router().net
    xs = [n[0] for n in router().main]
    ys = [n[1] for n in router().main]
    (lo0, la0), (lo1, la1) = N.to_ll(min(xs), min(ys)), N.to_ll(max(xs), max(ys))
    tr = net.time_range
    return {"bounds": [[la0, lo0], [la1, lo1]],
            "time_range": [tr[0].isoformat(timespec="minutes"), tr[1].isoformat(timespec="minutes")] if tr else None,
            "defaults": {"from": [45.7800, 4.8650], "to": [45.7845, 4.8790], "lam": 3.0, "unknown_penalty": 0.25}}


def api_basemap(_q, _b):
    if "basemap" in STATE:
        return STATE["basemap"]
    from shapely.geometry import mapping
    from shapely.ops import linemerge, transform

    ll = lambda x, y, z=None: tuple(round(c, 6) for c in N.to_ll(x, y))
    rt = router()
    feats = []
    for f in rt.net.features:
        g = f.geom
        if g.geom_type not in ("Polygon", "MultiPolygon") or g.area > 2e5:
            continue
        green = (any(str(v) in ("grass", "flowerbed", "meadow", "park") for v in f.props.get("P_LANDUSE", []))
                 or any(str(v) in ("scrub", "wood", "heath", "grassland") for v in f.props.get("P_NATURAL", [])))
        if green or "K_BUILDING" in f.kinds:
            feats.append({"type": "Feature", "geometry": mapping(transform(ll, g.simplify(0.4))),
                          "properties": {"k": "green" if green else "building"}})
    lines = [d["geom"] for u, v, d in rt.G.edges(data=True) if u[2] == 0 and v[2] == 0 and not d.get("bridge")]
    merged = linemerge(lines)
    for g in getattr(merged, "geoms", [merged]):
        feats.append({"type": "Feature", "geometry": mapping(transform(ll, g.simplify(0.3))),
                      "properties": {"k": "path"}})
    STATE["basemap"] = {"type": "FeatureCollection", "features": feats}
    return STATE["basemap"]


def api_personas(_q, _b):
    return [persona_json(p) for p in personas()]


def api_geocode(q, _b):
    text = (q.get("q") or "").strip()
    return osm.geocode(text) if text else []


def api_persona_yaml(q, _b):
    p = persona(q["id"])
    return {"yaml": yaml.safe_dump(P.to_dict(p), sort_keys=False, allow_unicode=True)}


def api_templates(_q, _b):
    return [{"name": f.stem, "yaml": f.read_text(encoding="utf-8")} for f in sorted(TEMPLATES.glob("*.yaml"))]


def api_properties(q, _b):
    net, vocab = router().net, P.vocabulary()
    rows = []
    for key in vocab:
        vals = net.coverage.get(key, {})
        rows.append({"key": key, "count": sum(vals.values()),
                     "top": [[v, c] for v, c in vals.most_common(8)] if vals else [],
                     "derived": DERIVED.get(key)})
    rows.sort(key=lambda r: (-r["count"], r["key"]))
    return rows


def _parse_yaml(text: str, source: str):
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return None, ["the YAML must be a mapping with id, label, category, requirements"], []
        p = P.from_dict(data, source=source)
    except (yaml.YAMLError, ValueError, TypeError, KeyError) as exc:
        return None, [str(exc)], []
    return p, None, None


def _validate(body: dict):
    p, errors, warnings = _parse_yaml(body.get("yaml", ""), "personas/(new)")
    if p is None:
        return None, errors, warnings
    replace = (body.get("replace") or "").lower()
    existing = {x.id for x in personas() if x.id.lower() != replace or not x.source.startswith("personas/")}
    errors, warnings = P.validate(p, coverage=router().net.coverage, existing_ids=existing)
    return p, errors, warnings


def api_check(_q, body):
    p, errors, warnings = _validate(body)
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "persona": persona_json(p) if p and not errors else None}


def api_save(_q, body):
    p, errors, warnings = _validate(body)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    replace = body.get("replace")
    if replace and replace.lower() != p.id.lower():
        old = kg.PERSONA_DIR / f"{replace}.ttl"
        if old.exists():
            old.unlink()
    path = P.save(p)
    invalidate_personas()
    return {"ok": True, "errors": [], "warnings": warnings, "id": p.id,
            "path": str(path.relative_to(kg.ROOT)).replace("\\", "/")}


def api_delete(q, _b):
    f = kg.PERSONA_DIR / f"{q['id']}.ttl"
    if not f.exists():
        raise KeyError(f"{q['id']} is not an added persona (ABox personas cannot be removed)")
    f.unlink()
    invalidate_personas()
    return {"ok": True}


def api_shortest(_q, body):
    try:
        rt, s, e, ds, de = _endpoints(body)
    except osm.OSMError as exc:
        return {"error": str(exc)}
    when = _when(body, rt)
    try:
        nodes = rt.shortest(s, e)
    except Exception:  # noqa: BLE001 - networkx NoPath
        return {"error": "A and B are not connected by walkable paths in this network"}
    length = sum(rt.G[u][v]["length"] for u, v in zip(nodes, nodes[1:]))
    return {"length": round(length, 1), "segments": segments(rt.G, nodes), "network": network_info(rt, when),
            "start": _ll(s, rt.G), "end": _ll(e, rt.G), "snap_m": [round(ds, 1), round(de, 1)]}


def api_route(_q, body):
    rt, s, e, _, _ = _endpoints(body)
    rt.s = _settings(body)
    p, when = persona(body["persona"]), _when(body, rt)
    try:
        route = rt.route(p, s, e, when)
    except RuntimeError as exc:
        return {"persona": p.id, "error": str(exc)}
    base = rt.evaluate_path(p, rt.shortest(s, e), when)
    return {"persona": p.id, "length": round(route.length, 1), "base_length": round(base.length, 1),
            "cost": round(route.cost, 1), "base_cost": round(base.cost, 1),
            "segments": segments(rt.G, route.nodes), "levels": sorted({n[2] for n in route.nodes}),
            "stats": stats_json(route, base), "waypoints": route.waypoints, "notes": route.notes,
            "explanation": R.explain(route, base)}


GET = {"/api/meta": api_meta, "/api/basemap": api_basemap, "/api/personas": api_personas,
       "/api/persona_yaml": api_persona_yaml, "/api/templates": api_templates, "/api/properties": api_properties,
       "/api/geocode": api_geocode}
POST = {"/api/check": api_check, "/api/save": api_save, "/api/delete": api_delete,
        "/api/shortest": api_shortest, "/api/route": api_route}
STATIC = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _api(self, fn, query, body):
        try:
            with LOCK:
                out = fn(query, body)
            self._send(200, json.dumps(out, default=str).encode(), "application/json")
        except KeyError as exc:
            self._send(404, json.dumps({"error": str(exc).strip("'\"")}).encode(), "application/json")
        except Exception as exc:  # noqa: BLE001 - report to the page
            traceback.print_exc()
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in GET:
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            return self._api(GET[url.path], q, {})
        rel = unquote(url.path).lstrip("/") or "index.html"
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents or not f.is_file() or f.suffix not in STATIC:
            return self._send(404, b"not found", "text/plain")
        self._send(200, f.read_bytes(), STATIC[f.suffix])

    def do_POST(self):
        url = urlparse(self.path)
        if url.path not in POST:
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error": "invalid JSON"}', "application/json")
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        self._api(POST[url.path], q, body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    print("loading knowledge graph and network ...", flush=True)
    with LOCK:
        router(), personas(), api_basemap(None, None)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"PIIRRITE routing UI on {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
