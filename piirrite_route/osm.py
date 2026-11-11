"""Routing anywhere: build a Network from OpenStreetMap (Overpass API) plus Open-Meteo weather/pollen.

The PIIRRITE knowledge graph is itself largely OSM-derived: its properties are OSM tag keys
in PIIRRITE spelling (``tactile_paving`` -> ``P_TactilePaving``, canonical key ``P_TACTILEPAVING``).
So OSM ways/nodes are converted into the same ``Feature`` objects, with the same derived
``D_*`` properties, and the rest of the pipeline (graph, costs, personas) is unchanged.

Differences from the campus graph:
* no indoor floors in practice (outdoor ways have no ``level``), no noise/crowd data;
* weather is one hourly series for the whole area (Open-Meteo, any place on Earth);
* pollen intensity (0..1) is the Open-Meteo CAMS European forecast: the highest species
  concentration relative to a "high" level, area-wide (no per-tree hotspots); Europe only.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon, box

from . import kg
from . import network as N
from .network import Feature, Network, Proj

USER_AGENT = "piirrite-routing-prototype/0.2 (research prototype)"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
CACHE_DIR = kg.DATA_DIR / "osm_cache"
CACHE_VERSION = 1
MAX_SPAN_M = 12_000          # largest bbox side we fetch
MARGIN_M = 400               # minimum padding around A/B

WALKABLE = {"footway", "path", "pedestrian", "steps", "living_street", "residential", "service",
            "unclassified", "tertiary", "tertiary_link", "secondary", "secondary_link", "primary",
            "primary_link", "track", "cycleway", "bridleway", "corridor", "road", "elevator", "platform"}
AREA_KEYS = {"amenity", "leisure", "landuse", "natural", "building", "area"}


class OSMError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ helpers
def _get(url: str, data: bytes | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def prop_key(tag: str) -> str:
    """OSM tag key -> canonical PIIRRITE key (``step_count`` -> ``P_STEPCOUNT``)."""
    return "P_" + "_".join(part.replace("_", "") for part in tag.split(":")).upper()


def bbox_for(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float, float]:
    """(south, west, north, east) around two (lat, lon) points, padded and snapped to a grid
    so that nearby queries share a cache entry."""
    lat0 = (a[0] + b[0]) / 2
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540
    dx, dy = abs(a[1] - b[1]) * kx, abs(a[0] - b[0]) * ky
    if max(dx, dy) > MAX_SPAN_M - 2 * MARGIN_M:
        raise OSMError(f"A and B are {math.hypot(dx, dy) / 1000:.1f} km apart; "
                       f"OpenStreetMap routing is limited to about {(MAX_SPAN_M - 2 * MARGIN_M) / 1000:.0f} km")
    pad = max(MARGIN_M, 0.2 * math.hypot(dx, dy))
    step = 0.005                                    # ~500 m grid
    s = math.floor((min(a[0], b[0]) - pad / ky) / step) * step
    n = math.ceil((max(a[0], b[0]) + pad / ky) / step) * step
    w = math.floor((min(a[1], b[1]) - pad / kx) / step) * step
    e = math.ceil((max(a[1], b[1]) + pad / kx) / step) * step
    return round(s, 4), round(w, 4), round(n, 4), round(e, 4)


def contains(bb, pt) -> bool:
    s, w, n, e = bb
    return s <= pt[0] <= n and w <= pt[1] <= e


# ------------------------------------------------------------------------------ Overpass
def _query(bb) -> str:
    s, w, n, e = bb
    b = f"({s},{w},{n},{e})"
    return f"""[out:json][timeout:80][maxsize:536870912];
(
  way["highway"]{b};
  way["railway"="tram"]{b};
  nw["amenity"]{b};
  nw["leisure"]{b};
  node["natural"="tree"]{b};
  way["natural"~"^(tree_row|wood|scrub|grassland|heath)$"]{b};
  way["landuse"~"^(grass|meadow|forest|flowerbed|recreation_ground|village_green)$"]{b};
  node["highway"~"^(street_lamp|crossing|elevator)$"]{b};
);
out tags geom;"""


def fetch(bb) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / ("overpass_" + "_".join(map(str, bb)) + ".json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    data = urllib.parse.urlencode({"data": _query(bb)}).encode()
    errors = []
    for url in OVERPASS:                      # one try per mirror; public servers are often busy
        try:
            out = _get(url, data, timeout=90)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{urllib.parse.urlparse(url).netloc}: {exc}")
            time.sleep(1)
            continue
        if out.get("remark", "").startswith("runtime error"):
            errors.append(f"{urllib.parse.urlparse(url).netloc}: {out['remark'][:120]}")
            continue
        f.write_text(json.dumps(out), encoding="utf-8")
        return out
    raise OSMError("OpenStreetMap servers are busy, try again in a minute (" + "; ".join(errors) + ")")


# ------------------------------------------------------------------------------ conversion
def _walkable(t: dict) -> bool:
    hw = t.get("highway")
    if hw not in WALKABLE:
        return False
    foot = t.get("foot", "")
    if foot == "no" or t.get("access") in ("no", "private") and foot not in ("yes", "designated", "permissive"):
        return False
    if hw in ("cycleway", "bridleway") and foot not in ("yes", "designated", "permissive"):
        return False
    return True


def _feature(el: dict, proj: Proj) -> Feature | None:
    t = el.get("tags") or {}
    iri = f"osm:{el['type']}/{el['id']}"
    if el["type"] == "node":
        geom = Point(proj.to_m(el["lon"], el["lat"]))
        cls = "PointOfInterest"
    else:
        pts = [proj.to_m(p["lon"], p["lat"]) for p in el.get("geometry") or [] if p]
        if len(pts) < 2:
            return None
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        if _walkable(t):
            geom, cls = LineString(pts), "NavigablePath"
        elif closed and (t.get("area") == "yes" or AREA_KEYS & t.keys()) and t.get("natural") != "tree_row":
            geom, cls = Polygon(pts), "AreaOfInterest"
            if not geom.is_valid:
                geom = geom.buffer(0)
        else:
            geom, cls = LineString(pts), "PolylineOfInterest"
    kinds = set()
    if t.get("indoor") not in (None, "no") or t.get("highway") == "corridor":
        kinds.add("K_INDOOR")
    if t.get("railway") == "tram":
        kinds.add("K_TRAM")
    if t.get("highway") == "elevator":
        kinds.add("K_ELEVATOR")
    if t.get("handrail") in ("yes", "both", "left", "right"):
        kinds.add("K_HANDRAIL")
    if "building" in t:
        kinds.add("K_BUILDING")
    props = defaultdict(list)
    for k, v in t.items():
        for part in str(v).split(";") if k not in ("level", "opening_hours", "name") else [v]:
            props[prop_key(k)].append(part.strip())
    if t.get("covered") in ("no",):
        props.pop("P_COVERED", None)                # _derive treats any P_COVERED as covered
    f = Feature(iri, cls, geom, kinds, dict(props))
    f.passes = []
    N._derive(f)
    return f


def _attach_pois(feats: list[Feature]) -> None:
    """POIs within 12 m of a path are 'passed by' it (like piirrite:passesBy on campus)."""
    paths = [f for f in feats if f.cls == "NavigablePath" and f.geom.geom_type == "LineString"]
    if not paths:
        return
    tree = STRtree([p.geom for p in paths])
    for f in feats:
        if f.cls != "PointOfInterest" or not ({"P_AMENITY", "P_LEISURE", "P_VENDING"} & f.props.keys()):
            continue
        for i in tree.query(f.geom.buffer(12)):
            if paths[i].geom.distance(f.geom) <= 12:
                paths[i].passes.append(f.iri)


def build(bb) -> Network:
    key = hashlib.md5(json.dumps([CACHE_VERSION, N.CACHE_VERSION, bb]).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"net_{key}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    raw = fetch(bb)
    s, w, n, e = bb
    proj = Proj((s + n) / 2, (w + e) / 2)
    feats = [f for el in raw.get("elements", []) if (f := _feature(el, proj)) is not None]
    if not any(f.cls == "NavigablePath" for f in feats):
        raise OSMError("OpenStreetMap has no walkable paths in this area")
    _attach_pois(feats)
    G = N._build_graph(feats)
    G.graph["proj"] = proj
    coverage: dict[str, Counter] = defaultdict(Counter)
    for f in feats:
        for k, vs in f.props.items():
            for v in vs:
                coverage[k][str(v)] += 1
    net = Network(G, feats, {f.iri: f for f in feats if f.cls == "NavigablePath"}, dict(coverage), None,
                  proj=proj, kind="osm")
    net.bbox = bb
    net.weather_days = set()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(net, fh)
    return net


# ------------------------------------------------------------------------------ weather / pollen
WMO = {0: "clear", 1: "clear", 2: "partly_cloudy", 3: "cloudy", 45: "fog", 48: "fog"}
for c in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
    WMO[c] = "rain"
for c in (71, 73, 75, 77, 85, 86):
    WMO[c] = "snow"
for c in (95, 96, 99):
    WMO[c] = "thunderstorm"
POLLEN_HIGH = {"alder_pollen": 100, "birch_pollen": 100, "grass_pollen": 50, "mugwort_pollen": 50,
               "olive_pollen": 200, "ragweed_pollen": 30}      # grains/m3 considered "high"
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _series(hourly: dict, field: str, conv=lambda v: v):
    out = []
    for t, v in zip(hourly.get("time", []), hourly.get(field, [])):
        if v is None:
            continue
        start = datetime.fromisoformat(t)
        out.append((start, start + timedelta(hours=1), conv(v)))
    return out


def weather_features(net: Network, day: date) -> tuple[list[Feature], list[str]]:
    """One area covering the network with hourly weather (and pollen) for `day`."""
    s, w, n, e = net.bbox
    lat, lon = (s + n) / 2, (w + e) / 2
    ds = day.isoformat()
    notes = []
    common = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "timezone": "auto",
              "start_date": ds, "end_date": ds}
    fields = "temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,surface_pressure," \
             "wind_speed_10m,wind_direction_10m"
    recent = (date.today() - day).days <= 80
    base = "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"
    timed: dict[str, list] = {}
    try:
        h = _get(base + "?" + urllib.parse.urlencode({**common, "hourly": fields}), timeout=30).get("hourly", {})
        timed = {
            "P_TEMPERATURE": _series(h, "temperature_2m", float),
            "P_HUMIDITY": _series(h, "relative_humidity_2m", float),
            "P_PRECIPITATION": _series(h, "precipitation", float),
            "P_WEATHER_CONDITION": _series(h, "weather_code", lambda c: WMO.get(int(c), "cloudy")),
            "P_CLOUD_COVER": _series(h, "cloud_cover", float),
            "P_PRESSURE": _series(h, "surface_pressure", float),
            "P_WIND_SPEED": _series(h, "wind_speed_10m", float),
            "P_WIND_DIRECTION": _series(h, "wind_direction_10m", lambda d: COMPASS[int((d + 22.5) % 360 // 45)]),
        }
        timed = {k: v for k, v in timed.items() if v}
        if not timed:
            notes.append(f"no weather available for {ds}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"weather unavailable ({exc})")
    if (date.today() - day).days <= 90 and (day - date.today()).days <= 4:
        try:
            q = {**common, "hourly": ",".join(POLLEN_HIGH)}
            h = _get("https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(q),
                     timeout=30).get("hourly", {})
            pol = []
            for i, t in enumerate(h.get("time", [])):
                vals = [h[k][i] / hi for k, hi in POLLEN_HIGH.items() if h.get(k) and h[k][i] is not None]
                if vals:
                    st = datetime.fromisoformat(t)
                    pol.append((st, st + timedelta(hours=1), round(min(1.0, max(vals)), 3)))
            if pol:
                timed["P_POLLEN_INTENSITY"] = pol
            else:
                notes.append("no pollen forecast here (Open-Meteo pollen covers Europe only)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"pollen unavailable ({exc})")
    else:
        notes.append("pollen data only exists for the last 90 days and next 4 days")
    if not timed:
        return [], notes
    x0, y0 = net.proj.to_m(w, s)
    x1, y1 = net.proj.to_m(e, n)
    return [Feature(f"open-meteo:{ds}", "AreaOfInterest", box(x0, y0, x1, y1), set(), {}, timed)], notes


def ensure_weather(net: Network, when: datetime) -> list[str]:
    days = getattr(net, "weather_days", None)
    if days is None:
        days = net.weather_days = set()
    notes_by_day = getattr(net, "weather_notes", None)
    if notes_by_day is None:
        notes_by_day = net.weather_notes = {}
    d = when.date()
    if d not in days:
        feats, notes = weather_features(net, d)
        net.add_features(feats)
        days.add(d)
        notes_by_day[d] = notes
    return notes_by_day.get(d, [])


# ------------------------------------------------------------------------------ geocoding
def geocode(q: str, limit: int = 5) -> list[dict]:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": limit})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.loads(r.read().decode("utf-8"))
    return [{"name": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in rows]
