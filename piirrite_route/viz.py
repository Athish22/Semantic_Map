"""Folium map comparing personas' routes with the plain shortest path."""
from __future__ import annotations

import html
from pathlib import Path

import folium

from . import network as N

PALETTE = ["#2f6fdf", "#e8590c", "#2b8a3e", "#ae3ec9", "#c92a2a", "#0b7285", "#5f3dc4", "#e67700",
           "#495057", "#d6336c", "#1971c2", "#66a80f"]


def _offline_base(router, shortest_nodes, routes):
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    pts = [n for r in routes for n in r.nodes] + list(shortest_nodes)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    frame = box(min(xs) - 250, min(ys) - 250, max(xs) + 250, max(ys) + 250)
    ll = lambda x, y, z=None: N.to_ll(x, y)
    feats = []
    for f in router.net.features:
        g = f.geom
        if g.geom_type not in ("Polygon", "MultiPolygon") or g.area > 2e5 or not g.intersects(frame):
            continue
        green = (any(str(v) in ("grass", "flowerbed") for v in f.props.get("P_LANDUSE", []))
                 or any(str(v) in ("scrub", "wood", "heath") for v in f.props.get("P_NATURAL", [])))
        if green or "K_BUILDING" in f.kinds:
            feats.append({"type": "Feature", "geometry": mapping(transform(ll, g)),
                          "properties": {"c": "#cfe8c4" if green else "#d9d4cc"}})
    for u, v, d in router.G.edges(data=True):
        if u[2] == 0 and not d.get("vertical") and d["geom"].intersects(frame):
            feats.append({"type": "Feature", "geometry": mapping(transform(ll, d["geom"])),
                          "properties": {"c": "#b9b4ab", "line": 1}})
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": feats}, name="Campus (from knowledge graph)",
        style_function=lambda ft: ({"color": ft["properties"]["c"], "weight": 1, "opacity": 1}
                                   if ft["properties"].get("line") else
                                   {"fillColor": ft["properties"]["c"], "color": ft["properties"]["c"],
                                    "weight": 0, "fillOpacity": 1}),
        control=False)


def compare_map(router, routes: list, shortest_nodes: list, explanations: dict, out: Path) -> Path:
    G = router.G
    start, end = shortest_nodes[0], shortest_nodes[-1]
    lon, lat = N.to_ll(*start[:2])
    # No tile server: draw the base map (buildings, green areas, paths) from the knowledge graph.
    m = folium.Map(location=(lat, lon), zoom_start=16, tiles=None, control_scale=True)
    m.get_root().html.add_child(folium.Element("<style>.leaflet-container{background:#f7f5f0}</style>"))
    _offline_base(router, shortest_nodes, routes).add_to(m)

    from .route import Route
    sp = Route("shortest", shortest_nodes, 0, 0)
    folium.PolyLine(sp.coords_ll(G), color="#000000", weight=7, opacity=0.35, dash_array="8 6",
                    tooltip="Shortest path").add_to(folium.FeatureGroup("Shortest path").add_to(m))
    for i, rt in enumerate(routes):
        colour = PALETTE[i % len(PALETTE)]
        fg = folium.FeatureGroup(rt.persona).add_to(m)
        text = "<br>".join(html.escape(l) for l in explanations.get(rt.persona, []))
        folium.PolyLine(rt.coords_ll(G), color=colour, weight=4, opacity=0.9,
                        tooltip=f"{rt.persona}: {rt.length:.0f} m",
                        popup=folium.Popup(f"<div style='font:12px sans-serif;width:420px'>{text}</div>",
                                           max_width=460)).add_to(fg)
    for node, label in ((start, "A"), (end, "B")):
        lo, la = N.to_ll(*node[:2])
        folium.Marker((la, lo), tooltip=label).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out
