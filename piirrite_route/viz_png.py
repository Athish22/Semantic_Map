"""Offline PNG maps: the base map is drawn from the knowledge graph itself (no tile server)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from .viz import PALETTE  # noqa: E402

BG, BUILDING, GREEN, PATH, TRAM = "#f7f5f0", "#d9d4cc", "#cfe8c4", "#b9b4ab", "#5c5c5c"


def _route_xy(G, nodes):
    pts = []
    for u, v in zip(nodes, nodes[1:]):
        line = list(G[u][v]["geom"].coords)
        if (line[0][0] - u[0]) ** 2 + (line[0][1] - u[1]) ** 2 > (line[-1][0] - u[0]) ** 2 + (line[-1][1] - u[1]) ** 2:
            line.reverse()
        pts += line
    return pts


def _offset(pts, d):
    if abs(d) < 1e-6 or len(pts) < 2:
        return pts
    try:
        g = LineString(pts).offset_curve(d, join_style="round")
        return list(g.coords) if g.geom_type == "LineString" and not g.is_empty else pts
    except Exception:
        return pts


def _base(ax, net, G, frame):
    fx = box(*frame)
    for f in net.features:
        g = f.geom
        if g.geom_type not in ("Polygon", "MultiPolygon") or not g.intersects(fx) or g.area > 2e5:
            continue
        green = (any(str(v) in ("grass", "flowerbed") for v in f.props.get("P_LANDUSE", []))
                 or "K_GARDEN" in f.kinds or any(str(v) in ("scrub", "wood", "heath") for v in f.props.get("P_NATURAL", [])))
        building = "K_BUILDING" in f.kinds
        if not (green or building):
            continue
        for poly in getattr(g, "geoms", [g]):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=GREEN if green else BUILDING, lw=0, zorder=1)
    for u, v, d in G.edges(data=True):
        if u[2] != 0 or d.get("vertical"):
            continue
        g = d["geom"]
        if not g.intersects(fx):
            continue
        x, y = g.xy
        ax.plot(x, y, color=PATH, lw=0.6, zorder=2, solid_capstyle="round")
    for f in net.features:
        if "D_TRAM" in f.props and f.geom.geom_type == "LineString" and f.geom.intersects(fx):
            x, y = f.geom.xy
            ax.plot(x, y, color=TRAM, lw=1.6, ls=(0, (4, 2)), zorder=3)


def _frame(all_pts, margin=90):
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _finish(ax, frame):
    ax.set_xlim(frame[0], frame[2])
    ax.set_ylim(frame[1], frame[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    # 100 m scale bar
    x0, y0 = frame[0] + 25, frame[1] + 25
    ax.plot([x0, x0 + 100], [y0, y0], color="#333", lw=2.5, zorder=9)
    ax.text(x0 + 50, y0 + 8, "100 m", ha="center", fontsize=8, color="#333", zorder=9)


def _ends(ax, s, e):
    for (x, y, _), lab in ((s, "A"), (e, "B")):
        ax.scatter([x], [y], s=170, color="#222", zorder=10, edgecolor="white", linewidth=1.5)
        ax.text(x, y, lab, color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=11)


def compare_png(router, routes, shortest_nodes, out: Path, title: str = "") -> Path:
    """One map with every persona's route (slightly offset so overlaps stay visible)."""
    G, net = router.G, router.net
    sp = _route_xy(G, shortest_nodes)
    rxy = [_route_xy(G, r.nodes) for r in routes]
    frame = _frame(sp + [p for r in rxy for p in r])
    fig, ax = plt.subplots(figsize=(13, 9), dpi=150)
    fig.patch.set_facecolor("white")
    _base(ax, net, G, frame)
    x, y = zip(*sp)
    ax.plot(x, y, color="#000", lw=7, alpha=0.22, zorder=4, solid_capstyle="round")
    handles = [Line2D([], [], color="#000", lw=6, alpha=0.3, label=f"Shortest path  {router_len(G, shortest_nodes):.0f} m")]
    n = len(routes)
    for i, (rt, pts) in enumerate(zip(routes, rxy)):
        c = PALETTE[i % len(PALETTE)]
        off = (i - (n - 1) / 2) * 2.2
        x, y = zip(*_offset(pts, off))
        ax.plot(x, y, color=c, lw=2.2, zorder=5 + i * 0.01, solid_capstyle="round")
        handles.append(Line2D([], [], color=c, lw=2.5, label=f"{rt.persona}  {rt.length:.0f} m"))
    handles.append(Line2D([], [], color=TRAM, lw=1.6, ls=(0, (4, 2)), label="Tram line"))
    _ends(ax, shortest_nodes[0], shortest_nodes[-1])
    _finish(ax, frame)
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9, framealpha=0.92)
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def grid_png(router, routes, shortest_nodes, explanations: dict, out: Path) -> Path:
    """Small multiples: one panel per persona, its route vs the shortest path, with key numbers."""
    G, net = router.G, router.net
    sp = _route_xy(G, shortest_nodes)
    rxy = [_route_xy(G, r.nodes) for r in routes]
    frame = _frame(sp + [p for r in rxy for p in r], margin=60)
    cols = 3 if len(routes) > 4 else 2
    rows = -(-len(routes) // cols)
    aspect = (frame[3] - frame[1]) / (frame[2] - frame[0])
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, (6.2 * aspect + 0.5) * rows), dpi=130, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (rt, pts) in enumerate(zip(routes, rxy)):
        ax = axes.flat[i]
        ax.axis("on")
        _base(ax, net, G, frame)
        x, y = zip(*sp)
        ax.plot(x, y, color="#000", lw=5, alpha=0.2, zorder=4)
        x, y = zip(*pts)
        ax.plot(x, y, color=PALETTE[i % len(PALETTE)], lw=2.6, zorder=5)
        _ends(ax, shortest_nodes[0], shortest_nodes[-1])
        _finish(ax, frame)
        lines = explanations.get(rt.persona, [])[1:]
        lines = [l.strip()[2:] for l in lines if not l.strip().startswith("!")][:5]
        note = "\n".join(_short(l) for l in lines)
        ax.set_title(f"{rt.persona}: {rt.length:.0f} m  (+{rt.length - router_len(G, shortest_nodes):.0f} m vs shortest)",
                     fontsize=11, loc="left", color=PALETTE[i % len(PALETTE)], weight="bold")
        ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=7, family="monospace",
                bbox=dict(facecolor="white", alpha=0.88, edgecolor="#ccc", boxstyle="round,pad=0.3"), zorder=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _short(line: str, n: int = 78) -> str:
    line = line.split("  | shortest path:")[0] if len(line) > n else line
    return line if len(line) <= n else line[: n - 1] + "…"


def router_len(G, nodes) -> float:
    return sum(G[u][v]["length"] for u, v in zip(nodes, nodes[1:]))
