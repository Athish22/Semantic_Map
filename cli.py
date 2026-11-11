"""Command-line interface.

    python cli.py personas list
    python cli.py personas show Anna
    python cli.py personas export Anna            # YAML template on stdout (or -o file)
    python cli.py personas check personas/new/lea.yaml
    python cli.py personas add personas/new/lea.yaml   # validate + write personas/<id>.ttl
    python cli.py personas remove Lea              # only personas you added
    python cli.py properties [--min 1] [--grep wheel]
    python cli.py route --persona Anna --from 45.7800,4.8650 --to 45.7845,4.8790
    python cli.py compare --from 45.7800,4.8650 --to 45.7845,4.8790 [--personas Anna,Lea]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("rdflib").setLevel(logging.ERROR)   # ABox has an "0.7"^^xsd:integer literal

from piirrite_route import kg, personas as P  # noqa: E402
from piirrite_route.ns import DERIVED  # noqa: E402


def _latlon(s: str):
    lat, lon = (float(x) for x in s.split(","))
    return lat, lon


def _when(s: str | None):
    return datetime.fromisoformat(s) if s else None


def _show(p: P.Persona) -> None:
    print(f"{p.id}  ({p.source})" + (f"  - {p.label}" if p.label else "") + (f"  [{p.category}]" if p.category else ""))
    for r in p.requirements:
        imp = "INF" if r.importance == float("inf") else f"{r.importance:g}"
        hard = "  HARD" if r.is_hard() else ""
        print(f"   - {r.label or '(no label)'}: importance {imp}, flexibility {r.flexibility:g}{hard}")
        print(f"       {r.describe()}")
    for n in p.notes:
        print(f"   ! {n}")


def _check(path: Path, existing: set[str]) -> tuple[P.Persona, list, list]:
    from piirrite_route import network
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        p = P.from_dict(data, source=str(path))
    except (ValueError, TypeError) as exc:
        return None, [str(exc)], []
    errors, warnings = P.validate(p, coverage=network.load().coverage, existing_ids=existing)
    return p, errors, warnings


def cmd_personas(a) -> int:
    if a.action == "list":
        for p in P.load_all():
            print(f"{p.id:<12} {len(p.requirements)} requirements  {p.source}"
                  + (f"  [{p.category}]" if p.category else ""))
        return 0
    if a.action == "show":
        _show(P.get(a.target))
        return 0
    if a.action == "export":
        text = yaml.safe_dump(P.to_dict(P.get(a.target)), sort_keys=False, allow_unicode=True)
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"wrote {a.output}")
        else:
            print(text)
        return 0
    if a.action in ("check", "add"):
        existing = {p.id for p in P.load_all()}
        status = 0
        for path in map(Path, a.target_and_more):
            p, errors, warnings = _check(path, existing if a.action == "add" else set())
            print(f"{path}:")
            for w in warnings:
                print(f"   warning: {w}")
            for e in errors:
                print(f"   ERROR: {e}")
            if errors:
                status = 1
                continue
            if a.action == "add":
                out = P.save(p)
                existing.add(p.id)
                print(f"   added {p.id} -> {out.relative_to(kg.ROOT)}")
            else:
                print("   ok")
        return status
    if a.action == "remove":
        f = kg.PERSONA_DIR / f"{a.target}.ttl"
        if not f.exists():
            print(f"{f} not found (personas shipped in ABox.ttl cannot be removed here)")
            return 1
        f.unlink()
        print(f"removed {f.relative_to(kg.ROOT)}")
        return 0
    return 2


def cmd_properties(a) -> int:
    from piirrite_route import network
    net = network.load()
    vocab = P.vocabulary()
    rows = []
    for key in sorted(vocab):
        vals = net.coverage.get(key, {})
        n = sum(vals.values())
        if n < a.min or (a.grep and a.grep.lower() not in key.lower()):
            continue
        top = ", ".join(f"{v} ({c})" for v, c in vals.most_common(6)) if vals else ""
        rows.append((key, n, top))
    for key, n, top in sorted(rows, key=lambda r: -r[1]):
        extra = f"  -- {DERIVED[key]}" if key in DERIVED else ""
        print(f"{key:<24} {n:>6}  {top}{extra}")
    print(f"\n{len(rows)} properties shown (of {len(vocab)} in GoT.ttl + derived)")
    return 0


def _routes(a, ids):
    from piirrite_route import route as R
    router = R.Router(settings=R.Settings(lam=a.lam, unknown_penalty=a.unknown_penalty))
    (s, ds), (e, de) = router.snap(*_latlon(a.src)), router.snap(*_latlon(a.dst))
    if max(ds, de) > 60:
        print(f"warning: start/end snapped {ds:.0f} m / {de:.0f} m away from the requested points")
    shortest = router.shortest(s, e)
    results, texts = [], {}
    for pid in ids:
        p = P.get(pid)
        try:
            rt = router.route(p, s, e, _when(a.time))
        except RuntimeError as exc:
            print(f"{pid}: {exc}")
            continue
        base = router.evaluate_path(p, shortest, _when(a.time))
        texts[p.id] = R.explain(rt, base)
        results.append(rt)
        print("\n".join(texts[p.id]) + "\n")
    return router, results, shortest, texts


def cmd_route(a) -> int:
    router, results, shortest, texts = _routes(a, [a.persona])
    if a.map and results:
        from piirrite_route.viz import compare_map
        print(f"map: {compare_map(router, results, shortest, texts, Path(a.map))}")
    return 0 if results else 1


def cmd_compare(a) -> int:
    ids = a.personas.split(",") if a.personas else [p.id for p in P.load_all()]
    router, results, shortest, texts = _routes(a, ids)
    from piirrite_route.viz import compare_map
    from piirrite_route.viz_png import compare_png, grid_png
    print(f"map: {compare_map(router, results, shortest, texts, Path(a.map))}")
    png = Path(a.map).with_suffix(".png")
    title = f"Personalised routes vs shortest path" + (f"  ({a.time})" if a.time else "")
    print(f"png: {compare_png(router, results, shortest, png, title)}")
    print(f"png: {grid_png(router, results, shortest, texts, png.with_name(png.stem + '_grid.png'))}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Personalised routing on the PIIRRITE knowledge graph")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("personas", help="list / show / export / check / add / remove personas")
    sp.add_argument("action", choices=["list", "show", "export", "check", "add", "remove"])
    sp.add_argument("target_and_more", nargs="*", metavar="target")
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_personas)

    pp = sub.add_parser("properties", help="properties usable in requirements, with data coverage")
    pp.add_argument("--min", type=int, default=1, help="hide properties with fewer occurrences (0 = all)")
    pp.add_argument("--grep")
    pp.set_defaults(func=cmd_properties)

    for name, func in (("route", cmd_route), ("compare", cmd_compare)):
        rp = sub.add_parser(name)
        if name == "route":
            rp.add_argument("--persona", required=True)
            rp.add_argument("--map", help="also write an HTML map")
        else:
            rp.add_argument("--personas", help="comma-separated ids (default: all)")
            rp.add_argument("--map", default="out/compare.html")
        rp.add_argument("--from", dest="src", required=True, help="lat,lon")
        rp.add_argument("--to", dest="dst", required=True, help="lat,lon")
        rp.add_argument("--time", help="ISO datetime for weather/pollen, e.g. 2026-06-06T14:00")
        rp.add_argument("--lam", type=float, default=3.0, help="inadequacy weight vs length")
        rp.add_argument("--unknown-penalty", type=float, default=0.25)
        rp.set_defaults(func=func)

    a = ap.parse_args(argv)
    if getattr(a, "cmd", None) == "personas":
        a.target = a.target_and_more[0] if a.target_and_more else None
        if a.action != "list" and not a.target:
            ap.error(f"personas {a.action} needs a target")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
