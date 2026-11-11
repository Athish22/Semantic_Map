"""Loading the PIIRRITE knowledge graph (TBox, GoT vocabulary, ABox, extra personas)."""
from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path

from rdflib import Graph

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PERSONA_DIR = ROOT / "personas"

EXPECTED_MD5 = {
    "TBox.ttl": "8b2d69083cf465c305e32e7ce7cf38a1",
    "GoT.ttl": "9c0cc3289a66c93e812a99e9176eb9ff",
    "ABox.ttl": "34ab52cb2818e6145b97cbb291930c91",
}
ZENODO = "https://zenodo.org/records/21496238/files/{name}?download=1"


def ensure_data() -> None:
    """Download the Zenodo files if missing and check their checksums."""
    import urllib.request

    DATA_DIR.mkdir(exist_ok=True)
    for name, md5 in EXPECTED_MD5.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(ZENODO.format(name=name), path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != md5:
            raise RuntimeError(f"{name}: checksum {digest} != expected {md5}")


def _cached_parse(path: Path) -> Graph:
    stamp = f"{path.stat().st_size}-{int(path.stat().st_mtime)}"
    cache = path.with_name(f".{path.stem}.{stamp}.pkl")
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    for old in path.parent.glob(f".{path.stem}.*.pkl"):
        old.unlink()
    g = Graph()
    g.parse(path, format="turtle")
    with cache.open("wb") as fh:
        pickle.dump(g, fh)
    return g


@lru_cache(maxsize=None)
def abox() -> Graph:
    ensure_data()
    return _cached_parse(DATA_DIR / "ABox.ttl")


@lru_cache(maxsize=None)
def got() -> Graph:
    ensure_data()
    return _cached_parse(DATA_DIR / "GoT.ttl")


def persona_files() -> list[Path]:
    return sorted(PERSONA_DIR.glob("*.ttl")) if PERSONA_DIR.exists() else []


def persona_sources() -> list[tuple[str, Graph]]:
    """(source name, graph) pairs holding personas: ABox.ttl first, then personas/*.ttl."""
    sources = [("ABox.ttl", abox())]
    for f in persona_files():
        g = Graph()
        g.parse(f, format="turtle")
        sources.append((f"personas/{f.name}", g))
    return sources
