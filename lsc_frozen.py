"""Lädt die vorgerechneten Messreihen (data/lsc_vorgerechnet.json, erzeugt von tools/precompute_lsc.py)."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "data" / "lsc_vorgerechnet.json"


@lru_cache(maxsize=1)
def load() -> dict:
    """Die Datei als Wörterbuch; leer, wenn sie fehlt (die App zeigt dann nur die Live-Teile)."""
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
