"""Rechnet die Messreihen der README vor und schreibt data/lsc_vorgerechnet.json (reproduzierbar: feste Zufalls-Seeds, SplitMix64).

    python tools/precompute_lsc.py            # alle Reihen (einige Minuten)
    python tools/precompute_lsc.py bifurkation quer   # nur die genannten Reihen (die übrigen bleiben aus der vorhandenen Datei)

Die App und die README nennen diese Zahlen „vorgerechnet“. Die Tests rechnen kleine Fassungen mit denselben Funktionen aus lsc_messreihen.py nach."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lsc_messreihen as M  # noqa: E402

OUT = ROOT / "data" / "lsc_vorgerechnet.json"
REIHEN = {
    "bifurkation": M.reihe_bifurkation,
    "quer": M.reihe_quer,
    "asym_b": M.reihe_asym_b,
    "asym_c": M.reihe_asym_c,
    "v9g": M.reihe_v9g,
    "nullsumme": M.reihe_nullsumme,
    "hedge_rpe": M.reihe_hedge_rpe,
    "hedge_random": M.reihe_hedge_random,
    "vergessen": M.reihe_vergessen,
}


def main(names):
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names or REIHEN:
        t0 = time.time()
        data[name] = REIHEN[name]()
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    data["meta"] = {"seeds": M.SEEDS, "python": platform.python_version(), "numpy": np.__version__, "s_shift": M.SH,
                    "hinweis": "vorgerechnet mit tools/precompute_lsc.py; Zufall: SplitMix64 (lsc_rng.py)"}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("geschrieben:", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
