"""Messreihen der README (vorgerechnet). Jede Funktion liefert ein JSON-fähiges Wörterbuch; tools/precompute_lsc.py schreibt sie nach data/lsc_vorgerechnet.json,
die Tests rechnen kleine Fassungen mit denselben Funktionen nach. Zufall nur über lsc_rng.Rng (SplitMix64), Seeds stehen in SEEDS.
Alle Schrittweiten der Reihen haben die Verschiebung S_SHIFT gegen Zweierpotenzen und Schwellen."""

import numpy as np

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
import lsc_evaluation as E
from lsc_rng import Rng

SEEDS = {"asym_b": 4, "asym_c": 7, "v9g": 99, "hedge_rpe": 11, "hedge_random": 23, "vergessen": 101, "vergessen_fenster": 1, "vergessen_gitter": 11, "vergessen_alle": 11}
SH = C.S_SHIFT
ES5 = (-0.4, -0.2, 0.0, 0.2, 0.4)


def _f(x):
    return float(x)


# ------------------------------------------------------------------------------------------------ Diagonale

def reihe_bifurkation(n=60000):
    """Gitter s = 1 bis 59,5 (Schritt 0,5) plus S_SHIFT in beiden Beispielen; dazu das erste nicht konvergierende s auf dem feinen Gitter (Schritt 0,05)."""
    ss = np.arange(1.0, 60.0, 0.5) + SH
    fine = np.arange(0.5, 60.0, 0.05) + SH
    out = {}
    for key, (_name, a1, a2) in C.EXAMPLES.items():
        scan = E.bifurcation_scan(a1, a2, ss, n=n)
        scan["first_unstable_fine"] = A.first_unstable(a1, a2, fine)
        out[str(key)] = scan
    return out


def reihe_quer(n=60000):
    """Beispiel 2: Längs- und Querexponent der Diagonale bei s = 20, 26, 30, 40, 50; Wachstum eines Anfangsunterschieds bei s = 40."""
    _name, a1, a2 = C.EXAMPLES[2]
    rows = []
    for s in (20.0, 26.0, 30.0, 40.0, 50.0):
        lp, lq, _sp = A.diag_lyapunov(s + SH, a1, a2, burn=4000, n=n)
        rows.append({"s": s, "lam_par": _f(lp[0]), "lam_perp": _f(lq[0])})
    growth = {}
    for dk in (3, 6, 9, 12):
        steps, dist = A.steps_to_leave(a1, a2, 40.0 + SH, 10.0 ** (-dk))                    # nach 4000 Einschwingschritten (auf der Bahn)
        steps0, dist0 = A.steps_to_leave(a1, a2, 40.0 + SH, 10.0 ** (-dk), burn=0)          # sofort ab u = 0,37 (Übergang)
        growth[str(dk)] = {"steps": int(steps), "rate": A.growth_rate(dist), "steps_ohne_einschwingen": int(steps0), "rate_ohne_einschwingen": A.growth_rate(dist0)}
    return {"rows": rows, "growth": growth}


# ------------------------------------------------------------------------------------------------ Asymmetrie (2 × 2)

def _counts(o: A.Outcomes) -> dict:
    return {"n": int(o.pure.size), "pure": int(o.pure.sum()), "still": int(o.still.sum()), "other": int(o.other.sum()), "chaos": int(o.chaos.sum()), "on_diagonal": int(o.on_diagonal.sum())}


def reihe_asym_b(n=400):
    """Unsymmetrische Starts in beiden Beispielen (sechs Konfigurationen, je n Starts, Logits in [−3, 3]²)."""
    rng = Rng(SEEDS["asym_b"])
    rows = []
    for key, svals in ((1, (5.0, 10.0, 40.0)), (2, (10.0, 20.0, 40.0))):
        _name, a1, a2 = C.EXAMPLES[key]
        for s in svals:
            u = rng.uniform((2, n), -C.START_BOX, C.START_BOX)
            o = A.run2(u[0], u[1], A.Pair.symmetric(a1, a2, s + SH))
            rows.append({"beispiel": key, "s": s, **_counts(o)})
    return {"rows": rows}


def reihe_asym_c(n=2000, symmetric_n=100):
    """Beispiel 2, s₁ = 26 und 40: Spieler 2 mit Schrittweite s₁·(1 + δ) oder Kosten auf Kante 2 a₂·(1 + δ), δ = 10⁻⁹ bis 0,1: je n Starts.
    Dazu je Variante und s₁ eine Zeile δ = 0 mit `symmetric_n` Starts exakt auf der Diagonale."""
    _name, a1, a2 = C.EXAMPLES[2]
    rng = Rng(SEEDS["asym_c"])
    rows = []
    for variant in ("Schrittweite", "Kosten Kante 2"):
        for s1 in (26.0, 40.0):
            for delta in C.DELTAS:
                u = rng.uniform((2, n), -C.START_BOX, C.START_BOX)
                u1, u2 = u[0], u[1].copy()
                if delta == 0.0:
                    u2[:symmetric_n] = u1[:symmetric_n]
                P = A.Pair.with_deviation(a1, a2, s1 + SH, delta if variant == "Schrittweite" else 0.0, delta if variant == "Kosten Kante 2" else 0.0)
                o = A.run2(u1, u2, P)
                row = {"variante": variant, "s1": s1, "delta": delta, **_counts(o)}
                if delta == 0.0:
                    row["on_diagonal_sym"] = int(o.on_diagonal[:symmetric_n].sum())
                    row["chaos_sym"] = int(o.chaos[:symmetric_n].sum())
                rows.append(row)
    return {"rows": rows}


def reihe_v9g(k=20000):
    """Gegenprobe der Absolutaussage: k zufällige Spiele mit eigenen Kosten a1, a2, b1, b2 ∈ (0,05; 1) und eigenen Schrittweiten ∈ (2; 80), Start-Logits in [−3, 3]², unsymmetrisch (|u1 − u2| ≥ 0,01)."""
    rng = Rng(SEEDS["v9g"])
    a1 = rng.uniform(k, 0.05, 1.0)
    a2 = rng.uniform(k, 0.05, 1.0)
    b1 = rng.uniform(k, 0.05, 1.0)
    b2 = rng.uniform(k, 0.05, 1.0)
    s1 = rng.uniform(k, 2.0, 80.0)
    s2 = rng.uniform(k, 2.0, 80.0)
    rows = []
    for name in ("jeder Spieler eigene Kosten und Schrittweite", "gleiche Kosten, nur Schrittweiten verschieden", "gleiche Kosten und Schrittweite (nur der Start unsymmetrisch)"):
        B1, B2, S2 = (b1, b2, s2) if name.startswith("jeder") else (a1, a2, s2)
        if name.startswith("gleiche Kosten und"):
            S2 = s1
        u1 = rng.uniform(k, -C.START_BOX, C.START_BOX)
        u2 = rng.uniform(k, -C.START_BOX, C.START_BOX)
        u2 = np.where(np.abs(u1 - u2) < 0.01, u1 + 0.5, u2)
        o = A.run2(u1, u2, A.Pair(a1, a2, s1, B1, B2, S2), burn=6000, meas=2000)
        row = {"variante": name, **_counts(o)}
        # Bahnen, die weder still stehen noch rein enden, noch einmal mit langem Einschwingen: sie sind langsame Übergänge, kein Chaos
        idx = np.nonzero(o.other)[0]
        if idx.size:
            o2 = A.run2(u1[idx], u2[idx], A.Pair(a1[idx], a2[idx], s1[idx], B1[idx], B2[idx], S2[idx]), burn=30000, meas=5000)
            row["nachpruefung"] = {"n": int(idx.size), "pure": int(o2.pure.sum()), "still": int(o2.still.sum()), "other": int(o2.other.sum()), "chaos": int(o2.chaos.sum())}
        rows.append(row)
    return {"rows": rows}


# ------------------------------------------------------------------------------------------------ Nullsumme

def reihe_nullsumme(steps=100000, quotient_steps=20000):
    rows = []
    R0 = B.rps(0.0)
    for eps in C.EPS_VALUES:
        vol, kl, minp, _trace = B.zero_sum_run(R0, eps, steps=steps, burn=min(5000, steps // 4))
        rows.append({"eps": eps, "vol": _f(vol), "kl": _f(kl), "minp": _f(minp)})
    quot = []
    p = np.array([[0.5, 0.2, -0.1]])
    q = np.array([[-0.3, 0.1, 0.4]])
    for s in (0.01, 0.1, 1.0):
        lam, minp, _rg = B.run_games(np.array([R0]), np.array([R0]), s, 0.0, p, q, burn=2000, meas=quotient_steps)
        quot.append({"s": s, "lam": _f(lam[0]), "minp_log10": _f(np.log10(max(minp[0], 1e-300)))})
    return {"rows": rows, "quotient": quot, "steps": steps, "quotient_steps": quotient_steps}


# ------------------------------------------------------------------------------------------------ 3 × 3

def _games_rpe(es, S):
    games = [(e1, e2) for e1 in es for e2 in es]
    A_ = np.array([B.rps(e1) for e1, e2 in games for _ in range(S)])
    C_ = np.array([B.rps(e2) for e1, e2 in games for _ in range(S)])
    return games, A_, C_


def reihe_hedge_rpe(S=40, svals=(0.3, 1.0, 3.0, 10.0), burn=4000, meas=4000):
    """Reines Hedge in R + e·I (25 Spiele e₁, e₂ ∈ {−0,4 … 0,4}, S Starts je Spiel): Anteile je Zelle und Summen."""
    games, A_, C_ = _games_rpe(ES5, S)
    rng = Rng(SEEDS["hedge_rpe"])
    nb = len(A_)
    p0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
    q0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
    out = []
    for s in svals:
        lam, minp, rg = B.run_games(A_, C_, s + SH, 0.0, p0, q0, burn=burn, meas=meas)
        conv, chaos, oth = B.classify(lam, minp, rg)
        loose = (~conv) & (lam > C.CHAOS_LAMBDA)
        cells = [{"e1": e1, "e2": e2, "conv": _f(conv[i * S:(i + 1) * S].mean()), "chaos": _f(chaos[i * S:(i + 1) * S].mean()), "other": _f(oth[i * S:(i + 1) * S].mean())}
                 for i, (e1, e2) in enumerate(games)]
        out.append({"s": s, "n": nb, "conv": int(conv.sum()), "chaos": int(chaos.sum()), "other": int(oth.sum()), "loose": int(loose.sum()), "cells": cells})
    return {"S": S, "rows": out, "burn": burn, "meas": meas}


def reihe_hedge_random(G=300, S=8, gammas=(-0.8, -0.4, 0.0, 0.4), svals=(0.5, 2.0, 8.0), burn=3000, meas=3000):
    rng = Rng(SEEDS["hedge_random"])
    rows = []
    for gamma in gammas:
        As, Cs = [], []
        for _ in range(G):
            a, c = B.random_game(rng, 3, gamma)
            As.append(a)
            Cs.append(c)
        A_ = np.repeat(np.array(As), S, axis=0)
        C_ = np.repeat(np.array(Cs), S, axis=0)
        nb = len(A_)
        p0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
        q0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
        for s in svals:
            lam, minp, rg = B.run_games(A_, C_, s + SH, 0.0, p0, q0, burn=burn, meas=meas)
            conv, chaos, oth = B.classify(lam, minp, rg)
            rows.append({"gamma": gamma, "s": s, "n": nb, "conv": int(conv.sum()), "chaos": int(chaos.sum()), "other": int(oth.sum())})
    return {"rows": rows, "G": G, "S": S}


# ------------------------------------------------------------------------------------------------ Vergessen

def _vergessen_run(e1, e2, s, g, n, seed, burn=6000, meas=6000, tangent_seed=0):
    rng = Rng(seed)
    A_ = np.array([B.rps(e1)] * n)
    C_ = np.array([B.rps(e2)] * n)
    p0 = rng.uniform((n, 3), -C.START_BOX_3, C.START_BOX_3)
    q0 = rng.uniform((n, 3), -C.START_BOX_3, C.START_BOX_3)
    lam, minp, rg = B.run_games(A_, C_, s + SH, g, p0, q0, burn=burn, meas=meas, tangent_seed=tangent_seed)
    return A_, C_, p0, q0, lam, minp, rg


def vergessen_anteil(e1, e2, s, g, n=100, seed=1, meas=6000):
    """Anteil chaotischer und konvergenter Starts bei Vergessensrate g (e1, e2: Spiel; s ohne Verschiebung)."""
    _A, _C, _p, _q, lam, minp, rg = _vergessen_run(e1, e2, s, g, n, seed, meas=meas)
    conv, chaos, _oth = B.classify(lam, minp, rg)
    return _f(chaos.mean()), _f(conv.mean())


def reihe_vergessen(n=400, n_fenster=100, n_gitter=40, meas_list=(2000, 6000, 18000), n_alle=40):
    e1, e2, s, g = 0.0, 0.4, 0.5, 0.02
    main = {"n": n, "meas": []}
    for meas in meas_list:
        A_, C_, p0, q0, lam, minp, rg = _vergessen_run(e1, e2, s, g, n, SEEDS["vergessen"], meas=meas, tangent_seed=meas)
        conv, chaos, _oth = B.classify(lam, minp, rg)
        main["meas"].append({"meas": meas, "chaos": int(chaos.sum()), "conv": int(conv.sum()), "lam_q": [_f(v) for v in np.quantile(lam, (0, 0.1, 0.5, 0.9, 1))]})
    m2 = min(n, 100)
    lam2 = B.two_orbit_lyapunov(A_[:m2], C_[:m2], s + SH, g, p0[:m2], q0[:m2])
    lam1, minp1, rg1 = B.run_games(A_[:m2], C_[:m2], s + SH, g, p0[:m2], q0[:m2], burn=6000, meas=6000, tangent_seed=3)
    main["two_orbit"] = {"n": m2, "lam_two": [_f(v) for v in np.quantile(lam2, (0, 0.5, 1))], "lam_tangent": [_f(v) for v in np.quantile(lam1, (0, 0.5, 1))], "max_diff": _f(np.abs(lam1 - lam2).max())}
    with np.errstate(divide="ignore"):
        mp = np.log10(np.maximum(minp, 1e-300))
    main["minp_log10_q"] = [_f(v) for v in np.quantile(mp, (0, 0.5, 1))]
    main["spanne_median"] = _f(np.median(rg))
    g_sweep = []
    for gg in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        ch, cv = vergessen_anteil(e1, e2, s, gg, n_fenster, SEEDS["vergessen_fenster"])
        g_sweep.append({"g": gg, "chaos": ch, "conv": cv})
    s_sweep = []
    for ss in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5):
        ch, cv = vergessen_anteil(e1, e2, ss, g, n_fenster, SEEDS["vergessen_fenster"])
        s_sweep.append({"s": ss, "chaos": ch, "conv": cv})
    es1 = (-0.2, -0.1, 0.0, 0.1, 0.2)
    es2 = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    grid = [[vergessen_anteil(a, b, s, g, n_gitter, SEEDS["vergessen_gitter"])[0] for b in es2] for a in es1]
    nullsumme = [{"e2": b, "chaos": vergessen_anteil(-b, b, s, g, n_fenster, SEEDS["vergessen_fenster"])[0]} for b in (0.2, 0.4, 0.6)]
    # andere Konfigurationen über alle 25 Spiele × n_alle Starts (g, s)
    games, A_, C_ = _games_rpe(ES5, n_alle)
    rng = Rng(SEEDS["vergessen_alle"])
    nb = len(A_)
    p0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
    q0 = rng.uniform((nb, 3), -C.START_BOX_3, C.START_BOX_3)
    alle = []
    for gg in (0.02, 0.1):
        for ss in (0.5, 2.0, 8.0):
            lam, minp, rg = B.run_games(A_, C_, ss + SH, gg, p0, q0, burn=6000, meas=6000)
            conv, chaos, oth = B.classify(lam, minp, rg)
            by_cell = [int(chaos[i * n_alle:(i + 1) * n_alle].sum()) for i in range(len(games))]
            alle.append({"g": gg, "s": ss, "n": nb, "chaos": int(chaos.sum()), "conv": int(conv.sum()), "cells_with_chaos": int(sum(1 for c in by_cell if c > 0))})
    return {"main": main, "g_sweep": g_sweep, "s_sweep": s_sweep, "grid": {"e1": list(es1), "e2": list(es2), "chaos": grid}, "nullsumme_kontrolle": nullsumme, "alle": alle,
            "game": {"e1": e1, "e2": e2, "s": s, "g": g}}
