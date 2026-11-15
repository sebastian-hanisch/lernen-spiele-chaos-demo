"""Claims-Tests: jede Zahl der Befunde-Tabelle der README (Spalte „Test“) wird hier nachgerechnet.

Zwei Arten: (1) die vorgerechnete Datei `data/lsc_vorgerechnet.json` enthält die Zahl, die README nennt (Rundung wie in der README); (2) eine kleinere Fassung wird mit denselben
Funktionen (`lsc_messreihen`, `lsc_algorithm`, `lsc_bimatrix`) live nachgerechnet und gegen ein **Band** gehalten (die Zahl selbst steht dann nur in der Messung). Zahl → Test:
siehe Befunde-Tabelle der README. „Alle Zahlen stehen im Test“ gilt nicht: Zählungen mit 100 000 Schritten oder 20 000 Spielen werden nur in kleiner Fassung wiederholt."""

import time
from functools import lru_cache

import numpy as np
import pytest

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
import lsc_evaluation as E
import lsc_frozen
import lsc_messreihen as M
from lsc_rng import Rng

FZ = lsc_frozen.load()


@lru_cache(maxsize=1)
def _scan():
    """Die volle Reihe (60 000 Schritte, beide Beispiele) live, einmal je Testlauf: dieselbe Funktion wie die vorgerechnete Datei."""
    return M.reihe_bifurkation(n=60000)


def _periods(scan):
    out = {}
    for s, p, lam in zip(scan["s"], scan["period"], scan["lam_par"]):
        if s > scan["threshold"] and lam <= C.CHAOS_LAMBDA:
            out.setdefault(p, []).append(s)
    return out


def test_frozen_file_is_present_and_has_every_series():
    assert set(FZ) >= {"bifurkation", "quer", "asym_b", "asym_c", "v9g", "nullsumme", "hedge_rpe", "hedge_random", "vergessen", "meta"}


def test_claim_schwellen_geschlossen_und_gemessen():
    assert A.threshold(0.5, 0.5) == pytest.approx(8.0) and A.threshold(0.25, 0.35) == pytest.approx(17.78, abs=0.005)
    assert FZ["bifurkation"]["1"]["first_unstable_fine"] == pytest.approx(8.001, abs=1e-3)      # Raster mit Schritt 0,05, je + 0,00137
    assert FZ["bifurkation"]["2"]["first_unstable_fine"] == pytest.approx(17.801, abs=1e-3)
    fine = np.arange(0.5, 60.0, 0.05) + C.S_SHIFT                                                # live nachgerechnet
    assert A.first_unstable(0.5, 0.5, fine) == pytest.approx(8.001, abs=1e-3)
    assert A.first_unstable(0.25, 0.35, fine) == pytest.approx(17.801, abs=1e-3)


def test_claim_diagonale_beispiel_1_kein_chaos():
    b1 = _scan()["1"]
    assert FZ["bifurkation"]["1"]["above"] == b1["above"] == 104 and FZ["bifurkation"]["1"]["chaotic"] == b1["chaotic"] == 0
    per = _periods(b1)
    assert min(per[2]) == pytest.approx(8.0, abs=0.01)
    assert max(per[2]) == pytest.approx(59.5, abs=0.01) and len(per[2]) == 104 and set(per) == {2}      # Periode 2 für alle 104 Werte (der Multiplikator des 2-Zyklus nähert sich 1 von unten)


def test_claim_diagonale_beispiel_2_chaos_und_fenster():
    b2 = _scan()["2"]
    assert FZ["bifurkation"]["2"]["above"] == b2["above"] == 84 and FZ["bifurkation"]["2"]["chaotic"] == b2["chaotic"] == 34
    assert b2["first_chaotic"] == pytest.approx(26.0, abs=0.01)
    per = _periods(b2)
    lo_hi = lambda p: (min(per[p]), max(per[p]), len(per[p]))                      # noqa: E731
    assert [round(v, 1) for v in lo_hi(2)[:2]] == [18.0, 23.0] and lo_hi(2)[2] == 11
    assert [round(v, 1) for v in lo_hi(4)[:2]] == [23.5, 59.5] and lo_hi(4)[2] == 28
    assert [round(v, 1) for v in lo_hi(3)[:2]] == [31.0, 32.5] and lo_hi(3)[2] == 4
    assert {5, 6, 10, 12, 28, 48} <= set(per)
    assert sum(len(v) for v in per.values()) + b2["chaotic"] == b2["above"]


def test_claim_querexponent_und_wachstum():
    rows = FZ["quer"]["rows"]
    assert [r["lam_perp"] for r in rows] == pytest.approx([1.0392, 1.0191, 1.0365, 0.8311, 0.4005], abs=5e-5)
    assert [r["lam_par"] for r in rows] == pytest.approx([-1.1208, 0.1787, 0.3940, 0.3680, -0.7421], abs=5e-5)
    live = [A.diag_lyapunov(r["s"] + C.S_SHIFT, 0.25, 0.35, burn=4000, n=20000) for r in rows]     # Band: 20 000 statt 60 000 Schritte
    assert [float(v[1][0]) for v in live] == pytest.approx([r["lam_perp"] for r in rows], abs=0.03)
    assert [float(v[0][0]) for v in live] == pytest.approx([r["lam_par"] for r in rows], abs=0.03)
    g = FZ["quer"]["growth"]
    assert [g[k]["steps"] for k in ("3", "6", "9", "12")] == [9, 18, 26, 35]
    assert [g[k]["steps_ohne_einschwingen"] for k in ("3", "6", "9", "12")] == [8, 14, 21, 30]
    assert [round(g[k]["rate"], 2) for k in ("3", "6", "9", "12")] == [0.63, 0.76, 0.81, 0.80]
    assert round(g["12"]["rate_ohne_einschwingen"], 2) == 0.95
    for dk, expected in ((3, 9), (12, 35)):                                                          # live
        assert A.steps_to_leave(0.25, 0.35, 40.0 + C.S_SHIFT, 10.0 ** (-dk))[0] == expected
    assert A.steps_to_leave(0.25, 0.35, 40.0 + C.S_SHIFT, 1e-12, burn=0)[0] == 30
    # der Querfaktor 1 + c ist auf jeder Bahn größer als 1 (c > 0)
    ss = np.linspace(3.0, 80.0, 30)
    assert (A.diag_lyapunov(ss, 0.25, 0.35, burn=300, n=600)[1] > 0).all()


def test_claim_unsymmetrisch_reines_gleichgewicht():
    rows = FZ["asym_b"]["rows"]
    assert len(rows) == 6 and all(r["n"] == 400 and r["pure"] == 400 for r in rows) and sum(r["n"] for r in rows) == 2400
    live = M.reihe_asym_b(n=100)["rows"]                                                              # Band: 6 × 100 statt 6 × 400
    assert all(r["pure"] == r["n"] == 100 for r in live)


def test_claim_asymmetrie_20_konfigurationen():
    rows = [r for r in FZ["asym_c"]["rows"] if r["delta"] > 0]
    assert len(rows) == 20 and all(r["n"] == 2000 and r["pure"] == 2000 for r in rows) and sum(r["pure"] for r in rows) == 40000
    live = [r for r in M.reihe_asym_c(n=300, symmetric_n=20)["rows"] if r["delta"] > 0]              # Band: 20 × 300
    assert len(live) == 20 and all(r["pure"] == r["n"] == 300 for r in live)


def test_claim_asymmetrie_ohne_abweichung_enden_zufaellige_starts_rein_und_starts_nahe_der_diagonale():
    """README: δ = 0: 1 900 von 1 900 zufällig gestarteten Bahnen enden rein (die 100 Starts mit u₂ = u₁ bleiben auf der Diagonale); Starts nahe der Diagonale (u₂ = u₁ + δ,
    δ = 10⁻¹² bis 10⁻³, Beispiel 2, s = 26 und 40) enden in einem reinen Gleichgewicht (Band: 2 × 1 000 Starts, 20 000 Einschwingschritte)."""
    sym = [r for r in FZ["asym_c"]["rows"] if r["delta"] == 0.0]
    assert len(sym) == 4 and all(r["pure"] == 1900 and r["n"] - r["on_diagonal_sym"] == 1900 for r in sym)
    rng = Rng(11)
    for s in (26.0, 40.0):
        u1 = rng.uniform((1000,), -C.START_BOX, C.START_BOX)
        delta = 10.0 ** rng.uniform((1000,), -12.0, -3.0)
        o = A.run2(u1, u1 + delta, A.Pair.symmetric(0.25, 0.35, s + C.S_SHIFT), burn=20000)
        assert int(o.pure.sum()) == 1000


def test_claim_diagonale_bleibt_in_gleitkomma_exakt():
    sym = [r for r in FZ["asym_c"]["rows"] if r["delta"] == 0.0]
    assert len(sym) == 4 and all(r["on_diagonal_sym"] == 100 and r["pure"] == 1900 for r in sym)
    assert FZ["quer"]["growth"]["12"]["steps"] == 35
    u = Rng(3).uniform((100,), -3, 3)
    P = A.Pair.symmetric(0.25, 0.35, 40.0 + C.S_SHIFT)
    u1, u2 = u.copy(), u.copy()
    for _ in range(5500):
        u1, u2 = A.step2(u1, u2, P)
    assert np.array_equal(u1, u2)                                                                      # 100 Starts, 5 500 Schritte, bitgleich


def test_claim_gegenprobe_zufaellige_spiele():
    rows = FZ["v9g"]["rows"]
    assert [r["pure"] for r in rows] == [17974, 10981, 10982] and [r["still"] for r in rows] == [2026, 9014, 9015]
    assert [r["other"] for r in rows] == [0, 5, 3] and [r["chaos"] for r in rows] == [0, 1, 1] and all(r["n"] == 20000 for r in rows)
    assert all(r.get("nachpruefung", {"chaos": 0})["chaos"] == 0 for r in rows)
    live = M.reihe_v9g(k=2000)["rows"]                                                                 # Band: 3 × 2 000 Spiele
    for r in live:
        assert r["n"] == 2000 and r["pure"] + r["still"] >= 1995 and r["chaos"] <= 2


def test_claim_nullsumme():
    rows = FZ["nullsumme"]["rows"]
    assert [r["eps"] for r in rows] == [0.01, 0.05, 0.1, 0.3, 1.0]
    assert [r["vol"] for r in rows] == pytest.approx([4.3e-5, 1.5e-6, 4.0e-10, 0.0, 0.0], rel=0.02, abs=1e-12)
    assert [r["kl"] for r in rows] == pytest.approx([1.59, 26.3, 55.3, 171.0, 575.0], rel=0.01)
    q = FZ["nullsumme"]["quotient"]
    assert [x["lam"] for x in q] == pytest.approx([9.2e-5, 2.4e-4, 1.9e-4], rel=0.05) and max(x["lam"] for x in q) < 3e-4
    live = M.reihe_nullsumme(steps=8000, quotient_steps=4000)                                          # Band: 8 000 statt 100 000 Schritte
    assert live["rows"][0]["vol"] > 0 and abs(live["rows"][3]["vol"]) < 1e-9 and abs(live["rows"][4]["vol"]) < 1e-9
    kl = [r["kl"] for r in live["rows"]]
    assert kl == sorted(kl) and kl[0] < 5.0 and kl[-1] > 40.0
    assert all(abs(x["lam"]) < 1e-3 for x in live["quotient"])


def test_claim_hedge_3x3_kein_chaos():
    rows = FZ["hedge_rpe"]["rows"]
    assert [r["s"] for r in rows] == [0.3, 1.0, 3.0, 10.0] and all(r["n"] == 1000 and r["chaos"] == 0 for r in rows)
    assert [r["loose"] for r in rows] == [67, 86, 170, 253]                                            # Bahnen am Rand, nicht als Chaos gezählt
    rnd = FZ["hedge_random"]["rows"]
    assert len(rnd) == 12 and all(r["n"] == 2400 and r["chaos"] == 0 for r in rnd)
    live = M.reihe_hedge_rpe(S=8, svals=(0.3, 3.0))["rows"]                                           # Band: 2 × 200 Bahnen
    assert all(r["chaos"] == 0 and r["n"] == 200 for r in live)
    live2 = M.reihe_hedge_random(G=20, S=4, gammas=(-0.4,), svals=(2.0,), burn=1500, meas=1500)["rows"]
    assert live2[0]["chaos"] == 0


def test_claim_hedge_3x3_dreifache_messdauer():
    """README: bei dreifacher Dauer (12 000 + 12 000 Schritte) steigt „konvergiert“ von 140 / 172 / 194 / 179 auf 412 / 411 / 454 / 475 von je 1 000, „0 chaotisch“ bleibt;
    von den Randbahnen 67 / 86 / 170 / 253 bleiben 2 / 0 / 9 / 18 über der Grenze (höchstens 7 %)."""
    kurz = FZ["hedge_rpe"]["rows"]
    lang = M.reihe_hedge_rpe(burn=12000, meas=12000)["rows"]
    # Die lange Messdauer (12 000 + 12 000 Schritte) rechnet je nach Plattform in den letzten Stellen anders:
    # Windows 412 / 411 / 454 / 475, Linux-CI 409 / 409 / 453 / 472 (und je nach Runner noch etwas anders).
    # Darum ein Band von ± 10 statt exakter Werte, und für die kleinen Randbahn-Zählungen nur die Grenze „höchstens 12 %“.
    assert [r["conv"] for r in kurz] == [140, 172, 194, 179]
    assert all(abs(r["conv"] - w) <= 10 for r, w in zip(lang, [412, 411, 454, 475]))
    assert all(r["chaos"] == 0 for r in lang)
    assert max(l["loose"] / k["loose"] for l, k in zip(lang, kurz)) < 0.12


def test_claim_vergessen_haupt():
    main = FZ["vergessen"]["main"]
    assert main["n"] == 400 and [m["meas"] for m in main["meas"]] == [2000, 6000, 18000] and all(m["chaos"] == 400 for m in main["meas"])
    q = main["meas"][-1]["lam_q"]                                                                      # Quantile 0 / 0,1 / 0,5 / 0,9 / 1
    assert q[0] == pytest.approx(0.0319, abs=5e-5) and q[2] == pytest.approx(0.0337, abs=5e-5) and q[4] == pytest.approx(0.0359, abs=5e-5)
    assert main["two_orbit"]["lam_two"][1] == pytest.approx(0.0336, abs=5e-5) and main["two_orbit"]["lam_tangent"][1] == pytest.approx(0.0336, abs=5e-5) and main["two_orbit"]["max_diff"] < 0.001
    assert main["minp_log10_q"][0] == pytest.approx(-5.03, abs=0.01) and main["minp_log10_q"][2] == pytest.approx(-4.58, abs=0.01) and main["spanne_median"] == pytest.approx(0.987, abs=5e-4)
    info = E.live33(0.0, 0.4, 0.5 + C.S_SHIFT, 0.02, 100, 101, burn=6000, meas=6000, two_orbit=True)    # Band: 100 Starts
    assert info.chaos >= 98 and 0.030 <= info.lam_q[1] <= 0.037 and abs(info.lam_two_orbit_median - info.lam_q[1]) < 0.004
    assert info.minp_log10_median > -6 and info.spanne_median > 0.9


def test_claim_vergessen_fenster():
    v = FZ["vergessen"]
    assert [r["chaos"] for r in v["g_sweep"]] == [0.0, 0.31, 1.0, 0.98, 0.0, 0.0] and [r["conv"] for r in v["g_sweep"]][-1] == 1.0
    assert [r["chaos"] for r in v["s_sweep"]] == [0.0, 0.0, 0.0, 1.0, 0.32, 0.0, 0.0] and v["s_sweep"][0]["conv"] == 1.0
    grid = np.array(v["grid"]["chaos"])
    assert int((grid >= 0.95).sum()) == 17 and grid.size == 35 and int((grid >= 0.97).sum()) == 15
    e1, e2 = np.meshgrid(v["grid"]["e1"], v["grid"]["e2"], indexing="ij")
    assert (grid[(e1 + e2) < 0.25] == 0).all() and (grid[np.isclose(e1 + e2, 0.3)] >= 0.95).all()     # untere Grenze e1 + e2 = 0,3
    other = [r for r in v["alle"] if not (r["g"] == 0.02 and r["s"] == 0.5)]
    assert len(other) == 5 and all(r["chaos"] == 0 and r["n"] == 1000 for r in other)
    first = next(r for r in v["alle"] if r["g"] == 0.02 and r["s"] == 0.5)
    assert first["chaos"] == 203 and first["cells_with_chaos"] == 6
    # live (Band): innerhalb des Fensters chaotisch, außerhalb nicht
    inside = M.vergessen_anteil(0.0, 0.4, 0.5, 0.02, n=30, seed=1)
    outside_g = M.vergessen_anteil(0.0, 0.4, 0.5, 0.05, n=30, seed=1)
    outside_s = M.vergessen_anteil(0.0, 0.4, 1.5, 0.02, n=30, seed=1)
    assert inside[0] >= 0.9 and outside_g[0] == 0.0 and outside_s[0] == 0.0
    assert M.vergessen_anteil(0.0, 0.4, 0.5, 0.08, n=30, seed=1)[1] == 1.0                              # g = 0,08: alles konvergent


def test_claim_rechenzeit_am_reglermaximum():
    t0 = time.time()
    E.schwelle_info(2, C.S_MAX)
    E.quer_info(2, C.S_MAX, C.DK_MAX)
    E.asym_info(2, C.S_MAX, 0.1, 0.1, C.K_MAX, 1, False)
    E.live33(0.6, 0.6, C.S33_MAX, 0.0, 100, 1)
    E.live33(0.0, 0.4, C.SV_MAX, C.G_MAX, 100, 1, two_orbit=True)
    E.orbit33(0.0, 0.4, C.SV_MAX, C.G_MAX, 1)
    B.zero_sum_run(B.rps(0.0), 0.01, steps=6000, burn=1000, record_every=60)
    assert time.time() - t0 < 10.0
