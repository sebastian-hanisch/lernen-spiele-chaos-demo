"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier; dazu je Preset die erwartete Verdikt-Zeile
(DEMO-PLAYBOOK §11a Punkt 7): ein Preset ohne Effekt darf keinen Effekt melden, und das Preset „Mit Vergessen“ darf keine Hedge-Aussage machen."""

import numpy as np
import pytest

import lsc_constants as C
import lsc_evaluation as E


def _info(regime, s=10.0, c=2.5, period=2, lam_par=-0.2, xs=0.5, thr=8.0):
    return E.SchwelleInfo(1, s, xs, thr, c, lam_par, 1.1, period, 0.7, regime)


def test_schwelle_stable_names_threshold_and_factor():
    kind, text = E.verdict_schwelle(_info("stabil", s=7.99, c=1.998), None)
    assert kind == "success" and "unter der Schwelle 8,00" in text and "x* = 0,50" in text and "Faktor 1 − c = -0,998" in text


def test_schwelle_unstable_regimes_and_prefilled_counts():
    kind, text = E.verdict_schwelle(_info("zyklus", s=10.001, c=2.5003, period=2), None)
    assert kind == "info" and "instabil" in text and "Periode 2" in text and "Faktor 1 − c = -1,500" in text
    kind, text = E.verdict_schwelle(_info("chaos", lam_par=0.3696), {"first_chaotic": 26.00137, "chaotic": 34, "above": 84})
    assert "chaotisch auf der Diagonale" in text and "0,370" in text and "erste chaotische s = 26,00 (34 von 84" in text
    kind, text = E.verdict_schwelle(_info("lang", lam_par=0.0004, period=None), {"first_chaotic": None, "chaotic": 0, "above": 104})
    assert "nicht chaotisch gemessen" in text and "keiner von 104 Werten" in text
    assert "längere Periode oder eine sehr langsame Annäherung an einen Zyklus" in text and "quasiperiodisch" not in text


def test_schwelle_example_1_above_44_reports_the_period_two_cycle_not_a_long_period():
    """Fehler vor der Korrektur: Beispiel 1 bei s = 50,001 wurde als „lange Periode oder quasiperiodisch“ gedeutet, obwohl ein stabiler 2-Zyklus vorliegt."""
    for s in (44.001, 50.001, 59.501):
        i = E.schwelle_info(1, s, n=2000)
        kind, text = E.verdict_schwelle(i, None)
        assert i.regime == "zyklus" and i.period == 2 and "Die Bahn auf der Diagonale hat die Periode 2" in text and "quasiperiodisch" not in text


def test_quer_verdict_names_growth_steps_and_both_exponents():
    q = E.QuerInfo(40.001, 12, 1e-12, 30, 0.8265, 0.83, 0.37, np.array([1e-12, 1.0]))
    kind, text = E.verdict_quer(q)
    assert kind == "info" and "10⁻¹²" in text and "0,83 je Schritt" in text and "nach 30 Schritten" in text and "Querexponent entlang der Diagonale ist 0,83" in text and "Sattel" in text
    kind, text = E.verdict_quer(E.QuerInfo(5.0, 9, 1e-9, 400, float("nan"), 0.0, 0.0, np.array([1e-9])))
    assert kind == "warning" and "10⁻⁹" in text and "nicht auf 1" in text


def test_asym_verdict_branches_and_counts():
    a = E.AsymInfo(400, 400, 0, 0, 0, 0, False, 0.0, 40.001, 0.001, 0.0)
    kind, text = E.verdict_asym(a)
    assert kind == "success" and "400 von 400 Starts enden in einem reinen Gleichgewicht" in text and "nur auf der symmetrischen Diagonale gemessen" in text
    kind, text = E.verdict_asym(E.AsymInfo(400, 300, 20, 80, 5, 0, False, 0.1, 40.001, 0.001, 0.0))
    assert kind == "info" and "300 von 400" in text and "20 stehen" in text and "80 kreisen" in text and "davon λ > 0,003: 5" in text and "übrig bleiben 100 von 400" in text
    kind, text = E.verdict_asym(E.AsymInfo(100, 0, 0, 100, 100, 100, True, 0.83, 40.001, 0.0, 0.0))
    assert kind == "info" and "100 von 100 bleiben dort" in text and "0,83" in text and "bitgleich" in text


def test_hedge33_verdict_distinguishes_zero_and_positive_chaos_counts():
    base = dict(n=100, conv=10, other=90, lam_q=(0.0, 0.0, 0.0), minp_log10_median=-5.0, spanne_median=0.5, g=0.0, s=3.001, e1=0.0, e2=0.4, lam_two_orbit_median=None)
    kind, text = E.verdict_hedge33(E.Live33(chaos=0, **base), {"chaos_total": 0, "n_total": 4000})
    assert kind == "success" and "10 konvergiert" in text and "0 chaotisch im Inneren" in text and "0 von 4 000" in text and "kein Beweis" in text
    kind, text = E.verdict_hedge33(E.Live33(chaos=3, **{**base, "other": 87}), None)
    assert kind == "warning" and "3 chaotisch im Inneren" in text


def test_vergessen_verdict_never_claims_hedge_and_refuses_zero_forgetting():
    base = dict(n=100, conv=0, other=0, lam_q=(0.03, 0.034, 0.037), minp_log10_median=-4.7, spanne_median=0.98, s=0.501, e1=0.0, e2=0.4, lam_two_orbit_median=0.034)
    kind, text = E.verdict_vergessen(E.Live33(chaos=100, g=0.02, **base), None)
    assert kind == "success" and "nicht mehr Hedge" in text and "100 von 100 Starts chaotisch" in text and "0,034" in text and "schmalen Parameterfenster" in text
    kind, text = E.verdict_vergessen(E.Live33(chaos=40, g=0.02, **{**base, "other": 60}), None)
    assert kind == "info" and "Am Rand des Fensters" in text
    kind, text = E.verdict_vergessen(E.Live33(chaos=0, g=0.05, **{**base, "other": 100}), None)
    assert kind == "info" and "Außerhalb des Fensters" in text
    kind, text = E.verdict_vergessen(E.Live33(chaos=0, g=0.0, **{**base, "other": 100}), None)
    assert kind == "warning" and "reines Hedge" in text


def test_nullsumme_verdict_uses_the_frozen_row_and_a_missing_row_warns():
    kind, text = E.verdict_nullsumme(0.1, {"vol": 4.01e-10, "kl": 55.3})
    assert kind == "info" and "ε = 0,10" in text and "+4,0e-10" in text and "55,3" in text and "Rand des Simplex gewandert" in text
    assert "vorgerechnet, 20 000 Schritte" in text and "10⁵ Schritte" not in text.split("Quotientenraum")[1] and "praktisch null" not in text
    assert "je Schritt 0)" in E.verdict_nullsumme(1.0, {"vol": 0.0, "kl": 575.0})[1]
    # ε = 0,01: die Bahn liegt am Ende noch im Inneren (kleinste Wahrscheinlichkeit 0,035), das Volumen wächst im Mittel: kein „nur anfangs“, kein „Rand“
    kind, text = E.verdict_nullsumme(0.01, {"vol": 4.35e-05, "kl": 1.59, "minp": 0.0354})
    assert kind == "info" and "noch im Inneren des Simplex" in text and "0,035" in text and "nur anfangs" not in text and "gewandert" not in text
    assert E.verdict_nullsumme(0.3, None)[0] == "warning"


def test_quer_verdict_tail_depends_on_chaos_on_the_diagonal():
    q = E.QuerInfo(8.001, 12, 1e-12, 26, 1.1, 0.0, -0.001, np.array([1e-12, 1.0]))
    assert "Sattel" not in E.verdict_quer(q)[1] and "Zyklus oder ein Punkt" in E.verdict_quer(q)[1]


def test_thousands_separator_uses_a_protected_space():
    assert E.ints(40000) == "40 000" and E.ints(999) == "999"


def test_kern_verdict_without_the_frozen_file_warns():
    assert E.verdict_kern(None)[0] == "warning" and E.kernzahlen({}) is None


# ---------------------------------------------------------------------------------------------------- Presets: erwartete Verdikt-Zeile je Preset

def _settings(name):
    return E.Settings(**{**C.DEFAULTS, **C.PRESETS[name]})


def test_preset_schwelle_von_hand_reports_a_period_two_orbit_just_above_the_threshold():
    s = _settings("Schwelle von Hand (Beispiel 1)")
    i = E.schwelle_info(s.beispiel, s.s)
    assert i.regime == "zyklus" and i.period == 2
    kind, text = E.verdict_schwelle(i, None)
    assert kind == "info" and "instabil" in text and "Periode 2" in text


def test_preset_chaos_auf_der_diagonale_reports_chaos_on_the_diagonal_and_a_growing_difference():
    s = _settings("Chaos auf der Diagonale (Beispiel 2)")
    kind, text = E.verdict_schwelle(E.schwelle_info(s.beispiel, s.s), None)
    assert "chaotisch auf der Diagonale" in text
    kind, text = E.verdict_quer(E.quer_info(s.beispiel, s.s, s.dk))
    assert "wächst" in text and "Sattel" in text


def test_preset_asymmetrie_reports_all_starts_in_a_pure_equilibrium():
    s = _settings("Asymmetrie: reines Gleichgewicht")
    info = E.asym_info(s.beispiel, s.s, s.delta_s, s.delta_k, 200, s.seed, s.sym)
    kind, text = E.verdict_asym(info)
    assert kind == "success" and "200 von 200 Starts enden in einem reinen Gleichgewicht" in text


def test_preset_hedge_ohne_vergessen_reports_no_interior_chaos():
    s = _settings("3×3 mit reinem Hedge")
    info = E.live33(s.e1, s.e2, s.s33, 0.0, 40, s.seed, burn=1500, meas=1500)
    kind, text = E.verdict_hedge33(info, None)
    assert kind == "success" and "0 chaotisch im Inneren" in text


def test_preset_vergessen_reports_chaos_and_says_it_is_not_hedge():
    s = _settings("Mit Vergessen (kein Hedge mehr)")
    info = E.live33(s.e1v, s.e2v, s.sv, s.g, 40, s.seed, burn=3000, meas=3000)
    kind, text = E.verdict_vergessen(info, None)
    assert kind == "success" and "nicht mehr Hedge" in text and "40 von 40 Starts chaotisch" in text


def test_default_settings_of_the_forgetting_experiment_report_no_chaos():
    s = E.Settings()
    kind, text = E.verdict_vergessen(E.live33(s.e1v, s.e2v, s.sv, s.g, 40, s.seed, burn=3000, meas=3000), None)
    assert kind == "info" and "Außerhalb des Fensters" in text
