"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): s → 0, sehr großes s, Wahrscheinlichkeit exakt 0 oder 1, entartete Kosten, kein gemischter Punkt, Nullsumme, Anfangsunterschied null."""

import numpy as np
import pytest

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
import lsc_evaluation as E


def test_step_size_tending_to_zero_leaves_the_state_unchanged():
    assert A.diag_step(0.8, 1e-12, 0.25, 0.35) == pytest.approx(0.8, abs=1e-11)
    A_ = np.array([B.rps(0.1)])
    p = np.array([[0.4, -0.2, 0.0]])
    pn, _qn, _x, _y = B.step(p, p, A_, A_, 1e-12, 0.0)
    assert np.allclose(pn, p, atol=1e-11)


def test_very_large_step_stays_finite():
    u = np.array([0.0, 1.0, -5.0])
    for _ in range(100):
        u = A.diag_step(u, 1e6, 0.25, 0.35)
    assert np.isfinite(u).all()


def test_probability_exactly_zero_or_one_means_infinite_logit_without_error():
    assert float(A.sigma(np.inf)) == 1.0 and float(A.sigma(-np.inf)) == 0.0
    assert A.diag_step(np.inf, 10.001, 0.25, 0.35) == np.inf


def test_zero_costs_have_no_mixed_point_and_no_dynamics():
    assert A.fixed_point(0.0, 0.0) is None and A.threshold(0.0, 0.0) is None
    assert A.diag_step(0.7, 40.001, 0.0, 0.0) == 0.7


def test_costs_without_mixed_point_report_none_and_the_verdict_says_so():
    # a1 = 1, a2 = 0,3: x* = (0,6 − 1)/1,3 < 0, Kante 1 ist immer teurer: kein gemischter Punkt
    assert A.fixed_point(1.0, 0.3) is None and A.threshold(1.0, 0.3) is None
    info = E.SchwelleInfo(1, 10.0, None, None, None, 0.0, 0.0, 1, 0.0, "stabil")
    kind, text = E.verdict_schwelle(info, None)
    assert kind == "warning" and "keinen gemischten Gleichgewichtspunkt" in text


def test_start_difference_exactly_zero_is_bitwise_symmetric_documented_property_of_the_implementation():
    u = np.array([0.37, -1.2])
    P = A.Pair.symmetric(0.25, 0.35, 40.001)
    u1, u2 = u.copy(), u.copy()
    for _ in range(500):
        u1, u2 = A.step2(u1, u2, P)
    assert np.array_equal(u1, u2)
    assert A.steps_to_leave(0.25, 0.35, 40.001, 0.0, limit=300)[0] == 300   # Unterschied 0: nie verlassen


def test_zero_sum_game_has_no_pure_or_mixed_attractor_in_the_interior():
    lam, minp, rg = B.run_games(np.array([B.rps(0.0)]), np.array([B.rps(0.0)]), 0.51, 0.0, np.array([[0.5, 0.2, -0.1]]), np.array([[-0.3, 0.1, 0.4]]), burn=2000, meas=2000)
    conv, chaos, _oth = B.classify(lam, minp, rg)
    assert not conv[0] and not chaos[0]                                 # er kreist/driftet, ist aber nicht chaotisch im Inneren gezählt


def test_classification_boundaries_are_inclusive_at_the_stated_limits():
    lam = np.array([C.CHAOS_LAMBDA, C.CHAOS_LAMBDA + 1e-6, 0.1])
    minp = np.array([0.1, 0.1, C.MIN_PROB])
    rg = np.array([0.5, 0.5, 0.5])
    conv, chaos, oth = B.classify(lam, minp, rg)
    assert chaos.tolist() == [False, True, False]                       # λ muss größer, minp muss größer als die Grenze sein
    assert oth.tolist() == [True, False, True] and not conv.any()
    assert B.classify(np.array([1.0]), np.array([0.5]), np.array([C.CONV_RANGE]))[0][0] == False  # noqa: E712 (Spannweite genau CONV_RANGE: nicht konvergiert)


def test_ties_in_the_start_list_do_not_matter_and_one_start_works():
    u1, u2 = np.array([0.5]), np.array([-0.5])
    o = A.run2(u1, u2, A.Pair.symmetric(0.25, 0.35, 10.001), burn=500, meas=200)
    assert o.pure.shape == (1,) and o.pure[0]


def test_bifurcation_scan_with_a_single_value_and_no_value_above_the_threshold():
    scan = E.bifurcation_scan(0.5, 0.5, np.array([5.001]), n=500, burn=500)
    assert scan["above"] == 0 and scan["chaotic"] == 0 and scan["first_chaotic"] is None and scan["period"] == [1]


def test_periodic_orbit_detection_returns_none_for_a_chaotic_orbit():
    assert A.diag_period(40.001, 0.25, 0.35) is None


def test_random_game_with_gamma_plus_or_minus_one_is_a_degenerate_but_valid_game():
    from lsc_rng import Rng
    a, c = B.random_game(Rng(1), 3, 1.0)
    assert np.allclose(c.T, a)                                          # γ = 1: C = Aᵀ (Koordinationsspiel)
    a, c = B.random_game(Rng(1), 3, -1.0)
    assert np.allclose(c.T, -a)                                         # γ = −1: Nullsumme
