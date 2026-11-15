"""Kern gegen Handrechnung (Handbeispiel aus dem Plan: Beispiel 1 a₁ = a₂ = ½, Beispiel 2 a₁ = ¼, a₂ = 7/20), je Einheit eine Mini-Instanz.

Schrittweiten sind nie Zweierpotenzen und nie genau eine Schwelle (8,001 und 7,99 statt 8; 3,999 statt 4): bei s = 8 ist der Mehrfachwert exakt −1, ein Rechner-Artefakt.
Die Handwerte selbst (8; 17,78; u = 1,776029 bei s = 10) sind Zahlen des Handbeispiels, keine Reglerwerte."""

import math

import numpy as np
import pytest

import lsc_algorithm as A
import lsc_constants as C
from lsc_rng import Rng


def test_fixed_point_and_threshold_by_hand():
    # x* = (2·a2 − a1)/(a1 + a2): Beispiel 1: (1 − ½)/1 = ½; Beispiel 2: (0,7 − 0,25)/0,6 = 0,75
    assert A.fixed_point(0.5, 0.5) == pytest.approx(0.5)
    assert A.fixed_point(0.25, 0.35) == pytest.approx(0.75)
    # Schwelle 2/((a1 + a2)·x*(1 − x*)): 2/(1·¼) = 8 und 2/(0,6·0,1875) = 160/9
    assert A.threshold(0.5, 0.5) == pytest.approx(8.0)
    assert A.threshold(0.25, 0.35) == pytest.approx(160 / 9)


def test_factor_along_and_across_the_diagonal_at_the_fixed_point():
    u_star = 0.0                                                   # Beispiel 1: x* = ½
    for s in (3.999, 8.001, 12.001):
        c = A.diag_factor(u_star, s, 0.5, 0.5)
        assert c == pytest.approx(s / 4)                           # c = s·(a1 + a2)·¼
        assert (1 - c, 1 + c) == pytest.approx((1 - s / 4, 1 + s / 4))
    assert abs(1 - A.diag_factor(0.0, 7.99, 0.5, 0.5)) < 1 < abs(1 - A.diag_factor(0.0, 8.001, 0.5, 0.5))


def test_one_step_by_hand():
    # Beispiel 1: u' = u − s·(σ(u) − ½); s = 3,999 und u = 1: σ(1) = 1/(1 + e⁻¹) = 0,7310586
    sg = 1 / (1 + math.exp(-1))
    assert A.diag_step(1.0, 3.999, 0.5, 0.5) == pytest.approx(1 - 3.999 * (sg - 0.5), abs=1e-14)
    assert A.diag_step(1.0, 3.999, 0.5, 0.5) == pytest.approx(0.0760, abs=1e-4)


def test_two_cycle_of_example_1_at_s_10():
    u = A.two_cycle_example1(10.0)
    assert u == pytest.approx(1.776029, abs=1e-6)                  # u = (s/4)·tanh(u/2) für s = 10
    assert float(A.sigma(-u)) == pytest.approx(0.144794, abs=1e-6)
    assert float(A.sigma(u)) == pytest.approx(0.855206, abs=1e-6)
    assert math.tanh(u / 2) == pytest.approx(0.710412, abs=1e-6)   # Spannweite der Wahrscheinlichkeit
    assert A.diag_step(u, 10.0, 0.5, 0.5) == pytest.approx(-u, abs=1e-12) and A.diag_step(-u, 10.0, 0.5, 0.5) == pytest.approx(u, abs=1e-12)
    c = A.diag_factor(u, 10.0, 0.5, 0.5)
    assert (1 - c) ** 2 == pytest.approx(0.05678, abs=1e-5)        # stabil längs der Diagonale
    assert (1 + c) ** 2 == pytest.approx(5.00993, abs=1e-4)        # quer instabil
    assert A.two_cycle_example1(7.99) is None and A.two_cycle_example1(8.0) is None


def test_two_dimensional_map_leaves_the_diagonal_after_about_26_steps_from_the_cycle():
    u = A.two_cycle_example1(10.0)
    P = A.Pair.symmetric(0.5, 0.5, 10.0)
    u1, u2 = u + 1e-9, u
    n = 0
    for n in range(1, 200):
        u1, u2 = A.step2(u1, u2, P)
        if abs(u1 - u2) > 1.0:
            break
    expected = math.log(1e9) / (0.5 * math.log(5.00993))           # ≈ 25,7
    assert abs(n - expected) <= 3


def test_hedge_as_mwu_epsilon():
    s = 10.0
    eps = 1 - math.exp(-s)
    assert eps == pytest.approx(0.9999546, abs=1e-7)
    assert -math.log(1 - eps) == pytest.approx(s)


def test_stable_below_and_period_two_above_the_threshold():
    below = A.diag_orbit(7.99, 0.5, 0.5, 50, burn=20000)[:, 0]   # Faktor −0,9975: langsame Annäherung, daher viele Einschwingschritte
    assert below.max() - below.min() < 1e-9 and below[0] == pytest.approx(0.5, abs=1e-6)
    above = A.diag_orbit(8.001, 0.5, 0.5, 50, burn=3000)[:, 0]
    assert above.max() - above.min() > 1e-5
    assert A.diag_period(10.001, 0.5, 0.5) == 2
    assert A.diag_period(7.99, 0.5, 0.5) == 1


@pytest.mark.parametrize("s", [44.00137, 46.00137, 50.00137, 55.00137, 59.50137])
def test_period_detection_finds_the_slowly_converging_two_cycle_of_example_1(s):
    """Fehler vor der Korrektur: Beispiel 1 (a1 = a2 = ½) hat für jedes s > 8 den stabilen 2-Zyklus {u, −u} mit u = (s/4)·tanh(u/2) und dem Multiplikator
    (1 − (s/4)·sech²(u/2))² < 1, der sich mit wachsendem s der 1 nähert (0,9985 bei s = 44). Bei s ≥ 44 reichten 20 000 Einschwingschritte für die Toleranz 1e-9 nicht,
    diag_period meldete None („Periode > 64“, in der App „lange Periode oder quasiperiodisch“)."""
    u = A.two_cycle_example1(s)
    assert A.diag_step(u, s, 0.5, 0.5) == pytest.approx(-u, abs=1e-9) and A.diag_step(-u, s, 0.5, 0.5) == pytest.approx(u, abs=1e-9)     # der Zyklus ist ein 2-Zyklus
    multiplikator = (1.0 - (s / 4.0) / math.cosh(u / 2.0) ** 2) ** 2
    assert 0.99 < multiplikator < 1.0                                                                                                   # stabil, aber langsam
    assert A.diag_period(s, 0.5, 0.5) == 2
    assert A.diag_period(s, 0.5, 0.5, burn=1000) == 2                                                                                   # unabhängig von der Einschwingzeit


def test_period_detection_for_every_example_1_grid_value_above_the_threshold():
    """README: Periode 2 für alle 104 gemessenen Werte von s = 8,0 bis 59,5 (Raster 0,5, je + 0,00137)."""
    ss = np.arange(8.0, 60.0, 0.5) + C.S_SHIFT
    assert ss.size == 104 and [A.diag_period(float(s), 0.5, 0.5) for s in ss] == [2] * 104


def test_slow_cycle_detection_does_not_turn_a_chaotic_orbit_into_a_period():
    assert A.diag_period(26.00137, 0.25, 0.35) is None and A.diag_period(40.00137, 0.25, 0.35) is None


def test_threshold_by_scan_of_the_spread_brackets_the_closed_form():
    # Raster mit Schritt 0,05 wie in der Messreihe; ein feineres Raster zeigt unterhalb der Schwelle die langsame Annäherung (Faktor nahe −1) als „nicht konvergent“
    grid = np.arange(7.5, 8.5, 0.05) + C.S_SHIFT
    first = A.first_unstable(0.5, 0.5, grid)
    assert 8.0 < first <= 8.06
    grid2 = np.arange(17.0, 18.5, 0.05) + C.S_SHIFT
    first2 = A.first_unstable(0.25, 0.35, grid2)
    assert 160 / 9 < first2 <= 160 / 9 + 0.07


def test_exponents_along_and_across_by_hand_values_at_s_20():
    # Beispiel 2, s = 20,001: stabiler 2-Zyklus (λ_längs < 0), Querexponent > 0 (Faktor 1 + c ≥ 2,05 in jedem Schritt)
    lp, lq, span = A.diag_lyapunov(20.001, 0.25, 0.35, burn=2000, n=4000)
    assert lp[0] < -1.0 and 1.0 < lq[0] < 1.1 and span[0] > 0.2
    # Mittelwert von ln(1 + c) liegt bei jedem s über 0: c > 0
    ss = np.array([5.001, 20.001, 40.001, 70.001])
    assert (A.diag_lyapunov(ss, 0.25, 0.35, burn=500, n=2000)[1] > 0).all()


def test_growth_of_a_start_difference_matches_the_cross_exponent():
    steps, dist = A.steps_to_leave(0.25, 0.35, 40.001, 1e-12)           # nach 4000 Einschwingschritten auf der Bahn
    lam_perp = A.diag_lyapunov(40.001, 0.25, 0.35, burn=3000, n=20000)[1][0]
    assert 25 <= steps <= 40
    assert A.growth_rate(dist) == pytest.approx(lam_perp, abs=0.1)
    steps0, _ = A.steps_to_leave(0.25, 0.35, 40.001, 1e-12, burn=0)     # ohne Einschwingen: Übergang, andere Zahl
    assert 25 <= steps0 <= 40


def test_symmetric_start_stays_exactly_on_the_diagonal_in_floating_point():
    u = np.linspace(-2, 2, 7)
    P = A.Pair.symmetric(0.25, 0.35, 40.001)
    u1, u2 = u.copy(), u.copy()
    for _ in range(3000):
        u1, u2 = A.step2(u1, u2, P)
    assert np.array_equal(u1, u2)                                  # bitgleich: dieselben Gleitkommaoperationen
    v1, v2 = u.copy() + 1e-12, u.copy()
    for _ in range(60):
        v1, v2 = A.step2(v1, v2, P)
    assert np.abs(v1 - v2).max() > 1e-3                            # ein Unterschied von 1e-12 verlässt die Diagonale


def test_classification_pure_equilibrium_from_asymmetric_starts():
    u = Rng(0).uniform((2, 100), -3, 3)
    o = A.run2(u[0], u[1], A.Pair.symmetric(0.25, 0.35, 40.001))
    assert o.pure.all() and not o.chaos.any() and not o.on_diagonal.any()
