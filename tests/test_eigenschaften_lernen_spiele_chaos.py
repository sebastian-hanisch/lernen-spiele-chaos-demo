"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die für Hedge gelten müssen, ohne dass das richtige Ergebnis bekannt ist.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64) und die Randfälle. Chaotische Bahnen verstärken Rundungsabweichungen
mit etwa e^(0,8 · Schritte); Vergleiche über viele Schritte laufen deshalb bei Schrittweiten ohne Chaos (s = 7,99 und 10,001 im Beispiel 1) oder über wenige Schritte."""

import numpy as np
import pytest

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
import lsc_evaluation as E
from lsc_rng import Rng


def _orbit(u0, s, a1, a2, n):
    u = np.array(u0, dtype=float)
    out = []
    for _ in range(n):
        u = A.diag_step(u, s, a1, a2)
        out.append(u.copy())
    return np.array(out)


def test_swapping_the_two_edges_mirrors_the_logit():
    rng = Rng(1)
    a = rng.uniform((30, 2), 0.05, 1.0)
    u0 = rng.uniform((30,), -3, 3)
    for k in range(30):
        a1, a2 = a[k]
        s = 10.0 + 10 * k / 30
        o1 = _orbit(u0[k], s, a1, a2, 12)
        o2 = _orbit(-u0[k], s, a2, a1, 12)       # Kanten vertauscht: x ↔ 1 − x, also u ↔ −u
        assert np.allclose(o1, -o2, atol=1e-6), k   # 12 Schritte, Band 1e-6 (Rundung wächst höchstens mit e^(1,1·12))


def test_swapping_the_players_swaps_the_outcome():
    rng = Rng(2)
    u = rng.uniform((2, 40), -3, 3)
    P = A.Pair(0.25, 0.35, 12.001, 0.3, 0.45, 15.001)
    Q = A.Pair(P.b1, P.b2, P.s2, P.a1, P.a2, P.s1)
    o = A.run2(u[0], u[1], P, burn=2000, meas=500)
    p = A.run2(u[1], u[0], Q, burn=2000, meas=500)
    assert np.array_equal(o.pure, p.pure) and np.array_equal(o.still, p.still) and np.array_equal(o.other, p.other)


def test_only_the_product_of_cost_and_step_matters():
    # Kosten mit c > 0 multipliziert und s durch c geteilt: dieselbe Bahn (c = 0,9 ist keine Zweierpotenz); 8 Schritte, Band 1e-9
    rng = Rng(3)
    u0 = rng.uniform((25,), -3, 3)
    for c in (0.9, 1.7, 3.3):
        o1 = _orbit(u0, 30.001, 0.25, 0.35, 8)
        o2 = _orbit(u0, 30.001 / c, 0.25 * c, 0.35 * c, 8)
        assert np.allclose(o1, o2, atol=1e-9), c


def test_adding_a_constant_to_all_payoffs_of_a_player_does_not_change_the_probabilities():
    rng = Rng(4)
    for g in (0.0, 0.05):
        A_, C_ = rng.normal((20, 3, 3)), rng.normal((20, 3, 3))
        p, q = B.normalise(rng.normal((20, 3))), B.normalise(rng.normal((20, 3)))
        p2, q2 = p.copy(), q.copy()
        shift_a, shift_c = rng.normal((20, 1, 1)), rng.normal((20, 1, 1))
        for _ in range(30):
            p, q, x, y = B.step(p, q, A_, C_, 0.7, g)
            p2, q2, x2, y2 = B.step(p2, q2, A_ + shift_a, C_ + shift_c, 0.7, g)
        assert np.allclose(x, x2, atol=1e-9) and np.allclose(y, y2, atol=1e-9), g   # 30 Schritte, Band 1e-9


def test_probabilities_stay_in_zero_one_and_sum_to_one_even_at_the_largest_step():
    rng = Rng(5)
    A_, C_ = rng.normal((50, 3, 3)), rng.normal((50, 3, 3))
    p, q = B.normalise(rng.normal((50, 3))), B.normalise(rng.normal((50, 3)))
    for _ in range(200):
        p, q, x, y = B.step(p, q, A_, C_, 80.0, 0.0)
        assert np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
        assert np.allclose(x.sum(axis=1), 1) and np.allclose(y.sum(axis=1), 1)
    u = np.linspace(-1e4, 1e4, 9)
    assert ((A.sigma(u) >= 0) & (A.sigma(u) <= 1)).all() and np.isfinite(A.diag_step(u, 80.0, 0.25, 0.35)).all()


def test_equal_probabilities_in_a_symmetric_zero_sum_game_stay_at_the_uniform_equilibrium():
    A_ = np.array([B.rps(0.0)])
    p = np.zeros((1, 3))
    q = np.zeros((1, 3))
    for _ in range(50):
        p, q, x, y = B.step(p, q, A_, A_, 3.001, 0.0)
    assert np.allclose(x, 1 / 3) and np.allclose(y, 1 / 3)


def test_same_seed_gives_the_same_output_and_another_seed_a_different_one():
    kw = dict(beispiel=2, s=26.001, delta_s=0.001, delta_k=0.0, n=60, symmetric=False)
    a = E.asym_info(seed=5, **kw)
    assert a == E.asym_info(seed=5, **kw)
    assert E.make_starts(5, 10)[0].tolist() != E.make_starts(6, 10)[0].tolist()
    l1 = E.live33(0.0, 0.4, 0.501, 0.02, 20, 5, burn=500, meas=500)
    assert l1 == E.live33(0.0, 0.4, 0.501, 0.02, 20, 5, burn=500, meas=500)


def test_the_diagonal_stays_invariant_under_swapping_players_in_a_symmetric_game():
    rng = Rng(6)
    u = rng.uniform((2, 20), -3, 3)
    u[1] = u[0]
    P = A.Pair.symmetric(0.25, 0.35, 26.001)
    u1, u2 = u[0].copy(), u[1].copy()
    for _ in range(500):
        u1, u2 = A.step2(u1, u2, P)
    assert np.array_equal(u1, u2)


def test_larger_forgetting_keeps_the_dual_coordinates_bounded():
    # |p| ≤ s·max|A|/(g) nach genügend Schritten (Vergessen hält die Dualkoordinaten beschränkt); ohne Vergessen wachsen sie
    rng = Rng(7)
    A_ = np.array([B.rps(0.0)] * 5)
    p, q = B.normalise(rng.normal((5, 3))), B.normalise(rng.normal((5, 3)))
    for _ in range(3000):
        p, q, _x, _y = B.step(p, q, A_, A_, 0.501, 0.05)
    assert np.abs(p).max() <= 2 * 0.501 * 1.0 / 0.05 + 1e-9
