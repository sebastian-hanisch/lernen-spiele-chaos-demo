"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg.

(a) Hedge-Schritt des Überlastungsspiels gegen 60-stellige `decimal`-Rechnung (exp statt tanh, keine Gleitkommazahlen).
(b) Jacobi-Produkt der m×m-Abbildung gegen Complex-Step (exakte Ableitung bis auf Rundung) und gegen finite Differenz.
(c) Exponenten der Diagonale gegen das Zwei-Bahn-Verfahren und die 2D-Tangente; Querexponent gegen das gemessene Wachstum.
(d) allgemeine m×m-Fassung mit der 2×2-Überlastungsmatrix gegen die Logit-Abbildung.
(e) Schwelle gegen Bisektion auf der Spannweite (siehe auch test_algorithm.py).
(f) Volumenänderung der Nullsumme gegen die Determinante einer numerisch gebildeten Jacobi-Matrix.
Bei chaotischer Dynamik wächst jede Rundungsabweichung mit etwa e^(0,8·Schritte); die Schrittzahlen und Bänder sind daraus gewählt (Band genannt)."""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

import lsc_algorithm as A
import lsc_bimatrix as B
from lsc_rng import Rng

getcontext().prec = 60


def _dec_step(u1, u2, a1, a2, b1, b2, s1, s2):
    """Ein Schritt in 60-stelliger Dezimalrechnung; Sigmoidfunktion über exp, nicht über tanh."""
    def sig(u):
        return 1 / (1 + (-u).exp())
    x1, x2 = sig(u1), sig(u2)
    return (u1 - s1 * (a1 * (1 + x2) - a2 * (2 - x2)), u2 - s2 * (b1 * (1 + x1) - b2 * (2 - x1)))


@pytest.mark.parametrize("s", [10.001, 26.001, 40.001])
def test_hedge_step_against_sixty_digit_decimal_arithmetic(s):
    # 12 Schritte; Band 1e-9 (Querexponent bis 1,04 je Schritt: e^(1,04·12) ≈ 2,6e5, mal 1e-16 Rundung je Schritt)
    a1, a2, b1, b2 = 0.25, 0.35, 0.25, 0.4
    rng = Rng(11)
    u = rng.uniform((2, 20), -3, 3)
    P = A.Pair(a1, a2, s, b1, b2, s * 1.05)
    d = lambda v: Decimal(float(v))   # noqa: E731 (die Gleitkommazahl exakt als Dezimalzahl)
    for k in range(20):
        u1, u2 = u[0, k], u[1, k]
        D1, D2 = d(u1), d(u2)
        for _ in range(12):
            u1, u2 = A.step2(u1, u2, P)
            D1, D2 = _dec_step(D1, D2, d(a1), d(a2), d(b1), d(b2), d(s), d(s * 1.05))
        assert abs(float(D1) - u1) < 1e-9 and abs(float(D2) - u2) < 1e-9, (s, k)


def test_first_step_matches_decimal_to_rounding_error():
    u1, u2 = 0.7, -1.3
    P = A.Pair(0.25, 0.35, 30.001, 0.25, 0.35, 30.001)
    got = A.step2(u1, u2, P)
    ref = _dec_step(Decimal(u1), Decimal(u2), Decimal(0.25), Decimal(0.35), Decimal(0.25), Decimal(0.35), Decimal(30.001), Decimal(30.001))
    assert abs(float(ref[0]) - got[0]) < 1e-13 and abs(float(ref[1]) - got[1]) < 1e-13


def _complex_step_tangent(p, q, v_p, v_q, A_, C_, s, g, h=1e-30):
    """Ableitung des Schritts in Richtung (v_p, v_q) per Complex-Step; der Schritt selbst ist unverändert (komplexe Eingabe)."""
    pn, qn, _x, _y = B.step(p + 1j * h * v_p, q + 1j * h * v_q, A_, C_, s, g)
    return pn.imag / h, qn.imag / h


@pytest.mark.parametrize("g", [0.0, 0.04])
def test_tangent_matches_complex_step_and_finite_difference(g):
    rng = Rng(5)
    for _ in range(25):
        A_, C_ = rng.normal((1, 3, 3)), rng.normal((1, 3, 3))
        p, q = B.normalise(rng.normal((1, 3))), B.normalise(rng.normal((1, 3)))
        vp, vq = B.normalise(rng.normal((1, 3))), B.normalise(rng.normal((1, 3)))
        s = 1.7
        _pn, _qn, x, y = B.step(p, q, A_, C_, s, g)
        tp, tq = B.tangent(vp, vq, x, y, A_, C_, s, g)
        cp, cq = _complex_step_tangent(p, q, vp, vq, A_, C_, s, g)
        assert np.abs(cp - tp).max() < 1e-12 and np.abs(cq - tq).max() < 1e-12
        h = 1e-6
        pn2, qn2, _, _ = B.step(p + h * vp, q + h * vq, A_, C_, s, g)
        pn1, qn1, _, _ = B.step(p, q, A_, C_, s, g)
        assert np.abs((pn2 - pn1) / h - tp).max() < 1e-5   # Vorwärtsdifferenz: Fehler ≈ h·Krümmung


def test_general_bimatrix_code_equals_the_logit_map_of_the_two_by_two_game():
    a1, a2, s = 0.25, 0.35, 26.001
    Am = B.congestion_bimatrix(a1, a2)[None]
    p = np.array([[0.37, 0.0]])
    q = np.array([[0.37, 0.0]])
    u = 0.37
    worst = 0.0
    for _ in range(15):
        p, q, _x, _y = B.step(p, q, Am, Am, s, 0.0)
        u = float(A.diag_step(u, s, a1, a2))
        worst = max(worst, abs(p[0, 0] - u), abs(q[0, 0] - u))
    assert worst < 1e-11     # 15 Schritte im chaotischen Bereich; gemessen in der Vormessung 2e-14


def test_asymmetric_start_in_general_code_equals_the_two_dimensional_logit_map():
    a1, a2, s = 0.5, 0.5, 5.001
    Am = B.congestion_bimatrix(a1, a2)[None]
    p, q = np.array([[0.9, 0.0]]), np.array([[-0.4, 0.0]])
    u1, u2 = 0.9, -0.4
    P = A.Pair.symmetric(a1, a2, s)
    for _ in range(25):
        p, q, _x, _y = B.step(p, q, Am, Am, s, 0.0)
        u1, u2 = A.step2(u1, u2, P)
    assert abs(p[0, 0] - u1) < 1e-10 and abs(q[0, 0] - u2) < 1e-10


def test_diagonal_exponent_against_two_orbit_method_and_two_dimensional_tangent():
    a1, a2, s = 0.25, 0.35, 40.001
    lam_par, lam_perp, _ = A.diag_lyapunov(s, a1, a2, burn=3000, n=40000)
    # Zwei-Bahn-Verfahren in der Diagonale (eindimensional, Abstand 1e-9, Neuskalierung jeden Schritt)
    u, d0 = 0.37, 1e-9
    for _ in range(3000):
        u = float(A.diag_step(u, s, a1, a2))
    v, acc = u + d0, 0.0
    for _ in range(40000):
        u = float(A.diag_step(u, s, a1, a2))
        v = float(A.diag_step(v, s, a1, a2))
        dist = abs(v - u)
        acc += math.log(dist / d0)
        v = u + (d0 if v >= u else -d0)
    assert acc / 40000 == pytest.approx(lam_par[0], abs=0.03)      # Statistik von 4e4 Schritten: Streuung ≈ 0,01
    # 2D-Tangente auf der Diagonale misst den größeren der beiden Exponenten, hier den Querexponent
    o = A.run2(np.array([0.37]), np.array([0.37]), A.Pair.symmetric(a1, a2, s), burn=3000, meas=6000)
    assert o.lam[0] == pytest.approx(lam_perp[0], abs=0.05) and o.lam[0] > lam_par[0]


def test_cross_exponent_equals_mean_of_ln_one_plus_c_over_the_orbit():
    a1, a2, s = 0.25, 0.35, 30.001
    u = 0.37
    for _ in range(3000):
        u = float(A.diag_step(u, s, a1, a2))
    acc = 0.0
    for _ in range(20000):
        x = 1 / (1 + math.exp(-u))
        acc += math.log(1 + s * (a1 + a2) * x * (1 - x))
        u = float(A.diag_step(u, s, a1, a2))
    assert acc / 20000 == pytest.approx(A.diag_lyapunov(s, a1, a2, burn=3000, n=20000)[1][0], abs=1e-9)


def test_zero_sum_volume_equals_the_determinant_of_a_numerical_jacobian():
    R0 = B.rps(0.0)
    eps = 0.3

    def dual_step(z):
        p, q = z[:3], z[3:]
        x, y = B._sm1(p), B._sm1(q)
        return np.concatenate([p + eps * R0 @ y, q - eps * R0.T @ x])

    z = np.array([0.5, 0.2, -0.1, -0.3, 0.1, 0.4])
    h = 1e-6
    Jn = np.empty((6, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = h
        Jn[:, k] = (dual_step(z + e) - dual_step(z - e)) / (2 * h)
    ld_numeric = np.linalg.slogdet(Jn)[1]
    x, y = B._sm1(z[:3]), B._sm1(z[3:])
    Sx, Sy = np.diag(x) - np.outer(x, x), np.diag(y) - np.outer(y, y)
    Bm, Cm = eps * R0 @ Sy, -eps * R0.T @ Sx
    assert np.linalg.slogdet(np.eye(3) - Cm @ Bm)[1] == pytest.approx(ld_numeric, abs=1e-8)
    # ein Schritt der Routine mit einem einzigen Messschritt liefert dieselbe Zahl
    vol, _kl, _minp, _ = B.zero_sum_run(R0, eps, steps=2, burn=0, p0=z[:3], q0=z[3:])
    assert vol == pytest.approx(np.mean([np.linalg.slogdet(np.eye(3) - Cm @ Bm)[1], np.linalg.slogdet(np.eye(3) - _next_cm_bm(z, R0, eps))[1]]), abs=1e-9)


def _next_cm_bm(z, R0, eps):
    p, q = z[:3], z[3:]
    x, y = B._sm1(p), B._sm1(q)
    pn, qn = p + eps * R0 @ y, q - eps * R0.T @ x
    xn, yn = B._sm1(pn), B._sm1(qn)
    Sx, Sy = np.diag(xn) - np.outer(xn, xn), np.diag(yn) - np.outer(yn, yn)
    return (-eps * R0.T @ Sx) @ (eps * R0 @ Sy)


def test_softmax_and_smul_against_explicit_jacobian():
    rng = Rng(9)
    p = rng.normal((4, 3))
    d = rng.normal((4, 3))
    y = B.softmax(p)
    for k in range(4):
        J = np.diag(y[k]) - np.outer(y[k], y[k])
        assert np.allclose(B.smul(y, d)[k], J @ d[k], atol=1e-14)
