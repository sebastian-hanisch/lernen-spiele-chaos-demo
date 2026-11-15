"""Kern 1: Hedge im Zwei-Spieler-Zwei-Kanten-Überlastungsspiel (Logit-Abbildung).

Zwei Spieler wählen je eine von zwei Kanten; die Kosten einer Kante sind a_Kante · Last, die Last ist 1 + (der andere wählt dieselbe Kante).
Hedge mit Schrittweite s (Gewichte w_k ← w_k · exp(−s · erwartete Kosten_k)) ist MWU_ε mit 1 − ε = exp(−s), also s = −ln(1 − ε).
Mit dem Logit u = ln(x/(1 − x)) der Wahrscheinlichkeit x für Kante 1 gilt für Spieler i, dessen Gegner mit x' spielt:
    u ← u − s · (a1 · (1 + x') − a2 · (2 − x')).
Auf der Diagonale u1 = u2 = u ist das eine eindimensionale Abbildung. Alle Funktionen sind überlaufsicher (tanh-Form der Sigmoidfunktion) und arbeiten
auf Feldern (Vektorisierung über Schrittweiten oder Anfangswerte). Die Messlogik ist neu geschrieben, Messreihen siehe README und tools/precompute_lsc.py."""

import math
from dataclasses import dataclass

import numpy as np

import lsc_constants as C


def sigma(u):
    """Sigmoidfunktion in der tanh-Form (kein Überlauf für große |u|)."""
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def fixed_point(a1: float, a2: float):
    """Gemischter Gleichgewichtspunkt x* = (2·a2 − a1)/(a1 + a2) der symmetrischen Diagonale; None, wenn x* nicht in (0, 1) liegt."""
    if a1 + a2 <= 0:
        return None
    xs = (2 * a2 - a1) / (a1 + a2)
    return xs if 0.0 < xs < 1.0 else None


def threshold(a1: float, a2: float):
    """Schrittweite, ab der der gemischte Punkt auf der Diagonale instabil wird: 2/((a1 + a2)·x*(1 − x*)); None ohne gemischten Punkt."""
    xs = fixed_point(a1, a2)
    return None if xs is None else 2.0 / ((a1 + a2) * xs * (1.0 - xs))


def diag_factor(u, s, a1, a2):
    """c = s·(a1 + a2)·x(1 − x): Die Jacobi-Matrix der Diagonale hat die Eigenwerte 1 − c (längs) und 1 + c (quer)."""
    x = sigma(u)
    return s * (a1 + a2) * x * (1.0 - x)


def diag_step(u, s, a1, a2):
    x = sigma(u)
    return u - s * (a1 * (1.0 + x) - a2 * (2.0 - x))


def diag_orbit(s, a1, a2, n, burn=1500, u0=0.37):
    """n Wahrscheinlichkeiten x nach `burn` Einschwingschritten; s ist ein Feld, das Ergebnis hat die Form (n, len(s))."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    u = np.full(s.shape, u0)
    for _ in range(burn):
        u = diag_step(u, s, a1, a2)
    xs = np.empty((n, s.size))
    for t in range(n):
        xs[t] = sigma(u)
        u = diag_step(u, s, a1, a2)
    return xs


def diag_lyapunov(s, a1, a2, burn=4000, n=60000, u0=0.37):
    """Längs- und Querexponent auf der Diagonale: Mittel von ln|1 − c| bzw. ln(1 + c) entlang der Bahn; dazu die Spannweite von x im Messfenster.

    Rückgabe (lam_längs, lam_quer, spanne), je ein Feld über s."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    u = np.full(s.shape, u0)
    for _ in range(burn):
        u = diag_step(u, s, a1, a2)
    acc_par = np.zeros(s.shape)
    acc_perp = np.zeros(s.shape)
    lo = np.ones(s.shape)
    hi = np.zeros(s.shape)
    for _ in range(n):
        x = sigma(u)
        c = s * (a1 + a2) * x * (1.0 - x)
        acc_par += np.log(np.abs(1.0 - c) + 1e-300)
        acc_perp += np.log1p(c)
        lo = np.minimum(lo, x)
        hi = np.maximum(hi, x)
        u = diag_step(u, s, a1, a2)
    return acc_par / n, acc_perp / n, hi - lo


def _diag_iterate(u, s, a1, a2, n):
    for _ in range(n):
        u = diag_step(u, s, a1, a2)
    return u


def _slow_cycle_period(u, s, a1, a2, max_period, window=4000):
    """Periode eines Zyklus, dem die Bahn ab u nur langsam zustrebt (Multiplikator nahe 1, also Abstand nach der Einschwingzeit noch über der Toleranz), sonst None.

    Erkennung: Die Teilfolge jeden p-ten Wertes ist im Fenster streng monoton (eine chaotische Bahn kommt darüber nicht); hinter ihr liegt ein Fixpunkt L von f^p
    (Vorzeichenwechsel von f^p(v) − v, Bisektion), der anziehend ist (|(f^p)'(L)| < 1) und dem die Teilfolge zustrebt. Zurückgegeben wird die kleinste Periode von L."""
    w = np.empty(window)
    for j in range(window):
        w[j] = u
        u = diag_step(u, s, a1, a2)
    for p in range(1, max_period + 1):
        sub = w[::p]
        d = np.diff(sub)
        if not (np.all(d > 0) or np.all(d < 0)):
            continue
        richtung = 1.0 if d[0] > 0 else -1.0
        g = lambda v, p=p: _diag_iterate(v, s, a1, a2, p) - v     # noqa: E731
        a = float(sub[0])
        ga = g(a)
        schritt = abs(float(d[0])) or 1e-9
        wurzel = None
        for _ in range(64):
            b = a + richtung * schritt
            if g(b) * ga < 0.0:
                lo, hi = a, b                                      # Bisektion zwischen a und b (g wechselt dort das Vorzeichen)
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    if g(mid) * ga < 0.0:
                        hi = mid
                    else:
                        lo = mid
                wurzel = 0.5 * (lo + hi)
                break
            schritt *= 2.0
        if wurzel is None or richtung * (wurzel - sub[-1]) <= 0.0 or abs(sub[-1] - wurzel) >= abs(sub[0] - wurzel):
            continue
        m, v = 1.0, wurzel
        for _ in range(p):
            m *= 1.0 - float(diag_factor(v, s, a1, a2))
            v = diag_step(v, s, a1, a2)
        if abs(m) >= 1.0:
            continue
        for q in range(1, p + 1):                                  # kleinste Periode des Fixpunkts von f^p
            if p % q == 0 and abs(_diag_iterate(wurzel, s, a1, a2, q) - wurzel) < 1e-8:
                return q
        return p
    return None


def diag_period(s, a1, a2, u0=0.37, burn=20000, max_period=64, tol=1e-9):
    """Kleinste Periode p ≤ max_period der Diagonalbahn (Abstand < tol nach p Schritten), sonst None.

    Strebt die Bahn nur sehr langsam einem Zyklus zu (Multiplikator nahe 1, z. B. Beispiel 1 ab s ≈ 44), reicht die Einschwingzeit für die Toleranz nicht;
    dann gilt die Periode des Zyklus, dem sie zustrebt (`_slow_cycle_period`). None bleibt für Bahnen ohne Zyklus bis max_period (chaotisch oder längere Periode)."""
    u = _diag_iterate(u0, s, a1, a2, burn)
    ref = u
    for p in range(1, max_period + 1):
        u = diag_step(u, s, a1, a2)
        if abs(u - ref) < tol:
            return p
    return _slow_cycle_period(u, s, a1, a2, max_period)


def two_cycle_example1(s: float):
    """Der 2-Zyklus {u, −u} von Beispiel 1 (a1 = a2): u = (s/4)·tanh(u/2), nichttrivial nur für s > 8. Rückgabe u > 0 oder None. Bisektion."""
    if s <= 8.0:
        return None
    g = lambda u: u - (s / 4.0) * math.tanh(u / 2.0)   # noqa: E731
    lo, hi = 1e-9, s / 4.0 + 1.0                       # g(lo) < 0 (Steigung 1 − s/8 < 0 bei 0), g(hi) > 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def first_unstable(a1, a2, s_grid, burn=3000, n=4000, tol=1e-5):
    """Erstes s der Gitterfolge, dessen Bahn nicht konvergiert (Spannweite von x im Messfenster > tol); None, wenn keines."""
    s_grid = np.asarray(s_grid, dtype=float)
    xs = diag_orbit(s_grid, a1, a2, n, burn=burn)
    span = xs.max(axis=0) - xs.min(axis=0)
    hit = np.nonzero(span > tol)[0]
    return None if hit.size == 0 else float(s_grid[hit[0]])


# ---------------------------------------------------------------------------------------------------- zwei Spieler getrennt (2D)

@dataclass(frozen=True)
class Pair:
    """Kosten und Schrittweite beider Spieler; Spieler 1: (a1, a2, s1), Spieler 2: (b1, b2, s2)."""
    a1: float
    a2: float
    s1: float
    b1: float
    b2: float
    s2: float

    @staticmethod
    def symmetric(a1, a2, s):
        return Pair(a1, a2, s, a1, a2, s)

    @staticmethod
    def with_deviation(a1, a2, s, delta_s=0.0, delta_k=0.0):
        """Spieler 2 lernt mit s·(1 + delta_s) und hat auf Kante 2 die Kosten a2·(1 + delta_k)."""
        return Pair(a1, a2, s, a1, a2 * (1.0 + delta_k), s * (1.0 + delta_s))


def step2(u1, u2, P: Pair):
    x1, x2 = sigma(u1), sigma(u2)
    return (u1 - P.s1 * (P.a1 * (1.0 + x2) - P.a2 * (2.0 - x2)),
            u2 - P.s2 * (P.b1 * (1.0 + x1) - P.b2 * (2.0 - x1)))


@dataclass(frozen=True)
class Outcomes:
    pure: np.ndarray       # reines Gleichgewicht (x1 > 0,999 und x2 < 0,001 oder umgekehrt)
    still: np.ndarray      # steht still, nicht rein (gemischtes Gleichgewicht)
    other: np.ndarray      # sonst (Zyklus oder unregelmäßig)
    chaos: np.ndarray      # sonst und λ > CHAOS_LAMBDA
    lam: np.ndarray        # größter Exponent (Tangentenverfahren im Messfenster)
    on_diagonal: np.ndarray  # |u1 − u2| < 1e-3 am Ende


def run2(u1, u2, P: Pair, burn=4000, meas=1500) -> Outcomes:
    """Lässt viele Anfangswerte (Felder u1, u2) laufen und klassifiziert den Ausgang; λ per Tangentenverfahren im Messfenster."""
    u1 = np.array(u1, dtype=float)
    u2 = np.array(u2, dtype=float)
    for _ in range(burn):
        u1, u2 = step2(u1, u2, P)
    v = np.stack([np.ones(u1.shape), 0.7 * np.ones(u1.shape)], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    acc = np.zeros(u1.shape)
    lo = np.ones(u1.shape)
    hi = np.zeros(u1.shape)
    for _ in range(meas):
        x1, x2 = sigma(u1), sigma(u2)
        j12 = -P.s1 * (P.a1 + P.a2) * x2 * (1.0 - x2)
        j21 = -P.s2 * (P.b1 + P.b2) * x1 * (1.0 - x1)
        v = np.stack([v[:, 0] + j12 * v[:, 1], j21 * v[:, 0] + v[:, 1]], axis=1)
        nv = np.linalg.norm(v, axis=1)
        v /= nv[:, None]
        acc += np.log(nv)
        u1, u2 = step2(u1, u2, P)
        lo = np.minimum(lo, np.minimum(x1, x2))
        hi = np.maximum(hi, np.maximum(x1, x2))
    x1, x2 = sigma(u1), sigma(u2)
    pure = ((x1 > 0.999) & (x2 < 0.001)) | ((x1 < 0.001) & (x2 > 0.999))
    still = ((hi - lo) < 1e-6) & ~pure
    other = ~pure & ~still
    lam = acc / meas
    return Outcomes(pure, still, other, other & (lam > C.CHAOS_LAMBDA), lam, np.abs(u1 - u2) < 1e-3)


def steps_to_leave(a1, a2, s, delta, start=0.37, limit=100000, burn=4000):
    """Schritte, bis ein Anfangsunterschied delta (in u) auf der Diagonale auf |u1 − u2| > 1 gewachsen ist (Start bei u = 0,37 nach `burn` Einschwingschritten auf der Diagonalbahn; burn = 0 beginnt mitten im Übergang).

    Gibt (Schritte, Abstände je Schritt) zurück; bei Erreichen von `limit` wird abgebrochen."""
    P = Pair.symmetric(a1, a2, s)
    u = start
    for _ in range(burn):
        u = diag_step(u, s, a1, a2)
    u1, u2 = u + delta, u
    dist = [abs(u1 - u2)]
    n = 0
    while dist[-1] < 1.0 and n < limit:
        u1, u2 = step2(u1, u2, P)
        dist.append(abs(float(u1) - float(u2)))
        n += 1
    return n, np.array(dist)


def growth_rate(dist, max_dist=0.05):
    """Wachstumsrate je Schritt (Steigung von ln Abstand über die Schritte, solange der Abstand unter max_dist bleibt)."""
    dist = np.asarray(dist, dtype=float)
    keep = np.nonzero(dist <= max_dist)[0]
    k = keep[-1] if keep.size else 0
    if k < 1:
        return float("nan")
    return float((math.log(dist[k]) - math.log(dist[0])) / k)
