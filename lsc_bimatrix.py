"""Kern 2: Hedge (mit optionalem Vergessen) in Zwei-Spieler-Spielen beliebiger Größe m × m, im Dualraum.

Dualkoordinaten p (Spieler 1) und q (Spieler 2), Wahrscheinlichkeiten x = softmax(p), y = softmax(q):
    p ← (1 − g)·p + s·A·y,   q ← (1 − g)·q + s·C·x,
A und C sind die Auszahlungsmatrizen (Zeile = eigene Aktion, Spalte = Aktion des Gegners). g = 0 ist reines Hedge (multiplikative Gewichte);
g > 0 ist eine andere Lernregel (Vergessen) und kein Hedge mehr. Die Dualkoordinaten werden um ihren letzten Eintrag normiert (Quotientenraum: die neutrale
Richtung (1, …, 1) ist entfernt). Lyapunov-Exponent per Tangentenverfahren (Benettin) im Quotientenraum, Gegenprobe per Zwei-Bahn-Verfahren.
Alle Funktionen arbeiten auf Stapeln von B Spielen/Anfangswerten (Form (B, m, m) bzw. (B, m)). Neu geschrieben nach der Messlogik der Vormessung, nicht kopiert."""

import numpy as np

import lsc_constants as C


def softmax(p):
    e = np.exp(p - p.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def smul(y, d):
    """(diag(y) − y yᵀ)·d zeilenweise: Jacobi-Matrix der Softmax-Funktion angewandt auf d."""
    return y * d - y * np.sum(y * d, axis=1, keepdims=True)


def normalise(p):
    return p - p[:, -1:]


def step(p, q, A, C_, s, g=0.0):
    """Ein Schritt; gibt die neuen Dualkoordinaten und die alten Wahrscheinlichkeiten (x, y) zurück."""
    x, y = softmax(p), softmax(q)
    pn = (1.0 - g) * p + s * np.einsum("bij,bj->bi", A, y)
    qn = (1.0 - g) * q + s * np.einsum("bij,bj->bi", C_, x)
    return normalise(pn), normalise(qn), x, y


def tangent(vp, vq, x, y, A, C_, s, g=0.0):
    """Jacobi-Matrix des Schritts angewandt auf (vp, vq)."""
    wp = (1.0 - g) * vp + s * np.einsum("bij,bj->bi", A, smul(y, vq))
    wq = (1.0 - g) * vq + s * np.einsum("bij,bj->bi", C_, smul(x, vp))
    return normalise(wp), normalise(wq)


def run_games(A, C_, s, g, p, q, burn=4000, meas=4000, tangent_seed=0):
    """Läuft `burn` Schritte ein und misst danach `meas` Schritte.

    Rückgabe (lam, minp, spanne): größter Exponent im Quotientenraum je Bahn, kleinste Wahrscheinlichkeit entlang der Messung,
    größte Spannweite einer Wahrscheinlichkeit im letzten Fenster (400 Schritte)."""
    from lsc_rng import Rng
    B, m = p.shape
    p, q = normalise(p.copy()), normalise(q.copy())
    for _ in range(burn):
        p, q, _x, _y = step(p, q, A, C_, s, g)
    rng = Rng(tangent_seed)
    vp = normalise(rng.normal((B, m)))
    vq = normalise(rng.normal((B, m)))
    nrm = np.sqrt((vp ** 2).sum(1) + (vq ** 2).sum(1))
    vp /= nrm[:, None]
    vq /= nrm[:, None]
    acc = np.zeros(B)
    minp = np.ones(B)
    window = min(400, meas)
    lo = np.full((B, 2 * m), 1.0)
    hi = np.zeros((B, 2 * m))
    for t in range(meas):
        pn, qn, x, y = step(p, q, A, C_, s, g)
        vp, vq = tangent(vp, vq, x, y, A, C_, s, g)
        nrm = np.sqrt((vp ** 2).sum(1) + (vq ** 2).sum(1))
        vp /= nrm[:, None]
        vq /= nrm[:, None]
        acc += np.log(nrm)
        minp = np.minimum(minp, np.minimum(x.min(1), y.min(1)))
        if t >= meas - window:
            xy = np.concatenate([x, y], axis=1)
            lo = np.minimum(lo, xy)
            hi = np.maximum(hi, xy)
        p, q = pn, qn
    return acc / meas, minp, (hi - lo).max(1)


def two_orbit_lyapunov(A, C_, s, g, p, q, burn=6000, meas=6000, d0=1e-9, seed=17):
    """Gegenprobe: Zwei-Bahn-Verfahren (Abstand d0, Neuskalierung in jedem Schritt) im Quotientenraum."""
    from lsc_rng import Rng
    B, m = p.shape
    p, q = normalise(p.copy()), normalise(q.copy())
    for _ in range(burn):
        p, q, _x, _y = step(p, q, A, C_, s, g)
    rng = Rng(seed)
    dp = normalise(rng.normal((B, m)))
    dq = normalise(rng.normal((B, m)))
    nr = np.sqrt((dp ** 2).sum(1) + (dq ** 2).sum(1))
    p2, q2 = p + d0 * dp / nr[:, None], q + d0 * dq / nr[:, None]
    acc = np.zeros(B)
    for _ in range(meas):
        p, q, _x, _y = step(p, q, A, C_, s, g)
        p2, q2, _x, _y = step(p2, q2, A, C_, s, g)
        dp, dq = p2 - p, q2 - q
        dist = np.sqrt((dp ** 2).sum(1) + (dq ** 2).sum(1))
        acc += np.log(dist / d0)
        p2, q2 = p + dp * (d0 / dist)[:, None], q + dq * (d0 / dist)[:, None]
    return acc / meas


def classify(lam, minp, spanne):
    """(konvergiert, chaotisch, nicht konvergent sonst): chaotisch heißt beschränkt nicht konvergent, λ > CHAOS_LAMBDA und alle Wahrscheinlichkeiten > MIN_PROB (Inneres)."""
    conv = spanne < C.CONV_RANGE
    chaos = (~conv) & (lam > C.CHAOS_LAMBDA) & (minp > C.MIN_PROB)
    return conv, chaos, (~conv) & ~chaos


# ------------------------------------------------------------------------------------------------------------ Spiele

R_RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def rps(e: float):
    """Stein-Schere-Papier plus e·Einheitsmatrix (e = 0: Nullsumme)."""
    return R_RPS + e * np.eye(3)


def congestion_bimatrix(a1: float, a2: float):
    """Auszahlungsmatrix (= −Kosten) des Überlastungsspiels für beide Spieler: Zeile = eigene Kante, Spalte = Kante des Gegners."""
    return -np.array([[2.0 * a1, a1], [a2, 2.0 * a2]])


def random_game(rng, m: int, gamma: float):
    """Zufälliges m × m-Spiel: A_ij ~ N(0, 1); die Auszahlung des Gegners hat die Korrelation gamma zu A (C = Ct^T mit Ct = γA + √(1−γ²)Z)."""
    A = rng.normal((m, m))
    Z = rng.normal((m, m))
    Ct = gamma * A + np.sqrt(1.0 - gamma ** 2) * Z
    return A, Ct.T


# ------------------------------------------------------------------------------------------------------------ Nullsumme

def zero_sum_run(A, eps, steps=100000, burn=5000, p0=(0.5, 0.2, -0.1), q0=(-0.3, 0.1, 0.4), record_every=0):
    """Stein-Schere-Papier-artiges Nullsummenspiel (C = −Aᵀ), ein Startpunkt, Dualraum (6-dimensional, ohne Quotient).

    Rückgabe: mittleres ln|det J| je Schritt (Volumenwachstum, nach `burn` Schritten gemessen), KL-Abstand beider Strategien zur Gleichverteilung
    am Ende (Summe über beide Spieler), kleinste Wahrscheinlichkeit am Ende und optional eine Spur (record_every > 0: Schritt, KL, kleinste Wahrscheinlichkeit)."""
    p = np.array(p0, dtype=float)
    q = np.array(q0, dtype=float)
    m = p.size
    acc, n = 0.0, 0
    trace = []
    uni = np.full(m, 1.0 / m)
    for t in range(steps):
        x, y = _sm1(p), _sm1(q)
        if t >= burn:
            Sx = np.diag(x) - np.outer(x, x)
            Sy = np.diag(y) - np.outer(y, y)
            Bm = eps * A @ Sy
            Cm = -eps * A.T @ Sx
            sign, ld = np.linalg.slogdet(np.eye(m) - Cm @ Bm)   # det [[I, B], [C, I]] = det(I − C·B)
            acc += ld
            n += 1
        if record_every and t % record_every == 0:
            trace.append((t, _kl(uni, x) + _kl(uni, y), min(x.min(), y.min())))
        p = p + eps * A @ y
        q = q - eps * A.T @ x
    x, y = _sm1(p), _sm1(q)
    return (acc / max(n, 1)), _kl(uni, x) + _kl(uni, y), float(min(x.min(), y.min())), trace


def _sm1(p):
    e = np.exp(p - p.max())
    return e / e.sum()


def _kl(u, x):
    return float(np.sum(u * np.log(u / np.maximum(x, 1e-300))))
