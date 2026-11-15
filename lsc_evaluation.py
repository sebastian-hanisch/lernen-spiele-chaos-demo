"""Messungen, Kennzahlen und Verdikte. Die Messfunktionen sind parametrisiert (Zahl der Starts, Messdauer): die App rechnet kleine Fassungen live,
tools/precompute_lsc.py die großen (vorgerechnet, data/lsc_vorgerechnet.json), tests/ die kleinen mit denselben Funktionen.
Verdikte sind reine Funktionen (Art, Text); die App ruft st.<Art>(Text)."""

from dataclasses import dataclass

import numpy as np

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
from lsc_rng import Rng


@dataclass(frozen=True)
class Settings:
    beispiel: int = C.DEFAULTS["beispiel"]
    s: float = C.DEFAULTS["s"]
    dk: int = C.DEFAULTS["dk"]
    delta_s: float = C.DEFAULTS["delta_s"]
    delta_k: float = C.DEFAULTS["delta_k"]
    k_starts: int = C.DEFAULTS["k_starts"]
    sym: bool = C.DEFAULTS["sym"]
    eps: float = C.DEFAULTS["eps"]
    e1: float = C.DEFAULTS["e1"]
    e2: float = C.DEFAULTS["e2"]
    s33: float = C.DEFAULTS["s33"]
    g: float = C.DEFAULTS["g"]
    sv: float = C.DEFAULTS["sv"]
    e1v: float = C.DEFAULTS["e1v"]
    e2v: float = C.DEFAULTS["e2v"]
    seed: int = C.DEFAULTS["seed"]

    @property
    def costs(self):
        _name, a1, a2 = C.EXAMPLES[self.beispiel]
        return a1, a2


def pct(x: float) -> str:
    """Prozentzahl mit Dezimalkomma, eine Stelle."""
    return f"{x:.1f}".replace(".", ",")


def ints(n: int) -> str:
    """Ganzzahl mit geschütztem Leerzeichen als Tausendertrenner (40 000)."""
    return f"{int(n):,}".replace(",", " ")


def num(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------------------------------------------ Experiment 1: Schwelle

@dataclass(frozen=True)
class SchwelleInfo:
    beispiel: int
    s: float
    xs: float | None
    threshold: float | None
    c: float | None            # c = s·(a1 + a2)·x*(1 − x*) am gemischten Punkt
    lam_par: float             # Exponent längs der Diagonalbahn
    lam_perp: float
    period: int | None
    span: float
    regime: str                # "stabil", "zyklus", "chaos", "lang"


def schwelle_info(beispiel: int, s: float, n=20000) -> SchwelleInfo:
    _name, a1, a2 = C.EXAMPLES[beispiel]
    xs = A.fixed_point(a1, a2)
    thr = A.threshold(a1, a2)
    c = None if xs is None else s * (a1 + a2) * xs * (1.0 - xs)
    lp, lq, sp = A.diag_lyapunov(s, a1, a2, burn=3000, n=n)
    period = A.diag_period(s, a1, a2)
    lam_par, lam_perp, span = float(lp[0]), float(lq[0]), float(sp[0])
    if thr is None or s <= thr:
        regime = "stabil" if span < 1e-5 else ("chaos" if lam_par > C.CHAOS_LAMBDA else "zyklus")
    elif lam_par > C.CHAOS_LAMBDA:
        regime = "chaos"
    elif period is not None:
        regime = "zyklus"
    else:
        regime = "lang"
    if span < 1e-5:
        regime = "stabil"
        period = 1
    return SchwelleInfo(beispiel, s, xs, thr, c, lam_par, lam_perp, period, span, regime)


def bifurcation_scan(a1, a2, ss, n=60000, burn=4000):
    """Diagonalabbildung über ein Gitter von Schrittweiten: Exponenten, Spannweite, Periode, Zählung oberhalb der Schwelle. Reine Rechnung (vorgerechnet)."""
    ss = np.asarray(ss, dtype=float)
    lp, lq, sp = A.diag_lyapunov(ss, a1, a2, burn=burn, n=n)
    periods = [A.diag_period(float(s), a1, a2) for s in ss]
    thr = A.threshold(a1, a2)
    above = ss > thr if thr is not None else np.zeros(ss.shape, dtype=bool)
    chaotic = above & (lp > C.CHAOS_LAMBDA)
    first = float(ss[chaotic][0]) if chaotic.any() else None
    return {"s": [float(v) for v in ss], "lam_par": [float(v) for v in lp], "lam_perp": [float(v) for v in lq], "span": [float(v) for v in sp],
            "period": periods, "above": int(above.sum()), "chaotic": int(chaotic.sum()), "first_chaotic": first, "threshold": thr}


def bifurcation_picture(a1, a2, s_lo=1.0, s_hi=60.0, points=420, burn=1500, keep=120):
    """Punkte (s, x) für das Bifurkationsbild der Diagonale (live) und λ(s) auf demselben Gitter (kurze Messung)."""
    ss = np.linspace(s_lo, s_hi, points) + C.S_SHIFT
    xs = A.diag_orbit(ss, a1, a2, keep, burn=burn)
    lp, lq, _sp = A.diag_lyapunov(ss, a1, a2, burn=1500, n=1500)
    return ss, xs, lp, lq


def verdict_schwelle(i: SchwelleInfo, scan: dict | None) -> tuple[str, str]:
    """Verdikt des Schwellen-Experiments. `scan`: vorgerechnete Zählung (oder None)."""
    if i.threshold is None:
        return "warning", "Bei diesen Kosten gibt es keinen gemischten Gleichgewichtspunkt (x* liegt nicht in (0, 1)); eine Schwelle gibt es nicht."
    thr = num(i.threshold, 2)
    if i.regime == "stabil":
        return "success", (f"Bei s = {num(i.s, 3)} liegt die Schrittweite unter der Schwelle {thr}: Der gemischte Punkt x* = {num(i.xs, 2)} ist stabil (Faktor 1 − c = {num(1 - i.c, 3)}), "
                           "die Bahn konvergiert zu ihm, solange beide Spieler gleich starten.")
    head = f"Bei s = {num(i.s, 3)} ist der gemischte Punkt instabil (Schwelle {thr}, Faktor 1 − c = {num(1 - i.c, 3)}). "
    if i.regime == "zyklus":
        tail = f"Die Bahn auf der Diagonale hat die Periode {i.period} (λ längs = {num(i.lam_par, 3)})."
    elif i.regime == "chaos":
        tail = f"Der Exponent längs der Diagonale ist positiv, λ = {num(i.lam_par, 3)} (Messdauer 20 000 Schritte): chaotisch auf der Diagonale."
    else:
        tail = f"Keine Periode bis 64 erkannt und kein positiver Exponent (λ = {num(i.lam_par, 3)}): eine längere Periode oder eine sehr langsame Annäherung an einen Zyklus; nicht chaotisch gemessen."
    if scan is not None and scan.get("first_chaotic") is not None:
        tail += f" In der vorgerechneten Reihe war das erste chaotische s = {num(scan['first_chaotic'], 2)} ({scan['chaotic']} von {scan['above']} Werten oberhalb der Schwelle)."
    elif scan is not None:
        tail += f" In der vorgerechneten Reihe war keiner von {scan['above']} Werten oberhalb der Schwelle chaotisch."
    return "info", head + tail


# ------------------------------------------------------------------------------------------------------ Experiment 2: quer

def sup_exp(k: int) -> str:
    """10^-k in Hochzahl-Schreibweise, z. B. 10⁻¹²."""
    table = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "10⁻" + str(k).translate(table)


@dataclass(frozen=True)
class QuerInfo:
    s: float
    dk: int
    delta: float
    steps: int
    rate: float
    lam_perp: float
    lam_par: float
    dist: np.ndarray


def quer_info(beispiel: int, s: float, dk: int, limit=400) -> QuerInfo:
    _name, a1, a2 = C.EXAMPLES[beispiel]
    delta = 10.0 ** (-dk)
    steps, dist = A.steps_to_leave(a1, a2, s, delta, limit=limit)
    lp, lq, _sp = A.diag_lyapunov(s, a1, a2, burn=3000, n=20000)
    return QuerInfo(s, dk, delta, steps, A.growth_rate(dist), float(lq[0]), float(lp[0]), dist)


def verdict_quer(i: QuerInfo) -> tuple[str, str]:
    if i.steps >= 400:
        return "warning", f"Ein Anfangsunterschied von {sup_exp(i.dk)} wuchs in 400 Schritten nicht auf 1: Bei diesem s ist die Diagonale quer nicht instabil genug, oder die Bahn liegt in einem Gleichgewicht."
    tail = ("Die Diagonale ist quer instabil: Das Chaos dort ist ein Sattel, kein Attraktor." if i.lam_par > C.CHAOS_LAMBDA
            else "Die Diagonale ist quer instabil: Auch ein Zyklus oder ein Punkt auf ihr zieht Bahnen mit einem kleinen Unterschied nicht an, sondern von ihr weg.")
    return "info", (f"Der Unterschied von {sup_exp(i.dk)} wächst mit {num(i.rate, 2)} je Schritt und ist nach {i.steps} Schritten auf 1 gewachsen; der Querexponent entlang der Diagonale ist {num(i.lam_perp, 2)}, "
                    f"der Exponent längs {num(i.lam_par, 2)}. " + tail)


# ------------------------------------------------------------------------------------------------------ Experiment 3: Zufallsstarts, Asymmetrie

def make_starts(seed: int, n: int, box: float = C.START_BOX):
    rng = Rng(seed)
    u = rng.uniform((2, n), -box, box)
    return u[0], u[1]


@dataclass(frozen=True)
class AsymInfo:
    n: int
    pure: int
    still: int
    other: int
    chaos: int
    on_diagonal: int
    symmetric_start: bool
    lam_median: float
    s: float
    delta_s: float
    delta_k: float


def asym_info(beispiel: int, s: float, delta_s: float, delta_k: float, n: int, seed: int, symmetric: bool, burn=4000, meas=1500) -> AsymInfo:
    _name, a1, a2 = C.EXAMPLES[beispiel]
    u1, u2 = make_starts(seed, n)
    if symmetric:
        u2 = u1.copy()
    o = A.run2(u1, u2, A.Pair.with_deviation(a1, a2, s, delta_s, delta_k), burn=burn, meas=meas)
    return AsymInfo(n, int(o.pure.sum()), int(o.still.sum()), int(o.other.sum()), int(o.chaos.sum()), int(o.on_diagonal.sum()), symmetric, float(np.median(o.lam)), s, delta_s, delta_k)


def verdict_asym(i: AsymInfo) -> tuple[str, str]:
    if i.symmetric_start:
        return "info", (f"Alle {i.n} Starts liegen exakt auf der Diagonale (u₂ = u₁): {i.on_diagonal} von {i.n} bleiben dort, weil beide Spieler bitgleich dieselbe Rechnung ausführen; "
                        f"{i.other} von {i.n} kreisen oder laufen unregelmäßig. Der Exponent im Raum beider Spieler ist im Median {num(i.lam_median, 2)} (er misst die Querrichtung). Das ist eine Eigenschaft der symmetrischen Rechnung, keine Aussage über Rundung.")
    if i.pure == i.n:
        return "success", (f"{i.pure} von {i.n} Starts enden in einem reinen Gleichgewicht (ein Spieler auf Kante 1, der andere auf Kante 2); kein Start blieb auf der Diagonale oder kreiste. "
                           "Chaos wurde in dieser Demo nur auf der symmetrischen Diagonale gemessen (Experiment 1), nicht bei diesen unsymmetrischen Starts.")
    rest = i.n - i.pure
    return "info", (f"{i.pure} von {i.n} Starts enden in einem reinen Gleichgewicht, {i.still} stehen in einem gemischten Punkt still, {i.other} kreisen oder laufen unregelmäßig "
                    f"(davon λ > {num(C.CHAOS_LAMBDA, 3)}: {i.chaos}); übrig bleiben {rest} von {i.n}.")


# ------------------------------------------------------------------------------------------------------ Experimente 5 und 6: 3×3

@dataclass(frozen=True)
class Live33:
    n: int
    conv: int
    chaos: int
    other: int
    lam_q: tuple[float, float, float]   # Quantile 0, 0,5, 1 von λ
    minp_log10_median: float
    spanne_median: float
    g: float
    s: float
    e1: float
    e2: float
    lam_two_orbit_median: float | None


def live33(e1: float, e2: float, s: float, g: float, n: int, seed: int, burn=3000, meas=3000, two_orbit=False) -> Live33:
    rng = Rng(seed)
    A_ = np.array([B.rps(e1)] * n)
    C_ = np.array([B.rps(e2)] * n)
    p0 = rng.uniform((n, 3), -C.START_BOX_3, C.START_BOX_3)
    q0 = rng.uniform((n, 3), -C.START_BOX_3, C.START_BOX_3)
    lam, minp, rg = B.run_games(A_, C_, s, g, p0, q0, burn=burn, meas=meas)
    conv, chaos, oth = B.classify(lam, minp, rg)
    lam2 = None
    if two_orbit:
        m2 = min(n, 20)
        lam2 = float(np.median(B.two_orbit_lyapunov(A_[:m2], C_[:m2], s, g, p0[:m2], q0[:m2], burn=burn, meas=meas)))
    with np.errstate(divide="ignore"):
        lm = float(np.median(np.log10(np.maximum(minp, 1e-300))))
    return Live33(n, int(conv.sum()), int(chaos.sum()), int(oth.sum()), tuple(float(v) for v in np.quantile(lam, (0, 0.5, 1))), lm, float(np.median(rg)), g, s, e1, e2, lam2)


def orbit33(e1, e2, s, g, seed, burn=3000, n=1500):
    """Eine Bahn (Wahrscheinlichkeiten x von Spieler 1, y von Spieler 2) im Simplex nach `burn` Einschwingschritten."""
    rng = Rng(seed)
    p = rng.uniform((1, 3), -C.START_BOX_3, C.START_BOX_3)
    q = rng.uniform((1, 3), -C.START_BOX_3, C.START_BOX_3)
    A_, C_ = np.array([B.rps(e1)]), np.array([B.rps(e2)])
    p, q = B.normalise(p), B.normalise(q)
    for _ in range(burn):
        p, q, _x, _y = B.step(p, q, A_, C_, s, g)
    xs, ys = [], []
    for _ in range(n):
        p, q, x, y = B.step(p, q, A_, C_, s, g)
        xs.append(x[0].copy())
        ys.append(y[0].copy())
    return np.array(xs), np.array(ys)


def verdict_hedge33(i: Live33, frozen: dict | None) -> tuple[str, str]:
    base = f"Spiel R + e·I mit e₁ = {num(i.e1, 2)}, e₂ = {num(i.e2, 2)}, s = {num(i.s, 3)}, reines Hedge (g = 0), {i.n} Starts: {i.conv} konvergiert, {i.other} nicht konvergent ohne inneres Chaos, {i.chaos} chaotisch im Inneren. "
    if i.chaos == 0:
        tail = "Das ist eine Messung in dieser Spielfamilie, kein Beweis."
        if frozen is not None:
            tail = (f"In der vorgerechneten Reihe (25 Spiele × 40 Starts × 4 Schrittweiten) waren {frozen['chaos_total']} von {ints(frozen['n_total'])} Bahnen im Inneren chaotisch; das ist eine Messung in dieser Spielfamilie, kein Beweis.")
        return "success", base + tail
    return "warning", base + "Hier wurden Bahnen mit positivem Exponent im Inneren gezählt: eine Abweichung von der vorgerechneten Reihe, bitte Messdauer und Randgrenze prüfen."


def verdict_vergessen(i: Live33, window: dict | None) -> tuple[str, str]:
    if i.g <= 0:
        return "warning", "Mit Vergessensrate g = 0 ist die Lernregel reines Hedge; Experiment 6 zeigt das Vergessen und braucht g > 0."
    base = (f"Mit Vergessen g = {num(i.g, 3)} (nicht mehr Hedge), s = {num(i.s, 3)}, Spiel e₁ = {num(i.e1, 2)}, e₂ = {num(i.e2, 2)}: {i.chaos} von {i.n} Starts chaotisch "
            f"(λ im Median {num(i.lam_q[1], 3)}), {i.conv} konvergiert. ")
    if i.chaos >= 0.9 * i.n:
        tail = "Das gilt in dieser Spielfamilie und in diesem schmalen Parameterfenster; die Karten unten zeigen, wie schmal es ist."
        return "success", base + tail
    if i.chaos > 0:
        return "info", base + "Am Rand des Fensters: nur ein Teil der Starts ist chaotisch."
    return "info", base + "Außerhalb des Fensters (Karten unten): hier gibt es für diese Parameter kein inneres Chaos."


# ------------------------------------------------------------------------------------------------------ Nullsumme

def verdict_nullsumme(eps: float, row: dict | None) -> tuple[str, str]:
    """`row`: vorgerechnete Zeile (10⁵ Schritte): vol (mittleres ln det je Schritt), kl, quot (Exponent im Quotientenraum, 20 000 Schritte, kann fehlen)."""
    if row is None:
        return "warning", "Für diese Schrittweite liegt keine vorgerechnete Messung vor."
    vol = row["vol"]
    vol_txt = "0" if abs(vol) < 1e-12 else f"{vol:+.1e}".replace(".", ",")
    quot = ("Der größte Exponent im Quotientenraum liegt unter der Chaos-Grenze " + num(C.CHAOS_LAMBDA, 3) + " (vorgerechnet, 20 000 Schritte); "
            "Volumenwachstum im Dualraum sagt nahe am Rand nichts über Instabilität.")
    if row.get("minp", 0.0) > 1e-3:
        return "info", (f"ε = {num(eps, 2)}: Das mittlere ln det je Schritt über die Messung (10⁵ Schritte) ist {vol_txt}: Das Volumen wächst im Dualraum im Mittel. Der KL-Abstand zum Gleichgewicht ist am Ende "
                        f"{num(row['kl'], 1)}, die kleinste Wahrscheinlichkeit {num(row['minp'], 3)}: Die Bahn liegt am Ende der Messung noch im Inneren des Simplex. " + quot)
    return "info", (f"ε = {num(eps, 2)}: Das Volumen wächst im Dualraum nur anfangs (mittleres ln det je Schritt {vol_txt}), der KL-Abstand zum Gleichgewicht ist am Ende {num(row['kl'], 1)}: "
                    "die Bahn ist an den Rand des Simplex gewandert. " + quot)


# ------------------------------------------------------------------------------------------------------ Kernfrage aus den vorgerechneten Reihen

def kernzahlen(fz: dict) -> dict | None:
    """Die Zählungen der Kernbotschaft aus den vorgerechneten Reihen (None, wenn die Datei fehlt oder unvollständig ist)."""
    try:
        b1, b2 = fz["bifurkation"]["1"], fz["bifurkation"]["2"]
        asym_b = fz["asym_b"]["rows"]
        asym_c = [r for r in fz["asym_c"]["rows"] if r["delta"] > 0]
        g9 = fz["v9g"]["rows"]
        h33 = fz["hedge_rpe"]["rows"]
        v = fz["vergessen"]["main"]["meas"][-1]
        return {
            "thr1": A.threshold(*C.EXAMPLES[1][1:]), "thr2": A.threshold(*C.EXAMPLES[2][1:]),
            "ch1": b1["chaotic"], "above1": b1["above"], "ch2": b2["chaotic"], "above2": b2["above"], "first2": b2["first_chaotic"],
            "asym_b_pure": sum(r["pure"] for r in asym_b), "asym_b_n": sum(r["n"] for r in asym_b),
            "asym_c_pure": sum(r["pure"] for r in asym_c), "asym_c_n": sum(r["n"] for r in asym_c), "asym_c_configs": len(asym_c),
            "v9g_chaos": sum(r["chaos"] for r in g9), "v9g_n": sum(r["n"] for r in g9),
            "h33_chaos": sum(r["chaos"] for r in h33), "h33_n": sum(r["n"] for r in h33),
            "verg_chaos": v["chaos"], "verg_n": fz["vergessen"]["main"]["n"], "verg_lam": v["lam_q"][2],
        }
    except (KeyError, IndexError, TypeError):
        return None


def verdict_kern(k: dict | None) -> tuple[str, str]:
    if k is None:
        return "warning", "Die vorgerechneten Reihen (data/lsc_vorgerechnet.json) fehlen; die Kernzahlen können nicht angezeigt werden."
    return "success", (
        f"Auf der symmetrischen Diagonale verliert Hedge den gemischten Punkt bei s = {num(k['thr1'], 2)} (Beispiel 1) bzw. {num(k['thr2'], 2)} (Beispiel 2) die Stabilität; quer zur Diagonale ist er bei s > 0 instabil. "
        f"Darüber gibt es Zyklen und, **nur auf der symmetrischen Diagonale**, Chaos "
        f"(Beispiel 2: {k['ch2']} von {k['above2']} gemessenen Werten oberhalb der Schwelle; Beispiel 1: {k['ch1']} von {k['above1']}). "
        f"Die Diagonale ist quer instabil: Von {ints(k['asym_c_n'])} unsymmetrischen Starts (Beispiel 2, {k['asym_c_configs']} Konfigurationen) endeten {ints(k['asym_c_pure'])} in einem reinen Gleichgewicht. "
        f"Das gilt für diese Spielklasse (2 Spieler, 2 Kanten) und diese Parameter. Mit Vergessen (kein Hedge mehr) gab es in einem 3×3-Spiel dagegen Chaos für {k['verg_chaos']} von {k['verg_n']} Starts."
    )
