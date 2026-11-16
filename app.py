"""Lernen in Spielen: Wann Hedge kreist, springt oder chaotisch wird - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Reihe „Dynamische Systeme“: Hedge (multiplikative Gewichte) in Zwei-Spieler-Spielen, die Schrittweite über der Schwelle. Sechs Experimente mit
Verdikten; schwere Messreihen sind vorgerechnet (data/lsc_vorgerechnet.json, tools/precompute_lsc.py), der Rest rechnet live.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lsc_algorithm as A
import lsc_bimatrix as B
import lsc_constants as C
import lsc_evaluation as E
import lsc_frozen
from lsc_layout import dunkles_design, kopf_css
from lsc_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lsc_visualization import (
    build_bifurcation,
    build_nullsumme,
    build_outcomes,
    build_quer,
    build_raster,
    build_simplex,
    build_window,
)

st.set_page_config(page_title="Lernen in Spielen – Sebastian Hanisch", layout="wide")

FZ = lsc_frozen.load()


@st.cache_data(show_spinner=False)
def _schwelle(bsp, s):
    return E.schwelle_info(bsp, s)


@st.cache_data(show_spinner=False)
def _picture(bsp):
    _n, a1, a2 = C.EXAMPLES[bsp]
    return E.bifurcation_picture(a1, a2)


@st.cache_data(show_spinner=False)
def _quer(bsp, s, dk):
    return E.quer_info(bsp, s, dk)


@st.cache_data(show_spinner=False)
def _asym(bsp, s, ds, dkost, n, seed, sym):
    return E.asym_info(bsp, s, ds, dkost, n, seed, sym)


@st.cache_data(show_spinner=False)
def _live33(e1, e2, s, g, n, seed, two_orbit):
    return E.live33(e1, e2, s, g, n, seed, two_orbit=two_orbit)


@st.cache_data(show_spinner=False)
def _orbit33(e1, e2, s, g, seed):
    return E.orbit33(e1, e2, s, g, seed)


@st.cache_data(show_spinner=False)
def _zero_sum_trace(eps):
    return B.zero_sum_run(B.rps(0.0), eps, steps=6000, burn=1000, record_every=60)[3]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lsc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Lernen in Spielen: Wann Hedge kreist, springt oder chaotisch wird")
st.markdown(
    """
Zwei Spieler lernen gleichzeitig mit **Hedge** (multiplikative Gewichte): Jeder gewichtet seine Aktionen nach den bisherigen Kosten, mit einer **Schrittweite s**. Bei kleinem s
konvergiert das, bei großem s nicht mehr. Diese Demo fährt s über die **Schwelle des gemischten Gleichgewichts** und misst, was dann passiert: Zyklen, **Chaos auf der symmetrischen Diagonale** (in Beispiel 2)
und was von diesem Chaos übrig bleibt, wenn die Spieler nicht exakt gleich starten.

Sie ist die Gegenseite zur Demo [noregret-demo](https://github.com/sebastian-hanisch/noregret-demo) (Regret und Konvergenz bei kleiner Schrittweite). Unten: sechs **🔬 Experimente**,
**🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Hedge in diesem Spiel", expanded=True):
    st.markdown(
        """
1. **Das Spiel:** Zwei Spieler wählen je eine von zwei Kanten. Die Kosten einer Kante sind ihr Kostensatz mal die Last; die Last ist 1, plus 1, wenn der andere dieselbe Kante wählt.
2. **Hedge:** Jeder Spieler hält Gewichte für seine Kanten und multipliziert sie nach jeder Runde mit `exp(−s · erwartete Kosten)`. Mit dem Logit `u = ln(x/(1 − x))` der Wahrscheinlichkeit x für Kante 1
   ist das `u ← u − s · (Kostendifferenz)`, eine Abbildung wie in der logistischen Abbildung.
3. **Die Diagonale:** Starten beide Spieler mit derselben Mischung, bleibt es so. Dort ist das Verhalten eine **eindimensionale Abbildung** (Periodenverdopplung, Fenster, Chaos). Die Frage der Demo ist,
   ob das auch gilt, wenn die Spieler nur *fast* gleich starten.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Überlastungsspiel (Experimente 1 bis 3)**")
    st.radio("Spielbeispiel", [1, 2], format_func=lambda k: C.EXAMPLES[k][0], key="beispiel", help="Kosten der beiden Kanten je Last. Beispiel 1 ist symmetrisch, Beispiel 2 hat den gemischten Punkt bei x* = 0,75.")
    st.slider("Schrittweite s", C.S_MIN, C.S_MAX, step=0.001, format="%.3f", key="s",
              help="Hedge mit exp(−s·Kosten); entspricht MWU_ε mit s = −ln(1 − ε). Die Schwelle des gemischten Punkts steht in Experiment 1.")
    st.markdown("**Zufall**")
    st.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="seed", step=1, help="Legt die Anfangswerte der Experimente 3, 5 und 6 fest.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})

ss = E.Settings(**{key: st.session_state[key] for key in SETTING_SPECS})
a1, a2 = ss.costs

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann kreist, springt oder wird Hedge chaotisch?")
kern = E.kernzahlen(FZ)
if kern is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Schwelle des gemischten Punkts", f"s = {E.num(kern['thr1'], 2)} / {E.num(kern['thr2'], 2)}", delta="Beispiel 1 / Beispiel 2", delta_color="off")
    c2.metric("Diagonale chaotisch (Beispiel 2)", f"{kern['ch2']} von {kern['above2']}", delta="Werte oberhalb der Schwelle", delta_color="off")
    c3.metric("Unsymmetrisch: reines Gleichgewicht", f"{E.ints(kern['asym_c_pure'])} von {E.ints(kern['asym_c_n'])}", delta="Starts, Beispiel 2", delta_color="off")
    c4.metric("Mit Vergessen: chaotische Starts", f"{kern['verg_chaos']} von {kern['verg_n']}", delta="kein Hedge mehr, 3×3-Spiel", delta_color="off")
kind, text = E.verdict_kern(kern)
getattr(st, kind)(text)
st.caption("Alle vier Zahlen sind vorgerechnet (tools/precompute_lsc.py, feste Zufalls-Seeds); die Experimente darunter rechnen kleine Fassungen live.")

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Schwelle von Hand")
st.caption("Beide Spieler starten gleich (Diagonale). Schrittweite s und Beispiel stehen in der Seitenleiste; Werte knapp um die Schwelle (7,99 und 8,001) zeigen den Wechsel.")
info1 = _schwelle(ss.beispiel, ss.s)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gemischter Punkt x*", E.num(info1.xs, 3) if info1.xs is not None else "–")
m2.metric("Schwelle s", E.num(info1.threshold, 3) if info1.threshold is not None else "–")
m3.metric("Faktor 1 − c (längs)", E.num(1 - info1.c, 3) if info1.c is not None else "–", delta="|Faktor| > 1: instabil" if info1.c is not None and abs(1 - info1.c) > 1 else "stabil", delta_color="off")
m4.metric("Periode der Bahn", str(info1.period) if info1.period is not None else "> 64", delta=f"λ längs {E.num(info1.lam_par, 3)}", delta_color="off")
ss_pic, xs_pic, lp_pic, lq_pic = _picture(ss.beispiel)
st.plotly_chart(build_bifurcation(ss_pic, xs_pic, lp_pic, lq_pic, ss.s, info1.threshold), width="stretch", key="bif_chart")
st.caption("Oben: die Wahrscheinlichkeit x nach dem Einschwingen über s (senkrechte Linien: Schwelle gestrichelt, gewähltes s rot). Unten: Exponent längs der Diagonale (grün) und quer dazu (rot); "
           "kurze Messung (1 500 Schritte), das Bild ist eine Skizze, die Zahlen der README sind vorgerechnet (60 000 Schritte).")
kind, text = E.verdict_schwelle(info1, FZ.get("bifurkation", {}).get(str(ss.beispiel)))
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Quer zur Diagonale")
st.slider("Anfangsunterschied δ = 10^(−k): k =", C.DK_MIN, C.DK_MAX, key="dk", help="Unterschied der Logits beider Spieler beim Start; k = 12 heißt 10⁻¹², weit unter jeder Messgenauigkeit.")
q = _quer(ss.beispiel, ss.s, ss.dk)
st.plotly_chart(build_quer(q.dist), width="stretch", key="quer_chart")
st.caption("Der Abstand beider Spieler in logarithmischer Achse: Eine Gerade nach oben ist exponentielles Wachstum mit der Rate λ_quer = Mittel von ln(1 + c) entlang der Diagonalbahn (Beispiel, s und δ wie eingestellt).")
kind, text = E.verdict_quer(q)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Zufällige Starts und Asymmetrie")
st.caption("Anfangswerte zufällig (Logits in [−3, 3]², Zufalls-Seed der Seitenleiste, SplitMix64). Spieler 2 darf eine abweichende Schrittweite oder abweichende Kosten auf Kante 2 haben.")
a, b, c, d = st.columns(4)
a.select_slider("Schrittweite von Spieler 2: Abweichung", C.DELTAS, key="delta_s", format_func=lambda v: f"+{v:g}" if v else "keine", help="Spieler 2 lernt mit s·(1 + Abweichung).")
b.select_slider("Kosten auf Kante 2 (Spieler 2): Abweichung", C.DELTAS, key="delta_k", format_func=lambda v: f"+{v:g}" if v else "keine", help="Spieler 2 hat die Kosten a₂·(1 + Abweichung).")
c.slider("Zahl der Starts", C.K_MIN, C.K_MAX, step=50, key="k_starts")
d.checkbox("Beide Spieler starten exakt gleich (Diagonale)", key="sym", help="u₂ = u₁ für alle Starts: die Rechnung beider Spieler ist bitgleich.")
with st.spinner("Rechne die Starts (rund 1 bis 2 Sekunden) …"):
    asym = _asym(ss.beispiel, ss.s, ss.delta_s, ss.delta_k, ss.k_starts, ss.seed, ss.sym)
st.plotly_chart(build_outcomes(asym), width="stretch", key="asym_chart")
kind, text = E.verdict_asym(asym)
getattr(st, kind)(text)
st.caption("Das bitgleiche Verhalten der Diagonale ist eine Eigenschaft der symmetrischen Rechnung (beide Spieler führen dieselben Gleitkommaoperationen aus), nicht der Rundung; "
           "ein Unterschied von 10⁻¹² verlässt die Diagonale in Experiment 2 nach rund 30 Schritten.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Nullsumme – das Volumen wächst, der Rand gewinnt")
st.select_slider("Schrittweite ε des Hedge-Updates", C.EPS_VALUES, key="eps", help="Stein-Schere-Papier (Nullsumme), Dualraum; die Messung ist vorgerechnet (10⁵ Schritte).")
row = next((r for r in FZ.get("nullsumme", {}).get("rows", []) if abs(r["eps"] - ss.eps) < 1e-12), None)
st.plotly_chart(build_nullsumme(_zero_sum_trace(ss.eps)), width="stretch", key="zero_chart")
st.caption("Die ersten 6 000 Schritte live (ein fester Startpunkt); die Zahlen im Verdikt stammen aus der vorgerechneten Messung (Volumen, KL-Abstand und kleinste Wahrscheinlichkeit über 10⁵ Schritte, der Exponent im Quotientenraum über 20 000 Schritte).")
kind, text = E.verdict_nullsumme(ss.eps, row)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: 3×3-Spiele mit reinem Hedge")
st.caption("Spiele R + e·I (R = Stein-Schere-Papier): Spieler 1 hat e₁, Spieler 2 hat e₂ auf der Diagonale seiner Auszahlungsmatrix. 100 zufällige Starts, 3 000 + 3 000 Schritte.")
a, b, c = st.columns(3)
a.slider("e₁", C.E_MIN, C.E_MAX, step=0.05, key="e1")
b.slider("e₂", C.E_MIN, C.E_MAX, step=0.05, key="e2")
c.slider("Schrittweite s", C.S33_MIN, C.S33_MAX, step=0.001, format="%.3f", key="s33")
l33 = _live33(ss.e1, ss.e2, ss.s33, 0.0, 100, ss.seed, False)
rows_rpe = FZ.get("hedge_rpe", {}).get("rows")
frozen33 = {"chaos_total": sum(r["chaos"] for r in rows_rpe), "n_total": sum(r["n"] for r in rows_rpe)} if rows_rpe else None
kind, text = E.verdict_hedge33(l33, frozen33)
getattr(st, kind)(text)
if rows_rpe:
    st.markdown("**Vorgerechnet: Anteil konvergenter Starts je Spiel** (25 Spiele × 40 Starts, vier Schrittweiten; chaotisch im Inneren: 0 in allen Zellen)")
    st.plotly_chart(build_raster(rows_rpe), width="stretch", key="raster_chart")

st.markdown("---")

# --- Experiment 6 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: 3×3 mit Vergessen – das ist nicht mehr Hedge")
st.caption("Dieselbe Lernregel plus Vergessen: p ← (1 − g)·p + s·A·y. Das ist ein anderes Verfahren; die Aussage gilt nur für diese Spielfamilie und dieses Parameterfenster.")
a, b, c, d = st.columns(4)
a.slider("Vergessensrate g", C.G_MIN, C.G_MAX, step=0.005, format="%.3f", key="g")
b.slider("Schrittweite s", C.SV_MIN, C.SV_MAX, step=0.001, format="%.3f", key="sv")
c.slider("e₁", C.E_MIN, C.E_MAX, step=0.05, key="e1v")
d.slider("e₂", C.E_MIN, C.E_MAX, step=0.05, key="e2v")
lv = _live33(ss.e1v, ss.e2v, ss.sv, ss.g, 100, ss.seed, ss.g > 0)
x_orb, _y_orb = _orbit33(ss.e1v, ss.e2v, ss.sv, ss.g, ss.seed)
kind, text = E.verdict_vergessen(lv, FZ.get("vergessen"))
getattr(st, kind)(text)
if lv.lam_two_orbit_median is not None:
    st.caption(f"Gegenprobe mit dem Zwei-Bahn-Verfahren (20 Starts): λ im Median {E.num(lv.lam_two_orbit_median, 3)}; Tangentenverfahren (100 Starts): {E.num(lv.lam_q[1], 3)}.")
st.plotly_chart(build_simplex(x_orb), width="stretch", key="simplex_chart")
st.caption("Bahn von Spieler 1 im Dreieck (Zufalls-Seed der Seitenleiste, 1 500 Schritte nach dem Einschwingen). Bei Chaos im Inneren füllt sie eine Fläche, ohne den Rand zu erreichen.")
if "vergessen" in FZ:
    st.markdown("**Vorgerechnet: das Parameterfenster** (Spiel e₁ = 0, e₂ = 0,4, g = 0,02, s = 0,5, falls nicht über die Achse verändert; 100 bzw. 40 Starts je Wert)")
    st.plotly_chart(build_window(FZ["vergessen"]), width="stretch", key="window_chart")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Beide Spieler starten exakt gleich und lernen gleich** (Diagonale) | Das Chaos der Diagonale ist ein Sattel: Schon ein Unterschied von 10⁻¹² wächst bei s = 40 mit rund +0,8 je Schritt (bei s = 26 mit rund +1,0, bei s = 50 mit rund +0,4), danach endet fast jeder Start in einem reinen Gleichgewicht. | Experimente 2 und 3 |
| **2 Spieler, 2 Kanten, die zwei gemessenen Beispiele** | Mehr als zwei Spieler oder Kanten sind nicht gemessen; die Folgerung der Quelle (Chaos für beliebig kleine Schrittweiten bei n Spielern) ist hier nicht nachgemessen. | – |
| **Reines Hedge** | Mit Vergessen (g > 0) gibt es in einem 3×3-Spiel inneres Chaos für alle gemessenen Starts; das ist ein anderes Verfahren, in einer Spielfamilie, in einem schmalen Fenster. | Experiment 6 |
| **Nullsumme: Volumenwachstum** | Volumenwachstum im Dualraum ist hier bei ε ≥ 0,05 nur anfangs vorhanden (bei ε = 0,01 liegt die Bahn am Ende der Messung noch im Inneren); der Exponent im Quotientenraum liegt unter der Chaos-Grenze 0,003. | Experiment 4 |
| **Messdauer und Randgrenze 10⁻⁹** | Bahnen am Rand (kleinste Wahrscheinlichkeit unter 10⁻⁹) werden nicht als Chaos gezählt; bei dreifacher Messdauer bleiben von ihnen nur 0 bis 7 % über der Grenze (README, nicht in der Messreihe). | README |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben ein Zwei-Spieler-Spiel mit Auszahlungsmatrizen $A$ (Spieler 1) und $C$ (Spieler 2), eine Schrittweite $s > 0$ und eine Vergessensrate $g \ge 0$. Gesucht ist das Verhalten der Bahn der gemischten Strategien.

| Symbol | Bedeutung |
|---|---|
| $x, y$ | gemischte Strategien von Spieler 1 und 2 (Wahrscheinlichkeitsvektoren) |
| $p, q$ | Dualkoordinaten (kumulierte Auszahlungen), $x = \mathrm{softmax}(p)$, $y = \mathrm{softmax}(q)$ |
| $s$ | Schrittweite; $s = -\ln(1-\varepsilon)$ für MWU mit Faktor $(1-\varepsilon)^{\text{Kosten}}$ |
| $g$ | Vergessensrate ($g = 0$: reines Hedge) |
| $a_1, a_2$ | Kosten je Last der Kanten 1 und 2 im Überlastungsspiel |
| $u$ | Logit $\ln\frac{x}{1-x}$ der Wahrscheinlichkeit für Kante 1 |
| $c$ | $c = s\,(a_1+a_2)\,x(1-x)$ |

**Hedge im Dualraum.** $p \leftarrow (1-g)\,p + s\,A\,y$ und $q \leftarrow (1-g)\,q + s\,C\,x$.

**Überlastungsspiel, zwei Kanten.** Jeder Spieler wählt eine Kante, die Kosten sind Kostensatz mal Last (1 plus 1, wenn der andere dieselbe Kante wählt). Für den Logit $u_i$ von Spieler $i$, dessen Gegner mit $x'$ spielt, gilt
$$u_i \leftarrow u_i - s\,\bigl(a_1(1+x') - a_2(2-x')\bigr), \qquad x' = \sigma(u_{3-i}).$$

**Diagonale und Schwelle.** Auf der Diagonale $u_1 = u_2 = u$ ist das eine eindimensionale Abbildung. Der gemischte Punkt ist $x^* = \dfrac{2a_2 - a_1}{a_1 + a_2}$ (nur für $0 < x^* < 1$). Die Jacobi-Matrix hat dort die festen Eigenrichtungen $(1,1)$ und $(1,-1)$ mit den Faktoren $1-c$ und $1+c$.
Der gemischte Punkt verliert die Stabilität längs der Diagonale bei
$$s^* = \frac{2}{(a_1+a_2)\,x^*(1-x^*)}.$$
Der Querfaktor $1 + c$ ist wegen $c > 0$ in jedem Schritt größer als 1: Die Diagonale ist quer instabil.

**Exponenten.** $\lambda_{\text{längs}} = \lim \frac1n\sum \ln|1-c_t|$ und $\lambda_{\text{quer}} = \lim \frac1n\sum \ln(1+c_t)$ entlang der Diagonalbahn. Für $m\times m$-Spiele: größter Exponent per Tangentenverfahren im Quotientenraum (Dualkoordinaten modulo $(1,\dots,1)$), Gegenprobe Zwei-Bahn-Verfahren.

**Nullsumme.** Für $C = -A^{\top}$ ist die Volumenänderung je Schritt $\ln\left|\det\begin{pmatrix} I & sA\,S(y)\\ -sA^{\top}S(x) & I\end{pmatrix}\right|$ mit $S(x) = \mathrm{diag}(x) - xx^{\top}$.

**Literatur** (Stufen der Lektüre: W = Wortlaut gelesen). Palaiopanos, Panageas, Piliouras (NeurIPS 2017) [W]: Hedge-Variante MWU$_\varepsilon$, Grenzzyklus der Periode 2 und Li-Yorke-Chaos in Lastverteilungsspielen.
Cheung, Piliouras (arXiv 2005.13996) [W bis Zeichen 80 000 von 137 000]: Volumenwachstum im Dualraum; Cheung, Tao (arXiv 2008.00540) [W bis Zeichen 80 000 von 129 000]: „fast überall“ bezieht sich auf die Klasse der Spiele.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html)."
)
