"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lsc_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_bifurcation(ss, xs, lam_par, lam_perp, s_mark, threshold):
    """Oben: Bifurkationsbild der Diagonale (Wahrscheinlichkeit x der Kante 1 über s); unten: Exponent längs (λ_längs) und quer (λ_quer) über s."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    xx = np.repeat(ss[None, :], xs.shape[0], axis=0).ravel()
    fig.add_trace(go.Scattergl(x=xx, y=xs.ravel(), mode="markers", marker=dict(size=2, color=C.COLOR_LINE), name="x nach Einschwingen", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ss, y=lam_par, mode="lines", line=dict(color=C.COLOR_MAIN, width=2), name="λ längs der Diagonale"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ss, y=lam_perp, mode="lines", line=dict(color=C.COLOR_ALT, width=2), name="λ quer zur Diagonale"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)
    if threshold is not None:
        fig.add_vline(x=threshold, line=dict(color="gray", width=1, dash="dash"))
    fig.add_vline(x=s_mark, line=dict(color=C.COLOR_ALT, width=1))
    fig.update_yaxes(title_text="x (Kante 1)", range=[-0.02, 1.02], row=1, col=1)
    fig.update_yaxes(title_text="Exponent", row=2, col=1)
    fig.update_xaxes(title_text="Schrittweite s", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_quer(dist):
    """Abstand |u₁ − u₂| der beiden Spieler über die Schritte (logarithmische Achse)."""
    fig = go.Figure(go.Scatter(x=list(range(len(dist))), y=list(dist), mode="lines+markers", line=dict(color=C.COLOR_ALT, width=2), marker=dict(size=5), name="|u₁ − u₂|",
                               hovertemplate="Schritt %{x}: %{y:.3g}<extra></extra>"))
    fig.update_yaxes(type="log", title_text="Abstand |u₁ − u₂| (Logit)")
    fig.update_xaxes(title_text="Schritt")
    return _base(fig, 320)


def build_outcomes(info):
    """Ausgang der Starts: reines Gleichgewicht, gemischt (still), kreisend oder unregelmäßig."""
    labels = ["reines Gleichgewicht", "gemischt, still", "kreisend oder unregelmäßig"]
    vals = [info.pure, info.still, info.other]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[C.COLOR_MAIN, C.COLOR_LINE, C.COLOR_ALT], text=vals, textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_yaxes(title_text=f"Zahl der Starts (von {info.n})", range=[0, max(info.n * 1.12, 1)])
    return _base(fig, 300)


def build_nullsumme(trace):
    """Nullsumme (vorgerechnet in der Tabelle, hier die ersten Schritte live): KL-Abstand zur Gleichverteilung und kleinste Wahrscheinlichkeit."""
    t = [r[0] for r in trace]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KL-Abstand zur Gleichverteilung", "kleinste Wahrscheinlichkeit (log₁₀)"))
    fig.add_trace(go.Scatter(x=t, y=[r[1] for r in trace], mode="lines", line=dict(color=C.COLOR_ALT, width=2), name="KL-Abstand"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[float(np.log10(max(r[2], 1e-300))) for r in trace], mode="lines", line=dict(color=C.COLOR_MAIN, width=2), name="log₁₀ min p"), row=1, col=2)
    fig.update_xaxes(title_text="Schritt")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_raster(rows):
    """Vier kleine Karten (je Schrittweite s): Anteil konvergenter Starts je Spiel (e₁ Zeile, e₂ Spalte); der Text nennt die Zahl chaotischer Bahnen (immer 0 in dieser Messung)."""
    titles = [f"s = {r['s']:g}" for r in rows]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, vertical_spacing=0.14)
    for k, r in enumerate(rows):
        es = sorted({c["e1"] for c in r["cells"]})
        z = [[next(c["conv"] for c in r["cells"] if c["e1"] == a and c["e2"] == b) for b in es] for a in es]
        fig.add_trace(go.Heatmap(z=z, x=[f"{b:+.1f}" for b in es], y=[f"{a:+.1f}" for a in es], zmin=0, zmax=1, colorscale="Teal", showscale=(k == 0),
                                 colorbar=dict(title="konvergiert", len=0.9), hovertemplate="e₁ %{y}, e₂ %{x}: konvergiert %{z:.2f}<extra></extra>"), row=k // 2 + 1, col=k % 2 + 1)
    fig.update_xaxes(title_text="e₂", type="category")
    fig.update_yaxes(title_text="e₁", type="category")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_simplex(xs):
    """Bahn der Wahrscheinlichkeiten von Spieler 1 im Dreieck (Simplex)."""
    fig = go.Figure(go.Scatterternary(a=xs[:, 0], b=xs[:, 1], c=xs[:, 2], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=1), marker=dict(size=3, color=C.COLOR_ALT),
                                      hovertemplate="Stein %{a:.3f}, Schere %{b:.3f}, Papier %{c:.3f}<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=30, r=30, t=20, b=20), ternary=dict(aaxis=dict(title="Stein"), baxis=dict(title="Schere"), caxis=dict(title="Papier")))
    return fig


def build_window(frozen):
    """Fenster des Vergessens (vorgerechnet): Anteil chaotischer Starts über g, über s und über das Spiel (e₁, e₂)."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("über die Vergessensrate g", "über die Schrittweite s", "über das Spiel (e₂ rechts, e₁ oben)"), column_widths=[0.28, 0.28, 0.44])
    gs = frozen["g_sweep"]
    fig.add_trace(go.Bar(x=[f"{r['g']:g}" for r in gs], y=[r["chaos"] for r in gs], marker_color=C.COLOR_MAIN, name="chaotisch", hovertemplate="g = %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    ss = frozen["s_sweep"]
    fig.add_trace(go.Bar(x=[f"{r['s']:g}" for r in ss], y=[r["chaos"] for r in ss], marker_color=C.COLOR_MAIN, name="chaotisch", hovertemplate="s = %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    gr = frozen["grid"]
    fig.add_trace(go.Heatmap(z=gr["chaos"], x=[f"{v:+.1f}" for v in gr["e2"]], y=[f"{v:+.1f}" for v in gr["e1"]], zmin=0, zmax=1, colorscale="Teal", showscale=False,
                             hovertemplate="e₁ %{y}, e₂ %{x}: %{z:.2f}<extra></extra>"), row=1, col=3)
    fig.update_yaxes(range=[0, 1.05], title_text="Anteil chaotisch", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed", row=1, col=3)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)
