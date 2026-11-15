"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Schrittweiten sind nie Zweierpotenzen und nie genau eine Schwelle: Presets und Tests nehmen s = 8,001 oder 7,99, nie 8,0 (Rechner-Artefakt bei Mehrfachwert -1,
DEMO-PLAYBOOK-Hinweis zu Zweierpotenzen). Die Reihen der README (vorgerechnet) benutzen durchweg s + S_SHIFT."""

S_SHIFT = 0.00137                       # Verschiebung der Messwerte gegen Zweierpotenzen und Schwellen (wie in der Vormessung)
CHAOS_LAMBDA = 0.003                    # Grenze, ab der ein Exponent als positiv gezählt wird (Messdauer-abhängig, siehe README)
MIN_PROB = 1e-9                         # Randgrenze: kleinere Wahrscheinlichkeiten zählen als Rand, nicht als inneres Chaos
CONV_RANGE = 1e-6                       # Spannweite der Wahrscheinlichkeiten im letzten Fenster, unter der eine Bahn als konvergiert gilt

# Beispiele des Zwei-Spieler-Zwei-Kanten-Überlastungsspiels: Kosten a1, a2 der Kanten je Last
EXAMPLES = {
    1: ("Beispiel 1: gleiche Kanten (a₁ = a₂ = ½)", 0.5, 0.5),
    2: ("Beispiel 2: a₁ = ¼, a₂ = 7/20", 0.25, 0.35),
}
DEFAULT_EXAMPLE = 1
S_MIN, S_MAX, DEFAULT_S = 1.0, 80.0, 8.001
DK_MIN, DK_MAX, DEFAULT_DK = 3, 12, 12             # Anfangsunterschied δ = 10^-dk (Experiment 2)
DELTAS = (0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1)         # Abweichung von Spieler 2 (Schrittweite bzw. Kosten) in Experiment 3
K_MIN, K_MAX, DEFAULT_K = 50, 2000, 400            # Zahl der Anfangswerte (Experiment 3)
EPS_VALUES = (0.01, 0.05, 0.1, 0.3, 1.0)           # Schrittweiten ε des Nullsummen-Experiments (vorgerechnet)
DEFAULT_EPS = 0.05
E_MIN, E_MAX = -0.6, 0.6                           # Diagonaleintrag e der 3×3-Spiele R + e·I
S33_MIN, S33_MAX, DEFAULT_S33 = 0.1, 12.0, 1.001
G_MIN, G_MAX, DEFAULT_G = 0.0, 0.2, 0.05           # Vergessensrate
SV_MIN, SV_MAX, DEFAULT_SV = 0.1, 3.0, 0.501
SEED_MAX, DEFAULT_SEED = 999999, 35
START_BOX = 3.0                                    # Anfangs-Logits in [-3, 3] (2×2) bzw. Dualkoordinaten in [-2, 2] (3×3)
START_BOX_3 = 2.0

COLOR_MAIN = "#2F6B65"
COLOR_ALT = "#c0504d"
COLOR_GRAY = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"

DEFAULTS = {
    "beispiel": DEFAULT_EXAMPLE, "s": DEFAULT_S, "dk": DEFAULT_DK, "delta_s": 0.0, "delta_k": 0.0, "k_starts": DEFAULT_K,
    "sym": False, "eps": DEFAULT_EPS, "e1": 0.0, "e2": 0.4, "s33": DEFAULT_S33, "g": DEFAULT_G, "sv": DEFAULT_SV,
    "e1v": 0.0, "e2v": 0.4, "seed": DEFAULT_SEED,
}

PRESETS = {
    "Schwelle von Hand (Beispiel 1)": {},
    "Chaos auf der Diagonale (Beispiel 2)": {"beispiel": 2, "s": 40.001},
    "Asymmetrie: reines Gleichgewicht": {"beispiel": 2, "s": 40.001, "delta_s": 1e-3},
    "Nullsumme: der Rand gewinnt": {"eps": 0.1},
    "3×3 mit reinem Hedge": {"s33": 3.001},
    "Mit Vergessen (kein Hedge mehr)": {"g": 0.02},
}
PRESET_HELP = {
    "Schwelle von Hand (Beispiel 1)": "Zwei Spieler, zwei gleiche Kanten, Schrittweite s = 8,001 knapp über der Schwelle 8: Auf der Diagonale ist der gemischte Punkt gerade instabil (quer zur Diagonale ist er es bei s > 0).",
    "Chaos auf der Diagonale (Beispiel 2)": "Beispiel 2 bei s = 40,001: beide Spieler starten gleich (Diagonale), dort ist der Exponent positiv. Experiment 2 zeigt, dass die Diagonale quer instabil ist.",
    "Asymmetrie: reines Gleichgewicht": "Wie zuvor, aber Spieler 2 lernt mit einer um 0,1 % größeren Schrittweite: Die zufälligen Starts enden in einem reinen Gleichgewicht, auch ohne die Abweichung (sie ändert nur, was die exakt symmetrisch gestartete Bahn tut).",
    "Nullsumme: der Rand gewinnt": "Stein-Schere-Papier mit ε = 0,1: Das Volumen wächst nur anfangs, die Bahn wandert an den Rand des Simplex.",
    "3×3 mit reinem Hedge": "Spiel R + e·I mit e₁ = 0, e₂ = 0,4 und s = 3,001: kein inneres Chaos gemessen.",
    "Mit Vergessen (kein Hedge mehr)": "Dasselbe Spiel, aber mit Vergessensrate g = 0,02 und s = 0,501: eine andere Lernregel, kein Hedge mehr.",
}
