"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import lsc_constants as C


def _to_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "ja")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type | object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Regler mit festen Werten (select_slider): der Permalink rastet auf den nächsten ein


SETTING_SPECS = {
    "beispiel": SettingSpec("bsp", int, C.DEFAULTS["beispiel"], 1, len(C.EXAMPLES)),
    "s": SettingSpec("s", float, C.DEFAULTS["s"], C.S_MIN, C.S_MAX),
    "dk": SettingSpec("dk", int, C.DEFAULTS["dk"], C.DK_MIN, C.DK_MAX),
    "delta_s": SettingSpec("dels", float, C.DEFAULTS["delta_s"], min(C.DELTAS), max(C.DELTAS), C.DELTAS),
    "delta_k": SettingSpec("delk", float, C.DEFAULTS["delta_k"], min(C.DELTAS), max(C.DELTAS), C.DELTAS),
    "k_starts": SettingSpec("starts", int, C.DEFAULTS["k_starts"], C.K_MIN, C.K_MAX),
    "sym": SettingSpec("sym", _to_bool, C.DEFAULTS["sym"]),
    "eps": SettingSpec("eps", float, C.DEFAULTS["eps"], min(C.EPS_VALUES), max(C.EPS_VALUES), C.EPS_VALUES),
    "e1": SettingSpec("e1", float, C.DEFAULTS["e1"], C.E_MIN, C.E_MAX),
    "e2": SettingSpec("e2", float, C.DEFAULTS["e2"], C.E_MIN, C.E_MAX),
    "s33": SettingSpec("s33", float, C.DEFAULTS["s33"], C.S33_MIN, C.S33_MAX),
    "g": SettingSpec("g", float, C.DEFAULTS["g"], C.G_MIN, C.G_MAX),
    "sv": SettingSpec("sv", float, C.DEFAULTS["sv"], C.SV_MIN, C.SV_MAX),
    "e1v": SettingSpec("e1v", float, C.DEFAULTS["e1v"], C.E_MIN, C.E_MAX),
    "e2v": SettingSpec("e2v", float, C.DEFAULTS["e2v"], C.E_MIN, C.E_MAX),
    "seed": SettingSpec("seed", int, C.DEFAULTS["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}   # Preset-Schlüssel = Reglerschlüssel


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                value = min(spec.options, key=lambda o: abs(o - value))
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die im Preset genannten auf ihren Wert, die übrigen auf die Voreinstellung (sonst hinge das Ergebnis an der Klickreihenfolge)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)
