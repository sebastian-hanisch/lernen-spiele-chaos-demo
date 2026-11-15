"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_select_slider_at_its_first_and_last_option_does_not_raise():
    from lsc_presets import SETTING_SPECS   # die Rohwerte der Optionen (die Anzeige ist formatiert)
    keys = [s.key for s in _fresh_app().select_slider]
    assert keys, "keine Auswahl-Regler gefunden"
    for key in keys:
        for edge in (0, -1):
            at = _fresh_app()
            at.select_slider(key=key).set_value(SETTING_SPECS[key].options[edge]).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_the_symmetric_start_checkbox_and_the_example_switch_do_not_raise():
    at = _fresh_app()
    at.checkbox(key="sym").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at.radio(key="beispiel").set_value(2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_shows_the_six_experiments_and_a_verdict_box_each():
    at = _fresh_app()
    for name in [b.label for b in at.sidebar.button if b.key and b.key.startswith("preset_")]:
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert len([h for h in at.subheader if h.value.startswith("🔬 Experiment")]) == 6
        assert len(at.success) + len(at.info) + len(at.warning) >= 7        # Kernfrage und sechs Verdikte
