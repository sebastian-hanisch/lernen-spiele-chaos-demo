"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, paarweise verschieden; Permalink rastet auf die Optionen ein."""

import pytest

import lsc_constants as C
import lsc_evaluation as E
import lsc_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            if spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings_and_all_presets_differ():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    assert E.Settings(**{**C.DEFAULTS, **C.PRESETS[first]}) == E.Settings()
    full = [tuple(sorted({**C.DEFAULTS, **p}.items())) for p in C.PRESETS.values()]
    assert len(set(full)) == len(full)       # sonst wären zwei Presets nicht unterscheidbar (Hervorhebung)


def test_setting_specs_cover_every_default_and_settings_field():
    assert set(P.SETTING_SPECS) == set(C.DEFAULTS) == set(E.Settings.__dataclass_fields__)


def test_no_preset_or_default_uses_a_power_of_two_or_a_threshold_exactly():
    forbidden = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 8.0, 160 / 9}
    for name, preset in C.PRESETS.items():
        for key, value in {**C.DEFAULTS, **preset}.items():
            if key in ("s", "s33", "sv"):
                assert value not in forbidden, (name, key, value)


def test_the_selection_options_are_sorted_and_contain_the_defaults():
    assert list(C.DELTAS) == sorted(C.DELTAS) and 0.0 in C.DELTAS
    assert C.DEFAULT_EPS in C.EPS_VALUES and C.EPS_VALUES == tuple(sorted(C.EPS_VALUES))


@pytest.mark.parametrize("key,raw,expected", [("delta_s", 0.0072, 1e-2), ("eps", 0.07, 0.05), ("eps", 7.0, 1.0)])
def test_permalink_value_snaps_to_the_nearest_option(key, raw, expected):
    spec = P.SETTING_SPECS[key]
    assert min(spec.options, key=lambda o: abs(o - raw)) == expected
