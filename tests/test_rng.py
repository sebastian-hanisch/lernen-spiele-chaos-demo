"""Zufallsgenerator SplitMix64: vektorisierte Fassung gegen eine schrittweise Schleife in reinem Python und gegen den bekannten Wert für Seed 0."""

import numpy as np

from lsc_rng import GAMMA, MASK, Rng


def _reference(seed, n):
    """Die schrittweise Fassung (Schleife, Python-Ganzzahlen) nach dem veröffentlichten Algorithmus."""
    state = seed & MASK
    out = []
    for _ in range(n):
        state = (state + GAMMA) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


def test_known_first_output_for_seed_zero():
    assert int(Rng(0).next_u64(1)[0]) == 16294208416658607535       # 0xE220A8397B1DCDAF, der bekannte erste Wert von SplitMix64 mit Seed 0


def test_vectorised_equals_stepwise_loop_for_many_seeds():
    for seed in (0, 1, 35, 99, 123456789, 2 ** 63 + 5, MASK):
        got = [int(v) for v in Rng(seed).next_u64(50)]
        assert got == _reference(seed, 50), seed


def test_consecutive_calls_continue_the_sequence():
    rng = Rng(7)
    first = [int(v) for v in rng.next_u64(10)]
    second = [int(v) for v in rng.next_u64(10)]
    assert first + second == _reference(7, 20)


def test_uniform_is_in_range_with_the_right_moments_and_is_reproducible():
    u = Rng(3).uniform((20000,), -2.0, 5.0)
    assert u.min() >= -2.0 and u.max() < 5.0
    assert abs(u.mean() - 1.5) < 0.1 and abs(u.var() - 49 / 12) < 0.2
    assert np.array_equal(u, Rng(3).uniform((20000,), -2.0, 5.0))
    assert not np.array_equal(u, Rng(4).uniform((20000,), -2.0, 5.0))


def test_normal_has_mean_zero_and_variance_one():
    z = Rng(5).normal((40000,))
    assert abs(z.mean()) < 0.03 and abs(z.var() - 1.0) < 0.05 and np.isfinite(z).all()
