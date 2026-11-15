"""Eigener Zufallsgenerator (SplitMix64), vektorisiert: gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen.

Herkunft: Algorithmus SplitMix64 (Steele, Lea, Flood 2014), Schreibweise nach restaurant-tischmix-demo/rtm_rng.py (kopiert und auf numpy-uint64-Felder
umgestellt, nicht importiert). Die n-te Ausgabe hängt nur vom Zustand seed + n·γ ab; deshalb lassen sich viele Ausgaben auf einmal berechnen
und stimmen mit der schrittweisen Fassung überein (tests/test_rng.py vergleicht beide). Python-`hash()` und `random` sind für feste
Zufalls-Seeds nicht zugesichert und werden nirgends benutzt."""

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)


class Rng:
    def __init__(self, seed: int):
        self.state = int(seed) & MASK

    def next_u64(self, n: int) -> np.ndarray:
        """Die nächsten n Ausgaben als uint64-Feld (Zustand rückt um n Schritte vor)."""
        n = int(n)
        idx = np.arange(1, n + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(GAMMA)
            z = (z ^ (z >> _S30)) * _M1
            z = (z ^ (z >> _S27)) * _M2
            z = z ^ (z >> _S31)
        self.state = (self.state + n * GAMMA) & MASK
        return z

    def random(self, shape) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        n = int(np.prod(shape))
        return ((self.next_u64(n) >> _S11).astype(np.float64) / float(1 << 53)).reshape(shape)

    def uniform(self, shape, lo: float, hi: float) -> np.ndarray:
        return lo + (hi - lo) * self.random(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormalverteilt (Box-Muller)."""
        u1 = np.maximum(self.random(shape), 1e-12)
        u2 = self.random(shape)
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
