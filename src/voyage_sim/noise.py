"""Counter-based random numbers keyed by (seed, voyage id, step, channel).

Every random draw in a voyage is a pure function of those four integers, so a
voyage's track does not depend on which other voyages share its batch, the batch
order, or how a grid is split across processes. That is what lets every scenario
grid rerun from its config with identical results.

The generator is SplitMix64 applied as a hash; its output passes BigCrush and is
far better than the statistics this model needs.
"""

from __future__ import annotations

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO_POW_53 = float(2**53)


def _mix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def _hash(seed, ids, step, channel) -> np.ndarray:
    ids = np.atleast_1d(np.asarray(ids)).astype(np.uint64)
    h = _mix(np.full(ids.shape, np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64))
    h = _mix(h ^ ids)
    h = _mix(h ^ np.uint64(int(step) & 0xFFFFFFFFFFFFFFFF))
    return _mix(h ^ np.uint64(int(channel) & 0xFFFFFFFFFFFFFFFF))


def uniform(seed: int, ids, step: int, channel: int) -> np.ndarray:
    """Uniform draws in the open interval (0, 1), one per id."""
    h = _hash(seed, ids, step, channel)
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / _TWO_POW_53


def normal(seed: int, ids, step: int, channel: int) -> np.ndarray:
    """Standard normal draws, one per id (Box-Muller on two derived channels)."""
    u1 = uniform(seed, ids, step, 2 * int(channel))
    u2 = uniform(seed, ids, step, 2 * int(channel) + 1)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
