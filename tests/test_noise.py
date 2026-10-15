import numpy as np

from voyage_sim import noise


def test_deterministic_and_order_independent():
    ids = np.arange(1000, dtype=np.int64)
    a = noise.normal(7, ids, step=3, channel=2)
    b = noise.normal(7, ids[::-1], step=3, channel=2)[::-1]
    assert np.array_equal(a, b)
    # A voyage's draw does not depend on which other voyages are in the batch.
    assert noise.normal(7, ids[500:501], 3, 2)[0] == a[500]


def test_streams_differ():
    ids = np.arange(1000)
    base = noise.uniform(1, ids, 0, 0)
    for other in (noise.uniform(2, ids, 0, 0), noise.uniform(1, ids, 1, 0), noise.uniform(1, ids, 0, 1)):
        assert not np.allclose(base, other)
        assert abs(np.corrcoef(base, other)[0, 1]) < 0.1


def test_distribution_moments():
    ids = np.arange(200_000)
    u = noise.uniform(11, ids, 5, 9)
    assert u.min() > 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005
    z = noise.normal(11, ids, 5, 9)
    assert abs(z.mean()) < 0.01
    assert abs(z.std() - 1.0) < 0.01
