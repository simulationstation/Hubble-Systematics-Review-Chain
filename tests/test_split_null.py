import numpy as np

from hubble_systematics.audit.split_null import _permute_within_groups


def test_permute_within_groups_keeps_nans_fixed() -> None:
    rng = np.random.default_rng(0)
    v = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    out = _permute_within_groups(v, groups=None, rng=rng)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert set(out[2:]) == {1.0, 2.0, 3.0, 4.0}


def test_permute_within_groups_respects_groups_and_nans() -> None:
    rng = np.random.default_rng(1)
    v = np.array([np.nan, 1.0, 2.0, 10.0, 11.0, np.nan])
    g = np.array([0, 0, 0, 1, 1, 1])
    out = _permute_within_groups(v, groups=g, rng=rng)
    assert np.isnan(out[0]) and np.isnan(out[5])
    # Group 0 finite values stay within {1,2}; group 1 finite values stay within {10,11}.
    assert set(out[1:3]) == {1.0, 2.0}
    assert set(out[3:5]) == {10.0, 11.0}

