from __future__ import annotations

import numpy as np

from hubble_systematics.cosmology import LCDM


def test_distance_monotone() -> None:
    cosmo = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.0)
    z = np.array([0.1, 0.3, 0.7])
    dc = cosmo.comoving_distance(z)
    assert np.all(np.diff(dc) > 0)


def test_dm_equals_dc_for_flat() -> None:
    cosmo = LCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0)
    z = np.array([0.2, 0.5])
    dc = cosmo.comoving_distance(z)
    dm = cosmo.transverse_comoving_distance(z)
    assert np.allclose(dm, dc, rtol=1e-10, atol=0.0)


def test_dv_positive() -> None:
    cosmo = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.0)
    z = np.array([0.2, 0.5])
    dv = cosmo.DV(z)
    assert np.all(np.isfinite(dv))
    assert np.all(dv > 0)

