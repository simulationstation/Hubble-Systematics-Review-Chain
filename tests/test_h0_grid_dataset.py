from __future__ import annotations

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.probes.h0_grid import H0GridPosteriorDataset


def test_h0_grid_dataset_linear_space_build_design_runs() -> None:
    H0_grid = np.linspace(60.0, 80.0, 401)
    mean = 70.0
    sigma = 2.0
    posterior = np.exp(-0.5 * ((H0_grid - mean) / sigma) ** 2)
    ds = H0GridPosteriorDataset(label="gauss", H0_grid=H0_grid, posterior=posterior, space="linear")

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level="L0", cfg={"shared_scale": {"enable": True, "params": ["delta_lnH0"]}})
    assert y.shape == (1,)
    assert y0.shape == (1,)
    assert cov.shape == (1,)
    assert X.shape == (1, 1)
    assert prior.param_names == ["delta_lnH0"]
    assert float(y[0]) == float(np.trapezoid((posterior / np.trapezoid(posterior, H0_grid)) * H0_grid, H0_grid))


def test_h0_grid_dataset_ln_space_build_design_runs() -> None:
    H0_grid = np.exp(np.linspace(np.log(50.0), np.log(100.0), 801))
    mu = np.log(70.0)
    sigma_ln = 0.03
    posterior = np.exp(-0.5 * ((np.log(H0_grid) - mu) / sigma_ln) ** 2) / H0_grid
    ds = H0GridPosteriorDataset(label="lognorm", H0_grid=H0_grid, posterior=posterior, space="ln")

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level="L0", cfg={"shared_scale": {"enable": True, "params": ["delta_lnH0"]}})
    assert y.shape == (1,)
    assert y0.shape == (1,)
    assert cov.shape == (1,)
    assert X.shape == (1, 1)
    assert prior.param_names == ["delta_lnH0"]
    assert np.isfinite(y[0]) and np.isfinite(cov[0])

