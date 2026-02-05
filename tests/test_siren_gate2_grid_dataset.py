from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.probes.siren_gate2_grid import load_siren_gate2_grid_dataset


def test_siren_gate2_grid_matches_shipped_posterior() -> None:
    path = Path("data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json")
    d = json.loads(path.read_text())

    ds = load_siren_gate2_grid_dataset({"data_path": str(path), "label": "gate2_test", "space": "ln"})
    assert len(ds) == int(d["n_events"])
    mjd = ds.get_column("event_mjd")
    assert int(np.sum(np.isfinite(mjd))) == len(ds)
    assert float(np.nanmin(mjd)) > 50000.0

    H0 = np.asarray(d["H0_grid"], dtype=float)
    p_ref = np.asarray(d["posterior"], dtype=float)
    p_ref = p_ref / float(np.trapezoid(p_ref, H0))
    p = ds.posterior()
    l1 = float(np.trapezoid(np.abs(p - p_ref), H0))
    assert l1 < 1e-12

    # Check the selection-corrected identity (up to an additive constant).
    logpost_ref = np.asarray(d["logL_H0_rel"], dtype=float)
    logpost_ref = logpost_ref - float(np.max(logpost_ref))
    logpost = ds.log_posterior_rel()
    logpost = logpost - float(np.max(logpost))
    assert float(np.max(np.abs(logpost - logpost_ref))) < 1e-12


def test_siren_gate2_grid_build_design_is_consistent() -> None:
    path = Path("data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json")
    ds = load_siren_gate2_grid_dataset({"data_path": str(path), "label": "gate2_test", "space": "ln"})

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    y, y0, cov, X, prior = ds.build_design(
        anchor=anchor,
        ladder_level="L0",
        cfg={"shared_scale": {"enable": True, "params": ["delta_lnH0"]}},
    )

    n = len(ds)
    assert y.shape == (n,)
    assert y0.shape == (n,)
    assert cov.shape == (n,)
    assert X.shape == (n, 1)
    assert prior.param_names == ["delta_lnH0"]

    # Fit and compare to posterior moments.
    fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior))
    mu_ln, sd_ln = ds.lnH0_moments()
    delta_expected = float(mu_ln - np.log(anchor.H0))
    assert fit.param_names == ["delta_lnH0"]
    assert abs(float(fit.mean[0]) - delta_expected) < 1e-12
    assert abs(float(fit.cov[0, 0]) - float(sd_ln**2)) < 1e-12
