from __future__ import annotations

from pathlib import Path

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.injection import run_injection_suite
from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


def test_load_and_design_shapes() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "raw_dat_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat"),
        "raw_cov_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov"),
        "include_calibrators": True,
        "include_hubble_flow": True,
        "z_column": "zHD",
        "z_hf_min": 0.023,
        "z_hf_max": 0.15,
        "tag": "pytest_cal+hf_zHD",
        "processed_dir": str(root / "data/processed/pantheon_plus_shoes_ladder"),
    }
    ds = load_pantheon_plus_shoes_ladder_dataset(cfg)
    assert ds.z.ndim == 1
    assert ds.cov.shape == (ds.z.size, ds.z.size)
    assert ds.is_calibrator.shape == (ds.z.size,)
    assert ds.is_hubble_flow.shape == (ds.z.size,)
    assert np.any(ds.is_calibrator)
    assert np.any(ds.is_hubble_flow)

    z_hf = ds.get_column("z_hf_max")
    assert np.all(np.isnan(z_hf[ds.is_calibrator]))
    assert np.all(np.isfinite(z_hf[ds.is_hubble_flow]))

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    model_cfg = {
        "shared_scale": {"enable": True, "params": ["delta_lnH0"]},
        "priors": {},
        "mechanisms": {
            "calibrator_offset": True,
            "hf_survey_offsets": True,
            "mwebv_linear": True,
            "host_mass_step": True,
            "pkmjd_linear": True,
            "pkmjd_bins": {"enable": True, "n_bins": 4},
        },
    }
    y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level="L1", cfg=model_cfg)
    assert y.shape == y0.shape == (ds.z.size,)
    assert cov.shape == (ds.z.size, ds.z.size)
    assert X.shape[0] == ds.z.size
    assert X.shape[1] == len(prior.param_names)
    assert "global_offset_mag" in prior.param_names
    assert "delta_lnH0" in prior.param_names
    assert "calibrator_offset_mag" in prior.param_names
    assert "mwebv_linear_mag" in prior.param_names
    assert "host_mass_step_mag" in prior.param_names
    assert "pkmjd_linear_mag" in prior.param_names
    assert any(n.startswith("pkmjd_bin_offset_") for n in prior.param_names)
    assert any(n.startswith("hf_survey_offset_") for n in prior.param_names)

    j = prior.param_names.index("delta_lnH0")
    assert np.allclose(X[ds.is_calibrator, j], 0.0)


def test_injection_suite_calibrator_offset_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "raw_dat_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat"),
        "raw_cov_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov"),
        "include_calibrators": True,
        "include_hubble_flow": True,
        "z_column": "zHD",
        "z_hf_min": 0.023,
        "z_hf_max": 0.15,
        "tag": "pytest_cal+hf_zHD",
        "processed_dir": str(root / "data/processed/pantheon_plus_shoes_ladder"),
    }
    ds = load_pantheon_plus_shoes_ladder_dataset(cfg)
    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)

    model_cfg = {
        "shared_scale": {"enable": True, "params": ["delta_lnH0"], "prior_sigma": 0.5},
        "priors": {"sigma_global_offset_mag": 10.0},
    }
    inj_cfg = {
        "mechanism": "calibrator_offset_mag",
        "amplitudes": [0.0],
        "n_mc": 3,
        "use_diagonal_errors": True,
        "param_of_interest": "delta_lnH0",
        "seed": 123,
    }
    rng = np.random.default_rng(123)
    res = run_injection_suite(dataset=ds, anchor=anchor, ladder_level="L1", model_cfg=model_cfg, inj_cfg=inj_cfg, rng=rng)
    assert res.mechanism == "calibrator_offset_mag"
    assert res.param_of_interest == "delta_lnH0"
    assert len(res.rows) == 1


def test_injection_suite_m_b_corr_err_linear_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "raw_dat_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat"),
        "raw_cov_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov"),
        "include_calibrators": True,
        "include_hubble_flow": True,
        "z_column": "zHD",
        "z_hf_min": 0.023,
        "z_hf_max": 0.15,
        "tag": "pytest_cal+hf_zHD",
        "processed_dir": str(root / "data/processed/pantheon_plus_shoes_ladder"),
    }
    ds = load_pantheon_plus_shoes_ladder_dataset(cfg)
    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)

    model_cfg = {
        "shared_scale": {"enable": True, "params": ["delta_lnH0"], "prior_sigma": 0.5},
        "priors": {"sigma_global_offset_mag": 10.0, "sigma_survey_offset_mag": 0.2},
    }
    inj_cfg = {
        "mechanism": "m_b_corr_err_linear_mag",
        "apply_to": "hf",
        "amplitudes": [0.0],
        "n_mc": 3,
        "use_diagonal_errors": True,
        "param_of_interest": "delta_lnH0",
        "seed": 123,
    }
    rng = np.random.default_rng(123)
    res = run_injection_suite(dataset=ds, anchor=anchor, ladder_level="L2", model_cfg=model_cfg, inj_cfg=inj_cfg, rng=rng)
    assert res.mechanism == "m_b_corr_err_linear_mag"
    assert res.param_of_interest == "delta_lnH0"
    assert len(res.rows) == 1
