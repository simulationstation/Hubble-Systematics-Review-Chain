from __future__ import annotations

from pathlib import Path

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.probes.pantheon_plus import load_pantheon_plus_dataset


def test_load_and_design_shapes() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "data_path": str(root / "data/processed/pantheon_plus/pantheon_plus_sky_cosmology_stat+sys_zHD.npz"),
        "z_min": 0.02,
        "z_max": 0.62,
    }
    ds = load_pantheon_plus_dataset(cfg)
    assert ds.z.ndim == 1
    assert ds.cov.shape == (ds.z.size, ds.z.size)

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level="L1", cfg={"priors": {}})
    assert y.shape == y0.shape == (ds.z.size,)
    assert cov.shape == (ds.z.size, ds.z.size)
    assert X.shape[0] == ds.z.size
    assert X.shape[1] == len(prior.param_names)
    assert "global_offset_mag" in prior.param_names

    y2, y02, cov2, X2, prior2 = ds.build_design(anchor=anchor, ladder_level="L2", cfg={"priors": {}})
    assert y2.shape == (ds.z.size,)
    assert X2.shape[0] == ds.z.size
    assert X2.shape[1] == len(prior2.param_names)
    assert prior2.param_names[0] == "global_offset_mag"

