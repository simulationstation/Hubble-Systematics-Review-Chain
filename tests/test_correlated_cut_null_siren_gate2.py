from __future__ import annotations

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.correlated_cut_null import run_correlated_cut_null_siren_gate2_event_gaussian_mc
from hubble_systematics.probes.siren_gate2_grid import load_siren_gate2_grid_dataset


def test_correlated_cut_null_siren_gate2_event_gaussian_runs() -> None:
    ds = load_siren_gate2_grid_dataset(
        {
            "data_path": "data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json",
            "label": "gate2",
            "space": "ln",
        }
    )
    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    cuts = np.linspace(250.0, 4000.0, 6)
    rng = np.random.default_rng(0)
    res = run_correlated_cut_null_siren_gate2_event_gaussian_mc(
        dataset=ds,
        anchor=anchor,
        cut_var="ess_min",
        cut_mode="geq",
        cuts=cuts,
        n_mc=20,
        rng=rng,
    )
    assert res.cut_var == "ess_min"
    assert res.cut_mode == "geq"
    assert res.param_name == "delta_lnH0"
    assert set(res.p_values) == {"end_to_end", "max_pair", "path_length"}
    for k, v in res.p_values.items():
        assert 0.0 <= float(v) <= 1.0, k

