from __future__ import annotations

import json

from hubble_systematics.audit.runner import RunContext
from hubble_systematics.audit.tasks import run_predictive_score_task


def test_predictive_score_supports_stack(tmp_path) -> None:
    cfg = {
        "anchor": {"H0": 67.4, "Omega_m": 0.315, "Omega_k": 0.0, "rd_Mpc": 147.09},
        "model": {"shared_scale": {"enable": True, "params": ["delta_lnH0"], "prior_sigma": 0.5}},
        "probe": {
            "name": "stack",
            "stack": [
                {
                    "name": "gaussian_measurement",
                    "label": "g1",
                    "quantity": "H0",
                    "mean": 67.4,
                    "sigma": 1.0,
                    "space": "ln",
                    "model": {"ladder_level": "L0"},
                },
                {
                    "name": "gaussian_measurement",
                    "label": "g2",
                    "quantity": "H0",
                    "mean": 70.0,
                    "sigma": 1.0,
                    "space": "ln",
                    "model": {"ladder_level": "L0"},
                },
            ],
        },
        "predictive_score": {
            "seed": 123,
            "mode": "random",
            "n_rep": 10,
            "train_frac": 0.6,
            "always_include_calibrators": False,
            "always_include_hubble_flow": False,
            "use_diagonal_errors": True,
            "models": [
                {"label": "base", "model": {}},
                {"label": "tight", "model": {"shared_scale": {"prior_sigma": 0.1}}},
            ],
        },
    }

    ctx = RunContext(run_dir=tmp_path, config=cfg)
    out = run_predictive_score_task(ctx)
    assert out["mode"] == "random"
    assert int(out["n_rep"]) == 10
    assert (tmp_path / "predictive_score.json").exists()
    saved = json.loads((tmp_path / "predictive_score.json").read_text())
    assert saved["mode"] == "random"

