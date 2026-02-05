from __future__ import annotations

import json

from hubble_systematics.audit.runner import RunContext
from hubble_systematics.audit.tasks import run_baseline_sweep_task


def test_baseline_sweep_supports_stack(tmp_path) -> None:
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
        "sweep": [
            {"label": "base", "model": {}},
            {"label": "tight_prior", "model": {"shared_scale": {"prior_sigma": 0.1}}},
        ],
    }

    ctx = RunContext(run_dir=tmp_path, config=cfg)
    out = run_baseline_sweep_task(ctx)
    assert out["base_label"] == "base"
    assert len(out["rows"]) == 2
    assert (tmp_path / "baseline_sweep.json").exists()
    saved = json.loads((tmp_path / "baseline_sweep.json").read_text())
    assert saved["base_label"] == "base"
