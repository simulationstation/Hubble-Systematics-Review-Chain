from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hubble_systematics.audit.tasks import (
    run_baseline_sweep_task,
    run_correlated_cut_null,
    run_cut_scan,
    run_fit_baseline,
    run_group_split_null_task,
    run_hemisphere_scan_task,
    run_injections,
    run_prior_mc_task,
    run_predictive_score_task,
    run_split_null_task,
    run_split_fit_task,
    run_sbc_task,
)
from hubble_systematics.audit.reporting import write_report
from hubble_systematics.io import copy_config_and_env, ensure_dir


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    config: dict[str, Any]


def run_from_config_path(config_path: Path) -> Path:
    config_path = config_path.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())

    run_id = cfg.get("run", {}).get("run_id")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")

    out_base = Path(cfg.get("run", {}).get("out_base", "outputs")).expanduser()
    run_dir = ensure_dir(out_base / run_id)

    ctx = RunContext(run_dir=run_dir, config=cfg)
    copy_config_and_env(config_path=config_path, run_dir=run_dir)

    tasks = cfg.get("run", {}).get("tasks", ["baseline_fit", "cut_scan", "correlated_cut_null"])
    results: dict[str, Any] = {"run_id": run_id, "tasks": {}, "created_utc": datetime.now(timezone.utc).isoformat()}

    for task in tasks:
        if task == "baseline_fit":
            results["tasks"][task] = run_fit_baseline(ctx)
        elif task == "baseline_sweep":
            results["tasks"][task] = run_baseline_sweep_task(ctx)
        elif task == "cut_scan":
            results["tasks"][task] = run_cut_scan(ctx)
        elif task == "correlated_cut_null":
            results["tasks"][task] = run_correlated_cut_null(ctx)
        elif task == "injection_suite":
            results["tasks"][task] = run_injections(ctx)
        elif task == "hemisphere_scan":
            results["tasks"][task] = run_hemisphere_scan_task(ctx)
        elif task == "predictive_score":
            results["tasks"][task] = run_predictive_score_task(ctx)
        elif task == "prior_mc":
            results["tasks"][task] = run_prior_mc_task(ctx)
        elif task == "split_fit":
            results["tasks"][task] = run_split_fit_task(ctx)
        elif task == "split_null":
            results["tasks"][task] = run_split_null_task(ctx)
        elif task == "group_split_null":
            results["tasks"][task] = run_group_split_null_task(ctx)
        elif task == "sbc":
            results["tasks"][task] = run_sbc_task(ctx)
        elif task == "report":
            write_report(run_dir)
            results["tasks"][task] = {"ok": True}
        else:
            raise ValueError(f"Unknown task: {task}")

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    return run_dir
