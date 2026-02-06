#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import re

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.stack_predictive_dataset import StackPredictiveDataset, StackPredictivePart
from hubble_systematics.probes.bao import load_bao_dataset
from hubble_systematics.probes.chronometers import load_chronometers_dataset
from hubble_systematics.probes.gaussian_measurement import load_gaussian_measurement_dataset
from hubble_systematics.probes.h0_grid import load_h0_grid_posterior_dataset
from hubble_systematics.probes.pantheon_plus import load_pantheon_plus_dataset
from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset
from hubble_systematics.probes.siren_gate2_grid import load_siren_gate2_grid_dataset


def _build_anchor(cfg: dict[str, Any]) -> AnchorLCDM:
    a = cfg.get("anchor", {}) or {}
    return AnchorLCDM(
        H0=float(a.get("H0", 67.4)),
        Omega_m=float(a.get("Omega_m", 0.315)),
        Omega_k=float(a.get("Omega_k", 0.0)),
        rd_Mpc=float(a.get("rd_Mpc", 147.09)),
    )


def _load_probe(probe_cfg: dict[str, Any]):
    name = (probe_cfg.get("name") or "").strip()
    if name == "pantheon_plus":
        return load_pantheon_plus_dataset(probe_cfg)
    if name == "pantheon_plus_shoes_ladder":
        return load_pantheon_plus_shoes_ladder_dataset(probe_cfg)
    if name == "bao":
        return load_bao_dataset(probe_cfg)
    if name == "chronometers":
        return load_chronometers_dataset(probe_cfg)
    if name == "h0_grid":
        return load_h0_grid_posterior_dataset(probe_cfg)
    if name == "gaussian_measurement":
        return load_gaussian_measurement_dataset(probe_cfg)
    if name == "siren_gate2_grid":
        return load_siren_gate2_grid_dataset(probe_cfg)
    raise ValueError(f"Unknown probe name: {name!r}")

def _stack_part_label(item: dict[str, Any]) -> str:
    if item.get("part_label") is not None:
        return str(item["part_label"])
    name = str(item.get("name") or "").strip()
    label = item.get("label")
    if label is not None:
        lab = str(label)
        if name == "h0_grid":
            return f"h0_grid:{lab}"
        if name == "gaussian_measurement":
            return f"gaussian:{lab}"
        if name == "siren_gate2_grid":
            return f"siren_gate2:{lab}"
        return lab
    return name


def _stack_predictive_dataset_from_config(*, cfg: dict[str, Any], anchor: AnchorLCDM) -> tuple[StackPredictiveDataset, np.ndarray | None]:
    probe = cfg.get("probe", {}) or {}
    items = probe.get("stack", []) or []
    if not items:
        raise ValueError("stack probe requires non-empty probe.stack")

    base_model = cfg.get("model", {}) or {}
    base_level = str(base_model.get("ladder_level", "L1"))
    pred_cfg = cfg.get("predictive_score", {}) or {}
    scope_part = pred_cfg.get("stack_scope_part")

    parts: list[StackPredictivePart] = []
    for item in items:
        item = dict(item)
        name = item.get("name")
        if name is None:
            raise ValueError("Each stack item must include a name")
        part_label = _stack_part_label(item)
        ds = _load_probe(item)
        parts.append(StackPredictivePart(name=str(part_label), dataset=ds, base_model=item.get("model", {}) or {}))

    ds = StackPredictiveDataset.from_parts(
        parts=parts,
        anchor=anchor,
        base_level=base_level,
        base_model_cfg=base_model,
        scope_part=str(scope_part) if scope_part is not None else None,
    )

    fixed_train_mask = None
    if scope_part is not None:
        scope_mask = ds.part_mask_full(str(scope_part))
        fixed_train_mask = ~scope_mask
    return ds, fixed_train_mask


def _group_holdout_labels(
    *,
    dataset,
    group_var: str,
    min_group_n: int,
    group_allowlist: list[Any] | None,
    group_prefix: str | None,
    group_regex: str | None,
    always_include_calibrators: bool,
    always_include_hubble_flow: bool,
    fixed_train_mask: np.ndarray | None,
) -> tuple[list[Any], np.ndarray]:
    v = dataset.get_column(group_var)
    v = np.asarray(v)
    n = int(v.size)

    fixed_train = np.zeros(n, dtype=bool)
    if fixed_train_mask is not None:
        fixed_train_mask = np.asarray(fixed_train_mask, dtype=bool)
        if fixed_train_mask.shape != (n,):
            raise ValueError("fixed_train_mask shape mismatch")
        fixed_train |= fixed_train_mask
    if always_include_calibrators and hasattr(dataset, "is_calibrator"):
        cal = np.asarray(getattr(dataset, "is_calibrator"), dtype=bool)
        if cal.shape == (n,):
            fixed_train |= cal
    if always_include_hubble_flow and hasattr(dataset, "is_hubble_flow"):
        hf = np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool)
        if hf.shape == (n,):
            fixed_train |= hf

    if np.issubdtype(v.dtype, np.number):
        good = np.isfinite(v)
    else:
        good = np.ones(n, dtype=bool)
        if v.dtype.kind in {"U", "S"}:
            good &= v != ""
        else:
            good &= np.asarray([x is not None for x in v], dtype=bool)

    groups = np.unique(v[good])

    allow = None
    if group_allowlist:
        if not isinstance(group_allowlist, list):
            raise ValueError("group_allowlist must be a list")
        allow = {str(x) for x in group_allowlist}
    prefix = None if group_prefix is None else str(group_prefix)
    rx = None
    if group_regex is not None:
        rx = re.compile(str(group_regex))

    labels: list[Any] = []
    test_counts = []
    for g in groups:
        sg = str(g)
        if allow is not None and sg not in allow:
            continue
        if prefix and not sg.startswith(prefix):
            continue
        if rx is not None and rx.search(sg) is None:
            continue
        test = (v == g) & good & (~fixed_train)
        if int(np.sum(test)) < int(min_group_n):
            continue
        labels.append(g)
        test_counts.append(int(np.sum(test)))
    return labels, np.asarray(test_counts, dtype=int)


def _format_group(g: Any) -> str:
    if isinstance(g, (np.floating, float, int, np.integer)):
        try:
            if float(g).is_integer():
                return str(int(g))
        except Exception:
            pass
        return str(float(g))
    return str(g)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank which held-out groups drive predictive-score gains.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Output run directory containing config.yaml + predictive_score.json")
    ap.add_argument("--top-k", type=int, default=10, help="Top-k groups to list per model")
    ap.add_argument("--out-md", type=Path, default=None, help="Write markdown report to this path (default: run_dir/driver_ranking.md)")
    ap.add_argument("--out-json", type=Path, default=None, help="Write JSON payload to this path (default: run_dir/driver_ranking.json)")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    pred = json.loads((run_dir / "predictive_score.json").read_text())

    pred_cfg = cfg.get("predictive_score", {}) or {}
    mode = str(pred_cfg.get("mode", "random")).lower()
    if mode != "group_holdout":
        raise ValueError(f"This report currently supports only predictive_score.mode=group_holdout; got {mode!r}")

    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = str(probe_cfg.get("name") or "")
    fixed_train_mask = None
    if probe_name == "stack":
        dataset, fixed_train_mask = _stack_predictive_dataset_from_config(cfg=cfg, anchor=anchor)
    else:
        dataset = _load_probe(probe_cfg)

    group_var = str(pred_cfg.get("group_var", "idsurvey"))
    min_group_n = int(pred_cfg.get("min_group_n", 20))
    group_allowlist = pred_cfg.get("group_allowlist")
    group_prefix = pred_cfg.get("group_prefix")
    group_regex = pred_cfg.get("group_regex")
    always_include_calibrators = bool(pred_cfg.get("always_include_calibrators", False))
    always_include_hubble_flow = bool(pred_cfg.get("always_include_hubble_flow", False))

    labels, test_counts = _group_holdout_labels(
        dataset=dataset,
        group_var=group_var,
        min_group_n=min_group_n,
        group_allowlist=group_allowlist,
        group_prefix=group_prefix,
        group_regex=group_regex,
        always_include_calibrators=always_include_calibrators,
        always_include_hubble_flow=always_include_hubble_flow,
        fixed_train_mask=fixed_train_mask,
    )

    # Determine baseline label from config ordering.
    model_list = pred_cfg.get("models") or []
    if not isinstance(model_list, list) or not model_list:
        raise ValueError("predictive_score.models missing or empty in config")
    base_label = str(model_list[0].get("label") or "baseline")

    models: dict[str, dict[str, Any]] = pred.get("models", {})
    if base_label not in models:
        raise ValueError(f"Baseline label {base_label!r} not found in predictive_score.json models={list(models.keys())}")

    # Validate split alignment.
    n_splits = len(labels)
    base_scores = np.asarray(models[base_label]["logp"], dtype=float)
    if base_scores.size != n_splits:
        raise ValueError(f"Split mismatch: reconstructed n_splits={n_splits}, baseline logp has {base_scores.size}")

    rows = []
    for label, obj in models.items():
        scores = np.asarray(obj["logp"], dtype=float)
        if scores.size != n_splits:
            raise ValueError(f"Split mismatch for model {label!r}: expected {n_splits}, got {scores.size}")
        d = scores - base_scores
        rows.append(
            {
                "label": label,
                "mean_logp": float(obj.get("mean_logp")),
                "mean_delta_logp": float(np.mean(d)),
                "win_rate": float(np.mean(d > 0.0)),
                "sd_delta_logp": float(np.std(d)),
                "delta_logp": d,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["mean_delta_logp"], reverse=True)

    top_k = int(args.top_k)
    md_lines = []
    md_lines.append("# Predictive-score driver ranking\n")
    md_lines.append(f"- Run dir: `{run_dir}`\n")
    md_lines.append(f"- Mode: `{mode}` (group_var=`{group_var}`, n_groups={n_splits}, min_group_n={min_group_n})\n")
    md_lines.append(f"- Baseline label: `{base_label}`\n")
    md_lines.append("\n## Model ranking\n\n")
    md_lines.append("| model | mean Δlogp vs base | win-rate | sd(Δlogp) |\n")
    md_lines.append("|---|---:|---:|---:|\n")
    for r in rows_sorted:
        md_lines.append(f"| `{r['label']}` | {r['mean_delta_logp']:+.3f} | {100.0*r['win_rate']:.1f}% | {r['sd_delta_logp']:.3f} |\n")

    md_lines.append("\n## Top group drivers (per model)\n\n")
    md_lines.append(f"Groups are ordered as in the underlying split generator (sorted unique `{group_var}` values after filters).\n\n")

    for r in rows_sorted:
        d = np.asarray(r["delta_logp"], dtype=float)
        order_hi = np.argsort(-d)[:top_k]
        order_lo = np.argsort(d)[:top_k]
        md_lines.append(f"### `{r['label']}`\n\n")
        md_lines.append("Top gains:\n\n")
        md_lines.append("| group | n_test | Δlogp |\n")
        md_lines.append("|---|---:|---:|\n")
        for j in order_hi.tolist():
            md_lines.append(f"| `{_format_group(labels[j])}` | {int(test_counts[j])} | {float(d[j]):+.3f} |\n")
        md_lines.append("\nTop losses:\n\n")
        md_lines.append("| group | n_test | Δlogp |\n")
        md_lines.append("|---|---:|---:|\n")
        for j in order_lo.tolist():
            md_lines.append(f"| `{_format_group(labels[j])}` | {int(test_counts[j])} | {float(d[j]):+.3f} |\n")
        md_lines.append("\n")

    out_md = (run_dir / "driver_ranking.md") if args.out_md is None else args.out_md.expanduser().resolve()
    out_md.write_text("".join(md_lines))

    out_json = (run_dir / "driver_ranking.json") if args.out_json is None else args.out_json.expanduser().resolve()
    out_payload = {
        "run_dir": str(run_dir),
        "mode": mode,
        "group_var": group_var,
        "base_label": base_label,
        "group_labels": [_format_group(x) for x in labels],
        "test_counts": test_counts.tolist(),
        "models": {r["label"]: {"mean_delta_logp": r["mean_delta_logp"], "win_rate": r["win_rate"], "delta_logp": r["delta_logp"].tolist()} for r in rows},
    }
    out_json.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
