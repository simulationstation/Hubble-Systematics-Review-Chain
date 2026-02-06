#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.predictive_score import run_predictive_score
from hubble_systematics.audit.stack_predictive_dataset import StackPredictiveDataset, StackPredictivePart
from hubble_systematics.probes.bao import load_bao_dataset
from hubble_systematics.probes.chronometers import load_chronometers_dataset
from hubble_systematics.probes.gaussian_measurement import load_gaussian_measurement_dataset
from hubble_systematics.probes.h0_grid import load_h0_grid_posterior_dataset
from hubble_systematics.probes.pantheon_plus import load_pantheon_plus_dataset
from hubble_systematics.probes.pantheon_plus_shoes_ladder import PantheonPlusShoesLadderDataset, load_pantheon_plus_shoes_ladder_dataset
from hubble_systematics.probes.siren_gate2_grid import load_siren_gate2_grid_dataset


ApplyTo = Literal["cal", "hf", "all"]
PermuteMode = Literal["global", "within_survey"]

# Globals for multiprocessing (fork start method).
_G_PARTS: list[StackPredictivePart] | None = None
_G_DS0: StackPredictiveDataset | None = None
_G_ANCHOR: AnchorLCDM | None = None
_G_BASE_LEVEL: str | None = None
_G_BASE_MODEL_CFG: dict[str, Any] | None = None
_G_PRED_CFG: dict[str, Any] | None = None
_G_TEST_LABEL: str | None = None
_G_LADDER_IDX: int | None = None
_G_LADDER_DS: PantheonPlusShoesLadderDataset | None = None
_G_PERMUTE_COLUMN: str | None = None
_G_APPLY_TO: ApplyTo | None = None
_G_PERMUTE_MODE: PermuteMode | None = None
_G_FIXED_TRAIN_MASK: np.ndarray | None = None


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _finite_or_none(x: float) -> float | None:
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if np.isfinite(xf) else None


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


def _stack_parts_from_config(*, cfg: dict[str, Any]) -> list[StackPredictivePart]:
    probe = cfg.get("probe", {}) or {}
    items = probe.get("stack", []) or []
    if not items:
        raise ValueError("Expected a stack probe with non-empty probe.stack")
    parts: list[StackPredictivePart] = []
    for item in items:
        item = dict(item)
        name = item.get("name")
        if name is None:
            raise ValueError("Each stack item must include a name")
        part_label = _stack_part_label(item)
        ds = _load_probe(item)
        parts.append(StackPredictivePart(name=str(part_label), dataset=ds, base_model=item.get("model", {}) or {}))
    return parts


def _fixed_train_mask_for_stack(ds: StackPredictiveDataset, *, scope_part: str | None) -> np.ndarray | None:
    if scope_part is None:
        return None
    scope_mask = ds.part_mask_full(str(scope_part))
    if int(np.sum(scope_mask)) < 2:
        raise ValueError("scope_part must match a stack part with at least 2 rows")
    return ~scope_mask


def _get_pred_model_entry(pred_cfg: dict[str, Any], label: str) -> dict[str, Any]:
    model_list = pred_cfg.get("models") or []
    if not isinstance(model_list, list) or not model_list:
        raise ValueError("predictive_score.models must be a non-empty list")
    for item in model_list:
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "") == str(label):
            return dict(item)
    raise KeyError(f"Model label not found in predictive_score.models: {label!r}")


def _select_mask(*, ds: PantheonPlusShoesLadderDataset, apply_to: ApplyTo) -> np.ndarray:
    if apply_to == "cal":
        return np.asarray(ds.is_calibrator, dtype=bool).reshape(-1)
    if apply_to == "hf":
        return np.asarray(ds.is_hubble_flow, dtype=bool).reshape(-1)
    if apply_to == "all":
        return np.ones_like(np.asarray(ds.is_calibrator, dtype=bool).reshape(-1))
    raise ValueError(f"Unsupported apply_to: {apply_to!r}")


def _permute_values(
    *,
    x: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    mode: PermuteMode,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(x).copy()
    ids = np.asarray(ids).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if x.shape != ids.shape or x.shape != mask.shape:
        raise ValueError("shape mismatch in permute_values")

    good = mask & np.isfinite(x)
    if mode == "global":
        idx = np.where(good)[0]
        if idx.size < 2:
            return x
        x[idx] = x[rng.permutation(idx)]
        return x

    if mode == "within_survey":
        idx_all = np.where(good)[0]
        if idx_all.size < 2:
            return x
        for sid in np.unique(ids[idx_all]):
            m = good & (ids == sid)
            idx = np.where(m)[0]
            if idx.size < 2:
                continue
            x[idx] = x[rng.permutation(idx)]
        return x

    raise ValueError(f"Unsupported permute mode: {mode!r}")


def _make_permuted_ladder_dataset(
    *,
    ds: PantheonPlusShoesLadderDataset,
    column: str,
    apply_to: ApplyTo,
    mode: PermuteMode,
    rng: np.random.Generator,
) -> PantheonPlusShoesLadderDataset:
    column = str(column)
    mask = _select_mask(ds=ds, apply_to=apply_to)
    ids = np.asarray(ds.idsurvey, dtype=int)
    if column == "pkmjd_err":
        x_new = _permute_values(x=np.asarray(ds.pkmjd_err, dtype=float), ids=ids, mask=mask, mode=mode, rng=rng)
        return dataclasses.replace(ds, pkmjd_err=x_new)
    if column == "host_logmass":
        x_new = _permute_values(x=np.asarray(ds.host_logmass, dtype=float), ids=ids, mask=mask, mode=mode, rng=rng)
        return dataclasses.replace(ds, host_logmass=x_new)
    raise ValueError(f"Unsupported column to permute: {column!r}")


def _one_permutation(perm_seed: int) -> float:
    parts = _G_PARTS
    ds0 = _G_DS0
    anchor = _G_ANCHOR
    base_level = _G_BASE_LEVEL
    base_model_cfg = _G_BASE_MODEL_CFG
    pred_cfg = _G_PRED_CFG
    test_label = _G_TEST_LABEL
    ladder_part_idx = _G_LADDER_IDX
    ladder_ds = _G_LADDER_DS
    permute_column = _G_PERMUTE_COLUMN
    apply_to = _G_APPLY_TO
    mode = _G_PERMUTE_MODE
    fixed_train_mask = _G_FIXED_TRAIN_MASK

    if (
        parts is None
        or ds0 is None
        or anchor is None
        or base_level is None
        or base_model_cfg is None
        or pred_cfg is None
        or test_label is None
        or ladder_part_idx is None
        or ladder_ds is None
        or permute_column is None
        or apply_to is None
        or mode is None
    ):
        raise RuntimeError("Permutation globals not initialized")

    rng = np.random.default_rng(int(perm_seed))
    ds_perm_ladder = _make_permuted_ladder_dataset(
        ds=ladder_ds,
        column=str(permute_column),
        apply_to=apply_to,
        mode=mode,
        rng=rng,
    )
    parts_perm = list(parts)
    parts_perm[ladder_part_idx] = StackPredictivePart(
        name=parts[ladder_part_idx].name,
        dataset=ds_perm_ladder,
        base_model=parts[ladder_part_idx].base_model,
    )

    ds_perm = dataclasses.replace(ds0, parts=parts_perm)
    res = run_predictive_score(
        dataset=ds_perm,
        anchor=anchor,
        base_level=base_level,
        base_model_cfg=base_model_cfg,
        pred_cfg=pred_cfg,
        rng=rng,
        fixed_train_mask=fixed_train_mask,
    )
    test_obj = res.models[test_label]
    return float(test_obj["mean_delta_logp_vs_base"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation-null test for metadata-proxy predictive-score gains.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory with config.yaml and predictive_score.json (stack probe).")
    ap.add_argument("--test-model", required=True, help="Model label to test (must exist in config predictive_score.models).")
    ap.add_argument("--permute-column", choices=["pkmjd_err", "host_logmass"], required=True, help="Which ladder column to permute.")
    ap.add_argument("--apply-to", choices=["cal", "hf", "all"], default="cal", help="Which rows to permute within (default: cal).")
    ap.add_argument("--mode", choices=["global", "within_survey"], default="within_survey", help="Permutation mode (default: within_survey).")
    ap.add_argument("--n-perm", type=int, default=300, help="Number of permutations.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument("--n-proc", type=int, default=1, help="Parallel worker processes (default: 1).")
    ap.add_argument("--chunksize", type=int, default=5, help="Multiprocessing map chunksize (default: 5).")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N permutations (0 disables).",
    )
    ap.add_argument(
        "--progress-secs",
        type=float,
        default=30.0,
        help="Also write progress at least every N seconds (0 disables).",
    )
    ap.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Optional progress JSON path (default: run_dir/permutation_progress_<model>.json).",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="Output JSON path (default: run_dir/permutation_null_<model>.json)")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    pred_obs = json.loads((run_dir / "predictive_score.json").read_text())

    anchor = _build_anchor(cfg)
    base_model_cfg = cfg.get("model", {}) or {}
    base_level = str(base_model_cfg.get("ladder_level", "L1"))
    pred_cfg_full = cfg.get("predictive_score", {}) or {}
    pred_mode = str(pred_cfg_full.get("mode", "random")).lower()

    base_label = str((pred_cfg_full.get("models") or [])[0].get("label") or "baseline")
    test_label = str(args.test_model)

    # Observed mean Δlogp from the finished run (for the exact split family).
    obs_models = pred_obs.get("models", {}) or {}
    if base_label not in obs_models or test_label not in obs_models:
        raise ValueError(f"predictive_score.json missing base/test labels: base={base_label!r}, test={test_label!r}")
    obs_base = np.asarray(obs_models[base_label]["logp"], dtype=float)
    obs_test = np.asarray(obs_models[test_label]["logp"], dtype=float)
    if obs_base.shape != obs_test.shape:
        raise ValueError("Observed base/test logp arrays shape mismatch")
    obs_delta = obs_test - obs_base
    obs_mean = float(np.mean(obs_delta))

    # Permutation null requires deterministic splits (group_holdout). If you need random-split
    # permutation nulls, extend predictive scoring to accept precomputed splits.
    if pred_mode != "group_holdout":
        raise ValueError(f"Permutation null currently supports only predictive_score.mode=group_holdout (got {pred_mode!r})")

    # Build baseline stack dataset (we will swap out only the ladder part). This is the expensive
    # precomputation step; for permutations we reuse all bookkeeping (diag sigma, row slices).
    parts = _stack_parts_from_config(cfg=cfg)
    scope_part = pred_cfg_full.get("stack_scope_part")
    ds0 = StackPredictiveDataset.from_parts(
        parts=parts,
        anchor=anchor,
        base_level=base_level,
        base_model_cfg=base_model_cfg,
        scope_part=str(scope_part) if scope_part is not None else None,
    )
    fixed_train_mask = _fixed_train_mask_for_stack(ds0, scope_part=str(scope_part) if scope_part is not None else None)

    # Find the ladder part dataset to permute.
    ladder_part_idx = None
    for i, part in enumerate(parts):
        if part.name == "pantheon_plus_shoes_ladder":
            ladder_part_idx = i
            break
    if ladder_part_idx is None:
        raise ValueError("Could not find stack part named 'pantheon_plus_shoes_ladder'")
    ladder_ds = parts[ladder_part_idx].dataset
    if not isinstance(ladder_ds, PantheonPlusShoesLadderDataset):
        raise TypeError("Expected pantheon_plus_shoes_ladder dataset to be PantheonPlusShoesLadderDataset")

    # Build a predictive-score cfg containing ONLY {baseline, test_model} to speed permutations.
    base_entry = _get_pred_model_entry(pred_cfg_full, base_label)
    test_entry = _get_pred_model_entry(pred_cfg_full, test_label)
    pred_cfg = dict(pred_cfg_full)
    pred_cfg["models"] = [base_entry, test_entry]

    rng = np.random.default_rng(int(args.seed))
    n_perm = int(args.n_perm)
    perm_seeds = rng.integers(0, 2**32 - 1, size=n_perm, dtype=np.uint32).astype(np.int64)

    global _G_PARTS, _G_DS0, _G_ANCHOR, _G_BASE_LEVEL, _G_BASE_MODEL_CFG, _G_PRED_CFG, _G_TEST_LABEL
    global _G_LADDER_IDX, _G_LADDER_DS, _G_PERMUTE_COLUMN, _G_APPLY_TO, _G_PERMUTE_MODE, _G_FIXED_TRAIN_MASK

    _G_PARTS = parts
    _G_DS0 = ds0
    _G_ANCHOR = anchor
    _G_BASE_LEVEL = base_level
    _G_BASE_MODEL_CFG = base_model_cfg
    _G_PRED_CFG = pred_cfg
    _G_TEST_LABEL = test_label
    _G_LADDER_IDX = int(ladder_part_idx)
    _G_LADDER_DS = ladder_ds
    _G_PERMUTE_COLUMN = str(args.permute_column)
    _G_APPLY_TO = str(args.apply_to)  # type: ignore[assignment]
    _G_PERMUTE_MODE = str(args.mode)  # type: ignore[assignment]
    _G_FIXED_TRAIN_MASK = fixed_train_mask

    n_proc = int(args.n_proc)
    if n_proc < 1:
        raise ValueError("--n-proc must be >= 1")

    safe = "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in test_label)[:80]
    progress_path = args.progress_path
    if progress_path is None:
        progress_path = run_dir / f"permutation_progress_{safe}_{args.permute_column}_{args.mode}.json"
    progress_path = progress_path.expanduser().resolve()

    print(
        f"Permutation null starting:\n"
        f"- run_dir: {run_dir}\n"
        f"- test_label: {test_label}\n"
        f"- permute_column: {args.permute_column}\n"
        f"- mode: {args.mode}\n"
        f"- apply_to: {args.apply_to}\n"
        f"- n_perm: {n_perm}\n"
        f"- n_proc: {n_proc}\n"
        f"- progress_path: {progress_path}\n"
        f"- observed_mean_delta_logp: {obs_mean:+.6f}\n",
        flush=True,
    )
    _write_json_atomic(
        progress_path,
        {
            "status": "starting",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "test_label": test_label,
            "permute_column": str(args.permute_column),
            "mode": str(args.mode),
            "apply_to": str(args.apply_to),
            "n_proc": int(n_proc),
            "n_done": 0,
            "n_total": int(n_perm),
            "fraction_done": 0.0,
            "elapsed_sec": 0.0,
            "rate_perm_per_sec": None,
            "eta_sec": None,
            "observed_mean_delta_logp": float(obs_mean),
            "perm_running_mean_delta_logp": None,
            "perm_running_sd_delta_logp": None,
            "n_perm_ge_obs": 0,
            "p_hat_ge_obs": None,
        },
    )

    # Avoid oversubscription; BLAS threads should be 1 even with many processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    draws = np.empty(n_perm, dtype=float)
    t0 = time.time()
    progress_every = int(args.progress_every)
    progress_secs = float(args.progress_secs)
    last_progress_time = 0.0
    # Welford moments for perm mean Δlogp draws.
    run_n = 0
    run_mean = 0.0
    run_M2 = 0.0
    run_ge = 0

    def maybe_emit_progress(i_done: int) -> None:
        nonlocal last_progress_time
        if progress_every <= 0:
            by_count = False
        else:
            by_count = i_done > 0 and (i_done % progress_every == 0 or i_done == n_perm)
        now = time.time()
        by_time = progress_secs > 0 and (now - last_progress_time) >= progress_secs
        if not (by_count or by_time or i_done == n_perm):
            return

        elapsed = max(time.time() - t0, 1e-9)
        rate = i_done / elapsed
        eta = (n_perm - i_done) / rate if rate > 0 else float("inf")
        sd = (run_M2 / (run_n - 1)) ** 0.5 if run_n >= 2 else float("nan")
        p_hat = (1.0 + float(run_ge)) / (float(run_n) + 1.0) if run_n >= 1 else float("nan")
        stamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "status": "running" if i_done < n_perm else "complete",
            "updated_utc": stamp,
            "run_dir": str(run_dir),
            "test_label": test_label,
            "permute_column": str(args.permute_column),
            "mode": str(args.mode),
            "apply_to": str(args.apply_to),
            "n_proc": int(n_proc),
            "n_done": int(i_done),
            "n_total": int(n_perm),
            "fraction_done": float(i_done / n_perm) if n_perm else None,
            "elapsed_sec": float(elapsed),
            "rate_perm_per_sec": _finite_or_none(rate),
            "eta_sec": _finite_or_none(eta),
            "observed_mean_delta_logp": float(obs_mean),
            "perm_running_mean_delta_logp": float(run_mean) if run_n else None,
            "perm_running_sd_delta_logp": _finite_or_none(sd),
            "n_perm_ge_obs": int(run_ge),
            "p_hat_ge_obs": _finite_or_none(p_hat),
        }
        _write_json_atomic(progress_path, payload)
        last_progress_time = now

        if progress_every > 0 or progress_secs > 0:
            print(
                f"progress: {i_done}/{n_perm} perms "
                f"({100.0 * i_done / n_perm:.1f}%) "
                f"elapsed={elapsed/60.0:.1f} min "
                f"eta={eta/60.0:.1f} min "
                f"p_hat≈{p_hat:.3g}",
                flush=True,
            )

    if n_proc == 1:
        for i, s in enumerate(perm_seeds.tolist()):
            val = float(_one_permutation(int(s)))
            draws[i] = val
            run_n += 1
            delta = val - run_mean
            run_mean += delta / run_n
            run_M2 += delta * (val - run_mean)
            if val >= obs_mean:
                run_ge += 1
            maybe_emit_progress(i + 1)
    else:
        ctx = mp.get_context("fork")
        chunksize = int(args.chunksize)
        with ctx.Pool(processes=n_proc) as pool:
            for i, val in enumerate(pool.imap(_one_permutation, perm_seeds.tolist(), chunksize=chunksize), start=1):
                val = float(val)
                draws[i - 1] = val
                run_n += 1
                delta = val - run_mean
                run_mean += delta / run_n
                run_M2 += delta * (val - run_mean)
                if val >= obs_mean:
                    run_ge += 1
                maybe_emit_progress(i)

    # One-sided p-value for “>= observed”.
    p = float((1.0 + float(np.sum(draws >= obs_mean))) / float(n_perm + 1))

    out = {
        "run_dir": str(run_dir),
        "base_label": base_label,
        "test_label": test_label,
        "permute_column": str(args.permute_column),
        "apply_to": str(args.apply_to),
        "mode": str(args.mode),
        "n_perm": int(n_perm),
        "seed": int(args.seed),
        "observed_mean_delta_logp": obs_mean,
        "perm_mean_delta_logp": draws.tolist(),
        "perm_summary": {
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws)),
            "p05": float(np.quantile(draws, 0.05)),
            "p50": float(np.quantile(draws, 0.50)),
            "p95": float(np.quantile(draws, 0.95)),
        },
        "p_value_ge_obs": p,
    }

    out_path = args.out_json
    if out_path is None:
        out_path = run_dir / f"permutation_null_{safe}_{args.permute_column}_{args.mode}.json"
    out_path = out_path.expanduser().resolve()
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    # Final progress write (complete).
    maybe_emit_progress(n_perm)
    print(f"Wrote: {out_path}")
    print(f"Observed mean Δlogp: {obs_mean:+.4f}")
    print(f"Permutation p-value (Δlogp >= obs): {p:.4g}")


if __name__ == "__main__":
    main()
