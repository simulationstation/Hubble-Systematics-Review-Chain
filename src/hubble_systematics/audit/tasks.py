from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.correlated_cut_null import run_correlated_cut_null_mc
from hubble_systematics.audit.correlated_cut_null import run_correlated_cut_null_siren_gate2_event_gaussian_mc
from hubble_systematics.audit.group_split_null import run_group_split_null_mc
from hubble_systematics.audit.injection import run_injection_suite
from hubble_systematics.audit.hemisphere_scan import run_hemisphere_scan
from hubble_systematics.audit.predictive_score import run_predictive_score
from hubble_systematics.audit.prior_mc import run_prior_mc
from hubble_systematics.audit.stack_predictive_dataset import StackPredictiveDataset, StackPredictivePart
from hubble_systematics.audit.split_null import run_split_null_mc
from hubble_systematics.audit.sbc import run_sbc
from hubble_systematics.audit.split_fit import run_split_fit
from hubble_systematics.audit.types import BaselineFitResult, CutScanResult, DriftNullResult
from hubble_systematics.audit.util import make_cut_grid, write_json
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model, log_marginal_likelihood
from hubble_systematics.joint import Part, stack_parts
from hubble_systematics.ledger import mechanism_ledger
from hubble_systematics.probes.bao import load_bao_dataset
from hubble_systematics.probes.chronometers import load_chronometers_dataset
from hubble_systematics.probes.gaussian_measurement import load_gaussian_measurement_dataset
from hubble_systematics.probes.h0_grid import load_h0_grid_posterior_dataset
from hubble_systematics.probes.pantheon_plus import load_pantheon_plus_dataset
from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset
from hubble_systematics.probes.siren_gate2_grid import load_siren_gate2_grid_dataset
from hubble_systematics.shared_scale import apply_shared_scale_prior


def _build_anchor(cfg: dict[str, Any]) -> AnchorLCDM:
    a = cfg.get("anchor", {})
    return AnchorLCDM(
        H0=float(a.get("H0", 67.4)),
        Omega_m=float(a.get("Omega_m", 0.315)),
        Omega_k=float(a.get("Omega_k", 0.0)),
        rd_Mpc=float(a.get("rd_Mpc", 147.09)),
    )


def _load_probe(cfg: dict[str, Any]):
    probe = cfg.get("probe", {})
    name = probe.get("name")
    if name == "pantheon_plus":
        return load_pantheon_plus_dataset(probe)
    if name == "pantheon_plus_shoes_ladder":
        return load_pantheon_plus_shoes_ladder_dataset(probe)
    if name == "bao":
        return load_bao_dataset(probe)
    if name == "chronometers":
        return load_chronometers_dataset(probe)
    if name == "h0_grid":
        return load_h0_grid_posterior_dataset(probe)
    if name == "gaussian_measurement":
        return load_gaussian_measurement_dataset(probe)
    if name == "siren_gate2_grid":
        return load_siren_gate2_grid_dataset(probe)
    raise ValueError(f"Unknown probe name: {name}")


def run_fit_baseline(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")

    model = cfg.get("model", {})
    base_level = str(model.get("ladder_level", "L1"))

    if probe_name == "stack":
        items = probe_cfg.get("stack", []) or []
        if not items:
            raise ValueError("stack probe requires non-empty probe.stack")
        parts: list[Part] = []
        part_labels: list[str] = []
        for item in items:
            item = dict(item)
            name = item.get("name")
            if name is None:
                raise ValueError("Each stack item must include a name")
            # Merge base model with any per-item overrides.
            item_model = _deep_merge_dicts(model, item.get("model", {}) or {})
            level = str(item_model.get("ladder_level", base_level))
            ds = _load_probe({"probe": item})
            y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level=level, cfg=item_model)
            parts.append(Part(y=y, y0=y0, cov=cov, X=X, prior=prior))
            part_labels.append(str(name))
        stacked = stack_parts(parts)
        y, y0, cov, X, prior = stacked.y, stacked.y0, stacked.cov, stacked.X, stacked.prior
        level = base_level
    else:
        probe_obj = _load_probe(cfg)
        level = base_level
        y, y0, cov, X, prior = probe_obj.build_design(anchor=anchor, ladder_level=level, cfg=model)

    # Apply shared-scale global priors once (avoid per-probe double counting).
    prior = apply_shared_scale_prior(prior, model_cfg=model)

    spec = GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior)
    fit = fit_gaussian_linear_model(spec)
    logZ = log_marginal_likelihood(spec)

    # Optional: per-part chi2 contributions for stack probes.
    if probe_name == "stack":
        name_to_idx = {str(nm): j for j, nm in enumerate(fit.param_names)}
        part_rows: list[dict[str, Any]] = []
        for label, part in zip(part_labels, parts, strict=True):
            y_p = np.asarray(part.y, dtype=float).reshape(-1)
            y0_p = np.asarray(part.y0, dtype=float).reshape(-1)
            X_p = np.asarray(part.X, dtype=float)
            cov_p = np.asarray(part.cov, dtype=float)
            local_names = list(part.prior.param_names)
            beta = np.zeros(len(local_names), dtype=float)
            for i, nm in enumerate(local_names):
                j = name_to_idx.get(str(nm))
                if j is None:
                    continue
                beta[i] = float(fit.mean[j])
            r = y_p - (y0_p + X_p @ beta)
            if cov_p.ndim == 1:
                chi2_p = float(np.sum((r * r) / cov_p))
            else:
                L = np.linalg.cholesky(cov_p)
                u = np.linalg.solve(L, r)
                chi2_p = float(u @ u)
            part_rows.append({"label": label, "n": int(y_p.size), "chi2": chi2_p})
        write_json(ctx.run_dir / "stack_parts.json", {"parts": part_rows})

    out = BaselineFitResult(
        ladder_level=level,
        n=int(y.shape[0]),
        param_names=fit.param_names,
        mean=fit.mean.tolist(),
        cov=np.asarray(fit.cov).tolist(),
        chi2=float(fit.chi2),
        dof=int(fit.dof),
        logdet_cov=float(fit.logdet_cov),
        log_evidence=logZ,
    )
    write_json(ctx.run_dir / "baseline_fit.json", asdict(out))
    write_json(
        ctx.run_dir / "mechanism_ledger.json",
        mechanism_ledger(anchor=anchor, param_names=fit.param_names, mean=fit.mean, cov=fit.cov),
    )
    return asdict(out)


def run_baseline_sweep_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")

    base_model = cfg.get("model", {}) or {}
    base_level = str(base_model.get("ladder_level", "L1"))
    sweep = cfg.get("sweep", []) or []
    if not isinstance(sweep, list) or not sweep:
        return {"skipped": True, "reason": "baseline_sweep requires a non-empty top-level sweep: [...] list"}

    stack_items = None
    if probe_name == "stack":
        stack_items = probe_cfg.get("stack", []) or []
        if not stack_items:
            raise ValueError("stack probe requires non-empty probe.stack")
    else:
        dataset = _load_probe(cfg)

    rows: list[dict[str, Any]] = []
    base_delta_lnH0 = None
    base_logZ = None

    for i, item in enumerate(sweep):
        if not isinstance(item, dict):
            raise ValueError("sweep entries must be mappings")
        label = str(item.get("label", f"var{i}"))
        model_cfg = _deep_merge_dicts(base_model, item.get("model", {}) or {})

        if stack_items is not None:
            # Stack-aware sweep: allow per-part overrides via sweep_item.stack_overrides.
            stack_overrides = item.get("stack_overrides", {}) or {}
            if not isinstance(stack_overrides, dict):
                raise ValueError("sweep.stack_overrides must be a mapping of stack-part-name -> model override dict")
            parts: list[Part] = []
            for part_cfg in stack_items:
                part_cfg = dict(part_cfg)
                part_name = part_cfg.get("name")
                if part_name is None:
                    raise ValueError("Each stack item must include a name")
                part_override = stack_overrides.get(str(part_name), {}) or {}
                part_model = _deep_merge_dicts(_deep_merge_dicts(model_cfg, part_cfg.get("model", {}) or {}), part_override)
                level = str(part_model.get("ladder_level", base_level))
                ds = _load_probe({"probe": part_cfg})
                y_p, y0_p, cov_p, X_p, prior_p = ds.build_design(anchor=anchor, ladder_level=level, cfg=part_model)
                parts.append(Part(y=y_p, y0=y0_p, cov=cov_p, X=X_p, prior=prior_p))
            stacked = stack_parts(parts)
            y, y0, cov, X, prior = stacked.y, stacked.y0, stacked.cov, stacked.X, stacked.prior
            prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
            level = "stack"
        else:
            level = str(item.get("ladder_level", model_cfg.get("ladder_level", base_level)))
            y, y0, cov, X, prior = dataset.build_design(anchor=anchor, ladder_level=level, cfg=model_cfg)
            prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)

        spec = GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior)
        fit = fit_gaussian_linear_model(spec)
        logZ = log_marginal_likelihood(spec)
        if i == 0:
            base_logZ = logZ
        ledger = mechanism_ledger(anchor=anchor, param_names=fit.param_names, mean=fit.mean, cov=fit.cov)

        delta_lnH0 = ledger.get("delta_lnH0", {}).get("mean") if isinstance(ledger.get("delta_lnH0"), dict) else None
        if i == 0:
            base_delta_lnH0 = delta_lnH0
        reduction = None
        if base_delta_lnH0 is not None and delta_lnH0 is not None and abs(float(base_delta_lnH0)) > 1e-6:
            reduction = float((float(base_delta_lnH0) - float(delta_lnH0)) / float(base_delta_lnH0))

        # Lightweight parameter summary for interpretability.
        names = list(fit.param_names)
        mean = np.asarray(fit.mean, dtype=float)
        sd = np.sqrt(np.clip(np.diag(np.asarray(fit.cov, dtype=float)), 0.0, np.inf))

        def param(name: str) -> dict[str, float] | None:
            if name not in names:
                return None
            j = names.index(name)
            return {"mean": float(mean[j]), "sd": float(sd[j])}

        def max_abs(prefix: str) -> dict[str, float] | None:
            idx = [j for j, nm in enumerate(names) if str(nm).startswith(prefix)]
            if not idx:
                return None
            return {"max_abs_mean": float(np.max(np.abs(mean[idx]))), "n": int(len(idx))}

        param_summary = {
            "calibrator_offset_mag": param("calibrator_offset_mag"),
            "cal_survey_offsets": max_abs("cal_survey_offset_"),
            "time_bin_offsets": max_abs("pkmjd_bin_offset_"),
            "survey_time_offsets": max_abs("survey_pkmjd_bin_offset_"),
            "hf_survey_offsets": max_abs("hf_survey_offset_"),
            "hf_z_spline": (max_abs("hf_z_spline_") or max_abs("z_spline_")),
            "sky_modes": max_abs("sky_"),
        }

        rows.append(
            {
                "label": label,
                "ladder_level": level,
                "n": int(y.shape[0]),
                "chi2": float(fit.chi2),
                "dof": int(fit.dof),
                "chi2_per_dof": float(fit.chi2 / fit.dof) if fit.dof > 0 else float("nan"),
                "log_evidence": logZ,
                "delta_log_evidence_vs_first": (None if (logZ is None or base_logZ is None) else float(logZ - base_logZ)),
                "ledger": ledger,
                "param_summary": param_summary,
                "tension_reduction_fraction_vs_first": reduction,
            }
        )

    out = {"base_label": rows[0]["label"], "rows": rows}
    write_json(ctx.run_dir / "baseline_sweep.json", out)
    return out

def run_cut_scan(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "cut scans are not implemented for stack probes"}

    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_leq"):
        return {"skipped": True, "reason": f"cut scans require a subset_leq-capable dataset (probe={probe_name})"}
    scan_cfg = cfg.get("scan", {})
    cut_var = str(scan_cfg.get("cut_var", "z"))
    cut_mode = str(scan_cfg.get("cut_mode", "leq")).lower()
    if cut_mode not in {"leq", "geq"}:
        raise ValueError(f"scan.cut_mode must be one of {{leq,geq}}; got {cut_mode!r}")
    if cut_mode == "geq" and not hasattr(dataset, "subset_geq"):
        return {"skipped": True, "reason": f"cut_mode=geq requires subset_geq on dataset (probe={probe_name})"}
    cut_grid = make_cut_grid(dataset=dataset, cut_var=cut_var, scan_cfg=scan_cfg)

    model = cfg.get("model", {})
    level = str(model.get("ladder_level", "L1"))

    subset_fn = dataset.subset_leq if cut_mode == "leq" else dataset.subset_geq
    rows = []
    for cut in cut_grid:
        subset = subset_fn(cut_var, cut)
        y, y0, cov, X, prior = subset.build_design(anchor=anchor, ladder_level=level, cfg=model)
        prior = apply_shared_scale_prior(prior, model_cfg=model)
        spec = GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior)
        fit = fit_gaussian_linear_model(spec)
        rows.append(
            {
                "cut": float(cut),
                "n": int(y.shape[0]),
                "chi2": float(fit.chi2),
                "dof": int(fit.dof),
                "params": dict(zip(fit.param_names, fit.mean.tolist(), strict=True)),
            }
        )

    out = CutScanResult(cut_var=cut_var, cuts=[float(c) for c in cut_grid], rows=rows, cut_mode=cut_mode)
    write_json(ctx.run_dir / "cut_scan.json", out.to_jsonable())
    return out.to_jsonable()


def run_correlated_cut_null(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "correlated-cut null is not implemented for stack probes"}

    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_leq"):
        return {"skipped": True, "reason": f"correlated-cut null requires a subset_leq-capable dataset (probe={probe_name})"}
    scan_cfg = cfg.get("scan", {})
    cut_var = str(scan_cfg.get("cut_var", "z"))
    cut_mode = str(scan_cfg.get("cut_mode", "leq")).lower()
    if cut_mode not in {"leq", "geq"}:
        raise ValueError(f"scan.cut_mode must be one of {{leq,geq}}; got {cut_mode!r}")
    if cut_mode == "geq" and not hasattr(dataset, "subset_geq"):
        return {"skipped": True, "reason": f"cut_mode=geq requires subset_geq on dataset (probe={probe_name})"}
    cut_grid = make_cut_grid(dataset=dataset, cut_var=cut_var, scan_cfg=scan_cfg)

    model = cfg.get("model", {})
    level = str(model.get("ladder_level", "L1"))
    # YAML gotcha: an unquoted "null:" key may parse as None under YAML 1.1.
    null_cfg = (cfg.get("null") or cfg.get(None) or {})  # type: ignore[arg-type]
    n_mc = int(null_cfg.get("n_mc", 200))
    seed = null_cfg.get("seed")
    drift_param = str(null_cfg.get("drift_param", "global_offset_mag"))
    use_diag = bool(null_cfg.get("use_diagonal_errors", True))
    rng = np.random.default_rng(seed)

    null_mode = str(null_cfg.get("mode", "")).lower()
    if probe_name == "siren_gate2_grid" and (null_mode == "event_gaussian"):
        drift = run_correlated_cut_null_siren_gate2_event_gaussian_mc(
            dataset=dataset,
            anchor=anchor,
            cut_var=cut_var,
            cut_mode=cut_mode,
            cuts=cut_grid,
            n_mc=n_mc,
            rng=rng,
        )
    else:
        drift = run_correlated_cut_null_mc(
            dataset=dataset,
            anchor=anchor,
            ladder_level=level,
            model_cfg=model,
            cut_var=cut_var,
            cut_mode=cut_mode,
            cuts=cut_grid,
            n_mc=n_mc,
            rng=rng,
            drift_param=drift_param,
            use_diagonal_errors=use_diag,
        )
    write_json(ctx.run_dir / "correlated_cut_null.json", drift.to_jsonable())
    return drift.to_jsonable()


def run_injections(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "injection suite is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "diag_sigma"):
        return {"skipped": True, "reason": f"injection suite requires a diag_sigma-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {})
    level = str(model.get("ladder_level", "L1"))

    inj_cfg = cfg.get("injection", {}) or {}

    def safe_label(s: str) -> str:
        return "".join(c if (c.isalnum() or c in {"-", "_"} or c == ".") else "_" for c in s)[:80]

    if isinstance(inj_cfg, list):
        out: dict[str, Any] = {}
        for i, item in enumerate(inj_cfg):
            if not isinstance(item, dict):
                raise ValueError("injection list entries must be mappings")
            label = str(item.get("label") or item.get("mechanism") or f"inj{i}")
            seed = item.get("seed")
            rng = np.random.default_rng(seed)
            res = run_injection_suite(dataset=dataset, anchor=anchor, ladder_level=str(item.get("ladder_level", level)), model_cfg=model, inj_cfg=item, rng=rng)
            out[label] = res.to_jsonable()
            write_json(ctx.run_dir / f"injection_suite_{safe_label(label)}.json", res.to_jsonable())
        write_json(ctx.run_dir / "injection_suite.json", out)
        return out

    seed = inj_cfg.get("seed")
    rng = np.random.default_rng(seed)
    res = run_injection_suite(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, inj_cfg=inj_cfg, rng=rng)
    write_json(ctx.run_dir / "injection_suite.json", res.to_jsonable())
    return res.to_jsonable()


def run_sbc_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "SBC is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "diag_sigma"):
        return {"skipped": True, "reason": f"SBC requires a diag_sigma-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {})
    level = str(model.get("ladder_level", "L1"))

    sbc_cfg = cfg.get("sbc", {}) or {}
    seed = sbc_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_sbc(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, sbc_cfg=sbc_cfg, rng=rng)
    write_json(ctx.run_dir / "sbc.json", res.to_jsonable())
    return res.to_jsonable()


def run_hemisphere_scan_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "hemisphere_scan is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_mask"):
        return {"skipped": True, "reason": f"hemisphere_scan requires a subset_mask-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {})
    level = str(cfg.get("hemisphere_scan", {}).get("ladder_level", model.get("ladder_level", "L1")))
    hemi_cfg = cfg.get("hemisphere_scan", {}) or {}
    seed = hemi_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_hemisphere_scan(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, hemi_cfg=hemi_cfg, rng=rng)
    write_json(ctx.run_dir / "hemisphere_scan.json", res.to_jsonable())
    return res.to_jsonable()


def run_predictive_score_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")
    dataset = None
    fixed_train_mask = None

    if probe_name == "stack":
        items = probe_cfg.get("stack", []) or []
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
            ds = _load_probe({"probe": item})
            parts.append(StackPredictivePart(name=str(name), dataset=ds, base_model=item.get("model", {}) or {}))

        dataset = StackPredictiveDataset.from_parts(
            parts=parts,
            anchor=anchor,
            base_level=base_level,
            base_model_cfg=base_model,
            scope_part=str(scope_part) if scope_part is not None else None,
        )
        if scope_part is not None:
            scope_mask = dataset.part_mask_full(str(scope_part))
            if int(np.sum(scope_mask)) < 2:
                raise ValueError("predictive_score.stack_scope_part must match a part with at least 2 rows")
            fixed_train_mask = ~scope_mask

    else:
        dataset = _load_probe(cfg)
        if not hasattr(dataset, "subset_mask"):
            return {"skipped": True, "reason": f"predictive_score requires a subset_mask-capable dataset (probe={probe_name})"}

    base_model = cfg.get("model", {}) or {}
    base_level = str(base_model.get("ladder_level", "L1"))
    pred_cfg = cfg.get("predictive_score", {}) or {}
    seed = pred_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_predictive_score(
        dataset=dataset,
        anchor=anchor,
        base_level=base_level,
        base_model_cfg=base_model,
        pred_cfg=pred_cfg,
        rng=rng,
        fixed_train_mask=fixed_train_mask,
    )
    out = res.to_jsonable()
    write_json(ctx.run_dir / "predictive_score.json", out)
    return out


def run_prior_mc_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")

    dataset = None
    if probe_name == "stack":
        items = probe_cfg.get("stack", []) or []
        if not items:
            raise ValueError("stack probe requires non-empty probe.stack")

        base_model = cfg.get("model", {}) or {}
        base_level = str(base_model.get("ladder_level", "L1"))
        mc_cfg = cfg.get("prior_mc", {}) or {}
        scope_part = mc_cfg.get("stack_scope_part")

        parts: list[StackPredictivePart] = []
        for item in items:
            item = dict(item)
            name = item.get("name")
            if name is None:
                raise ValueError("Each stack item must include a name")
            ds = _load_probe({"probe": item})
            parts.append(StackPredictivePart(name=str(name), dataset=ds, base_model=item.get("model", {}) or {}))

        dataset = StackPredictiveDataset.from_parts(
            parts=parts,
            anchor=anchor,
            base_level=base_level,
            base_model_cfg=base_model,
            scope_part=str(scope_part) if scope_part is not None else None,
        )
    else:
        dataset = _load_probe(cfg)
        if not hasattr(dataset, "diag_sigma"):
            return {"skipped": True, "reason": f"prior_mc requires a diag_sigma-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {}) or {}
    level = str(model.get("ladder_level", "L1"))
    mc_cfg = cfg.get("prior_mc", {}) or {}
    seed = mc_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_prior_mc(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, prior_mc_cfg=mc_cfg, rng=rng)
    write_json(ctx.run_dir / "prior_mc.json", res.to_jsonable())
    return res.to_jsonable()


def run_split_fit_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {})
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "split_fit is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_mask"):
        return {"skipped": True, "reason": f"split_fit requires a subset_mask-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {})
    split_cfg = cfg.get("split_fit", {}) or {}
    if isinstance(split_cfg, dict):
        level = str(split_cfg.get("ladder_level", model.get("ladder_level", "L1")))
    else:
        level = str(model.get("ladder_level", "L1"))

    def safe_label(s: str) -> str:
        return "".join(c if (c.isalnum() or c in {"-", "_"} or c == ".") else "_" for c in s)[:80]

    if isinstance(split_cfg, list):
        out: dict[str, Any] = {}
        for i, item in enumerate(split_cfg):
            if not isinstance(item, dict):
                raise ValueError("split_fit list entries must be mappings")
            label = str(item.get("label") or item.get("split_var") or f"split{i}")
            res = run_split_fit(dataset=dataset, anchor=anchor, ladder_level=str(item.get("ladder_level", level)), model_cfg=model, split_cfg=item)
            out[label] = res.to_jsonable()
            write_json(ctx.run_dir / f"split_fit_{safe_label(label)}.json", res.to_jsonable())
        write_json(ctx.run_dir / "split_fit.json", out)
        return out

    res = run_split_fit(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, split_cfg=split_cfg)
    write_json(ctx.run_dir / "split_fit.json", res.to_jsonable())
    return res.to_jsonable()


def run_split_null_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "split_null is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_mask"):
        return {"skipped": True, "reason": f"split_null requires a subset_mask-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {}) or {}
    split_null_cfg = cfg.get("split_null", {}) or {}
    level = str(split_null_cfg.get("ladder_level", model.get("ladder_level", "L1")))
    seed = split_null_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_split_null_mc(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, split_null_cfg=split_null_cfg, rng=rng)
    write_json(ctx.run_dir / "split_null.json", res.to_jsonable())
    return res.to_jsonable()


def run_group_split_null_task(ctx) -> dict[str, Any]:
    cfg = ctx.config
    anchor = _build_anchor(cfg)
    probe_cfg = cfg.get("probe", {}) or {}
    probe_name = probe_cfg.get("name")
    if probe_name == "stack":
        return {"skipped": True, "reason": "group_split_null is not implemented for stack probes"}
    dataset = _load_probe(cfg)
    if not hasattr(dataset, "subset_mask"):
        return {"skipped": True, "reason": f"group_split_null requires a subset_mask-capable dataset (probe={probe_name})"}

    model = cfg.get("model", {}) or {}
    group_cfg = cfg.get("group_split_null", {}) or {}
    level = str(group_cfg.get("ladder_level", model.get("ladder_level", "L1")))
    seed = group_cfg.get("seed")
    rng = np.random.default_rng(seed)

    res = run_group_split_null_mc(dataset=dataset, anchor=anchor, ladder_level=level, model_cfg=model, group_cfg=group_cfg, rng=rng)
    out = res.to_jsonable()
    write_json(ctx.run_dir / "group_split_null.json", out)
    # Also write per-group files for convenience.
    for gid, item in (out.get("results") or {}).items():
        write_json(ctx.run_dir / f"group_split_null_{gid}.json", item)
    return out


def _stack_design_matrices(datasets, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
    parts = []
    for ds in datasets:
        y, y0, cov, X, prior = ds.build_design(anchor=anchor, ladder_level=ladder_level, cfg=cfg)
        parts.append(Part(y=y, y0=y0, cov=cov, X=X, prior=prior))
    stacked = stack_parts(parts)
    return stacked.y, stacked.y0, stacked.cov, stacked.X, stacked.prior


def _block_diag(mats: list[np.ndarray]) -> np.ndarray:
    if len(mats) == 1:
        return mats[0]
    ndims = {np.asarray(m).ndim for m in mats}
    if ndims == {1}:
        return np.concatenate([np.asarray(m) for m in mats], axis=0)
    if ndims == {2}:
        sizes = [m.shape[0] for m in mats]
        out = np.zeros((sum(sizes), sum(sizes)))
        i = 0
        for m in mats:
            n = m.shape[0]
            out[i : i + n, i : i + n] = m
            i += n
        return out

    # Mixed: fall back to dense block-diagonal.
    dense = []
    for m in mats:
        m = np.asarray(m)
        if m.ndim == 1:
            dense.append(np.diag(m))
        else:
            dense.append(m)
    sizes = [m.shape[0] for m in dense]
    out = np.zeros((sum(sizes), sum(sizes)))
    i = 0
    for m in dense:
        n = m.shape[0]
        out[i : i + n, i : i + n] = m
        i += n
    return out


def _deep_merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
