from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class PriorMCComponent:
    label: str
    mechanism: str
    sigmas: dict[str, float]
    deltas: dict[str, np.ndarray]


@dataclass(frozen=True)
class PriorMCResult:
    param_of_interest: str
    n_mc: int
    use_diagonal_errors: bool
    tension_target: float | None
    components: list[dict[str, Any]]
    draws: list[float]
    shift: list[float]
    fraction_of_tension: list[float] | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "param_of_interest": self.param_of_interest,
            "n_mc": self.n_mc,
            "use_diagonal_errors": self.use_diagonal_errors,
            "tension_target": self.tension_target,
            "components": self.components,
            "draws": self.draws,
            "shift": self.shift,
            "fraction_of_tension": self.fraction_of_tension,
        }


def _load_sigma_overrides(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[3] / p).resolve()
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        raise ValueError("sigma_overrides_path must point to a JSON mapping")
    out: dict[str, float] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = float(v)
    return out


def _pkmjd_bin_id(*, t: np.ndarray, edges: list[float]) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    edges = [float(x) for x in edges]
    edges = sorted(edges)
    if len(edges) < 3:
        raise ValueError("edges must have length >=3")
    good = np.isfinite(t) & (t > 0.0)
    inner = np.asarray(edges[1:-1], dtype=float)
    bid = np.zeros_like(t, dtype=int)
    bid[good] = np.digitize(t[good], inner, right=False)
    return bid


def _component_masks(
    *,
    dataset,
    mechanism: str,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Build a dict of delta vectors (amp=1.0) keyed by a param-name-like label.
    """
    n = int(np.asarray(dataset.z).size)
    mechanism = str(mechanism)

    if mechanism == "calibrator_offset_mag":
        if not hasattr(dataset, "is_calibrator"):
            raise ValueError("calibrator_offset_mag requires dataset.is_calibrator")
        w = np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        if w.shape != (n,):
            raise ValueError("dataset.is_calibrator shape mismatch")
        return {"calibrator_offset_mag": w.astype(float)}

    if mechanism == "pkmjd_bin_offset_mag":
        if not hasattr(dataset, "pkmjd"):
            raise ValueError("pkmjd_bin_offset_mag requires dataset.pkmjd")
        edges = cfg.get("edges")
        if edges is None:
            raise ValueError("pkmjd_bin_offset_mag requires edges")
        edges = [float(x) for x in edges]
        t = np.asarray(getattr(dataset, "pkmjd"), dtype=float).reshape(-1)
        if t.shape != (n,):
            raise ValueError("dataset.pkmjd shape mismatch")
        bid = _pkmjd_bin_id(t=t, edges=edges)

        bins = cfg.get("bins")
        if bins is None:
            bins = list(range(1, len(edges) - 1))
        bins = [int(b) for b in bins]
        apply_to = str(cfg.get("apply_to", "all")).lower()
        base = np.ones(n, dtype=bool)
        if apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("pkmjd_bin_offset_mag apply_to=cal requires dataset.is_calibrator")
            base = base & np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        elif apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("pkmjd_bin_offset_mag apply_to=hf requires dataset.is_hubble_flow")
            base = base & np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).reshape(-1)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")

        out: dict[str, np.ndarray] = {}
        for k in bins:
            out[f"pkmjd_bin_offset_{k}"] = (base & (bid == k)).astype(float)
        return out

    if mechanism == "survey_pkmjd_bin_offset_mag":
        if not hasattr(dataset, "pkmjd"):
            raise ValueError("survey_pkmjd_bin_offset_mag requires dataset.pkmjd")
        if not hasattr(dataset, "idsurvey"):
            raise ValueError("survey_pkmjd_bin_offset_mag requires dataset.idsurvey")
        edges = cfg.get("edges")
        if edges is None:
            raise ValueError("survey_pkmjd_bin_offset_mag requires edges")
        edges = [float(x) for x in edges]
        t = np.asarray(getattr(dataset, "pkmjd"), dtype=float).reshape(-1)
        ids = np.asarray(getattr(dataset, "idsurvey"), dtype=int).reshape(-1)
        if t.shape != (n,) or ids.shape != (n,):
            raise ValueError("dataset.pkmjd/idsurvey shape mismatch")
        bid = _pkmjd_bin_id(t=t, edges=edges)

        sids = cfg.get("idsurveys")
        if sids is None:
            sids = np.unique(ids[np.isfinite(ids)]).tolist()
        sids = [int(s) for s in sids]
        bins = cfg.get("bins")
        if bins is None:
            bins = list(range(1, len(edges) - 1))
        bins = [int(b) for b in bins]
        apply_to = str(cfg.get("apply_to", "cal")).lower()
        base = np.ones(n, dtype=bool)
        if apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("survey_pkmjd_bin_offset_mag apply_to=cal requires dataset.is_calibrator")
            base = base & np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        elif apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("survey_pkmjd_bin_offset_mag apply_to=hf requires dataset.is_hubble_flow")
            base = base & np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).reshape(-1)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")

        out: dict[str, np.ndarray] = {}
        for sid in sids:
            in_s = ids == int(sid)
            for k in bins:
                out[f"survey_pkmjd_bin_offset_{sid}_{k}"] = (base & in_s & (bid == k)).astype(float)
        return out

    raise ValueError(f"Unsupported mechanism for prior_mc: {mechanism!r}")


def run_prior_mc(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    prior_mc_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> PriorMCResult:
    n_mc = int(prior_mc_cfg.get("n_mc", 10_000))
    use_diag = bool(prior_mc_cfg.get("use_diagonal_errors", True))
    poi = str(prior_mc_cfg.get("param_of_interest", "delta_lnH0"))
    tension_target = prior_mc_cfg.get("tension_target")
    tension_target_str = None
    if isinstance(tension_target, str):
        tension_target_str = str(tension_target).strip().lower()
        tension_target = None
    tension_target = None if tension_target is None else float(tension_target)

    # Fit once to define a baseline "true" parameter vector.
    y_obs, y0, cov, X, prior = dataset.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
    if use_diag:
        sigma = dataset.diag_sigma()
        cov = sigma**2
    else:
        sigma = None
    fit0 = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_obs, y0=y0, cov=cov, X=X, prior=prior))
    if poi not in fit0.param_names:
        raise ValueError(f"param_of_interest '{poi}' not in params {fit0.param_names}")
    j_poi = fit0.param_names.index(poi)
    beta_true = np.asarray(fit0.mean, dtype=float).reshape(-1)
    if tension_target_str == "baseline_poi":
        tension_target = float(beta_true[j_poi])
    elif tension_target_str == "baseline_poi_abs":
        tension_target = float(abs(beta_true[j_poi]))

    # Baseline noiseless expectation.
    y_base = y0 + X @ beta_true

    # Build components.
    sig_overrides = _load_sigma_overrides(str(prior_mc_cfg.get("sigma_overrides_path") or ""))
    comps_cfg = prior_mc_cfg.get("components") or []
    if not isinstance(comps_cfg, list) or not comps_cfg:
        raise ValueError("prior_mc.components must be a non-empty list")

    components: list[PriorMCComponent] = []
    for item in comps_cfg:
        if not isinstance(item, dict):
            raise ValueError("prior_mc.components entries must be mappings")
        label = str(item.get("label") or item.get("mechanism") or "component")
        mech = str(item.get("mechanism"))
        fallback_sigma = float(item.get("fallback_sigma_mag", 0.02))
        scale = float(item.get("scale", 1.0))

        deltas = _component_masks(dataset=dataset, mechanism=mech, cfg=item)
        sigmas: dict[str, float] = {}
        for name in deltas.keys():
            sig = sig_overrides.get(str(name))
            if sig is None:
                sig = fallback_sigma
            sigmas[str(name)] = float(scale) * float(sig)

        components.append(PriorMCComponent(label=label, mechanism=mech, sigmas=sigmas, deltas=deltas))

    # Simulation loop.
    draws = np.empty(n_mc, dtype=float)
    shift = np.empty(n_mc, dtype=float)
    frac = np.empty(n_mc, dtype=float) if (tension_target is not None and abs(float(tension_target)) > 0.0) else None

    for t in range(n_mc):
        delta = np.zeros_like(y_base, dtype=float)
        # Draw each component coefficient and add its delta.
        for comp in components:
            for name, m in comp.deltas.items():
                sig = float(comp.sigmas.get(name, 0.0))
                if sig <= 0.0:
                    continue
                amp = float(rng.normal(0.0, sig))
                delta = delta + amp * np.asarray(m, dtype=float)

        if use_diag:
            assert sigma is not None
            eps = rng.normal(0.0, sigma, size=sigma.size)
        else:
            eps = rng.multivariate_normal(mean=np.zeros_like(y_base), cov=cov)

        y_sim = y_base + delta + eps
        fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_sim, y0=y0, cov=cov, X=X, prior=prior))
        val = float(fit.mean[j_poi])
        draws[t] = val
        shift[t] = val - float(beta_true[j_poi])
        if frac is not None:
            frac[t] = abs(shift[t]) / abs(float(tension_target))

    comp_meta: list[dict[str, Any]] = []
    for comp in components:
        comp_meta.append(
            {
                "label": comp.label,
                "mechanism": comp.mechanism,
                "n_terms": int(len(comp.deltas)),
                "sigmas": {k: float(v) for k, v in comp.sigmas.items()},
            }
        )

    return PriorMCResult(
        param_of_interest=poi,
        n_mc=int(n_mc),
        use_diagonal_errors=bool(use_diag),
        tension_target=tension_target,
        components=comp_meta,
        draws=draws.tolist(),
        shift=shift.tolist(),
        fraction_of_tension=(None if frac is None else frac.tolist()),
    )
