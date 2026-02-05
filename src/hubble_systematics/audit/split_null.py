from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class SplitNullResult:
    split_var: str
    mode: str
    edges: list[float] | None
    param: str
    shuffle_within: str | None
    observed: dict[str, Any]
    mc: dict[str, list[float]]
    p_values: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "split_var": self.split_var,
            "mode": self.mode,
            "edges": self.edges,
            "param": self.param,
            "shuffle_within": self.shuffle_within,
            "observed": self.observed,
            "mc": self.mc,
            "p_values": self.p_values,
        }


def run_split_null_mc(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    split_null_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> SplitNullResult:
    split_var = str(split_null_cfg.get("split_var", "pkmjd"))
    mode = str(split_null_cfg.get("mode", "bins"))
    param = str(split_null_cfg.get("param", "delta_lnH0"))
    use_diag = bool(split_null_cfg.get("use_diagonal_errors", True))
    n_mc = int(split_null_cfg.get("n_mc", 200))
    shuffle_within = split_null_cfg.get("shuffle_within")
    shuffle_within = str(shuffle_within) if shuffle_within is not None else None
    include_nan = bool(split_null_cfg.get("include_nan", False))
    composition_var = split_null_cfg.get("composition_var")
    composition_var = str(composition_var) if composition_var is not None else None
    composition_top_k = int(split_null_cfg.get("composition_top_k", 5))

    v = np.asarray(dataset.get_column(split_var), dtype=float)

    if mode != "bins":
        raise ValueError("split_null currently supports mode=bins only")
    edges = list(split_null_cfg.get("edges", []))
    if len(edges) < 3:
        raise ValueError("split_null mode=bins requires edges with >=3 values")
    edges = [float(x) for x in edges]
    edges = sorted(edges)

    if shuffle_within is not None:
        g = np.asarray(dataset.get_column(shuffle_within))
    else:
        g = None

    obs = _fit_bins(
        dataset=dataset,
        anchor=anchor,
        ladder_level=ladder_level,
        model_cfg=model_cfg,
        param=param,
        v=v,
        edges=edges,
        use_diag=use_diag,
        include_nan=include_nan,
        composition_var=composition_var,
        composition_top_k=composition_top_k,
    )
    obs_metrics = _bin_metrics(obs)

    mc_span = np.empty(n_mc, dtype=float)
    mc_chi2 = np.empty(n_mc, dtype=float)
    for t in range(n_mc):
        v_perm = _permute_within_groups(v, groups=g, rng=rng)
        r = _fit_bins(
            dataset=dataset,
            anchor=anchor,
            ladder_level=ladder_level,
            model_cfg=model_cfg,
            param=param,
            v=v_perm,
            edges=edges,
            use_diag=use_diag,
            include_nan=include_nan,
            composition_var=composition_var,
            composition_top_k=composition_top_k,
        )
        m = _bin_metrics(r)
        mc_span[t] = float(m["span"])
        mc_chi2[t] = float(m["chi2_const"])

    p_span = float(np.mean(mc_span >= float(obs_metrics["span"])))
    p_chi2 = float(np.mean(mc_chi2 >= float(obs_metrics["chi2_const"])))

    return SplitNullResult(
        split_var=split_var,
        mode=mode,
        edges=edges,
        param=param,
        shuffle_within=shuffle_within,
        observed={"bins": obs, "metrics": obs_metrics},
        mc={"span": mc_span.tolist(), "chi2_const": mc_chi2.tolist()},
        p_values={"span": p_span, "chi2_const": p_chi2},
    )


def _permute_within_groups(v: np.ndarray, *, groups: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    nan = np.isnan(v)
    if groups is None:
        out = v.copy()
        idx = np.where(~nan)[0]
        out[idx] = rng.permutation(v[idx])
        return out
    groups = np.asarray(groups)
    out = v.copy()
    for g in np.unique(groups):
        idx = np.where((groups == g) & (~nan))[0]
        out[idx] = rng.permutation(v[idx])
    return out


def _fit_bins(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    param: str,
    v: np.ndarray,
    edges: list[float],
    use_diag: bool,
    include_nan: bool,
    composition_var: str | None,
    composition_top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v = np.asarray(v, dtype=float)
    comp = None
    if composition_var is not None:
        try:
            comp = np.asarray(dataset.get_column(composition_var))
        except Exception:
            comp = None
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (v >= lo) & (v < hi)
        if include_nan:
            mask = np.isnan(v) | mask
        row: dict[str, Any] = {"label": f"[{lo},{hi})", "n": int(np.sum(mask))}
        if comp is not None:
            row["composition"] = _composition_summary(comp, mask=mask, top_k=composition_top_k)
        if int(np.sum(mask)) < 5:
            row["skipped"] = True
            rows.append(row)
            continue
        sub = dataset.subset_mask(mask)
        y, y0, cov, X, prior = sub.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
        prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
        if use_diag:
            cov = sub.diag_sigma() ** 2
        fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior))
        if param not in fit.param_names:
            row["skipped"] = True
            row["reason"] = f"param '{param}' not in model"
            rows.append(row)
            continue
        j = fit.param_names.index(param)
        row["mean"] = float(fit.mean[j])
        row["var"] = float(fit.cov[j, j])
        rows.append(row)
    return rows


def _bin_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    means = []
    vars_ = []
    for r in rows:
        if r.get("skipped"):
            continue
        means.append(float(r["mean"]))
        vars_.append(float(r["var"]))
    if len(means) < 2:
        return {"span": 0.0, "chi2_const": 0.0}
    m = np.asarray(means, dtype=float)
    v = np.asarray(vars_, dtype=float)
    span = float(np.max(m) - np.min(m))
    w = 1.0 / v
    m0 = float(np.sum(w * m) / np.sum(w))
    chi2 = float(np.sum(((m - m0) ** 2) / v))
    return {"span": span, "chi2_const": chi2}


def _composition_summary(values: np.ndarray, *, mask: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    x = values[mask]
    if x.size == 0:
        return []
    # Treat values as numeric where possible.
    try:
        x_num = x.astype(float)
        ok = np.isfinite(x_num)
        x_use = x_num[ok]
    except Exception:
        x_use = x
    if x_use.size == 0:
        return []
    uniq, counts = np.unique(x_use, return_counts=True)
    order = np.argsort(-counts)
    out: list[dict[str, Any]] = []
    for i in order[: max(0, int(top_k))]:
        v = uniq[i]
        if isinstance(v, (np.floating, float)):
            if np.isfinite(float(v)) and abs(float(v) - round(float(v))) < 1e-6:
                v_out: Any = int(round(float(v)))
            else:
                v_out = float(v)
        else:
            v_out = v.item() if hasattr(v, "item") else v
        out.append({"value": v_out, "count": int(counts[i])})
    return out
