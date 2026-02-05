from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class SplitFitResult:
    split_var: str
    param: str
    rows: list[dict[str, Any]]

    def to_jsonable(self) -> dict[str, Any]:
        return {"split_var": self.split_var, "param": self.param, "rows": self.rows}


def run_split_fit(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> SplitFitResult:
    split_var = str(split_cfg.get("split_var", "idsurvey"))
    param = str(split_cfg.get("param", "global_offset_mag"))
    use_diag = bool(split_cfg.get("use_diagonal_errors", True))
    include_nan = bool(split_cfg.get("include_nan", False))

    v = dataset.get_column(split_var)
    rows: list[dict[str, Any]] = []

    mode = str(split_cfg.get("mode", "categories"))
    if mode == "categories":
        cats = np.unique(v.astype(int))
        for c in cats:
            mask = v.astype(int) == int(c)
            rows.append(_fit_row(dataset, mask=mask, anchor=anchor, ladder_level=ladder_level, model_cfg=model_cfg, param=param, label=str(c), use_diag=use_diag))
    elif mode == "bins":
        edges = list(split_cfg.get("edges", []))
        if len(edges) < 2:
            raise ValueError("split_fit mode=bins requires split_cfg.edges with >=2 values")
        edges = [float(x) for x in edges]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (v >= lo) & (v < hi)
            if include_nan:
                mask = np.isnan(v) | mask
            rows.append(_fit_row(dataset, mask=mask, anchor=anchor, ladder_level=ladder_level, model_cfg=model_cfg, param=param, label=f"[{lo},{hi})", use_diag=use_diag))
    else:
        raise ValueError(f"Unknown split_fit mode: {mode}")

    return SplitFitResult(split_var=split_var, param=param, rows=rows)


def _fit_row(dataset, *, mask: np.ndarray, anchor: AnchorLCDM, ladder_level: str, model_cfg: dict[str, Any], param: str, label: str, use_diag: bool) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) < 5:
        return {"label": label, "n": int(np.sum(mask)), "skipped": True}
    sub = dataset.subset_mask(mask)
    y, y0, cov, X, prior = sub.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
    if use_diag:
        cov = sub.diag_sigma() ** 2
    fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior))
    if param not in fit.param_names:
        return {"label": label, "n": int(y.size), "skipped": True, "reason": f"param '{param}' not in model"}
    j = fit.param_names.index(param)
    return {"label": label, "n": int(y.size), "mean": float(fit.mean[j]), "sd": float(np.sqrt(fit.cov[j, j]))}
