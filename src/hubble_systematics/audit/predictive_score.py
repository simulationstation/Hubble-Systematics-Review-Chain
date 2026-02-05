from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class PredictiveScoreResult:
    mode: str
    use_diagonal_errors: bool
    seed: int | None
    n_rep: int
    train_frac: float | None
    always_include_calibrators: bool
    always_include_hubble_flow: bool
    models: dict[str, dict[str, Any]]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "use_diagonal_errors": self.use_diagonal_errors,
            "seed": self.seed,
            "n_rep": self.n_rep,
            "train_frac": self.train_frac,
            "always_include_calibrators": self.always_include_calibrators,
            "always_include_hubble_flow": self.always_include_hubble_flow,
            "models": self.models,
        }


def run_predictive_score(
    *,
    dataset,
    anchor: AnchorLCDM,
    base_level: str,
    base_model_cfg: dict[str, Any],
    pred_cfg: dict[str, Any],
    rng: np.random.Generator,
    fixed_train_mask: np.ndarray | None = None,
) -> PredictiveScoreResult:
    mode = str(pred_cfg.get("mode", "random")).lower()
    use_diag = bool(pred_cfg.get("use_diagonal_errors", True))
    seed = pred_cfg.get("seed")
    always_include_cal = bool(pred_cfg.get("always_include_calibrators", False))
    always_include_hf = bool(pred_cfg.get("always_include_hubble_flow", False))

    model_list = pred_cfg.get("models") or pred_cfg.get("compare") or []
    if not isinstance(model_list, list) or not model_list:
        raise ValueError("predictive_score.models must be a non-empty list")

    if mode == "random":
        n_rep = int(pred_cfg.get("n_rep", 50))
        train_frac = float(pred_cfg.get("train_frac", 0.7))
        if not (0.1 < train_frac < 0.95):
            raise ValueError("predictive_score.train_frac must be in (0.1,0.95)")
        splits = _random_splits(
            dataset=dataset,
            n_rep=n_rep,
            train_frac=train_frac,
            rng=rng,
            always_include_calibrators=always_include_cal,
            always_include_hubble_flow=always_include_hf,
            fixed_train_mask=fixed_train_mask,
        )
    elif mode == "group_holdout":
        n_rep = 0
        train_frac = None
        group_var = str(pred_cfg.get("group_var", "idsurvey"))
        min_group_n = int(pred_cfg.get("min_group_n", 20))
        splits = _group_holdout_splits(
            dataset=dataset,
            group_var=group_var,
            min_group_n=min_group_n,
            always_include_calibrators=always_include_cal,
            always_include_hubble_flow=always_include_hf,
            fixed_train_mask=fixed_train_mask,
        )
        n_rep = len(splits)
    else:
        raise ValueError(f"Unsupported predictive_score.mode: {mode}")

    out_models: dict[str, dict[str, Any]] = {}
    base_label = None

    for i_model, item in enumerate(model_list):
        if not isinstance(item, dict):
            raise ValueError("predictive_score.models entries must be mappings")
        label = str(item.get("label") or f"model{i_model}")
        if base_label is None:
            base_label = label

        model_cfg = _deep_merge_dicts(base_model_cfg, item.get("model", {}) or {})
        if "stack_overrides" in item:
            # Stack-specific: per-part model overrides keyed by stack part name.
            so = item.get("stack_overrides", {}) or {}
            if not isinstance(so, dict):
                raise ValueError("predictive_score.models[*].stack_overrides must be a mapping")
            model_cfg = dict(model_cfg)
            model_cfg["stack_overrides"] = _deep_merge_dicts(model_cfg.get("stack_overrides", {}) or {}, so)
        level = str(item.get("ladder_level", model_cfg.get("ladder_level", base_level)))

        scores = np.empty(len(splits), dtype=float)
        for j, (train_mask, test_mask) in enumerate(splits):
            ds_train = dataset.subset_mask(train_mask)
            ds_test = dataset.subset_mask(test_mask)

            y_tr, y0_tr, cov_tr, X_tr, prior_tr = ds_train.build_design(anchor=anchor, ladder_level=level, cfg=model_cfg)
            prior_tr = apply_shared_scale_prior(prior_tr, model_cfg=model_cfg)
            if use_diag:
                cov_tr = ds_train.diag_sigma() ** 2
            spec_tr = GaussianLinearModelSpec(y=y_tr, y0=y0_tr, cov=cov_tr, X=X_tr, prior=prior_tr)
            fit_tr = fit_gaussian_linear_model(spec_tr)

            y_te, y0_te, cov_te, X_te, prior_te = ds_test.build_design(anchor=anchor, ladder_level=level, cfg=model_cfg)
            if use_diag:
                cov_te = ds_test.diag_sigma() ** 2
            X_te_aligned = _align_X(X_te, prior_te.param_names, fit_tr.param_names)

            scores[j] = _predictive_logpdf(
                y=y_te,
                y0=y0_te,
                cov=cov_te,
                X=X_te_aligned,
                post_mean=fit_tr.mean,
                post_cov=fit_tr.cov,
            )

        out_models[label] = {
            "ladder_level": level,
            "mean_logp": float(np.mean(scores)),
            "sd_logp": float(np.std(scores)),
            "logp": scores.tolist(),
        }

    assert base_label is not None
    base_scores = np.asarray(out_models[base_label]["logp"], dtype=float)
    for label, obj in out_models.items():
        scores = np.asarray(obj["logp"], dtype=float)
        d = scores - base_scores
        obj["mean_delta_logp_vs_base"] = float(np.mean(d))
        obj["sd_delta_logp_vs_base"] = float(np.std(d))

    return PredictiveScoreResult(
        mode=mode,
        use_diagonal_errors=use_diag,
        seed=int(seed) if seed is not None else None,
        n_rep=int(len(splits)),
        train_frac=train_frac,
        always_include_calibrators=always_include_cal,
        always_include_hubble_flow=always_include_hf,
        models=out_models,
    )


def _random_splits(
    *,
    dataset,
    n_rep: int,
    train_frac: float,
    rng: np.random.Generator,
    always_include_calibrators: bool,
    always_include_hubble_flow: bool,
    fixed_train_mask: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = int(dataset.z.size)
    all_idx = np.arange(n)

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

    free_idx = all_idx[~fixed_train]
    if free_idx.size < 2:
        raise ValueError("Not enough free rows to split after applying fixed_train constraints")
    n_train_free = int(round(train_frac * free_idx.size))
    n_train_free = int(np.clip(n_train_free, 1, max(free_idx.size - 1, 1)))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_rep):
        pick = rng.choice(free_idx, size=n_train_free, replace=False)
        train = fixed_train.copy()
        train[pick] = True
        test = ~train
        # Keep calibrators out of test if requested.
        if always_include_calibrators:
            test &= ~fixed_train
        splits.append((train, test))
    return splits


def _group_holdout_splits(
    *,
    dataset,
    group_var: str,
    min_group_n: int,
    always_include_calibrators: bool,
    always_include_hubble_flow: bool,
    fixed_train_mask: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
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

    # Only hold out "valid" group labels.
    if np.issubdtype(v.dtype, np.number):
        good = np.isfinite(v)
    else:
        good = np.ones(n, dtype=bool)
        if v.dtype.kind in {"U", "S"}:
            good &= v != ""
        else:
            good &= np.asarray([x is not None for x in v], dtype=bool)
    groups = np.unique(v[good])

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for g in groups:
        test = (v == g) & good & (~fixed_train)
        if int(np.sum(test)) < min_group_n:
            continue
        train = (~test) | fixed_train
        splits.append((train.astype(bool), test.astype(bool)))
    return splits


def _align_X(X: np.ndarray, names: list[str], target_names: list[str]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if names == target_names:
        return X
    n = X.shape[0]
    out = np.zeros((n, len(target_names)), dtype=float)
    src = {str(nm): j for j, nm in enumerate(names)}
    for j, nm in enumerate(target_names):
        k = src.get(str(nm))
        if k is None:
            continue
        out[:, j] = X[:, k]
    return out


def _predictive_logpdf(*, y: np.ndarray, y0: np.ndarray, cov: np.ndarray, X: np.ndarray, post_mean: np.ndarray, post_cov: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if y.shape != y0.shape or X.shape[0] != y.size:
        raise ValueError("Shape mismatch in predictive_logpdf")

    mu = y0 + X @ np.asarray(post_mean, dtype=float).reshape(-1)
    r = y - mu
    S = np.asarray(post_cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("post_cov must be square")

    # Predictive covariance = C + X S X^T.
    if cov.ndim == 1:
        var = np.asarray(cov, dtype=float).reshape(-1)
        if var.shape != (y.size,):
            raise ValueError("diag cov shape mismatch")
        if not (np.all(np.isfinite(var)) and np.all(var > 0)):
            raise ValueError("diag cov must be finite and >0")
        return _logpdf_diag_plus_lowrank(r=r, var=var, X=X, S=S)

    C = np.asarray(cov, dtype=float)
    if C.shape != (y.size, y.size):
        raise ValueError("dense cov shape mismatch")
    Sigma = C + X @ S @ X.T
    return _logpdf_dense(r=r, Sigma=Sigma)


def _logpdf_dense(*, r: np.ndarray, Sigma: np.ndarray) -> float:
    r = np.asarray(r, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = r.size
    # Cholesky with tiny jitter fallback.
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * float(np.mean(np.diag(Sigma)))
        L = np.linalg.cholesky(Sigma + jitter * np.eye(n))
    u = np.linalg.solve(L, r)
    quad = float(u @ u)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return float(-0.5 * (n * np.log(2.0 * np.pi) + logdet + quad))


def _logpdf_diag_plus_lowrank(*, r: np.ndarray, var: np.ndarray, X: np.ndarray, S: np.ndarray) -> float:
    r = np.asarray(r, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    n = r.size

    Dinv = 1.0 / var
    # K = S^{-1} + X^T D^{-1} X
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    XtDinv = X.T * Dinv[None, :]
    K = Sinv + XtDinv @ X
    try:
        Lk = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * float(np.mean(np.diag(K)))
        Lk = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))

    # quad = r^T D^{-1} r - (X^T D^{-1} r)^T K^{-1} (X^T D^{-1} r)
    q0 = float(np.sum((r * r) * Dinv))
    z = XtDinv @ r
    w = np.linalg.solve(Lk, z)
    w = np.linalg.solve(Lk.T, w)
    quad = q0 - float(z @ w)

    signS, logdetS = np.linalg.slogdet(S)
    if signS <= 0 or not np.isfinite(logdetS):
        # Fallback: build dense Sigma.
        Sigma = np.diag(var) + X @ S @ X.T
        return _logpdf_dense(r=r, Sigma=Sigma)

    logdetD = float(np.sum(np.log(var)))
    logdetK = 2.0 * float(np.sum(np.log(np.diag(Lk))))
    logdet = logdetD + logdetS + logdetK
    return float(-0.5 * (n * np.log(2.0 * np.pi) + logdet + quad))


def _deep_merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
