from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FragilisticData:
    cov: np.ndarray
    labels: np.ndarray

    @property
    def n(self) -> int:
        return int(self.labels.size)


_SANITIZE_RE = re.compile(r"[^0-9A-Za-z]+")


def sanitize_label(label: str) -> str:
    s = str(label).strip()
    s = _SANITIZE_RE.sub("_", s).strip("_")
    return s if s else "UNKNOWN"


def frag_key(label: str) -> str:
    """Return a FRAGILISTIC group key for a label.

    Labels are of the form "<SURVEY> <BAND>" or "<SURVEY> p1 <BAND>" (CFA4 variants). We group
    by "<SURVEY>" except for the CFA4 p1/p2 style where we group by the first two tokens.
    """

    parts = str(label).strip().split()
    if len(parts) >= 2 and parts[1].startswith("p"):
        return " ".join(parts[:2])
    return parts[0]


def groups_from_labels(labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    out: dict[str, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        out.setdefault(frag_key(str(lab)), []).append(int(i))
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def default_idsurvey_to_name() -> dict[int, str]:
    # Pantheon+SH0ES README_DataRelease_4_DISTANCES_AND_COVAR.txt (IDSURVEY mapping).
    return {
        1: "SDSS",
        4: "SNLS",
        5: "CSP",
        10: "DES",
        15: "PS1MD",
        18: "CNIa0.02",
        50: "LOWZ/JRK07",
        51: "LOSS1",
        56: "SOUSA",
        57: "LOSS2",
        61: "CFA1",
        62: "CFA2",
        63: "CFA3S",
        64: "CFA3K",
        65: "CFA4p2",
        66: "CFA4p3",
        100: "HST",
        101: "SNAP",
        106: "CANDELS",
        150: "FOUND",
    }


def default_survey_to_frag_keys() -> dict[str, list[str]]:
    # Pragmatic mapping used in scripts/derive_pantheon_shoes_fragilistic_priors.py
    return {
        "SDSS": ["SDSS"],
        "SNLS": ["SNLS"],
        "CSP": ["CSPDR3"],
        "DES": ["DES3YR", "DES5YR"],
        "PS1MD": ["PS1", "oldPS1", "newPS1"],
        "FOUND": ["PS1", "oldPS1", "newPS1"],
        # DataRelease uses "CNIa0.02"; FRAGILISTIC labels this family as "CNIa0.2".
        "CNIa0.02": ["CNIa0.2"],
        "CFA3S": ["CFA3S"],
        "CFA3K": ["CFA3K"],
        "CFA4p2": ["CFA4 p2"],
        # FRAGILISTIC only ships CFA4 p1/p2; treat CFA4p3 as closest available.
        "CFA4p3": ["CFA4 p1"],
        "LOSS1": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        "LOSS2": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        "SOUSA": ["SWIFT"],
        # No direct FRAGILISTIC key; use a conservative fallback.
        "LOWZ/JRK07": [],
        "CFA1": [],
        "CFA2": [],
        "HST": ["HST"],
        "SNAP": ["SNAP"],
        "CANDELS": ["CANDELS"],
    }


@lru_cache(maxsize=8)
def _load_fragilistic_npz_cached(abs_path: str) -> FragilisticData:
    p = Path(abs_path)
    z = np.load(p)
    cov = np.asarray(z["cov"], dtype=float)
    labels = np.asarray(z["labels"])
    if cov.shape != (labels.size, labels.size):
        raise ValueError("FRAGILISTIC cov/labels shape mismatch")
    return FragilisticData(cov=cov, labels=labels)


def load_fragilistic_npz(path: str | Path) -> FragilisticData:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return _load_fragilistic_npz_cached(str(p))


@dataclass(frozen=True)
class FragilisticDesign:
    X: np.ndarray
    names: list[str]
    label_indices: np.ndarray
    unmapped_idsurvey: np.ndarray


def build_fragilistic_filter_design(
    *,
    idsurvey: np.ndarray,
    labels: np.ndarray,
    apply_weight: np.ndarray,
    idsurvey_to_name: dict[int, str] | None = None,
    survey_to_frag_keys: dict[str, list[str]] | None = None,
    include_only_used_labels: bool = True,
    param_prefix: str = "frag_zp_",
) -> FragilisticDesign:
    """Build a filter-level calibration-offset design matrix from FRAGILISTIC labels.

    Each Pantheon IDSURVEY is mapped to 0+ FRAGILISTIC "group keys". We then assign each SN row a
    *uniform* weight over all FRAGILISTIC filter labels included for that survey.

    This is not a full photometry-level forward model; it's an explicit, reproducible way to:
      - keep the *filter-level* correlated prior as shipped by Pantheon+,
      - while using only the public standardized-magnitude table.
    """

    ids = np.asarray(idsurvey, dtype=int).reshape(-1)
    n = int(ids.size)
    labels = np.asarray(labels)
    apply_weight = np.asarray(apply_weight, dtype=float).reshape(-1)
    if apply_weight.shape != (n,):
        raise ValueError("apply_weight shape mismatch")

    id_to_name = default_idsurvey_to_name() if idsurvey_to_name is None else dict(idsurvey_to_name)
    survey_to_keys = default_survey_to_frag_keys() if survey_to_frag_keys is None else dict(survey_to_frag_keys)
    groups = groups_from_labels(labels)

    per_sid_label_idx: dict[int, np.ndarray] = {}
    unmapped: list[int] = []
    for sid in np.unique(ids).tolist():
        sid = int(sid)
        sname = id_to_name.get(sid, f"IDSURVEY_{sid}")
        keys = survey_to_keys.get(sname, [])
        idx: list[int] = []
        for k in keys:
            g = groups.get(str(k))
            if g is None:
                continue
            idx.extend(g.tolist())
        if not idx:
            unmapped.append(sid)
            per_sid_label_idx[sid] = np.asarray([], dtype=int)
            continue
        per_sid_label_idx[sid] = np.unique(np.asarray(idx, dtype=int))

    if include_only_used_labels:
        used = np.unique(np.concatenate([v for v in per_sid_label_idx.values() if v.size], axis=0)) if per_sid_label_idx else np.asarray([], dtype=int)
        label_indices = used
    else:
        label_indices = np.arange(labels.size, dtype=int)

    label_indices = np.asarray(label_indices, dtype=int).reshape(-1)
    if label_indices.size == 0:
        return FragilisticDesign(
            X=np.zeros((n, 0), dtype=float),
            names=[],
            label_indices=np.asarray([], dtype=int),
            unmapped_idsurvey=np.asarray(unmapped, dtype=int),
        )

    # Map original label index -> column index in returned matrix.
    col_of_label = {int(li): j for j, li in enumerate(label_indices.tolist())}

    B = np.zeros((n, label_indices.size), dtype=float)
    for sid in np.unique(ids).tolist():
        sid = int(sid)
        idx = per_sid_label_idx.get(sid)
        if idx is None or idx.size == 0:
            continue
        cols = [col_of_label[int(i)] for i in idx.tolist() if int(i) in col_of_label]
        if not cols:
            continue
        cols = sorted(set(cols))
        w = 1.0 / float(len(cols))
        m = ids == sid
        if not np.any(m):
            continue
        B[np.ix_(m, cols)] = apply_weight[m, None] * w

    # Parameter names: include the original label index to avoid accidental collisions.
    names = []
    for li in label_indices.tolist():
        lab = str(labels[int(li)])
        names.append(f"{param_prefix}{int(li):03d}_{sanitize_label(lab)}")

    return FragilisticDesign(
        X=B,
        names=names,
        label_indices=label_indices,
        unmapped_idsurvey=np.asarray(unmapped, dtype=int),
    )


def fragilistic_precision_from_cov(*, cov: np.ndarray, scale: float = 1.0, jitter: float = 0.0) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and >0")
    C = cov * float(scale) ** 2
    if jitter:
        C = C + float(jitter) * np.eye(C.shape[0])
    # Small matrix (<=102). Use a direct inverse; upstream code checks positive-definiteness.
    return np.linalg.inv(C)


@lru_cache(maxsize=16)
def cached_fragilistic_precision(*, abs_npz_path: str, scale: float = 1.0, jitter: float = 0.0) -> np.ndarray:
    """Cached full precision matrix for a given FRAGILISTIC npz + scale/jitter.

    Note: this returns the precision for *all* labels in the file. If callers select a subset of
    labels, they must compute `inv(cov_sub)` themselves (a marginal prior), not a sub-block of this
    conditional precision.
    """

    frag = _load_fragilistic_npz_cached(str(Path(abs_npz_path)))
    return fragilistic_precision_from_cov(cov=frag.cov, scale=scale, jitter=jitter)


def parse_fragilistic_prior_cfg(prior_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalized fragilistic prior config dict from model.priors.*."""

    raw = (prior_cfg or {}).get("fragilistic") or {}
    if not isinstance(raw, dict):
        raise ValueError("priors.fragilistic must be a mapping")
    out: dict[str, Any] = {}
    out["enable"] = bool(raw.get("enable", False))
    out["npz_path"] = str(raw.get("npz_path", "data/raw/pantheon_plus_calibration/FRAGILISTIC_COVARIANCE.npz"))
    out["scale"] = float(raw.get("scale", 1.0))
    out["jitter"] = float(raw.get("jitter", 0.0))
    return out
