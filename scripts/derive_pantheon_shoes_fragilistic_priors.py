from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


@dataclass(frozen=True)
class FragilisticGroup:
    key: str
    idx: np.ndarray


def _frag_key(label: str) -> str:
    parts = str(label).strip().split()
    if len(parts) >= 2 and parts[1].startswith("p"):
        return " ".join(parts[:2])
    return parts[0]


def _groups_from_labels(labels: np.ndarray) -> dict[str, FragilisticGroup]:
    labels = np.asarray(labels)
    out: dict[str, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        k = _frag_key(str(lab))
        out.setdefault(k, []).append(int(i))
    return {k: FragilisticGroup(key=k, idx=np.asarray(v, dtype=int)) for k, v in out.items()}


def _sigma_mean_offset(*, cov: np.ndarray, idx: np.ndarray) -> float | None:
    cov = np.asarray(cov, dtype=float)
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if idx.size == 0:
        return None
    idx = np.unique(idx)
    w = np.ones(idx.size, dtype=float) / float(idx.size)
    C = cov[np.ix_(idx, idx)]
    v = float(w @ C @ w)
    if not np.isfinite(v) or v < 0.0:
        return None
    return float(np.sqrt(v))


def derive_sigma_overrides(
    *,
    cov: np.ndarray,
    labels: np.ndarray,
    idsurvey_levels: np.ndarray,
    include_survey_offsets: bool,
    include_cal_survey_offsets: bool,
    include_hf_survey_offsets: bool,
    scale: float,
    fallback_sigma_mag: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    cov = np.asarray(cov, dtype=float)
    labels = np.asarray(labels)
    idsurvey_levels = np.asarray(idsurvey_levels, dtype=int)

    groups = _groups_from_labels(labels)

    # Mapping from Pantheon+SH0ES IDSURVEY -> short survey label (from the DataRelease README).
    idsurvey_to_name: dict[int, str] = {
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

    # Mapping from survey label -> FRAGILISTIC group keys.
    # Note: this is a pragmatic compression (zeropoint covariance is per-survey+band, while our audit
    # model uses per-survey offsets on the standardized magnitude).
    survey_to_frag_keys: dict[str, list[str]] = {
        "SDSS": ["SDSS"],
        "SNLS": ["SNLS"],
        "CSP": ["CSPDR3"],
        "DES": ["DES3YR", "DES5YR"],
        "PS1MD": ["PS1", "oldPS1", "newPS1"],
        "FOUND": ["PS1", "oldPS1", "newPS1"],
        # The DataRelease uses "CNIa0.02"; FRAGILISTIC labels this family as "CNIa0.2".
        "CNIa0.02": ["CNIa0.2"],
        "CFA3S": ["CFA3S"],
        "CFA3K": ["CFA3K"],
        "CFA4p2": ["CFA4 p2"],
        # FRAGILISTIC only ships CFA4 p1/p2; treat CFA4p3 as the closest available variant.
        "CFA4p3": ["CFA4 p1"],
        # LOSS datasets are KAIT/Nickel reductions (FRAGILISTIC provides KAIT1-4, NICKEL1-2).
        "LOSS1": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        "LOSS2": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        # SOUSA = Swift Optical/UV SN Archive (FRAGILISTIC provides SWIFT filters).
        "SOUSA": ["SWIFT"],
        # No direct FRAGILISTIC key; use a conservative fallback.
        "LOWZ/JRK07": [],
        "CFA1": [],
        "CFA2": [],
    }

    overrides: dict[str, float] = {}
    meta: dict[str, Any] = {
        "created_by": "scripts/derive_pantheon_shoes_fragilistic_priors.py",
        "scale": float(scale),
        "fallback_sigma_mag": float(fallback_sigma_mag),
        "idsurvey_to_name": {str(k): v for k, v in sorted(idsurvey_to_name.items())},
        "survey_to_frag_keys": survey_to_frag_keys,
        "fragilistic_keys": sorted(groups.keys()),
        "computed": {},
        "notes": [
            "FRAGILISTIC_COVARIANCE.npz is the zeropoint-offset covariance from Brout+21 (as shipped by Pantheon+SH0ES DataRelease).",
            "We compress per-survey+band zeropoint offsets into a single per-survey offset by taking the mean across that survey’s bands.",
            "This is a pragmatic prior for our audit model’s per-survey standardized-magnitude offset parameters; it is not a full photometry-level forward model.",
        ],
    }

    computed: dict[str, Any] = {}
    for sid in idsurvey_levels.tolist():
        sid = int(sid)
        sname = idsurvey_to_name.get(sid, f"IDSURVEY_{sid}")
        keys = survey_to_frag_keys.get(sname, [])

        idx: list[int] = []
        missing = []
        for k in keys:
            g = groups.get(k)
            if g is None:
                missing.append(k)
                continue
            idx.extend(g.idx.tolist())

        sig = _sigma_mean_offset(cov=cov, idx=np.asarray(idx, dtype=int))
        if sig is None:
            sig = float(fallback_sigma_mag)
        sig = float(scale) * float(sig)

        computed[str(sid)] = {
            "name": sname,
            "frag_keys": keys,
            "missing_frag_keys": missing,
            "sigma_mag": sig,
        }

        if include_survey_offsets:
            overrides[f"survey_offset_{sid}"] = sig
        if include_cal_survey_offsets:
            overrides[f"cal_survey_offset_{sid}"] = sig
        if include_hf_survey_offsets:
            overrides[f"hf_survey_offset_{sid}"] = sig

    meta["computed"] = computed
    return overrides, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fragilistic-npz",
        default="data/raw/pantheon_plus_calibration/FRAGILISTIC_COVARIANCE.npz",
        help="Path to FRAGILISTIC_COVARIANCE.npz (labels + covariance).",
    )
    ap.add_argument(
        "--out",
        default="data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_fragilistic_v1.json",
        help="Output JSON path for sigma_overrides mapping.",
    )
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fallback-sigma-mag", type=float, default=0.01)
    ap.add_argument("--include-survey-offsets", action="store_true", default=True)
    ap.add_argument("--include-cal-survey-offsets", action="store_true", default=False)
    ap.add_argument("--include-hf-survey-offsets", action="store_true", default=False)
    ap.add_argument(
        "--pantheon-plus-cosmology-npz",
        default="data/processed/pantheon_plus/pantheon_plus_sky_cosmology_stat+sys_zHD.npz",
        help="Optional Pantheon+ cosmology-subset NPZ used to include all IDSURVEY levels present in the main SN set.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    npz_path = Path(args.fragilistic_npz).expanduser()
    if not npz_path.is_absolute():
        npz_path = (root / npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    z = np.load(npz_path)
    cov = z["cov"]
    labels = z["labels"]

    # Use the same ladder selection that our audit configs use, to get relevant idsurvey levels.
    ds = load_pantheon_plus_shoes_ladder_dataset(
        {
            "raw_dat_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat"),
            "raw_cov_path": str(root / "data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov"),
            "include_calibrators": True,
            "include_hubble_flow": True,
            "z_column": "zHD",
            "z_hf_min": 0.023,
            "z_hf_max": 0.15,
            "tag": "cal+hf_stat+sys_zHD",
            "processed_dir": str(root / "data/processed/pantheon_plus_shoes_ladder"),
        }
    )
    id_levels = np.asarray(ds.idsurvey_levels, dtype=int).reshape(-1)

    # Union with Pantheon+ cosmology IDSURVEY levels (so the override file can also be used for the
    # main SN-only Pantheon+ probe).
    pp_path = Path(args.pantheon_plus_cosmology_npz).expanduser()
    if not pp_path.is_absolute():
        pp_path = (root / pp_path).resolve()
    if pp_path.exists():
        pp = np.load(pp_path)
        if "idsurvey" in pp:
            pp_levels = np.unique(np.asarray(pp["idsurvey"], dtype=int))
            id_levels = np.unique(np.concatenate([id_levels, pp_levels], axis=0))

    overrides, meta = derive_sigma_overrides(
        cov=cov,
        labels=labels,
        idsurvey_levels=id_levels,
        include_survey_offsets=bool(args.include_survey_offsets),
        include_cal_survey_offsets=bool(args.include_cal_survey_offsets),
        include_hf_survey_offsets=bool(args.include_hf_survey_offsets),
        scale=float(args.scale),
        fallback_sigma_mag=float(args.fallback_sigma_mag),
    )

    out_path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(overrides)} sigma overrides to {out_path}")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
