from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


@dataclass(frozen=True)
class KcorSummary:
    key: str
    path: str
    n: int
    median_zpoff: float
    robust_sigma_zpoff: float
    max_abs_zpoff: float


def _kcor_key_from_filename(name: str) -> str:
    name = str(name)
    if name.startswith("kcor_"):
        name = name[len("kcor_") :]
    if name.endswith(".fits"):
        name = name[: -len(".fits")]
    # Drop a trailing version suffix (e.g. _v6_1).
    if "_v" in name:
        head, tail = name.rsplit("_v", 1)
        # Require a numeric-looking tail (e.g. 6_1) to avoid stripping "ADAMRECAL".
        if all((c.isdigit() or c == "_") for c in tail):
            name = head
    return name


def _summarize_kcor_zpoff(*, fits_path: Path, clip_abs: float) -> KcorSummary | None:
    with fits.open(fits_path) as hdul:
        if "ZPoff" not in hdul:
            return None
        t = hdul["ZPoff"].data
        zp = np.asarray(t["ZPoff(Primary)"], dtype=float).reshape(-1)
        zp = zp[np.isfinite(zp)]
        if clip_abs > 0:
            zp = zp[np.abs(zp) < float(clip_abs)]
        if zp.size == 0:
            return None

        med = float(np.median(zp))
        mad = float(np.median(np.abs(zp - med)))
        robust_sigma = float(1.4826 * mad)
        return KcorSummary(
            key=_kcor_key_from_filename(fits_path.name),
            path=str(fits_path),
            n=int(zp.size),
            median_zpoff=med,
            robust_sigma_zpoff=robust_sigma,
            max_abs_zpoff=float(np.max(np.abs(zp))),
        )


def derive_sigma_overrides(
    *,
    kcor_summaries: dict[str, KcorSummary],
    idsurvey_levels: np.ndarray,
    pkmjd_edges: list[float],
    include_survey_time_bins: bool,
    include_time_bins: bool,
    include_survey_offsets: bool,
    include_cal_survey_offsets: bool,
    include_hf_survey_offsets: bool,
    fallback_sigma_mag: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    idsurvey_levels = np.asarray(idsurvey_levels, dtype=int).reshape(-1)
    edges = [float(x) for x in pkmjd_edges]
    if len(edges) < 3:
        raise ValueError("pkmjd_edges must have length >=3")
    n_bins = len(edges) - 1

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
        150: "FOUND",
    }

    # Mapping from survey label -> kcor keys (multiple keys mean multiple calibration variants).
    # Notes:
    # - Keys are derived from the SNANA kcor FITS filenames (see _kcor_key_from_filename).
    # - This is a pragmatic mapping for audit priors; it is not a full photometry-level model.
    survey_to_kcor_keys: dict[str, list[str]] = {
        "SDSS": ["SDSS"],
        "SNLS": ["SNLS"],
        "DES": ["DES_3yr", "DES_5yr"],
        "PS1MD": ["PS1"],
        "FOUND": ["Foundation"],
        "CSP": ["CSPDR3", "CSPDR3_supercal"],
        "LOSS1": ["KAIT_Mo", "KAIT_Stahl"],
        "LOSS2": ["KAIT_Mo", "KAIT_Stahl"],
        "SOUSA": ["SWIFT"],
        "CNIa0.02": ["ASASSN"],
        # Legacy Landolt-like low-z collections (heuristic).
        "LOWZ/JRK07": ["Land2"],
        "CFA1": ["Land2"],
        "CFA2": ["Land2"],
        "CFA3S": ["Land2"],
        "CFA3K": ["Land2"],
        "CFA4p2": ["Land2"],
        "CFA4p3": ["Land2"],
    }

    def sigma_from_keys(keys: list[str]) -> tuple[float, dict[str, Any]]:
        keys = [str(k) for k in keys if str(k)]
        medians: list[float] = []
        found: list[str] = []
        missing: list[str] = []
        for k in keys:
            s = kcor_summaries.get(k)
            if s is None:
                missing.append(k)
                continue
            found.append(k)
            medians.append(float(s.median_zpoff))
        if len(medians) >= 2:
            sig = float(np.std(np.asarray(medians, dtype=float), ddof=1))
        else:
            sig = float(fallback_sigma_mag)
        return sig, {"found": found, "missing": missing, "medians": medians}

    overrides: dict[str, float] = {}
    computed: dict[str, Any] = {}

    # Per-survey sigmas from variant spread.
    survey_sigma: dict[int, float] = {}
    for sid in idsurvey_levels.tolist():
        sid = int(sid)
        sname = idsurvey_to_name.get(sid, f"IDSURVEY_{sid}")
        keys = survey_to_kcor_keys.get(sname, [])
        sig, info = sigma_from_keys(keys)
        survey_sigma[sid] = float(sig)
        computed[str(sid)] = {"name": sname, "sigma_mag": float(sig), "kcor": info}

        if include_survey_offsets:
            overrides[f"survey_offset_{sid}"] = float(sig)
        if include_cal_survey_offsets:
            overrides[f"cal_survey_offset_{sid}"] = float(sig)
        if include_hf_survey_offsets:
            overrides[f"hf_survey_offset_{sid}"] = float(sig)

        if include_survey_time_bins:
            for k in range(1, n_bins):
                overrides[f"survey_pkmjd_bin_offset_{sid}_{k}"] = float(sig)

    # Global time-bin offsets (survey-agnostic): set to the median survey sigma.
    if include_time_bins:
        sig_global = float(np.median(list(survey_sigma.values()))) if survey_sigma else float(fallback_sigma_mag)
        for k in range(1, n_bins):
            overrides[f"pkmjd_bin_offset_{k}"] = float(sig_global)

    meta: dict[str, Any] = {
        "created_by": "scripts/derive_pantheon_shoes_kcor_variant_priors.py",
        "notes": [
            "This file encodes per-parameter sigma overrides derived from the spread of SNANA_kcor calibration variants (Pantheon+ DataRelease 2_CALIBRATION/SNANA_kcor).",
            "We summarize each kcor FITS by the median ZPoff(Primary) across filters (after optional clipping) and use the stddev across variants as a heuristic calibration-uncertainty scale per survey.",
            "This is an external calibration metadata proxy for audit priors; it is not a full photometry-level forward model.",
        ],
        "pkmjd_edges": edges,
        "kcor_keys": sorted(kcor_summaries.keys()),
        "computed": computed,
    }
    return overrides, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kcor-dir",
        default="data/raw/pantheon_plus_calibration/SNANA_kcor",
        help="Directory containing SNANA kcor FITS files (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--out",
        default="data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_kcor_variants_v1.json",
        help="Output JSON path for sigma_overrides mapping.",
    )
    ap.add_argument("--clip-abs", type=float, default=0.2, help="Clip |ZPoff(Primary)| to this value before summarizing (<=0 disables).")
    ap.add_argument("--fallback-sigma-mag", type=float, default=0.01)
    ap.add_argument(
        "--pkmjd-edges",
        default="44672.6,53410.4,54131.12,54885.0,57269.22,59385.6",
        help="Comma-separated pkmjd bin edges (MJD).",
    )
    ap.add_argument("--include-survey-time-bins", action="store_true", default=True)
    ap.add_argument("--include-time-bins", action="store_true", default=True)
    ap.add_argument("--include-survey-offsets", action="store_true", default=True)
    ap.add_argument("--include-cal-survey-offsets", action="store_true", default=False)
    ap.add_argument("--include-hf-survey-offsets", action="store_true", default=False)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    kcor_dir = Path(args.kcor_dir).expanduser()
    if not kcor_dir.is_absolute():
        kcor_dir = (root / kcor_dir).resolve()
    if not kcor_dir.exists():
        raise FileNotFoundError(kcor_dir)

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    edges = [float(x) for x in str(args.pkmjd_edges).split(",") if str(x).strip()]

    # Summarize all kcor FITS files we can find.
    kcor_summaries: dict[str, KcorSummary] = {}
    for p in sorted(kcor_dir.glob("kcor_*.fits")):
        s = _summarize_kcor_zpoff(fits_path=p, clip_abs=float(args.clip_abs))
        if s is None:
            continue
        kcor_summaries[str(s.key)] = s

    if not kcor_summaries:
        raise ValueError(f"No kcor FITS summaries found under {kcor_dir}")

    # Use the ladder selection to get the relevant idsurvey levels.
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

    overrides, meta = derive_sigma_overrides(
        kcor_summaries=kcor_summaries,
        idsurvey_levels=id_levels,
        pkmjd_edges=edges,
        include_survey_time_bins=bool(args.include_survey_time_bins),
        include_time_bins=bool(args.include_time_bins),
        include_survey_offsets=bool(args.include_survey_offsets),
        include_cal_survey_offsets=bool(args.include_cal_survey_offsets),
        include_hf_survey_offsets=bool(args.include_hf_survey_offsets),
        fallback_sigma_mag=float(args.fallback_sigma_mag),
    )

    out_path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")

    meta_out: dict[str, Any] = {
        "meta": meta,
        "kcor_summaries": {k: s.__dict__ for k, s in sorted(kcor_summaries.items())},
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(overrides)} sigma overrides to {out_path}")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()

