#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _constraint_for_key(*, key: str, sigma_mag: float, source: str) -> dict[str, Any]:
    key = str(key)
    sigma_mag = float(sigma_mag)
    if not (sigma_mag > 0.0):
        raise ValueError(f"Invalid sigma for {key!r}: {sigma_mag}")

    # Common structured patterns.
    if key == "calibrator_offset_mag":
        return {"mechanism": "calibrator_offset_mag", "sigma_mag": sigma_mag, "source": source}

    if key.startswith("survey_offset_"):
        sid = int(key.removeprefix("survey_offset_"))
        return {"mechanism": "survey_offset_mag", "idsurvey": sid, "sigma_mag": sigma_mag, "source": source}
    if key.startswith("cal_survey_offset_"):
        sid = int(key.removeprefix("cal_survey_offset_"))
        return {"mechanism": "cal_survey_offset_mag", "idsurvey": sid, "sigma_mag": sigma_mag, "source": source}
    if key.startswith("hf_survey_offset_"):
        sid = int(key.removeprefix("hf_survey_offset_"))
        return {"mechanism": "hf_survey_offset_mag", "idsurvey": sid, "sigma_mag": sigma_mag, "source": source}

    if key.startswith("pkmjd_bin_offset_"):
        k = int(key.removeprefix("pkmjd_bin_offset_"))
        return {"mechanism": "pkmjd_bin_offset_mag", "bin": k, "sigma_mag": sigma_mag, "source": source}
    if key.startswith("survey_pkmjd_bin_offset_"):
        tail = key.removeprefix("survey_pkmjd_bin_offset_")
        parts = tail.split("_")
        if len(parts) < 2:
            raise ValueError(f"Bad survey_pkmjd_bin_offset key: {key!r}")
        sid = int(parts[0])
        k = int(parts[1])
        return {"mechanism": "survey_pkmjd_bin_offset_mag", "idsurvey": sid, "bin": k, "sigma_mag": sigma_mag, "source": source}

    # Scalar metadata proxies.
    scalar_mechs = {
        "pkmjd_err_linear_mag",
        "mwebv_linear_mag",
        "c_linear_mag",
        "x1_linear_mag",
        "biascor_m_b_linear_mag",
        "host_mass_step_mag",
    }
    if key in scalar_mechs:
        return {"mechanism": key, "sigma_mag": sigma_mag, "source": source}

    # Fallback: constrain the param name directly.
    return {"param": key, "sigma_mag": sigma_mag, "source": source}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export a sigma_overrides mapping (param->sigma) to the external-constraints list format "
            "used by scripts/build_sigma_overrides_from_external_constraints.py."
        )
    )
    ap.add_argument("sigma_overrides", help="Input sigma_overrides JSON mapping (relative to repo root unless absolute).")
    ap.add_argument(
        "--out",
        required=True,
        help="Output external-constraints JSON (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Optional source label attached to each constraint (default: input file path).",
    )
    ap.add_argument(
        "--notes",
        default=None,
        help="Optional freeform note string to include at top-level.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_path = Path(args.sigma_overrides).expanduser()
    if not in_path.is_absolute():
        in_path = (root / in_path).resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = str(args.source) if args.source is not None else str(in_path)

    mapping = json.loads(in_path.read_text())
    if not isinstance(mapping, dict):
        raise ValueError("Input must be a JSON mapping param_name -> sigma")

    constraints: list[dict[str, Any]] = []
    for k in sorted(mapping.keys()):
        v = mapping[k]
        if v is None:
            continue
        constraints.append(_constraint_for_key(key=str(k), sigma_mag=float(v), source=src))

    payload: dict[str, Any] = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sigma_overrides": str(in_path),
        "constraints": constraints,
        "notes": [
            "Auto-exported from a sigma_overrides mapping. Edit/replace entries with real external calibration products as they become available.",
        ],
    }
    if args.notes:
        payload["notes"].append(str(args.notes))

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(constraints)} constraints to {out_path}")


if __name__ == "__main__":
    main()

