from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve_param(entry: dict[str, Any]) -> str:
    if "param" in entry and entry["param"] is not None:
        return str(entry["param"])

    mech = entry.get("mechanism")
    if mech is None:
        raise ValueError("Constraint entry must include either 'param' or 'mechanism'")
    mech = str(mech)

    # Normalize common aliases.
    aliases = {
        "idsurvey_offset": "survey_offset_mag",
        "survey_offset": "survey_offset_mag",
        "cal_survey_offset": "cal_survey_offset_mag",
        "hf_survey_offset": "hf_survey_offset_mag",
        "pkmjd_bin_offset": "pkmjd_bin_offset_mag",
        "survey_pkmjd_bin_offset": "survey_pkmjd_bin_offset_mag",
        "calibrator_offset": "calibrator_offset_mag",
        "pkmjd_err_linear": "pkmjd_err_linear_mag",
        "mwebv_linear": "mwebv_linear_mag",
        "c_linear": "c_linear_mag",
        "x1_linear": "x1_linear_mag",
        "biascor_m_b_linear": "biascor_m_b_linear_mag",
        "host_mass_step": "host_mass_step_mag",
    }
    mech = aliases.get(mech, mech)

    if mech in {"calibrator_offset_mag", "pkmjd_err_linear_mag", "mwebv_linear_mag", "c_linear_mag", "x1_linear_mag", "biascor_m_b_linear_mag", "host_mass_step_mag"}:
        return mech

    if mech == "survey_offset_mag":
        sid = entry.get("idsurvey")
        if sid is None:
            raise ValueError("survey_offset_mag requires 'idsurvey'")
        return f"survey_offset_{int(sid)}"

    if mech == "cal_survey_offset_mag":
        sid = entry.get("idsurvey")
        if sid is None:
            raise ValueError("cal_survey_offset_mag requires 'idsurvey'")
        return f"cal_survey_offset_{int(sid)}"

    if mech == "hf_survey_offset_mag":
        sid = entry.get("idsurvey")
        if sid is None:
            raise ValueError("hf_survey_offset_mag requires 'idsurvey'")
        return f"hf_survey_offset_{int(sid)}"

    if mech == "pkmjd_bin_offset_mag":
        k = entry.get("bin")
        if k is None:
            raise ValueError("pkmjd_bin_offset_mag requires 'bin'")
        return f"pkmjd_bin_offset_{int(k)}"

    if mech == "survey_pkmjd_bin_offset_mag":
        sid = entry.get("idsurvey")
        k = entry.get("bin")
        if sid is None or k is None:
            raise ValueError("survey_pkmjd_bin_offset_mag requires 'idsurvey' and 'bin'")
        return f"survey_pkmjd_bin_offset_{int(sid)}_{int(k)}"

    raise ValueError(f"Unsupported mechanism for external constraints: {mech!r}")


def _resolve_sigma(entry: dict[str, Any]) -> float:
    for key in ("sigma", "sigma_mag", "sigma_ln"):
        if key in entry and entry[key] is not None:
            v = float(entry[key])
            if not (v > 0.0):
                raise ValueError(f"Invalid sigma in entry (key={key}): {v}")
            return v
    raise ValueError("Constraint entry missing sigma (expected one of: sigma, sigma_mag, sigma_ln)")


def _load_constraints(path: Path) -> list[dict[str, Any]]:
    d = json.loads(path.read_text())
    if isinstance(d, dict) and "constraints" in d:
        c = d.get("constraints") or []
        if not isinstance(c, list):
            raise ValueError("constraints must be a list")
        out = []
        for x in c:
            if not isinstance(x, dict):
                raise ValueError("constraints entries must be objects")
            out.append(dict(x))
        return out
    if isinstance(d, list):
        out = []
        for x in d:
            if not isinstance(x, dict):
                raise ValueError("list entries must be objects")
            out.append(dict(x))
        return out
    if isinstance(d, dict):
        # Backwards-compat: allow plain mapping param -> sigma.
        out = []
        for k, v in d.items():
            if v is None:
                continue
            out.append({"param": str(k), "sigma": float(v)})
        return out
    raise ValueError("Unsupported external constraint file format (expected {constraints:[...]}, [...], or {param:sigma,...})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile external calibration constraints into a sigma_overrides JSON mapping."
    )
    ap.add_argument("--out", required=True, help="Output sigma-overrides JSON path (relative to repo root unless absolute).")
    ap.add_argument("inputs", nargs="+", help="Input constraint JSON file(s) (relative to repo root unless absolute).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    merged: dict[str, float] = {}
    sources: list[dict[str, Any]] = []
    n_entries = 0
    for raw in list(args.inputs):
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        entries = _load_constraints(p)
        sources.append({"path": str(p), "n_entries": int(len(entries))})
        n_entries += int(len(entries))
        for ent in entries:
            param = _resolve_param(ent)
            sigma = _resolve_sigma(ent)
            # Later inputs override earlier ones for the same param.
            merged[str(param)] = float(sigma)

    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    meta = {
        "created_by": "scripts/build_sigma_overrides_from_external_constraints.py",
        "inputs": sources,
        "n_entries_total": int(n_entries),
        "n_keys_out": int(len(merged)),
        "notes": [
            "This file is a compiled per-parameter sigma-override mapping intended to be referenced via",
            "prior_cfg.sigma_overrides_path in audit configs.",
            "Entries may specify params directly, or via mechanism+selector fields (idsurvey/bin).",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"Wrote: {out_path} (n={len(merged)})")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()

