from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_mapping(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path} must be a JSON mapping")
    out: dict[str, float] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = float(v)
    return out


def _merge(*, base: dict[str, float], other: dict[str, float], mode: str) -> dict[str, float]:
    out = dict(base)
    if mode == "override":
        out.update(other)
        return out
    if mode == "min":
        for k, v in other.items():
            if k in out:
                out[k] = float(min(float(out[k]), float(v)))
            else:
                out[k] = float(v)
        return out
    if mode == "max":
        for k, v in other.items():
            if k in out:
                out[k] = float(max(float(out[k]), float(v)))
            else:
                out[k] = float(v)
        return out
    raise ValueError(f"Unsupported merge mode: {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path for merged sigma overrides (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--mode",
        choices=["override", "min", "max"],
        default="override",
        help="Merge mode for overlapping keys. 'override' means later files win; 'min'/'max' take min/max sigma per key.",
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON mapping(s) to merge, in order (relative to repo root unless absolute).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    merged: dict[str, float] = {}
    resolved_inputs: list[str] = []
    n_keys_by_input: list[dict[str, Any]] = []

    for raw in list(args.inputs):
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        m = _load_mapping(p)
        merged = _merge(base=merged, other=m, mode=str(args.mode))
        resolved_inputs.append(str(p))
        n_keys_by_input.append({"path": str(p), "n_keys": int(len(m))})

    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    meta = {
        "created_by": "scripts/merge_sigma_overrides.py",
        "mode": str(args.mode),
        "inputs": resolved_inputs,
        "n_inputs": int(len(resolved_inputs)),
        "n_keys_by_input": n_keys_by_input,
        "n_keys_out": int(len(merged)),
        "notes": [
            "This file is a mechanical merge of one or more per-parameter sigma-override mappings.",
            "It is intended for audit experiments that combine multiple calibration-gate sources.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"Wrote merged overrides to {out_path} (n={len(merged)})")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()

