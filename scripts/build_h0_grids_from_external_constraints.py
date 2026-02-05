from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _pdf_from_constraint(*, H0: np.ndarray, mean: float, sigma: float, space: str) -> np.ndarray:
    space = str(space).lower()
    if space == "linear":
        z = (H0 - float(mean)) / float(sigma)
        return np.exp(-0.5 * z * z)
    if space == "ln":
        # The upstream external-constraints file records sigma in linear H0 units.
        # For a Gaussian in ln(H0) we use the small-error approximation sigma_ln ≈ sigma/mean.
        mu = float(np.log(mean))
        sigma_ln = float(sigma / mean)
        x = np.log(H0)
        z = (x - mu) / sigma_ln
        # Convert a density in x=ln(H0) to a density in H0: p(H0)=p(x)/H0.
        return np.exp(-0.5 * z * z) / H0
    raise ValueError(f"Unsupported space: {space!r}")


def _grid_for_constraint(*, mean: float, sigma: float, space: str, nsig: float, n: int) -> np.ndarray:
    space = str(space).lower()
    if space == "linear":
        lo = float(mean - nsig * sigma)
        hi = float(mean + nsig * sigma)
        lo = max(lo, 1e-6)
        return np.linspace(lo, hi, int(n), dtype=float)
    if space == "ln":
        mu = float(np.log(mean))
        sigma_ln = float(sigma / mean)
        lo = mu - nsig * sigma_ln
        hi = mu + nsig * sigma_ln
        return np.exp(np.linspace(lo, hi, int(n), dtype=float))
    raise ValueError(f"Unsupported space: {space!r}")


def build_h0_grids(*, constraints_path: Path, out_dir: Path, nsig: float, n_grid: int) -> list[Path]:
    d = json.loads(constraints_path.read_text())
    constraints = d.get("constraints") or []
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for item in constraints:
        if not isinstance(item, dict):
            continue
        if str(item.get("quantity", "")).lower() != "h0":
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        mean = float(item["mean"])
        sigma = float(item["sigma"])
        space = str(item.get("space", "ln")).lower()

        H0_grid = _grid_for_constraint(mean=mean, sigma=sigma, space=space, nsig=nsig, n=n_grid)
        posterior = _pdf_from_constraint(H0=H0_grid, mean=mean, sigma=sigma, space=space)

        out_json = out_dir / f"{label}.json"
        out_meta = out_dir / f"{label}.meta.json"

        out_json.write_text(
            json.dumps(
                {
                    "H0_grid": [float(x) for x in H0_grid.tolist()],
                    "posterior": [float(x) for x in posterior.tolist()],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        meta: dict[str, Any] = {
            "label": label,
            "quantity": "H0",
            "mean": mean,
            "sigma": sigma,
            "space": space,
            "grid": {"nsig": float(nsig), "n_grid": int(n_grid)},
            "notes": [
                "This is an approximately-Gaussian external constraint encoded as an H0_grid posterior for use by probe.name=h0_grid.",
                "For space='ln' we interpret the published symmetric sigma in linear H0 units and use sigma_ln≈sigma/mean.",
            ],
            "reference": item.get("reference", {}),
            "source_constraints_file": str(constraints_path),
            "created_by": "scripts/build_h0_grids_from_external_constraints.py",
        }
        out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

        written.append(out_json)
        written.append(out_meta)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--constraints",
        default="data/processed/external_constraints/h0_constraints_2026-02-05.json",
        help="Path to the external constraints JSON.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed/h0_grids",
        help="Output directory for h0_grid JSONs.",
    )
    ap.add_argument("--nsig", type=float, default=6.0, help="Grid span in +/- nsig.")
    ap.add_argument("--n-grid", type=int, default=401, help="Number of grid points.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    constraints_path = Path(args.constraints).expanduser()
    if not constraints_path.is_absolute():
        constraints_path = (root / constraints_path).resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    written = build_h0_grids(constraints_path=constraints_path, out_dir=out_dir, nsig=float(args.nsig), n_grid=int(args.n_grid))
    print(f"Wrote {len(written)} files to {out_dir}")


if __name__ == "__main__":
    main()

