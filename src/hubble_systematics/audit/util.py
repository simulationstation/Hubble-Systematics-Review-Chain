from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def make_cut_grid(dataset, cut_var: str, scan_cfg: dict[str, Any]) -> np.ndarray:
    cut_min = scan_cfg.get("cut_min")
    cut_max = scan_cfg.get("cut_max")
    n_cuts = int(scan_cfg.get("n_cuts", 12))

    vals = dataset.get_column(cut_var)
    if cut_min is None:
        cut_min = float(np.nanmin(vals))
    if cut_max is None:
        cut_max = float(np.nanmax(vals))
    # Keep cuts inside the available support to avoid empty subsets at endpoints.
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    cut_min = max(float(cut_min), vmin)
    cut_max = min(float(cut_max), vmax)

    if n_cuts < 2:
        raise ValueError("n_cuts must be >=2")
    return np.linspace(float(cut_min), float(cut_max), n_cuts)
