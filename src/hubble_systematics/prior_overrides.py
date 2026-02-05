from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import json


def load_sigma_overrides(prior_cfg: dict[str, Any]) -> dict[str, float]:
    """
    Load optional per-parameter sigma overrides from:

    - prior_cfg["sigma_overrides"] (mapping param_name -> sigma)
    - prior_cfg["sigma_overrides_path"] (JSON mapping param_name -> sigma)

    Inline overrides take precedence over file overrides.
    """
    out: dict[str, float] = {}

    path = prior_cfg.get("sigma_overrides_path")
    if path:
        out.update(_load_sigma_overrides_file(str(path)))

    inline = prior_cfg.get("sigma_overrides") or {}
    if isinstance(inline, dict):
        for k, v in inline.items():
            if v is None:
                continue
            out[str(k)] = float(v)

    return out


def sigma_from_mapping(mapping: Any, key: int | str) -> float | None:
    """
    Read a sigma value from a mapping that may have int or str keys.
    """
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        raise ValueError("sigma mapping must be a dict")

    if key in mapping:
        v = mapping[key]
        return None if v is None else float(v)
    skey = str(key)
    if skey in mapping:
        v = mapping[skey]
        return None if v is None else float(v)
    try:
        ikey = int(key)
    except Exception:
        ikey = None
    if ikey is not None and ikey in mapping:
        v = mapping[ikey]
        return None if v is None else float(v)
    return None


@lru_cache(maxsize=32)
def _load_sigma_overrides_file(path_str: str) -> dict[str, float]:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[2] / p).resolve()
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        raise ValueError("sigma_overrides_path must point to a JSON mapping")
    out: dict[str, float] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = float(v)
    return out

