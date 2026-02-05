from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.audit.split_null import SplitNullResult, run_split_null_mc
from hubble_systematics.anchors import AnchorLCDM


@dataclass(frozen=True)
class GroupSplitNullResult:
    group_var: str
    group_values: list[int]
    results: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "group_var": self.group_var,
            "group_values": self.group_values,
            "results": self.results,
        }


def run_group_split_null_mc(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    group_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> GroupSplitNullResult:
    group_var = str(group_cfg.get("group_var", "idsurvey"))
    group_on = str(group_cfg.get("group_on", "hf")).lower()
    min_hf_per_group = int(group_cfg.get("min_hf_per_group", 20))
    max_groups = group_cfg.get("max_groups")
    max_groups = int(max_groups) if max_groups is not None else None
    include_calibrators = bool(group_cfg.get("include_calibrators", True))

    if not hasattr(dataset, "subset_mask"):
        raise ValueError("dataset must support subset_mask for group_split_null")

    g = np.asarray(dataset.get_column(group_var))
    if group_on == "hf":
        if not hasattr(dataset, "is_hubble_flow"):
            raise ValueError("group_on=hf requires dataset.is_hubble_flow")
        base = np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool)
    elif group_on == "cal":
        if not hasattr(dataset, "is_calibrator"):
            raise ValueError("group_on=cal requires dataset.is_calibrator")
        base = np.asarray(getattr(dataset, "is_calibrator"), dtype=bool)
    elif group_on == "all":
        base = np.ones(g.shape, dtype=bool)
    else:
        raise ValueError(f"Unsupported group_on: {group_on}")

    if include_calibrators:
        if not hasattr(dataset, "is_calibrator"):
            raise ValueError("include_calibrators=true requires dataset.is_calibrator")
        cal = np.asarray(getattr(dataset, "is_calibrator"), dtype=bool)
    else:
        cal = np.zeros(g.shape, dtype=bool)

    # Candidate groups ranked by HF counts (or group_on selection counts).
    uniq = np.unique(g[base].astype(int))
    counts = {int(u): int(np.sum(base & (g.astype(int) == int(u)))) for u in uniq}
    keep = [u for u in uniq if counts[int(u)] >= min_hf_per_group]
    keep = sorted(keep, key=lambda u: -counts[int(u)])
    if max_groups is not None:
        keep = keep[:max_groups]

    split_null_cfg = dict(group_cfg.get("split_null", {}) or {})
    if not split_null_cfg:
        raise ValueError("group_split_null requires group_cfg.split_null to be set (like the regular split_null config)")

    results: dict[str, Any] = {}
    group_values: list[int] = []
    for u in keep:
        u = int(u)
        group_values.append(u)
        mask = cal | (base & (g.astype(int) == u))
        sub = dataset.subset_mask(mask)
        res = run_split_null_mc(
            dataset=sub,
            anchor=anchor,
            ladder_level=ladder_level,
            model_cfg=model_cfg,
            split_null_cfg=split_null_cfg,
            rng=rng,
        )
        results[str(u)] = {
            "n_total": int(np.sum(mask)),
            "n_group": int(counts[u]),
            "result": res.to_jsonable(),
        }

    return GroupSplitNullResult(group_var=group_var, group_values=group_values, results=results)

