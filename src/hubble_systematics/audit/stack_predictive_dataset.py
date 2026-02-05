from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.joint import Part, stack_parts
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class StackPredictivePart:
    name: str
    dataset: Any
    base_model: dict[str, Any]


@dataclass(frozen=True)
class StackPredictiveDataset:
    parts: list[StackPredictivePart]
    base_level: str
    base_model_cfg: dict[str, Any]
    scope_part: str | None
    mask_full: np.ndarray  # bool mask on the full stacked rows
    diag_sigma_full: np.ndarray  # diag sigma for the full stacked rows
    part_slices: list[tuple[str, slice]]
    z_full: np.ndarray
    is_calibrator_full: np.ndarray
    is_hubble_flow_full: np.ndarray

    @classmethod
    def from_parts(
        cls,
        *,
        parts: list[StackPredictivePart],
        anchor: AnchorLCDM,
        base_level: str,
        base_model_cfg: dict[str, Any],
        scope_part: str | None,
    ) -> "StackPredictiveDataset":
        if not parts:
            raise ValueError("StackPredictiveDataset requires non-empty parts")

        model_cfg = dict(base_model_cfg or {})

        # Precompute diag sigma and row bookkeeping from a baseline build.
        part_slices: list[tuple[str, slice]] = []
        diag_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        cal_chunks: list[np.ndarray] = []
        hf_chunks: list[np.ndarray] = []
        row0 = 0

        for part in parts:
            part_model = _deep_merge_dicts(model_cfg, part.base_model or {})
            lvl = str(part_model.get("ladder_level", base_level))
            y, _, cov, _, _ = part.dataset.build_design(anchor=anchor, ladder_level=lvl, cfg=part_model)
            y = np.asarray(y, dtype=float).reshape(-1)
            n = int(y.size)

            if cov.ndim == 1:
                var = np.asarray(cov, dtype=float).reshape(-1)
                if var.shape != (n,):
                    raise ValueError("Part cov shape mismatch")
            else:
                C = np.asarray(cov, dtype=float)
                if C.shape != (n, n):
                    raise ValueError("Part cov shape mismatch")
                var = np.diag(C)
            if not (np.all(np.isfinite(var)) and np.all(var > 0)):
                raise ValueError("Invalid covariance diagonal in stack part")
            diag_chunks.append(np.sqrt(var))

            # z bookkeeping: used only for split sizing; prefer actual z if present.
            z = None
            if hasattr(part.dataset, "get_column"):
                try:
                    z = np.asarray(part.dataset.get_column("z"), dtype=float).reshape(-1)
                except Exception:
                    z = None
            if z is None and hasattr(part.dataset, "z"):
                try:
                    z = np.asarray(getattr(part.dataset, "z"), dtype=float).reshape(-1)
                except Exception:
                    z = None
            if z is None or z.shape != (n,):
                z = np.full(n, np.nan, dtype=float)
            z_chunks.append(z)

            cal = np.zeros(n, dtype=bool)
            if hasattr(part.dataset, "is_calibrator"):
                try:
                    v = np.asarray(getattr(part.dataset, "is_calibrator"), dtype=bool)
                    if v.shape == (n,):
                        cal = v
                except Exception:
                    pass
            cal_chunks.append(cal)

            hf = np.zeros(n, dtype=bool)
            if hasattr(part.dataset, "is_hubble_flow"):
                try:
                    v = np.asarray(getattr(part.dataset, "is_hubble_flow"), dtype=bool)
                    if v.shape == (n,):
                        hf = v
                except Exception:
                    pass
            hf_chunks.append(hf)

            part_slices.append((str(part.name), slice(row0, row0 + n)))
            row0 += n

        n_total = row0
        return cls(
            parts=list(parts),
            base_level=str(base_level),
            base_model_cfg=model_cfg,
            scope_part=str(scope_part) if scope_part is not None else None,
            mask_full=np.ones(n_total, dtype=bool),
            diag_sigma_full=np.concatenate(diag_chunks, axis=0),
            part_slices=part_slices,
            z_full=np.concatenate(z_chunks, axis=0),
            is_calibrator_full=np.concatenate(cal_chunks, axis=0),
            is_hubble_flow_full=np.concatenate(hf_chunks, axis=0),
        )

    @property
    def z(self) -> np.ndarray:
        return self.z_full[self.mask_full]

    @property
    def is_calibrator(self) -> np.ndarray:
        return self.is_calibrator_full[self.mask_full]

    @property
    def is_hubble_flow(self) -> np.ndarray:
        return self.is_hubble_flow_full[self.mask_full]

    def part_mask_full(self, part_name: str) -> np.ndarray:
        part_name = str(part_name)
        out = np.zeros_like(self.mask_full)
        for nm, sl in self.part_slices:
            if nm == part_name:
                out[sl] = True
        return out

    def subset_mask(self, mask: np.ndarray) -> "StackPredictiveDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == self.mask_full.shape:
            new_mask = mask
        else:
            # Allow a mask defined on the current subset rows.
            idx = np.where(self.mask_full)[0]
            if mask.shape != (idx.size,):
                raise ValueError("mask shape mismatch")
            new_mask = np.zeros_like(self.mask_full)
            new_mask[idx] = mask
        return StackPredictiveDataset(
            parts=self.parts,
            base_level=self.base_level,
            base_model_cfg=self.base_model_cfg,
            scope_part=self.scope_part,
            mask_full=new_mask,
            diag_sigma_full=self.diag_sigma_full,
            part_slices=self.part_slices,
            z_full=self.z_full,
            is_calibrator_full=self.is_calibrator_full,
            is_hubble_flow_full=self.is_hubble_flow_full,
        )

    def diag_sigma(self) -> np.ndarray:
        sig = np.asarray(self.diag_sigma_full[self.mask_full], dtype=float)
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0)):
            raise ValueError("Invalid diag sigma")
        return sig

    def get_column(self, name: str) -> np.ndarray:
        name = str(name)
        out = np.full(self.mask_full.shape[0], np.nan, dtype=float)
        for part, (nm, sl) in zip(self.parts, self.part_slices, strict=True):
            if self.scope_part is not None and nm != self.scope_part:
                continue
            ds = part.dataset
            if not hasattr(ds, "get_column"):
                continue
            try:
                v = np.asarray(ds.get_column(name), dtype=float).reshape(-1)
            except Exception:
                continue
            if v.shape != (sl.stop - sl.start,):
                continue
            out[sl] = v
        return out[self.mask_full]

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        cfg = dict(cfg or {})
        stack_overrides = cfg.get("stack_overrides", {}) or {}
        if not isinstance(stack_overrides, dict):
            raise ValueError("model.stack_overrides must be a mapping")

        lvl_default = str(ladder_level)
        parts: list[Part] = []
        for part in self.parts:
            part_model = _deep_merge_dicts(cfg, part.base_model or {})
            override = stack_overrides.get(str(part.name), {}) or {}
            if override:
                part_model = _deep_merge_dicts(part_model, override)
            lvl = str(part_model.get("ladder_level", lvl_default))
            y, y0, cov, X, prior = part.dataset.build_design(anchor=anchor, ladder_level=lvl, cfg=part_model)
            parts.append(Part(y=y, y0=y0, cov=cov, X=X, prior=prior))

        stacked = stack_parts(parts)
        prior = apply_shared_scale_prior(stacked.prior, model_cfg=cfg)

        idx = np.where(self.mask_full)[0]
        y = np.asarray(stacked.y, dtype=float).reshape(-1)[idx]
        y0 = np.asarray(stacked.y0, dtype=float).reshape(-1)[idx]
        X = np.asarray(stacked.X, dtype=float)[idx, :]
        cov = np.asarray(stacked.cov, dtype=float)
        if cov.ndim == 1:
            cov = cov[idx]
        else:
            cov = cov[np.ix_(idx, idx)]

        return y, y0, cov, X, prior


def _deep_merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

