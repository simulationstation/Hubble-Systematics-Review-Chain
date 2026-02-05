from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineFitResult:
    ladder_level: str
    n: int
    param_names: list[str]
    mean: list[float]
    cov: list[list[float]]
    chi2: float
    dof: int
    logdet_cov: float
    log_evidence: float | None = None


@dataclass(frozen=True)
class CutScanResult:
    cut_var: str
    cuts: list[float]
    rows: list[dict[str, Any]]
    cut_mode: str = "leq"

    def to_jsonable(self) -> dict[str, Any]:
        return {"cut_var": self.cut_var, "cuts": self.cuts, "rows": self.rows, "cut_mode": self.cut_mode}


@dataclass(frozen=True)
class DriftNullResult:
    cut_var: str
    cuts: list[float]
    param_name: str
    observed: dict[str, float]
    mc: dict[str, list[float]]
    p_values: dict[str, float]
    cut_mode: str = "leq"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "cut_var": self.cut_var,
            "cuts": self.cuts,
            "param_name": self.param_name,
            "observed": self.observed,
            "mc": self.mc,
            "p_values": self.p_values,
            "cut_mode": self.cut_mode,
        }
