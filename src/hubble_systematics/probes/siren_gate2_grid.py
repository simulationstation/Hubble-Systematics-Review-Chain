from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import re

import numpy as np
from astropy.time import Time

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params


@dataclass(frozen=True)
class SirenGate2GridDataset:
    """Selection-corrected Gate-2 dark-siren H0 grid with per-event metadata.

    This adapter ingests the Gate-2 JSON outputs produced by the Dark_Siren_Ladder_Audit project.

    Key detail (selection correction):
      p(H0 | {d_i}) ∝ Π_i L_i(H0) / alpha(H0)^{N}
    where `alpha(H0)` is the selection factor (provided on the same grid) and N is the number of
    events in the subset being analyzed.

    For compatibility with the rest of this repo's linear-Gaussian audit runner, we summarize each
    subset by the mean and standard deviation of ln(H0) under the *selection-corrected* grid
    posterior, and return a diagonal Gaussian design with a single effective degree of freedom.
    """

    label: str
    H0_grid: np.ndarray  # (n_grid,)
    log_alpha_grid: np.ndarray  # (n_grid,)
    event: np.ndarray  # (n,) str
    logL_H0: np.ndarray  # (n, n_grid)
    pe_analysis_id: np.ndarray  # (n,) int
    pe_analysis: np.ndarray  # (n,) str
    ess_min: np.ndarray  # (n,) float
    n_good_min: np.ndarray  # (n,) float
    event_mjd: np.ndarray  # (n,) float (UTC)
    space: str = "ln"  # "ln" (default) or "linear"

    def __post_init__(self) -> None:
        H0 = np.asarray(self.H0_grid, dtype=float).reshape(-1)
        la = np.asarray(self.log_alpha_grid, dtype=float).reshape(-1)
        if H0.shape != la.shape:
            raise ValueError("H0_grid/log_alpha_grid shape mismatch")
        if not (np.all(np.isfinite(H0)) and np.all(H0 > 0) and np.all(np.diff(H0) > 0)):
            raise ValueError("H0_grid must be finite, positive, and strictly increasing")
        if not np.all(np.isfinite(la)):
            raise ValueError("log_alpha_grid must be finite")

        n = int(np.asarray(self.event).shape[0])
        if np.asarray(self.logL_H0).shape != (n, H0.size):
            raise ValueError("logL_H0 must have shape (n_events, n_grid)")
        for name in [
            "pe_analysis_id",
            "pe_analysis",
            "ess_min",
            "n_good_min",
            "event_mjd",
        ]:
            v = getattr(self, name)
            if np.asarray(v).shape != (n,):
                raise ValueError(f"{name} must have shape (n,)")

    def __len__(self) -> int:
        return int(np.asarray(self.event).shape[0])

    def get_column(self, name: str) -> np.ndarray:
        name = str(name)
        if name == "event_mjd":
            return np.asarray(self.event_mjd, dtype=float)
        if name == "event_year":
            mjd = np.asarray(self.event_mjd, dtype=float)
            out = np.full_like(mjd, np.nan, dtype=float)
            ok = np.isfinite(mjd)
            if np.any(ok):
                t = Time(mjd[ok], format="mjd", scale="utc").to_datetime()
                out[ok] = np.array([dt.year + (dt.timetuple().tm_yday - 1) / 365.25 for dt in t], dtype=float)
            return out
        if name == "pe_analysis_id":
            return np.asarray(self.pe_analysis_id, dtype=int)
        if name == "ess_min":
            return np.asarray(self.ess_min, dtype=float)
        if name == "n_good_min":
            return np.asarray(self.n_good_min, dtype=float)
        raise KeyError(name)

    def subset_mask(self, mask: np.ndarray) -> "SirenGate2GridDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.event.shape:
            raise ValueError("mask shape mismatch")
        return SirenGate2GridDataset(
            label=self.label,
            H0_grid=self.H0_grid,
            log_alpha_grid=self.log_alpha_grid,
            event=self.event[mask],
            logL_H0=self.logL_H0[mask, :],
            pe_analysis_id=self.pe_analysis_id[mask],
            pe_analysis=self.pe_analysis[mask],
            ess_min=self.ess_min[mask],
            n_good_min=self.n_good_min[mask],
            event_mjd=self.event_mjd[mask],
            space=self.space,
        )

    def subset_leq(self, cut_var: str, cut: float) -> "SirenGate2GridDataset":
        v = np.asarray(self.get_column(cut_var), dtype=float)
        return self.subset_mask(np.isnan(v) | (v <= float(cut)))

    def subset_geq(self, cut_var: str, cut: float) -> "SirenGate2GridDataset":
        v = np.asarray(self.get_column(cut_var), dtype=float)
        return self.subset_mask(np.isnan(v) | (v >= float(cut)))

    def log_posterior_rel(self) -> np.ndarray:
        """Unnormalized log posterior on the H0 grid (relative; max-subtracted)."""

        H0 = np.asarray(self.H0_grid, dtype=float).reshape(-1)
        logL = np.asarray(self.logL_H0, dtype=float)
        if logL.ndim != 2 or logL.shape[1] != H0.size:
            raise ValueError("logL_H0 shape mismatch")
        n = int(logL.shape[0])
        if n < 1:
            raise ValueError("Siren subset contains no events")

        logL_sum = np.sum(logL, axis=0)
        logpost = logL_sum - float(n) * np.asarray(self.log_alpha_grid, dtype=float).reshape(-1)
        m = float(np.nanmax(logpost))
        if not np.isfinite(m):
            raise ValueError("Non-finite max in log posterior grid")
        return logpost - m

    def posterior(self) -> np.ndarray:
        """Normalized posterior density p(H0) on the grid (with respect to dH0)."""

        H0 = np.asarray(self.H0_grid, dtype=float).reshape(-1)
        logpost = self.log_posterior_rel()
        p = np.exp(logpost)
        if not (np.all(np.isfinite(p)) and np.any(p > 0)):
            raise ValueError("Invalid posterior weights")
        Z = float(np.trapezoid(p, H0))
        if not (np.isfinite(Z) and Z > 0):
            raise ValueError("Posterior normalization failed")
        return p / Z

    def lnH0_moments(self) -> tuple[float, float]:
        """Mean and SD of ln(H0) under the selection-corrected posterior."""

        H0 = np.asarray(self.H0_grid, dtype=float).reshape(-1)
        p = self.posterior()
        lnH0 = np.log(H0)
        mu = float(np.trapezoid(p * lnH0, H0))
        var = float(np.trapezoid(p * (lnH0 - mu) ** 2, H0))
        return mu, float(np.sqrt(max(var, 0.0)))

    def H0_moments(self) -> tuple[float, float]:
        """Mean and SD of H0 under the selection-corrected posterior."""

        H0 = np.asarray(self.H0_grid, dtype=float).reshape(-1)
        p = self.posterior()
        mu = float(np.trapezoid(p * H0, H0))
        var = float(np.trapezoid(p * (H0 - mu) ** 2, H0))
        return mu, float(np.sqrt(max(var, 0.0)))

    def diag_sigma(self) -> np.ndarray:
        # One effective Gaussian constraint replicated across events, with per-row variance scaled
        # by N so the aggregate weight is unchanged.
        n = len(self)
        if n < 1:
            raise ValueError("Empty siren dataset has no sigma")
        space = str(self.space).lower()
        if space == "ln":
            _, sd = self.lnH0_moments()
            sigma_eff = float(sd)
        elif space == "linear":
            _, sd = self.H0_moments()
            sigma_eff = float(sd)
        else:
            raise ValueError(f"Unsupported siren_gate2_grid.space: {self.space!r}")
        return np.full(n, float(np.sqrt(n) * sigma_eff), dtype=float)

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        _ = ladder_level  # no per-level behavior for this probe
        shared_params = set(shared_scale_params(cfg or {}))

        n = len(self)
        space = str(self.space).lower()
        if space == "ln":
            mu, sd = self.lnH0_moments()
            y_scalar = float(mu)
            y0_scalar = float(np.log(anchor.H0))
            sigma_eff = float(sd)
        elif space == "linear":
            mu, sd = self.H0_moments()
            y_scalar = float(mu)
            y0_scalar = float(anchor.H0)
            sigma_eff = float(sd)
        else:
            raise ValueError(f"Unsupported siren_gate2_grid.space: {self.space!r}")

        y = np.full(n, y_scalar, dtype=float)
        y0 = np.full(n, y0_scalar, dtype=float)
        cov = np.full(n, float(n) * float(sigma_eff**2), dtype=float)

        names: list[str] = []
        X = np.zeros((n, 0), dtype=float)
        if "delta_lnH0" in shared_params:
            if space == "ln":
                X = np.ones((n, 1), dtype=float)
            else:
                X = y0.reshape(-1, 1)
            names = ["delta_lnH0"]

        # No per-probe priors on shared params; runner applies global priors once.
        prior = GaussianPrior.from_sigmas(names, [float("inf")] * len(names), mean=0.0)
        return y, y0, cov, X, prior


_GW_EVENT_RE = re.compile(
    r"^GW(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})_(?P<hh>\d{2})(?P<mi>\d{2})(?P<ss>\d{2})$"
)


def _event_mjd_utc(name: str) -> float:
    m = _GW_EVENT_RE.match(str(name))
    if m is None:
        return float("nan")
    yy = int(m.group("yy"))
    year = 2000 + yy if yy < 90 else 1900 + yy
    dt = datetime(
        year=year,
        month=int(m.group("mm")),
        day=int(m.group("dd")),
        hour=int(m.group("hh")),
        minute=int(m.group("mi")),
        second=int(m.group("ss")),
    )
    return float(Time(dt, scale="utc").mjd)


def load_siren_gate2_grid_dataset(probe_cfg: dict[str, Any]) -> SirenGate2GridDataset:
    data_path = Path(probe_cfg["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[3] / data_path).resolve()

    label = str(probe_cfg.get("label", data_path.name))
    d = json.loads(data_path.read_text())

    H0_grid = np.asarray(d["H0_grid"], dtype=float)
    if "log_alpha_grid" in d:
        log_alpha_grid = np.asarray(d["log_alpha_grid"], dtype=float)
    elif "selection_alpha_grid" in d:
        alpha = np.asarray(d["selection_alpha_grid"], dtype=float)
        if not (np.all(np.isfinite(alpha)) and np.all(alpha > 0)):
            raise ValueError("selection_alpha_grid must be finite and >0")
        log_alpha_grid = np.log(alpha)
    else:
        raise ValueError("Gate-2 siren JSON missing log_alpha_grid / selection_alpha_grid")

    events = list(d.get("events") or [])
    if not events:
        raise ValueError("Gate-2 siren JSON contains no events")

    ev_names: list[str] = []
    ess: list[float] = []
    ngood: list[float] = []
    pe_analysis: list[str] = []
    mjd: list[float] = []
    logL_rows: list[np.ndarray] = []

    for ev in events:
        name = str(ev.get("event"))
        logL = np.asarray(ev.get("logL_H0"), dtype=float).reshape(-1)
        if logL.shape != H0_grid.shape:
            raise ValueError(f"Event {name}: logL_H0 shape mismatch vs H0_grid")
        ev_names.append(name)
        ess.append(float(ev.get("ess_min", float("nan"))))
        ngood.append(float(ev.get("n_good_min", float("nan"))))
        pe_analysis.append(str(ev.get("pe_analysis", "")))
        mjd.append(_event_mjd_utc(name))
        logL_rows.append(logL)

    # Stable integer levels for PE analysis labels.
    uniq = sorted(set(pe_analysis))
    mapping = {u: i for i, u in enumerate(uniq)}
    pe_id = np.array([mapping.get(x, -1) for x in pe_analysis], dtype=int)

    space = str(probe_cfg.get("space", "ln"))

    return SirenGate2GridDataset(
        label=label,
        H0_grid=H0_grid,
        log_alpha_grid=log_alpha_grid,
        event=np.asarray(ev_names, dtype="U"),
        logL_H0=np.stack(logL_rows, axis=0),
        pe_analysis_id=pe_id,
        pe_analysis=np.asarray(pe_analysis, dtype="U"),
        ess_min=np.asarray(ess, dtype=float),
        n_good_min=np.asarray(ngood, dtype=float),
        event_mjd=np.asarray(mjd, dtype=float),
        space=space,
    )
