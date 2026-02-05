from __future__ import annotations

from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM


def mechanism_ledger(*, anchor: AnchorLCDM, param_names: list[str], mean: np.ndarray, cov: np.ndarray) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    out: dict[str, Any] = {"anchor": {"H0": float(anchor.H0), "rd_Mpc": float(anchor.rd_Mpc)}}

    def add_param(name: str) -> None:
        if name not in param_names:
            return
        j = param_names.index(name)
        mu = float(mean[j])
        sd = float(np.sqrt(cov[j, j])) if cov.size else float("nan")
        out[name] = {"mean": mu, "sd": sd}

    def add_ln_param(name: str, base: float, label: str) -> None:
        if name not in param_names:
            return
        j = param_names.index(name)
        mu = float(mean[j])
        sd = float(np.sqrt(cov[j, j])) if cov.size else float("nan")
        out[name] = {"mean": mu, "sd": sd}
        out[label] = {"mean": float(base * np.exp(mu)), "sd_lin": float(base * np.exp(mu) * sd)}

    add_ln_param("delta_lnH0", anchor.H0, "H0_eff")
    add_ln_param("delta_lnrd", anchor.rd_Mpc, "rd_eff")

    # Equivalent distance-modulus shift for H0-like scaling (exact for FRW distances):
    if "delta_lnH0" in param_names:
        j = param_names.index("delta_lnH0")
        mu = float(mean[j])
        sd = float(np.sqrt(cov[j, j])) if cov.size else float("nan")
        dmu = -(5.0 / np.log(10.0)) * mu
        dmu_sd = abs(5.0 / np.log(10.0)) * sd
        out["delta_mu_equiv_mag"] = {"mean": float(dmu), "sd": float(dmu_sd)}

    # For interpretability: if a calibrator↔HF offset exists, convert to an "equivalent" H0 scaling.
    # This is not a statement of identifiability; it's a unit conversion for the gap accounting.
    add_param("calibrator_offset_mag")
    if "calibrator_offset_mag" in out:
        cal = out["calibrator_offset_mag"]
        dmu = float(cal.get("mean"))
        dmu_sd = float(cal.get("sd"))
        dlnH0 = -(np.log(10.0) / 5.0) * dmu
        dlnH0_sd = abs(np.log(10.0) / 5.0) * dmu_sd
        out["calibrator_offset_equiv_delta_lnH0"] = {"mean": float(dlnH0), "sd": float(dlnH0_sd)}
        out["calibrator_offset_equiv_H0"] = {"mean": float(anchor.H0 * np.exp(dlnH0)), "sd_lin": float(anchor.H0 * np.exp(dlnH0) * dlnH0_sd)}

    return out
