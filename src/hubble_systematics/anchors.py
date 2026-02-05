from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hubble_systematics.cosmology import LCDM


@dataclass(frozen=True)
class AnchorLCDM:
    """Early-time 'anchor' cosmology used to define late-time targets."""

    H0: float
    Omega_m: float
    Omega_k: float = 0.0
    rd_Mpc: float = 147.09
    _cosmo: LCDM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rd_Mpc <= 0 or not np.isfinite(self.rd_Mpc):
            raise ValueError("rd_Mpc must be positive and finite")
        object.__setattr__(self, "_cosmo", LCDM(H0=self.H0, Omega_m=self.Omega_m, Omega_k=self.Omega_k))

    @property
    def cosmo(self) -> LCDM:
        return self._cosmo

    def mu(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.distance_modulus(z)

    def H(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.H(z)

    def DM_over_rd(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.transverse_comoving_distance(z) / self.rd_Mpc

    def DH_over_rd(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.DH(z) / self.rd_Mpc

    def DV_over_rd(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.DV(z) / self.rd_Mpc
