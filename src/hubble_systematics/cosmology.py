from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import integrate

_C_KM_S = 299_792.458


@dataclass
class LCDM:
    """Minimal FRW background for distance predictions.

    Units:
      - H0 in km/s/Mpc
      - distances in Mpc
    """

    H0: float
    Omega_m: float
    Omega_k: float = 0.0
    _z_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _I_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.H0) or self.H0 <= 0:
            raise ValueError("H0 must be positive and finite")
        if not np.isfinite(self.Omega_m):
            raise ValueError("Omega_m must be finite")
        if not np.isfinite(self.Omega_k):
            raise ValueError("Omega_k must be finite")

    @property
    def Omega_lambda(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_k

    def E(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.sqrt(
            self.Omega_m * (1.0 + z) ** 3
            + self.Omega_k * (1.0 + z) ** 2
            + self.Omega_lambda
        )

    def H(self, z: np.ndarray) -> np.ndarray:
        return self.H0 * self.E(z)

    def _ensure_I_cache(self, z_max: float, *, n_grid: int = 8192) -> None:
        z_max = float(z_max)
        if z_max < 0:
            raise ValueError("z_max must be >= 0")
        if self._z_cache is not None and self._I_cache is not None:
            if z_max <= float(self._z_cache[-1]) and self._z_cache.size == n_grid:
                return
        if z_max == 0.0:
            self._z_cache = np.array([0.0])
            self._I_cache = np.array([0.0])
            return
        zg = np.linspace(0.0, z_max, int(n_grid), dtype=float)
        invE = 1.0 / self.E(zg)
        Ig = integrate.cumulative_trapezoid(invE, zg, initial=0.0)
        self._z_cache = zg
        self._I_cache = Ig

    def _I_of_z(self, z: np.ndarray, *, n_grid: int = 8192) -> np.ndarray:
        """Dimensionless comoving distance integral I(z)=∫0^z dz'/E(z')."""

        z = np.asarray(z, dtype=float)
        z_max = float(np.max(z)) if z.size else 0.0
        self._ensure_I_cache(z_max, n_grid=n_grid)
        assert self._z_cache is not None and self._I_cache is not None
        return np.interp(z, self._z_cache, self._I_cache)

    def comoving_distance(self, z: np.ndarray) -> np.ndarray:
        """Line-of-sight comoving distance D_C(z) in Mpc."""

        return (_C_KM_S / self.H0) * self._I_of_z(z)

    def transverse_comoving_distance(self, z: np.ndarray) -> np.ndarray:
        """Transverse comoving distance D_M(z) in Mpc."""

        z = np.asarray(z, dtype=float)
        dc = self.comoving_distance(z)
        if np.isclose(self.Omega_k, 0.0):
            return dc

        I = (self.H0 / _C_KM_S) * dc  # dimensionless
        ok = float(self.Omega_k)
        if ok > 0:
            return (_C_KM_S / self.H0) / np.sqrt(ok) * np.sinh(np.sqrt(ok) * I)
        return (_C_KM_S / self.H0) / np.sqrt(abs(ok)) * np.sin(np.sqrt(abs(ok)) * I)

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return (1.0 + z) * self.transverse_comoving_distance(z)

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        dL = self.luminosity_distance(z)
        return 5.0 * np.log10(dL) + 25.0

    def DH(self, z: np.ndarray) -> np.ndarray:
        return _C_KM_S / self.H(z)

    def DV(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        dm = self.transverse_comoving_distance(z)
        return (dm**2 * self.DH(z) * z) ** (1.0 / 3.0)
