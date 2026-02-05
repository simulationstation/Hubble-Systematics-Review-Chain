from __future__ import annotations

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.audit.correlated_cut_null import run_correlated_cut_null_mc
from hubble_systematics.design import one_hot, ones
from hubble_systematics.gaussian_linear_model import GaussianPrior


class _ToyCategoryDataset:
    def __init__(self, *, x: np.ndarray, cat: np.ndarray, y: np.ndarray, sigma: float = 0.1, var: np.ndarray | None = None):
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.cat = np.asarray(cat, dtype=int).reshape(-1)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if not (self.x.shape == self.cat.shape == self.y.shape):
            raise ValueError("shape mismatch")
        if var is None:
            self._var = np.full(self.x.size, float(sigma) ** 2)
        else:
            self._var = np.asarray(var, dtype=float).reshape(-1)
            if self._var.shape != self.x.shape:
                raise ValueError("var shape mismatch")

    def get_column(self, name: str) -> np.ndarray:
        if name != "x":
            raise KeyError(name)
        return self.x

    def diag_sigma(self) -> np.ndarray:
        return np.sqrt(self._var)

    def subset_leq(self, col: str, value: float) -> "_ToyCategoryDataset":
        if col != "x":
            raise KeyError(col)
        m = self.x <= float(value)
        return self.subset_mask(m)

    def subset_geq(self, col: str, value: float) -> "_ToyCategoryDataset":
        if col != "x":
            raise KeyError(col)
        m = self.x >= float(value)
        return self.subset_mask(m)

    def subset_mask(self, mask: np.ndarray) -> "_ToyCategoryDataset":
        mask = np.asarray(mask, dtype=bool)
        return _ToyCategoryDataset(x=self.x[mask], cat=self.cat[mask], y=self.y[mask], var=self._var[mask])

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict):
        n = self.y.size
        y = self.y.copy()
        y0 = np.zeros_like(y)
        cov = self._var.copy()

        # The category one-hot changes when a cut removes a category.
        dm = one_hot(self.cat, prefix="cat_", reference=0, drop_reference=True)

        # Use x as the "drift param" column to avoid exact collinearity with the global offset.
        cols = [ones(n), dm.X, self.x.reshape(-1, 1)]
        names = ["global_offset_mag", *dm.names, "delta_lnH0"]
        X = np.concatenate([c if c.size else np.zeros((n, 0)) for c in cols], axis=1)
        prior = GaussianPrior.from_sigmas(names, [float("inf")] * len(names), mean=0.0)
        return y, y0, cov, X, prior


def test_correlated_cut_null_handles_varying_param_names() -> None:
    # Construct data where the strictest cut removes category==1 entirely,
    # changing the one-hot columns and therefore the parameter vector length.
    x = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    cat = np.array([0, 0, 0, 1, 1, 1])
    y = np.zeros_like(x)
    ds = _ToyCategoryDataset(x=x, cat=cat, y=y, sigma=0.01)

    anchor = AnchorLCDM(H0=67.4, Omega_m=0.315, Omega_k=0.0, rd_Mpc=147.09)
    cuts = np.array([0.25, 0.75, 1.25])

    res = run_correlated_cut_null_mc(
        dataset=ds,
        anchor=anchor,
        ladder_level="L1",
        model_cfg={"shared_scale": {"enable": True, "params": ["delta_lnH0"]}},
        cut_var="x",
        cut_mode="leq",
        cuts=cuts,
        n_mc=5,
        rng=np.random.default_rng(0),
        drift_param="delta_lnH0",
        use_diagonal_errors=True,
    )
    assert res.param_name == "delta_lnH0"
    assert set(res.p_values) == {"end_to_end", "max_pair", "path_length"}
