from __future__ import annotations

import numpy as np

from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.joint import Part, stack_parts


def test_stack_parts_unions_params_and_rows() -> None:
    p1 = Part(
        y=np.array([1.0, 2.0]),
        y0=np.array([0.0, 0.0]),
        cov=np.array([1.0, 1.0]),
        X=np.ones((2, 1)),
        prior=GaussianPrior.from_sigmas(["a"], [1.0]),
    )
    p2 = Part(
        y=np.array([3.0]),
        y0=np.array([0.0]),
        cov=np.array([4.0]),
        X=np.ones((1, 1)) * 2.0,
        prior=GaussianPrior.from_sigmas(["b"], [2.0]),
    )
    s = stack_parts([p1, p2])
    assert s.y.shape == (3,)
    assert s.y0.shape == (3,)
    assert s.cov.shape == (3,)
    assert s.X.shape == (3, 2)
    assert s.prior.param_names == ["a", "b"]
    # First block: only "a" column non-zero.
    assert np.allclose(s.X[:2, 0], 1.0)
    assert np.allclose(s.X[:2, 1], 0.0)
    # Second block: only "b" column non-zero.
    assert np.allclose(s.X[2:, 0], 0.0)
    assert np.allclose(s.X[2:, 1], 2.0)

