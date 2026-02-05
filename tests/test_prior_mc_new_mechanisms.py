from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hubble_systematics.audit.prior_mc import _component_masks


def _dummy_dataset() -> SimpleNamespace:
    n = 6
    return SimpleNamespace(
        z=np.linspace(0.01, 0.2, n),
        idsurvey=np.array([1, 1, 5, 5, 5, 1], dtype=int),
        idsurvey_levels=np.array([1, 5], dtype=int),
        is_calibrator=np.array([True, False, True, False, True, False], dtype=bool),
        is_hubble_flow=np.array([False, True, False, True, False, True], dtype=bool),
        mwebv=np.array([0.02, 0.03, 0.10, 0.01, 0.05, 0.02], dtype=float),
        mwebv_mu=0.04,
        mwebv_sd=0.03,
        pkmjd_err=np.array([5.0, 6.0, 4.5, 7.0, 5.5, 6.2], dtype=float),
        pkmjd_err_mu=5.7,
        pkmjd_err_sd=0.9,
        c=np.array([0.01, -0.02, 0.10, 0.05, -0.01, 0.03], dtype=float),
        c_mu=0.03,
        c_sd=0.04,
        x1=np.array([0.5, -0.3, 1.2, 0.1, -0.2, 0.0], dtype=float),
        x1_mu=0.2,
        x1_sd=0.6,
        biascor_m_b=np.array([0.02, 0.01, -0.03, 0.00, 0.01, 0.02], dtype=float),
        biascor_m_b_mu=0.005,
        biascor_m_b_sd=0.02,
    )


def test_component_masks_survey_offsets() -> None:
    ds = _dummy_dataset()
    out = _component_masks(dataset=ds, mechanism="survey_offset_mag", cfg={})
    assert "survey_offset_5" in out
    assert out["survey_offset_5"].shape == ds.z.shape
    assert np.all(np.isin(out["survey_offset_5"], [0.0, 1.0]))

    out_cal = _component_masks(dataset=ds, mechanism="cal_survey_offset_mag", cfg={})
    assert "cal_survey_offset_5" in out_cal
    assert out_cal["cal_survey_offset_5"].shape == ds.z.shape

    out_hf = _component_masks(dataset=ds, mechanism="hf_survey_offset_mag", cfg={})
    assert "hf_survey_offset_5" in out_hf
    assert out_hf["hf_survey_offset_5"].shape == ds.z.shape


def test_component_masks_linear_proxies_apply_to_cal() -> None:
    ds = _dummy_dataset()

    c = _component_masks(dataset=ds, mechanism="c_linear_mag", cfg={"apply_to": "cal"})
    assert "c_linear_mag" in c
    assert c["c_linear_mag"].shape == ds.z.shape

    x1 = _component_masks(dataset=ds, mechanism="x1_linear_mag", cfg={"apply_to": "cal"})
    assert "x1_linear_mag" in x1
    assert x1["x1_linear_mag"].shape == ds.z.shape

    b = _component_masks(dataset=ds, mechanism="biascor_m_b_linear_mag", cfg={"apply_to": "cal"})
    assert "biascor_m_b_linear_mag" in b
    assert b["biascor_m_b_linear_mag"].shape == ds.z.shape


def test_component_masks_other_linear_proxies() -> None:
    ds = _dummy_dataset()

    mwebv = _component_masks(dataset=ds, mechanism="mwebv_linear_mag", cfg={"apply_to": "hf"})
    assert "mwebv_linear_mag" in mwebv
    assert mwebv["mwebv_linear_mag"].shape == ds.z.shape

    pkmjderr = _component_masks(dataset=ds, mechanism="pkmjd_err_linear_mag", cfg={"apply_to": "cal"})
    assert "pkmjd_err_linear_mag" in pkmjderr
    assert pkmjderr["pkmjd_err_linear_mag"].shape == ds.z.shape

