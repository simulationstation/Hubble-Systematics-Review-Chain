from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_report(run_dir: Path) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[str] = [f"# Audit report\n\nRun dir: `{run_dir}`\n"]

    base = _read_json(run_dir / "baseline_fit.json")
    ledger = _read_json(run_dir / "mechanism_ledger.json")
    stack_parts = _read_json(run_dir / "stack_parts.json")
    if base is not None:
        chi2 = float(base.get("chi2", float("nan")))
        dof = int(base.get("dof", 0))
        n = int(base.get("n", 0))
        lvl = base.get("ladder_level")
        logZ = base.get("log_evidence")
        pieces.append("## Baseline fit\n\n")
        pieces.append(f"- N: {n}\n")
        pieces.append(f"- Ladder level: {lvl}\n")
        pieces.append(f"- chi2/dof: {chi2:.2f} / {dof}\n")
        if logZ is not None:
            try:
                pieces.append(f"- log evidence: {float(logZ):.2f}\n")
            except Exception:
                pass
        if ledger is not None and "H0_eff" in ledger:
            H0 = ledger["H0_eff"]
            dm = ledger.get("delta_mu_equiv_mag")
            pieces.append(f"- H0_eff (from delta_lnH0): {H0.get('mean'):.3f} ± {H0.get('sd_lin'):.3f}\n")
            if dm is not None:
                pieces.append(f"- Equivalent Δμ [mag]: {dm.get('mean'):.4f} ± {dm.get('sd'):.4f}\n")
            cal_equiv = ledger.get("calibrator_offset_equiv_H0")
            if isinstance(cal_equiv, dict) and "mean" in cal_equiv and "sd_lin" in cal_equiv:
                pieces.append(f"- Calibrator-offset-equivalent H0 [km/s/Mpc]: {cal_equiv.get('mean'):.3f} ± {cal_equiv.get('sd_lin'):.3f}\n")
        # Quick systematic-parameter summary (if present).
        try:
            names = list(base.get("param_names") or [])
            mean = np.asarray(base.get("mean") or [], dtype=float)
            cov = np.asarray(base.get("cov") or [], dtype=float)
            if mean.size == len(names) and cov.shape == (mean.size, mean.size) and mean.size > 0:
                sd = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

                def show(name: str) -> None:
                    if name not in names:
                        return
                    j = names.index(name)
                    pieces.append(f"- {name}: {mean[j]:+.4f} ± {sd[j]:.4f}\n")

                def show_prefix(prefix: str, label: str) -> None:
                    idx = [i for i, nm in enumerate(names) if str(nm).startswith(prefix)]
                    if not idx:
                        return
                    a = np.max(np.abs(mean[idx]))
                    pieces.append(f"- {label}: max |mean|={a:.4f} (n={len(idx)})\n")

                show("calibrator_offset_mag")
                show_prefix("cal_survey_offset_", "Calibrator survey offsets")
                show_prefix("pkmjd_bin_offset_", "Time-bin offsets")
                show_prefix("survey_pkmjd_bin_offset_", "Survey×time offsets")
                show_prefix("hf_survey_offset_", "HF survey offsets")
                show_prefix("hf_z_spline_", "HF z-spline coeffs")
                show_prefix("z_spline_", "HF z-spline coeffs (legacy)")
                show_prefix("sky_", "Sky low-ℓ modes")
        except Exception:
            pass

        if stack_parts is not None:
            pieces.append("- Per-part chi2:\n")
            for item in (stack_parts.get("parts") or [])[:20]:
                pieces.append(f"  - {item.get('label')}: chi2={item.get('chi2')}, n={item.get('n')}\n")

    tension_delta_lnH0 = None
    if isinstance(ledger, dict):
        d = ledger.get("delta_lnH0")
        if isinstance(d, dict) and "mean" in d:
            tension_delta_lnH0 = float(d["mean"])

    cut_scan = _read_json(run_dir / "cut_scan.json")
    if cut_scan is not None:
        png = fig_dir / "cut_scan_param.png"
        _plot_cut_scan(cut_scan, png)
        pieces.append("## Cut scan\n\n")
        pieces.append(f"- Cut var: {cut_scan.get('cut_var')}\n")
        pieces.append(f"- Cut mode: {cut_scan.get('cut_mode', 'leq')}\n")
        pieces.append(f"- Figure: `{png}`\n")

    drift = _read_json(run_dir / "correlated_cut_null.json")
    if drift is not None:
        png = fig_dir / "correlated_cut_null_hist.png"
        _plot_correlated_cut_null(drift, png)
        pvals = drift.get("p_values", {})
        pieces.append("## Correlated-cut null\n\n")
        pieces.append(f"- Drift param: {drift.get('param_name')}\n")
        pieces.append(f"- Cut mode: {drift.get('cut_mode', 'leq')}\n")
        pieces.append(f"- p-values (end_to_end / max_pair / path_length): {pvals}\n")
        pieces.append(f"- Figure: `{png}`\n")

    inj = _read_json(run_dir / "injection_suite.json")
    if inj is not None:
        pieces.append("## Injection/recovery\n\n")
        if "rows" in inj:
            png = fig_dir / "injection_suite.png"
            _plot_injection(inj, png)
            pieces.append(f"- Mechanism: {inj.get('mechanism')}\n")
            pieces.append(f"- Param of interest: {inj.get('param_of_interest')}\n")
            slope, req = _injection_slope_and_required_amp(inj, tension_delta_lnH0=tension_delta_lnH0)
            if slope is not None:
                pieces.append(f"- Approx bias slope d(param)/d(amp): {slope:.4f}\n")
            if req is not None:
                pieces.append(f"- Amp to explain |delta_lnH0|: {req:.3f}\n")
            pieces.append(f"- Figure: `{png}`\n")
        else:
            for label, obj in inj.items():
                safe = "".join(c if (c.isalnum() or c in {"-", "_"} or c == ".") else "_" for c in str(label))[:80]
                png = fig_dir / f"injection_suite_{safe}.png"
                _plot_injection(obj, png)
                slope, req = _injection_slope_and_required_amp(obj, tension_delta_lnH0=tension_delta_lnH0)
                extra = ""
                if slope is not None:
                    extra += f", slope={slope:.4f}"
                if req is not None:
                    extra += f", amp_for_|delta_lnH0|={req:.3f}"
                pieces.append(f"- {label}: {obj.get('mechanism')} → {obj.get('param_of_interest')}{extra} (`{png}`)\n")

    sbc = _read_json(run_dir / "sbc.json")
    if sbc is not None:
        png = fig_dir / "sbc_coverage.png"
        _plot_sbc(sbc, png)
        pieces.append(f"## SBC\n\n- Figure: `{png}`\n")

    hemi = _read_json(run_dir / "hemisphere_scan.json")
    if hemi is not None:
        pieces.append("## Hemisphere scan\n\n")
        pieces.append(f"- Best axis (lon,lat deg): {hemi.get('best_axis')}\n")
        pieces.append(f"- TRAIN z: {hemi.get('train')}\n")
        pieces.append(f"- TEST z: {hemi.get('test')}\n")

    split_fit = _read_json(run_dir / "split_fit.json")
    if split_fit is not None:
        pieces.append("## Split fit\n\n")
        if isinstance(split_fit, dict):
            for label, obj in split_fit.items():
                if not isinstance(obj, dict):
                    continue
                pieces.append(f"- {label}: split_var={obj.get('split_var')}, param={obj.get('param')}\n")
                rows = obj.get("rows", []) or []
                for r in rows[:10]:
                    if r.get("skipped"):
                        pieces.append(f"  - {r.get('label')}: skipped\n")
                    else:
                        pieces.append(f"  - {r.get('label')}: mean={r.get('mean')}, sd={r.get('sd')}, n={r.get('n')}\n")

    split_null = _read_json(run_dir / "split_null.json")
    if split_null is not None:
        png = fig_dir / "split_null_hist.png"
        _plot_split_null(split_null, png)
        pieces.append("## Split null\n\n")
        pieces.append(f"- Split var: {split_null.get('split_var')} ({split_null.get('mode')})\n")
        pieces.append(f"- Param: {split_null.get('param')}\n")
        pieces.append(f"- Shuffle within: {split_null.get('shuffle_within')}\n")
        pieces.append(f"- p-values (span / chi2_const): {split_null.get('p_values')}\n")
        pieces.append(f"- Figure: `{png}`\n")

    group_split = _read_json(run_dir / "group_split_null.json")
    if group_split is not None:
        pieces.append("## Grouped split null\n\n")
        pieces.append(f"- Group var: {group_split.get('group_var')}\n")
        results = group_split.get("results", {}) or {}
        order = group_split.get("group_values") or list(results.keys())
        for gid in order:
            item = results.get(str(gid), {})
            r = item.get("result", {}) or {}
            pv = r.get("p_values", {}) or {}
            obs = ((r.get("observed") or {}).get("metrics") or {}) if isinstance(r, dict) else {}
            n_group = item.get("n_group")
            span = obs.get("span")
            pieces.append(f"- {gid}: n_hf={n_group}, span={span}, p(span)={pv.get('span')}, p(chi2)={pv.get('chi2_const')}\n")

    sweep = _read_json(run_dir / "baseline_sweep.json")
    if sweep is not None:
        pieces.append("## Baseline sweep\n\n")
        base_label = sweep.get("base_label")
        pieces.append(f"- Base: {base_label}\n")
        rows = sweep.get("rows", []) or []
        for r in rows:
            label = r.get("label")
            try:
                chi2pd = float(r.get("chi2_per_dof"))
            except Exception:
                chi2pd = float("nan")
            dlogZ = r.get("delta_log_evidence_vs_first")
            frac = r.get("tension_reduction_fraction_vs_first")
            pieces.append(f"- {label}: chi2/dof={chi2pd:.3f}, ΔlogZ={dlogZ}, frac={frac}\n")

    pred = _read_json(run_dir / "predictive_score.json")
    if pred is not None:
        pieces.append("## Predictive score\n\n")
        pieces.append(f"- Mode: {pred.get('mode')}\n")
        pieces.append(f"- N splits: {pred.get('n_rep')}\n")
        pieces.append(f"- Train frac: {pred.get('train_frac')}\n")
        if "always_include_calibrators" in pred:
            pieces.append(f"- Always include calibrators: {pred.get('always_include_calibrators')}\n")
        if "always_include_hubble_flow" in pred:
            pieces.append(f"- Always include hubble-flow: {pred.get('always_include_hubble_flow')}\n")
        base_label = None
        models = pred.get("models", {}) or {}
        if isinstance(models, dict) and models:
            base_label = next(iter(models.keys()))
        for label, obj in models.items() if isinstance(models, dict) else []:
            try:
                mean_logp = float(obj.get("mean_logp"))
            except Exception:
                mean_logp = float("nan")
            try:
                dlogp = float(obj.get("mean_delta_logp_vs_base"))
            except Exception:
                dlogp = float("nan")
            pieces.append(f"- {label}: mean_logp={mean_logp:.2f}, Δlogp_vs_base={dlogp:.2f}\n")

    (run_dir / "report.md").write_text("\n".join(pieces))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _plot_cut_scan(obj: dict[str, Any], out: Path) -> None:
    rows = obj["rows"]
    cuts = np.array([r["cut"] for r in rows], dtype=float)
    # Choose a parameter to plot: prefer 'global_offset_mag' if present.
    params0 = rows[0].get("params", {})
    if "delta_lnH0" in params0:
        name = "delta_lnH0"
    elif "global_offset_mag" in params0:
        name = "global_offset_mag"
    else:
        name = next(iter(params0.keys()))
    vals = np.array([r["params"][name] for r in rows], dtype=float)
    ns = np.array([r["n"] for r in rows], dtype=int)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(cuts, vals, marker="o", lw=1.5)
    plt.xlabel(obj.get("cut_var", "cut"))
    plt.ylabel(name)
    plt.title("Cut scan")
    ax2 = plt.gca().twinx()
    ax2.plot(cuts, ns, color="gray", alpha=0.35, lw=1.0)
    ax2.set_ylabel("N")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_correlated_cut_null(obj: dict[str, Any], out: Path) -> None:
    mc = obj["mc"]
    obs = obj["observed"]
    metrics = ["end_to_end", "max_pair", "path_length"]

    plt.figure(figsize=(10.0, 3.2))
    for i, m in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        x = np.array(mc[m], dtype=float)
        plt.hist(x, bins=30, color="#4C72B0", alpha=0.8)
        plt.axvline(float(obs[m]), color="k", lw=2)
        plt.title(m)
        plt.xlabel("metric")
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_injection(obj: dict[str, Any], out: Path) -> None:
    rows = obj["rows"]
    amp = np.array([r["amp"] for r in rows], dtype=float)
    bias = np.array([r["bias"] for r in rows], dtype=float)
    sd = np.array([r["std_hat"] for r in rows], dtype=float)

    plt.figure(figsize=(6.2, 4.0))
    plt.axhline(0.0, color="k", lw=1)
    plt.errorbar(amp, bias, yerr=sd, fmt="o-", lw=1.5)
    plt.xlabel("injection amplitude")
    plt.ylabel("bias (hat - true)")
    plt.title(f"Injection: {obj.get('mechanism')} → {obj.get('param_of_interest')}")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_sbc(obj: dict[str, Any], out: Path) -> None:
    names = obj.get("param_names", [])
    if not names:
        return
    cov68 = obj["coverage_68"]
    cov95 = obj["coverage_95"]
    x = np.arange(len(names))
    y68 = np.array([cov68[n] for n in names], dtype=float)
    y95 = np.array([cov95[n] for n in names], dtype=float)

    plt.figure(figsize=(7.0, 3.6))
    w = 0.38
    plt.bar(x - w / 2, y68, width=w, label="68%")
    plt.bar(x + w / 2, y95, width=w, label="95%")
    plt.axhline(0.68, color="k", lw=1, alpha=0.6)
    plt.axhline(0.95, color="k", lw=1, alpha=0.6)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("empirical coverage")
    plt.title("SBC coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_split_null(obj: dict[str, Any], out: Path) -> None:
    try:
        mc = obj["mc"]
        obs = obj["observed"]["metrics"]
        span = np.asarray(mc["span"], dtype=float)
        chi2 = np.asarray(mc["chi2_const"], dtype=float)
        span_obs = float(obs["span"])
        chi2_obs = float(obs["chi2_const"])
    except Exception:
        return

    plt.figure(figsize=(10.0, 3.2))
    plt.subplot(1, 2, 1)
    plt.hist(span, bins=30, color="#4C72B0", alpha=0.8)
    plt.axvline(span_obs, color="k", lw=2)
    plt.title("span")
    plt.xlabel("max(mean) - min(mean)")
    plt.ylabel("count")

    plt.subplot(1, 2, 2)
    plt.hist(chi2, bins=30, color="#4C72B0", alpha=0.8)
    plt.axvline(chi2_obs, color="k", lw=2)
    plt.title("chi2_const")
    plt.xlabel("chi2 vs constant mean")
    plt.ylabel("count")

    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _injection_slope_and_required_amp(obj: dict[str, Any], *, tension_delta_lnH0: float | None) -> tuple[float | None, float | None]:
    rows = obj.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None, None
    try:
        amp = np.array([r["amp"] for r in rows], dtype=float)
        bias = np.array([r["bias"] for r in rows], dtype=float)
    except Exception:
        return None, None
    # Fit bias ≈ slope * amp with zero intercept (mechanism amplitude is defined to be zero at null).
    denom = float(np.sum(amp**2))
    if denom <= 0:
        return None, None
    slope = float(np.sum(amp * bias) / denom)
    req = None
    if tension_delta_lnH0 is not None and abs(slope) > 0:
        req = float(abs(tension_delta_lnH0) / abs(slope))
    return slope, req
