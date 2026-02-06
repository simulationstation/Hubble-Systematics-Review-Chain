#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


@dataclass(frozen=True)
class HeroRow:
    cid: str
    idsurvey: int
    survey: str
    is_cal: bool
    is_hf: bool
    z: float
    m_b_corr: float
    m_b_corr_err_diag: float
    mu_shoes: float
    mu_shoes_err_diag: float
    ceph_dist: float
    pkmjd: float
    pkmjd_err: float
    mwebv: float
    host_logmass: float
    fitprob: float


def _idsurvey_to_name() -> dict[int, str]:
    # Keep in sync with Pantheon+SH0ES README_DataRelease_4_DISTANCES_AND_COVAR.txt.
    return {
        1: "SDSS",
        4: "SNLS",
        5: "CSP",
        10: "DES",
        15: "PS1MD",
        18: "CNIa0.02",
        50: "LOWZ/JRK07",
        51: "LOSS1",
        56: "SOUSA",
        57: "LOSS2",
        61: "CFA1",
        62: "CFA2",
        63: "CFA3S",
        64: "CFA3K",
        65: "CFA4p2",
        66: "CFA4p3",
        100: "HST",
        101: "SNAP",
        106: "CANDELS",
        150: "FOUND",
    }


def _frag_key(label: str) -> str:
    parts = str(label).strip().split()
    if len(parts) >= 2 and parts[1].startswith("p"):
        return " ".join(parts[:2])
    return parts[0]


def _survey_to_frag_keys() -> dict[str, list[str]]:
    # Pragmatic mapping used by scripts/derive_pantheon_shoes_fragilistic_priors.py
    return {
        "SDSS": ["SDSS"],
        "SNLS": ["SNLS"],
        "CSP": ["CSPDR3"],
        "DES": ["DES3YR", "DES5YR"],
        "PS1MD": ["PS1", "oldPS1", "newPS1"],
        "FOUND": ["PS1", "oldPS1", "newPS1"],
        # The DataRelease uses "CNIa0.02"; FRAGILISTIC labels this family as "CNIa0.2".
        "CNIa0.02": ["CNIa0.2"],
        "CFA3S": ["CFA3S"],
        "CFA3K": ["CFA3K"],
        "CFA4p2": ["CFA4 p2"],
        # FRAGILISTIC only ships CFA4 p1/p2; treat CFA4p3 as closest available.
        "CFA4p3": ["CFA4 p1"],
        "LOSS1": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        "LOSS2": ["KAIT1", "KAIT2", "KAIT3", "KAIT4", "NICKEL1", "NICKEL2"],
        "SOUSA": ["SWIFT"],
        # No direct FRAGILISTIC key; use fallback.
        "LOWZ/JRK07": [],
        "CFA1": [],
        "CFA2": [],
        "HST": ["HST"],
        "SNAP": ["SNAP"],
        "CANDELS": ["CANDELS"],
    }


def _survey_to_kcor_keys() -> dict[str, list[str]]:
    # Pragmatic mapping used by scripts/derive_pantheon_shoes_kcor_variant_priors.py
    return {
        "SDSS": ["SDSS"],
        "SNLS": ["SNLS"],
        "DES": ["DES_3yr", "DES_5yr"],
        "PS1MD": ["PS1"],
        "FOUND": ["Foundation"],
        "CSP": ["CSPDR3", "CSPDR3_supercal"],
        "LOSS1": ["KAIT_Mo", "KAIT_Stahl"],
        "LOSS2": ["KAIT_Mo", "KAIT_Stahl"],
        "SOUSA": ["SWIFT"],
        "CNIa0.02": ["ASASSN"],
        "LOWZ/JRK07": ["Land2"],
        "CFA1": ["Land2"],
        "CFA2": ["Land2"],
        "CFA3S": ["Land2"],
        "CFA3K": ["Land2"],
        "CFA4p2": ["Land2"],
        "CFA4p3": ["Land2"],
        "HST": [],
        "SNAP": [],
        "CANDELS": [],
    }


def _load_fragilistic_labels(*, root: Path) -> dict[str, list[str]]:
    npz_path = root / "data/raw/pantheon_plus_calibration/FRAGILISTIC_COVARIANCE.npz"
    if not npz_path.exists():
        return {}
    z = np.load(npz_path)
    labels = np.asarray(z["labels"])
    groups: dict[str, list[str]] = {}
    for lab in labels.tolist():
        k = _frag_key(str(lab))
        groups.setdefault(k, []).append(str(lab).strip())
    return groups


def _format_float(x: float, ndp: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.{ndp}f}"


def _sanitize_md(text: str) -> str:
    return str(text).replace("|", "\\|")


def _parse_driver_ranking(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "driver_ranking.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _top_groups(*, ranking: dict[str, Any], model: str, top_k: int) -> list[tuple[str, float, int]]:
    models = ranking.get("models", {}) or {}
    if model not in models:
        raise KeyError(f"Model {model!r} not found; available={sorted(models.keys())}")
    d = np.asarray(models[model]["delta_logp"], dtype=float).reshape(-1)
    groups = [str(x) for x in (ranking.get("group_labels") or [])]
    counts = np.asarray(ranking.get("test_counts") or [], dtype=int).reshape(-1)
    if not (len(groups) == d.size == counts.size):
        raise ValueError("driver_ranking.json shape mismatch")
    order = np.argsort(-d)[: int(top_k)]
    out: list[tuple[str, float, int]] = []
    for j in order.tolist():
        out.append((groups[j], float(d[j]), int(counts[j])))
    return out


def _build_rows_for_cid(*, ds, cid: str, id_to_name: dict[int, str]) -> list[HeroRow]:
    mask = np.asarray(ds.cid == cid, dtype=bool)
    if not np.any(mask):
        return []
    out: list[HeroRow] = []
    idxs = np.where(mask)[0].tolist()
    for i in idxs:
        sid = int(ds.idsurvey[i])
        out.append(
            HeroRow(
                cid=str(ds.cid[i]),
                idsurvey=sid,
                survey=str(id_to_name.get(sid, f"IDSURVEY_{sid}")),
                is_cal=bool(ds.is_calibrator[i]),
                is_hf=bool(ds.is_hubble_flow[i]),
                z=float(ds.z[i]),
                m_b_corr=float(ds.m_b_corr[i]),
                m_b_corr_err_diag=float(ds.m_b_corr_err_diag[i]),
                mu_shoes=float(ds.mu_shoes[i]),
                mu_shoes_err_diag=float(ds.mu_shoes_err_diag[i]),
                ceph_dist=float(ds.ceph_dist[i]),
                pkmjd=float(ds.pkmjd[i]),
                pkmjd_err=float(ds.pkmjd_err[i]),
                mwebv=float(ds.mwebv[i]),
                host_logmass=float(ds.host_logmass[i]),
                fitprob=float(ds.fitprob[i]),
            )
        )
    return out


def _load_stack_ladder_probe_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    probe = cfg.get("probe", {}) or {}
    if str(probe.get("name") or "") != "stack":
        raise ValueError("This script currently expects a stack config (probe.name=stack).")
    items = probe.get("stack", []) or []
    for item in items:
        if str(item.get("name") or "") == "pantheon_plus_shoes_ladder":
            return dict(item)
    raise ValueError("Config has no pantheon_plus_shoes_ladder stack item.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect high-leverage (hero) calibrators from driver_ranking.json.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Existing outputs/* run directory containing config.yaml + driver_ranking.json.")
    ap.add_argument("--model", type=str, default="+bounded_fields_plus_metadata_bounded", help="Model label to use for top-driver selection.")
    ap.add_argument("--top-k", type=int, default=10, help="Top-k groups (CIDs) to report.")
    ap.add_argument("--out-md", type=Path, default=None, help="Output markdown path (default: run_dir/hero_calibrators_provenance.md).")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ranking = _parse_driver_ranking(run_dir)

    model = str(args.model)
    if model not in (ranking.get("models", {}) or {}):
        # Fall back to best model by mean_delta_logp.
        items = []
        for k, v in (ranking.get("models", {}) or {}).items():
            items.append((str(k), float(v.get("mean_delta_logp", float("nan")))))
        items = [x for x in items if np.isfinite(x[1])]
        if not items:
            raise ValueError("No models in driver_ranking.json")
        model = sorted(items, key=lambda x: x[1], reverse=True)[0][0]

    top = _top_groups(ranking=ranking, model=model, top_k=int(args.top_k))

    root = Path(__file__).resolve().parents[1]
    ladder_probe_cfg = _load_stack_ladder_probe_cfg(cfg)
    ds = load_pantheon_plus_shoes_ladder_dataset(ladder_probe_cfg)

    id_to_name = _idsurvey_to_name()
    survey_to_frag = _survey_to_frag_keys()
    survey_to_kcor = _survey_to_kcor_keys()
    frag_labels_by_key = _load_fragilistic_labels(root=root)

    md = []
    md.append("# Hero calibrator provenance\n\n")
    md.append(f"- Source run: `{run_dir}`\n")
    md.append(f"- Driver model: `{model}`\n")
    md.append(f"- Top-k: {int(args.top_k)}\n\n")
    md.append(
        "This report lists the highest-leverage **calibrator** SNe (by Δlogp in a CID holdout) and shows "
        "their key table metadata and which surveys/photometry reductions they appear under.\n\n"
    )

    md.append("## Top drivers\n\n")
    md.append("| CID | n_test | Δlogp |\n")
    md.append("|---|---:|---:|\n")
    for cid, dlogp, n_test in top:
        md.append(f"| `{_sanitize_md(cid)}` | {int(n_test)} | {float(dlogp):+.3f} |\n")

    md.append("\n## Per-CID details\n\n")

    for cid, dlogp, n_test in top:
        rows = _build_rows_for_cid(ds=ds, cid=str(cid), id_to_name=id_to_name)
        if not rows:
            continue
        surveys = sorted({r.survey for r in rows})
        md.append(f"### `{_sanitize_md(cid)}`\n\n")
        md.append(f"- Holdout Δlogp: {float(dlogp):+.3f} (n_test={int(n_test)})\n")
        md.append(f"- Rows in ladder table: {len(rows)}\n")
        md.append(f"- Surveys present: {', '.join(f'`{_sanitize_md(s)}`' for s in surveys)}\n\n")

        md.append(
            "| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |\n"
        )
        md.append("|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda x: (x.idsurvey, x.pkmjd)):
            md.append(
                "| "
                + f"{r.idsurvey} | `{_sanitize_md(r.survey)}` | "
                + ("Y" if r.is_cal else "N")
                + " | "
                + ("Y" if r.is_hf else "N")
                + " | "
                + f"{_format_float(r.z, 5)} | {_format_float(r.m_b_corr, 5)} | {_format_float(r.m_b_corr_err_diag, 5)} | "
                + f"{_format_float(r.mu_shoes, 5)} | {_format_float(r.mu_shoes_err_diag, 5)} | {_format_float(r.ceph_dist, 5)} | "
                + f"{_format_float(r.pkmjd, 2)} | {_format_float(r.pkmjd_err, 4)} | {_format_float(r.mwebv, 5)} | {_format_float(r.host_logmass, 3)} | {_format_float(r.fitprob, 4)}"
                + " |\n"
            )

        md.append("\n")

        md.append("Photometry provenance (external products used for calibration priors):\n\n")
        md.append("| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |\n")
        md.append("|---|---|---|---|\n")
        for s in surveys:
            frag_keys = survey_to_frag.get(s, [])
            frag_filters = []
            for k in frag_keys:
                frag_filters.extend(frag_labels_by_key.get(k, []))
            frag_filters = sorted({x for x in frag_filters})
            if len(frag_filters) > 12:
                frag_filters_str = ", ".join(f"`{_sanitize_md(x)}`" for x in frag_filters[:12]) + f", …(+{len(frag_filters)-12})"
            else:
                frag_filters_str = ", ".join(f"`{_sanitize_md(x)}`" for x in frag_filters) if frag_filters else ""
            kcor_keys = survey_to_kcor.get(s, [])
            md.append(
                f"| `{_sanitize_md(s)}` | "
                + (", ".join(f"`{_sanitize_md(k)}`" for k in frag_keys) if frag_keys else "")
                + " | "
                + frag_filters_str
                + " | "
                + (", ".join(f"`{_sanitize_md(k)}`" for k in kcor_keys) if kcor_keys else "")
                + " |\n"
            )
        md.append("\n")

    out_md = (run_dir / "hero_calibrators_provenance.md") if args.out_md is None else args.out_md.expanduser().resolve()
    out_md.write_text("".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

