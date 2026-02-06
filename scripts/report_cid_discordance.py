#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


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


def _sanitize_md(text: str) -> str:
    return str(text).replace("|", "\\|")


def _format_float(x: float, ndp: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.{ndp}f}"


def _load_stack_ladder_probe_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    probe = cfg.get("probe", {}) or {}
    if str(probe.get("name") or "") != "stack":
        raise ValueError("This script expects a stack config (probe.name=stack).")
    items = probe.get("stack", []) or []
    for item in items:
        if str(item.get("name") or "") == "pantheon_plus_shoes_ladder":
            return dict(item)
    raise ValueError("Config has no pantheon_plus_shoes_ladder stack item.")


def _parse_driver_ranking(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "driver_ranking.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _choose_model_label(*, ranking: dict[str, Any], preferred: str | None) -> str | None:
    models = ranking.get("models", {}) or {}
    if not models:
        return None
    if preferred and preferred in models:
        return str(preferred)
    # Fall back to best by mean Δlogp.
    items: list[tuple[str, float]] = []
    for k, v in models.items():
        try:
            items.append((str(k), float(v.get("mean_delta_logp", float("nan")))))
        except Exception:
            continue
    items = [x for x in items if np.isfinite(x[1])]
    if not items:
        return None
    return sorted(items, key=lambda x: x[1], reverse=True)[0][0]


def _top_groups(*, ranking: dict[str, Any], model: str, top_k: int) -> list[tuple[str, float, int]]:
    models = ranking.get("models", {}) or {}
    if model not in models:
        return []
    d = np.asarray(models[model].get("delta_logp", []), dtype=float).reshape(-1)
    groups = [str(x) for x in (ranking.get("group_labels") or [])]
    counts = np.asarray(ranking.get("test_counts") or [], dtype=int).reshape(-1)
    if not (len(groups) == d.size == counts.size):
        return []
    order = np.argsort(-d)[: int(top_k)]
    out: list[tuple[str, float, int]] = []
    for j in order.tolist():
        out.append((groups[j], float(d[j]), int(counts[j])))
    return out


def _max_pairwise_std(delta: np.ndarray, sigma: np.ndarray) -> float:
    # Compute max_{i<j} |delta_i - delta_j| / sqrt(sigma_i^2 + sigma_j^2)
    n = int(delta.size)
    if n <= 1:
        return float("nan")
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            denom = float(np.sqrt(float(sigma[i]) ** 2 + float(sigma[j]) ** 2))
            if not np.isfinite(denom) or denom <= 0.0:
                continue
            v = float(abs(float(delta[i]) - float(delta[j])) / denom)
            if np.isfinite(v) and v > best:
                best = v
    return float(best)


@dataclass(frozen=True)
class CidDiscordance:
    cid: str
    n: int
    surveys: list[str]
    idsurveys: list[int]
    mb_range: float
    mb_stdmax: float
    mb_sigma_med: float
    fitprob_min: float
    fitprob_max: float
    fitchi2ndof_med: float


def _iter_cids(ds, *, mask: np.ndarray) -> Iterable[str]:
    c = np.asarray(ds.cid, dtype=str)
    return np.unique(c[mask]).tolist()


def _cid_summary(ds, cid: str, *, mask: np.ndarray, id_to_name: dict[int, str]) -> CidDiscordance | None:
    m = (np.asarray(ds.cid == cid, dtype=bool)) & mask
    if not np.any(m):
        return None
    idxs = np.where(m)[0]
    if idxs.size <= 1:
        return None
    mb = np.asarray(ds.m_b_corr[idxs], dtype=float)
    sigma = np.asarray(ds.m_b_corr_err_diag[idxs], dtype=float)
    ids = np.asarray(ds.idsurvey[idxs], dtype=int)
    fitprob = np.asarray(ds.fitprob[idxs], dtype=float)
    fitchi2 = np.asarray(ds.fitchi2[idxs], dtype=float)
    ndof = np.asarray(ds.ndof[idxs], dtype=float)

    surveys = [str(id_to_name.get(int(s), f"IDSURVEY_{int(s)}")) for s in ids.tolist()]
    surveys_u = sorted(set(surveys))
    id_u = sorted(set(int(x) for x in ids.tolist()))

    mb_range = float(np.nanmax(mb) - np.nanmin(mb)) if mb.size else float("nan")
    mb_stdmax = _max_pairwise_std(mb, sigma)
    mb_sigma_med = float(np.nanmedian(sigma)) if sigma.size else float("nan")
    fitprob_min = float(np.nanmin(fitprob)) if fitprob.size else float("nan")
    fitprob_max = float(np.nanmax(fitprob)) if fitprob.size else float("nan")
    chi2ndof = fitchi2 / np.where(ndof > 0, ndof, np.nan)
    fitchi2ndof_med = float(np.nanmedian(chi2ndof)) if chi2ndof.size else float("nan")

    return CidDiscordance(
        cid=str(cid),
        n=int(idxs.size),
        surveys=surveys_u,
        idsurveys=id_u,
        mb_range=mb_range,
        mb_stdmax=mb_stdmax,
        mb_sigma_med=mb_sigma_med,
        fitprob_min=fitprob_min,
        fitprob_max=fitprob_max,
        fitchi2ndof_med=fitchi2ndof_med,
    )


def _rows_markdown(ds, cid: str, *, mask: np.ndarray, id_to_name: dict[int, str]) -> str:
    m = (np.asarray(ds.cid == cid, dtype=bool)) & mask
    if not np.any(m):
        return ""
    idxs = np.where(m)[0].tolist()
    lines = []
    lines.append("| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |")
    lines.append("|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i in idxs:
        sid = int(ds.idsurvey[i])
        survey = str(id_to_name.get(sid, f"IDSURVEY_{sid}"))
        chi2ndof = float(ds.fitchi2[i]) / float(ds.ndof[i]) if float(ds.ndof[i]) > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(sid),
                    f"`{_sanitize_md(survey)}`",
                    "Y" if bool(ds.is_calibrator[i]) else "N",
                    "Y" if bool(ds.is_hubble_flow[i]) else "N",
                    _format_float(float(ds.z[i]), 5),
                    _format_float(float(ds.m_b_corr[i]), 5),
                    _format_float(float(ds.m_b_corr_err_diag[i]), 5),
                    _format_float(float(ds.fitprob[i]), 4),
                    _format_float(chi2ndof, 3),
                    _format_float(float(ds.pkmjd[i]), 2),
                    _format_float(float(ds.pkmjd_err[i]), 4),
                    _format_float(float(ds.mwebv[i]), 5),
                    _format_float(float(ds.host_logmass[i]), 3),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Report discordant duplicate Pantheon+SH0ES ladder CIDs (multiple survey reductions).")
    ap.add_argument("--run-dir", type=Path, required=True, help="Existing outputs/* directory containing config.yaml (and optionally driver_ranking.json).")
    ap.add_argument("--scope", type=str, default="cal", choices=["cal", "hf", "all"], help="Which subset to analyze.")
    ap.add_argument("--threshold-mag", type=float, default=0.05, help="Flag CID if m_b_corr range exceeds this (mag).")
    ap.add_argument("--threshold-std", type=float, default=2.0, help="Flag CID if max pairwise |Δm|/σ_comb exceeds this.")
    ap.add_argument("--top-n", type=int, default=30, help="Show top N duplicate CIDs by stdmax.")
    ap.add_argument("--hero-top-k", type=int, default=10, help="Also detail top-k hero CIDs from driver_ranking.json (if present).")
    ap.add_argument("--model", type=str, default=None, help="Model label for hero selection (default: best by mean Δlogp).")
    ap.add_argument("--out-md", type=Path, default=None, help="Output markdown path (default: run_dir/cid_discordance.md).")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ladder_probe_cfg = _load_stack_ladder_probe_cfg(cfg)
    ds = load_pantheon_plus_shoes_ladder_dataset(ladder_probe_cfg)
    id_to_name = _idsurvey_to_name()

    if args.scope == "cal":
        base_mask = np.asarray(ds.is_calibrator, dtype=bool)
    elif args.scope == "hf":
        base_mask = np.asarray(ds.is_hubble_flow, dtype=bool)
    else:
        base_mask = np.ones_like(ds.is_calibrator, dtype=bool)

    cids = _iter_cids(ds, mask=base_mask)
    summaries: list[CidDiscordance] = []
    for cid in cids:
        s = _cid_summary(ds, cid, mask=base_mask, id_to_name=id_to_name)
        if s is not None:
            summaries.append(s)

    def sort_key(x: CidDiscordance) -> tuple[float, float]:
        a = x.mb_stdmax if np.isfinite(x.mb_stdmax) else -1.0
        b = x.mb_range if np.isfinite(x.mb_range) else -1.0
        return (a, b)

    summaries_sorted = sorted(summaries, key=sort_key, reverse=True)
    flagged = [
        s
        for s in summaries_sorted
        if (np.isfinite(s.mb_range) and s.mb_range >= float(args.threshold_mag))
        or (np.isfinite(s.mb_stdmax) and s.mb_stdmax >= float(args.threshold_std))
    ]

    ranking = _parse_driver_ranking(run_dir)
    hero_block = ""
    if ranking is not None:
        model = _choose_model_label(ranking=ranking, preferred=args.model)
        if model:
            top_hero = _top_groups(ranking=ranking, model=model, top_k=int(args.hero_top_k))
            if top_hero:
                hero_block += "## Hero CIDs (from driver ranking)\n\n"
                hero_block += f"- Model: `{_sanitize_md(model)}`\n\n"
                hero_block += "| CID | n_test | Δlogp | n_rows | m_b_corr range | max |Δm|/σ |\n"
                hero_block += "|---|---:|---:|---:|---:|---:|\n"
                by_cid = {s.cid: s for s in summaries_sorted}
                for cid, dlogp, n_test in top_hero:
                    s = by_cid.get(str(cid))
                    hero_block += (
                        "| "
                        + " | ".join(
                            [
                                f"`{_sanitize_md(str(cid))}`",
                                str(int(n_test)),
                                _format_float(float(dlogp), 3),
                                str(int(s.n) if s else 1),
                                _format_float(float(s.mb_range) if s else 0.0, 4),
                                _format_float(float(s.mb_stdmax) if s else float("nan"), 3),
                            ]
                        )
                        + " |\n"
                    )
                hero_block += "\n"
                hero_block += "### Hero per-row details\n\n"
                for cid, _, _ in top_hero:
                    hero_block += f"#### `{_sanitize_md(str(cid))}`\n\n"
                    hero_block += _rows_markdown(ds, str(cid), mask=base_mask, id_to_name=id_to_name) + "\n"

    lines = []
    lines.append("# CID duplicate discordance report\n")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Scope: `{args.scope}`")
    lines.append(f"- Duplicate CIDs (n>1): {len(summaries_sorted)}")
    lines.append(
        f"- Flag thresholds: range ≥ {float(args.threshold_mag):.3f} mag OR max |Δm|/σ ≥ {float(args.threshold_std):.2f}"
    )
    lines.append(f"- Flagged: {len(flagged)}")
    lines.append("")

    lines.append("## Top duplicate CIDs\n")
    lines.append("| CID | n_rows | surveys | m_b_corr range | max |Δm|/σ | σ_diag med | FITPROB min..max | chi2/ndof med |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|")
    for s in summaries_sorted[: int(args.top_n)]:
        fp = f"{_format_float(s.fitprob_min, 4)}..{_format_float(s.fitprob_max, 4)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_sanitize_md(s.cid)}`",
                    str(int(s.n)),
                    ", ".join(f"`{_sanitize_md(x)}`" for x in s.surveys),
                    _format_float(s.mb_range, 5),
                    _format_float(s.mb_stdmax, 3),
                    _format_float(s.mb_sigma_med, 5),
                    fp,
                    _format_float(s.fitchi2ndof_med, 3),
                ]
            )
            + " |"
        )
    lines.append("")

    if flagged:
        lines.append("## Flagged per-row details\n")
        for s in flagged[: min(len(flagged), 30)]:
            lines.append(f"### `{_sanitize_md(s.cid)}`\n")
            lines.append(
                f"- n_rows={s.n}, m_b_corr range={_format_float(s.mb_range,5)} mag, max |Δm|/σ={_format_float(s.mb_stdmax,3)}"
            )
            lines.append("")
            lines.append(_rows_markdown(ds, s.cid, mask=base_mask, id_to_name=id_to_name))
            lines.append("")

    if hero_block:
        lines.append(hero_block)

    out_md = args.out_md
    if out_md is None:
        out_md = run_dir / "cid_discordance.md"
    out_md = out_md.expanduser().resolve()
    out_md.write_text("\n".join(lines))
    print(str(out_md))


if __name__ == "__main__":
    main()
