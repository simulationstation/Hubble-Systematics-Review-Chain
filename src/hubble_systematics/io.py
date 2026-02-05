from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_config_and_env(config_path: Path, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(config_path.read_text())
    env = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "numpy": np.__version__,
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True))
    _write_git_like(run_dir)


def _run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and proc.stderr:
        out = f"{out}\n{proc.stderr.strip()}".strip()
    return int(proc.returncode), out


def _write_git_like(run_dir: Path) -> None:
    code, top = _run_git(["rev-parse", "--show-toplevel"])
    if code != 0:
        (run_dir / "git_like.json").write_text(json.dumps({"ok": False, "reason": "not a git repo"}, indent=2, sort_keys=True))
        return

    code, head = _run_git(["rev-parse", "HEAD"])
    code, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    code, describe = _run_git(["describe", "--tags", "--always", "--dirty"])
    code, status = _run_git(["status", "--porcelain"])
    dirty = bool(status.strip())
    code, remotes = _run_git(["remote", "-v"])

    payload = {
        "ok": True,
        "toplevel": top,
        "head": head,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
        "status_porcelain": [ln for ln in status.splitlines() if ln.strip()],
        "remotes": [ln for ln in remotes.splitlines() if ln.strip()],
    }
    (run_dir / "git_like.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
