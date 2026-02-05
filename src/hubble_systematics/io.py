from __future__ import annotations

import json
import os
import platform
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

