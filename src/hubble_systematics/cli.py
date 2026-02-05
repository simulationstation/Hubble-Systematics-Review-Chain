from __future__ import annotations

import argparse
from pathlib import Path

from hubble_systematics.audit.runner import run_from_config_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hubble-audit",
        description="Systematics-first audit runner for late-time distance-scale probes.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run an audit packet from a YAML config.")
    run_p.add_argument("config", type=Path, help="Path to YAML config.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        run_from_config_path(args.config)
        return

    raise RuntimeError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    main()

