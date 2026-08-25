#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_as_skill.protocol import sha256_file


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Create a frozen held-out benchmark manifest.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--controller-version", default="")
    p.add_argument("--confirm-no-feedback", action="store_true", help="Confirm test tasks were finalized without controller-error feedback.")
    args = p.parse_args()
    if not args.confirm_no_feedback:
        raise SystemExit("Refusing to mark a test set frozen without --confirm-no-feedback.")
    benchmark = Path(args.benchmark)
    if not benchmark.is_absolute():
        benchmark = ROOT / benchmark
    output = Path(args.output) if args.output else benchmark.with_suffix(".manifest.json")
    if not output.is_absolute():
        output = ROOT / output
    payload = {
        "split": "test",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "controller_version": args.controller_version or git_head(),
        "benchmark_sha256": sha256_file(benchmark),
        "created_without_controller_feedback": True,
        "model_parameter_training_on_benchmark": False,
        "model_fine_tuning_on_benchmark": False,
        "notes": "Freeze this manifest before running the controller on held-out tasks."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
