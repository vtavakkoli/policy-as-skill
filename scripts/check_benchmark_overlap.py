#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_as_skill.data_loader import load_tasks_from_path
from policy_as_skill.protocol import max_ngram_overlap


def main() -> None:
    p = argparse.ArgumentParser(description="Measure lexical overlap between development and proposed held-out tasks.")
    p.add_argument("--development", default="data/tasks/benchmark_tasks.jsonl")
    p.add_argument("--test", required=True)
    p.add_argument("--ngram", type=int, default=4)
    p.add_argument("--warn-threshold", type=float, default=0.50)
    p.add_argument("--fail-threshold", type=float, default=0.80)
    p.add_argument("--output", default="result/benchmark_overlap.json")
    args = p.parse_args()
    dev_path, test_path = ROOT / args.development, ROOT / args.test
    dev, test = load_tasks_from_path(dev_path), load_tasks_from_path(test_path)
    dev_questions = [x.question for x in dev]
    rows = [{"task_id": task.id, "max_development_ngram_overlap": max_ngram_overlap(task.question, dev_questions, n=args.ngram)} for task in test]
    payload = {
        "development": str(dev_path), "test": str(test_path), "ngram": args.ngram,
        "warn_threshold": args.warn_threshold, "fail_threshold": args.fail_threshold,
        "max_overlap": max((x["max_development_ngram_overlap"] for x in rows), default=0.0),
        "warning_count": sum(x["max_development_ngram_overlap"] >= args.warn_threshold for x in rows),
        "failure_count": sum(x["max_development_ngram_overlap"] >= args.fail_threshold for x in rows), "rows": rows,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    if payload["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
