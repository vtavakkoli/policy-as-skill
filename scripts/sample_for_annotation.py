#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Create a stratified citation-faithfulness annotation sample.")
    p.add_argument("--traces", default="result/traces.jsonl")
    p.add_argument("--per-method", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="data/annotations/citation_faithfulness_sample.csv")
    args = p.parse_args()
    traces = []
    for line in Path(args.traces).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "decision_trace":
            traces.append(obj)
    by_method = {}
    for tr in traces:
        by_method.setdefault(tr.get("method", ""), []).append(tr)
    rng = random.Random(args.seed)
    rows = []
    for method, items in sorted(by_method.items()):
        for tr in rng.sample(items, min(args.per_method, len(items))):
            evidence = {e.get("citation_id", ""): e.get("text", "") for e in tr.get("evidence", []) or []}
            for citation in (tr.get("citations", []) or [""])[:2]:
                rows.append({"task_id": tr.get("task_id", ""), "method": method, "citation_id": citation, "answer_span": tr.get("answer", ""), "evidence_span": evidence.get(citation, ""), "supported": "", "contradiction": "", "policy_ref_correct": "", "annotator_id": "", "reviewed_at": "", "notes": ""})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "method", "citation_id", "answer_span", "evidence_span", "supported", "contradiction", "policy_ref_correct", "annotator_id", "reviewed_at", "notes"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
