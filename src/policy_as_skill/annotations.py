from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _supported_to_score(value: str) -> float | None:
    v = (value or "").strip().lower()
    if v in {"yes", "y", "true", "supported", "1"}:
        return 1.0
    if v in {"partial", "partly", "partially", "0.5"}:
        return 0.5
    if v in {"no", "n", "false", "unsupported", "0"}:
        return 0.0
    return None


def load_manual_citation_annotations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load optional human citation-faithfulness annotations.

    Expected CSV columns are provided by
    data/annotations/citation_faithfulness_annotation_template.csv. Empty files
    or missing files simply return an empty map, preserving fully automated runs.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows_by_key: dict[tuple[str, str], list[float]] = {}
    contradictions_by_key: dict[tuple[str, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = (row.get("task_id") or "").strip()
            method = (row.get("method") or "").strip()
            if not task_id or not method:
                continue
            score = _supported_to_score(row.get("supported", ""))
            if score is None:
                continue
            key = (task_id, method)
            rows_by_key.setdefault(key, []).append(score)
            contradiction = (row.get("contradiction") or "").strip().lower() in {"yes", "y", "true", "1"}
            contradictions_by_key.setdefault(key, []).append(1.0 if contradiction else 0.0)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, vals in rows_by_key.items():
        out[key] = {
            "manual_evidence_faithfulness": sum(vals) / len(vals),
            "manual_contradiction_rate": sum(contradictions_by_key.get(key, [0.0])) / max(1, len(contradictions_by_key.get(key, []))),
            "manual_annotation_count": len(vals),
        }
    return out
