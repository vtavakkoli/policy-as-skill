from __future__ import annotations

import csv
from collections import defaultdict
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


def _supported_label(value: str) -> str | None:
    score = _supported_to_score(value)
    if score is None:
        return None
    return "yes" if score == 1.0 else "partial" if score == 0.5 else "no"


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({x for pair in pairs for x in pair})
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)
    counts_a = {label: sum(1 for a, _ in pairs if a == label) / len(pairs) for label in labels}
    counts_b = {label: sum(1 for _, b in pairs if b == label) / len(pairs) for label in labels}
    expected = sum(counts_a[label] * counts_b[label] for label in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def load_manual_citation_annotations(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
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
        contradictions = contradictions_by_key.get(key, [])
        out[key] = {"manual_evidence_faithfulness": sum(vals) / len(vals), "manual_contradiction_rate": sum(contradictions) / max(1, len(contradictions)), "manual_annotation_count": len(vals)}
    return out


def annotation_agreement(path: Path) -> dict[str, Any]:
    """Compute pairwise Cohen's kappa when two annotators label the same item."""
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "not_available", "rows": 0, "paired_items": 0, "supported_kappa": None, "contradiction_kappa": None}
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    total = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            annotator = (row.get("annotator_id") or "").strip()
            if not annotator:
                continue
            key = ((row.get("task_id") or "").strip(), (row.get("method") or "").strip(), (row.get("citation_id") or "").strip(), (row.get("answer_span") or "").strip())
            if not key[0] or not key[1]:
                continue
            grouped[key][annotator] = row
            total += 1
    supported_pairs, contradiction_pairs = [], []
    for annotations in grouped.values():
        ids = sorted(annotations)
        if len(ids) < 2:
            continue
        a, b = annotations[ids[0]], annotations[ids[1]]
        sa, sb = _supported_label(a.get("supported", "")), _supported_label(b.get("supported", ""))
        if sa is not None and sb is not None:
            supported_pairs.append((sa, sb))
        ca = "yes" if (a.get("contradiction") or "").strip().lower() in {"yes", "y", "true", "1"} else "no"
        cb = "yes" if (b.get("contradiction") or "").strip().lower() in {"yes", "y", "true", "1"} else "no"
        contradiction_pairs.append((ca, cb))
    return {"status": "available" if supported_pairs or contradiction_pairs else "insufficient_duplicate_annotations", "rows": total, "paired_items": max(len(supported_pairs), len(contradiction_pairs)), "supported_kappa": _cohen_kappa(supported_pairs), "contradiction_kappa": _cohen_kappa(contradiction_pairs), "supported_pair_count": len(supported_pairs), "contradiction_pair_count": len(contradiction_pairs)}
