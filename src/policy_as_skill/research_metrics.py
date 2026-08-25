from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from .evaluators import apply_run_normalization as _legacy_apply_run_normalization
from .evaluators import evaluate as _legacy_evaluate
from .evaluators import expected_review as _legacy_expected_review

DECISION_LABELS = ("allowed", "not_allowed", "conditional", "needs_review", "unknown")


def evaluate(task, trace: dict, manual_annotations: dict | None = None) -> dict:
    """Add format-neutral task metrics while preserving the legacy composite."""
    base = _legacy_evaluate(task, trace, manual_annotations=manual_annotations)
    expected_decision = str(task.expected_decision or "unknown")
    predicted_decision = str(trace.get("decision", "unknown"))
    expected_review = bool(task.expected_human_review) if task.expected_human_review is not None else bool(_legacy_expected_review(task))
    predicted_review = bool(trace.get("human_review_required", False))
    base["expected_decision"] = expected_decision
    base["predicted_decision"] = predicted_decision
    base["decision_exact_match"] = 1.0 if predicted_decision == expected_decision else 0.0
    base["expected_human_review"] = expected_review
    base["predicted_human_review"] = predicted_review
    base["review_exact_match"] = 1.0 if predicted_review == expected_review else 0.0
    base["decision_agreement_graded"] = float(base.get("decision_accuracy", 0.0))
    return base


def apply_run_normalization(rows: list[dict]) -> list[dict]:
    rows = _legacy_apply_run_normalization(rows)
    for row in rows:
        row["governance_readiness_index"] = float(row.get("normalized_score", 0.0))
    return rows


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def decision_confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix = {e: {p: 0 for p in DECISION_LABELS} for e in DECISION_LABELS}
    for r in rows:
        e = str(r.get("expected_decision", "unknown"))
        p = str(r.get("predicted_decision", "unknown"))
        matrix.setdefault(e, {x: 0 for x in DECISION_LABELS})
        if p not in matrix[e]:
            for row in matrix.values():
                row.setdefault(p, 0)
        matrix[e][p] += 1
    return matrix


def macro_f1(rows: list[dict]) -> float:
    labels = sorted(set(DECISION_LABELS) | {str(r.get("expected_decision", "unknown")) for r in rows})
    scores = []
    for label in labels:
        tp = sum(1 for r in rows if r.get("expected_decision") == label and r.get("predicted_decision") == label)
        fp = sum(1 for r in rows if r.get("expected_decision") != label and r.get("predicted_decision") == label)
        fn = sum(1 for r in rows if r.get("expected_decision") == label and r.get("predicted_decision") != label)
        if tp + fp + fn == 0:
            continue
        scores.append(_f1(tp, fp, fn)[2])
    return mean(scores) if scores else 0.0


def bootstrap_ci(values: list[float], *, seed: int = 7, iterations: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(max(100, iterations)):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return lo, hi


def method_summary(rows: list[dict], *, seed: int = 7, bootstrap_iterations: int = 1000) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("method", ""))].append(r)
    out: dict[str, dict[str, Any]] = {}
    for method, mr in grouped.items():
        exact = [float(r.get("decision_exact_match", 0.0)) for r in mr]
        expected_review = [bool(r.get("expected_human_review", False)) for r in mr]
        predicted_review = [bool(r.get("predicted_human_review", False)) for r in mr]
        tp = sum(1 for e, p in zip(expected_review, predicted_review) if e and p)
        fp = sum(1 for e, p in zip(expected_review, predicted_review) if not e and p)
        fn = sum(1 for e, p in zip(expected_review, predicted_review) if e and not p)
        rp, rr, rf1 = _f1(tp, fp, fn)
        lo, hi = bootstrap_ci(exact, seed=seed, iterations=bootstrap_iterations)
        out[method] = {
            "n": len(mr),
            "decision_exact_accuracy": mean(exact) if exact else 0.0,
            "decision_exact_accuracy_ci95": [lo, hi],
            "decision_macro_f1": macro_f1(mr),
            "review_precision": rp,
            "review_recall": rr,
            "review_f1": rf1,
            "review_exact_accuracy": mean(float(r.get("review_exact_match", 0.0)) for r in mr),
            "citation_precision": mean(float(r.get("citation_precision", 0.0)) for r in mr),
            "policy_ref_recall": mean(float(r.get("policy_ref_recall", 0.0)) for r in mr),
            "evidence_faithfulness": mean(float(r.get("evidence_faithfulness", 0.0)) for r in mr),
            "unsupported_claim_rate": mean(float(r.get("unsupported_claim_rate", 0.0)) for r in mr),
            "audit_completeness": mean(float(r.get("audit_completeness", 0.0)) for r in mr),
            "traceability_score": mean(float(r.get("traceability_score", 0.0)) for r in mr),
            "governance_quality_score": mean(float(r.get("governance_quality_score", 0.0)) for r in mr),
            "governance_readiness_index": mean(float(r.get("governance_readiness_index", 0.0)) for r in mr),
            "latency_seconds_mean": mean(float(r.get("latency_seconds", 0.0)) for r in mr),
            "latency_seconds_std": pstdev(float(r.get("latency_seconds", 0.0)) for r in mr) if len(mr) > 1 else 0.0,
            "confusion_matrix": decision_confusion(mr),
        }
    return out


def _weight_grid(step: float = 0.1):
    units = max(2, int(round(1.0 / step)))
    for d in range(units + 1):
        for e in range(units - d + 1):
            for g in range(units - d - e + 1):
                s = units - d - e - g
                yield d / units, e / units, g / units, s / units


def sensitivity_analysis(rows: list[dict], *, step: float = 0.1) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("method", ""))].append(r)
    components: dict[str, tuple[float, float, float, float]] = {}
    for method, mr in grouped.items():
        components[method] = (
            mean(float(r.get("decision_quality_score", 0.0)) for r in mr),
            mean(float(r.get("evidence_quality_score", 0.0)) for r in mr),
            mean(float(r.get("governance_quality_score", 0.0)) for r in mr),
            mean(float(r.get("answer_similarity", 0.0)) for r in mr),
        )
    top_counts = {m: 0.0 for m in components}
    ranks = {m: [] for m in components}
    weightings = 0
    for weights in _weight_grid(step):
        scores = {m: sum(w * c for w, c in zip(weights, comps)) for m, comps in components.items()}
        ordered = sorted(scores, key=lambda m: (-scores[m], m))
        if not ordered:
            continue
        weightings += 1
        best = scores[ordered[0]]
        winners = [m for m in ordered if math.isclose(scores[m], best, rel_tol=1e-12, abs_tol=1e-12)]
        for m in winners:
            top_counts[m] += 1.0 / len(winners)
        for rank, m in enumerate(ordered, start=1):
            ranks[m].append(rank)
    return {
        "step": step,
        "weightings_evaluated": weightings,
        "components": {m: {"decision": c[0], "evidence": c[1], "governance": c[2], "answer_similarity": c[3]} for m, c in components.items()},
        "top_rank_share": {m: _safe_div(v, weightings) for m, v in top_counts.items()},
        "mean_rank": {m: mean(v) if v else None for m, v in ranks.items()},
        "rank_std": {m: pstdev(v) if len(v) > 1 else 0.0 for m, v in ranks.items()},
    }


def readiness_threshold_sensitivity(rows: list[dict], thresholds=(0.65, 0.70, 0.72, 0.75, 0.80)) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("method", ""))].append(r)
    return {
        method: {
            f"{threshold:.2f}": mean(1.0 if float(r.get("governance_readiness_index", 0.0)) >= threshold else 0.0 for r in mr)
            for threshold in thresholds
        }
        for method, mr in grouped.items()
    }
