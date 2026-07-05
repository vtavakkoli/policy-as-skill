from __future__ import annotations

from .data_loader import BenchmarkTask
from .utils import containment, token_similarity


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def expected_review(task: BenchmarkTask) -> bool:
    if task.expected_human_review is not None:
        return bool(task.expected_human_review)
    return task.task_type in {"compliance_check", "risk_classification", "policy_conflict_detection", "human_review_routing", "policy_update_adaptation"} or any(
        x in task.question.lower() for x in ["citizen", "adverse", "external", "vulnerable", "benefits", "sensitive"]
    )


def _expected_decision(task: BenchmarkTask) -> str:
    if task.expected_decision:
        return task.expected_decision
    x = task.expected_answer.lower()
    if x.startswith("no") or "not allowed" in x or "must not" in x:
        return "not_allowed"
    if x.startswith("yes") or "allowed" in x:
        return "allowed"
    if "review" in x or "approval" in x or "requires" in x:
        return "needs_review"
    return "conditional"


def _citation_stats(task: BenchmarkTask, trace: dict) -> tuple[float, float, float]:
    citations = [str(c).lower() for c in trace.get("citations", [])]
    evidence = trace.get("evidence", []) or []
    evidence_ids = {str(e.get("citation_id", "")).lower() for e in evidence}
    evidence_text = " ".join(str(e.get("text", "")) + " " + " ".join(e.get("tags", [])) + " " + str(e.get("citation_id", "")) for e in evidence).lower()
    if not citations:
        precision = 0.0
    else:
        precision = sum(1 for c in citations if c in evidence_ids or any(c in eid for eid in evidence_ids)) / max(1, len(citations))
    expected = [r.lower() for r in task.expected_policy_refs]
    ref_recall = sum(1 for r in expected if r in " ".join(citations) or r in evidence_text or r in str(trace.get("answer", "")).lower()) / max(1, len(expected))
    citation_coverage = 1.0 if citations else 0.0
    return citation_coverage, precision, ref_recall


def _audit_completeness(trace: dict) -> float:
    fields = [
        "timestamp",
        "task_id",
        "method",
        "question",
        "evidence",
        "citations",
        "decision",
        "reasoning_summary",
        "human_review_required",
        "confidence",
        "validation",
        "policy_hashes",
        "prompt_hash",
        "selected_skill",
        "policy_skill_version",
        "skill_metadata",
    ]
    score = sum(1 for f in fields if f in trace and trace.get(f) not in [None, "", []]) / len(fields)
    validation = trace.get("validation") or {}
    if validation.get("citation_validation_applied"):
        score = min(1.0, score + 0.05)
    return score


def _traceability(trace: dict) -> float:
    checks = [
        bool(trace.get("evidence")),
        bool(trace.get("reasoning_summary")),
        bool(trace.get("decision")),
        "human_review_required" in trace,
        bool(trace.get("policy_hashes")),
        bool(trace.get("prompt_hash")),
        bool(trace.get("validation")),
    ]
    return sum(checks) / len(checks)


def _unsupported_claim_rate(trace: dict) -> float:
    answer = str(trace.get("answer", ""))
    evidence_text = " ".join(str(e.get("text", "")) for e in trace.get("evidence", []) or [])
    if not answer.strip():
        return 1.0
    support = containment(answer, evidence_text)
    return _clip01(1.0 - support)


def _contradiction_rate(task: BenchmarkTask, trace: dict) -> float:
    exp = _expected_decision(task)
    got = str(trace.get("decision", "unknown"))
    if got == exp:
        return 0.0
    compatible = {("conditional", "needs_review"), ("needs_review", "conditional")}
    return 0.35 if (exp, got) in compatible else 1.0


def _update_adaptation(task: BenchmarkTask, trace: dict) -> float:
    if task.task_type != "policy_update_adaptation" and not task.policy_version:
        return 1.0
    expected_version = (task.policy_version or "v2").lower()
    citations = " ".join(str(c) for c in trace.get("citations", [])).lower()
    evidence = " ".join(str(e.get("citation_id", "")) + " " + str(e.get("version", "")) for e in trace.get("evidence", []) or []).lower()
    return 1.0 if expected_version in citations or expected_version in evidence else 0.0


def _task_success(score: float, decision_accuracy: float, human_review_correctness: float, evidence_quality: float, governance_quality: float) -> float:
    success = score >= 0.72 and decision_accuracy >= 0.65 and human_review_correctness >= 0.90 and evidence_quality >= 0.60 and governance_quality >= 0.75
    return 1.0 if success else 0.0


def _quality_components(
    sim: float,
    citation_precision: float,
    policy_ref_recall: float,
    faithfulness: float,
    unsupported: float,
    contradiction: float,
    traceability: float,
    hrc: float,
    audit: float,
    update: float,
    governance: float,
) -> tuple[float, float, float, float]:
    decision_accuracy = _clip01(1.0 - contradiction)
    decision_quality = _clip01(0.70 * decision_accuracy + 0.30 * hrc)
    evidence_quality = _clip01(0.30 * citation_precision + 0.30 * policy_ref_recall + 0.25 * faithfulness + 0.15 * (1.0 - unsupported))
    governance_quality = _clip01(0.30 * traceability + 0.30 * audit + 0.20 * governance + 0.20 * update)
    raw_quality = _clip01(0.40 * decision_quality + 0.25 * evidence_quality + 0.25 * governance_quality + 0.10 * sim)
    # Strong penalty for semantically wrong decisions. This prevents a fast
    # lexical baseline from winning solely through citation overlap.
    raw_quality = _clip01(raw_quality * (1.0 - 0.25 * contradiction))
    return decision_quality, evidence_quality, governance_quality, raw_quality


def evaluate(task: BenchmarkTask, trace: dict, manual_annotations: dict | None = None) -> dict:
    answer = str(trace.get("answer", ""))
    sim = token_similarity(task.expected_answer, answer)
    citation_coverage, citation_precision, policy_ref_recall = _citation_stats(task, trace)
    evidence_text = " ".join(str(e.get("text", "")) for e in trace.get("evidence", []) or [])
    faithfulness = containment(answer, evidence_text) if evidence_text else 0.0
    manual = (manual_annotations or {}).get((task.id, str(trace.get("method", ""))), {})
    if manual:
        faithfulness = float(manual.get("manual_evidence_faithfulness", faithfulness))
    unsupported = _unsupported_claim_rate(trace)
    contradiction = _contradiction_rate(task, trace)
    if manual and manual.get("manual_contradiction_rate") is not None:
        contradiction = max(contradiction, float(manual.get("manual_contradiction_rate", 0.0)))
    decision_accuracy = _clip01(1.0 - contradiction)
    traceability = _traceability(trace)
    hrc = 1.0 if bool(trace.get("human_review_required")) == expected_review(task) else 0.0
    audit = _audit_completeness(trace)
    update = _update_adaptation(task, trace)
    governance = _clip01(0.30 * traceability + 0.25 * audit + 0.15 * citation_precision + 0.20 * hrc + 0.10 * update)
    decision_quality, evidence_quality, governance_quality, raw_quality = _quality_components(
        sim, citation_precision, policy_ref_recall, faithfulness, unsupported, contradiction, traceability, hrc, audit, update, governance
    )
    success = _task_success(raw_quality, decision_accuracy, hrc, evidence_quality, governance_quality)
    return {
        "answer_similarity": round(sim, 6),
        "citation_coverage": round(citation_coverage, 6),
        "citation_precision": round(citation_precision, 6),
        "policy_ref_recall": round(policy_ref_recall, 6),
        "evidence_faithfulness": round(faithfulness, 6),
        "unsupported_claim_rate": round(unsupported, 6),
        "contradiction_rate": round(contradiction, 6),
        "decision_accuracy": round(decision_accuracy, 6),
        "traceability_score": round(traceability, 6),
        "human_review_correctness": round(hrc, 6),
        "audit_completeness": round(audit, 6),
        "governance_readiness_score": round(governance, 6),
        "update_adaptation_score": round(update, 6),
        "decision_quality_score": round(decision_quality, 6),
        "evidence_quality_score": round(evidence_quality, 6),
        "governance_quality_score": round(governance_quality, 6),
        "raw_quality_score": round(raw_quality, 6),
        "latency_efficiency_score": 0.0,
        "normalized_score": round(raw_quality, 6),
        "task_success": round(success, 6),
        "latency_seconds": round(float(trace.get("latency_seconds", 0)), 6),
        "overall_score": round(raw_quality, 6),
        "manual_annotation_count": int(manual.get("manual_annotation_count", 0)) if manual else 0,
    }


def apply_run_normalization(rows: list[dict]) -> list[dict]:
    """Add run-level normalized score after all latencies are known.

    All semantic/governance metrics are already in [0, 1]. Latency is the only
    unbounded metric, so it is converted to an inverse min-max efficiency score
    and given a small weight. This keeps the benchmark from declaring a lexical
    method best only because it is fast.
    """
    if not rows:
        return rows
    latencies = [float(r.get("latency_seconds", 0.0)) for r in rows]
    lo, hi = min(latencies), max(latencies)
    denom = hi - lo
    for r in rows:
        lat = float(r.get("latency_seconds", 0.0))
        efficiency = 1.0 if denom <= 1e-12 else 1.0 - ((lat - lo) / denom)
        raw = float(r.get("raw_quality_score", r.get("overall_score", 0.0)))
        normalized = _clip01(0.95 * raw + 0.05 * efficiency)
        r["latency_efficiency_score"] = round(efficiency, 6)
        r["normalized_score"] = round(normalized, 6)
        r["overall_score"] = round(normalized, 6)
        r["task_success"] = _task_success(
            normalized,
            float(r.get("decision_accuracy", 0.0)),
            float(r.get("human_review_correctness", 0.0)),
            float(r.get("evidence_quality_score", 0.0)),
            float(r.get("governance_quality_score", 0.0)),
        )
    return rows
