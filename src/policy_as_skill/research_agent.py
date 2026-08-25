from __future__ import annotations

import time
from typing import Any

from .agents import VALID_DECISIONS, _prompt_for, _select_evidence, canonicalize_decision, normalize_decision, parse_model_json, run_method as _legacy_run_method
from .data_loader import BenchmarkTask
from .governance import assess_policy
from .ollama_client import OllamaClient
from .retrieval import PolicyChunk, PolicyRetriever
from .skills import PolicySkill
from .utils import first_sentence, now, stable_hash, token_similarity, tokens

PAS_RETRIEVAL = "Policy-as-Skill Retrieval"
PAS_CONTROLLER = "Policy-as-Skill + Controller"
PAS_AUDIT = "Policy-as-Skill + Audit"
PAS_FULL = "Policy-as-Skill"
PAS_ALIASES = {"Policy-as-Skill No Audit": PAS_CONTROLLER}
PAS_VARIANTS = {PAS_RETRIEVAL, PAS_CONTROLLER, PAS_AUDIT, PAS_FULL, *PAS_ALIASES}


def _research_fallback(task: BenchmarkTask, chunks: list[PolicyChunk], *, use_controller: bool) -> dict[str, Any]:
    """Offline-only fallback that preserves the requested ablation semantics."""
    selected = []
    q = " ".join(tokens(task.question))
    for c in chunks:
        candidates = [s.strip() for s in c.text.replace("\n", " ").split(".") if s.strip()]
        ranked = sorted(candidates, key=lambda s: token_similarity(q, s), reverse=True)
        if ranked:
            selected.append(first_sentence(ranked[0] + ".", 240))
    if use_controller:
        assessment = assess_policy(task.task_type, task.question, [c.to_dict() for c in chunks])
        decision = assessment.decision
        review = assessment.human_review_required
        confidence = assessment.confidence if chunks else 0.15
        reasoning = assessment.rationale
    else:
        decision = "unknown"
        review = False
        confidence = 0.20 if chunks else 0.10
        reasoning = "Offline fallback for a no-controller ablation; evidence is returned but no governed decision is inferred."
    return {
        "answer": " ".join(selected[:2]) if selected else "Insufficient trusted policy evidence was retrieved.",
        "decision": decision,
        "reasoning_summary": reasoning,
        "citations": [c.citation_id for c in chunks[:3]],
        "human_review_required": review,
        "confidence": confidence,
        "risks": [],
        "missing_information": [] if chunks else ["policy evidence"],
    }


def _validate_citations(out: dict[str, Any], chunks: list[PolicyChunk], skill: PolicySkill | None, *, strict: bool) -> dict[str, Any]:
    evidence_ids = {c.citation_id for c in chunks}
    short_map = {f"{c.source}#{c.chunk_id}": c.citation_id for c in chunks}
    valid, invalid = [], []
    for citation in out.get("citations", []) or []:
        c = str(citation)
        if c in evidence_ids:
            valid.append(c)
        elif c in short_map:
            valid.append(short_map[c])
        else:
            invalid.append(c)
    available_tags = {tag for c in chunks for tag in c.tags}
    missing_tags = [tag for tag in (skill.required_evidence_tags if skill else []) if tag not in available_tags]
    if strict:
        out["citations"] = valid
        if chunks and not valid:
            out["citations"] = [chunks[0].citation_id]
        if missing_tags:
            out.setdefault("missing_information", [])
            out["missing_information"] = sorted(set(out["missing_information"] + [f"missing evidence tag: {tag}" for tag in missing_tags]))
    return {"schema_valid": bool(out.get("answer") and out.get("decision") in VALID_DECISIONS), "citation_validation_applied": strict, "valid_citations": valid, "invalid_citations": invalid, "missing_required_evidence_tags": missing_tags}


def _pas_flags(method: str) -> tuple[bool, bool]:
    canonical = PAS_ALIASES.get(method, method)
    return canonical in {PAS_CONTROLLER, PAS_FULL}, canonical in {PAS_AUDIT, PAS_FULL}


def _run_pas_variant(method: str, task: BenchmarkTask, retriever: PolicyRetriever, client: OllamaClient, top_k: int) -> dict[str, Any]:
    start = time.perf_counter()
    canonical_method = PAS_ALIASES.get(method, method)
    use_controller, use_audit = _pas_flags(method)
    chunks, skill = _select_evidence("Policy-as-Skill", task, retriever, top_k)
    fallback = _research_fallback(task, chunks, use_controller=use_controller)
    prompt = _prompt_for("Policy-as-Skill", task, chunks, skill, retriever)
    raw = client.generate(prompt, {"method": canonical_method, "task_id": task.id, "expect_json": True})
    parsed = parse_model_json(raw)
    out = normalize_decision(parsed or {}, fallback)
    out["question"] = task.question
    model_decision = canonicalize_decision(out.get("decision"), fallback="unknown")
    model_review = bool(out.get("human_review_required", False))
    controller = None
    if use_controller:
        controller = assess_policy(task.task_type, task.question, [c.to_dict() for c in chunks], model_decision=model_decision, model_review=model_review)
        out["decision"] = controller.decision
        out["human_review_required"] = controller.human_review_required
        out["confidence"] = min(float(out.get("confidence", 0.5)), controller.confidence)
    validation = _validate_citations(out, chunks, skill, strict=use_audit)
    validation.update({
        "research_ablation": canonical_method,
        "generic_policy_controller": use_controller,
        "audit_controls_enabled": use_audit,
        "legacy_benchmark_phrase_controller_used": False,
        "model_decision_before_control": model_decision,
        "model_review_before_control": model_review,
        "offline_fallback_preserves_ablation": True,
    })
    if controller is not None:
        validation["controller_assessment"] = controller.to_dict()
    trace = {
        "timestamp": now(), "method": canonical_method, "task_id": task.id, "task_type": task.task_type, "question": task.question,
        "selected_skill": skill.name if skill else None, "policy_skill_version": skill.version if skill else None, "skill_metadata": skill.to_public_dict() if skill else None,
        "prompt_version": "research-prompt-v3", "prompt_hash": stable_hash(prompt) if use_audit else "",
        "policy_hashes": sorted({f"{c.source}@{c.version}:{c.sha256}" for c in chunks}) if use_audit else [],
        "evidence": [c.to_dict() for c in chunks], "raw_model_output": raw, "llm_provider": "ollama_or_offline", "parsed_model_output": parsed, "validation": validation,
        **out, "latency_seconds": time.perf_counter() - start,
    }
    return trace


def run_method(method: str, task: BenchmarkTask, retriever: PolicyRetriever, client: OllamaClient, top_k: int = 5) -> dict[str, Any]:
    """Run baselines unchanged and Policy-as-Skill through clean research ablations."""
    if method in PAS_VARIANTS:
        return _run_pas_variant(method, task, retriever, client, top_k)
    return _legacy_run_method(method, task, retriever, client, top_k)
