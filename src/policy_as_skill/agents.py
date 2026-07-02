from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .data_loader import BenchmarkTask
from .ollama_client import OllamaClient
from .retrieval import PolicyChunk, PolicyRetriever
from .skills import PolicySkill, choose_skill, review_required_by_skill
from .utils import first_sentence, now, stable_hash, token_similarity, tokens

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"allowed", "not_allowed", "conditional", "needs_review", "unknown"}


def evidence_text(chunks: list[PolicyChunk]) -> str:
    return "\n\n".join(
        f"[{c.citation_id} score={c.score:.3f} tags={','.join(c.tags)} sha={c.sha256}]\n{c.text}" for c in chunks
    )


def evidence_json(chunks: list[PolicyChunk]) -> str:
    return json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False)


def parse_model_json(raw: str) -> dict[str, Any] | None:
    if not raw or raw.startswith("OLLAMA_"):
        return None
    raw = raw.strip()
    candidates = [raw]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if m:
        candidates.insert(0, m.group(1))
    m = re.search(r"(\{.*\})", raw, re.S)
    if m:
        candidates.append(m.group(1))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _decision_from_text(question: str, evidence: str, skill: PolicySkill | None = None) -> str:
    q = question.lower()
    e = evidence.lower()
    not_allowed_markers = ["not allowed", "must not", "excluded", "no,", "cannot", "not without", "without explicit"]
    conditional_markers = ["may", "if", "requires", "required", "conditional", "approval", "review", "safeguards"]
    if any(x in e for x in not_allowed_markers) or ("without" in q and "external" in q):
        return "not_allowed"
    if skill and skill.name in {"RiskClassificationSkill", "HumanReviewRoutingSkill", "ConflictDetectionSkill"}:
        return "needs_review" if review_required_by_skill(skill, question) else "conditional"
    if any(x in e for x in conditional_markers):
        return "conditional"
    return "allowed" if evidence.strip() else "unknown"


def _extractive_answer(method: str, task: BenchmarkTask, chunks: list[PolicyChunk], skill: PolicySkill | None = None) -> dict[str, Any]:
    """Deterministic offline answer. It never uses expected_answer."""
    e = "\n".join(c.text for c in chunks)
    review = review_required_by_skill(skill, task.question) if skill else _generic_review_required(task.question)
    decision = _decision_from_text(task.question, e, skill)
    selected = []
    q_terms = tokens(task.question)
    for c in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", c.text.replace("\n", " "))
        ranked = sorted(sentences, key=lambda s: token_similarity(" ".join(q_terms), s), reverse=True)
        if ranked and ranked[0].strip():
            selected.append(first_sentence(ranked[0], 220))
    if selected:
        answer = " ".join(selected[:2])
    else:
        answer = "Insufficient policy evidence was retrieved to make a reliable decision."
        decision = "unknown"
        review = True
    if method == "Keyword Search":
        reasoning = "Lexical overlap selected policy evidence; no agentic validation was performed."
    elif skill:
        reasoning = f"Applied {skill.name} with scoped retrieval and deterministic policy controls."
    else:
        reasoning = "Retrieved policy evidence was summarized without skill-level governance controls."
    return {
        "answer": answer,
        "decision": decision,
        "reasoning_summary": reasoning,
        "citations": [c.citation_id for c in chunks[:3]],
        "human_review_required": bool(review),
        "confidence": 0.55 if selected else 0.15,
        "risks": _risk_labels(task.question, e),
        "missing_information": [] if selected else ["policy evidence"],
    }


def _risk_labels(question: str, evidence: str) -> list[str]:
    text = f"{question} {evidence}".lower()
    labels = []
    for label, words in {
        "sensitive-data": ["citizen", "image", "personal", "sensitive"],
        "external-cloud": ["external", "cloud", "vendor"],
        "high-risk-public-service": ["benefit", "eligibility", "essential service", "ranking"],
        "policy-conflict": ["conflict", "stricter", "less strict"],
        "auditability": ["audit", "trace", "citation", "version"],
    }.items():
        if any(w in text for w in words):
            labels.append(label)
    return labels


def _generic_review_required(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in ["citizen", "image", "sensitive", "external", "benefit", "eligibility", "conflict", "high-risk", "vulnerable"])


def normalize_decision(obj: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)
    if not obj:
        return out
    if isinstance(obj.get("answer"), str) and obj["answer"].strip():
        out["answer"] = obj["answer"].strip()
    if isinstance(obj.get("decision"), str):
        d = obj["decision"].strip().lower().replace(" ", "_").replace("-", "_")
        out["decision"] = d if d in VALID_DECISIONS else out["decision"]
    if isinstance(obj.get("reasoning_summary"), str) and obj["reasoning_summary"].strip():
        out["reasoning_summary"] = obj["reasoning_summary"].strip()
    if isinstance(obj.get("citations"), list):
        out["citations"] = [str(x) for x in obj["citations"] if str(x).strip()]
    if "human_review_required" in obj:
        out["human_review_required"] = bool(obj["human_review_required"])
    if "confidence" in obj:
        try:
            out["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
        except Exception:
            pass
    for key in ["risks", "missing_information"]:
        if isinstance(obj.get(key), list):
            out[key] = [str(x) for x in obj[key]]
    return out


def validate_output(out: dict[str, Any], chunks: list[PolicyChunk], skill: PolicySkill | None, strict: bool) -> dict[str, Any]:
    evidence_ids = {c.citation_id for c in chunks}
    short_ids = {f"{c.source}#{c.chunk_id}" for c in chunks}
    valid_cites = []
    invalid_cites = []
    for cit in out.get("citations", []):
        if cit in evidence_ids:
            valid_cites.append(cit)
        elif cit in short_ids:
            # normalize short citation to versioned citation
            for c in chunks:
                if cit == f"{c.source}#{c.chunk_id}":
                    valid_cites.append(c.citation_id)
                    break
        else:
            invalid_cites.append(cit)

    missing_tags: list[str] = []
    if skill:
        available_tags = set(t for c in chunks for t in c.tags)
        missing_tags = [t for t in skill.required_evidence_tags if t not in available_tags]

    review_by_trigger = review_required_by_skill(skill, out.get("question", "")) if skill else bool(out.get("human_review_required"))
    if strict:
        if not valid_cites and chunks:
            valid_cites = [chunks[0].citation_id]
        if missing_tags:
            out["human_review_required"] = True
            if out.get("decision") == "allowed":
                out["decision"] = "needs_review"
            out.setdefault("missing_information", [])
            out["missing_information"] = sorted(set(out["missing_information"] + [f"missing evidence tag: {t}" for t in missing_tags]))
    out["citations"] = valid_cites if strict else out.get("citations", [])
    return {
        "valid_citations": valid_cites,
        "invalid_citations": invalid_cites,
        "missing_required_evidence_tags": missing_tags,
        "citation_validation_applied": strict,
        "review_by_trigger": review_by_trigger,
        "schema_valid": bool(out.get("answer") and out.get("decision") in VALID_DECISIONS),
    }


def _prompt_for(method: str, task: BenchmarkTask, chunks: list[PolicyChunk], skill: PolicySkill | None, retriever: PolicyRetriever) -> str:
    schema = "Return only JSON with answer, decision, reasoning_summary, citations, human_review_required, confidence, risks, missing_information."
    if method == "Direct LLM":
        return f"{schema}\nQuestion: {task.question}\nNo policy context is provided. State uncertainty if needed."
    if method == "Standard RAG":
        return f"{schema}\nAnswer with citations from the provided context only.\nQuestion: {task.question}\nContext:\n{evidence_text(chunks)}"
    if method == "Hybrid RAG":
        return f"{schema}\nUse hybrid retrieved evidence only.\nQuestion: {task.question}\nEvidence:\n{evidence_text(chunks)}"
    if method == "Policy-as-Prompt":
        return f"Policy-as-Prompt: apply the policy text directly. {schema}\nTask type: {task.task_type}\nQuestion: {task.question}\nPolicies:\n{evidence_text(chunks)}"
    if method == "Structured Policy-as-Prompt":
        return f"Policy-as-Prompt with structured policy artifacts. {schema}\nTask type: {task.task_type}\nQuestion: {task.question}\nPolicy artifacts JSON:\n{evidence_json(chunks)}"
    if method in {"Policy-as-Skill", "Policy-as-Skill No Audit"} and skill:
        return f"{skill.prompt_template}\nSelected skill metadata:\n{json.dumps(skill.to_public_dict(), indent=2)}\nQuestion: {task.question}\nTrusted evidence JSON:\n{evidence_json(chunks)}"
    return f"{schema}\nQuestion: {task.question}\nContext:\n{evidence_text(chunks)}"


def _select_evidence(method: str, task: BenchmarkTask, retriever: PolicyRetriever, top_k: int) -> tuple[list[PolicyChunk], PolicySkill | None]:
    skill = choose_skill(task.task_type, task.question) if method in {"Policy-as-Skill", "Policy-as-Skill No Audit"} else None
    if method == "Direct LLM":
        return [], None
    if method == "Keyword Search":
        return retriever.retrieve(task.question, top_k=top_k, mode="keyword"), None
    if method == "Hybrid RAG":
        return retriever.retrieve(task.question, top_k=top_k, mode="hybrid"), None
    if method == "Policy-as-Prompt":
        return retriever.retrieve(task.question, top_k=top_k, mode="bm25"), None
    if method == "Structured Policy-as-Prompt":
        return retriever.retrieve(task.question, top_k=top_k, mode="hybrid"), None
    if method in {"Policy-as-Skill", "Policy-as-Skill No Audit"} and skill:
        chunks = retriever.retrieve(task.question, top_k=top_k, scope_terms=skill.retrieval_scope, mode="hybrid")
        if method == "Policy-as-Skill":
            # Evidence validation expansion: add more chunks if required tags are missing.
            available = set(t for c in chunks for t in c.tags)
            missing = [t for t in skill.required_evidence_tags if t not in available]
            if missing:
                expanded = retriever.retrieve(task.question + " " + " ".join(missing), top_k=top_k + 3, scope_terms=skill.retrieval_scope + missing, mode="hybrid")
                by_id = {c.citation_id: c for c in chunks}
                for c in expanded:
                    by_id.setdefault(c.citation_id, c)
                chunks = list(by_id.values())[: top_k + 3]
        return chunks, skill
    return retriever.retrieve(task.question, top_k=top_k, mode="bm25"), None


def run_method(method: str, task: BenchmarkTask, retriever: PolicyRetriever, client: OllamaClient, top_k: int = 5) -> dict[str, Any]:
    start = time.perf_counter()
    chunks, skill = _select_evidence(method, task, retriever, top_k)
    fallback = _extractive_answer(method, task, chunks, skill)
    prompt = _prompt_for(method, task, chunks, skill, retriever)
    raw = ""
    parsed = None

    if method != "Keyword Search":
        raw = client.generate(prompt, {"method": method, "task_id": task.id, "expect_json": True})
        parsed = parse_model_json(raw)
    out = normalize_decision(parsed or {}, fallback)
    out["question"] = task.question

    strict_validation = method == "Policy-as-Skill"
    validation = validate_output(out, chunks, skill, strict=strict_validation)
    if method == "Policy-as-Skill" and skill:
        # Enforce policy-skill review routing after model output.
        out["human_review_required"] = bool(out.get("human_review_required")) or review_required_by_skill(skill, task.question)
    if method == "Policy-as-Skill No Audit":
        # Ablation: uses skill prompt/scoped retrieval but deliberately omits the
        # governance controls that make the full method auditable.
        validation = {"schema_valid": validation.get("schema_valid", False), "citation_validation_applied": False, "ablation": "audit_controls_removed"}

    policy_hashes = sorted({f"{c.source}@{c.version}:{c.sha256}" for c in chunks})
    if method == "Policy-as-Skill No Audit":
        policy_hashes = []
    trace: dict[str, Any] = {
        "timestamp": now(),
        "method": method,
        "task_id": task.id,
        "task_type": task.task_type,
        "question": task.question,
        "selected_skill": skill.name if skill else None,
        "policy_skill_version": skill.version if skill else None,
        "skill_metadata": skill.to_public_dict() if skill else None,
        "prompt_version": "prompt-v2.1" if method != "Keyword Search" else "keyword-v1",
        "prompt_hash": stable_hash(prompt),
        "policy_hashes": policy_hashes,
        "evidence": [c.to_dict() for c in chunks],
        "raw_model_output": raw,
        "parsed_model_output": parsed,
        "validation": validation,
        **out,
        "latency_seconds": time.perf_counter() - start,
    }
    logger.debug("Trace generated task_id=%s method=%s decision=%s", task.id, method, trace.get("decision"))
    return trace
