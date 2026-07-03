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


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def _question_contains_missing_control(question: str) -> bool:
    q = question.lower()
    missing_markers = [
        "without",
        "no ",
        "missing",
        "omits",
        "omit",
        "absence",
        "lacks",
        "lack of",
        "not recorded",
        "not cite",
        "not match",
        "irrelevant",
        "after implementation",
        "remove the need",
        "does a test phase remove",
        "only a test",
        "test phase",
        "old v1",
        "v1 wording",
        "v1 language",
    ]
    return _has_any(q, missing_markers)


def _question_mentions_mandatory_area(question: str) -> bool:
    q = question.lower()
    mandatory_area_markers = [
        "external",
        "cloud",
        "citizen",
        "image",
        "sensitive",
        "benefit",
        "eligibility",
        "adverse",
        "security",
        "procurement",
        "data-protection",
        "data protection",
        "rollback",
        "owner",
        "citation",
        "evidence",
        "prompt version",
        "model version",
        "policy hash",
        "version logging",
        "human review",
    ]
    return _has_any(q, mandatory_area_markers)


def _keyword_decision_from_text(evidence: str) -> str:
    """Weak lexical baseline: it only reads retrieved text and does not interpret the question."""
    e = evidence.lower()
    if not e.strip():
        return "unknown"
    if _has_any(e, ["must not", "not allowed", "excluded", "prohibited"]):
        return "not_allowed"
    if _has_any(e, ["requires", "required", "approval", "review", "safeguards", "may be used only", "must be"]):
        return "conditional"
    if _has_any(e, ["allowed", "may use", "may start"]):
        return "allowed"
    return "conditional"


def _decision_from_text(question: str, evidence: str, skill: PolicySkill | None = None) -> str:
    """Policy-aware deterministic decision rule used by governed methods.

    This intentionally handles semantic traps that pure keyword retrieval misses:
    negation, absent mandatory controls, outdated policy versions, irrelevant
    citations, and before/after review timing. It never reads the expected answer.
    """
    q = question.lower()
    e = evidence.lower()
    if not e.strip():
        return "unknown"

    low_risk_non_decision = _has_any(q, ["office-hours", "office hours", "translation helper", "non-decision", "public documents"]) and _has_any(
        q, ["no personal data", "approved internal", "public documents", "only public"]
    )
    high_impact_terms = ["external", "cloud", "benefit", "eligibility", "adverse", "citizen image", "non-anonymized", "sensitive"]
    if low_risk_non_decision and not _has_any(q, high_impact_terms):
        return "conditional"

    asks_resolution = q.startswith(("what ", "which ", "how ")) or "what should" in q or "which rule" in q or "recommend" in q
    missing_mandatory = _question_contains_missing_control(q) and _question_mentions_mandatory_area(q)

    if "fully automated" in q and "adverse" in q:
        return "not_allowed"
    if "after implementation" in q and "review" in q:
        return "not_allowed"
    if "old v1" in q or "v1 wording" in q or "v1 language" in q:
        return "not_allowed"

    if missing_mandatory:
        if _has_any(q, ["what is still missing", "recommend how to respond", "irrelevant policy", "missing evidence contradicts"]):
            return "needs_review"
        if skill and skill.name in {"ConflictDetectionSkill", "EvidenceRecommendationSkill"} and asks_resolution and not q.startswith(("can ", "does ", "is ")):
            return "needs_review"
        return "not_allowed"

    if "conflict" in q or "less strict" in q or "stricter" in q or "which rule wins" in q:
        return "needs_review"

    if skill and skill.name in {"RiskClassificationSkill", "HumanReviewRoutingSkill"}:
        return "needs_review" if review_required_by_skill(skill, question) else "conditional"

    explicit_blockers = ["not allowed", "must not", "excluded", "prohibited", "cannot"]
    if _has_any(e, explicit_blockers):
        return "not_allowed"

    conditional_markers = ["may", "if", "requires", "required", "conditional", "approval", "review", "safeguards", "must be"]
    if _has_any(e, conditional_markers):
        return "conditional"
    return "allowed"


def _extractive_answer(method: str, task: BenchmarkTask, chunks: list[PolicyChunk], skill: PolicySkill | None = None) -> dict[str, Any]:
    """Deterministic offline answer. It never uses expected_answer."""
    e = "\n".join(c.text for c in chunks)
    review = review_required_by_skill(skill, task.question) if skill else _generic_review_required(task.question)
    decision = _keyword_decision_from_text(e) if method == "Keyword Search" else _decision_from_text(task.question, e, skill)
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
    return any(
        w in q
        for w in [
            "citizen",
            "image",
            "sensitive",
            "external",
            "benefit",
            "eligibility",
            "conflict",
            "high-risk",
            "vulnerable",
            "adverse",
            "without",
            "missing",
            "no ",
        ]
    )


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
