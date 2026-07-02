import json
import random
from collections import Counter
from pathlib import Path

from .data_loader import load_tasks_from_path


SCENARIOS = [
    {
        "type": "compliance_check",
        "stem": "Can a department upload non-anonymized citizen images to an external AI cloud service during a test phase?",
        "answer": "No, not without explicit legal basis, a data-protection review, security assessment, procurement approval, human oversight, and documented safeguards.",
        "refs": ["data-protection", "external-cloud", "human-review"],
        "decision": "not_allowed",
        "review": True,
        "difficulty": "hard",
    },
    {
        "type": "risk_classification",
        "stem": "Is an AI tool for ranking eligibility for public benefits a high-risk use?",
        "answer": "Yes. Public-benefit eligibility and access to essential services are high-risk uses and require risk management, logging, transparency, and human oversight.",
        "refs": ["high-risk-ai", "human-review", "transparency"],
        "decision": "needs_review",
        "review": True,
        "difficulty": "medium",
    },
    {
        "type": "policy_question_answering",
        "stem": "What must be logged for an AI-assisted public-sector recommendation?",
        "answer": "The audit trail should record the question, evidence, reasoning summary, decision, confidence, human-review flag, responsible official, timestamp, model version, prompt or skill version, and policy hashes.",
        "refs": ["audit-trail", "accountability", "model-governance"],
        "decision": "conditional",
        "review": False,
        "difficulty": "easy",
    },
    {
        "type": "policy_conflict_detection",
        "stem": "What happens when an internal department rule is less strict than the city-wide public-sector data rule?",
        "answer": "The stricter rule applies. The conflict should be recorded, escalated to the policy owner, and resolved before operational use.",
        "refs": ["policy-conflict", "data-protection", "accountability"],
        "decision": "needs_review",
        "review": True,
        "difficulty": "hard",
    },
    {
        "type": "human_review_routing",
        "stem": "Should a low-risk office-hours chatbot without sensitive data require mandatory human review for every answer?",
        "answer": "Usually no. It may operate with monitoring and fallback if it is limited to low-risk administrative information, uses no sensitive data, and has rollback procedures.",
        "refs": ["pilot-deployment", "human-review"],
        "decision": "conditional",
        "review": False,
        "difficulty": "medium",
    },
    {
        "type": "evidence_grounded_recommendation",
        "stem": "What should a team do before using an external AI service with sensitive personal data?",
        "answer": "They should perform data-protection, security, procurement, and risk reviews; document safeguards; restrict access; and route the case for human approval.",
        "refs": ["external-cloud", "data-protection", "access-control", "human-review"],
        "decision": "conditional",
        "review": True,
        "difficulty": "medium",
    },
    {
        "type": "policy_update_adaptation",
        "stem": "Under the current v2 pilot policy, may a pilot use external generative AI without a named responsible owner?",
        "answer": "No. The current v2 policy requires a named owner, policy-version logging, rollback criteria, and review before external generative AI pilots can proceed.",
        "refs": ["pilot-policy-v2", "accountability", "audit-trail"],
        "decision": "not_allowed",
        "review": True,
        "difficulty": "hard",
        "policy_version": "v2",
    },
    {
        "type": "citation_audit",
        "stem": "Can a decision be considered auditable if it gives a recommendation but no evidence citations?",
        "answer": "No. An auditable decision must include cited evidence, decision metadata, review status, model or skill version, and a reproducible trace.",
        "refs": ["audit-trail", "evidence-grounding", "model-governance"],
        "decision": "not_allowed",
        "review": True,
        "difficulty": "medium",
    },
]

VARIANTS = [
    "Answer for a city administration use case: {stem}",
    "For a regulated enterprise workflow, {stem}",
    "In a public-sector AI pilot, {stem}",
    "As a compliance officer reviewing an AI case, {stem}",
    "Considering governance and auditability, {stem}",
    "For a department preparing an AI solution, {stem}",
    "Based on the current policy set, {stem}",
    "Using policy-grounded reasoning only, {stem}",
]


def ensure_research_benchmark(data_dir: Path, result_dir: Path, tasks_per_type: int, seed: int = 7) -> Path:
    """Create a deterministic balanced benchmark with the same N for each task type."""
    source = data_dir / "tasks" / "benchmark_tasks.jsonl"
    base = load_tasks_from_path(source)
    out = result_dir / "benchmark_generated.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows: list[dict] = []

    # Keep a balanced subset of curated tasks first.
    base_by_type: dict[str, list[dict]] = {}
    for t in base:
        base_by_type.setdefault(t.task_type, []).append(t.__dict__ | {"metadata": t.metadata | {"origin": "curated"}})

    all_types = [s["type"] for s in SCENARIOS]
    counts: Counter[str] = Counter()

    for task_type in all_types:
        for row in base_by_type.get(task_type, [])[:tasks_per_type]:
            rows.append(row)
            counts[task_type] += 1

    scenario_map = {s["type"]: s for s in SCENARIOS}
    seq = 1
    while any(counts[t] < tasks_per_type for t in all_types):
        pending = [t for t in all_types if counts[t] < tasks_per_type]
        task_type = pending[seq % len(pending)] if len(pending) > 1 else pending[0]
        s = scenario_map[task_type]
        template = rng.choice(VARIANTS)
        noise = rng.choice([
            "Include only policy-grounded information.",
            "Return the safest governance interpretation.",
            "Mention whether human review is required.",
            "Consider policy updates and traceability.",
            "Do not rely on memory; use evidence.",
            "State the decision class and the justification.",
        ])
        q = f"{template.format(stem=s['stem'])} {noise}"
        rows.append(
            {
                "id": f"G{seq:04d}",
                "task_type": s["type"],
                "question": q,
                "expected_answer": s["answer"],
                "expected_policy_refs": s["refs"],
                "expected_decision": s["decision"],
                "expected_human_review": s["review"],
                "policy_version": s.get("policy_version", ""),
                "difficulty": s["difficulty"],
                "metadata": {"origin": "generated", "scenario": s["type"], "variant": template},
            }
        )
        counts[task_type] += 1
        seq += 1

    # deterministic ordering by type then id helps paper tables
    rows.sort(key=lambda r: (r["task_type"], str(r["id"])))

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return out
