from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .retrieval import PolicyChunk
from .skills import PolicySkill


@dataclass(frozen=True)
class FormalSkillContract:
    """Executable mapping of S=<n,v,R,E,D,H,A,F,P,C> used in the paper."""

    n: str
    v: str
    R: tuple[str, ...]
    E: tuple[str, ...]
    D: dict[str, Any]
    H: dict[str, Any]
    A: tuple[str, ...]
    F: str
    P: str
    C: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_formal_contract(skill: PolicySkill) -> FormalSkillContract:
    return FormalSkillContract(
        n=skill.name,
        v=skill.version,
        R=tuple(skill.retrieval_scope),
        E=tuple(skill.required_evidence_tags),
        D=dict(skill.decision_schema),
        H={
            "default_review_required": bool(skill.human_review_required),
            "triggers": tuple(skill.human_review_triggers),
        },
        A=tuple(skill.audit_fields),
        F=skill.failure_policy,
        P=skill.prompt_template,
        C={
            "required_inputs": ("task_type", "question", "trusted_evidence"),
            "evidence_must_be_scoped": True,
            "citations_must_resolve_to_retrieved_evidence": True,
            "decision_vocabulary": ("allowed", "not_allowed", "conditional", "needs_review", "unknown"),
            "required_outputs": (
                "answer", "decision", "reasoning_summary", "citations",
                "human_review_required", "confidence", "risks", "missing_information",
            ),
        },
    )


def validate_runtime_context(skill: PolicySkill, task_type: str, question: str, chunks: list[PolicyChunk]) -> dict[str, Any]:
    """Validate C and E before/after model execution without benchmark labels."""
    contract = build_formal_contract(skill)
    missing_inputs = []
    if not str(task_type or "").strip():
        missing_inputs.append("task_type")
    if not str(question or "").strip():
        missing_inputs.append("question")
    if not chunks:
        missing_inputs.append("trusted_evidence")
    available_tags = {tag for chunk in chunks for tag in chunk.tags}
    missing_evidence_tags = [tag for tag in contract.E if tag not in available_tags]
    return {
        "contract_valid": not missing_inputs,
        "missing_context_inputs": missing_inputs,
        "missing_required_evidence_tags": missing_evidence_tags,
        "retrieval_scope": list(contract.R),
        "formal_components_executed": ["n", "v", "R", "E", "D", "H", "A", "F", "P", "C"],
    }
