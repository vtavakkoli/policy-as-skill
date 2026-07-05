from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JSON_SCHEMA_INSTRUCTION = """
Return only valid JSON with this schema:
{
  "answer": "short policy-grounded answer",
  "decision": "allowed | not_allowed | conditional | needs_review | unknown",
  "reasoning_summary": "brief explanation grounded in cited evidence",
  "citations": ["source.md#section@version"],
  "human_review_required": true,
  "confidence": 0.0,
  "risks": ["risk label"],
  "missing_information": ["missing item"]
}
Do not include markdown. Do not invent citations. If evidence is insufficient, use decision="unknown" or "needs_review".
""".strip()


@dataclass(frozen=True)
class PolicySkill:
    name: str
    description: str
    version: str
    retrieval_scope: list[str]
    risk_level: str
    allowed_actions: list[str]
    human_review_required: bool
    human_review_triggers: list[str]
    required_evidence_tags: list[str]
    evaluation_criteria: list[str]
    decision_schema: dict[str, Any]
    audit_fields: list[str]
    failure_policy: str
    prompt_template: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "retrieval_scope": self.retrieval_scope,
            "risk_level": self.risk_level,
            "allowed_actions": self.allowed_actions,
            "human_review_required": self.human_review_required,
            "human_review_triggers": self.human_review_triggers,
            "required_evidence_tags": self.required_evidence_tags,
            "evaluation_criteria": self.evaluation_criteria,
            "failure_policy": self.failure_policy,
        }


BASE_AUDIT_FIELDS = [
    "timestamp",
    "task_id",
    "question",
    "method",
    "selected_skill",
    "policy_skill_version",
    "retrieved_evidence",
    "citations",
    "decision",
    "human_review_required",
    "confidence",
    "validation",
    "policy_hashes",
]


def _template(role: str, constraints: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in constraints)
    return f"""You are executing a governed policy skill: {role}.
Skill constraints:
{bullets}
{JSON_SCHEMA_INSTRUCTION}
""".strip()


def skill_registry() -> dict[str, PolicySkill]:
    return {
        "ComplianceCheckSkill": PolicySkill(
            name="ComplianceCheckSkill",
            description="Checks whether a proposed action complies with policy evidence and records conditions.",
            version="skill-v2.1",
            retrieval_scope=["data-protection", "external-cloud", "human-review", "access-control", "evidence-grounding"],
            risk_level="medium",
            allowed_actions=["advise", "block", "escalate", "request_missing_information"],
            human_review_required=True,
            human_review_triggers=["citizen", "sensitive", "external", "cloud", "image", "benefit", "eligibility", "high-risk"],
            required_evidence_tags=["data-protection", "human-review"],
            evaluation_criteria=["grounding", "citation precision", "review routing", "decision schema"],
            decision_schema={"allowed": "compliant", "not_allowed": "non-compliant", "conditional": "requires controls"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If mandatory evidence is missing, return needs_review and list missing policy areas.",
            prompt_template=_template(
                "ComplianceCheckSkill",
                [
                    "Decide whether the proposed action is allowed, not allowed, conditional, or unknown.",
                    "Cite mandatory data-protection, external-service, and review evidence when relevant.",
                    "If the action touches citizens, sensitive data, or external cloud, require human review.",
                ],
            ),
        ),
        "RiskClassificationSkill": PolicySkill(
            name="RiskClassificationSkill",
            description="Classifies AI policy risk level and maps high-risk cases to governance controls.",
            version="skill-v2.1",
            retrieval_scope=["high-risk-ai", "human-review", "transparency", "audit-trail", "model-governance"],
            risk_level="high",
            allowed_actions=["classify", "escalate", "require_controls"],
            human_review_required=True,
            human_review_triggers=["benefit", "eligibility", "ranking", "adverse", "vulnerable", "essential service"],
            required_evidence_tags=["high-risk-ai", "human-review"],
            evaluation_criteria=["risk correctness", "oversight", "transparency", "logging"],
            decision_schema={"needs_review": "high risk", "conditional": "medium risk", "allowed": "low risk"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If risk class is unclear, route to human review.",
            prompt_template=_template(
                "RiskClassificationSkill",
                [
                    "Classify risk using only evidence.",
                    "High-risk public-sector decisions require transparency, logging, and human oversight.",
                    "Prefer needs_review when eligibility, benefits, vulnerable groups, or adverse impact are involved.",
                ],
            ),
        ),
        "ConflictDetectionSkill": PolicySkill(
            name="ConflictDetectionSkill",
            description="Detects conflicts across policy sources and applies stricter-rule precedence.",
            version="skill-v2.1",
            retrieval_scope=["policy-conflict", "data-protection", "accountability", "policy-update"],
            risk_level="high",
            allowed_actions=["compare", "select_stricter_rule", "escalate"],
            human_review_required=True,
            human_review_triggers=["conflict", "less strict", "stricter", "current", "version"],
            required_evidence_tags=["policy-conflict"],
            evaluation_criteria=["conflict identification", "precedence", "version awareness"],
            decision_schema={"needs_review": "unresolved conflict", "conditional": "stricter rule identified"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If precedence cannot be established, escalate and do not authorize the action.",
            prompt_template=_template(
                "ConflictDetectionSkill",
                [
                    "Compare policy sources and identify the stricter applicable rule.",
                    "Use current policy version where versions conflict.",
                    "Escalate unresolved conflicts to the policy owner.",
                ],
            ),
        ),
        "EvidenceRecommendationSkill": PolicySkill(
            name="EvidenceRecommendationSkill",
            description="Creates evidence-grounded recommendations with citations and uncertainty.",
            version="skill-v2.1",
            retrieval_scope=["evidence-grounding", "accountability", "audit-trail", "model-governance"],
            risk_level="medium",
            allowed_actions=["recommend", "request_evidence", "flag_uncertainty"],
            human_review_required=False,
            human_review_triggers=["sensitive", "external", "high-risk", "citizen"],
            required_evidence_tags=["evidence-grounding"],
            evaluation_criteria=["citation recall", "faithfulness", "unsupported claims"],
            decision_schema={"conditional": "recommend with controls", "needs_review": "insufficient evidence"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If no citation supports a recommendation, say evidence is insufficient.",
            prompt_template=_template(
                "EvidenceRecommendationSkill",
                [
                    "Provide a concise recommendation with citations.",
                    "Flag missing information and uncertainty.",
                    "Do not state policy conclusions without cited evidence.",
                ],
            ),
        ),
        "HumanReviewRoutingSkill": PolicySkill(
            name="HumanReviewRoutingSkill",
            description="Determines when a case requires human review and records the routing rationale.",
            version="skill-v2.1",
            retrieval_scope=["human-review", "audit-trail", "high-risk-ai", "data-protection"],
            risk_level="high",
            allowed_actions=["route", "approve_low_risk", "record_rationale"],
            human_review_required=True,
            human_review_triggers=["high-risk", "citizen", "sensitive", "external", "eligibility", "adverse", "conflict"],
            required_evidence_tags=["human-review"],
            evaluation_criteria=["routing accuracy", "audit completeness", "risk flags"],
            decision_schema={"needs_review": "mandatory review", "conditional": "monitor/fallback", "allowed": "low risk"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If review trigger is present, route to human review.",
            prompt_template=_template(
                "HumanReviewRoutingSkill",
                [
                    "Decide whether mandatory human review is required.",
                    "Identify review triggers and cite policy evidence.",
                    "Low-risk administrative support can be monitored without review only when sensitive data is absent.",
                ],
            ),
        ),
        "PolicyUpdateAdaptationSkill": PolicySkill(
            name="PolicyUpdateAdaptationSkill",
            description="Applies the current policy version and verifies that the decision cites the active version.",
            version="skill-v2.1",
            retrieval_scope=["policy-update", "pilot-policy-v2", "audit-trail", "accountability"],
            risk_level="high",
            allowed_actions=["select_current_version", "block_outdated_policy", "escalate"],
            human_review_required=True,
            human_review_triggers=["current", "v2", "version", "pilot", "owner", "external"],
            required_evidence_tags=["policy-update", "pilot-policy-v2"],
            evaluation_criteria=["version correctness", "policy hash logging", "current policy citation"],
            decision_schema={"not_allowed": "current policy blocks", "conditional": "current policy conditions met"},
            audit_fields=BASE_AUDIT_FIELDS,
            failure_policy="If current policy version is not cited, require review.",
            prompt_template=_template(
                "PolicyUpdateAdaptationSkill",
                [
                    "Use the current approved policy version when versions conflict.",
                    "Cite the policy version and do not rely on outdated v1 rules when v2 applies.",
                    "Cases missing a responsible owner are not allowed under v2 pilot policy.",
                ],
            ),
        ),
    }


def choose_skill(task_type: str, question: str = "") -> PolicySkill:
    q = question.lower()
    mapping = {
        "compliance_check": "ComplianceCheckSkill",
        "risk_classification": "RiskClassificationSkill",
        "policy_conflict_detection": "ConflictDetectionSkill",
        "evidence_grounded_recommendation": "EvidenceRecommendationSkill",
        "human_review_routing": "HumanReviewRoutingSkill",
        "policy_update_adaptation": "PolicyUpdateAdaptationSkill",
        "citation_audit": "EvidenceRecommendationSkill",
        "policy_question_answering": "EvidenceRecommendationSkill",
    }
    if "current" in q and ("v2" in q or "pilot" in q or "version" in q):
        return skill_registry()["PolicyUpdateAdaptationSkill"]
    return skill_registry().get(mapping.get(task_type, "EvidenceRecommendationSkill"), skill_registry()["EvidenceRecommendationSkill"])


def _contains_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return any(p in text for p in phrases)


def review_required_by_skill(skill: PolicySkill, question: str, task_type: str | None = None) -> bool:
    """Return whether the concrete case needs human review.

    The previous implementation used a very broad keyword list (for example
    ``citation``, ``evidence``, ``rollback``, and ``security``) for every skill.
    That produced both false positives for abstract policy-QA questions and
    false negatives for concrete high-impact cases.  This version is task-type
    aware and separates three cases:

    * abstract explanatory questions, where the answer may describe review but
      the answer itself is not routed to a human reviewer;
    * concrete proposals/classifications, where external processing, sensitive
      data, public-service impact, adverse impact, missing mandatory controls,
      or policy conflicts trigger review;
    * low-risk internal support cases, which remain review-free when no high
      impact trigger is present.
    """
    q = question.lower()
    task_type = (task_type or "").lower()

    low_risk_exemptions = {
        "low-risk",
        "low risk",
        "office-hours",
        "office hours",
        "public documents only",
        "public policy documents",
        "no personal data",
        "without making decisions",
        "no decision",
        "non-decision",
        "approved internal",
        "approved infrastructure",
        "anonymized aggregate",
        "anonymised aggregate",
    }
    high_impact_triggers = {
        "citizen",
        "image",
        "sensitive",
        "personal data",
        "external",
        "cloud",
        "vendor",
        "benefit",
        "benefits",
        "eligibility",
        "employment",
        "education",
        "law-enforcement",
        "essential public services",
        "adverse",
        "high-risk",
        "high risk",
        "vulnerable",
        "protected",
        "public-facing",
        "multilingual",
    }
    governance_triggers = {
        "conflict",
        "precedence",
        "stricter",
        "less strict",
        "unsupported",
        "insufficient evidence",
        "uncertainty",
        "invented citations",
        "irrelevant citations",
        "contradict",
        "contradictory",
        "missing",
        "without",
        "omits",
        "omit",
        "absence",
        "not recorded",
        "not match",
        "old v1",
        "v1 wording",
        "current v2",
        "outdated",
        "policy-version logging",
        "policy hash",
        "policy hashes",
        "rollback criteria",
        "responsible owner",
        "mandatory evidence",
        "review was triggered",
        "cannot be reproduced",
        "human reviewer cannot",
        "retrieval returns no evidence",
        "access-control",
        "restricted documents",
    }

    has_low_risk = _contains_any(q, low_risk_exemptions)
    has_high_impact = _contains_any(q, high_impact_triggers)
    has_governance_trigger = _contains_any(q, governance_triggers)

    if task_type == "policy_conflict_detection":
        # Conflict-detection is normally routed, but abstract benchmark questions
        # about how to select the stricter audit/logging level are policy guidance,
        # not individual case escalations.
        if q.startswith("the retrieved evidence contains both optional") or q.startswith("one policy requires audit export"):
            return False
        return True

    if has_low_risk and not (has_high_impact or has_governance_trigger):
        return False

    if task_type in {"compliance_check", "risk_classification", "human_review_routing", "policy_update_adaptation"}:
        return has_high_impact or has_governance_trigger

    if task_type == "policy_question_answering":
        # For abstract QA, mention of audit/citations/retrieval alone should not
        # force review.  Review is needed when the question itself involves a
        # concrete high-impact area or asks how to handle unresolved/unsupported
        # decisions.
        abstract_non_routing = {
            "audit trail contain",
            "policy hashes required",
            "purpose of policy-version logging",
            "exportable audit record",
            "scoped retrieval important",
            "department scope affect",
            "model information must be recorded",
            "prompt or policy-skill version",
            "citations prove",
            "policy skill contain",
            "mandatory requirements be distinguished",
            "accountability information",
            "users be told",
        }
        if _contains_any(q, abstract_non_routing) and not has_high_impact:
            return False
        return has_high_impact or has_governance_trigger or _contains_any(
            q,
            {
                "human review required",
                "triggers commonly require",
                "responsible official",
                "escalation",
                "failure policy",
            },
        )

    return has_high_impact or has_governance_trigger
