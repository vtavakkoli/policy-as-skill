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


def review_required_by_skill(skill: PolicySkill, question: str) -> bool:
    """Return whether the concrete case needs human review.

    The skill metadata may mark a skill as governance-sensitive, but that does
    not mean every low-risk question handled by that skill requires manual
    review. Review is triggered by the case content: high-impact domains,
    sensitive data, external processing, policy conflict, missing mandatory
    controls, or outdated/current-version ambiguity.
    """
    q = question.lower()
    trigger_terms = {t.lower() for t in skill.human_review_triggers}
    trigger_terms.update(
        {
            "citizen",
            "image",
            "sensitive",
            "external",
            "cloud",
            "benefit",
            "eligibility",
            "adverse",
            "conflict",
            "vulnerable",
            "protected",
            "high-risk",
            "without",
            "missing",
            "absence",
            "omits",
            "no ",
            "test phase",
            "v2",
            "current",
            "owner",
            "security",
            "procurement",
            "data-protection",
            "data protection",
            "not match",
            "irrelevant",
            "policy hash",
            "citation",
            "evidence",
            "rollback",
        }
    )
    low_risk_exemptions = [
        "low-risk",
        "office-hours",
        "office hours",
        "non-sensitive internal",
        "public documents only",
        "no decision",
        "no personal data",
    ]
    has_low_risk_exemption = any(x in q for x in low_risk_exemptions)
    has_trigger = any(t in q for t in trigger_terms)
    if has_low_risk_exemption and not any(x in q for x in ["external", "cloud", "without", "missing", "absence", "omits", "no ", "citizen", "benefit", "eligibility", "adverse", "conflict", "v2", "not match", "policy hash", "rollback"]):
        return False
    return has_trigger
