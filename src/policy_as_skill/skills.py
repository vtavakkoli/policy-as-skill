from dataclasses import dataclass

@dataclass(frozen=True)
class PolicySkill:
    name: str
    description: str
    retrieval_scope: list[str]
    risk_level: str
    allowed_actions: list[str]
    human_review_required: bool
    evaluation_criteria: list[str]
    prompt_template: str

def skill_registry() -> dict[str, PolicySkill]:
    base='Use only cited policy evidence. Return JSON with answer, decision, reasoning_summary, citations, human_review_required.'
    return {
      'ComplianceCheckSkill': PolicySkill('ComplianceCheckSkill','Checks whether an action complies with policy.',['data-protection','external-cloud','human-review'],'medium',['advise','escalate'],True,['grounding','review'],base),
      'RiskClassificationSkill': PolicySkill('RiskClassificationSkill','Classifies AI policy risk level.',['high-risk-ai','prohibited-use'],'high',['classify','escalate'],True,['risk','oversight'],base),
      'ConflictDetectionSkill': PolicySkill('ConflictDetectionSkill','Detects conflicts and stricter-rule precedence.',['policy-conflict'],'high',['compare','escalate'],True,['conflict','precedence'],base),
      'EvidenceRecommendationSkill': PolicySkill('EvidenceRecommendationSkill','Creates evidence-grounded recommendations.',['evidence-grounding','accountability'],'medium',['recommend'],False,['citations','uncertainty'],base),
      'HumanReviewRoutingSkill': PolicySkill('HumanReviewRoutingSkill','Determines whether human review is required.',['human-review','audit-trail'],'high',['route','record'],True,['routing','audit'],base),
    }

def choose_skill(task_type: str) -> PolicySkill:
    mapping={'compliance_check':'ComplianceCheckSkill','risk_classification':'RiskClassificationSkill','policy_conflict_detection':'ConflictDetectionSkill','evidence_grounded_recommendation':'EvidenceRecommendationSkill','human_review_routing':'HumanReviewRoutingSkill'}
    return skill_registry().get(mapping.get(task_type,'EvidenceRecommendationSkill'))
