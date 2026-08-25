from policy_as_skill.retrieval import PolicyChunk
from policy_as_skill.skill_contract import build_formal_contract, validate_runtime_context
from policy_as_skill.skills import skill_registry


def test_formal_tuple_maps_all_components():
    skill = skill_registry()["ComplianceCheckSkill"]
    contract = build_formal_contract(skill)
    assert contract.n == skill.name
    assert contract.v == skill.version
    assert contract.R == tuple(skill.retrieval_scope)
    assert contract.E == tuple(skill.required_evidence_tags)
    assert contract.D == skill.decision_schema
    assert contract.A == tuple(skill.audit_fields)
    assert contract.F == skill.failure_policy
    assert contract.P == skill.prompt_template
    assert set(contract.C["required_inputs"]) == {"task_type", "question", "trusted_evidence"}


def test_runtime_contract_validation_is_label_free():
    skill = skill_registry()["ComplianceCheckSkill"]
    chunks = [PolicyChunk("p.md", "x", "Human review and data protection are required.", tags=["human-review", "data-protection"])]
    result = validate_runtime_context(skill, "compliance_check", "Can the action proceed?", chunks)
    assert result["contract_valid"] is True
    assert result["missing_required_evidence_tags"] == []
    assert result["formal_components_executed"] == ["n", "v", "R", "E", "D", "H", "A", "F", "P", "C"]
