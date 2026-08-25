import inspect

from policy_as_skill.governance import assess_policy


def _evidence(text, tags=None):
    return [{"citation_id": "policy.md#control@v1", "text": text, "tags": tags or []}]


def test_generic_controller_blocks_missing_mandatory_control():
    result = assess_policy(
        "compliance_check",
        "Can the service be used without procurement approval?",
        _evidence("Before external service use, procurement approval is required.", ["external-cloud"]),
    )
    assert result.decision == "not_allowed"
    assert result.missing_mandatory_control is True


def test_generic_controller_routes_explicit_review_rule():
    result = assess_policy(
        "risk_classification",
        "A public service system affects eligibility decisions.",
        _evidence("Eligibility decisions require trained human review before implementation.", ["human-review"]),
    )
    assert result.decision == "needs_review"
    assert result.human_review_required is True


def test_generic_controller_abstains_without_evidence():
    result = assess_policy("compliance_check", "Can this proceed?", [])
    assert result.decision == "unknown"
    assert result.human_review_required is True


def test_controller_has_no_oracle_inputs():
    params = inspect.signature(assess_policy).parameters
    assert "expected_decision" not in params
    assert "expected_answer" not in params
    assert "task_id" not in params
