from policy_as_skill.research_metrics import macro_f1, sensitivity_analysis


def test_macro_f1_is_one_for_exact_predictions():
    rows = [
        {"expected_decision": "allowed", "predicted_decision": "allowed"},
        {"expected_decision": "not_allowed", "predicted_decision": "not_allowed"},
        {"expected_decision": "conditional", "predicted_decision": "conditional"},
        {"expected_decision": "needs_review", "predicted_decision": "needs_review"},
    ]
    assert macro_f1(rows) == 1.0


def test_sensitivity_evaluates_multiple_weightings():
    rows = [
        {"method": "A", "decision_quality_score": 0.9, "evidence_quality_score": 0.8, "governance_quality_score": 0.7, "answer_similarity": 0.6},
        {"method": "B", "decision_quality_score": 0.7, "evidence_quality_score": 0.9, "governance_quality_score": 0.8, "answer_similarity": 0.7},
    ]
    result = sensitivity_analysis(rows, step=0.5)
    assert result["weightings_evaluated"] > 1
    assert set(result["top_rank_share"]) == {"A", "B"}
