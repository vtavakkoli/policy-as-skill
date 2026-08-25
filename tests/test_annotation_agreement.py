from policy_as_skill.annotations import annotation_agreement


def test_annotation_agreement_reports_perfect_kappa(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "task_id,method,citation_id,answer_span,evidence_span,supported,contradiction,policy_ref_correct,annotator_id,reviewed_at,notes\n"
        "T1,Policy-as-Skill,p#1,claim,evidence,yes,no,yes,A1,2026-08-25,\n"
        "T1,Policy-as-Skill,p#1,claim,evidence,yes,no,yes,A2,2026-08-25,\n"
        "T2,Policy-as-Skill,p#2,claim2,evidence2,no,yes,no,A1,2026-08-25,\n"
        "T2,Policy-as-Skill,p#2,claim2,evidence2,no,yes,no,A2,2026-08-25,\n",
        encoding="utf-8",
    )
    result = annotation_agreement(path)
    assert result["status"] == "available"
    assert result["paired_items"] == 2
    assert result["supported_kappa"] == 1.0
    assert result["contradiction_kappa"] == 1.0
