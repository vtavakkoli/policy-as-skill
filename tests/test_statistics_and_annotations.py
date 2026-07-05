from pathlib import Path

from policy_as_skill.annotations import load_manual_citation_annotations
from policy_as_skill.stats import compute_pairwise_statistics


def test_pairwise_statistics_detects_target_wins():
    rows = [
        {"task_id": "T1", "method": "Policy-as-Skill", "normalized_score": 0.9},
        {"task_id": "T1", "method": "Baseline", "normalized_score": 0.4},
        {"task_id": "T2", "method": "Policy-as-Skill", "normalized_score": 0.8},
        {"task_id": "T2", "method": "Baseline", "normalized_score": 0.6},
    ]
    stats = compute_pairwise_statistics(rows, bootstrap_iterations=10, seed=1)
    assert stats[0]["target_wins"] == 2
    assert stats[0]["mean_difference"] > 0


def test_manual_annotation_loader_empty_file(tmp_path: Path):
    p = tmp_path / "manual.csv"
    p.write_text("", encoding="utf-8")
    assert load_manual_citation_annotations(p) == {}
