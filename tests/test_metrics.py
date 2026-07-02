from policy_as_skill.data_loader import BenchmarkTask
from policy_as_skill.evaluators import evaluate


def test_metrics_policy_ref_recall_and_audit():
    task = BenchmarkTask('T', 'compliance_check', 'q', 'No without review', ['data-protection', 'human-review'], expected_decision='not_allowed')
    trace = {
        'timestamp': 'now',
        'task_id': 'T',
        'method': 'Policy-as-Skill',
        'question': 'q',
        'answer': 'No without review',
        'citations': ['x.md#data-protection@v1', 'y.md#human-review@v1'],
        'evidence': [
            {'citation_id': 'x.md#data-protection@v1', 'text': 'data protection', 'tags': ['data-protection']},
            {'citation_id': 'y.md#human-review@v1', 'text': 'human review', 'tags': ['human-review']},
        ],
        'reasoning_summary': 'because policy',
        'decision': 'not_allowed',
        'human_review_required': True,
        'confidence': 0.9,
        'validation': {'schema_valid': True},
        'policy_hashes': ['x'],
        'prompt_hash': 'abc',
        'selected_skill': 'ComplianceCheckSkill',
        'policy_skill_version': 'skill-v2.1',
        'skill_metadata': {'name': 'ComplianceCheckSkill'},
        'latency_seconds': 0.1,
    }
    ev = evaluate(task, trace)
    assert ev['policy_ref_recall'] == 1.0
    assert ev['citation_precision'] == 1.0
    assert ev['audit_completeness'] == 1.0
