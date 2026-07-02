from policy_as_skill.data_loader import BenchmarkTask
from policy_as_skill.evaluators import evaluate

def test_metrics_policy_ref_recall():
    task=BenchmarkTask('T','compliance_check','q','No without review',['data-protection','human-review'])
    trace={'answer':'No without review','citations':['x#data-protection','y#human-review'],'evidence':[{'source':'x'}],'reasoning_summary':'because policy','decision':'not_allowed','human_review_required':True,'latency_seconds':0.1}
    ev=evaluate(task, trace)
    assert ev['policy_ref_recall'] == 1.0
    assert ev['traceability_score'] == 1.0
