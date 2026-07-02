from .data_loader import BenchmarkTask
from .utils import token_similarity

def expected_review(task: BenchmarkTask) -> bool:
    return task.task_type in {'compliance_check','risk_classification','policy_conflict_detection','human_review_routing'} or any(x in task.question.lower() for x in ['citizen','adverse','external','vulnerable','benefits'])

def evaluate(task: BenchmarkTask, trace: dict) -> dict:
    text=' '.join([trace.get('answer',''), trace.get('reasoning_summary',''), ' '.join(trace.get('citations',[])), str(trace.get('evidence',''))]).lower()
    recall=sum(1 for r in task.expected_policy_refs if r.lower() in text)/max(1,len(task.expected_policy_refs))
    citation=1.0 if trace.get('citations') else 0.0
    sim=token_similarity(task.expected_answer, trace.get('answer',''))
    tr=sum([bool(trace.get('evidence')), bool(trace.get('reasoning_summary')), bool(trace.get('decision')), 'human_review_required' in trace])/4
    hrc=1.0 if bool(trace.get('human_review_required'))==expected_review(task) else 0.0
    overall=0.25*sim+0.2*citation+0.25*recall+0.2*tr+0.1*hrc
    return {'answer_similarity':sim,'citation_coverage':citation,'policy_ref_recall':recall,'traceability_score':tr,'human_review_correctness':hrc,'latency_seconds':trace.get('latency_seconds',0),'overall_score':overall}
