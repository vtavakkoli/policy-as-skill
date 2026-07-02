import json, logging, random
import csv
from statistics import mean
from .config import Config
from .data_loader import load_policies, load_tasks
from .retrieval import PolicyRetriever
from .ollama_client import OllamaClient
from .agents import run_method
from .evaluators import evaluate
from .report_generator import generate_report
from .utils import append_jsonl

def main() -> None:
    cfg=Config(); random.seed(cfg.seed); cfg.result_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    trace_path=cfg.result_dir/'traces.jsonl'; trace_path.write_text('', encoding='utf-8')
    retriever=PolicyRetriever(load_policies(cfg.data_dir)); tasks=load_tasks(cfg.data_dir)
    client=OllamaClient(cfg.ollama_base_url,cfg.ollama_model,cfg.timeout_seconds,trace_path)
    methods=['Keyword Search','Standard RAG','Policy-as-Prompt','Policy-as-Skill']
    rows=[]; traces=[]; failures={m:[] for m in methods}
    for task in tasks:
        for method in methods:
            try:
                tr=run_method(method,task,retriever,client); traces.append(tr); append_jsonl(trace_path, {'kind':'decision_trace', **tr})
                ev=evaluate(task,tr); row={'task_id':task.id,'task_type':task.task_type,'method':method,**ev}; rows.append(row)
                if ev['overall_score'] < 0.65: failures[method].append({'task_id':task.id,'overall_score':round(ev['overall_score'],3),'reason':'below threshold'})
            except Exception as e:
                logging.exception('Task failed'); failures[method].append({'task_id':task.id,'error':str(e)})
    
    with (cfg.result_dir/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (cfg.result_dir/'metrics.json').write_text(json.dumps({'rows':rows,'aggregate':{m:{k:mean(r[k] for r in rows if r['method']==m) for k in ['answer_similarity','citation_coverage','policy_ref_recall','traceability_score','human_review_correctness','latency_seconds','overall_score']} for m in methods}}, indent=2), encoding='utf-8')
    (cfg.result_dir/'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    generate_report(cfg.result_dir, rows, traces, failures)
    logging.info('Report written to %s', cfg.result_dir/'report.html')

if __name__ == '__main__':
    main()
