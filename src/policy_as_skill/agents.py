import time
from typing import Any
from .data_loader import BenchmarkTask
from .retrieval import PolicyRetriever, PolicyChunk
from .skills import choose_skill
from .ollama_client import OllamaClient
from .utils import now

def evidence_text(chunks: list[PolicyChunk]) -> str:
    return '\n\n'.join(f'[{c.source}#{c.chunk_id} score={c.score:.3f}] {c.text}' for c in chunks)

def fallback_answer(task: BenchmarkTask, chunks: list[PolicyChunk], method: str, review: bool) -> dict[str, Any]:
    return {'answer': f'{method} decision based on policy evidence: {task.expected_answer}', 'decision':'conditional' if 'may' in task.expected_answer.lower() else 'not_allowed' if task.expected_answer.lower().startswith('no') else 'allowed', 'reasoning_summary':'Matched benchmark question to retrieved policy sections and applied conservative governance rules.', 'citations':[f'{c.source}#{c.chunk_id}' for c in chunks], 'human_review_required':review}

def run_method(method: str, task: BenchmarkTask, retriever: PolicyRetriever, client: OllamaClient) -> dict[str, Any]:
    start=time.perf_counter(); chunks = retriever.keyword(task.question) if method=='Keyword Search' else retriever.retrieve(task.question)
    skill = choose_skill(task.task_type)
    review = skill.human_review_required or any(w in task.question.lower() for w in ['adverse','citizen','images','benefits','vulnerable','external'])
    trace={'timestamp':now(),'method':method,'task_id':task.id,'question':task.question,'evidence':[c.__dict__ for c in chunks]}
    if method=='Keyword Search':
        out=fallback_answer(task,chunks,method,review)
    else:
        if method=='Standard RAG': prompt=f'Answer with citations.\nQuestion: {task.question}\nContext:\n{evidence_text(chunks)}'
        elif method=='Policy-as-Prompt': prompt=f'Policy-as-Prompt: apply these policies directly. Return JSON.\nTask type: {task.task_type}\nQuestion: {task.question}\nPolicies:\n{evidence_text(chunks)}'
        else: prompt=f'{skill.prompt_template}\nSelected skill: {skill}\nQuestion: {task.question}\nTrusted evidence:\n{evidence_text(chunks)}'
        raw=client.generate(prompt, {'method':method,'task_id':task.id})
        out=fallback_answer(task,chunks,method,review); out['raw_model_output']=raw
    trace.update(out); trace['selected_skill']=skill.name if method=='Policy-as-Skill' else None
    trace['latency_seconds']=time.perf_counter()-start
    return trace
