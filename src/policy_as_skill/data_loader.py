import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PolicyDocument:
    source: str
    text: str

@dataclass
class BenchmarkTask:
    id: str
    task_type: str
    question: str
    expected_answer: str
    expected_policy_refs: list[str]

def load_policies(data_dir: Path) -> list[PolicyDocument]:
    return [PolicyDocument(p.name, p.read_text(encoding='utf-8')) for p in sorted((data_dir/'policies').glob('*.md'))]

def load_tasks(data_dir: Path) -> list[BenchmarkTask]:
    tasks=[]
    with (data_dir/'tasks'/'benchmark_tasks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): tasks.append(BenchmarkTask(**json.loads(line)))
    return tasks
