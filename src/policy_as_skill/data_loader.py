import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import file_sha256


@dataclass
class PolicyDocument:
    source: str
    text: str
    version: str = "v1"
    sha256: str = ""
    path: str = ""


@dataclass
class BenchmarkTask:
    id: str
    task_type: str
    question: str
    expected_answer: str
    expected_policy_refs: list[str]
    expected_decision: str = ""
    expected_human_review: bool | None = None
    policy_version: str = ""
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_version(text: str, path: Path) -> str:
    m = re.search(r"(?im)^version\s*:\s*([\w.-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[_-])(v\d+)(?:[_\-.]|$)", path.stem, flags=re.I)
    return m.group(1).lower() if m else "v1"


def load_policies(data_dir: Path) -> list[PolicyDocument]:
    docs: list[PolicyDocument] = []
    for p in sorted((data_dir / "policies").glob("*.md")):
        text = p.read_text(encoding="utf-8")
        docs.append(PolicyDocument(p.name, text, _parse_version(text, p), file_sha256(p), str(p)))
    return docs


def _derive_decision(expected_answer: str) -> str:
    x = expected_answer.lower()
    if x.startswith("no") or "not allowed" in x or "must not" in x:
        return "not_allowed"
    if x.startswith("yes") or "allowed" in x:
        return "allowed"
    if "review" in x or "approval" in x or "conditional" in x or "if" in x:
        return "conditional"
    return "needs_review"


def load_tasks_from_path(path: Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw.setdefault("expected_decision", _derive_decision(raw.get("expected_answer", "")))
            tasks.append(BenchmarkTask(**raw))
    return tasks


def load_tasks(data_dir: Path, path: Path | None = None) -> list[BenchmarkTask]:
    return load_tasks_from_path(path or data_dir / "tasks" / "benchmark_tasks.jsonl")
