from dataclasses import dataclass
from pathlib import Path
import os


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    result_dir: Path = root / "result"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    healthcheck_seconds: float = float(os.getenv("OLLAMA_HEALTHCHECK_SECONDS", "2"))
    ollama_enabled: bool = _bool_env("OLLAMA_ENABLED", True)
    seed: int = int(os.getenv("SEED", "7"))
    max_tasks: int = int(os.getenv("MAX_TASKS", "0"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    manual_citation_annotations_path: str = os.getenv("MANUAL_CITATION_ANNOTATIONS_PATH", "data/annotations/manual_citation_faithfulness.csv")
    bootstrap_iterations: int = int(os.getenv("BOOTSTRAP_ITERATIONS", "1000"))
    evaluation_split: str = os.getenv("EVALUATION_SPLIT", "development")
    benchmark_path: str = os.getenv("BENCHMARK_PATH", "data/tasks/benchmark_tasks.jsonl")
    benchmark_manifest_path: str = os.getenv("BENCHMARK_MANIFEST_PATH", "data/tasks/development_manifest.json")
    frozen_evaluation: bool = _bool_env("FROZEN_EVALUATION", False)
    sensitivity_step: float = float(os.getenv("SENSITIVITY_STEP", "0.1"))
    methods: str = os.getenv(
        "METHODS",
        "Direct LLM,LLM,Keyword Search,Standard RAG,Hybrid RAG,Hybrid RAG + Reranker,LLM + RAG,Policy-as-Prompt,Structured Policy-as-Prompt,Policy-as-Skill Retrieval,Policy-as-Skill + Controller,Policy-as-Skill + Audit,Policy-as-Skill",
    )

    def method_list(self) -> list[str]:
        aliases = {"Commercial LLM": "LLM", "Commercial LLM + RAG": "LLM + RAG", "Policy-as-Skill No Audit": "Policy-as-Skill + Controller"}
        methods: list[str] = []
        seen: set[str] = set()
        for item in self.methods.split(","):
            method = aliases.get(item.strip(), item.strip())
            if method and method not in seen:
                methods.append(method)
                seen.add(method)
        return methods

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path
