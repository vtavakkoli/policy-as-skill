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
    tasks_per_type: int = int(os.getenv("TASKS_PER_TYPE", "10"))
    benchmark_size: int = int(os.getenv("BENCHMARK_SIZE", "0"))
    max_tasks: int = int(os.getenv("MAX_TASKS", "0"))  # 0 means all generated tasks
    top_k: int = int(os.getenv("TOP_K", "5"))
    methods: str = os.getenv(
        "METHODS",
        "Direct LLM,Keyword Search,Standard RAG,Hybrid RAG,Policy-as-Prompt,Structured Policy-as-Prompt,Policy-as-Skill No Audit,Policy-as-Skill",
    )

    def method_list(self) -> list[str]:
        return [m.strip() for m in self.methods.split(",") if m.strip()]
