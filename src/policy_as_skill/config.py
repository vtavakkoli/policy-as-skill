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
    max_tasks: int = int(os.getenv("MAX_TASKS", "0"))  # 0 means all curated tasks in data/tasks
    top_k: int = int(os.getenv("TOP_K", "5"))
    commercial_llm_enabled: bool = _bool_env("COMMERCIAL_LLM_ENABLED", False)
    commercial_llm_provider: str = os.getenv("COMMERCIAL_LLM_PROVIDER", "openai")
    commercial_llm_base_url: str = os.getenv("COMMERCIAL_LLM_BASE_URL", "")
    commercial_llm_model: str = os.getenv("COMMERCIAL_LLM_MODEL", "gpt-4o-mini")
    commercial_llm_api_key: str = os.getenv("COMMERCIAL_LLM_API_KEY", "")
    manual_citation_annotations_path: str = os.getenv("MANUAL_CITATION_ANNOTATIONS_PATH", "data/annotations/manual_citation_faithfulness.csv")
    bootstrap_iterations: int = int(os.getenv("BOOTSTRAP_ITERATIONS", "1000"))
    methods: str = os.getenv(
        "METHODS",
        "Direct LLM,Keyword Search,Standard RAG,Hybrid RAG,Hybrid RAG + Reranker,Policy-as-Prompt,Structured Policy-as-Prompt,Commercial LLM,Commercial LLM + RAG,Policy-as-Skill No Audit,Policy-as-Skill",
    )

    def method_list(self) -> list[str]:
        return [m.strip() for m in self.methods.split(",") if m.strip()]
