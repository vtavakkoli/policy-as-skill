from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / 'data'
    result_dir: Path = root / 'result'
    ollama_base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
    ollama_model: str = os.getenv('OLLAMA_MODEL', 'gemma4:e2b')
    timeout_seconds: float = float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '120'))
    seed: int = 7
