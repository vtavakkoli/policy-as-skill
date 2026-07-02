import json, re, time
from pathlib import Path
from typing import Any

def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def tokens(text: str) -> set[str]:
    return set(re.findall(r'[a-z0-9-]+', text.lower()))

def token_similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))
