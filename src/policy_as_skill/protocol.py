from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evaluation_protocol(benchmark_path: Path, *, evaluation_split: str, frozen_evaluation: bool, manifest_path: Path | None = None) -> dict[str, Any]:
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")
    split = (evaluation_split or "development").strip().lower()
    if split not in {"development", "test", "custom"}:
        raise ValueError("EVALUATION_SPLIT must be development, test, or custom")
    manifest = load_protocol_manifest(manifest_path)
    benchmark_sha = sha256_file(benchmark_path)
    declared_sha = str(manifest.get("benchmark_sha256", "")).strip()
    if declared_sha and declared_sha != benchmark_sha:
        raise ValueError(f"Benchmark SHA-256 does not match the manifest: expected={declared_sha} actual={benchmark_sha}")
    if split == "test" and frozen_evaluation:
        required = {"frozen_at", "controller_version", "benchmark_sha256", "created_without_controller_feedback", "model_parameter_training_on_benchmark"}
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"Frozen test manifest missing required fields: {', '.join(missing)}")
        if manifest.get("created_without_controller_feedback") is not True:
            raise ValueError("Frozen test runs require created_without_controller_feedback=true")
        if manifest.get("model_parameter_training_on_benchmark") is not False:
            raise ValueError("Manifest must state model_parameter_training_on_benchmark=false")
    system_unseen = bool(split == "test" and frozen_evaluation and manifest.get("created_without_controller_feedback") is True)
    return {
        "evaluation_split": split,
        "frozen_evaluation": bool(frozen_evaluation),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_sha,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "model_parameter_training_on_benchmark": False,
        "model_level_unseen": True,
        "system_development_unseen": system_unseen,
        "system_development_informed": not system_unseen,
        "claim_guidance": "The benchmark is unseen by the LLM at the parameter-adaptation level. System-level held-out generalization may be claimed only when system_development_unseen=true.",
    }


def ngram_set(text: str, n: int = 4) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i+n]) for i in range(len(toks)-n+1)}


def max_ngram_overlap(query: str, development_questions: list[str], n: int = 4) -> float:
    q = ngram_set(query, n=n)
    if not q:
        return 0.0
    best = 0.0
    for text in development_questions:
        d = ngram_set(text, n=n)
        if d:
            best = max(best, len(q & d) / max(1, len(q)))
    return best
