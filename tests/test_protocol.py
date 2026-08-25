import hashlib
import json

import pytest

from policy_as_skill.protocol import validate_evaluation_protocol


def test_development_split_distinguishes_model_unseen_from_system_unseen(tmp_path):
    benchmark = tmp_path / "dev.jsonl"
    benchmark.write_text('{"id":"T1"}\n', encoding="utf-8")
    result = validate_evaluation_protocol(benchmark, evaluation_split="development", frozen_evaluation=False)
    assert result["model_level_unseen"] is True
    assert result["model_parameter_training_on_benchmark"] is False
    assert result["system_development_unseen"] is False
    assert result["system_development_informed"] is True


def test_frozen_test_accepts_matching_sha_and_no_feedback(tmp_path):
    benchmark = tmp_path / "test.jsonl"
    benchmark.write_text('{"id":"T1"}\n', encoding="utf-8")
    sha = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    manifest = tmp_path / "test.manifest.json"
    manifest.write_text(json.dumps({
        "frozen_at": "2026-08-25T00:00:00Z",
        "controller_version": "abc123",
        "benchmark_sha256": sha,
        "created_without_controller_feedback": True,
        "model_parameter_training_on_benchmark": False,
    }), encoding="utf-8")
    result = validate_evaluation_protocol(benchmark, evaluation_split="test", frozen_evaluation=True, manifest_path=manifest)
    assert result["system_development_unseen"] is True
    assert result["system_development_informed"] is False


def test_frozen_test_rejects_bad_sha(tmp_path):
    benchmark = tmp_path / "test.jsonl"
    benchmark.write_text('{"id":"T1"}\n', encoding="utf-8")
    manifest = tmp_path / "test.manifest.json"
    manifest.write_text(json.dumps({
        "frozen_at": "2026-08-25T00:00:00Z",
        "controller_version": "abc123",
        "benchmark_sha256": "bad",
        "created_without_controller_feedback": True,
        "model_parameter_training_on_benchmark": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_evaluation_protocol(benchmark, evaluation_split="test", frozen_evaluation=True, manifest_path=manifest)
