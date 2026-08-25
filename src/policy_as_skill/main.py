from __future__ import annotations

import csv
import json
import logging
import platform
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from .annotations import annotation_agreement, load_manual_citation_annotations
from .config import Config
from .data_loader import load_policies, load_tasks
from .ollama_client import OllamaClient
from .protocol import validate_evaluation_protocol
from .research_agent import run_method
from .research_metrics import apply_run_normalization, evaluate, method_summary, readiness_threshold_sensitivity, sensitivity_analysis
from .research_report import generate_report
from .retrieval import PolicyRetriever
from .stats import write_statistics
from .system_profile import write_system_profile
from .utils import append_jsonl, now, stable_hash


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], methods: list[str]) -> dict:
    numeric = [
        "answer_similarity", "citation_coverage", "citation_precision", "policy_ref_recall", "evidence_faithfulness",
        "unsupported_claim_rate", "contradiction_rate", "decision_accuracy", "decision_agreement_graded", "decision_exact_match",
        "review_exact_match", "traceability_score", "human_review_correctness", "audit_completeness", "governance_readiness_score",
        "update_adaptation_score", "decision_quality_score", "evidence_quality_score", "governance_quality_score", "raw_quality_score",
        "latency_efficiency_score", "normalized_score", "governance_readiness_index", "task_success", "latency_seconds", "overall_score",
    ]
    agg = {}
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        if not mr:
            continue
        agg[method] = {}
        for key in numeric:
            vals = [float(r[key]) for r in mr if key in r and isinstance(r[key], (int, float))]
            if vals:
                agg[method][key] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0}
        agg[method]["n"] = len(mr)
    return agg


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("policy_as_skill")
    trace_path = cfg.result_dir / "traces.jsonl"
    trace_path.write_text("", encoding="utf-8")

    policies = load_policies(cfg.data_dir)
    benchmark_path = cfg.resolve_path(cfg.benchmark_path)
    benchmark_manifest_path = cfg.resolve_path(cfg.benchmark_manifest_path) if cfg.benchmark_manifest_path else None
    evaluation_protocol = validate_evaluation_protocol(
        benchmark_path,
        evaluation_split=cfg.evaluation_split,
        frozen_evaluation=cfg.frozen_evaluation,
        manifest_path=benchmark_manifest_path,
    )
    tasks = load_tasks(cfg.data_dir, benchmark_path)
    if cfg.max_tasks > 0:
        tasks = tasks[: cfg.max_tasks]

    methods = cfg.method_list()
    retriever = PolicyRetriever(policies)
    client = OllamaClient(cfg.ollama_base_url, cfg.ollama_model, cfg.timeout_seconds, trace_path, enabled=cfg.ollama_enabled, healthcheck_seconds=cfg.healthcheck_seconds)
    annotation_path = cfg.resolve_path(cfg.manual_citation_annotations_path)
    manual_annotations = load_manual_citation_annotations(annotation_path)
    agreement = annotation_agreement(annotation_path)
    (cfg.result_dir / "annotation_agreement.json").write_text(json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8")
    system_profile = write_system_profile(cfg.result_dir / "system_profile.json")

    task_type_counts = Counter(t.task_type for t in tasks)
    manifest = {
        "timestamp": now(), "platform": platform.platform(), "python": platform.python_version(),
        "ollama_base_url": cfg.ollama_base_url, "ollama_model": cfg.ollama_model, "ollama_enabled": cfg.ollama_enabled,
        "ollama_available": client.is_available(), "model_parameter_training_on_benchmark": False, "model_fine_tuning_on_benchmark": False,
        "manual_citation_annotation_path": str(annotation_path),
        "manual_citation_annotation_rows": sum(v.get("manual_annotation_count", 0) for v in manual_annotations.values()),
        "annotation_agreement": agreement, "seed": cfg.seed, "bootstrap_iterations": cfg.bootstrap_iterations,
        "benchmark_path": str(benchmark_path), "benchmark_source": "explicit benchmark path; see evaluation_protocol for development-vs-test provenance",
        "evaluation_protocol": evaluation_protocol, "tasks_evaluated": len(tasks), "task_type_distribution": dict(sorted(task_type_counts.items())),
        "methods": methods,
        "policy_documents": [{"source": d.source, "version": d.version, "sha256": d.sha256, "chars": len(d.text)} for d in policies],
        "research_runner": {
            "policy_as_skill_controller": "generic evidence-driven normative controller",
            "legacy_benchmark_phrase_controller_used_for_policy_as_skill": False,
            "ablations": ["Policy-as-Skill Retrieval", "Policy-as-Skill + Controller", "Policy-as-Skill + Audit", "Policy-as-Skill"],
        },
        "run_id": stable_hash(now() + cfg.ollama_model + str(cfg.seed) + evaluation_protocol["benchmark_sha256"]),
    }
    (cfg.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Starting reviewer-hardened evaluation tasks=%s methods=%s split=%s frozen=%s model=%s", len(tasks), len(methods), cfg.evaluation_split, cfg.frozen_evaluation, cfg.ollama_model)

    rows, traces = [], []
    runtime_failures = {m: [] for m in methods}
    for ti, task in enumerate(tasks, start=1):
        logger.info("Processing task %s/%s task_id=%s task_type=%s", ti, len(tasks), task.id, task.task_type)
        for method in methods:
            try:
                trace = run_method(method, task, retriever, client, cfg.top_k)
                traces.append(trace)
                append_jsonl(trace_path, {"kind": "decision_trace", **trace})
                metrics = evaluate(task, trace, manual_annotations=manual_annotations)
                rows.append({"task_id": task.id, "task_type": task.task_type, "difficulty": task.difficulty, "policy_version": task.policy_version, "method": trace.get("method", method), **metrics})
            except Exception as exc:
                logger.exception("Task failed task_id=%s method=%s", task.id, method)
                runtime_failures[method].append({"task_id": task.id, "method": method, "error": str(exc)})

    rows = apply_run_normalization(rows)
    failures = {m: list(runtime_failures.get(m, [])) for m in methods}
    for row in rows:
        method = row["method"]
        failures.setdefault(method, [])
        if float(row.get("task_success", 0.0)) < 1.0:
            failures[method].append({
                "task_id": row["task_id"], "task_type": row["task_type"],
                "governance_readiness_index": row.get("governance_readiness_index", 0.0),
                "reason": "failed legacy composite governance-readiness gates",
                "success_gate_failures": str(row.get("success_gate_failures", "")).split(";") if row.get("success_gate_failures") else [],
            })

    _write_csv(cfg.result_dir / "metrics.csv", rows)
    statistics = write_statistics(cfg.result_dir, rows, bootstrap_iterations=cfg.bootstrap_iterations, seed=cfg.seed)
    summary = method_summary(rows, seed=cfg.seed, bootstrap_iterations=cfg.bootstrap_iterations)
    sensitivity = sensitivity_analysis(rows, step=cfg.sensitivity_step)
    threshold_sensitivity = readiness_threshold_sensitivity(rows)
    (cfg.result_dir / "sensitivity.json").write_text(json.dumps({"weight_sensitivity": sensitivity, "readiness_threshold_sensitivity": threshold_sensitivity}, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_payload = {
        "rows": rows,
        "aggregate": _aggregate(rows, list(dict.fromkeys(r["method"] for r in rows))),
        "paper_facing_summary": summary,
        "statistics": statistics,
        "sensitivity": sensitivity,
        "readiness_threshold_sensitivity": threshold_sensitivity,
        "manifest": manifest,
    }
    (cfg.result_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.result_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_report(cfg.result_dir, rows, traces, failures, manifest, sensitivity=sensitivity, threshold_sensitivity=threshold_sensitivity, annotation_agreement=agreement, system_profile=system_profile)
    logger.info("Reviewer-hardened report written to %s", cfg.result_dir / "report.html")


if __name__ == "__main__":
    main()
