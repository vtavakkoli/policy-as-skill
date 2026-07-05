from __future__ import annotations

import csv
import json
import logging
import platform
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from .agents import run_method
from .annotations import load_manual_citation_annotations
from .config import Config
from .data_loader import load_policies, load_tasks
from .evaluators import apply_run_normalization, evaluate
from .ollama_client import OllamaClient
from .report_generator import generate_report
from .retrieval import PolicyRetriever
from .stats import write_statistics
from .utils import append_jsonl, now, stable_hash


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], methods: list[str]) -> dict:
    numeric = [
        "answer_similarity",
        "citation_coverage",
        "citation_precision",
        "policy_ref_recall",
        "evidence_faithfulness",
        "unsupported_claim_rate",
        "contradiction_rate",
        "decision_accuracy",
        "traceability_score",
        "human_review_correctness",
        "audit_completeness",
        "governance_readiness_score",
        "update_adaptation_score",
        "decision_quality_score",
        "evidence_quality_score",
        "governance_quality_score",
        "raw_quality_score",
        "latency_efficiency_score",
        "normalized_score",
        "task_success",
        "latency_seconds",
        "overall_score",
    ]
    agg: dict = {}
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        if not mr:
            continue
        agg[m] = {
            k: {"mean": mean(float(r[k]) for r in mr), "std": (pstdev(float(r[k]) for r in mr) if len(mr) > 1 else 0.0)}
            for k in numeric
        }
        agg[m]["n"] = len(mr)
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
    benchmark_path = cfg.data_dir / "tasks" / "benchmark_tasks.jsonl"
    tasks = load_tasks(cfg.data_dir, benchmark_path)
    if cfg.max_tasks > 0:
        tasks = tasks[: cfg.max_tasks]

    methods = cfg.method_list()
    retriever = PolicyRetriever(policies)
    client = OllamaClient(
        cfg.ollama_base_url,
        cfg.ollama_model,
        cfg.timeout_seconds,
        trace_path,
        enabled=cfg.ollama_enabled,
        healthcheck_seconds=cfg.healthcheck_seconds,
    )
    annotation_path = Path(cfg.manual_citation_annotations_path)
    if not annotation_path.is_absolute():
        annotation_path = cfg.root / annotation_path
    manual_annotations = load_manual_citation_annotations(annotation_path)

    task_type_counts = Counter(t.task_type for t in tasks)
    manifest = {
        "timestamp": now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ollama_base_url": cfg.ollama_base_url,
        "ollama_model": cfg.ollama_model,
        "ollama_enabled": cfg.ollama_enabled,
        "ollama_available": client.is_available(),
        "manual_citation_annotation_path": str(annotation_path),
        "manual_citation_annotation_rows": sum(v.get("manual_annotation_count", 0) for v in manual_annotations.values()),
        "seed": cfg.seed,
        "benchmark_path": str(benchmark_path),
        "benchmark_source": "static curated tasks in data/tasks/benchmark_tasks.jsonl",
        "tasks_evaluated": len(tasks),
        "task_type_distribution": dict(sorted(task_type_counts.items())),
        "methods": methods,
        "policy_documents": [
            {"source": d.source, "version": d.version, "sha256": d.sha256, "chars": len(d.text)} for d in policies
        ],
        "run_id": stable_hash(now() + cfg.ollama_model + str(cfg.seed)),
    }
    (cfg.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Starting Policy-as-Skill platform tasks=%s methods=%s result_dir=%s ollama=%s available=%s",
        len(tasks),
        len(methods),
        cfg.result_dir,
        cfg.ollama_model,
        manifest["ollama_available"],
    )

    rows: list[dict] = []
    traces: list[dict] = []
    runtime_failures: dict[str, list[dict]] = {m: [] for m in methods}

    for ti, task in enumerate(tasks, start=1):
        logger.info("Processing task %s/%s task_id=%s task_type=%s", ti, len(tasks), task.id, task.task_type)
        for method in methods:
            try:
                tr = run_method(method, task, retriever, client, cfg.top_k)
                traces.append(tr)
                append_jsonl(trace_path, {"kind": "decision_trace", **tr})
                ev = evaluate(task, tr, manual_annotations=manual_annotations)
                row = {
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "difficulty": task.difficulty,
                    "policy_version": task.policy_version,
                    "method": method,
                    **ev,
                }
                rows.append(row)
            except Exception as e:
                logger.exception("Task failed task_id=%s method=%s", task.id, method)
                runtime_failures[method].append({"task_id": task.id, "method": method, "error": str(e)})

    rows = apply_run_normalization(rows)
    failures: dict[str, list[dict]] = {m: list(runtime_failures.get(m, [])) for m in methods}
    for row in rows:
        method = row["method"]
        if float(row.get("task_success", 0.0)) < 1.0:
            failures[method].append(
                {
                    "task_id": row["task_id"],
                    "task_type": row["task_type"],
                    "overall_score": row["overall_score"],
                    "reason": "failed strict task-success gates",
                    "success_gate_failures": str(row.get("success_gate_failures", "")).split(";") if row.get("success_gate_failures") else [],
                    "low_metrics": {
                        k: row[k]
                        for k in [
                            "decision_accuracy",
                            "human_review_correctness",
                            "evidence_quality_score",
                            "governance_quality_score",
                            "citation_precision",
                            "policy_ref_recall",
                            "evidence_faithfulness",
                            "governance_readiness_score",
                        ]
                        if float(row.get(k, 1.0)) < 0.6
                    },
                }
            )

    _write_csv(cfg.result_dir / "metrics.csv", rows)
    statistics = write_statistics(cfg.result_dir, rows, bootstrap_iterations=cfg.bootstrap_iterations, seed=cfg.seed)
    metrics_payload = {"rows": rows, "aggregate": _aggregate(rows, methods), "statistics": statistics, "manifest": manifest}
    (cfg.result_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.result_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_report(cfg.result_dir, rows, traces, failures, manifest)
    logger.info("Report written to %s", cfg.result_dir / "report.html")


if __name__ == "__main__":
    main()
