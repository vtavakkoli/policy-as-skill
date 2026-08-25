#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_as_skill.config import Config
from policy_as_skill.data_loader import load_policies, load_tasks
from policy_as_skill.ollama_client import OllamaClient
from policy_as_skill.research_agent import run_method
from policy_as_skill.retrieval import PolicyRetriever
from policy_as_skill.system_profile import collect_system_profile, network_snapshot, power_supply_snapshot, process_resource_snapshot
from policy_as_skill.utils import now


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[min(len(vals)-1, max(0, round((len(vals)-1)*p)))]


def _counter_delta(before: dict, after: dict, key: str):
    a, b = before.get(key), after.get(key)
    return int(b) - int(a) if isinstance(a, int) and isinstance(b, int) else None


def main() -> None:
    p = argparse.ArgumentParser(description="Run reproducible edge/hardware Policy-as-Skill measurements.")
    p.add_argument("--method", default="Policy-as-Skill")
    p.add_argument("--tasks", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", default="result/edge_benchmark.json")
    args = p.parse_args()
    cfg = Config()
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    policies = load_policies(cfg.data_dir)
    tasks = load_tasks(cfg.data_dir, cfg.resolve_path(cfg.benchmark_path))[:max(args.tasks + args.warmup, 1)]
    retriever = PolicyRetriever(policies)
    client = OllamaClient(cfg.ollama_base_url, cfg.ollama_model, cfg.timeout_seconds, cfg.result_dir / "edge_traces.jsonl", enabled=cfg.ollama_enabled, healthcheck_seconds=cfg.healthcheck_seconds)

    warmup_latencies = []
    for task in tasks[:args.warmup]:
        t0 = time.perf_counter()
        run_method(args.method, task, retriever, client, cfg.top_k)
        warmup_latencies.append(time.perf_counter() - t0)

    resource_before = process_resource_snapshot()
    network_before = network_snapshot()
    power_before = power_supply_snapshot()
    latencies = []
    start = time.perf_counter()
    measured = tasks[args.warmup:args.warmup + args.tasks]
    for task in measured:
        t0 = time.perf_counter()
        run_method(args.method, task, retriever, client, cfg.top_k)
        latencies.append(time.perf_counter() - t0)
    wall = time.perf_counter() - start
    resource_after = process_resource_snapshot()
    network_after = network_snapshot()
    power_after = power_supply_snapshot()

    rx_delta = _counter_delta(network_before, network_after, "network_rx_bytes")
    tx_delta = _counter_delta(network_before, network_after, "network_tx_bytes")
    payload = {
        "timestamp": now(),
        "method": args.method,
        "model": cfg.ollama_model,
        "ollama_available": client.is_available(),
        "tasks_measured": len(measured),
        "warmup_tasks": len(warmup_latencies),
        "cold_or_warmup_latency_seconds": warmup_latencies,
        "steady_state": {
            "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
            "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_seconds": percentile(latencies, 0.95),
            "p99_latency_seconds": percentile(latencies, 0.99),
            "total_wall_seconds": wall,
            "throughput_tasks_per_second": len(measured) / wall if wall else 0.0,
        },
        "resource_before": resource_before,
        "resource_after": resource_after,
        "network_before": network_before,
        "network_after": network_after,
        "network_delta": {
            "rx_bytes": rx_delta,
            "tx_bytes": tx_delta,
            "total_bytes": (rx_delta + tx_delta) if rx_delta is not None and tx_delta is not None else None,
            "bytes_per_task": ((rx_delta + tx_delta) / len(measured)) if rx_delta is not None and tx_delta is not None and measured else None,
        },
        "power_supply_before": power_before,
        "power_supply_after": power_after,
        "system_profile": collect_system_profile(),
        "notes": [
            "Run this script on the actual target edge device for paper-quality claims.",
            "Network deltas use host /proc/net/dev counters when available and can include unrelated host traffic; isolate the device/network for precise measurements.",
            "Battery/energy telemetry is recorded when Linux power_supply counters are available.",
            "GPU/power fields are recorded when nvidia-smi or tegrastats is available.",
            "No device, bandwidth, battery, or power measurement is fabricated when counters are unavailable.",
        ],
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
