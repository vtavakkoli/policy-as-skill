from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _exact_two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_ci(diffs: list[float], iterations: int, seed: int) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    rng = random.Random(seed)
    vals = []
    n = len(diffs)
    for _ in range(max(1, iterations)):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        vals.append(mean(sample))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def compute_pairwise_statistics(
    rows: list[dict],
    target_method: str = "Policy-as-Skill",
    metric: str = "normalized_score",
    bootstrap_iterations: int = 1000,
    seed: int = 7,
) -> list[dict[str, Any]]:
    by_method_task: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_method_task[str(r["method"])][str(r["task_id"])] = float(r.get(metric, 0.0))
    target = by_method_task.get(target_method, {})
    out: list[dict[str, Any]] = []
    for method, vals in sorted(by_method_task.items()):
        if method == target_method:
            continue
        task_ids = sorted(set(target) & set(vals))
        diffs = [target[t] - vals[t] for t in task_ids]
        wins = sum(1 for d in diffs if d > 1e-12)
        losses = sum(1 for d in diffs if d < -1e-12)
        ties = len(diffs) - wins - losses
        ci_lo, ci_hi = _bootstrap_ci(diffs, bootstrap_iterations, seed)
        out.append(
            {
                "target_method": target_method,
                "baseline_method": method,
                "metric": metric,
                "n_pairs": len(diffs),
                "target_mean": round(mean([target[t] for t in task_ids]) if task_ids else 0.0, 6),
                "baseline_mean": round(mean([vals[t] for t in task_ids]) if task_ids else 0.0, 6),
                "mean_difference": round(mean(diffs) if diffs else 0.0, 6),
                "ci95_low": round(ci_lo, 6),
                "ci95_high": round(ci_hi, 6),
                "target_wins": wins,
                "baseline_wins": losses,
                "ties": ties,
                "sign_test_p": round(_exact_two_sided_sign_test(wins, losses), 8),
            }
        )
    return out


def write_statistics(result_dir: Path, rows: list[dict], bootstrap_iterations: int = 1000, seed: int = 7) -> list[dict[str, Any]]:
    stats = compute_pairwise_statistics(rows, bootstrap_iterations=bootstrap_iterations, seed=seed)
    (result_dir / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats:
        with (result_dir / "statistics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            writer.writeheader()
            writer.writerows(stats)
    else:
        (result_dir / "statistics.csv").write_text("", encoding="utf-8")
    return stats
