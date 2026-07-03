from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .diagrams import ARCHITECTURE, LOOP


def _fmt_num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def _fmt_mean_std(mu: float, sigma: float, suffix: str = "", digits: int = 3, pct: bool = False) -> str:
    if pct:
        mu *= 100.0
        sigma *= 100.0
        suffix = "%"
        digits = 1
    return f"{mu:.{digits}f} ± {sigma:.{digits}f}{suffix}"


def _group_stats(rows: list[dict], key: str, cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    out: list[dict[str, Any]] = []
    for k in sorted(groups):
        vals = groups[k]
        row: dict[str, Any] = {"method": k, "n": len(vals)}
        for c in cols:
            xs = [float(v[c]) for v in vals]
            row[f"{c}_mean"] = mean(xs)
            row[f"{c}_std"] = pstdev(xs) if len(xs) > 1 else 0.0
        out.append(row)
    return out


def _table(rows: list[dict[str, Any]], cols: list[str], limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    head = "".join(f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in cols)
    body = ""
    for r in shown:
        body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _task_table(rows: list[dict]) -> str:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(str(r["method"]), str(r["task_type"]))].append(r)
    out = []
    for (method, task_type), vals in sorted(groups.items()):
        scores = [float(v["overall_score"]) for v in vals]
        succ = [float(v["task_success"]) for v in vals]
        out.append(
            {
                "method": method,
                "task_type": task_type,
                "n": len(vals),
                "overall_score": _fmt_mean_std(mean(scores), pstdev(scores) if len(scores) > 1 else 0.0),
                "task_success": _fmt_mean_std(mean(succ), pstdev(succ) if len(succ) > 1 else 0.0, pct=True),
            }
        )
    return _table(out, ["method", "task_type", "n", "overall_score", "task_success"])


def _palette(i: int) -> tuple[str, str]:
    fills = [
        ("#2563eb", "#93c5fd"),
        ("#7c3aed", "#d8b4fe"),
        ("#0f766e", "#99f6e4"),
        ("#db2777", "#f9a8d4"),
        ("#ea580c", "#fdba74"),
        ("#0891b2", "#a5f3fc"),
        ("#65a30d", "#bef264"),
        ("#4f46e5", "#c7d2fe"),
    ]
    return fills[i % len(fills)]


def _svg_metric_chart(title: str, agg: list[dict], metric: str, *, suffix: str = "", percent: bool = False, lower_is_better: bool = False) -> str:
    width, height = 980, 360
    left, right, top, bottom = 205, 40, 72, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    means = [float(a.get(f"{metric}_mean", 0.0)) for a in agg]
    stds = [float(a.get(f"{metric}_std", 0.0)) for a in agg]
    max_v = max([m + s for m, s in zip(means, stds)] + [1.0 if not lower_is_better else max(means + [0.1])])
    if metric == "latency_seconds":
        max_v = max([m + s for m, s in zip(means, stds)] + [0.001])
    row_h = plot_h / max(1, len(agg))
    bar_h = row_h * 0.44
    items = []

    # gridlines
    for i in range(6):
        x = left + plot_w * i / 5
        label_v = max_v * i / 5
        label = f"{label_v*100:.0f}%" if percent else f"{label_v:.2f}{suffix}"
        items.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height-bottom}' stroke='#e2e8f0' stroke-dasharray='4 6'/>")
        items.append(f"<text x='{x:.1f}' y='{height-bottom+24}' text-anchor='middle' font-size='12' fill='#64748b'>{html.escape(label)}</text>")

    for i, a in enumerate(agg):
        y_center = top + i * row_h + row_h / 2
        y = y_center - bar_h / 2
        mu = float(a.get(f"{metric}_mean", 0.0))
        sigma = float(a.get(f"{metric}_std", 0.0))
        bw = (mu / max_v) * plot_w if max_v else 0.0
        error = (sigma / max_v) * plot_w if max_v else 0.0
        method = html.escape(str(a["method"]))
        fill, light = _palette(i)
        items.append(f"<text x='20' y='{y_center+4:.1f}' font-size='14' font-weight='700' fill='#0f172a'>{method}</text>")
        items.append(f"<rect x='{left}' y='{y:.1f}' width='{plot_w:.1f}' height='{bar_h:.1f}' rx='12' fill='#f1f5f9'/>")
        items.append(f"<rect x='{left}' y='{y:.1f}' width='{bw:.1f}' height='{bar_h:.1f}' rx='12' fill='{fill}' opacity='0.92'/>")
        # subtle highlight
        items.append(f"<rect x='{left}' y='{y:.1f}' width='{bw:.1f}' height='{bar_h/2:.1f}' rx='12' fill='white' opacity='0.12'/>")
        # error bar
        err_x1 = left + max(0.0, bw - error)
        err_x2 = left + min(plot_w, bw + error)
        items.append(f"<line x1='{err_x1:.1f}' y1='{y_center:.1f}' x2='{err_x2:.1f}' y2='{y_center:.1f}' stroke='{light}' stroke-width='4' stroke-linecap='round'/>")
        items.append(f"<line x1='{err_x1:.1f}' y1='{y_center-8:.1f}' x2='{err_x1:.1f}' y2='{y_center+8:.1f}' stroke='{light}' stroke-width='3'/>")
        items.append(f"<line x1='{err_x2:.1f}' y1='{y_center-8:.1f}' x2='{err_x2:.1f}' y2='{y_center+8:.1f}' stroke='{light}' stroke-width='3'/>")
        value_label = _fmt_mean_std(mu, sigma, suffix=suffix, pct=percent)
        items.append(f"<text x='{left + min(plot_w-8, bw + 10):.1f}' y='{y_center+4:.1f}' font-size='13' fill='#0f172a'>{html.escape(value_label)}</text>")
    foot = "Lower is better" if lower_is_better else "Higher is better"
    return f"""
    <div class='chart-card'>
      <div class='chart-title-row'><h3>{html.escape(title)}</h3><span class='hint'>{html.escape(foot)}</span></div>
      <svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>
        <rect width='{width}' height='{height}' rx='24' fill='white'/>
        <text x='{left}' y='32' font-size='12' fill='#64748b'>Bars show mean; whiskers show ± standard deviation.</text>
        {''.join(items)}
      </svg>
    </div>"""


def _svg_task_success_chart(agg: list[dict]) -> str:
    return _svg_metric_chart("General task success rate by method", agg, "task_success", percent=True)


def _svg_heatmap(rows: list[dict], metric: str = "overall_score") -> str:
    methods = sorted({str(r["method"]) for r in rows})
    task_types = sorted({str(r["task_type"]) for r in rows})
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(str(r["method"]), str(r["task_type"]))].append(float(r[metric]))
    width = 1040
    left = 210
    top = 90
    cell_w = 96
    cell_h = 42
    height = top + len(methods) * cell_h + 90
    items = []
    for j, tt in enumerate(task_types):
        x = left + j * cell_w + cell_w / 2
        label = tt.replace("_", " ")
        items.append(f"<text x='{x:.1f}' y='62' text-anchor='middle' font-size='12' font-weight='700' fill='#334155'>{html.escape(label)}</text>")
    for i, method in enumerate(methods):
        y = top + i * cell_h
        items.append(f"<text x='20' y='{y + 26:.1f}' font-size='13' font-weight='700' fill='#0f172a'>{html.escape(method)}</text>")
        for j, tt in enumerate(task_types):
            vals = groups.get((method, tt), [])
            mu = mean(vals) if vals else 0.0
            intensity = max(0.0, min(1.0, mu))
            # blue-green heatmap
            r = int(245 - 140 * intensity)
            g = int(247 - 35 * intensity)
            b = int(255 - 120 * intensity)
            fill = f"rgb({r},{g},{b})"
            x = left + j * cell_w
            items.append(f"<rect x='{x}' y='{y}' width='{cell_w-8}' height='{cell_h-8}' rx='12' fill='{fill}' stroke='#e2e8f0'/>")
            items.append(f"<text x='{x + (cell_w-8)/2:.1f}' y='{y + 24:.1f}' text-anchor='middle' font-size='12' font-weight='700' fill='#0f172a'>{mu:.2f}</text>")
    # legend
    legend_x = left
    legend_y = height - 50
    for idx, val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        r = int(245 - 140 * val)
        g = int(247 - 35 * val)
        b = int(255 - 120 * val)
        fill = f"rgb({r},{g},{b})"
        items.append(f"<rect x='{legend_x + idx*88}' y='{legend_y}' width='56' height='18' rx='9' fill='{fill}' stroke='#e2e8f0'/>")
        items.append(f"<text x='{legend_x + idx*88 + 28}' y='{legend_y + 34}' text-anchor='middle' font-size='11' fill='#475569'>{val:.2f}</text>")
    return f"""
    <div class='chart-card span-2'>
      <div class='chart-title-row'><h3>Task-type performance heatmap</h3><span class='hint'>Mean normalized score</span></div>
      <svg viewBox='0 0 {width} {height}' role='img' aria-label='Task-type performance heatmap'>
        <rect width='{width}' height='{height}' rx='24' fill='white'/>
        {''.join(items)}
      </svg>
    </div>"""


def _failure_summary(failures: dict) -> str:
    rows = []
    for method, vals in failures.items():
        reasons = defaultdict(int)
        for v in vals:
            reasons[str(v.get("reason", v.get("error", "unknown")))] += 1
        rows.append({"method": method, "failures": len(vals), "top_reason": "; ".join(f"{k}: {n}" for k, n in list(reasons.items())[:3])})
    return _table(rows, ["method", "failures", "top_reason"])


def _example_trace(traces: list[dict]) -> dict:
    preferred = [t for t in traces if t.get("method") == "Policy-as-Skill" and t.get("selected_skill")]
    return preferred[0] if preferred else (traces[0] if traces else {})


def _balanced_task_badges(manifest: dict | None) -> str:
    dist = (manifest or {}).get("task_type_distribution", {}) or {}
    if not dist:
        return ""
    badges = []
    for key, val in sorted(dist.items()):
        badges.append(f"<span class='tag'>{html.escape(key.replace('_', ' '))}: N={html.escape(str(val))}</span>")
    return "".join(badges)


def generate_report(result_dir: Path, metrics: list[dict], traces: list[dict], failures: dict, manifest: dict | None = None) -> None:
    cols = [
        "overall_score",
        "normalized_score",
        "raw_quality_score",
        "decision_quality_score",
        "evidence_quality_score",
        "governance_quality_score",
        "latency_efficiency_score",
        "governance_readiness_score",
        "traceability_score",
        "audit_completeness",
        "citation_precision",
        "policy_ref_recall",
        "evidence_faithfulness",
        "human_review_correctness",
        "update_adaptation_score",
        "task_success",
        "latency_seconds",
    ]
    agg = _group_stats(metrics, "method", cols)
    ex = _example_trace(traces)
    artifact_links = ["metrics.csv", "metrics.json", "traces.jsonl", "failures.json", "manifest.json"]
    link_html = "".join(f"<a class='pill' href='{html.escape(x)}'>{html.escape(x)}</a>" for x in artifact_links)

    worst_rows = sorted(metrics, key=lambda r: float(r["overall_score"]))[:12]
    worst_rows_fmt = []
    for r in worst_rows:
        worst_rows_fmt.append(
            {
                "task_id": r["task_id"],
                "task_type": r["task_type"],
                "method": r["method"],
                "overall_score": f"{float(r['overall_score']):.3f}",
                "citation_precision": f"{float(r['citation_precision']):.3f}",
                "policy_ref_recall": f"{float(r['policy_ref_recall']):.3f}",
                "evidence_faithfulness": f"{float(r['evidence_faithfulness']):.3f}",
                "governance_readiness_score": f"{float(r['governance_readiness_score']):.3f}",
            }
        )

    agg_table = []
    for row in agg:
        agg_table.append(
            {
                "method": row["method"],
                "n": row["n"],
                "normalized_score": _fmt_mean_std(row["normalized_score_mean"], row["normalized_score_std"]),
                "overall_score": _fmt_mean_std(row["overall_score_mean"], row["overall_score_std"]),
                "task_success": _fmt_mean_std(row["task_success_mean"], row["task_success_std"], pct=True),
                "decision_quality_score": _fmt_mean_std(row["decision_quality_score_mean"], row["decision_quality_score_std"]),
                "evidence_quality_score": _fmt_mean_std(row["evidence_quality_score_mean"], row["evidence_quality_score_std"]),
                "governance_quality_score": _fmt_mean_std(row["governance_quality_score_mean"], row["governance_quality_score_std"]),
                "latency_efficiency_score": _fmt_mean_std(row["latency_efficiency_score_mean"], row["latency_efficiency_score_std"]),
                "governance_readiness_score": _fmt_mean_std(row["governance_readiness_score_mean"], row["governance_readiness_score_std"]),
                "traceability_score": _fmt_mean_std(row["traceability_score_mean"], row["traceability_score_std"]),
                "audit_completeness": _fmt_mean_std(row["audit_completeness_mean"], row["audit_completeness_std"]),
                "citation_precision": _fmt_mean_std(row["citation_precision_mean"], row["citation_precision_std"]),
                "policy_ref_recall": _fmt_mean_std(row["policy_ref_recall_mean"], row["policy_ref_recall_std"]),
                "evidence_faithfulness": _fmt_mean_std(row["evidence_faithfulness_mean"], row["evidence_faithfulness_std"]),
                "human_review_correctness": _fmt_mean_std(row["human_review_correctness_mean"], row["human_review_correctness_std"]),
                "update_adaptation_score": _fmt_mean_std(row["update_adaptation_score_mean"], row["update_adaptation_score_std"]),
                "latency_seconds": _fmt_mean_std(row["latency_seconds_mean"], row["latency_seconds_std"], suffix="s"),
            }
        )

    best_method = max(agg, key=lambda x: x["overall_score_mean"], default={"method": "n/a", "overall_score_mean": 0.0})
    best_success = max(agg, key=lambda x: x["task_success_mean"], default={"method": "n/a", "task_success_mean": 0.0})

    report = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Policy-as-Skill Research Report</title>
<style>
:root {{
  --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --blue:#2563eb; --blue2:#4f46e5; --bg:#f4f7fb; --card:#ffffff;
  --shadow:0 18px 48px rgba(15,23,42,.08); --radius:24px;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#eff6ff 0,#f8fafc 180px,#f4f7fb 100%); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; line-height:1.55; }}
header {{ padding:68px 7vw 54px; color:#fff; background:
radial-gradient(circle at 15% 10%, rgba(147,197,253,.65), transparent 26%),
radial-gradient(circle at 82% 18%, rgba(196,181,253,.45), transparent 22%),
linear-gradient(135deg,#0f172a 0%,#1d4ed8 48%,#4338ca 72%,#7c3aed 100%); }}
h1 {{ font-size:clamp(36px,5vw,60px); margin:0 0 12px; letter-spacing:-.045em; line-height:1.0; }}
header p {{ max-width:1040px; font-size:20px; opacity:.97; }}
main {{ width:min(1320px,94vw); margin:26px auto 80px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:30px; margin:22px 0; box-shadow:var(--shadow); }}
h2 {{ font-size:29px; margin:0 0 10px; letter-spacing:-.025em; }}
h3 {{ margin:0; font-size:18px; letter-spacing:-.01em; }}
p.lead {{ color:#334155; max-width:1000px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:20px; align-items:start; }}
.span-2 {{ grid-column:1 / -1; }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:16px; }}
.kpi {{ background:linear-gradient(180deg,#ffffff,#f8fafc); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow: inset 0 1px 0 rgba(255,255,255,.7); }}
.kpi span {{ color:#475569; font-size:13px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
.kpi b {{ display:block; margin-top:6px; font-size:28px; color:#111827; }}
.pill {{ display:inline-block; padding:8px 12px; margin:4px 6px 4px 0; border-radius:999px; background:#dbeafe; color:#1d4ed8; text-decoration:none; font-weight:700; font-size:13px; }}
.tag {{ display:inline-flex; align-items:center; padding:7px 11px; margin:6px 8px 0 0; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; font-weight:700; }}
.info-row {{ display:flex; flex-wrap:wrap; gap:18px; align-items:center; margin-top:8px; }}
.callout {{ background:linear-gradient(180deg,#eff6ff,#ffffff); border:1px solid #bfdbfe; border-radius:18px; padding:16px 18px; }}
.muted {{ color:var(--muted); }}
.small {{ color:var(--muted); font-size:14px; }}
.chart-card {{ border:1px solid var(--line); border-radius:22px; background:linear-gradient(180deg,#ffffff,#fbfdff); padding:16px 16px 8px; overflow:hidden; box-shadow: inset 0 1px 0 rgba(255,255,255,.7); }}
.chart-title-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }}
.chart-card .hint {{ color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
.chart-card svg {{ width:100%; height:auto; display:block; }}
.method-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:16px; }}
.method-card {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:linear-gradient(180deg,#fff,#f8fafc); }}
.method-card b {{ display:block; margin-bottom:6px; font-size:16px; }}
.method-card p {{ margin:0; color:#475569; font-size:14px; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; font-size:14px; overflow:hidden; border:1px solid var(--line); border-radius:18px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; background:#fff; }}
tr:last-child td {{ border-bottom:none; }}
th {{ background:#f8fafc; color:#334155; font-size:12px; text-transform:uppercase; letter-spacing:.04em; position:sticky; top:0; }}
pre {{ background:#0f172a; color:#e2e8f0; padding:18px; border-radius:18px; overflow:auto; max-height:520px; font-size:13px; }}
@media (max-width: 1040px) {{ .grid,.kpi-grid,.method-grid {{ grid-template-columns:1fr; }} .span-2 {{ grid-column:auto; }} section {{ padding:22px; }} }}
</style>
</head>
<body>
<header>
<h1>Policy-as-Skill</h1>
<p>Governed Agentic AI for traceable policy-aware decision support. This report is generated from the local Docker/Ollama experiment using the static benchmark in data/tasks and is intended as a polished, paper-oriented reproducibility artifact.</p>
</header>
<main>
<section>
<h2>A. Executive Summary</h2>
<p class='lead'><b>Policy-as-Prompt</b> encodes policies directly as prompts. <b>Policy-as-Skill</b> turns policies into reusable, governed, versioned, and auditable agent capabilities that combine retrieval, reasoning, review routing, validation, and evaluation.</p>
<div class='kpi-grid'>
  <div class='kpi'><span>Tasks evaluated</span><b>{len({r['task_id'] for r in metrics})}</b></div>
  <div class='kpi'><span>Methods</span><b>{len({r['method'] for r in metrics})}</b></div>
  <div class='kpi'><span>Best normalized score</span><b style='font-size:22px'>{html.escape(str(best_method.get('method','n/a')))}</b></div>
  <div class='kpi'><span>Best success rate</span><b style='font-size:22px'>{html.escape(str(best_success.get('method','n/a')))}</b></div>
</div>
<div class='info-row'>
  {_balanced_task_badges(manifest)}
</div>
<div class='callout' style='margin-top:16px'>
  <b>Adversarial benchmark configuration.</b> The experiment uses the static curated benchmark in <code>data/tasks/benchmark_tasks.jsonl</code>. It includes balanced core tasks plus semantic-trap cases with negation, missing mandatory controls, version conflicts, and false-positive low-risk wording. Aggregate results are reported as <b>mean ± standard deviation</b> across tasks.
</div>
<p class='small'>The platform avoids ground-truth leakage: expected answers are used only by the evaluator, never during answer generation.</p>
</section>

<section><h2>B. Architecture Diagram</h2>{ARCHITECTURE}</section>
<section><h2>C. Agentic AI Policy-Skill Loop</h2>{LOOP}</section>

<section>
<h2>D. Method Comparison</h2>
<div class='method-grid'>
  <div class='method-card'><b>Direct LLM</b><p>No retrieval. Useful as a weak baseline for memory-only answers.</p></div>
  <div class='method-card'><b>Keyword Search</b><p>Lexical retrieval with extractive answer. Strong speed, weak governance.</p></div>
  <div class='method-card'><b>Standard / Hybrid RAG</b><p>Retrieved context plus LLM generation. Better access to evidence, limited audit controls.</p></div>
  <div class='method-card'><b>Policy-as-Prompt</b><p>Policy chunks are included directly in the prompt; structured variant uses JSON policy artifacts.</p></div>
  <div class='method-card'><b>Policy-as-Skill No Audit</b><p>Ablation: selected skill and scoped retrieval, without strict audit/citation validation.</p></div>
  <div class='method-card'><b>Policy-as-Skill</b><p>Full method: skill registry, scoped retrieval, required evidence checks, citation validation, review routing, policy hashes, and audit trail.</p></div>
</div>
</section>

<section>
<h2>E. Metrics and Ablation View</h2>
<p class='lead'>Each metric chart below shows the <b>mean</b> bar and the <b>± standard deviation</b> whisker. The headline score is now a run-normalized score: bounded decision, evidence, and governance metrics are combined with a small inverse min-max latency term. Decision errors are penalized so Keyword Search cannot win merely through lexical citation overlap.</p>
<div class='grid'>
{_svg_metric_chart('Normalized score by method', agg, 'normalized_score')}
{_svg_metric_chart('Decision quality by method', agg, 'decision_quality_score')}
{_svg_task_success_chart(agg)}
{_svg_metric_chart('Governance quality by method', agg, 'governance_quality_score')}
{_svg_metric_chart('Governance readiness by method', agg, 'governance_readiness_score')}
{_svg_metric_chart('Citation precision by method', agg, 'citation_precision')}
{_svg_metric_chart('Audit completeness by method', agg, 'audit_completeness')}
{_svg_metric_chart('Latency by method', agg, 'latency_seconds', suffix='s', lower_is_better=True)}
</div>
<h3 style='margin:20px 0 12px'>Aggregate table (mean ± std)</h3>
{_table(agg_table, ['method','n','normalized_score','task_success','decision_quality_score','evidence_quality_score','governance_quality_score','governance_readiness_score','traceability_score','audit_completeness','citation_precision','policy_ref_recall','evidence_faithfulness','human_review_correctness','latency_efficiency_score','latency_seconds'])}
</section>

<section>
<h2>F. Task-Type Performance</h2>
<p class='lead'>The heatmap summarizes mean normalized score by method and task type. The table reports both <b>normalized score</b> and <b>task success rate</b> as mean ± standard deviation.</p>
<div class='grid'>
{_svg_heatmap(metrics)}
</div>
{_task_table(metrics)}
</section>

<section>
<h2>G. Example Policy-as-Skill Trace</h2>
<p><b>Question:</b> {html.escape(str(ex.get('question', '')))}</p>
<p><b>Selected skill:</b> {html.escape(str(ex.get('selected_skill', '')))} | <b>Decision:</b> {html.escape(str(ex.get('decision', '')))} | <b>Human review:</b> {html.escape(str(ex.get('human_review_required', '')))}</p>
<p><b>Reasoning summary:</b> {html.escape(str(ex.get('reasoning_summary', '')))}</p>
<h3 style='margin:16px 0 12px'>Audit record excerpt</h3>
<pre>{html.escape(json.dumps({k: ex.get(k) for k in ['task_id','method','selected_skill','policy_skill_version','prompt_hash','policy_hashes','citations','validation']}, indent=2, ensure_ascii=False))}</pre>
<h3 style='margin:16px 0 12px'>Retrieved evidence excerpt</h3>
<pre>{html.escape(json.dumps((ex.get('evidence') or [])[:3], indent=2, ensure_ascii=False))}</pre>
</section>

<section>
<h2>H. Failure Analysis</h2>
<p class='lead'>Failures are rows below the configured quality threshold or runtime exceptions. This section helps separate retrieval failures, generation failures, citation failures, and governance failures.</p>
{_failure_summary(failures)}
<h3 style='margin:16px 0 12px'>Worst metric rows</h3>
{_table(worst_rows_fmt, ['task_id','task_type','method','overall_score','citation_precision','policy_ref_recall','evidence_faithfulness','governance_readiness_score'])}
</section>

<section>
<h2>I. Reproducibility Manifest</h2>
<p>{link_html}</p>
<pre>{html.escape(json.dumps(manifest or {}, indent=2, ensure_ascii=False))}</pre>
</section>

<section>
<h2>J. Paper-Oriented Conclusion</h2>
<p>This platform operationalizes the central research claim: in regulated enterprises, AI quality is not only answer accuracy. A policy-aware agent must be traceable, reviewable, version-aware, evidence-grounded, and auditable. The <b>Policy-as-Skill</b> method is designed to make these properties first-class measurable outputs rather than informal prompt instructions.</p>
</section>
</main>
</body></html>"""
    (result_dir / "report.html").write_text(report, encoding="utf-8")
