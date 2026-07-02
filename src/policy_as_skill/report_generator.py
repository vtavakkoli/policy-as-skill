from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .diagrams import ARCHITECTURE, LOOP


def _group_mean(rows: list[dict], key: str, cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return [
        {"method": k, **{c: round(mean(float(x[c]) for x in v), 3) for c in cols}}
        for k, v in sorted(groups.items())
    ]


def _table(rows: list[dict[str, Any]], cols: list[str], limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    head = "".join(f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in cols)
    body = ""
    for r in shown:
        body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _task_table(rows: list[dict]) -> str:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(str(r["method"]), str(r["task_type"]))].append(float(r["overall_score"]))
    out = [
        {"method": k[0], "task_type": k[1], "n": len(v), "overall_score": f"{mean(v):.3f}"}
        for k, v in sorted(groups.items())
    ]
    return _table(out, ["method", "task_type", "n", "overall_score"])


def _svg_bar_chart(title: str, agg: list[dict], metric: str, suffix: str = "") -> str:
    width, height = 860, 320
    left, right, top, bottom = 190, 30, 48, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(a.get(metric, 0)) for a in agg]
    max_v = max(values + [1e-9])
    if metric != "latency_seconds":
        max_v = max(1.0, max_v)
    bar_h = plot_h / max(1, len(agg)) * 0.62
    gap = plot_h / max(1, len(agg)) * 0.38
    items = []
    for i, a in enumerate(agg):
        y = top + i * (bar_h + gap)
        v = float(a.get(metric, 0))
        bw = 0 if max_v == 0 else (v / max_v) * plot_w
        label = html.escape(str(a["method"]))
        val = f"{v:.3f}{suffix}"
        items.append(f"<text x='18' y='{y + bar_h * .68:.1f}' font-size='13' fill='#334155'>{label}</text>")
        items.append(f"<rect x='{left}' y='{y:.1f}' width='{bw:.1f}' height='{bar_h:.1f}' rx='8' fill='#3156d4'/>")
        items.append(f"<text x='{left + bw + 8:.1f}' y='{y + bar_h * .68:.1f}' font-size='13' fill='#0f172a'>{val}</text>")
    return f"""
    <div class='chart-card'>
      <h3>{html.escape(title)}</h3>
      <svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>
        <rect width='{width}' height='{height}' rx='18' fill='#f8fafc'/>
        <line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#cbd5e1'/>
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


def generate_report(result_dir: Path, metrics: list[dict], traces: list[dict], failures: dict, manifest: dict | None = None) -> None:
    cols = [
        "overall_score",
        "governance_readiness_score",
        "traceability_score",
        "audit_completeness",
        "citation_precision",
        "policy_ref_recall",
        "evidence_faithfulness",
        "human_review_correctness",
        "update_adaptation_score",
        "latency_seconds",
    ]
    agg = _group_mean(metrics, "method", cols)
    ex = _example_trace(traces)
    artifact_links = ["metrics.csv", "metrics.json", "traces.jsonl", "failures.json", "manifest.json", "benchmark_generated.jsonl"]
    link_html = "".join(f"<a class='pill' href='{html.escape(x)}'>{html.escape(x)}</a>" for x in artifact_links)

    top_rows = sorted(metrics, key=lambda r: (r["method"], r["task_id"]))[:16]
    worst_rows = sorted(metrics, key=lambda r: float(r["overall_score"]))[:12]

    report = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Policy-as-Skill Research Report</title>
<style>
:root {{ --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --blue:#3156d4; --bg:#f6f8fb; --card:#ffffff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; line-height:1.55; }}
header {{ padding:64px 7vw; color:#fff; background: radial-gradient(circle at 20% 10%, #60a5fa, transparent 30%), linear-gradient(135deg,#0f172a,#1d4ed8 55%,#7c3aed); }}
h1 {{ font-size:clamp(34px,5vw,58px); margin:0 0 14px; letter-spacing:-.04em; line-height:1.0; }}
header p {{ max-width:980px; font-size:20px; opacity:.95; }}
main {{ width:min(1220px,94vw); margin:28px auto 80px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:24px; padding:30px; margin:24px 0; box-shadow:0 16px 40px rgba(15,23,42,.06); }}
h2 {{ font-size:27px; margin:0 0 14px; letter-spacing:-.02em; }}
h3 {{ margin:6px 0 12px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:14px; }}
.kpi {{ background:#f8fafc; border:1px solid var(--line); border-radius:18px; padding:18px; }}
.kpi b {{ display:block; font-size:28px; color:var(--blue); }}
.pill {{ display:inline-block; padding:8px 12px; margin:5px 6px 5px 0; border-radius:999px; background:#dbeafe; color:#1e3a8a; text-decoration:none; font-weight:700; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#f8fafc; color:#334155; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
pre {{ background:#0f172a; color:#e2e8f0; padding:18px; border-radius:18px; overflow:auto; max-height:520px; font-size:13px; }}
.chart-card {{ border:1px solid var(--line); border-radius:20px; background:#fff; padding:16px; overflow:hidden; }}
.chart-card svg {{ width:100%; height:auto; }}
.small {{ color:var(--muted); font-size:14px; }}
.method-card {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:#f8fafc; }}
@media (max-width: 900px) {{ .grid,.kpi-grid {{ grid-template-columns:1fr; }} section {{ padding:20px; }} }}
</style>
</head>
<body>
<header>
<h1>Policy-as-Skill</h1>
<p>Governed Agentic AI for traceable policy-aware decision support. This report is generated automatically from the local Docker/Ollama experiment and is intended as a paper-oriented reproducibility artifact.</p>
</header>
<main>
<section>
<h2>A. Executive Summary</h2>
<p><b>Policy-as-Prompt</b> encodes policies directly as prompts. <b>Policy-as-Skill</b> turns policies into reusable, governed, versioned, and auditable agent capabilities that combine retrieval, reasoning, review routing, validation, and evaluation.</p>
<div class='kpi-grid'>
  <div class='kpi'><span>Tasks evaluated</span><b>{len({r['task_id'] for r in metrics})}</b></div>
  <div class='kpi'><span>Methods</span><b>{len({r['method'] for r in metrics})}</b></div>
  <div class='kpi'><span>Trace records</span><b>{len(traces)}</b></div>
  <div class='kpi'><span>Generated at</span><b style='font-size:15px'>{html.escape(str((manifest or {}).get('timestamp', 'n/a')))}</b></div>
</div>
<p class='small'>The platform avoids ground-truth leakage: expected answers are used only by the evaluator, never during answer generation.</p>
</section>

<section><h2>B. Architecture Diagram</h2>{ARCHITECTURE}</section>
<section><h2>C. Agentic AI Policy-Skill Loop</h2>{LOOP}</section>

<section>
<h2>D. Method Comparison</h2>
<div class='grid'>
  <div class='method-card'><b>Direct LLM</b><p>No retrieval. Useful as a weak baseline for memory-only answers.</p></div>
  <div class='method-card'><b>Keyword Search</b><p>Lexical retrieval with extractive answer. Low governance.</p></div>
  <div class='method-card'><b>Standard / Hybrid RAG</b><p>Retrieved context plus LLM generation. Better access to evidence, limited audit controls.</p></div>
  <div class='method-card'><b>Policy-as-Prompt</b><p>Policy chunks are included directly in the prompt; structured variant uses JSON policy artifacts.</p></div>
  <div class='method-card'><b>Policy-as-Skill No Audit</b><p>Ablation: selected skill and scoped retrieval, without strict audit/citation validation.</p></div>
  <div class='method-card'><b>Policy-as-Skill</b><p>Full method: skill registry, scoped retrieval, required evidence checks, citation validation, review routing, policy hashes, and audit trail.</p></div>
</div>
</section>

<section>
<h2>E. Metrics and Ablation View</h2>
<p>Metrics include answer similarity, citation precision, policy-reference recall, evidence faithfulness, decision accuracy, traceability, audit completeness, governance readiness, and policy-update adaptation.</p>
<div class='grid'>
{_svg_bar_chart('Overall score by method', agg, 'overall_score')}
{_svg_bar_chart('Governance readiness by method', agg, 'governance_readiness_score')}
{_svg_bar_chart('Traceability by method', agg, 'traceability_score')}
{_svg_bar_chart('Citation precision by method', agg, 'citation_precision')}
{_svg_bar_chart('Audit completeness by method', agg, 'audit_completeness')}
{_svg_bar_chart('Latency by method', agg, 'latency_seconds', 's')}
</div>
<h3>Aggregate table</h3>
{_table(agg, ['method'] + cols)}
</section>

<section>
<h2>F. Task-Type Performance</h2>
{_task_table(metrics)}
</section>

<section>
<h2>G. Example Policy-as-Skill Trace</h2>
<p><b>Question:</b> {html.escape(str(ex.get('question', '')))}</p>
<p><b>Selected skill:</b> {html.escape(str(ex.get('selected_skill', '')))} | <b>Decision:</b> {html.escape(str(ex.get('decision', '')))} | <b>Human review:</b> {html.escape(str(ex.get('human_review_required', '')))}</p>
<p><b>Reasoning summary:</b> {html.escape(str(ex.get('reasoning_summary', '')))}</p>
<h3>Audit record excerpt</h3>
<pre>{html.escape(json.dumps({k: ex.get(k) for k in ['task_id','method','selected_skill','policy_skill_version','prompt_hash','policy_hashes','citations','validation']}, indent=2, ensure_ascii=False))}</pre>
<h3>Retrieved evidence excerpt</h3>
<pre>{html.escape(json.dumps((ex.get('evidence') or [])[:3], indent=2, ensure_ascii=False))}</pre>
</section>

<section>
<h2>H. Failure Analysis</h2>
<p>Failures are rows below the configured score threshold or runtime exceptions. For paper analysis, inspect the trace file to separate retrieval failures, generation failures, citation failures, and governance failures.</p>
{_failure_summary(failures)}
<h3>Worst metric rows</h3>
{_table(worst_rows, ['task_id','task_type','method','overall_score','citation_precision','policy_ref_recall','evidence_faithfulness','governance_readiness_score'])}
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
