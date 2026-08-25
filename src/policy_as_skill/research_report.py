from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .research_metrics import method_summary


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _num(value: float) -> str:
    return f"{float(value):.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _write_paper_tables(result_dir: Path, summary: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Paper-facing result tables", "",
        "Primary outcome metrics are reported separately from evidence and governance metrics.",
        "The composite governance-readiness index is secondary and must not be described as decision accuracy.", "",
        "## Decision and review routing", "",
        "| Method | Exact decision accuracy | Macro-F1 | Review precision | Review recall | Review F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['decision_exact_accuracy']:.3f} | {s['decision_macro_f1']:.3f} | {s['review_precision']:.3f} | {s['review_recall']:.3f} | {s['review_f1']:.3f} |")
    lines += ["", "## Evidence grounding", "", "| Method | Citation precision | Policy-reference recall | Evidence faithfulness | Unsupported claim rate |", "|---|---:|---:|---:|---:|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['citation_precision']:.3f} | {s['policy_ref_recall']:.3f} | {s['evidence_faithfulness']:.3f} | {s['unsupported_claim_rate']:.3f} |")
    lines += ["", "## Governance and efficiency", "", "| Method | Audit completeness | Traceability | Governance quality | Governance-readiness index | Mean latency (s) |", "|---|---:|---:|---:|---:|---:|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['audit_completeness']:.3f} | {s['traceability_score']:.3f} | {s['governance_quality_score']:.3f} | {s['governance_readiness_index']:.3f} | {s['latency_seconds_mean']:.3f} |")
    (result_dir / "paper_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(result_dir: Path, rows: list[dict], traces: list[dict], failures: dict[str, list[dict]], manifest: dict, *, sensitivity: dict | None = None, threshold_sensitivity: dict | None = None, annotation_agreement: dict | None = None, system_profile: dict | None = None) -> None:
    summary = method_summary(rows, seed=int(manifest.get("seed", 7)), bootstrap_iterations=int(manifest.get("bootstrap_iterations", 1000)))
    _write_paper_tables(result_dir, summary)
    (result_dir / "research_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    protocol = manifest.get("evaluation_protocol", {})
    protocol_rows = [
        ["Evaluation split", str(protocol.get("evaluation_split", "unknown"))],
        ["Model parameter training on benchmark", str(protocol.get("model_parameter_training_on_benchmark", False))],
        ["Model-level unseen", str(protocol.get("model_level_unseen", True))],
        ["System-development unseen", str(protocol.get("system_development_unseen", False))],
        ["System-development informed", str(protocol.get("system_development_informed", True))],
        ["Benchmark SHA-256", str(protocol.get("benchmark_sha256", ""))],
    ]

    decision_rows, evidence_rows, governance_rows = [], [], []
    for method, s in summary.items():
        ci = s["decision_exact_accuracy_ci95"]
        decision_rows.append([method, _pct(s["decision_exact_accuracy"]), f"{_pct(ci[0])}–{_pct(ci[1])}", _num(s["decision_macro_f1"]), _num(s["review_precision"]), _num(s["review_recall"]), _num(s["review_f1"])])
        evidence_rows.append([method, _num(s["citation_precision"]), _num(s["policy_ref_recall"]), _num(s["evidence_faithfulness"]), _num(s["unsupported_claim_rate"])])
        governance_rows.append([method, _num(s["audit_completeness"]), _num(s["traceability_score"]), _num(s["governance_quality_score"]), _num(s["governance_readiness_index"]), _num(s["latency_seconds_mean"])])

    sensitivity_rows = []
    if sensitivity:
        for method in summary:
            sensitivity_rows.append([method, _pct(sensitivity.get("top_rank_share", {}).get(method, 0.0)), _num(sensitivity.get("mean_rank", {}).get(method, 0.0) or 0.0), _num(sensitivity.get("rank_std", {}).get(method, 0.0) or 0.0)])

    annotation = annotation_agreement or {}
    annotation_rows = [["Status", str(annotation.get("status", "not_available"))], ["Rows", str(annotation.get("rows", 0))], ["Paired items", str(annotation.get("paired_items", 0))], ["Supported-label Cohen κ", str(annotation.get("supported_kappa"))], ["Contradiction Cohen κ", str(annotation.get("contradiction_kappa"))]]
    style = "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#1f2937}h1,h2{color:#111827}.note{padding:12px 16px;background:#f3f4f6;border-left:4px solid #6b7280;margin:16px 0}.good{background:#ecfdf5;border-left-color:#059669}.warn{background:#fffbeb;border-left-color:#d97706}table{border-collapse:collapse;width:100%;margin:12px 0 28px;font-size:14px}th,td{border:1px solid #d1d5db;padding:8px;text-align:left}th{background:#f9fafb}code{background:#f3f4f6;padding:2px 4px;border-radius:4px}"
    system_unseen = bool(protocol.get("system_development_unseen"))
    protocol_note = "This run satisfies the repository's frozen held-out protocol." if system_unseen else "This run is a development/system-informed evaluation. It is unseen by the LLM at the parameter-training level, but it must not be presented as a frozen system-level held-out result."

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Policy-as-Skill Research Evaluation</title><style>{style}</style></head><body>
<h1>Policy-as-Skill — reviewer-hardened research evaluation</h1><div class="note {'good' if system_unseen else 'warn'}">{html.escape(protocol_note)}</div>
<h2>Evaluation provenance</h2>{_table(['Field','Value'], protocol_rows)}<p>{html.escape(str(protocol.get('claim_guidance','')))}</p>
<h2>Primary: decision and human-review performance</h2><p>These metrics do not reward audit fields or Policy-as-Skill-specific metadata.</p>{_table(['Method','Exact decision accuracy','95% bootstrap CI','Decision macro-F1','Review precision','Review recall','Review F1'], decision_rows)}
<h2>Evidence grounding</h2>{_table(['Method','Citation precision','Policy-ref recall','Evidence faithfulness','Unsupported claim rate'], evidence_rows)}
<h2>Governance and efficiency</h2><p>The governance-readiness index is retained as a secondary diagnostic, not as a synonym for decision accuracy.</p>{_table(['Method','Audit completeness','Traceability','Governance quality','Governance-readiness index','Mean latency (s)'], governance_rows)}
<h2>Composite-weight sensitivity</h2><p>Ranks are recomputed over a simplex of decision/evidence/governance/answer-similarity weights.</p>{_table(['Method','Share ranked first','Mean rank','Rank std.'], sensitivity_rows) if sensitivity_rows else '<p>Not computed.</p>'}
<h2>Human annotation agreement</h2>{_table(['Field','Value'], annotation_rows)}<p>If agreement is unavailable, the report does not claim completed independent human validation.</p>
<h2>Hardware/system profile</h2><pre>{html.escape(json.dumps(system_profile or {}, indent=2, ensure_ascii=False))}</pre>
<h2>Ablation interpretation</h2><p>The four Policy-as-Skill rows isolate skill-scoped retrieval, deterministic controller, audit/validation, and the full combination. The research runner records <code>legacy_benchmark_phrase_controller_used=false</code>.</p>
<h2>Artifacts</h2><p><code>metrics.csv</code>, <code>metrics.json</code>, <code>research_summary.json</code>, <code>sensitivity.json</code>, <code>annotation_agreement.json</code>, <code>system_profile.json</code>, <code>paper_tables.md</code>, and <code>traces.jsonl</code>.</p>
</body></html>'''
    (result_dir / "report.html").write_text(page, encoding="utf-8")
