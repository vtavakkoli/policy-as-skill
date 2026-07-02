import html, json
from pathlib import Path
from statistics import mean
from .diagrams import ARCHITECTURE, LOOP

def _group_mean(rows, key, cols):
    groups={}
    for r in rows: groups.setdefault(r[key], []).append(r)
    return [{'method':k, **{c:round(mean(x[c] for x in v),3) for c in cols}} for k,v in groups.items()]

def _task_table(rows):
    groups={}
    for r in rows: groups.setdefault((r['method'],r['task_type']), []).append(r['overall_score'])
    trs=''.join(f'<tr><td>{html.escape(k[0])}</td><td>{html.escape(k[1])}</td><td>{mean(v):.3f}</td></tr>' for k,v in sorted(groups.items()))
    return '<table><tr><th>Method</th><th>Task type</th><th>Overall score</th></tr>'+trs+'</table>'

def generate_report(result_dir: Path, metrics: list[dict], traces: list[dict], failures: dict) -> None:
    agg=_group_mean(metrics,'method',['overall_score','traceability_score','citation_coverage','latency_seconds'])
    ex=next((t for t in traces if t['method']=='Policy-as-Skill'), traces[0])
    labels=[a['method'] for a in agg]
    def arr(col): return [a[col] for a in agg]
    html_doc=f"""<!doctype html><html><head><meta charset='utf-8'><title>Policy-as-Skill Report</title>
<script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script><script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<style>body{{font-family:Inter,Arial,sans-serif;margin:0;background:#f6f8fb;color:#172033}} header{{background:linear-gradient(135deg,#172033,#3156d4);color:white;padding:56px 80px}} section{{background:white;margin:24px auto;padding:28px;border-radius:18px;max-width:1180px;box-shadow:0 10px 30px #d9e0ee}} h1{{font-size:42px}} h2{{color:#1f3a93}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #e5e7eb;padding:10px;text-align:left}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} .card{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:14px;padding:16px}} code,pre{{background:#0f172a;color:#e2e8f0;border-radius:10px;padding:12px;white-space:pre-wrap}} .badge{{background:#dbeafe;color:#1e40af;padding:4px 8px;border-radius:999px}}</style></head><body>
<header><h1>Policy-as-Skill: Governed Agentic AI</h1><p>Traceable policy-aware decision support research prototype using external Ollama model <b>gemma4:e2b</b>.</p></header>
<section><h2>A. Executive Summary</h2><p>This prototype evaluates whether policies should be treated merely as prompt text or as governed reusable agent capabilities. <b>Policy-as-Skill</b> packages retrieval scope, risk level, allowed actions, human-review rules, evaluation criteria, reasoning prompts, and audit traces into reusable skills.</p></section>
<section><h2>B. Architecture Diagram</h2><div class='mermaid'>{ARCHITECTURE}</div></section>
<section><h2>C. Agentic AI Policy-Skill Loop</h2><div class='mermaid'>{LOOP}</div></section>
<section><h2>D. Method Comparison</h2><table><tr><th>Method</th><th>Policy representation</th><th>Governance capability</th></tr><tr><td>Keyword Search</td><td>Lexical matches</td><td>Low</td></tr><tr><td>Standard RAG</td><td>Retrieved chunks</td><td>Medium</td></tr><tr><td>Policy-as-Prompt</td><td>Policy text in prompt</td><td>Medium</td></tr><tr><td><b>Policy-as-Skill</b></td><td>Versionable skill capability</td><td>High</td></tr></table></section>
<section><h2>E. Metrics</h2><div class='grid'><div class='card'><canvas id='overall'></canvas></div><div class='card'><canvas id='trace'></canvas></div><div class='card'><canvas id='citation'></canvas></div><div class='card'><canvas id='latency'></canvas></div></div><h3>Task-type Performance</h3>{_task_table(metrics)}</section>
<section><h2>F. Example Policy-as-Skill Trace</h2><p><span class='badge'>{html.escape(ex.get('selected_skill') or '')}</span></p><p><b>Question:</b> {html.escape(ex['question'])}</p><p><b>Decision:</b> {html.escape(str(ex.get('decision')))} | <b>Human review:</b> {ex.get('human_review_required')}</p><p><b>Reasoning summary:</b> {html.escape(ex.get('reasoning_summary',''))}</p><pre>{html.escape(json.dumps(ex.get('evidence',[])[:2], indent=2))}</pre></section>
<section><h2>G. Failure Analysis</h2><p>Common failures include missing citations, incomplete policy-reference recall, weak reasoning summaries, and absent audit fields.</p><pre>{html.escape(json.dumps(failures, indent=2))}</pre></section>
<section><h2>H. Governance and Auditability Analysis</h2><p>Policy-as-Skill binds each decision to a selected skill, evidence, human-review flag, reasoning summary, decision status, timestamps, and evaluation metrics.</p></section>
<section><h2>I. Paper-Oriented Conclusion</h2><p>Policy-as-Skill improves not only answer quality, but also governance readiness, traceability, and auditability compared with prompt-only and retrieval-only baselines.</p></section>
<script>mermaid.initialize({{startOnLoad:true}}); const labels={json.dumps(labels)}; function chart(id,label,data){{new Chart(document.getElementById(id),{{type:'bar',data:{{labels,datasets:[{{label,data,backgroundColor:'#3156d4'}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true}}}}}}}})}} chart('overall','Overall Score',{json.dumps(arr('overall_score'))}); chart('trace','Traceability',{json.dumps(arr('traceability_score'))}); chart('citation','Citation Coverage',{json.dumps(arr('citation_coverage'))}); chart('latency','Latency Seconds',{json.dumps(arr('latency_seconds'))});</script>
</body></html>"""
    (result_dir/'report.html').write_text(html_doc, encoding='utf-8')
