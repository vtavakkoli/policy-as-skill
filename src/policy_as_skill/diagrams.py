ARCHITECTURE = """
<svg viewBox="0 0 1180 420" role="img" aria-label="Policy-as-Skill architecture" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g1" x1="0" x2="1"><stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#7c3aed"/></linearGradient>
    <filter id="shadow"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#94a3b8" flood-opacity="0.25"/></filter>
  </defs>
  <rect width="1180" height="420" rx="26" fill="#f8fafc"/>
  <text x="50" y="54" font-family="Inter,Arial" font-size="28" font-weight="800" fill="#0f172a">Policy-as-Skill Reference Architecture</text>
  <g font-family="Inter,Arial" font-size="15" font-weight="700" fill="#0f172a" filter="url(#shadow)">
    <rect x="45" y="105" width="135" height="82" rx="18" fill="#dbeafe"/><text x="68" y="138">Policy</text><text x="68" y="162">Repository</text>
    <rect x="215" y="105" width="135" height="82" rx="18" fill="#e0e7ff"/><text x="240" y="138">Trusted</text><text x="240" y="162">Retrieval</text>
    <rect x="385" y="105" width="135" height="82" rx="18" fill="#ede9fe"/><text x="409" y="138">Skill</text><text x="409" y="162">Registry</text>
    <rect x="555" y="105" width="135" height="82" rx="18" fill="#fce7f3"/><text x="579" y="138">Agent</text><text x="579" y="162">Planner</text>
    <rect x="725" y="105" width="135" height="82" rx="18" fill="#fee2e2"/><text x="746" y="138">Ollama</text><text x="746" y="162">Reasoning</text>
    <rect x="895" y="105" width="135" height="82" rx="18" fill="#dcfce7"/><text x="918" y="138">Human</text><text x="918" y="162">Review</text>
    <rect x="45" y="265" width="155" height="82" rx="18" fill="#fef9c3"/><text x="72" y="298">Audit</text><text x="72" y="322">Trail</text>
    <rect x="240" y="265" width="155" height="82" rx="18" fill="#cffafe"/><text x="268" y="298">Evaluation</text><text x="268" y="322">Engine</text>
    <rect x="435" y="265" width="155" height="82" rx="18" fill="#ffedd5"/><text x="463" y="298">Metrics +</text><text x="463" y="322">Ablations</text>
    <rect x="630" y="265" width="155" height="82" rx="18" fill="#dcfce7"/><text x="658" y="298">HTML</text><text x="658" y="322">Report</text>
    <rect x="825" y="265" width="205" height="82" rx="18" fill="#e2e8f0"/><text x="854" y="298">Paper-ready</text><text x="854" y="322">Evidence Package</text>
  </g>
  <g stroke="url(#g1)" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M180 146H215"/><path d="M350 146H385"/><path d="M520 146H555"/><path d="M690 146H725"/><path d="M860 146H895"/>
    <path d="M963 187C963 230 170 220 123 265"/><path d="M200 306H240"/><path d="M395 306H435"/><path d="M590 306H630"/><path d="M785 306H825"/>
  </g>
</svg>
"""

LOOP = """
<svg viewBox="0 0 1000 560" role="img" aria-label="Agentic AI policy skill loop" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="loopg" x1="0" x2="1"><stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#16a34a"/></linearGradient>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#2563eb"/></marker>
  </defs>
  <rect width="1000" height="560" rx="28" fill="#f8fafc"/>
  <text x="60" y="60" font-family="Inter,Arial" font-size="30" font-weight="800" fill="#0f172a">Agentic AI Loop: From Policy Knowledge to Policy Improvement</text>
  <circle cx="500" cy="290" r="168" fill="none" stroke="url(#loopg)" stroke-width="10" stroke-dasharray="16 14"/>
  <text x="422" y="280" font-family="Inter,Arial" font-size="25" font-weight="800" fill="#0f172a">Policy-as-Skill</text>
  <text x="405" y="314" font-family="Inter,Arial" font-size="15" fill="#334155">Reusable, governed, auditable</text>
  <g font-family="Inter,Arial" font-size="15" font-weight="700" fill="#0f172a">
    <rect x="405" y="95" width="190" height="60" rx="18" fill="#dbeafe"/><text x="436" y="130">Policy Knowledge</text>
    <rect x="655" y="165" width="190" height="60" rx="18" fill="#e0e7ff"/><text x="690" y="200">Trusted Retrieval</text>
    <rect x="705" y="340" width="190" height="60" rx="18" fill="#ede9fe"/><text x="764" y="375">Reasoning</text>
    <rect x="405" y="445" width="190" height="60" rx="18" fill="#dcfce7"/><text x="430" y="480">Tool/Action Support</text>
    <rect x="105" y="340" width="190" height="60" rx="18" fill="#fef9c3"/><text x="151" y="375">Human Review</text>
    <rect x="155" y="165" width="190" height="60" rx="18" fill="#fee2e2"/><text x="195" y="200">Audit Trail</text>
  </g>
  <g stroke="#2563eb" stroke-width="4" fill="none" marker-end="url(#arrow)">
    <path d="M590 135 C665 130 708 145 742 165"/>
    <path d="M818 225 C862 270 858 320 820 340"/>
    <path d="M725 395 C665 438 610 454 592 462"/>
    <path d="M405 475 C320 455 270 420 235 395"/>
    <path d="M190 340 C145 290 160 240 205 220"/>
    <path d="M340 165 C378 138 405 130 405 130"/>
  </g>
  <text x="360" y="532" font-family="Inter,Arial" font-size="15" fill="#475569">Evaluation closes the loop: failures become policy, retrieval, skill, and governance improvements.</text>
</svg>
"""
