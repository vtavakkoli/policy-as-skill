# Policy-as-Skill

**Policy-as-Skill (PaS)** is a research prototype for governed agentic AI in regulated enterprise and public-sector workflows. It treats policy as a reusable, versioned, executable, auditable capability rather than only as text injected into a prompt.

> Paper direction: **Policy-as-Skill: Governed Agentic AI for Traceable Policy-Aware Decision Support**

## Important evaluation statement

The repository does **not train or fine-tune the configured LLM on the benchmark**. The bundled 600 tasks are therefore unseen by the LLM in the parameter-training / parameter-adaptation sense.

However, historical trace-level error analysis of those 600 tasks informed earlier controller refinements. For the stronger claim of **system-level held-out generalization**, the bundled 600-task file is now explicitly treated as a **development benchmark**. A separate frozen-test protocol is provided below.

This distinction is deliberate:

```text
model-level unseen                  yes
LLM training/fine-tuning on tasks   no
600-task system-development set     yes
frozen system-level held-out set    requires a separately frozen test file
```

See `docs/evaluation_protocol.md` and `docs/reviewer_hardening.md`.

## Reviewer-hardened research design

The current research runner addresses the main methodological weaknesses of the earlier prototype:

- primary decision metrics are separated from governance/audit metrics;
- exact decision accuracy and macro-F1 are reported rather than calling a graded agreement score “accuracy”;
- human-review precision, recall and F1 are reported separately;
- citation/evidence metrics form a separate evidence block;
- the historical composite is retained only as a secondary **governance-readiness index**;
- a weight-sensitivity analysis tests whether rankings depend on hand-selected composite weights;
- four Policy-as-Skill ablations isolate retrieval, controller, audit, and the full system;
- the Policy-as-Skill research runner bypasses the historical benchmark-informed phrase-refinement controller;
- a generic evidence-driven controller derives decisions from normative clauses in retrieved policy text;
- human citation annotation supports two independent annotators and Cohen's kappa;
- policy-corpus provenance is documented explicitly;
- physical edge benchmarking has a dedicated measurement script;
- a frozen held-out benchmark protocol records hashes and system-development provenance.

## Methods

The default run compares the same configured Ollama backend across model-based methods:

1. `Direct LLM`
2. `LLM`
3. `Keyword Search`
4. `Standard RAG`
5. `Hybrid RAG`
6. `Hybrid RAG + Reranker`
7. `LLM + RAG`
8. `Policy-as-Prompt`
9. `Structured Policy-as-Prompt`
10. `Policy-as-Skill Retrieval`
11. `Policy-as-Skill + Controller`
12. `Policy-as-Skill + Audit`
13. `Policy-as-Skill`

### Clean Policy-as-Skill ablation matrix

| Variant | Skill-scoped retrieval | Generic controller | Audit / citation validation |
|---|---:|---:|---:|
| Policy-as-Skill Retrieval | yes | no | no |
| Policy-as-Skill + Controller | yes | yes | no |
| Policy-as-Skill + Audit | yes | no | yes |
| Policy-as-Skill | yes | yes | yes |

The research trace records `legacy_benchmark_phrase_controller_used=false` for these PaS variants.

## Generic governance controller

`src/policy_as_skill/governance.py` does not accept task IDs, expected labels, expected answers, or expert annotations. It extracts applicable normative clauses from retrieved evidence and reasons over generic policy modalities such as:

```text
must not / prohibited  -> prohibition
must / required        -> mandatory control
may / permitted        -> permission
human review / approval / escalation -> review requirement
conflict / precedence  -> governed resolution or escalation
```

For a compliance scenario that explicitly omits an applicable mandatory control, the controller can block or route the case without matching a benchmark-specific sentence. If no trusted evidence supports a decision, it abstains.

## Benchmark

`data/tasks/benchmark_tasks.jsonl` contains 600 balanced development tasks:

```text
Policy question answering: 150
Compliance checking:       150
Risk classification:       150
Conflict detection:        150
Total:                      600
```

Each task contains a reference decision, expected review route, expected policy references, rationale, domain metadata and difficulty. The current reference labels are described as **single-expert curated**; they are not represented as an independently adjudicated legal gold standard.

The development status is recorded in:

```text
data/tasks/development_manifest.json
```

## Policy corpus provenance

The 11 included policy files are synthetic, de-identified, illustrative, or real-world-inspired research fixtures across privacy, cloud procurement, cybersecurity/access control, HR/workplace AI, model governance, policy update/versioning and public-sector scenarios.

Their status is documented in:

```text
data/policies/manifest.json
```

The repository does not claim that this small corpus establishes production or legal external validity. `eu_ai_act_sample.md` is an illustrative research summary, not an authoritative legal text.

## Run with Ollama

Install and start Ollama on the host, then pull/configure the desired model, for example:

```bash
ollama pull gemma4:e2b
ollama serve
```

Docker reaches host Ollama at `http://host.docker.internal:11434`.

Run the complete development benchmark:

```bash
docker compose up --build
```

Open:

```text
result/report.html
```

For a fast pipeline smoke test without model calls:

```bash
OLLAMA_ENABLED=false MAX_TASKS=8 docker compose up --build
```

Offline mode is for testing code, metrics and report generation. It is **not** a substitute for the paper's LLM experiment.

## Outputs

A run generates research artifacts including:

```text
result/report.html
result/paper_tables.md
result/metrics.csv
result/metrics.json
result/research_summary.json
result/traces.jsonl
result/failures.json
result/statistics.csv
result/statistics.json
result/sensitivity.json
result/annotation_agreement.json
result/system_profile.json
result/manifest.json
```

`report.html` begins with evaluation provenance so a development run cannot silently appear as a frozen held-out run.

## Primary metrics: task performance

These metrics do not reward the presence of PaS-specific audit fields:

- exact decision accuracy;
- decision macro-F1;
- decision confusion matrix;
- human-review precision;
- human-review recall;
- human-review F1;
- human-review exact accuracy.

The historical partial-credit quantity is retained as `decision_agreement_graded` for backward comparison.

## Evidence metrics

Reported separately:

- citation precision;
- policy-reference recall;
- evidence faithfulness;
- unsupported-claim rate;
- optional human citation-faithfulness scores.

## Governance metrics

Reported separately:

- traceability;
- audit completeness;
- governance quality;
- policy-version/update adaptation;
- secondary `governance_readiness_index`.

The governance-readiness index must not be described as decision accuracy.

## Weight and threshold sensitivity

Every run writes `result/sensitivity.json`. The analysis evaluates method rankings over a grid of non-negative weights summing to one for decision, evidence, governance and answer-similarity components. It also recomputes readiness pass rates over several thresholds.

This allows a paper to show whether its conclusion is robust rather than relying only on one hand-chosen weighting.

## Human validation

The existing citation-faithfulness protocol is in:

```text
docs/manual_citation_faithfulness_protocol.md
```

Generate a stratified sample from a completed run:

```bash
python scripts/sample_for_annotation.py --per-method 20
```

Use at least two independent annotators per sampled item and preserve their individual rows. The evaluator calculates paired Cohen's kappa when duplicate independent annotations are present.

For stronger decision ground truth, follow:

```text
docs/expert_ground_truth_protocol.md
```

An empty or single-annotator annotation file is reported as such; the repository does not fabricate a human-validation result.

## Frozen held-out system evaluation

Create a genuinely independent test file after the controller/prompt design is frozen. Do not merely paraphrase development questions.

First check lexical overlap:

```bash
python scripts/check_benchmark_overlap.py \
  --test data/tasks/heldout/test.jsonl
```

Then freeze the test manifest **before running the controller on it**:

```bash
python scripts/freeze_test_manifest.py \
  --benchmark data/tasks/heldout/test.jsonl \
  --confirm-no-feedback
```

Run:

```bash
EVALUATION_SPLIT=test \
BENCHMARK_PATH=data/tasks/heldout/test.jsonl \
BENCHMARK_MANIFEST_PATH=data/tasks/heldout/test.manifest.json \
FROZEN_EVALUATION=true \
docker compose up --build
```

The run validates the benchmark SHA-256 and required provenance fields. Only a valid frozen test is reported as `system_development_unseen=true`.

## Physical edge evaluation

Run this **on the actual target device**:

```bash
python scripts/edge_benchmark.py \
  --method Policy-as-Skill \
  --warmup 5 \
  --tasks 50 \
  --output result/edge_benchmark.json
```

The script records warm-up latency, steady-state mean/median/p95/p99 latency, throughput, process resources, system profile, optional `nvidia-smi` counters and optional Jetson/`tegrastats` information.

See `docs/edge_benchmark_protocol.md`. The software records missing device counters as missing; it does not invent Jetson or power measurements.

## Policy-as-Skill object

A `PolicySkill` contains:

```text
name
version
retrieval scope
risk level
allowed actions
human-review triggers
required evidence tags
evaluation criteria
decision schema
audit fields
failure policy
prompt template
```

This maps the conceptual Policy-as-Skill object to concrete runtime fields rather than treating it only as a formal tuple in the paper.

## Repository structure

```text
policy-as-skill/
  data/
    policies/                 research policy corpus + provenance manifest
    tasks/                    600-task development benchmark + held-out protocol
    annotations/              reference labels and human annotation files
    case_studies/             external-validity case-study template
  docs/
    evaluation_protocol.md
    reviewer_hardening.md
    expert_ground_truth_protocol.md
    edge_benchmark_protocol.md
  scripts/
    check_benchmark_overlap.py
    freeze_test_manifest.py
    sample_for_annotation.py
    edge_benchmark.py
  src/policy_as_skill/
    agents.py                 historical methods/baselines
    governance.py             generic evidence-driven controller
    research_agent.py         clean PaS ablation runner
    research_metrics.py       separated metrics + sensitivity analysis
    research_report.py        reviewer-facing HTML/paper tables
    protocol.py               development/frozen-test provenance
    system_profile.py         hardware/resource capture
    skills.py                 PolicySkill registry
    retrieval.py              retrieval and reranking
    evaluators.py             legacy metrics retained for reproducibility
  tests/
  result/
```

## Reproducibility and CI

The CI runs unit tests on Python 3.11 and 3.12 and performs a small offline end-to-end smoke test that must generate the reviewer-facing artifacts.

For local tests:

```bash
python -m pip install pytest
PYTHONPATH=src pytest -q
```

## Claim boundaries

The repository now makes three limitations explicit rather than hiding them:

1. the current 600-task benchmark is model-unseen but system-development-informed;
2. independent human validation still requires real independent annotators;
3. edge claims require an actual run on the stated physical edge hardware.

Those items cannot be solved by changing labels in code. The repository provides the protocols, checks and reporting needed to collect them correctly.
