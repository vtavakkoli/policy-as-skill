# Policy-as-Skill

**Policy-as-Skill** is a research prototype for governed Agentic AI in regulated enterprises and public-sector workflows.

It extends the idea of **Policy-as-Prompt** by treating policies, regulations, and internal guidelines as reusable, governed, versioned, and auditable agent capabilities rather than only as text inserted into prompts.

> Paper direction: **Policy-as-Skill: Governed Agentic AI for Traceable Policy-Aware Decision Support**

## What this prototype implements

The platform runs a local benchmark and compares several methods:

1. **Direct LLM** — no retrieval, weak baseline.
2. **Keyword Search** — lexical retrieval and extractive decision.
3. **Standard RAG** — retrieved policy chunks + LLM answer.
4. **Hybrid RAG** — BM25-like retrieval plus keyword/tag scoring.
5. **Hybrid RAG + Reranker** — stronger retrieval baseline with deterministic second-stage reranking.
6. **Policy-as-Prompt** — policy evidence directly encoded in prompt.
7. **Structured Policy-as-Prompt** — policy artifacts encoded as structured JSON.
8. **Commercial LLM** — optional commercial model baseline without local policy retrieval.
9. **Commercial LLM + RAG** — optional commercial model with reranked policy evidence.
10. **Policy-as-Skill No Audit** — ablation with skill selection but without strict validation.
11. **Policy-as-Skill** — full method with skill registry, scoped retrieval, required evidence checks, citation validation, human-review routing, policy hashes, and audit trace.

The run generates:

```text
result/report.html
result/metrics.csv
result/metrics.json
result/traces.jsonl
result/failures.json
result/statistics.csv
result/statistics.json
result/manifest.json
```

## Core distinction

### Policy-as-Prompt

Policies are encoded directly as prompt text.

### Policy-as-Skill

Policies become reusable governed agent capabilities:

```text
Policy knowledge
→ trusted retrieval
→ skill selection
→ contextual reasoning
→ validation
→ human review routing
→ audit trail
→ evaluation
→ policy improvement
```

Each `PolicySkill` includes:

```text
name
version
retrieval scope
risk level
allowed actions
human-review triggers
required evidence tags
decision schema
audit fields
failure policy
prompt template
```

## Repository structure

```text
policy-as-skill/
  data/
    policies/                 policy documents across public-sector, privacy, cloud, cyber, and HR domains
    tasks/                    curated 200-task benchmark
    annotations/              expert labels and manual citation-faithfulness templates
    case_studies/             public-sector / enterprise case-study template
  src/policy_as_skill/
    main.py                   main experiment runner
    agents.py                 method implementations
    skills.py                 PolicySkill registry
    retrieval.py              BM25-like, keyword, hybrid, and reranked retrieval
    evaluators.py             research metrics with optional manual annotation override
    stats.py                  paired bootstrap CIs and exact sign tests
    annotations.py            manual citation-faithfulness annotation loader
    report_generator.py       self-contained HTML report
    ollama_client.py          host Ollama REST client
    commercial_llm_client.py  optional OpenAI-compatible / Anthropic-compatible adapter
  result/                     generated outputs
  tests/                      lightweight tests
```

## Prerequisites

Install Ollama on the host machine. Ollama is **not** started inside Docker.

Pull the test model:

```bash
ollama pull gemma4:e2b
```

Start Ollama on the host:

```bash
ollama serve
```

The Docker container calls host Ollama using:

```text
http://host.docker.internal:11434
```

Linux compatibility is enabled in `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## Run

```bash
docker compose up --build
```

Open:

```text
result/report.html
```

## Fast offline smoke test

To run without Ollama and use deterministic offline outputs:

```bash
OLLAMA_ENABLED=false MAX_TASKS=8 docker compose up --build
```

This is useful for testing Docker, metrics, traces, and report generation.

## Full 200-task research run

The repository now contains a static balanced expert-labeled benchmark in `data/tasks/benchmark_tasks.jsonl` with 200 tasks:

```text
Policy question answering: 50
Compliance checking:      50
Risk classification:      50
Conflict detection:       50
Total:                   200
```

The default Docker configuration sets `MAX_TASKS=0`, so the normal command evaluates all 200 tasks:

```bash
docker compose up --build
```

`MAX_TASKS=0` means evaluate all curated tasks. Use `MAX_TASKS=8` only for a small smoke test.

For local debugging without Docker:

```bash
PYTHONPATH=src OLLAMA_ENABLED=false MAX_TASKS=0 python -m policy_as_skill.main
```


## Expert labels and policy domains

Each benchmark task contains metadata with a policy domain, expert-curated decision label, expected human-review route, expected policy references, and a short rationale. The same labels are exported in:

```text
data/annotations/expert_labels.jsonl
```

The benchmark and policy corpus now cover multiple real-world-inspired regulated domains:

```text
data_protection_privacy
public_sector_case_study
procurement_cloud
cybersecurity_access_control
hr_workplace_ai
```

The included policy documents are de-identified and synthetic/real-world-inspired. Do not claim completed external validity until a real public-sector or enterprise partner validates the case-study template in `data/case_studies/public_sector_case_study_template.json`.

## Stronger baselines

The default method list now includes a deterministic reranker baseline and optional commercial-LLM baselines. The commercial baselines run in deterministic fallback mode unless you explicitly enable them and provide an API key.

OpenAI-compatible example:

```bash
COMMERCIAL_LLM_ENABLED=true \
COMMERCIAL_LLM_PROVIDER=openai \
COMMERCIAL_LLM_MODEL=gpt-4o-mini \
COMMERCIAL_LLM_API_KEY=your_key \
docker compose up --build
```

Anthropic-compatible example:

```bash
COMMERCIAL_LLM_ENABLED=true \
COMMERCIAL_LLM_PROVIDER=anthropic \
COMMERCIAL_LLM_MODEL=claude-3-5-sonnet-latest \
COMMERCIAL_LLM_API_KEY=your_key \
docker compose up --build
```

## Statistical tests and confidence intervals

Every run writes paired statistics comparing `Policy-as-Skill` against each baseline on per-task normalized score:

```text
result/statistics.csv
result/statistics.json
```

The statistics include paired mean difference, bootstrap 95% confidence interval, target wins, baseline wins, ties, and exact two-sided sign-test p-value. Configure iterations with `BOOTSTRAP_ITERATIONS`.

## Manual citation-faithfulness annotation

A human-annotation template and protocol are included:

```text
data/annotations/citation_faithfulness_annotation_template.csv
docs/manual_citation_faithfulness_protocol.md
```

After manual annotation, save the adjudicated file as:

```text
data/annotations/manual_citation_faithfulness.csv
```

The evaluator automatically uses this file when populated and records `manual_annotation_count` per metric row.

## Metrics

The platform evaluates more than answer quality:

| Metric | Meaning |
|---|---|
| `answer_similarity` | Token overlap with expected answer; used only after generation. |
| `citation_coverage` | Whether the answer contains citations. |
| `citation_precision` | Whether cited policy IDs were actually retrieved. |
| `policy_ref_recall` | Whether expected policy references appear in answer/evidence/citations. |
| `evidence_faithfulness` | Whether answer tokens are supported by retrieved evidence. |
| `unsupported_claim_rate` | Approximate share of answer content not grounded in evidence. |
| `contradiction_rate` | Decision mismatch against expected decision. |
| `decision_accuracy` | One minus contradiction rate. |
| `traceability_score` | Whether evidence, decision, review flag, validation, hashes, and prompt hash exist. |
| `human_review_correctness` | Whether review routing matches expected review need. |
| `audit_completeness` | Completeness of audit fields. |
| `governance_readiness_score` | Weighted governance score using traceability, audit, citation precision, review correctness, and update adaptation. |
| `update_adaptation_score` | Whether current policy-version tasks cite/use the expected version. |
| `overall_score` | Weighted research score across correctness, citations, faithfulness, decision, traceability, and governance. |

## No ground-truth leakage

The platform is designed so that `expected_answer` is used **only** by `evaluators.py`.

`agents.py` never uses expected answers during generation. If Ollama is unavailable, it creates deterministic offline outputs from retrieved evidence only.
