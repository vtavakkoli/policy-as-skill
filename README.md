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
5. **Policy-as-Prompt** — policy evidence directly encoded in prompt.
6. **Structured Policy-as-Prompt** — policy artifacts encoded as structured JSON.
7. **Policy-as-Skill No Audit** — ablation with skill selection but without strict validation.
8. **Policy-as-Skill** — full method with skill registry, scoped retrieval, required evidence checks, citation validation, human-review routing, policy hashes, and audit trace.

The run generates:

```text
result/report.html
result/metrics.csv
result/metrics.json
result/traces.jsonl
result/failures.json
result/manifest.json
result/benchmark_generated.jsonl
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
    policies/                 synthetic policy documents
    tasks/                    curated seed tasks
  src/policy_as_skill/
    main.py                   main experiment runner
    agents.py                 method implementations
    skills.py                 PolicySkill registry
    retrieval.py              BM25-like, keyword, hybrid retrieval
    evaluators.py             research metrics
    benchmark_generator.py    deterministic synthetic benchmark generator
    report_generator.py       self-contained HTML report
    ollama_client.py          host Ollama REST client
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

## Larger research run

For a paper-style run, increase the benchmark size and run all generated tasks:

```bash
BENCHMARK_SIZE=300 MAX_TASKS=0 docker compose up --build
```

`MAX_TASKS=0` means evaluate all generated tasks.

For local debugging without Docker:

```bash
PYTHONPATH=src OLLAMA_ENABLED=false MAX_TASKS=8 python -m policy_as_skill.main
```

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

## Research extensions still recommended before A* submission

This repository is now a stronger research platform, but an A* paper still needs:

- expert-labeled policy tasks,
- multiple real policy domains,
- stronger baselines with rerankers and commercial LLMs,
- statistical tests and confidence intervals,
- manual citation-faithfulness annotation,
- external validity through a public-sector or enterprise case study.

## Scientific framing

The central claim is:

> In regulated enterprises, AI quality is not only answer accuracy. A policy-aware agent must be traceable, reviewable, version-aware, evidence-grounded, and auditable.

Policy-as-Skill makes these properties first-class measurable outputs.
