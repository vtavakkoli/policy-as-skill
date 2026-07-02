# Policy-as-Skill

**Policy-as-Skill: Governed Agentic AI for Traceable Policy-Aware Decision Support** is a research prototype for transforming regulations, public-sector rules, and internal guidelines into reusable, governed, auditable agentic AI skills.

The prototype compares four methods:

1. **Keyword Search Baseline** — simple lexical policy matching.
2. **Standard RAG** — TF-IDF retrieval plus LLM generation.
3. **Policy-as-Prompt** — policy context encoded directly into structured prompts.
4. **Policy-as-Skill** — policies represented as governed, reusable, auditable skills combining trusted retrieval, task planning, reasoning, human-review routing, and evaluation.

This distinction is central to the paper idea: **Policy-as-Prompt** treats policy as text inside a prompt, while **Policy-as-Skill** treats policy as a reusable, versionable agent capability with traceability and governance metadata.

## Architecture

Policy Repository → Retrieval → Skill Registry → Agent Planner → Ollama Reasoning → Human Review → Audit Trail → Evaluation → Report

The complete experiment runs locally with Docker Compose. **Ollama runs outside Docker on the host machine**, and the container calls it through `http://host.docker.internal:11434`.

## Prerequisites

- Docker and Docker Compose
- Python is only required if running tests outside Docker
- Ollama installed on the host machine

## Install and run Ollama

Install Ollama from <https://ollama.com/download>, then pull and serve the required model on the host:

```bash
ollama pull gemma4:e2b
ollama serve
```

Do not run Ollama inside Docker.

## Configuration

Copy the example environment file if desired:

```bash
cp .env.example .env
```

Default values:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gemma4:e2b
```

## Run the experiment

```bash
docker compose up --build
```

or:

```bash
docker compose run --rm policy-as-skill
```

The main entry point is `src/policy_as_skill/main.py`. All outputs are written to `result/`, including the final report:

```text
result/report.html
```

## Outputs

- `result/report.html` — polished self-contained paper-oriented HTML report.
- `result/metrics.csv` — per-task and per-method metrics.
- `result/metrics.json` — aggregate and detailed metrics.
- `result/traces.jsonl` — prompts, outputs, retrieved evidence, decisions, and audit traces.
- `result/failures.json` — failure cases grouped by method.

## Metrics

- **answer_similarity**: token-overlap similarity against expected answer.
- **citation_coverage**: whether generated evidence/citations include source identifiers.
- **policy_ref_recall**: fraction of expected policy references found in evidence or answer text.
- **traceability_score**: transparent score for logging source, reasoning, decision, and review flag.
- **human_review_correctness**: agreement with expected human-review routing heuristics.
- **latency_seconds**: wall-clock execution time per method/task.
- **overall_score**: weighted average of quality, grounding, traceability, and routing metrics.

## Research framing

This repository is a research prototype for a future IEEE/ACM paper. It demonstrates how policy-aware decision support can move beyond prompt engineering toward governed agentic skills that are reusable, auditable, and measurable.
