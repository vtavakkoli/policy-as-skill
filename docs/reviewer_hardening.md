# Reviewer-hardening changes

This document maps the main methodological concerns raised during peer review to concrete repository changes. It also distinguishes changes that can be implemented in code from evidence that still has to be collected independently.

## What the repository can now demonstrate

| Reviewer concern | Repository response |
|---|---|
| Composite score may reward Policy-as-Skill-specific audit metadata | Decision accuracy, decision macro-F1, and review-routing precision/recall/F1 are now reported separately from evidence and governance metrics. The old composite is retained only as a secondary `governance_readiness_index`. |
| Metric weights are hand chosen | `sensitivity.json` evaluates rankings over a simplex of decision/evidence/governance/answer-similarity weights, and separately varies readiness thresholds. |
| Controller contribution is unclear | Four explicit ablations isolate skill-scoped retrieval, generic deterministic controller, audit/validation, and the full system. |
| Controller may be benchmark-specific | The research runner does not invoke the historical phrase-refinement controller for Policy-as-Skill. `governance.py` instead extracts applicable normative clauses from retrieved policy evidence and has no task-ID, expected-answer, expected-decision, or expert-label input. |
| The 600 tasks are called unseen even though system error analysis influenced later controller work | The repository now states two different facts: the benchmark was not used to train or fine-tune the LLM, but the bundled 600-task set is system-development-informed. A frozen held-out protocol is provided for system-level generalization claims. |
| Ground truth / expert validation is unclear | The current labels remain explicitly identified as single-expert-curated reference labels. The manual annotation pipeline now supports duplicate independent annotations and computes Cohen's kappa. |
| Citation faithfulness is automatic | A stratified annotation sampler, existing annotation protocol, loader, and inter-annotator agreement calculation support a human-validated citation subset. Absence of human annotations is reported rather than hidden. |
| Only 11 policy documents and corpus provenance unclear | `data/policies/manifest.json` documents every policy file, domain, provenance status, and the lack of external production-corpus validation. |
| Edge deployment is claimed without device evidence | `scripts/edge_benchmark.py` records warm-up/steady-state latency, p95/p99, throughput, process resources, optional NVIDIA GPU counters, and Jetson/tegrastats information. No hardware result is fabricated if the device/counter is unavailable. |
| Need reproducibility | Evaluation provenance, benchmark SHA-256, policy hashes, prompt hashes for audit-enabled variants, model configuration, system profile, traces, metrics, bootstrap confidence intervals, and CI are generated automatically. |

## Four clean Policy-as-Skill ablations

The research runner reports:

1. `Policy-as-Skill Retrieval`: skill selection + scoped retrieval + LLM; no deterministic controller and no audit validation.
2. `Policy-as-Skill + Controller`: adds the generic evidence-driven controller; no audit validation.
3. `Policy-as-Skill + Audit`: adds citation/audit validation to skill-scoped retrieval but no deterministic controller.
4. `Policy-as-Skill`: combines scoped retrieval, generic controller, and audit/validation.

This design allows the paper to attribute improvements to specific components rather than comparing only a full architecture against unrelated baselines.

## Model-level unseen vs system-level held out

The repository never trains or fine-tunes the configured LLM on the benchmark. Therefore the bundled tasks are unseen by the model in the parameter-adaptation sense.

However, historical trace-level analysis of the 600 tasks informed earlier controller refinements. The 600-task file is therefore declared a **development benchmark for system-level claims**. This is not model-training leakage; it is a stricter distinction about system development.

A paper can claim system-level held-out generalization only after a test file is independently finalized, frozen before controller feedback, hashed in a manifest, and run with `FROZEN_EVALUATION=true`. The report exposes both statuses separately.

## What code cannot manufacture

Three reviewer concerns require new empirical evidence rather than software changes:

- **Independent held-out test set.** New scenarios must actually be authored/finalized without controller-error feedback. The repository provides overlap checks and a freeze manifest, but does not pretend the existing 600 tasks satisfy this stronger system-level criterion.
- **Independent human validation.** At least two people must actually annotate a stratified subset. The repository calculates agreement and consumes adjudicated results, but an empty annotation file remains explicitly reported as unavailable validation.
- **Physical edge measurements.** The benchmark script must actually be run on the target Jetson/edge hardware. Desktop or offline runs are not relabeled as Jetson evidence.

These distinctions are intentional: the repository should make unsupported claims difficult rather than make them easy.
