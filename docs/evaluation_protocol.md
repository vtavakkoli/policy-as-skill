# Evaluation protocol

## 1. Evaluation claims

The evaluation distinguishes three levels of evidence:

- **Model-level unseen:** benchmark instances are not used for LLM training, fine-tuning, or parameter adaptation. This is true for the bundled benchmark.
- **System-development-informed:** controller/prompt/error analysis has used the benchmark during development. This is the correct status of the bundled 600-task benchmark.
- **System-level held out:** the complete system is frozen before a separately authored test set is evaluated. This status requires a valid frozen-test manifest.

Do not collapse these statements into the single word `unseen`.

## 2. Primary outcomes

Primary task outcomes are deliberately format-neutral and do not reward the presence of Policy-as-Skill audit metadata:

- exact decision accuracy;
- decision macro-F1;
- human-review precision;
- human-review recall;
- human-review F1;
- human-review exact accuracy;
- confusion matrix.

The legacy graded decision-agreement quantity is retained as `decision_agreement_graded`, not presented as exact accuracy.

## 3. Evidence outcomes

Evidence quality is reported separately:

- citation precision;
- policy-reference recall;
- evidence faithfulness;
- unsupported-claim rate;
- optional human citation-faithfulness annotations.

## 4. Governance outcomes

Governance capability is a separate construct:

- traceability;
- audit completeness;
- governance quality;
- update/version adaptation;
- the historical composite, renamed `governance_readiness_index`.

The governance-readiness index is a secondary diagnostic. It must not be described as decision accuracy.

## 5. Sensitivity analysis

The repository recomputes method rankings over a grid of non-negative weights summing to one for:

`decision quality + evidence quality + governance quality + answer similarity`.

This tests whether a ranking depends on one hand-selected weighting. The repository also recomputes readiness pass rates over multiple thresholds. Results are written to `result/sensitivity.json`.

## 6. Component ablations

Use the same configured LLM for all model-based methods. Policy-as-Skill is decomposed as:

| Variant | Skill-scoped retrieval | Generic controller | Audit/citation validation |
|---|---:|---:|---:|
| Policy-as-Skill Retrieval | yes | no | no |
| Policy-as-Skill + Controller | yes | yes | no |
| Policy-as-Skill + Audit | yes | no | yes |
| Policy-as-Skill | yes | yes | yes |

The research runner records that the historical benchmark-informed phrase controller is not used by these Policy-as-Skill rows.

## 7. Human validation

For paper-quality citation validation, sample a stratified subset after the model run and obtain at least two independent annotations per item. Do not let annotators see the system's automatic faithfulness score. Adjudicate disagreements separately.

Generate a sample with:

```bash
python scripts/sample_for_annotation.py --per-method 20
```

After two annotators complete the CSV, point `MANUAL_CITATION_ANNOTATIONS_PATH` to it and rerun the evaluator. `annotation_agreement.json` reports Cohen's kappa where paired annotations exist.

For decision ground truth, the strongest version of the experiment should likewise use two independent domain-qualified annotators on the frozen test set, followed by adjudication. The bundled labels are currently declared as single-expert-curated reference labels.

## 8. Frozen held-out evaluation

Create a genuinely independent test file, check overlap, then freeze it before running the system:

```bash
python scripts/check_benchmark_overlap.py --test data/tasks/heldout/test.jsonl
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

The run fails if the frozen benchmark hash differs from its manifest or if the manifest does not explicitly state that the test set was finalized without controller-error feedback.

## 9. Reporting

For the paper, use `result/paper_tables.md` as the starting point. Report primary task metrics first, evidence outcomes second, governance outcomes third, and hardware efficiency separately. Include uncertainty intervals and the sensitivity result rather than relying on one aggregate score.
