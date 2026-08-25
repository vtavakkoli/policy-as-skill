# Expert ground-truth protocol

The bundled benchmark contains single-expert-curated reference labels. That is useful for development, but a stronger paper-quality test should add independent annotation and adjudication.

## Recommended roles

Use at least two independent annotators who are qualified for the policy domain being evaluated. Record the role/qualification category, not personally identifying information, in the study metadata. If one annotator is an author or system developer, disclose that and include at least one annotator who did not build the controller.

## Blind annotation

Annotators should receive the scenario and the applicable policy corpus, but not:

- the Policy-as-Skill prediction;
- baseline predictions;
- the automatic evaluator score;
- the controller trace;
- the other annotator's label.

For each task collect:

- decision: `allowed`, `not_allowed`, `conditional`, `needs_review`, or `unknown`;
- whether human review is required;
- applicable policy references;
- short rationale;
- optional confidence and ambiguity flag.

## Agreement and adjudication

Compute agreement before adjudication. For categorical decision labels use Cohen's kappa for two annotators or an appropriate multi-rater statistic for more annotators. Also report raw agreement because kappa can be sensitive to label prevalence.

Resolve disagreements through a separate adjudication pass. Preserve both original labels and the adjudicated label so the process remains auditable.

## Sampling

For a full frozen test set, label every task. If resources only permit a validation subset, stratify by task type, domain, and difficulty and report the sampling scheme. Do not select only examples where methods disagree after seeing the predictions unless the analysis is explicitly described as an error-analysis study.

## Separation from model training

Human annotation does not imply model training. The configured LLM remains frozen and is not fine-tuned on these labels. Labels are used only to evaluate outputs unless a future experiment explicitly defines a training stage.

## Reporting

The paper should state:

- number of annotators and qualification categories;
- whether annotators were independent from system development;
- number of tasks/items annotated;
- agreement before adjudication;
- adjudication procedure;
- whether the final test set was frozen before controller feedback.
