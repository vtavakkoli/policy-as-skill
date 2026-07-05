# Manual Citation-Faithfulness Annotation Protocol

This protocol supports human annotation of whether generated answers are faithfully grounded in cited policy evidence.

## Unit of annotation
Each row evaluates one answer--citation pair for one task and method. Annotators inspect the generated answer span, the cited evidence span, and the expected policy references.

## Labels
- `supported=yes`: the citation directly supports the answer span.
- `supported=partial`: the citation supports part of the claim but misses a required condition, exception, or scope.
- `supported=no`: the citation does not support the answer span.
- `contradiction=yes`: the citation contradicts the answer span or decision.
- `policy_ref_correct=yes/no`: whether the cited policy reference matches the expected policy area.

## Adjudication
For paper-quality experiments, use at least two annotators per sampled row and adjudicate disagreements. Store the final adjudicated CSV as `data/annotations/manual_citation_faithfulness.csv`. The runner automatically uses this file when populated.
