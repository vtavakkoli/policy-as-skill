# Frozen held-out benchmark

Do not copy or paraphrase the bundled 600 development questions into this folder and then call them held out.

A paper-quality system-level test set should be created or finalized **after** the controller/prompt implementation is frozen, without using error feedback from these test instances. This is separate from model training: the repository does not train or fine-tune the LLM on either split.

Recommended procedure:

1. Freeze the controller/prompt commit.
2. Create independently authored scenario families with different wording and, where possible, different policy combinations from the development set.
3. Obtain at least two independent labels for the evaluation subset and adjudicate disagreements.
4. Run `python scripts/check_benchmark_overlap.py --test data/tasks/heldout/test.jsonl`.
5. Before executing the controller on the test file, run:
   `python scripts/freeze_test_manifest.py --benchmark data/tasks/heldout/test.jsonl --confirm-no-feedback`.
6. Evaluate with:
   `EVALUATION_SPLIT=test BENCHMARK_PATH=data/tasks/heldout/test.jsonl BENCHMARK_MANIFEST_PATH=data/tasks/heldout/test.manifest.json FROZEN_EVALUATION=true docker compose up --build`.

The generated report will only mark `system_development_unseen=true` when the frozen-test manifest passes validation.
