# Executable Policy-as-Skill contract

The paper defines a policy skill as:

`S = <n, v, R, E, D, H, A, F, P, C>`

The research implementation now maps every component to executable runtime behavior rather than leaving the tuple as descriptive notation.

| Symbol | Meaning | Runtime implementation |
|---|---|---|
| `n` | skill name | `PolicySkill.name`; recorded as `selected_skill` |
| `v` | skill version | `PolicySkill.version`; recorded as `policy_skill_version` |
| `R` | retrieval scope | `PolicySkill.retrieval_scope`; passed to scoped retrieval/reranking |
| `E` | required evidence | `PolicySkill.required_evidence_tags`; checked against retrieved chunk tags |
| `D` | decision schema | `PolicySkill.decision_schema` plus canonical decision vocabulary |
| `H` | human-review policy | `human_review_required` and `human_review_triggers`; evaluated by the governed workflow/controller |
| `A` | audit fields | `PolicySkill.audit_fields`; audit-enabled variants generate hashes, validation and trace records |
| `F` | failure policy | `PolicySkill.failure_policy`; represented in the executable contract and used as governed failure semantics |
| `P` | prompt template | `PolicySkill.prompt_template`; included in the model request and prompt hash |
| `C` | context contract | required runtime inputs, evidence scoping, citation resolution and required output schema |

`src/policy_as_skill/skill_contract.py` constructs the formal object and validates runtime context without access to expected benchmark labels.

For each Policy-as-Skill research trace, `formal_skill_contract` records the complete contract used for the decision. The contract is also appended to the governed model prompt. `validation.formal_components_executed` records the ten formal components and `validation.missing_context_inputs` / `missing_required_evidence_tags` expose contract failures.

This makes the formalization testable. A reviewer can inspect a trace and determine which skill version, retrieval scope, evidence contract, decision schema, review rules, audit fields, failure policy, prompt and context contract were active for that case.
