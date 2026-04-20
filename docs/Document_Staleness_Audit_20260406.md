# Document Staleness Audit (2026-04-06)

This file lists potentially outdated docs. No files are deleted.

## Summary
- Audited folder: `docs/`
- Scope: consistency with current code/runtime behavior

## 1) `docs/Project Requirement.md` — **Partially outdated**
Main mismatches:
1. Early sections still state RL/training pipeline are out of scope, but RL module is now implemented and runnable.
2. Development-task sections still describe pre-RL stage as active baseline.
3. Current runtime overrides and dense-to-real RL workflow are not reflected.
4. Minor text artifact exists in user-choice section (`offer vs <control-char>ase`).

Recommendation:
- Keep as historical requirement baseline, but add a clear “Current Implementation Delta” section.

## 2) `docs/RL_SPEC.md` — **Outdated defaults / partially mismatched**
Main mismatches:
1. Hyperparameter defaults in doc differ from current code (e.g., batch/warmup/target update).
2. Doc states evaluation should include aggregated EDL statistics; current `eval_summary.csv` does not yet output explicit EDL aggregates.
3. Runtime controls added recently (resume, omega profile overrides) are not fully captured.

Recommendation:
- Update this spec to match current executable interfaces and current metric outputs.

## 3) `docs/RL_RUNBOOK.md` — **Outdated as primary runbook**
Main mismatches:
1. Uses old command patterns and output dirs.
2. Does not present dense-pretrain -> real-finetune workflow.
3. Does not document new CLI overrides (`resume-checkpoint`, omega profile overrides, `or-input-path`).

Recommendation:
- Keep for historical reference or merge into the dense-to-real protocol.

## 4) `docs/RL_DENSE_TO_REAL_PROTOCOL.md` — **Current / usable**
Status:
- Aligned with current code interfaces and strategy direction.

Note:
- Episode counts are strategy-scale examples; may be reduced for faster debugging runs.

## 5) `docs/thesis_rewrite_guidelines.md` — **Not stale for code ops**
Status:
- Writing guideline document, not an operations spec. No code-runtime mismatch concern.

---

## Suggested action order (without deleting files)
1. Add “Implementation Delta” to `Project Requirement.md`.
2. Refresh `RL_SPEC.md` with current defaults and actual evaluation outputs.
3. Mark `RL_RUNBOOK.md` as legacy or redirect to `RL_DENSE_TO_REAL_PROTOCOL.md`.
