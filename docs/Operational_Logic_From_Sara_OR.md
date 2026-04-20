# Operational Logic and Workflow (Starting After Sara OR Run)

## 1. Scope and Start Point
This document describes the **current end-to-end operational logic** of the project from the point where Sara's OR model has already finished running and produced outputs.

Start condition:
- Sara OR run completed (e.g., 0-56 slots for one day)
- OR output files exist (`detailed_results_*.xlsx`, `window_results_*.csv`)

Out of scope:
- Re-explaining Sara OR internals
- Scenario 2 implementation

---

## 2. Data and Artifact Flow (Post-OR)

## 2.1 OR output artifacts (from sara_repo)
Typical files:
- `sara_repo/results/RH_hyb/detailed_results_0_56.xlsx`
- `sara_repo/results/RH_hyb/window_results_0_56.csv`

The simulator does not use the OR objective internals directly. It only consumes translated relocation decisions.

## 2.2 Translation to simulator input `U_odit`
Translator:
- `or_model/sara_adapter.py::convert_sara_output_to_uodit(...)`

Output schema:
- `origin`
- `original_dest`
- `recommended_dest`
- `time_slot`
- `incentive_amount`
- `quota`

Runtime input path expected by simulator:
- `data/input/u_odit.csv`

Notes:
- This is the operationally critical file. If missing, simulator may fall back to synthetic OR placeholders.
- Current setup ensures this file exists and is loaded.

---

## 3. Simulator Runtime Pipeline (Current)

## 3.1 High-level runtime graph
1. Load Sara-aligned spatial system and initial fleet state
2. Build trip generator (currently `TRIP_SOURCE=omega_od`)
3. Load OR interface from `data/input/u_odit.csv`
4. For each trip event:
- Check OR opportunity `(o,d,t)`
- Layer-1 user participation (`ride/opt_out`)
- Layer-2 acceptance (`offer/base`) if applicable
- Execute trip and update fleet
- Apply battery Markov transition
- Update metrics and (if RL enabled) append transition

## 3.2 Main modules used
- `main.py` (wiring + runtime overrides)
- `simulation/simulation_engine.py` (event loop)
- `simulation/user_choice_model.py` (two-layer behavior)
- `simulation/battery_transition.py` + `simulation/edl_markov.py`
- `or_model/or_interface.py`
- `rl/*` (DDQN training/evaluation)

---

## 4. Key Design Choices (Current)

## 4.1 Spatial and fleet
- Sara H3 station map is source of zones
- Zone IDs follow row order (1..N)
- Initial inventory can be CSV-driven or uniform by battery categories

## 4.2 Trip demand source
Current default:
- `TRIP_SOURCE = omega_od`

Meaning:
- Request generation comes from OD-slot expected values built from `30sep-omega_h.csv`
- Poisson sampling is applied at runtime per `(o,d,slot)`

## 4.3 Two-layer user behavior
Layer 1:
- participation `ride` vs `opt_out`
- mode switch: `FIRST_LAYER_MODE in {aggregated_prob, realtime_choice}`

Layer 2:
- conditional acceptance `offer` vs `base`
- mode switch: `RELOCATION_ACCEPTANCE_MODE in {deterministic, stochastic}`

Scooter assignment rule (Scenario 1):
- remains system rule (highest-battery-first), not user scooter-level choice.

## 4.4 OR integration
- OR is external input only
- Decision lookup key is effectively `(origin, original_dest, time_slot)`
- Quota consumption policy configurable via `OR_QUOTA_CONSUME_POLICY`
- Incentive can be force-fixed to 1 EUR

## 4.5 Battery model
- CSV-conditioned Markov transitions
- Context by weekend/hour
- fallback hierarchy when data sparse
- policy for `high->inactive`: currently `strict-paper`

---

## 5. Current Switches and Their Meaning

## 5.1 Core simulation (`config.py`)
- `TRIP_SOURCE`: `omega_od | sara_profile | replay | poisson`
- `OR_INPUT_PATH`: path to `u_odit.csv`
- `OR_QUOTA_CONSUME_POLICY`: `consume_on_accept | consume_on_offer`
- `OR_FORCE_FIXED_INCENTIVE` + `OR_FIXED_INCENTIVE_EUR`
- `FIRST_LAYER_MODE`: `aggregated_prob | realtime_choice`
- `RELOCATION_ACCEPTANCE_MODE`: `deterministic | stochastic`
- `BATTERY_HIGH_TO_INACTIVE_POLICY`: `strict-paper | strict-data`

## 5.2 omega sampling profile switches (important)
These control trip-density shaping:
- `OMEGA_GLOBAL_SCALE`
- `OMEGA_WINDOW_START_SLOT`
- `OMEGA_WINDOW_END_SLOT`
- `OMEGA_WINDOW_SCALE`
- `OMEGA_OD_TARGET_SCALE`

Current target evaluation profile (`D_eval`):
- `global=1.0`, `window=0-7`, `window_scale=2.0`, `od_target_scale=5.0`

## 5.3 RL runtime overrides (CLI)
`rl.train` and `rl.evaluate` now support:
- `--or-input-path`
- `--omega-global-scale`
- `--omega-window-start-slot`
- `--omega-window-end-slot`
- `--omega-window-scale`
- `--omega-od-target-scale`

`rl.train` additionally supports:
- `--resume-checkpoint`
- `--lr`, `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-steps`
- `--seed-start`

---

## 6. RL Logic and Current Strategy

## 6.1 Action scope (Scenario 1)
- RL acts only on matched OR opportunities
- binary action: `offer / no_offer`

## 6.2 Reward
Hybrid reward combines:
- realized loss term
- delta EDL term
- cost term
- reject penalty

EDL computation path:
- `simulation/edl_markov.py` used by engine
- transition logs include `delta_edl`, `base_cum_edl`, `actual_cum_edl`

## 6.3 Dense-to-real training strategy
Current recommended workflow:
1. Probe candidate dense profiles (`D_train_dense`)
2. Pretrain on selected dense profile
3. Resume fine-tune on `D_eval`
4. Final evaluation on `D_eval`

Reason:
- direct training on sparse `D_eval` gives weak learning signal

---

## 7. What Was Actually Run in the Latest Full Chain

Artifacts:
- Probe summary: `experiments/pre_rl_20260404_021356/rl_setup/dense_probe_summary.csv`
- Full-run summary: `experiments/pre_rl_20260404_021356/rl_setup/dense_to_real_fullrun_20260405_summary.md`

Selected dense profile in latest run:
- `D_train_dense = (global=2.0, window_scale=5.0, od_target_scale=5.0)`

Then:
- pretrain (dense)
- fine-tune (real)
- eval (real)

---

## 8. Current Evaluation Outputs (What is and is not reported)

Currently reported in RL evaluation:
- service rates (OR vs RL)
- offer counts (OR vs RL)
- accepted counts (episode-level file)
- RL mean reward

Currently **not** explicitly summarized in `eval_summary.csv`:
- aggregated EDL metrics (`mean_delta_edl`, cumulative EDL improvements)

Important nuance:
- EDL is already used in reward computation and transition logs,
- but final eval report still lacks explicit EDL aggregation columns.

---

## 9. Known Gaps / Risks
1. Sparse matched opportunities in real distribution still limit learning speed.
2. Evaluation report needs explicit EDL summary columns for thesis-facing analysis.
3. Some historical docs still describe pre-RL-only status and outdated defaults.

---

## 10. Minimal Reproducible Command Skeleton

1. Ensure OR input exists:
- `data/input/u_odit.csv`

2. Dense pretrain (example):
- `python -m rl.train ... --omega-global-scale 2 --omega-window-scale 5 --omega-od-target-scale 5`

3. Real fine-tune with resume:
- `python -m rl.train ... --resume-checkpoint <pre_ckpt> --omega-global-scale 1 --omega-window-scale 2 --omega-od-target-scale 5`

4. Real final eval:
- `python -m rl.evaluate ... --omega-global-scale 1 --omega-window-scale 2 --omega-od-target-scale 5`

For full concrete commands, use:
- `docs/RL_DENSE_TO_REAL_PROTOCOL.md`

---

## 11. Document Version
- Generated against current repository state on 2026-04-06.
- If switches or output schema change, update this file first.
