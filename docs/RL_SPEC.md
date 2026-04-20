# RL Specification (Scenario 1, Current Implementation)

## 1. Scope
This document specifies the currently implemented RL design in the repository.

In scope now:
- OR remains external (injected via `U_odit`), no OR solver internals modified.
- Scenario 1 only: binary decision `offer` vs `no_offer` at OR-matched events.
- DDQN training/evaluation pipeline is implemented and runnable on CPU/GPU.
- Dense-to-real workflow is supported (pretrain on dense sampling profile, fine-tune/evaluate on target profile).

Out of scope now:
- Scenario 2 joint action space.
- Upstream OR redesign.
- Full thesis-level hyperparameter sensitivity completion.

---

## 2. Decision Process and Episode Definition

## 2.1 Decision trigger
RL is queried only when a trip event matches OR opportunity `(o, d, i, t)` from `U_odit`.

## 2.2 Action space
- `a=0`: no offer
- `a=1`: offer (incentive fixed by experiment policy)

## 2.3 Episode
- episode length: 2 hours (`episode_minutes=120`)
- slot length: 15 minutes (`planning_period=15`)
- slots per episode: 8

---

## 3. State, Transition, Reward

## 3.1 State (implemented feature vector)
State includes:
1. time-in-episode (slot normalized)
2. event identity `(o,d,i)` (normalized zone IDs)
3. zone inventory tuples for `o/d/i`
4. EDL features for `o/d/i` at decision time
5. trip/offer attributes (`rt_base`, `rt_offer`, `extra_walk`, incentive)
6. remaining quota/budget context

## 3.2 Transition logging
Each RL transition logs at least:
- `state`, `action`, `next_state`, `done`
- `reward`
- `offered`, `accepted`, `reject_flag`
- `delta_edl`, `base_cum_edl`, `actual_cum_edl`
- `realized_loss`, `cost_term`
- decomposed reward terms (`reward_realized_term`, `reward_edl_term`)

## 3.3 Reward (hybrid)
Implemented formula:

\[
r_t
= \lambda \cdot \left(-\tilde{L}^{real}_t\right)
+ (1-\lambda)\cdot \tilde{\Delta EDL}_t
- \beta_c \cdot Cost_t
- \beta_r \cdot \mathbf{1}_{reject}
\]

where:

\[
\tilde{L}^{real}_t = \mathrm{clip}\left(\frac{L^{real}_t}{L_{ref}}, 0, 1\right),
\quad
\tilde{\Delta EDL}_t = \mathrm{clip}\left(\frac{\Delta EDL_t}{E_{ref}}, -1, 1\right)
\]

Current cost setting:
- cost accounted on accept (`Cost_t = incentive * 1_accept`)

---

## 4. Current Defaults (Aligned to `rl/config.py`)

RL defaults:
- `lr = 1e-3`
- `gamma_rl = 0.99`
- `batch_size = 32`
- `replay_capacity = 50000`
- `target_update_every = 200`
- `warmup_steps = 32`
- `grad_clip = 10.0`
- `epsilon_start = 1.0`
- `epsilon_end = 0.05`
- `epsilon_decay_steps = 15000`
- `train_episodes = 2000`
- `eval_episodes = 200`
- `seed_start_train = 11000`
- `seed_start_eval = 21000`

Reward defaults:
- `reward_lambda = 0.7`
- `beta_c = 1.0`
- `beta_r = 0.1`
- `l_ref = 1.0`
- `e_ref = 1.0`

---

## 5. Omega Sampling Profiles

## 5.1 Target evaluation profile (`D_eval`)
Current target profile used for final comparison:
- `omega_global_scale = 1.0`
- `omega_window_start_slot = 0`
- `omega_window_end_slot = 7`
- `omega_window_scale = 2.0`
- `omega_od_target_scale = 5.0`

## 5.2 Dense pretraining profile (`D_train_dense`)
Chosen by short probe runs, then used for pretraining.
Typical selected profile in latest runs:
- `omega_global_scale = 2.0`
- `omega_window_start_slot = 0`
- `omega_window_end_slot = 7`
- `omega_window_scale = 5.0`
- `omega_od_target_scale = 5.0`

---

## 6. Runtime Interfaces (CLI)

## 6.1 `python -m rl.train`
Supports:
- `--episodes`, `--seed-start`, `--device`, `--output-dir`
- reward args: `--reward-lambda`, `--beta-c`, `--beta-r`, `--l-ref`, `--e-ref`
- fine-tune args: `--resume-checkpoint`, `--lr`, `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-steps`
- input/profile overrides:
  - `--or-input-path`
  - `--omega-global-scale`
  - `--omega-window-start-slot`
  - `--omega-window-end-slot`
  - `--omega-window-scale`
  - `--omega-od-target-scale`

## 6.2 `python -m rl.evaluate`
Supports:
- `--checkpoint`, `--episodes`, `--seed-start`, `--device`, `--output-dir`
- `--or-input-path`
- the same omega profile overrides as training

---

## 7. Training/Evaluation Protocol (Current)

Recommended chain:
1. Ensure OR input exists at `data/input/u_odit.csv`.
2. Probe candidate dense profiles with short runs.
3. Pretrain on selected `D_train_dense`.
4. Resume fine-tune on `D_eval`.
5. Final evaluation on `D_eval` using held-out seeds.

Reference procedure:
- `docs/RL_DENSE_TO_REAL_PROTOCOL.md`

---

## 8. Current Evaluation Outputs and Limitations

## 8.1 Outputs currently summarized in `eval_summary.csv`
- `or_mean_service_rate`, `rl_mean_service_rate`
- `or_mean_offers`, `rl_mean_offers`
- `rl_mean_reward`
- profile metadata (`or_input_path`, omega profile values)

## 8.2 Important limitation
- EDL is already computed and used in reward/transition logs.
- But `eval_summary.csv` currently does not include explicit aggregated EDL columns (e.g., mean `delta_edl`, cumulative EDL gain).

This is a known follow-up task for thesis-facing result reporting.

---

## 9. Acceptance Checklist
1. `data/input/u_odit.csv` exists and is loaded (no synthetic OR fallback).
2. Dense pretrain and real fine-tune both run successfully.
3. Fine-tune run metadata contains `resumed_from` checkpoint.
4. Final evaluation runs on `D_eval` with fixed hold-out seeds.
5. Transition logs are non-empty and include reward decomposition fields.

---

## 10. Version Note
This document is aligned to repository state on 2026-04-06.
If defaults, CLI options, or evaluation schema change, update this file and `docs/RL_DENSE_TO_REAL_PROTOCOL.md` together.
