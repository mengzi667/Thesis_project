# RL Specification (Scenario 1, Current)

## 1) Scope
- OR remains external and injected via `data/input/u_odit.csv`.
- Scenario 1 only: RL action is binary (`offer` / `no_offer`) at OR-matched events.
- Trip arrivals are generated from `omega_od` with NB2 count sampling and intra-slot timestamp sampling.

---

## 2) Event and Trip Generation
- For each `(o,d,slot)`, sample arrival count:
  - Poisson mode: `N ~ Poisson(mu)`
  - NB2 mode: Gamma-Poisson mixture (`Var = mu + phi*mu^2`)
- For each sampled event, timestamp is sampled as:
  - `t = slot_start + Uniform(0, slot_length)`
- All sampled events are sorted by `request_time` and fed into event-driven simulation.

This explicitly separates:
- **How many** arrivals (NB2/Poisson),
- **When** arrivals occur (Uniform within slot).

---

## 3) State, Transition, Reward

### 3.1 State (implemented)
State contains:
- normalized time-in-episode
- normalized `(o,d,i)`
- zone inventory tuples for `o/d/i`
- EDL features for `o/d/i` at decision time
- trip attributes (`rt_base`, `rt_offer`, `extra_walk`, incentive)
- remaining quota/budget context

### 3.2 Transition logging
Each transition logs:
- `state`, `action`, `next_state`, `done`, `reward`
- `offered`, `accepted`, `reject_flag`
- `delta_edl`, `base_cum_edl`, `actual_cum_edl`
- `realized_loss`, `cost_term`
- decomposed terms: `reward_realized_term`, `reward_edl_term`, `reward_accept_term`

### 3.3 Realized-loss scope
Realized-loss is computed in a decision window with **joint O + D1 + D2 scope**:
- between two consecutive RL decision points,
- count `no_supply` losses whose origin is in `{origin, destination, recommended_dest}`.

### 3.4 Reward (independent primary weights)
\[
r_t = -w_L \tilde{L}^{real}_t + w_E \widetilde{\Delta EDL}_t
      + \beta_a I_{accept} - \beta_c Cost_t - \beta_r I_{reject}
\]

\[
\tilde{L}^{real}_t=\mathrm{clip}\left(\frac{L^{real}_t}{L_{ref}},0,1\right),\quad
\widetilde{\Delta EDL}_t=\mathrm{clip}\left(\frac{\Delta EDL_t}{E_{ref}},-1,1\right)
\]

Notes:
- `L_ref`, `E_ref` are normalization anchors for numeric comparability/stability (not physical constants).
- current cost accounting is accept-based: payout counted only when relocation is accepted.

---

## 4) Current Defaults (from `rl/config.py`)
- `w_l=0.5`, `w_e=0.5`
- `beta_a=0.2`, `beta_c=0.3`, `beta_r=0.02`
- `l_ref=1.0`, `e_ref=1.0`
- `replay_capacity=1000`
- `hidden_dim=64`
- `train_episodes=180`, `eval_episodes=100` (CLI override allowed)

---

## 5) Train / Eval Protocol
- **Training:** fixed scale profile (do not tune scale during training).
- **Evaluation:** scale/supply may be varied for robustness/stress tests.
- Three-policy evaluation in one run:
  - `always_offer`
  - `no_offer`
  - `checkpoint`

Recommended result tags:
- `main_comparable`
- `diagnostic_stress`
- `final_retrained_stress`

---

## 6) CLI Interfaces

### 6.1 Train
`python -m rl.train`
- reward args: `--w-l`, `--w-e`, `--beta-a`, `--beta-c`, `--beta-r`, `--l-ref`, `--e-ref`
- profile args: `--omega-*` family
- continuation args: `--resume-checkpoint`, `--lr`, `--epsilon-*`

### 6.2 Evaluate
`python -m rl.evaluate`
- reward args: `--w-l`, `--w-e`, `--beta-a`, `--beta-c`, `--beta-r`, `--l-ref`, `--e-ref`
- profile args: `--omega-*` family
- result tagging: `--result-tag {main_comparable,diagnostic_stress,final_retrained_stress}`

### 6.3 Reference calibration
`python -m rl.calibrate_refs`
- estimates `L_ref` and `E_ref` from transition samples (quantile-based, default P95)
- writes JSON artifact for reproducible anchor selection.

---

## 7) Key Output Fields
- Main metrics: `service_rate`, `offers`, `accepted`, `mean_reward`
- Signal metrics: `mean_realized_loss`, `mean_delta_edl`
- Decomposition: `reward_realized_term`, `reward_edl_term`, `reward_accept_term`
- Diagnostics (eval summary): `sum_no_supply`, `window_od123_loss`, `mean_transitions`

