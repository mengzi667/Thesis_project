# Scenario1 Fast-Screen and Formal Run Commands

This file documents the two-stage execution process for the anti-collapse setup.

## Stage A: Fast screen (parallel 3 runs)

Use 3 terminals with different `--beta-a` values (`0.1`, `0.2`, `0.3`).

Common settings:
- train episodes: `180`
- eval episodes: `100`
- hidden dim: `64`
- checkpoint every: `200`
- early stop: `ep=120`, `mean_offers < 0.05`

Example (beta_a = 0.2):

```powershell
python -m rl.train `
  --episodes 180 `
  --output-dir results/rl_s1_stageA_ba02 `
  --device cuda `
  --checkpoint-every 200 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.2 `
  --beta-c 0.3 `
  --beta-r 0.02 `
  --hidden-dim 64 `
  --transition-dump-every 20 `
  --early-stop-episode 120 `
  --early-stop-min-offers 0.05 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour
```

```powershell
python -m rl.evaluate `
  --checkpoint results/rl_s1_stageA_ba02/checkpoints/ddqn_final.pt `
  --episodes 100 `
  --output-dir results/rl_s1_stageA_ba02_eval `
  --device cuda `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a 0.2 `
  --beta-c 0.3 `
  --beta-r 0.02 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour
```

## Stage B: Formal run (winner only)

After selecting the winner from Stage A:

```powershell
python -m rl.train `
  --episodes 800 `
  --output-dir results/rl_s1_stageB_winner `
  --device cuda `
  --checkpoint-every 200 `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a <winner_beta_a> `
  --beta-c 0.3 `
  --beta-r 0.02 `
  --hidden-dim 64 `
  --transition-dump-every 20 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour
```

```powershell
python -m rl.evaluate `
  --checkpoint results/rl_s1_stageB_winner/checkpoints/ddqn_final.pt `
  --episodes 400 `
  --output-dir results/rl_s1_stageB_winner_eval `
  --device cuda `
  --w-l 0.5 `
  --w-e 0.5 `
  --beta-a <winner_beta_a> `
  --beta-c 0.3 `
  --beta-r 0.02 `
  --omega-arrival-dist nb2 `
  --omega-nb-phi-mode by_hour
```

## Pass criteria

- `rl_mean_offers > 0`
- `rl_mean_reward > -0.02`
- Prefer higher `rl_mean_service_rate` and lower `rl_mean_realized_loss` when scores are close.
