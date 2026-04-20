# RL Dense-to-Real Training Protocol (Scenario 1)

## Goal
Use a denser trip-sampling distribution for pretraining, then switch to the target evaluation distribution for fine-tuning and final evaluation.

## Fixed evaluation distribution (`D_eval`)
- `omega_global_scale = 1.0`
- `omega_window_start_slot = 0`
- `omega_window_end_slot = 7`
- `omega_window_scale = 2.0`
- `omega_od_target_scale = 5.0`

## Prerequisites
1. OR input exists at `data/input/u_odit.csv`
2. GPU available (`torch.cuda.is_available() == True`)

## Step 0: pick `D_train_dense` candidate
Run short probes (200 episodes each) and compare `transitions` in `train_episode_metrics.csv`:

```powershell
cd D:\TUD\Thesis_project

# candidate A: 2,3,5
python -m rl.train --episodes 200 --seed-start 12000 --device cuda --output-dir results\rl_dense_probe_a --or-input-path data/input/u_odit.csv --omega-global-scale 2.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 3.0 --omega-od-target-scale 5.0

# candidate B: 2,5,5
python -m rl.train --episodes 200 --seed-start 13000 --device cuda --output-dir results\rl_dense_probe_b --or-input-path data/input/u_odit.csv --omega-global-scale 2.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 5.0 --omega-od-target-scale 5.0

# candidate C: 3,3,5
python -m rl.train --episodes 200 --seed-start 14000 --device cuda --output-dir results\rl_dense_probe_c --or-input-path data/input/u_odit.csv --omega-global-scale 3.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 3.0 --omega-od-target-scale 5.0
```

Pick the one with stable and higher transitions/episode (target >= 3-5).

## Step 1: Dense pretrain
Example with chosen dense profile `2,5,5`:

```powershell
cd D:\TUD\Thesis_project
python -m rl.train --episodes 3000 --seed-start 2000 --device cuda --checkpoint-every 250 --output-dir results\rl_s1_pretrain_dense --or-input-path data/input/u_odit.csv --omega-global-scale 2.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 5.0 --omega-od-target-scale 5.0
```

## Step 2: Real-distribution fine-tune (resume)
Switch to `D_eval` and continue from dense checkpoint:

```powershell
cd D:\TUD\Thesis_project
python -m rl.train --episodes 2000 --seed-start 6000 --device cuda --checkpoint-every 200 --output-dir results\rl_s1_finetune_real --resume-checkpoint results\rl_s1_pretrain_dense\checkpoints\ddqn_final.pt --lr 0.0003 --epsilon-start 0.2 --epsilon-end 0.02 --epsilon-decay-steps 15000 --or-input-path data/input/u_odit.csv --omega-global-scale 1.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 2.0 --omega-od-target-scale 5.0
```

## Step 3: Final evaluation on `D_eval`

```powershell
cd D:\TUD\Thesis_project
python -m rl.evaluate --checkpoint results\rl_s1_finetune_real\checkpoints\ddqn_final.pt --episodes 500 --seed-start 4000 --device cuda --output-dir results\rl_s1_eval_real --or-input-path data/input/u_odit.csv --omega-global-scale 1.0 --omega-window-start-slot 0 --omega-window-end-slot 7 --omega-window-scale 2.0 --omega-od-target-scale 5.0
```

## Audit checklist
- `train_meta.json` has `resumed_from` in fine-tune run.
- `eval_summary.csv` includes omega profile fields matching `D_eval`.
- Compare OR-only vs RL in `eval_or_vs_rl.csv`.
