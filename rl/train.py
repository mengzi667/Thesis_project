from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rl.agent import DDQNAgent
from rl.config import RLConfig
from rl.replay_buffer import ReplayBuffer
from rl.trip_report import write_trip_run_report
from rl.trainer import (
    EpsilonPolicy,
    dump_training_meta,
    epsilon_by_step,
    run_episode,
    transitions_to_replay,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDQN for Scenario 1")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--w-l", type=float, default=None)
    p.add_argument("--w-e", type=float, default=None)
    p.add_argument("--beta-a", type=float, default=None)
    p.add_argument("--beta-c", type=float, default=None)
    p.add_argument("--beta-r", type=float, default=None)
    p.add_argument("--l-ref", type=float, default=None)
    p.add_argument("--e-ref", type=float, default=None)
    p.add_argument("--seed-start", type=int, default=None, help="Start seed for training episodes")
    p.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from checkpoint")

    # fine-tune friendly overrides
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--transition-dump-every", type=int, default=None)
    p.add_argument("--early-stop-episode", type=int, default=None)
    p.add_argument("--early-stop-min-offers", type=float, default=None)

    # optional OR input path override
    p.add_argument("--or-input-path", type=str, default=None)

    # omega sampling profile override
    p.add_argument("--omega-global-scale", type=float, default=None)
    p.add_argument("--omega-window-start-slot", type=int, default=None)
    p.add_argument("--omega-window-end-slot", type=int, default=None)
    p.add_argument("--omega-window-scale", type=float, default=None)
    p.add_argument("--omega-od-target-scale", type=float, default=None)
    p.add_argument("--omega-arrival-dist", type=str, choices=["poisson", "nb2"], default=None)
    p.add_argument("--omega-nb-phi-mode", type=str, choices=["global", "by_hour", "by_hour_weektype"], default=None)
    p.add_argument("--omega-nb-phi-global", type=float, default=None)
    p.add_argument("--omega-nb-phi-csv", type=str, default=None)
    p.add_argument("--omega-nb-phi-min", type=float, default=None)
    p.add_argument("--omega-nb-phi-max", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RLConfig()

    if args.episodes is not None:
        cfg.train_episodes = int(args.episodes)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.w_l is not None:
        cfg.w_l = float(args.w_l)
    if args.w_e is not None:
        cfg.w_e = float(args.w_e)
    if args.beta_a is not None:
        cfg.beta_a = float(args.beta_a)
    if args.beta_c is not None:
        cfg.beta_c = float(args.beta_c)
    if args.beta_r is not None:
        cfg.beta_r = float(args.beta_r)
    if args.l_ref is not None:
        cfg.l_ref = float(args.l_ref)
    if args.e_ref is not None:
        cfg.e_ref = float(args.e_ref)
    if args.seed_start is not None:
        cfg.seed_start_train = int(args.seed_start)

    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.hidden_dim is not None:
        cfg.hidden_dim = int(args.hidden_dim)
    if args.epsilon_start is not None:
        cfg.epsilon_start = float(args.epsilon_start)
    if args.epsilon_end is not None:
        cfg.epsilon_end = float(args.epsilon_end)
    if args.epsilon_decay_steps is not None:
        cfg.epsilon_decay_steps = int(args.epsilon_decay_steps)
    if args.transition_dump_every is not None:
        cfg.transition_dump_every = max(0, int(args.transition_dump_every))
    if args.early_stop_episode is not None:
        cfg.early_stop_episode = int(args.early_stop_episode)
    if args.early_stop_min_offers is not None:
        cfg.early_stop_min_offers = float(args.early_stop_min_offers)

    if args.or_input_path is not None:
        cfg.or_input_path = str(args.or_input_path)

    if args.omega_global_scale is not None:
        cfg.omega_global_scale = float(args.omega_global_scale)
    if args.omega_window_start_slot is not None:
        cfg.omega_window_start_slot = int(args.omega_window_start_slot)
    if args.omega_window_end_slot is not None:
        cfg.omega_window_end_slot = int(args.omega_window_end_slot)
    if args.omega_window_scale is not None:
        cfg.omega_window_scale = float(args.omega_window_scale)
    if args.omega_od_target_scale is not None:
        cfg.omega_od_target_scale = float(args.omega_od_target_scale)
    if args.omega_arrival_dist is not None:
        cfg.omega_arrival_dist = str(args.omega_arrival_dist).strip().lower()
    if args.omega_nb_phi_mode is not None:
        cfg.omega_nb_phi_mode = str(args.omega_nb_phi_mode).strip().lower()
    if args.omega_nb_phi_global is not None:
        cfg.omega_nb_phi_global = float(args.omega_nb_phi_global)
    if args.omega_nb_phi_csv is not None:
        cfg.omega_nb_phi_csv = str(args.omega_nb_phi_csv)
    if args.omega_nb_phi_min is not None:
        cfg.omega_nb_phi_min = float(args.omega_nb_phi_min)
    if args.omega_nb_phi_max is not None:
        cfg.omega_nb_phi_max = float(args.omega_nb_phi_max)

    out = cfg.ensure_output_dirs()
    rng = np.random.default_rng(12345)

    # State dim from Scenario1FeatureBuilder spec
    state_dim = 22
    agent = DDQNAgent(
        state_dim=state_dim,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        gamma_rl=cfg.gamma_rl,
        grad_clip=cfg.grad_clip,
        device=args.device,
    )

    resumed_from = None
    if args.resume_checkpoint:
        ckpt = Path(args.resume_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt}")
        agent.load(str(ckpt))
        resumed_from = str(ckpt)

    replay = ReplayBuffer(capacity=cfg.replay_capacity)

    episode_rows = []
    loss_rows = []
    global_step = 0
    trip_counts = []

    seeds = cfg.train_seeds()
    all_trip_rows: list[dict] = []
    pbar = tqdm(seeds, total=cfg.train_episodes, desc="RL train", unit="ep")
    for ep_idx, seed in enumerate(pbar, start=1):
        eps = epsilon_by_step(global_step, cfg)
        policy = EpsilonPolicy(agent=agent, epsilon=eps, rng=rng)
        ep_result, tlog = run_episode(seed=seed, cfg=cfg, policy=policy)
        if tlog.trip_rows:
            for r in tlog.trip_rows:
                row = dict(r)
                row["episode"] = ep_idx
                row["seed"] = seed
                all_trip_rows.append(row)

        transitions_to_replay(tlog, replay)

        # Train once per new transition
        updates_this_episode = 0
        ep_losses = []
        if len(replay) >= cfg.warmup_steps:
            for _ in range(len(tlog)):
                batch = replay.sample(cfg.batch_size, rng)
                loss = agent.train_step(batch)
                ep_losses.append(float(loss))
                loss_rows.append({"episode": ep_idx, "global_step": global_step, "loss": loss})
                global_step += 1
                updates_this_episode += 1
                if global_step % cfg.target_update_every == 0:
                    agent.sync_target()
        else:
            global_step += len(tlog)

        offers = int(ep_result.relocation_offers)
        acc = int(ep_result.relocation_accepted)
        accept_rate = (acc / offers) if offers > 0 else 0.0
        mean_loss_ep = float(sum(ep_losses) / len(ep_losses)) if ep_losses else float("nan")
        generated_trips = int(ep_result.total_requests)
        trip_counts.append(generated_trips)
        tc = np.asarray(trip_counts, dtype=np.float64)
        trip_count_mean = float(tc.mean()) if tc.size else 0.0
        trip_count_var = float(tc.var(ddof=1)) if tc.size > 1 else 0.0
        trip_count_fano = float(trip_count_var / trip_count_mean) if trip_count_mean > 1e-12 else 0.0
        l_ref = max(1e-9, float(cfg.l_ref))
        e_ref = max(1e-9, float(cfg.e_ref))
        realized_vals = [float(r.get("realized_loss", 0.0)) for r in tlog.rows]
        delta_vals = [float(r.get("delta_edl", 0.0)) for r in tlog.rows]
        if realized_vals:
            realized_norm_vals = [float(min(max(v / l_ref, 0.0), 1.0)) for v in realized_vals]
            delta_norm_vals = [float(min(max(v / e_ref, -1.0), 1.0)) for v in delta_vals]
            realized_clip_hit_rate = float(sum(1 for v in realized_vals if (v / l_ref) >= 1.0) / len(realized_vals))
            delta_clip_hit_rate = float(sum(1 for v in delta_vals if abs(v / e_ref) >= 1.0) / len(delta_vals))
            mean_realized_norm = float(sum(realized_norm_vals) / len(realized_norm_vals))
            mean_delta_norm = float(sum(delta_norm_vals) / len(delta_norm_vals))
        else:
            realized_clip_hit_rate = 0.0
            delta_clip_hit_rate = 0.0
            mean_realized_norm = 0.0
            mean_delta_norm = 0.0

        episode_rows.append(
            {
                "episode": ep_idx,
                "seed": ep_result.seed,
                "epsilon": eps,
                "total_requests": ep_result.total_requests,
                "served_trips": ep_result.served_trips,
                "unserved_no_supply": ep_result.unserved_no_supply,
                "service_rate": ep_result.service_rate,
                "relocation_offers": offers,
                "relocation_accepted": acc,
                "accept_rate": accept_rate,
                "transitions": ep_result.transitions,
                "mean_reward": ep_result.mean_reward,
                "mean_reward_realized_term": ep_result.mean_reward_realized_term,
                "sum_reward_realized_term": ep_result.sum_reward_realized_term,
                "mean_reward_edl_term": ep_result.mean_reward_edl_term,
                "sum_reward_edl_term": ep_result.sum_reward_edl_term,
                "mean_reward_accept_term": ep_result.mean_reward_accept_term,
                "sum_reward_accept_term": ep_result.sum_reward_accept_term,
                "mean_realized_loss": ep_result.mean_realized_loss,
                "sum_realized_loss": ep_result.sum_realized_loss,
                "mean_delta_edl": ep_result.mean_delta_edl,
                "sum_delta_edl": ep_result.sum_delta_edl,
                "mean_realized_norm": mean_realized_norm,
                "mean_delta_edl_norm": mean_delta_norm,
                "realized_clip_hit_rate": realized_clip_hit_rate,
                "delta_edl_clip_hit_rate": delta_clip_hit_rate,
                "generated_trips": generated_trips,
                "trip_count_mean": trip_count_mean,
                "trip_count_var": trip_count_var,
                "fano_like": trip_count_fano,
                "replay_size": len(replay),
                "updates_this_episode": updates_this_episode,
                "mean_loss_episode": mean_loss_ep,
            }
        )

        # Persist transitions for audit/debug
        if len(tlog) > 0 and (cfg.transition_dump_every > 0) and (ep_idx % cfg.transition_dump_every == 0):
            pd.DataFrame(tlog.rows).drop(columns=["state", "next_state"]).to_csv(
                out / "metrics" / f"transitions_ep{ep_idx:04d}.csv", index=False
            )

        if ep_idx % max(1, int(args.checkpoint_every)) == 0:
            agent.save(str(out / "checkpoints" / f"ddqn_ep{ep_idx:04d}.pt"))

        pbar.set_postfix(
            {
                "eps": f"{eps:.3f}",
                "loss": f"{mean_loss_ep:.4f}" if not np.isnan(mean_loss_ep) else "na",
                "R": f"{ep_result.mean_reward:.3f}",
                "offers": offers,
                "acc%": f"{accept_rate:.1%}",
                "Lr": f"{ep_result.mean_reward_realized_term:.3f}",
                "Le": f"{ep_result.mean_reward_edl_term:.3f}",
                "La": f"{ep_result.mean_reward_accept_term:.3f}",
                "rLoss": f"{ep_result.mean_realized_loss:.3f}",
                "Lclip%": f"{realized_clip_hit_rate:.1%}",
                "Eclip%": f"{delta_clip_hit_rate:.1%}",
            }
        )

        if cfg.early_stop_episode is not None and cfg.early_stop_min_offers is not None and ep_idx >= cfg.early_stop_episode:
            mean_offers_so_far = float(np.mean([float(r["relocation_offers"]) for r in episode_rows]))
            if mean_offers_so_far < float(cfg.early_stop_min_offers):
                print(
                    f"[EARLY STOP] ep={ep_idx} mean_offers={mean_offers_so_far:.4f} "
                    f"< threshold={float(cfg.early_stop_min_offers):.4f}"
                )
                break

        if ep_idx >= cfg.train_episodes:
            break

    # Final artifacts
    agent.save(str(out / "checkpoints" / "ddqn_final.pt"))
    pd.DataFrame(episode_rows).to_csv(out / "metrics" / "train_episode_metrics.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out / "metrics" / "train_loss_curve.csv", index=False)
    write_trip_run_report(
        trip_rows=all_trip_rows,
        output_dir=out,
        planning_period_min=float(cfg.planning_period),
        file_prefix="train_trip",
    )

    dump_training_meta(
        str(out / "metrics" / "train_meta.json"),
        {
            "episodes": cfg.train_episodes,
            "state_dim": state_dim,
            "batch_size": cfg.batch_size,
            "replay_capacity": cfg.replay_capacity,
            "warmup_steps": cfg.warmup_steps,
            "target_update_every": cfg.target_update_every,
            "epsilon_start": cfg.epsilon_start,
            "epsilon_end": cfg.epsilon_end,
            "epsilon_decay_steps": cfg.epsilon_decay_steps,
            "w_l": cfg.w_l,
            "w_e": cfg.w_e,
            "beta_a": cfg.beta_a,
            "beta_c": cfg.beta_c,
            "beta_r": cfg.beta_r,
            "l_ref": cfg.l_ref,
            "e_ref": cfg.e_ref,
            "or_input_path": cfg.or_input_path,
            "omega_profile": {
                "global": cfg.omega_global_scale,
                "window_start": cfg.omega_window_start_slot,
                "window_end": cfg.omega_window_end_slot,
                "window_scale": cfg.omega_window_scale,
                "od_target_scale": cfg.omega_od_target_scale,
                "arrival_dist": cfg.omega_arrival_dist,
                "nb_phi_mode": cfg.omega_nb_phi_mode,
                "nb_phi_global": cfg.omega_nb_phi_global,
                "nb_phi_csv": cfg.omega_nb_phi_csv,
                "nb_phi_min": cfg.omega_nb_phi_min,
                "nb_phi_max": cfg.omega_nb_phi_max,
            },
            "transition_dump_every": cfg.transition_dump_every,
            "early_stop_episode": cfg.early_stop_episode,
            "early_stop_min_offers": cfg.early_stop_min_offers,
            "resumed_from": resumed_from,
            "mean_loss": agent.stats.mean_loss,
            "num_updates": agent.stats.updates,
        },
    )

    print(f"Training complete. Output: {out}")


if __name__ == "__main__":
    main()
