from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rl.agent import DDQNAgent
from rl.config import RLConfig
from rl.trainer import GreedyPolicy, run_episode, run_or_only_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OR-only vs OR+RL")
    p.add_argument("--checkpoint", type=str, default="results/rl_scenario1/checkpoints/ddqn_final.pt")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed-start", type=int, default=None, help="Start seed for evaluation episodes")

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
        cfg.eval_episodes = int(args.episodes)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.seed_start is not None:
        cfg.seed_start_eval = int(args.seed_start)

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
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    state_dim = 22
    agent = DDQNAgent(
        state_dim=state_dim,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        gamma_rl=cfg.gamma_rl,
        grad_clip=cfg.grad_clip,
        device=args.device,
    )
    agent.load(str(ckpt))
    rng = np.random.default_rng(54321)
    policy = GreedyPolicy(agent=agent, rng=rng)

    rows = []
    for ep_idx, seed in enumerate(cfg.eval_seeds(), start=1):
        r_or = run_or_only_episode(seed=seed, cfg=cfg)
        r_rl, _ = run_episode(seed=seed, cfg=cfg, policy=policy)
        rows.append(
            {
                "episode": ep_idx,
                "seed": seed,
                "or_total_requests": r_or.total_requests,
                "or_served_trips": r_or.served_trips,
                "or_service_rate": r_or.service_rate,
                "or_relocation_offers": r_or.relocation_offers,
                "or_relocation_accepted": r_or.relocation_accepted,
                "or_mean_reward_realized_term": r_or.mean_reward_realized_term,
                "or_sum_reward_realized_term": r_or.sum_reward_realized_term,
                "or_mean_reward_edl_term": r_or.mean_reward_edl_term,
                "or_sum_reward_edl_term": r_or.sum_reward_edl_term,
                "or_mean_realized_loss": r_or.mean_realized_loss,
                "or_sum_realized_loss": r_or.sum_realized_loss,
                "or_mean_delta_edl": r_or.mean_delta_edl,
                "or_sum_delta_edl": r_or.sum_delta_edl,
                "rl_total_requests": r_rl.total_requests,
                "rl_served_trips": r_rl.served_trips,
                "rl_service_rate": r_rl.service_rate,
                "rl_relocation_offers": r_rl.relocation_offers,
                "rl_relocation_accepted": r_rl.relocation_accepted,
                "rl_transitions": r_rl.transitions,
                "rl_mean_reward": r_rl.mean_reward,
                "rl_mean_reward_realized_term": r_rl.mean_reward_realized_term,
                "rl_sum_reward_realized_term": r_rl.sum_reward_realized_term,
                "rl_mean_reward_edl_term": r_rl.mean_reward_edl_term,
                "rl_sum_reward_edl_term": r_rl.sum_reward_edl_term,
                "rl_mean_realized_loss": r_rl.mean_realized_loss,
                "rl_sum_realized_loss": r_rl.sum_realized_loss,
                "rl_mean_delta_edl": r_rl.mean_delta_edl,
                "rl_sum_delta_edl": r_rl.sum_delta_edl,
                "or_generated_trips": r_or.total_requests,
                "rl_generated_trips": r_rl.total_requests,
            }
        )
        if ep_idx >= cfg.eval_episodes:
            break

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics" / "eval_or_vs_rl.csv", index=False)

    summary = {
        "episodes": int(len(df)),
        "or_mean_service_rate": float(df["or_service_rate"].mean()) if len(df) else 0.0,
        "rl_mean_service_rate": float(df["rl_service_rate"].mean()) if len(df) else 0.0,
        "or_mean_offers": float(df["or_relocation_offers"].mean()) if len(df) else 0.0,
        "rl_mean_offers": float(df["rl_relocation_offers"].mean()) if len(df) else 0.0,
        "rl_mean_reward": float(df["rl_mean_reward"].mean()) if len(df) else 0.0,
        "or_mean_reward_realized_term": float(df["or_mean_reward_realized_term"].mean()) if len(df) else 0.0,
        "or_sum_reward_realized_term": float(df["or_sum_reward_realized_term"].sum()) if len(df) else 0.0,
        "or_mean_reward_edl_term": float(df["or_mean_reward_edl_term"].mean()) if len(df) else 0.0,
        "or_sum_reward_edl_term": float(df["or_sum_reward_edl_term"].sum()) if len(df) else 0.0,
        "rl_mean_reward_realized_term": float(df["rl_mean_reward_realized_term"].mean()) if len(df) else 0.0,
        "rl_sum_reward_realized_term": float(df["rl_sum_reward_realized_term"].sum()) if len(df) else 0.0,
        "rl_mean_reward_edl_term": float(df["rl_mean_reward_edl_term"].mean()) if len(df) else 0.0,
        "rl_sum_reward_edl_term": float(df["rl_sum_reward_edl_term"].sum()) if len(df) else 0.0,
        "or_mean_realized_loss": float(df["or_mean_realized_loss"].mean()) if len(df) else 0.0,
        "or_sum_realized_loss": float(df["or_sum_realized_loss"].sum()) if len(df) else 0.0,
        "or_mean_delta_edl": float(df["or_mean_delta_edl"].mean()) if len(df) else 0.0,
        "or_sum_delta_edl": float(df["or_sum_delta_edl"].sum()) if len(df) else 0.0,
        "rl_mean_realized_loss": float(df["rl_mean_realized_loss"].mean()) if len(df) else 0.0,
        "rl_sum_realized_loss": float(df["rl_sum_realized_loss"].sum()) if len(df) else 0.0,
        "rl_mean_delta_edl": float(df["rl_mean_delta_edl"].mean()) if len(df) else 0.0,
        "rl_sum_delta_edl": float(df["rl_sum_delta_edl"].sum()) if len(df) else 0.0,
        "or_trip_count_mean": float(df["or_generated_trips"].mean()) if len(df) else 0.0,
        "or_trip_count_var": float(df["or_generated_trips"].var(ddof=1)) if len(df) > 1 else 0.0,
        "or_fano_like": (
            float(df["or_generated_trips"].var(ddof=1) / df["or_generated_trips"].mean())
            if len(df) > 1 and float(df["or_generated_trips"].mean()) > 1e-12
            else 0.0
        ),
        "rl_trip_count_mean": float(df["rl_generated_trips"].mean()) if len(df) else 0.0,
        "rl_trip_count_var": float(df["rl_generated_trips"].var(ddof=1)) if len(df) > 1 else 0.0,
        "rl_fano_like": (
            float(df["rl_generated_trips"].var(ddof=1) / df["rl_generated_trips"].mean())
            if len(df) > 1 and float(df["rl_generated_trips"].mean()) > 1e-12
            else 0.0
        ),
        "or_input_path": cfg.or_input_path,
        "omega_global_scale": cfg.omega_global_scale,
        "omega_window_start_slot": cfg.omega_window_start_slot,
        "omega_window_end_slot": cfg.omega_window_end_slot,
        "omega_window_scale": cfg.omega_window_scale,
        "omega_od_target_scale": cfg.omega_od_target_scale,
        "omega_arrival_dist": cfg.omega_arrival_dist,
        "omega_nb_phi_mode": cfg.omega_nb_phi_mode,
        "omega_nb_phi_global": cfg.omega_nb_phi_global,
        "omega_nb_phi_csv": cfg.omega_nb_phi_csv,
        "omega_nb_phi_min": cfg.omega_nb_phi_min,
        "omega_nb_phi_max": cfg.omega_nb_phi_max,
    }
    pd.DataFrame([summary]).to_csv(out / "metrics" / "eval_summary.csv", index=False)
    print(f"Evaluation complete. Output: {out}")


if __name__ == "__main__":
    main()
