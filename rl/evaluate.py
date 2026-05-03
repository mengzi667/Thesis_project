from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rl.agent import DDQNAgent
from rl.config import RLConfig
from rl.trip_report import write_trip_run_report
from rl.trainer import AlwaysOfferPolicy, GreedyPolicy, NoOfferPolicy, run_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate three policies: always_offer / no_offer / checkpoint")
    p.add_argument("--checkpoint", type=str, default="results/rl_scenario1/checkpoints/ddqn_final.pt")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed-start", type=int, default=None, help="Start seed for evaluation episodes")
    p.add_argument(
        "--result-tag",
        type=str,
        default="main_comparable",
        choices=["main_comparable", "diagnostic_stress", "final_retrained_stress"],
    )
    p.add_argument("--w-l", type=float, default=None)
    p.add_argument("--w-e", type=float, default=None)
    p.add_argument("--beta-a", type=float, default=None)
    p.add_argument("--beta-c", type=float, default=None)
    p.add_argument("--beta-r", type=float, default=None)
    p.add_argument("--l-ref", type=float, default=None)
    p.add_argument("--e-ref", type=float, default=None)

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


def _apply_cfg_overrides(cfg: RLConfig, args: argparse.Namespace) -> None:
    if args.episodes is not None:
        cfg.eval_episodes = int(args.episodes)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.seed_start is not None:
        cfg.seed_start_eval = int(args.seed_start)
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


def _to_prefixed_metrics(prefix: str, result) -> dict:
    return {
        f"{prefix}_total_requests": result.total_requests,
        f"{prefix}_served_trips": result.served_trips,
        f"{prefix}_unserved_no_supply": result.unserved_no_supply,
        f"{prefix}_total_no_supply": result.unserved_no_supply,
        f"{prefix}_service_rate": result.service_rate,
        f"{prefix}_relocation_offers": result.relocation_offers,
        f"{prefix}_relocation_accepted": result.relocation_accepted,
        f"{prefix}_transitions": result.transitions,
        f"{prefix}_mean_reward": result.mean_reward,
        f"{prefix}_mean_reward_realized_term": result.mean_reward_realized_term,
        f"{prefix}_sum_reward_realized_term": result.sum_reward_realized_term,
        f"{prefix}_mean_reward_edl_term": result.mean_reward_edl_term,
        f"{prefix}_sum_reward_edl_term": result.sum_reward_edl_term,
        f"{prefix}_mean_reward_accept_term": result.mean_reward_accept_term,
        f"{prefix}_sum_reward_accept_term": result.sum_reward_accept_term,
        f"{prefix}_mean_realized_loss": result.mean_realized_loss,
        f"{prefix}_sum_realized_loss": result.sum_realized_loss,
        f"{prefix}_window_od123_loss": result.sum_realized_loss,
        f"{prefix}_mean_delta_edl": result.mean_delta_edl,
        f"{prefix}_sum_delta_edl": result.sum_delta_edl,
        f"{prefix}_generated_trips": result.total_requests,
    }


def _summary_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) else 0.0


def _summary_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if len(df) else 0.0


def _fano_like(df: pd.DataFrame, col: str) -> tuple[float, float, float]:
    if len(df) == 0:
        return 0.0, 0.0, 0.0
    m = float(df[col].mean())
    v = float(df[col].var(ddof=1)) if len(df) > 1 else 0.0
    f = float(v / m) if m > 1e-12 else 0.0
    return m, v, f


def main() -> None:
    args = parse_args()
    cfg = RLConfig()
    _apply_cfg_overrides(cfg, args)
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

    ao_policy = AlwaysOfferPolicy()
    no_policy = NoOfferPolicy()
    ckpt_policy = GreedyPolicy(agent=agent, rng=rng)

    rows = []
    trip_rows: list[dict] = []
    seeds = cfg.eval_seeds()
    pbar = tqdm(seeds, total=cfg.eval_episodes, desc="RL eval (3 policies)", unit="ep")
    for ep_idx, seed in enumerate(pbar, start=1):
        r_ao, t_ao = run_episode(seed=seed, cfg=cfg, policy=ao_policy)
        r_no, t_no = run_episode(seed=seed, cfg=cfg, policy=no_policy)
        r_ck, t_ck = run_episode(seed=seed, cfg=cfg, policy=ckpt_policy)

        if t_ao.trip_rows:
            for r in t_ao.trip_rows:
                rr = dict(r)
                rr["episode"] = ep_idx
                rr["seed"] = seed
                rr["policy"] = "always_offer"
                trip_rows.append(rr)
        if t_no.trip_rows:
            for r in t_no.trip_rows:
                rr = dict(r)
                rr["episode"] = ep_idx
                rr["seed"] = seed
                rr["policy"] = "no_offer"
                trip_rows.append(rr)
        if t_ck.trip_rows:
            for r in t_ck.trip_rows:
                rr = dict(r)
                rr["episode"] = ep_idx
                rr["seed"] = seed
                rr["policy"] = "checkpoint"
                trip_rows.append(rr)

        row = {"episode": ep_idx, "seed": seed}
        row.update(_to_prefixed_metrics("ao", r_ao))
        row.update(_to_prefixed_metrics("no", r_no))
        row.update(_to_prefixed_metrics("ckpt", r_ck))
        rows.append(row)

        pbar.set_postfix(
            {
                "ao_sr": f"{r_ao.service_rate:.3f}",
                "no_sr": f"{r_no.service_rate:.3f}",
                "ck_sr": f"{r_ck.service_rate:.3f}",
                "ao_off": int(r_ao.relocation_offers),
                "no_off": int(r_no.relocation_offers),
                "ck_off": int(r_ck.relocation_offers),
            }
        )
        if ep_idx >= cfg.eval_episodes:
            break

    df = pd.DataFrame(rows)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics" / "eval_three_policies.csv", index=False)
    # Compatibility alias for existing tooling expecting old filename.
    df.to_csv(out / "metrics" / "eval_or_vs_rl.csv", index=False)

    summary = {"episodes": int(len(df)), "strategy_set": "always_offer|no_offer|checkpoint"}

    for prefix in ("ao", "no", "ckpt"):
        summary[f"{prefix}_mean_service_rate"] = _summary_mean(df, f"{prefix}_service_rate")
        summary[f"{prefix}_sum_no_supply"] = _summary_sum(df, f"{prefix}_unserved_no_supply")
        summary[f"{prefix}_total_no_supply"] = summary[f"{prefix}_sum_no_supply"]
        summary[f"{prefix}_mean_transitions"] = _summary_mean(df, f"{prefix}_transitions")
        summary[f"{prefix}_mean_offers"] = _summary_mean(df, f"{prefix}_relocation_offers")
        summary[f"{prefix}_mean_reward"] = _summary_mean(df, f"{prefix}_mean_reward")
        summary[f"{prefix}_mean_reward_realized_term"] = _summary_mean(df, f"{prefix}_mean_reward_realized_term")
        summary[f"{prefix}_sum_reward_realized_term"] = _summary_sum(df, f"{prefix}_sum_reward_realized_term")
        summary[f"{prefix}_mean_reward_edl_term"] = _summary_mean(df, f"{prefix}_mean_reward_edl_term")
        summary[f"{prefix}_sum_reward_edl_term"] = _summary_sum(df, f"{prefix}_sum_reward_edl_term")
        summary[f"{prefix}_mean_reward_accept_term"] = _summary_mean(df, f"{prefix}_mean_reward_accept_term")
        summary[f"{prefix}_sum_reward_accept_term"] = _summary_sum(df, f"{prefix}_sum_reward_accept_term")
        summary[f"{prefix}_mean_realized_loss"] = _summary_mean(df, f"{prefix}_mean_realized_loss")
        summary[f"{prefix}_sum_realized_loss"] = _summary_sum(df, f"{prefix}_sum_realized_loss")
        summary[f"{prefix}_window_od123_loss"] = summary[f"{prefix}_sum_realized_loss"]
        summary[f"{prefix}_mean_delta_edl"] = _summary_mean(df, f"{prefix}_mean_delta_edl")
        summary[f"{prefix}_sum_delta_edl"] = _summary_sum(df, f"{prefix}_sum_delta_edl")
        tc_m, tc_v, tc_f = _fano_like(df, f"{prefix}_generated_trips")
        summary[f"{prefix}_trip_count_mean"] = tc_m
        summary[f"{prefix}_trip_count_var"] = tc_v
        summary[f"{prefix}_fano_like"] = tc_f

    summary.update(
        {
            "result_tag": str(args.result_tag),
            "checkpoint_path": str(ckpt),
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
            "w_l": cfg.w_l,
            "w_e": cfg.w_e,
            "beta_a": cfg.beta_a,
            "beta_c": cfg.beta_c,
            "beta_r": cfg.beta_r,
            "l_ref": cfg.l_ref,
            "e_ref": cfg.e_ref,
        }
    )

    pd.DataFrame([summary]).to_csv(out / "metrics" / "eval_summary.csv", index=False)
    write_trip_run_report(
        trip_rows=trip_rows,
        output_dir=out,
        planning_period_min=float(cfg.planning_period),
        file_prefix="eval_trip",
    )
    print(f"Evaluation complete. Output: {out}")


if __name__ == "__main__":
    main()
