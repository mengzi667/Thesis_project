from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from rl.agent import DDQNAgent
from rl.config import RLConfig
from rl.trainer import AlwaysOfferPolicy, EpsilonPolicy, NoOfferPolicy, run_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate L_ref / E_ref from transition samples (P95).")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed-start", type=int, default=31000)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--output-json", type=str, default="results/ref_calibration.json")
    p.add_argument("--or-input-path", type=str, default=None)
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
    p.add_argument(
        "--policy",
        type=str,
        choices=["epsilon_greedy", "always_offer", "no_offer"],
        default="epsilon_greedy",
        help="Policy used during calibration sampling.",
    )
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for epsilon_greedy policy.")
    p.add_argument("--epsilon", type=float, default=0.1, help="Exploration epsilon for epsilon_greedy calibration.")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _apply_cfg_overrides(cfg: RLConfig, args: argparse.Namespace) -> None:
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


def main() -> None:
    args = parse_args()
    q = float(args.quantile)
    if not (0.5 <= q <= 0.999):
        raise ValueError("quantile should be in [0.5, 0.999]")

    cfg = RLConfig()
    _apply_cfg_overrides(cfg, args)

    realized_vals: list[float] = []
    delta_abs_vals: list[float] = []
    transitions = 0
    if args.policy == "always_offer":
        policy = AlwaysOfferPolicy()
    elif args.policy == "no_offer":
        policy = NoOfferPolicy()
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --policy epsilon_greedy")
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        rng = np.random.default_rng(20260429)
        agent = DDQNAgent(
            state_dim=22,
            hidden_dim=cfg.hidden_dim,
            lr=cfg.lr,
            gamma_rl=cfg.gamma_rl,
            grad_clip=cfg.grad_clip,
            device=args.device,
        )
        agent.load(str(ckpt_path))
        policy = EpsilonPolicy(agent=agent, epsilon=float(args.epsilon), rng=rng)

    pbar = tqdm(range(int(args.episodes)), total=int(args.episodes), desc="Calibrate refs", unit="ep")
    for ep in pbar:
        seed = int(args.seed_start) + ep
        _, tlog = run_episode(seed=seed, cfg=cfg, policy=policy)
        transitions += len(tlog.rows)
        for row in tlog.rows:
            realized_vals.append(float(row.get("realized_loss", 0.0)))
            delta_abs_vals.append(abs(float(row.get("delta_edl", 0.0))))
        pbar.set_postfix(
            {
                "seed": seed,
                "trans": transitions,
                "rows": len(tlog.rows),
            }
        )

    # Fallback anchors when samples are zero-dominant.
    l_ref = float(np.quantile(np.asarray(realized_vals), q)) if realized_vals else 1.0
    e_ref = float(np.quantile(np.asarray(delta_abs_vals), q)) if delta_abs_vals else 1.0
    l_ref = max(l_ref, 1.0)
    e_ref = max(e_ref, 1e-6)

    payload = {
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "quantile": q,
        "transitions": int(transitions),
        "l_ref": float(l_ref),
        "e_ref": float(e_ref),
        "policy": str(args.policy),
        "epsilon": float(args.epsilon),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
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
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Calibration complete. L_ref={l_ref:.6g}, E_ref={e_ref:.6g}, transitions={transitions}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
