from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import (
    OMEGA_ARRIVAL_DIST,
    OMEGA_GLOBAL_SCALE,
    OMEGA_NB_PHI_CSV,
    OMEGA_NB_PHI_GLOBAL,
    OMEGA_NB_PHI_MAX,
    OMEGA_NB_PHI_MIN,
    OMEGA_NB_PHI_MODE,
    OMEGA_OD_TARGET_SCALE,
    OMEGA_WINDOW_END_SLOT,
    OMEGA_WINDOW_SCALE,
    OMEGA_WINDOW_START_SLOT,
)


@dataclass
class RLConfig:
    # Training horizon
    episode_minutes: float = 120.0
    planning_period: float = 15.0

    # Hybrid reward coefficients
    # r = lambda*(-clip(L_real/L_ref,0,1)) + (1-lambda)*clip(DeltaEDL/E_ref,-1,1)
    #     - beta_c*cost - beta_r*1_reject
    reward_lambda: float = 0.7
    beta_c: float = 1.0
    beta_r: float = 0.1
    l_ref: float = 1.0
    e_ref: float = 1.0

    # Cost accounting
    # "offer" -> cost when offer is issued
    # "accept" -> cost only when accepted
    cost_mode: str = "accept"

    # DDQN hyper-parameters
    lr: float = 1e-3
    gamma_rl: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 1_000
    target_update_every: int = 200
    warmup_steps: int = 32
    grad_clip: float = 10.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15_000

    # Net
    hidden_dim: int = 128
    # hidden_dim: int = 64
    # hidden_dim: int = 32

    # Experiment
    train_episodes: int = 2000
    eval_episodes: int = 200
    seed_start_train: int = 11000
    seed_start_eval: int = 21000

    output_dir: str = "results/rl_scenario1"

    # Optional runtime override for OR input path.
    # If None, falls back to config.OR_INPUT_PATH inside main.build_simulation.
    or_input_path: str | None = None

    # Omega sampling profile (can be overridden by CLI in train/evaluate)
    omega_global_scale: float = float(OMEGA_GLOBAL_SCALE)
    omega_window_start_slot: int = int(OMEGA_WINDOW_START_SLOT)
    omega_window_end_slot: int = int(OMEGA_WINDOW_END_SLOT)
    omega_window_scale: float = float(OMEGA_WINDOW_SCALE)
    omega_od_target_scale: float = float(OMEGA_OD_TARGET_SCALE)
    omega_arrival_dist: str = str(OMEGA_ARRIVAL_DIST)
    omega_nb_phi_mode: str = str(OMEGA_NB_PHI_MODE)
    omega_nb_phi_global: float = float(OMEGA_NB_PHI_GLOBAL)
    omega_nb_phi_csv: str = str(OMEGA_NB_PHI_CSV)
    omega_nb_phi_min: float = float(OMEGA_NB_PHI_MIN)
    omega_nb_phi_max: float = float(OMEGA_NB_PHI_MAX)

    def train_seeds(self) -> List[int]:
        return [self.seed_start_train + i for i in range(self.train_episodes)]

    def eval_seeds(self) -> List[int]:
        return [self.seed_start_eval + i for i in range(self.eval_episodes)]

    def ensure_output_dirs(self) -> Path:
        root = Path(self.output_dir)
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        return root
