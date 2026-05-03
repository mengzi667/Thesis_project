from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
from dataclasses import asdict

from main import build_simulation
from rl.agent import DDQNAgent
from rl.config import RLConfig
from rl.replay_buffer import ReplayBuffer, Transition
from rl.runtime import Scenario1FeatureBuilder, TransitionLogger


@dataclass
class EpisodeResult:
    seed: int
    total_requests: int
    served_trips: int
    unserved_no_supply: int
    relocation_offers: int
    relocation_accepted: int
    service_rate: float
    transitions: int
    mean_reward: float
    mean_reward_realized_term: float
    sum_reward_realized_term: float
    mean_reward_edl_term: float
    sum_reward_edl_term: float
    mean_reward_accept_term: float
    sum_reward_accept_term: float
    mean_realized_loss: float
    sum_realized_loss: float
    mean_delta_edl: float
    sum_delta_edl: float


class EpsilonPolicy:
    def __init__(self, agent: DDQNAgent, epsilon: float, rng: np.random.Generator) -> None:
        self.agent = agent
        self.epsilon = float(epsilon)
        self.rng = rng

    def act(self, state: np.ndarray) -> int:
        return self.agent.act(state, epsilon=self.epsilon, rng=self.rng)


class GreedyPolicy:
    def __init__(self, agent: DDQNAgent, rng: np.random.Generator) -> None:
        self.agent = agent
        self.rng = rng

    def act(self, state: np.ndarray) -> int:
        return self.agent.act(state, epsilon=0.0, rng=self.rng)


class AlwaysOfferPolicy:
    """OR baseline policy: never suppress OR offers (action=1)."""

    def act(self, state: np.ndarray) -> int:
        _ = state
        return 1


class NoOfferPolicy:
    """No-offer baseline policy: always suppress offers (action=0)."""

    def act(self, state: np.ndarray) -> int:
        _ = state
        return 0


def epsilon_by_step(step: int, cfg: RLConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = float(step) / float(max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def build_rl_engine(seed: int, cfg: RLConfig, policy, transition_logger: TransitionLogger):
    engine = build_simulation(
        seed=seed,
        verbose=False,
        omega_global_scale=cfg.omega_global_scale,
        omega_window_start_slot=cfg.omega_window_start_slot,
        omega_window_end_slot=cfg.omega_window_end_slot,
        omega_window_scale=cfg.omega_window_scale,
        omega_od_target_scale=cfg.omega_od_target_scale,
        omega_arrival_dist=cfg.omega_arrival_dist,
        omega_nb_phi_mode=cfg.omega_nb_phi_mode,
        omega_nb_phi_global=cfg.omega_nb_phi_global,
        omega_nb_phi_csv=cfg.omega_nb_phi_csv,
        omega_nb_phi_min=cfg.omega_nb_phi_min,
        omega_nb_phi_max=cfg.omega_nb_phi_max,
        or_input_path_override=cfg.or_input_path,
    )
    engine.print_snapshots = False
    cap = next(iter(engine.spatial.zones.values())).capacity if engine.spatial.zones else 10
    feat_builder = Scenario1FeatureBuilder(
        zone_ids=engine.spatial.all_zone_ids(),
        zone_capacity=float(cap),
        planning_period=cfg.planning_period,
        episode_minutes=cfg.episode_minutes,
    )
    engine.rl_policy = policy
    engine.rl_feature_builder = feat_builder
    engine.rl_transition_logger = transition_logger
    engine.rl_reward_cfg = {
        "w_l": cfg.w_l,
        "w_e": cfg.w_e,
        "beta_a": cfg.beta_a,
        "beta_c": cfg.beta_c,
        "beta_r": cfg.beta_r,
        "l_ref": cfg.l_ref,
        "e_ref": cfg.e_ref,
    }
    engine.episode_minutes = cfg.episode_minutes
    return engine


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _transition_metrics(tlog: TransitionLogger) -> Dict[str, float]:
    if len(tlog.rows) == 0:
        return {
            "mean_reward": 0.0,
            "mean_reward_realized_term": 0.0,
            "sum_reward_realized_term": 0.0,
            "mean_reward_edl_term": 0.0,
            "sum_reward_edl_term": 0.0,
            "mean_reward_accept_term": 0.0,
            "sum_reward_accept_term": 0.0,
            "mean_realized_loss": 0.0,
            "sum_realized_loss": 0.0,
            "mean_delta_edl": 0.0,
            "sum_delta_edl": 0.0,
        }

    rewards = [float(r.get("reward", 0.0)) for r in tlog.rows]
    realized_terms = [float(r.get("reward_realized_term", 0.0)) for r in tlog.rows]
    edl_terms = [float(r.get("reward_edl_term", 0.0)) for r in tlog.rows]
    accept_terms = [float(r.get("reward_accept_term", 0.0)) for r in tlog.rows]
    realized_loss = [float(r.get("realized_loss", 0.0)) for r in tlog.rows]
    delta_edl = [float(r.get("delta_edl", 0.0)) for r in tlog.rows]

    return {
        "mean_reward": _safe_mean(rewards),
        "mean_reward_realized_term": _safe_mean(realized_terms),
        "sum_reward_realized_term": float(sum(realized_terms)),
        "mean_reward_edl_term": _safe_mean(edl_terms),
        "sum_reward_edl_term": float(sum(edl_terms)),
        "mean_reward_accept_term": _safe_mean(accept_terms),
        "sum_reward_accept_term": float(sum(accept_terms)),
        "mean_realized_loss": _safe_mean(realized_loss),
        "sum_realized_loss": float(sum(realized_loss)),
        "mean_delta_edl": _safe_mean(delta_edl),
        "sum_delta_edl": float(sum(delta_edl)),
    }


def run_episode(seed: int, cfg: RLConfig, policy) -> tuple[EpisodeResult, TransitionLogger]:
    tlog = TransitionLogger()
    engine = build_rl_engine(seed=seed, cfg=cfg, policy=policy, transition_logger=tlog)
    logger = engine.run(sim_duration=cfg.episode_minutes)
    tlog.trip_rows = [asdict(r) for r in logger.trip_records]
    s = logger.summary()
    tm = _transition_metrics(tlog)
    result = EpisodeResult(
        seed=seed,
        total_requests=int(s["total_requests"]),
        served_trips=int(s["served_trips"]),
        unserved_no_supply=int(s.get("unserved_no_supply", 0)),
        relocation_offers=int(s["relocation_offers"]),
        relocation_accepted=int(s["relocation_accepted"]),
        service_rate=float(s["service_rate"]),
        transitions=len(tlog),
        mean_reward=float(tm["mean_reward"]),
        mean_reward_realized_term=float(tm["mean_reward_realized_term"]),
        sum_reward_realized_term=float(tm["sum_reward_realized_term"]),
        mean_reward_edl_term=float(tm["mean_reward_edl_term"]),
        sum_reward_edl_term=float(tm["sum_reward_edl_term"]),
        mean_reward_accept_term=float(tm["mean_reward_accept_term"]),
        sum_reward_accept_term=float(tm["sum_reward_accept_term"]),
        mean_realized_loss=float(tm["mean_realized_loss"]),
        sum_realized_loss=float(tm["sum_realized_loss"]),
        mean_delta_edl=float(tm["mean_delta_edl"]),
        sum_delta_edl=float(tm["sum_delta_edl"]),
    )
    return result, tlog


def run_or_only_episode(seed: int, cfg: RLConfig) -> EpisodeResult:
    # Keep OR decisions untouched (always-offer policy), but enable transition
    # logging so OR metrics are computed with the same reward/EDL decomposition
    # as RL, making comparisons apples-to-apples.
    tlog = TransitionLogger()
    engine = build_rl_engine(seed=seed, cfg=cfg, policy=AlwaysOfferPolicy(), transition_logger=tlog)
    logger = engine.run(sim_duration=cfg.episode_minutes)
    s = logger.summary()
    tm = _transition_metrics(tlog)
    return EpisodeResult(
        seed=seed,
        total_requests=int(s["total_requests"]),
        served_trips=int(s["served_trips"]),
        unserved_no_supply=int(s.get("unserved_no_supply", 0)),
        relocation_offers=int(s["relocation_offers"]),
        relocation_accepted=int(s["relocation_accepted"]),
        service_rate=float(s["service_rate"]),
        transitions=len(tlog),
        mean_reward=float(tm["mean_reward"]),
        mean_reward_realized_term=float(tm["mean_reward_realized_term"]),
        sum_reward_realized_term=float(tm["sum_reward_realized_term"]),
        mean_reward_edl_term=float(tm["mean_reward_edl_term"]),
        sum_reward_edl_term=float(tm["sum_reward_edl_term"]),
        mean_reward_accept_term=float(tm["mean_reward_accept_term"]),
        sum_reward_accept_term=float(tm["sum_reward_accept_term"]),
        mean_realized_loss=float(tm["mean_realized_loss"]),
        sum_realized_loss=float(tm["sum_realized_loss"]),
        mean_delta_edl=float(tm["mean_delta_edl"]),
        sum_delta_edl=float(tm["sum_delta_edl"]),
    )


def transitions_to_replay(tlog: TransitionLogger, replay: ReplayBuffer) -> None:
    for row in tlog.rows:
        replay.push(
            Transition(
                state=np.asarray(row["state"], dtype=np.float32),
                action=int(row["action"]),
                reward=float(row["reward"]),
                next_state=np.asarray(row["next_state"], dtype=np.float32),
                done=float(row["done"]),
                info={
                    "request_id": int(row["request_id"]),
                    "offered": int(row["offered"]),
                    "accepted": int(row["accepted"]),
                    "reject_flag": int(row["reject_flag"]),
                    "delta_edl": float(row["delta_edl"]),
                    "cost_term": float(row["cost_term"]),
                    "realized_loss": float(row.get("realized_loss", 0.0)),
                    "reward_realized_term": float(row.get("reward_realized_term", 0.0)),
                    "reward_edl_term": float(row.get("reward_edl_term", 0.0)),
                    "reward_accept_term": float(row.get("reward_accept_term", 0.0)),
                },
            )
        )


def dump_training_meta(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
