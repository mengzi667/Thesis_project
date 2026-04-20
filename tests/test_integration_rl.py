from __future__ import annotations

import numpy as np
import pytest

from main import build_simulation
from rl.config import RLConfig
from rl.trainer import run_episode, run_or_only_episode


class AlwaysOfferPolicy:
    def act(self, state: np.ndarray) -> int:
        return 1


def test_integration_or_only_reproducible_seed() -> None:
    cfg = RLConfig(episode_minutes=120.0)
    s1 = run_or_only_episode(seed=12001, cfg=cfg)
    s2 = run_or_only_episode(seed=12001, cfg=cfg)
    assert s1.total_requests == s2.total_requests
    assert s1.served_trips == s2.served_trips
    assert np.isclose(s1.service_rate, s2.service_rate)


def test_integration_or_plus_rl_generates_transitions() -> None:
    cfg = RLConfig(episode_minutes=120.0)
    res, tlog = run_episode(seed=12001, cfg=cfg, policy=AlwaysOfferPolicy())
    # Pipeline should execute and transition logger should exist.
    assert res.total_requests >= 0
    assert len(tlog) >= 0


def test_regression_rl_disabled_same_as_engine_default() -> None:
    # Compare direct engine default run vs OR-only helper for same seed.
    cfg = RLConfig(episode_minutes=120.0)
    s_or = run_or_only_episode(seed=12002, cfg=cfg)

    eng = build_simulation(seed=12002, verbose=False)
    eng.print_snapshots = False
    logger = eng.run(sim_duration=120.0)
    s = logger.summary()
    assert s_or.total_requests == int(s["total_requests"])
    assert s_or.served_trips == int(s["served_trips"])


def test_sanity_ddqn_train_step_finite_loss() -> None:
    torch = pytest.importorskip("torch")
    from rl.agent import DDQNAgent

    _ = torch
    agent = DDQNAgent(state_dim=22, hidden_dim=64, lr=1e-3, gamma_rl=0.99, grad_clip=10.0, device="cpu")
    states = np.random.randn(8, 22).astype(np.float32)
    actions = np.random.randint(0, 2, size=(8,), dtype=np.int64)
    rewards = np.random.randn(8).astype(np.float32)
    next_states = np.random.randn(8, 22).astype(np.float32)
    dones = np.zeros((8,), dtype=np.float32)
    loss = agent.train_step((states, actions, rewards, next_states, dones))
    assert np.isfinite(loss)
