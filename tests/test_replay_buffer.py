from __future__ import annotations

import numpy as np

from rl.replay_buffer import ReplayBuffer, Transition


def _mk(i: int) -> Transition:
    s = np.full((22,), float(i), dtype=np.float32)
    ns = np.full((22,), float(i + 1), dtype=np.float32)
    return Transition(
        state=s,
        action=i % 2,
        reward=float(i),
        next_state=ns,
        done=0.0,
        info={"idx": i},
    )


def test_replay_push_and_len() -> None:
    rb = ReplayBuffer(capacity=3)
    rb.push(_mk(0))
    rb.push(_mk(1))
    assert len(rb) == 2


def test_replay_capacity_rollover() -> None:
    rb = ReplayBuffer(capacity=2)
    rb.push(_mk(0))
    rb.push(_mk(1))
    rb.push(_mk(2))
    assert len(rb) == 2
    infos = rb.dump_infos()
    idxs = [x["idx"] for x in infos]
    assert idxs == [1, 2]


def test_replay_sample_shapes() -> None:
    rb = ReplayBuffer(capacity=10)
    for i in range(6):
        rb.push(_mk(i))
    batch = rb.sample(batch_size=4, rng=np.random.default_rng(7))
    states, actions, rewards, next_states, dones = batch
    assert states.shape == (4, 22)
    assert next_states.shape == (4, 22)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
