from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float
    info: Dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._buf) < batch_size:
            raise ValueError("not enough transitions to sample")

        idx = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[int(i)] for i in idx]

        states = np.stack([t.state for t in batch], axis=0)
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch], axis=0)
        dones = np.asarray([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def dump_infos(self) -> List[Dict[str, Any]]:
        return [t.info for t in self._buf]
