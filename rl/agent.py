from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DDQNStats:
    updates: int = 0
    mean_loss: float = 0.0


class DDQNAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        lr: float,
        gamma_rl: float,
        grad_clip: float,
        device: str = "gpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma_rl = float(gamma_rl)
        self.grad_clip = float(grad_clip)

        self.q_online = QNetwork(state_dim, hidden_dim).to(self.device)
        self.q_target = QNetwork(state_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.stats = DDQNStats()

    def act(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, 2))
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_online(s)
            return int(torch.argmax(q, dim=1).item())

    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch

        s = torch.from_numpy(states).float().to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        r = torch.from_numpy(rewards).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        d = torch.from_numpy(dones).float().to(self.device)

        q_values = self.q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.q_online(ns), dim=1, keepdim=True)
            next_q = self.q_target(ns).gather(1, next_actions).squeeze(1)
            target = r + (1.0 - d) * self.gamma_rl * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), self.grad_clip)
        self.optimizer.step()

        self.stats.updates += 1
        # Online mean update
        self.stats.mean_loss += (float(loss.item()) - self.stats.mean_loss) / self.stats.updates
        return float(loss.item())

    def sync_target(self) -> None:
        self.q_target.load_state_dict(self.q_online.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_online": self.q_online.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "stats": {"updates": self.stats.updates, "mean_loss": self.stats.mean_loss},
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(ckpt["q_online"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        st = ckpt.get("stats", {})
        self.stats.updates = int(st.get("updates", 0))
        self.stats.mean_loss = float(st.get("mean_loss", 0.0))
