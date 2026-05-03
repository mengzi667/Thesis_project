from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class RLPolicy(Protocol):
    def act(self, state: np.ndarray) -> int: ...


@dataclass
class DecisionContext:
    request_time: float
    planning_period: float
    episode_minutes: float
    origin: int
    destination: int
    recommended: int
    rt_base_min: float
    rt_offer_min: float
    walk_extra_min: float
    incentive_amount: float
    offered: bool
    accepted: bool
    rejected: bool
    quota_remaining: float
    budget_remaining: float
    # zone states at decision time
    zone_state_before: Dict[int, tuple]
    zone_state_after: Dict[int, tuple]


class TransitionLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.trip_rows: List[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_csv(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(p, index=False)


class Scenario1FeatureBuilder:
    def __init__(
        self,
        zone_ids: List[int],
        zone_capacity: float,
        planning_period: float,
        episode_minutes: float,
    ) -> None:
        self.zone_ids = sorted(zone_ids)
        self.zone_capacity = max(1.0, float(zone_capacity))
        self.planning_period = float(planning_period)
        self.episode_minutes = max(float(episode_minutes), 1.0)
        self.zone_id_max = max(self.zone_ids) if self.zone_ids else 1

    @staticmethod
    def _rentable_count(state_tuple: tuple) -> float:
        return float(state_tuple[1] + state_tuple[2])

    def build(
        self,
        ctx: DecisionContext,
        edl_before: Dict[int, float],
    ) -> np.ndarray:
        def _zone_norm(z: int) -> float:
            return float(z) / float(self.zone_id_max)

        def _state_triplet(zone: int) -> tuple:
            v = ctx.zone_state_before.get(zone, (0, 0, 0))
            return (
                float(v[0]) / self.zone_capacity,
                float(v[1]) / self.zone_capacity,
                float(v[2]) / self.zone_capacity,
            )

        o = ctx.origin
        d = ctx.destination
        i = ctx.recommended
        o_n, o_l, o_h = _state_triplet(o)
        d_n, d_l, d_h = _state_triplet(d)
        i_n, i_l, i_h = _state_triplet(i)

        slot_idx = int(ctx.request_time // self.planning_period)
        slot_norm = (slot_idx * self.planning_period) / self.episode_minutes

        feat = np.asarray(
            [
                slot_norm,
                _zone_norm(o),
                _zone_norm(d),
                _zone_norm(i),
                o_n,
                o_l,
                o_h,
                d_n,
                d_l,
                d_h,
                i_n,
                i_l,
                i_h,
                float(edl_before.get(o, 0.0)),
                float(edl_before.get(d, 0.0)),
                float(edl_before.get(i, 0.0)),
                float(ctx.rt_base_min),
                float(ctx.rt_offer_min),
                float(ctx.walk_extra_min),
                float(ctx.incentive_amount),
                float(max(0.0, ctx.quota_remaining)),
                float(max(0.0, ctx.budget_remaining)),
            ],
            dtype=np.float32,
        )
        return feat


def estimate_zone_edl(
    zone_id: int,
    slot_idx: int,
    zone_state: Dict[int, tuple],
    planning_period: float,
    demand_profile=None,
    zone_ids: Optional[List[int]] = None,
    edl_model=None,
) -> float:
    s = zone_state.get(zone_id, (0, 0, 0))
    if edl_model is not None:
        return float(edl_model.station_edl_total(zone_id, slot_idx, s))
    if demand_profile is None or zone_ids is None:
        return 0.0
    expected_rate = float(demand_profile.dest_arrival_rate(zone_id, slot_idx, zone_ids))
    expected_arrivals = expected_rate * float(planning_period)
    rentable = float(s[1] + s[2])
    return max(0.0, expected_arrivals - rentable)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def reward_hybrid(
    w_l: float,
    w_e: float,
    beta_a: float,
    beta_c: float,
    beta_r: float,
    l_ref: float,
    e_ref: float,
    realized_loss: float,
    delta_edl: float,
    cost_term: float,
    accept_flag: bool,
    reject_flag: bool,
) -> float:
    """
    Hybrid reward with independently weighted primary terms.

    Normalization anchors:
      - l_ref: scale for realized-loss term
      - e_ref: scale for delta-EDL term
    They are numerical anchors (not physical constants) to keep terms
    comparable and stable across settings.
    """
    l_ref = max(1e-9, float(l_ref))
    e_ref = max(1e-9, float(e_ref))
    realized_norm = _clip(float(realized_loss) / l_ref, 0.0, 1.0)
    delta_edl_norm = _clip(float(delta_edl) / e_ref, -1.0, 1.0)
    return float(
        -float(w_l) * realized_norm
        + float(w_e) * delta_edl_norm
        + float(beta_a) * (1.0 if accept_flag else 0.0)
        - float(beta_c) * float(cost_term)
        - float(beta_r) * (1.0 if reject_flag else 0.0)
    )


# Backward-compatible alias retained for old tests/imports.
def reward_edl(
    alpha: float,
    beta: float,
    gamma: float,
    edl_before_d: float,
    edl_before_i: float,
    edl_after_d: float,
    edl_after_i: float,
    cost_term: float,
    reject_flag: bool,
) -> float:
    delta_edl = (edl_before_d + edl_before_i) - (edl_after_d + edl_after_i)
    return reward_hybrid(
        w_l=0.0,
        w_e=1.0,
        beta_a=0.0,
        beta_c=beta,
        beta_r=gamma,
        l_ref=1.0,
        e_ref=1.0,
        realized_loss=0.0,
        delta_edl=alpha * delta_edl,
        cost_term=cost_term,
        accept_flag=False,
        reject_flag=reject_flag,
    )


