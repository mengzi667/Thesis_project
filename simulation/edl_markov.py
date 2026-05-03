from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, name)


def _enumerate_states(capacity: int) -> List[Tuple[int, int, int]]:
    return [
        (n, l, h)
        for n in range(capacity + 1)
        for l in range(capacity + 1)
        for h in range(capacity + 1)
        if n + l + h <= capacity
    ]


@dataclass
class SaraMarkovEDL:
    """
    Sara-aligned Markov EDL calculator for station-level EDL_total.

    Formula alignment:
      EDL = dt * ((prob_l*pr + prob_h*pr)*P_empty + (dr_low + dr_high)*P_full)
    where P_empty and P_full are derived from one-step propagated state
    distribution under the CTMC transition matrix.
    """

    station_ids: List[int]
    capacity: int
    dt: float
    slots_per_hour: int
    prob_l: float
    prob_h: float
    pickup_rates_by_hour: Dict[int, Dict[int, float]]
    dropoff_rates_by_hour: Dict[int, Dict[int, np.ndarray]]
    phi1_by_hour: Dict[int, float]
    phi2_by_hour: Dict[int, float]

    def __post_init__(self) -> None:
        self.station_ids = sorted(int(s) for s in self.station_ids)
        self._states: Dict[int, List[Tuple[int, int, int]]] = {}
        self._state_index: Dict[int, Dict[Tuple[int, int, int], int]] = {}
        self._empty_mask: Dict[int, np.ndarray] = {}
        self._full_mask: Dict[int, np.ndarray] = {}
        self._P_step: Dict[int, Dict[int, np.ndarray]] = {}

        for sid in self.station_ids:
            states = _enumerate_states(int(self.capacity))
            idx = {s: i for i, s in enumerate(states)}
            arr = np.asarray(states, dtype=float)
            self._states[sid] = states
            self._state_index[sid] = idx
            self._empty_mask[sid] = (arr[:, 1] == 0) & (arr[:, 2] == 0)
            self._full_mask[sid] = arr.sum(axis=1) == float(self.capacity)
            self._P_step[sid] = self._build_transition_mats_for_station(sid, states, idx)

    @classmethod
    def from_sara_csv(
        cls,
        data_dir: str,
        station_ids: List[int],
        capacity: int,
        is_weekend: int,
        dt: float = 0.25,
        slots_per_hour: int = 4,
        prob_l: float = 0.18,
        prob_h: float = 0.70,
    ) -> "SaraMarkovEDL":
        pickup = pd.read_csv(_path(data_dir, "30sep-df_pickup_rates.csv"))
        dropoff = pd.read_csv(_path(data_dir, "30sep-df_power_dropoff_rates.csv"))
        phi1_df = pd.read_csv(_path(data_dir, "30sep-df_phi1.csv"))
        phi2_df = pd.read_csv(_path(data_dir, "30sep-df_phi2.csv"))

        for df in (pickup, dropoff, phi1_df, phi2_df):
            for c in list(df.columns):
                if str(c).startswith("Unnamed:"):
                    df.drop(columns=[c], inplace=True)

        pickup = pickup[pickup["is_weekend"].astype(int) == int(is_weekend)].copy()
        dropoff = dropoff[dropoff["is_weekend"].astype(int) == int(is_weekend)].copy()

        hour_cols_pick = [int(c) for c in pickup.columns if str(c).isdigit() and 0 <= int(c) <= 23]
        hour_cols_drop = [int(c) for c in dropoff.columns if str(c).isdigit() and 0 <= int(c) <= 23]

        pickup_rates_by_hour: Dict[int, Dict[int, float]] = {}
        for _, row in pickup.iterrows():
            sid = int(row["start_station"])
            pickup_rates_by_hour[sid] = {h: float(row[str(h)] if str(h) in row else row[h]) for h in hour_cols_pick}

        cls_order = ["inactive", "low", "high"]
        dropoff_rates_by_hour: Dict[int, Dict[int, np.ndarray]] = {}
        for sid in station_ids:
            dropoff_rates_by_hour[int(sid)] = {h: np.zeros(3, dtype=float) for h in range(24)}

        for _, row in dropoff.iterrows():
            sid = int(row["end_station"])
            pcls = str(row["end_power_class"]).strip().lower()
            if sid not in dropoff_rates_by_hour or pcls not in cls_order:
                continue
            cidx = cls_order.index(pcls)
            for h in hour_cols_drop:
                val = float(row[str(h)] if str(h) in row else row[h])
                dropoff_rates_by_hour[sid][h][cidx] = val

        # Sara uses phi keyed by weekend flag columns: "0"/"1"
        wcol = str(int(is_weekend))
        if wcol not in phi1_df.columns or wcol not in phi2_df.columns:
            raise ValueError(f"phi files missing weekend column {wcol}")
        phi1_by_hour = {int(i): float(v) for i, v in enumerate(phi1_df[wcol].tolist())}
        phi2_by_hour = {int(i): float(v) for i, v in enumerate(phi2_df[wcol].tolist())}

        return cls(
            station_ids=[int(s) for s in station_ids],
            capacity=int(capacity),
            dt=float(dt),
            slots_per_hour=int(slots_per_hour),
            prob_l=float(prob_l),
            prob_h=float(prob_h),
            pickup_rates_by_hour=pickup_rates_by_hour,
            dropoff_rates_by_hour=dropoff_rates_by_hour,
            phi1_by_hour=phi1_by_hour,
            phi2_by_hour=phi2_by_hour,
        )

    def _build_transition_mats_for_station(
        self,
        sid: int,
        states: List[Tuple[int, int, int]],
        idx: Dict[Tuple[int, int, int], int],
    ) -> Dict[int, np.ndarray]:
        S = len(states)
        I = np.eye(S)
        out: Dict[int, np.ndarray] = {}

        for hr in range(24):
            pr = float(self.pickup_rates_by_hour.get(sid, {}).get(hr, 0.0))
            dr = np.asarray(self.dropoff_rates_by_hour.get(sid, {}).get(hr, np.zeros(3)), dtype=float)
            low_pickr = pr * self.prob_l
            high_pickr = pr * self.prob_h
            low_dropr = float(dr[1])
            high_dropr = float(dr[2])
            phi1 = float(self.phi1_by_hour.get(hr, 0.0))
            phi2 = float(self.phi2_by_hour.get(hr, 0.0))

            Q = np.zeros((S, S), dtype=float)
            for si, (n, l, h) in enumerate(states):
                total = n + l + h
                if h > 0:
                    Q[si, idx[(n, l, h - 1)]] = self.prob_h * high_pickr
                if l > 0:
                    Q[si, idx[(n, l - 1, h)]] = self.prob_l * low_pickr
                if total < self.capacity:
                    rate_low_drop = (1.0 - phi2) * low_dropr + phi1 * high_dropr
                    rate_high_drop = (1.0 - phi1) * high_dropr
                    rate_no_drop = phi2 * low_dropr
                    Q[si, idx[(n, l + 1, h)]] = max(0.0, rate_low_drop)
                    Q[si, idx[(n, l, h + 1)]] = max(0.0, rate_high_drop)
                    Q[si, idx[(n + 1, l, h)]] = max(0.0, rate_no_drop)

            np.fill_diagonal(Q, -Q.sum(axis=1))
            # Sara-style Taylor discretisation P ≈ (I + Q*dt/100)^100
            A = Q * self.dt
            out[hr] = np.linalg.matrix_power(I + A / 100.0, 100)

        return out

    def _nearest_state(self, sid: int, inv: Tuple[int, int, int]) -> Tuple[int, int, int]:
        n, l, h = (int(round(inv[0])), int(round(inv[1])), int(round(inv[2])))
        n = max(0, n)
        l = max(0, l)
        h = max(0, h)
        if n + l + h <= self.capacity and (n, l, h) in self._state_index[sid]:
            return (n, l, h)
        arr = np.asarray(self._states[sid], dtype=int)
        target = np.asarray([n, l, h], dtype=int)
        d = np.abs(arr - target).sum(axis=1)
        return tuple(arr[int(np.argmin(d))].tolist())

    def station_edl_total(self, sid: int, slot_idx: int, inv_state: Tuple[int, int, int]) -> float:
        """
        Compute Sara-aligned EDL_total for one slot, starting from given inventory state.
        """
        sid = int(sid)
        if sid not in self._state_index:
            return 0.0

        state = self._nearest_state(sid, inv_state)
        s_idx = self._state_index[sid][state]
        pi = np.zeros(len(self._states[sid]), dtype=float)
        pi[s_idx] = 1.0

        hr = int(min(max(slot_idx // self.slots_per_hour, 0), 23))
        P = self._P_step[sid][hr]
        pi = pi @ P

        p_empty = float(pi[self._empty_mask[sid]].sum())
        p_full = float(pi[self._full_mask[sid]].sum())

        pr = float(self.pickup_rates_by_hour.get(sid, {}).get(hr, 0.0))
        dr = np.asarray(self.dropoff_rates_by_hour.get(sid, {}).get(hr, np.zeros(3)), dtype=float)
        edl = self.dt * (((self.prob_l * pr + self.prob_h * pr) * p_empty) + ((float(dr[1]) + float(dr[2])) * p_full))
        return float(max(0.0, edl))

    def station_cumulative_edl(
        self,
        sid: int,
        t_intervention: int,
        t_end_exclusive: int,
        inv_state: Tuple[int, int, int],
    ) -> float:
        """
        Sara-aligned cumulative EDL from t_intervention to t_end_exclusive-1.
        Equivalent to compute_EDL_from_t_to_end(...) tail sum behavior.
        """
        sid = int(sid)
        t0 = int(max(0, t_intervention))
        t1 = int(max(t0, t_end_exclusive))
        if sid not in self._state_index or t1 <= t0:
            return 0.0

        state = self._nearest_state(sid, inv_state)
        s_idx = self._state_index[sid][state]
        pi = np.zeros(len(self._states[sid]), dtype=float)
        pi[s_idx] = 1.0

        total = 0.0
        for t in range(t0, t1):
            hr = int(min(max(t // self.slots_per_hour, 0), 23))
            P = self._P_step[sid][hr]
            pi = pi @ P
            p_empty = float(pi[self._empty_mask[sid]].sum())
            p_full = float(pi[self._full_mask[sid]].sum())
            pr = float(self.pickup_rates_by_hour.get(sid, {}).get(hr, 0.0))
            dr = np.asarray(self.dropoff_rates_by_hour.get(sid, {}).get(hr, np.zeros(3)), dtype=float)
            edl_step = self.dt * (
                ((self.prob_l * pr + self.prob_h * pr) * p_empty)
                + ((float(dr[1]) + float(dr[2])) * p_full)
            )
            total += float(max(0.0, edl_step))
        return float(total)
