from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


STATE_HIGH = "high"
STATE_LOW = "low"
STATE_INACTIVE = "inactive"
_STATES = (STATE_HIGH, STATE_LOW, STATE_INACTIVE)


def _normalize_row(row: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, row.get(s, 0.0)) for s in _STATES)
    if total <= 0.0:
        return {STATE_HIGH: 1.0, STATE_LOW: 0.0, STATE_INACTIVE: 0.0}
    return {s: max(0.0, row.get(s, 0.0)) / total for s in _STATES}


def _apply_high_to_inactive_policy(
    init_state: str,
    probs: Dict[str, float],
    policy: str,
) -> Dict[str, float]:
    if init_state != STATE_HIGH or policy != "strict-paper":
        return _normalize_row(probs)

    # Paper-consistent baseline: disallow direct high -> inactive.
    p_high = max(0.0, probs.get(STATE_HIGH, 0.0))
    p_low = max(0.0, probs.get(STATE_LOW, 0.0))
    s = p_high + p_low
    if s <= 0.0:
        return {STATE_HIGH: 1.0, STATE_LOW: 0.0, STATE_INACTIVE: 0.0}
    return {
        STATE_HIGH: p_high / s,
        STATE_LOW: p_low / s,
        STATE_INACTIVE: 0.0,
    }


@dataclass(frozen=True)
class TransitionContext:
    is_weekend: int
    hour: int


class BatteryTransitionModel:
    """
    CSV-based battery Markov transition model with fallback hierarchy:
      1) (is_weekend, hour, init_state)
      2) (is_weekend, init_state) aggregated across hours
      3) (init_state) global aggregated across all rows
    """

    def __init__(
        self,
        primary_probs: Dict[Tuple[int, int, str], Dict[str, float]],
        primary_n_from: Dict[Tuple[int, int, str], int],
        weekend_probs: Dict[Tuple[int, str], Dict[str, float]],
        global_probs: Dict[str, Dict[str, float]],
        high_to_inactive_policy: str = "strict-paper",
        min_primary_n_from: int = 30,
    ) -> None:
        self._primary_probs = primary_probs
        self._primary_n_from = primary_n_from
        self._weekend_probs = weekend_probs
        self._global_probs = global_probs
        self.high_to_inactive_policy = high_to_inactive_policy
        self.min_primary_n_from = min_primary_n_from

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        high_to_inactive_policy: str = "strict-paper",
        min_primary_n_from: int = 30,
    ) -> "BatteryTransitionModel":
        primary_raw: Dict[Tuple[int, int, str], Dict[str, float]] = {}
        primary_n_from: Dict[Tuple[int, int, str], int] = {}
        weekend_counts: Dict[Tuple[int, str], Dict[str, float]] = {}
        global_counts: Dict[str, Dict[str, float]] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = int(row["is_weekend"])
                h = int(row["hour"])
                init_state = str(row["init_power_class"]).strip().lower()
                end_state = str(row["end_power_class"]).strip().lower()
                n = float(row["n"])
                n_from = int(float(row["n_from"]))
                p = float(row["p"])

                if init_state not in _STATES or end_state not in _STATES:
                    continue

                key1 = (w, h, init_state)
                primary_raw.setdefault(key1, {s: 0.0 for s in _STATES})
                primary_raw[key1][end_state] = p
                primary_n_from[key1] = n_from

                key2 = (w, init_state)
                weekend_counts.setdefault(key2, {s: 0.0 for s in _STATES})
                weekend_counts[key2][end_state] += n

                global_counts.setdefault(init_state, {s: 0.0 for s in _STATES})
                global_counts[init_state][end_state] += n

        primary_probs: Dict[Tuple[int, int, str], Dict[str, float]] = {}
        for key, probs in primary_raw.items():
            init_state = key[2]
            primary_probs[key] = _apply_high_to_inactive_policy(
                init_state=init_state,
                probs=probs,
                policy=high_to_inactive_policy,
            )

        weekend_probs: Dict[Tuple[int, str], Dict[str, float]] = {}
        for key, counts in weekend_counts.items():
            init_state = key[1]
            weekend_probs[key] = _apply_high_to_inactive_policy(
                init_state=init_state,
                probs=_normalize_row(counts),
                policy=high_to_inactive_policy,
            )

        global_probs: Dict[str, Dict[str, float]] = {}
        for init_state, counts in global_counts.items():
            global_probs[init_state] = _apply_high_to_inactive_policy(
                init_state=init_state,
                probs=_normalize_row(counts),
                policy=high_to_inactive_policy,
            )

        # Ensure every state has a valid fallback row.
        for init_state in _STATES:
            if init_state not in global_probs:
                if init_state == STATE_INACTIVE:
                    global_probs[init_state] = {
                        STATE_HIGH: 0.0,
                        STATE_LOW: 0.0,
                        STATE_INACTIVE: 1.0,
                    }
                elif init_state == STATE_LOW:
                    global_probs[init_state] = {
                        STATE_HIGH: 0.0,
                        STATE_LOW: 0.7,
                        STATE_INACTIVE: 0.3,
                    }
                else:
                    global_probs[init_state] = {
                        STATE_HIGH: 0.9,
                        STATE_LOW: 0.1,
                        STATE_INACTIVE: 0.0,
                    }

        return cls(
            primary_probs=primary_probs,
            primary_n_from=primary_n_from,
            weekend_probs=weekend_probs,
            global_probs=global_probs,
            high_to_inactive_policy=high_to_inactive_policy,
            min_primary_n_from=min_primary_n_from,
        )

    def probs_for(self, init_state: str, context: TransitionContext) -> Dict[str, float]:
        init_state = init_state.lower()
        if init_state == STATE_INACTIVE:
            return {STATE_HIGH: 0.0, STATE_LOW: 0.0, STATE_INACTIVE: 1.0}

        key1 = (context.is_weekend, context.hour, init_state)
        n_from = self._primary_n_from.get(key1, 0)
        if n_from >= self.min_primary_n_from and key1 in self._primary_probs:
            return self._primary_probs[key1]

        key2 = (context.is_weekend, init_state)
        if key2 in self._weekend_probs:
            return self._weekend_probs[key2]

        return self._global_probs[init_state]

    def sample_next_state(self, init_state: str, context: TransitionContext, rng) -> str:
        probs = self.probs_for(init_state=init_state, context=context)
        return rng.choices(
            population=list(_STATES),
            weights=[probs[s] for s in _STATES],
            k=1,
        )[0]

    def validate(self) -> None:
        def _validate_rows(rows: Iterable[Dict[str, float]]) -> None:
            for row in rows:
                s = sum(row.get(state, 0.0) for state in _STATES)
                if abs(s - 1.0) > 1e-6:
                    raise ValueError(f"Battery transition row not normalized: sum={s}")

        _validate_rows(self._primary_probs.values())
        _validate_rows(self._weekend_probs.values())
        _validate_rows(self._global_probs.values())
