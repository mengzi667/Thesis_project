# user_choice_model.py
# Discrete-choice models for two user sub-decisions:
#   1. Scooter selection  (high / low / opt-out)
#   2. Relocation acceptance  (accept recommended target / reject)
#
# This module is intentionally independent of the OR interface and any future
# RL agent.  The only inputs are observable trip/scooter attributes.

from __future__ import annotations

import math
import random
from typing import List, Optional

from config import (
    BETA_BATTERY,
    BETA_WALKING,
    BETA_PRICE,
    OPT_OUT_UTILITY,
    BETA_INCENTIVE,
    BETA_EXTRA_WALK,
    BASE_RELOC_UTILITY,
    RELOCATION_INCENTIVE,
    WALKING_THRESHOLD,
)
from fleet_manager import Scooter, BatteryCategory


# ── Utility helpers ───────────────────────────────────────────────────────────

def _softmax(utilities: List[float]) -> List[float]:
    """Numerically stable softmax over a list of utility values."""
    max_u = max(utilities)
    exp_u = [math.exp(u - max_u) for u in utilities]
    total = sum(exp_u)
    return [e / total for e in exp_u]


# ── UserChoiceModel ───────────────────────────────────────────────────────────

class UserChoiceModel:
    """
    Models user decision-making via a Multinomial Logit (MNL) framework.

    Parameters are set in config.py; the model itself is stateless across trips
    (all randomness is channelled through self.rng for reproducibility).
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    # ── Sub-decision 1: Scooter selection ────────────────────────────────────

    def choose_scooter(
        self,
        available_scooters: List[Scooter],
        walking_distances: Optional[List[float]] = None,
        user_type: str = "normal",
    ) -> Optional[Scooter]:
        """
        Return the chosen Scooter or None (opt-out / no scooter selected).

        Parameters
        ----------
        available_scooters : scooters at the origin zone, high-battery-first order.
        walking_distances  : metres to each scooter (parallel list).  If None,
                             all walking distances are assumed zero.
        user_type          : currently used as a label; extend for type-specific β.
        """
        if not available_scooters:
            return None

        if walking_distances is None:
            walking_distances = [0.0] * len(available_scooters)

        utilities: List[float] = []
        for scooter, walk_m in zip(available_scooters, walking_distances):
            u_battery = BETA_BATTERY * scooter.battery_level
            u_walk    = BETA_WALKING * (walk_m / max(WALKING_THRESHOLD, 1.0))
            # High-battery scooters carry a small price premium
            price_premium = 0.10 if scooter.battery_category == BatteryCategory.HIGH else 0.0
            u_price   = BETA_PRICE * price_premium
            utilities.append(u_battery + u_walk + u_price)

        # Add opt-out alternative as the last option
        all_utilities = utilities + [OPT_OUT_UTILITY]
        probs = _softmax(all_utilities)

        choices: List[Optional[Scooter]] = list(available_scooters) + [None]
        idx = self.rng.choices(range(len(choices)), weights=probs, k=1)[0]
        return choices[idx]

    # ── Sub-decision 2: Relocation acceptance ────────────────────────────────

    def accept_relocation(
        self,
        incentive_amount: float = RELOCATION_INCENTIVE,
        extra_walking_meters: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """
        Return True if the user accepts the recommended alternative drop-off zone.

        Parameters
        ----------
        incentive_amount     : monetary reward offered for accepting relocation.
        extra_walking_meters : additional distance to the recommended zone vs.
                               the original destination.
        user_type            : label for future type-specific parameterisation.
        """
        u_accept = (
            BASE_RELOC_UTILITY
            + BETA_INCENTIVE  * incentive_amount
            + BETA_EXTRA_WALK * (extra_walking_meters / max(WALKING_THRESHOLD, 1.0))
        )
        u_reject = 0.0   # reject is the baseline alternative

        probs = _softmax([u_accept, u_reject])
        return self.rng.choices([True, False], weights=probs, k=1)[0]
