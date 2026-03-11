# user_choice_model.py
# User behavioural model — relocation acceptance only.
#
# Design assumption (PRD §14):
#   Scooter selection is a deterministic system rule (highest-battery first);
#   it is NOT a user choice and this module plays no part in it.
#
# This module's sole responsibility is:
#   accept_relocation() — MNL model for whether a user accepts an OR / RL
#                          drop-off recommendation.
#
# It is intentionally independent of the OR interface, fleet state, and any
# future RL agent.

from __future__ import annotations

import math
import random
from typing import List, Optional

from config import (
    BETA_INCENTIVE,
    BETA_EXTRA_WALK,
    BASE_RELOC_UTILITY,
    RELOCATION_INCENTIVE,
    WALKING_THRESHOLD,
)


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
    MNL behavioural model for relocation acceptance only.

    Scooter selection is NOT handled here — that is a deterministic system
    rule (highest-battery first, PRD §14) owned by the simulation engine.

    This class is used solely to decide whether a user accepts an OR / RL
    drop-off recommendation via accept_relocation().
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    # ── Relocation acceptance ─────────────────────────────────────────────────

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
