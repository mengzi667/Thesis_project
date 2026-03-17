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
    RELOCATION_ACCEPTANCE_MODE,
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

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        acceptance_mode: str = RELOCATION_ACCEPTANCE_MODE,
    ) -> None:
        self.rng = rng or random.Random()
        mode = str(acceptance_mode).strip().lower()
        if mode not in {"deterministic", "stochastic"}:
            raise ValueError(
                "acceptance_mode must be 'deterministic' or 'stochastic', "
                f"got: {acceptance_mode!r}"
            )
        self.acceptance_mode = mode

    # ── Relocation acceptance ─────────────────────────────────────────────────

    def accept_relocation(
        self,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer: float = 0.0,
        walk_base: float = 0.0,
        rho0_offer: float = 0.0,
        rho0_base: float = 0.0,
        rho1_offer: float = 0.0,
        rho1_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """
        Return True if the user accepts the recommended alternative drop-off zone.

        Decision rule is configurable:
          - deterministic: accept iff P_offer > P_base
          - stochastic   : sample Bernoulli(P_offer)

        Utility of both alternatives is computed explicitly:
          - offer: accept recommended destination
          - base : keep original destination
        """
        _ = user_type  # reserved for future type-specific coefficients

        walk_offer_n = walk_offer / max(WALKING_THRESHOLD, 1.0)
        walk_base_n = walk_base / max(WALKING_THRESHOLD, 1.0)
        duration_n = trip_duration / 60.0

        # Shared coefficient scaffold so both alternatives are represented symmetrically.
        beta_rho0 = -0.30
        beta_rho1 = 0.35
        beta_battery = 0.25
        beta_duration = -0.08

        v_offer = (
            BASE_RELOC_UTILITY
            + BETA_INCENTIVE * incentive_amount
            + BETA_EXTRA_WALK * walk_offer_n
            + beta_rho0 * rho0_offer
            + beta_rho1 * rho1_offer
            + beta_duration * duration_n
            + beta_battery * battery_offer
        )
        v_base = (
            BETA_EXTRA_WALK * walk_base_n
            + beta_rho0 * rho0_base
            + beta_rho1 * rho1_base
            + beta_duration * duration_n
            + beta_battery * battery_base
        )

        p_offer, p_base = _softmax([v_offer, v_base])
        if self.acceptance_mode == "stochastic":
            return self.rng.random() < p_offer
        return p_offer > p_base
