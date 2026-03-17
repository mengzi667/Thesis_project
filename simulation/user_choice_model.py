# user_choice_model.py
# User behavioural model — labeled user decision with explicit opt-out.
#
# Design assumption (PRD §14):
#   Scooter selection is a deterministic system rule (highest-battery first);
#   it is NOT a user choice and this module plays no part in it.
#
# This module's sole responsibility is:
#   user decision over labeled alternatives with explicit opt-out:
#   offer / base / opt_out.
#
# It is intentionally independent of the OR interface, fleet state, and any
# future RL agent.

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

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
    MNL behavioural model for trip-level user decision with explicit opt-out.

    Scooter selection is NOT handled here — that is a deterministic system
    rule (highest-battery first, PRD §14) owned by the simulation engine.

    This class outputs one labeled action among:
      - offer
      - base
      - opt_out
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

    def choose_relocation_action(
        self,
        has_offer: bool,
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
    ) -> str:
        """
        Return one of: "offer", "base", "opt_out".

        Choice set:
          - has_offer=True  : {"offer", "base", "opt_out"}
          - has_offer=False : {"base", "opt_out"}

        Probabilities are computed by MNL over the active set.
        Opt-out utility is normalized to zero.
        """
        _ = user_type  # reserved for future type-specific coefficients

        walk_offer_n = walk_offer / max(WALKING_THRESHOLD, 1.0)
        walk_base_n = walk_base / max(WALKING_THRESHOLD, 1.0)
        duration_n = trip_duration / 60.0

        # Shared coefficient scaffold so both alternatives are represented symmetrically.
        beta_rho0 = -0.30
        beta_rho1 = 0.35
        # Placeholder: battery effect intentionally disabled for Scenario 1
        # until a validated battery attribute mapping is finalized.
        beta_battery = 0.0
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
        v_opt_out = 0.0

        if has_offer:
            actions = ["offer", "base", "opt_out"]
            probs = _softmax([v_offer, v_base, v_opt_out])
        else:
            actions = ["base", "opt_out"]
            probs = _softmax([v_base, v_opt_out])

        if self.acceptance_mode == "stochastic":
            return self.rng.choices(actions, weights=probs, k=1)[0]
        return actions[max(range(len(actions)), key=lambda i: probs[i])]

    def choice_probabilities(
        self,
        has_offer: bool,
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
    ) -> Dict[str, float]:
        """Return MNL probabilities for the active choice set."""
        _ = user_type

        walk_offer_n = walk_offer / max(WALKING_THRESHOLD, 1.0)
        walk_base_n = walk_base / max(WALKING_THRESHOLD, 1.0)
        duration_n = trip_duration / 60.0

        beta_rho0 = -0.30
        beta_rho1 = 0.35
        # Placeholder: battery effect intentionally disabled for Scenario 1
        # until a validated battery attribute mapping is finalized.
        beta_battery = 0.0
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
        v_opt_out = 0.0

        if has_offer:
            p_offer, p_base, p_opt_out = _softmax([v_offer, v_base, v_opt_out])
            return {"offer": p_offer, "base": p_base, "opt_out": p_opt_out}
        p_base, p_opt_out = _softmax([v_base, v_opt_out])
        return {"base": p_base, "opt_out": p_opt_out}

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
        Backward-compatible helper:
        return True iff chosen action is "offer".
        """
        action = self.choose_relocation_action(
            has_offer=True,
            incentive_amount=incentive_amount,
            walk_offer=walk_offer,
            walk_base=walk_base,
            rho0_offer=rho0_offer,
            rho0_base=rho0_base,
            rho1_offer=rho1_offer,
            rho1_base=rho1_base,
            trip_duration=trip_duration,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        return action == "offer"

