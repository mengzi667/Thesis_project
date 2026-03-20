# user_choice_model.py
# Two-layer user behavioural model for Scenario 1 (Sara-aligned structure).
#
# Layer 1: participation decision (ride vs opt_out)
# Layer 2: conditional offer acceptance (accept_offer vs reject_offer)
#
# Scooter assignment remains a deterministic system rule in SimulationEngine
# (highest-battery-first) and is intentionally outside this module.

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from config import (
    FIRST_LAYER_MODE,
    RELOCATION_ACCEPTANCE_MODE,
    RELOCATION_INCENTIVE,
    SARA_BETA_ALONE,
    SARA_BETA_BATT,
    SARA_BETA_BIKE,
    SARA_BETA_ES,
    SARA_BETA_INCOME,
    SARA_BETA_PREV,
    SARA_BETA_RIDE,
    SARA_BETA_SHARED,
    SARA_BETA_TYPE,
    SARA_BETA_UNLOCK,
    SARA_BETA_WALK,
    SARA_ETA_ATT,
    SARA_ETA_RANGE,
    SARA_FIRST_LAYER_RIDE_FEE_TERM,
    SARA_FIRST_LAYER_UNLOCK_FEE,
    SARA_FIRST_LAYER_WALK_MIN,
    SARA_FIRST_LAYER_PCT_HIGH,
    SARA_FIRST_LAYER_PCT_LOW,
    SARA_PROB_OUT,
    SARA_RANGE_PER_PCT_KM,
    SARA_USER_ATTITUDE,
    SARA_USER_BIKE,
    SARA_USER_INCOME_LOW,
    SARA_USER_LIVING_ALONE,
    SARA_USER_LIVING_SHARED,
    SARA_USER_PREVIOUS_USE,
    SARA_USER_RANGE_ANXIETY,
    SARA_USER_VEHICLE_TYPE_25,
)


def _softmax(utilities: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_u = max(utilities)
    exp_u = [math.exp(u - max_u) for u in utilities]
    total = sum(exp_u)
    return [e / total for e in exp_u]


class UserChoiceModel:
    """
    Two-layer user behaviour model.

    Layer 1 (participation):
      - aggregated_prob: fixed P(opt_out)=SARA_PROB_OUT
      - realtime_choice: P(ride)=P(high)+P(low) from Sara-style utilities

    Layer 2 (acceptance):
      - conditional on ride=1
      - binary choice between offer and base
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        first_layer_mode: str = FIRST_LAYER_MODE,
        acceptance_mode: str = RELOCATION_ACCEPTANCE_MODE,
    ) -> None:
        self.rng = rng or random.Random()

        flm = str(first_layer_mode).strip().lower()
        if flm not in {"aggregated_prob", "realtime_choice"}:
            raise ValueError(
                "first_layer_mode must be 'aggregated_prob' or 'realtime_choice', "
                f"got: {first_layer_mode!r}"
            )
        self.first_layer_mode = flm

        am = str(acceptance_mode).strip().lower()
        if am not in {"deterministic", "stochastic"}:
            raise ValueError(
                "acceptance_mode must be 'deterministic' or 'stochastic', "
                f"got: {acceptance_mode!r}"
            )
        self.acceptance_mode = am

    # ── Utility primitives ───────────────────────────────────────────────────

    def _base_common_utility(self, walk_base: float, ride_fee_term: float) -> float:
        asc_share = 0.5 * SARA_BETA_ES
        return (
            asc_share
            + SARA_BETA_WALK * max(0.0, walk_base)
            + SARA_BETA_UNLOCK * SARA_FIRST_LAYER_UNLOCK_FEE
            + SARA_BETA_RIDE * max(0.0, ride_fee_term)
            + SARA_BETA_TYPE * SARA_USER_VEHICLE_TYPE_25
            + SARA_BETA_PREV * SARA_USER_PREVIOUS_USE
            + SARA_BETA_BIKE * SARA_USER_BIKE
            + SARA_BETA_INCOME * SARA_USER_INCOME_LOW
            + SARA_BETA_ALONE * SARA_USER_LIVING_ALONE
            + SARA_BETA_SHARED * SARA_USER_LIVING_SHARED
            + SARA_ETA_ATT * SARA_USER_ATTITUDE
            + SARA_ETA_RANGE * SARA_USER_RANGE_ANXIETY
        )

    def _offer_common_utility(
        self, walk_offer: float, trip_duration: float, incentive_amount: float
    ) -> float:
        ride_fee_base = 0.30 * max(0.0, trip_duration)
        ride_fee_offer = max(0.0, ride_fee_base - max(0.0, incentive_amount))
        asc_share = 0.5 * SARA_BETA_ES
        return (
            asc_share
            + SARA_BETA_WALK * max(0.0, walk_offer)
            + SARA_BETA_UNLOCK * 1.0
            + SARA_BETA_RIDE * ride_fee_offer
            + SARA_BETA_TYPE * SARA_USER_VEHICLE_TYPE_25
            + SARA_BETA_PREV * SARA_USER_PREVIOUS_USE
            + SARA_BETA_BIKE * SARA_USER_BIKE
            + SARA_BETA_INCOME * SARA_USER_INCOME_LOW
            + SARA_BETA_ALONE * SARA_USER_LIVING_ALONE
            + SARA_BETA_SHARED * SARA_USER_LIVING_SHARED
            + SARA_ETA_ATT * SARA_USER_ATTITUDE
            + SARA_ETA_RANGE * SARA_USER_RANGE_ANXIETY
        )

    # ── Layer 1: participation (ride vs opt_out) ────────────────────────────

    def participation_probabilities(
        self,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_high: float = 0.0,
        battery_low: float = 0.0,
        user_type: str = "normal",
    ) -> Dict[str, float]:
        """
        Return Layer 1 probabilities: {'ride': ..., 'opt_out': ...}.
        """
        _ = user_type

        if self.first_layer_mode == "aggregated_prob":
            p_opt = min(1.0, max(0.0, float(SARA_PROB_OUT)))
            return {"ride": 1.0 - p_opt, "opt_out": p_opt}

        # Match Sara compute_probs_for_class defaults:
        # walk_min=2.0, unlock_fee=1.0, ride_fee_term=0.30.
        walk_m = float(SARA_FIRST_LAYER_WALK_MIN if walk_base == 0.0 else walk_base)
        ride_fee_term = float(SARA_FIRST_LAYER_RIDE_FEE_TERM)
        common = self._base_common_utility(
            walk_base=walk_m,
            ride_fee_term=ride_fee_term,
        )
        # Strict Sara default input setting for first-layer realtime model.
        # Keep method arguments for interface compatibility, but do not use
        # request-specific battery values in this mode.
        _ = (battery_high, battery_low)
        pct_high = max(0.0, float(SARA_FIRST_LAYER_PCT_HIGH))
        pct_low = max(0.0, float(SARA_FIRST_LAYER_PCT_LOW))

        v_high = common + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * pct_high)
        v_low = common + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * pct_low)
        v_out = 0.0
        p_high, p_low, p_out = _softmax([v_high, v_low, v_out])
        return {"ride": p_high + p_low, "opt_out": p_out}

    def decide_participation(
        self,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_high: float = 0.0,
        battery_low: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """Return True iff user participates (ride)."""
        probs = self.participation_probabilities(
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_high=battery_high,
            battery_low=battery_low,
            user_type=user_type,
        )
        p_ride = probs["ride"]
        return self.rng.random() < p_ride

    # ── Layer 2: conditional acceptance (offer vs base) ─────────────────────

    def acceptance_probabilities(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer: float = 0.0,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> Dict[str, float]:
        """
        Return Layer 2 probabilities over {'offer', 'base'}.
        When has_offer=False, returns {'offer': 0.0, 'base': 1.0}.
        """
        _ = user_type
        if not has_offer:
            return {"offer": 0.0, "base": 1.0}

        v_offer = self._offer_common_utility(
            walk_offer=walk_offer,
            trip_duration=trip_duration,
            incentive_amount=incentive_amount,
        ) + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * max(0.0, battery_offer))
        v_base = self._base_common_utility(
            walk_base=walk_base,
            ride_fee_term=0.30 * max(0.0, trip_duration),
        ) + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * max(0.0, battery_base))

        p_offer, p_base = _softmax([v_offer, v_base])
        return {"offer": p_offer, "base": p_base}

    def decide_offer_acceptance(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer: float = 0.0,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """Return True iff offer is accepted in Layer 2."""
        probs = self.acceptance_probabilities(
            has_offer=has_offer,
            incentive_amount=incentive_amount,
            walk_offer=walk_offer,
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        if not has_offer:
            return False
        if self.acceptance_mode == "stochastic":
            return self.rng.random() < probs["offer"]
        return probs["offer"] >= probs["base"]

    # ── Backward compatibility ───────────────────────────────────────────────

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
        Legacy adapter returning 'offer'/'base'/'opt_out' from two-layer logic.
        """
        _ = (rho0_offer, rho0_base, rho1_offer, rho1_base)
        ride = self.decide_participation(
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_high=battery_base,
            battery_low=battery_base,
            user_type=user_type,
        )
        if not ride:
            return "opt_out"
        accepted = self.decide_offer_acceptance(
            has_offer=has_offer,
            incentive_amount=incentive_amount,
            walk_offer=walk_offer,
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        return "offer" if accepted else "base"

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
        """Backward-compatible helper: True iff offer accepted."""
        _ = (rho0_offer, rho0_base, rho1_offer, rho1_base)
        return self.decide_offer_acceptance(
            has_offer=True,
            incentive_amount=incentive_amount,
            walk_offer=walk_offer,
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
