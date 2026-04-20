# user_choice_model.py
# Single-layer user behavioural model for Scenario 1 (Sara-aligned utility).
#
# Decision set:
#   - has_offer=True  -> {offer, base, opt_out}
#   - has_offer=False -> {base, opt_out}
#
# Utility is computed in real time and converted via softmax probabilities.
# Final action is sampled stochastically from that distribution.

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
    SARA_RANGE_PER_PCT_KM,
    SARA_RIDE_PRICE_PER_MIN,
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
    """Single-layer user behaviour model."""

    ACTIONS_WITH_OFFER = ("offer", "base", "opt_out")
    ACTIONS_NO_OFFER = ("base", "opt_out")

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        first_layer_mode: str = FIRST_LAYER_MODE,
        acceptance_mode: str = RELOCATION_ACCEPTANCE_MODE,
    ) -> None:
        self.rng = rng or random.Random()
        # Keep old config knobs for compatibility, but single-layer decision
        # always uses real-time utility + stochastic sampling.
        self.first_layer_mode = str(first_layer_mode).strip().lower()
        self.acceptance_mode = str(acceptance_mode).strip().lower()

    # ── Utility primitives ───────────────────────────────────────────────────

    def _base_common_utility(self, walk_base: float, ride_fee_term: float) -> float:
        asc_share = 0.5 * SARA_BETA_ES
        return (
            asc_share
            + SARA_BETA_WALK * max(0.0, walk_base)
            + SARA_BETA_UNLOCK * 1.0
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
        self,
        walk_offer_min: float,
        rt_offer_min: float,
        incentive_amount: float,
    ) -> float:
        # No floor-clipping for (ride_price - incentive), consistent with
        # Sara-style direct utility comparison.
        ride_fee_offer = (
            SARA_RIDE_PRICE_PER_MIN * max(0.0, rt_offer_min) - max(0.0, incentive_amount)
        )
        asc_share = 0.5 * SARA_BETA_ES
        return (
            asc_share
            + SARA_BETA_WALK * max(0.0, walk_offer_min)
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

    # ── Single-layer choice ──────────────────────────────────────────────────

    def choice_probabilities(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> Dict[str, float]:
        """Return action probabilities over {'offer','base','opt_out'}."""
        _ = user_type

        v_base = self._base_common_utility(
            walk_base=walk_base_min,
            ride_fee_term=SARA_RIDE_PRICE_PER_MIN * max(0.0, rt_base_min),
        ) + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * max(0.0, battery_base))
        v_out = 0.0

        if has_offer:
            v_offer = self._offer_common_utility(
                walk_offer_min=walk_offer_min,
                rt_offer_min=rt_offer_min,
                incentive_amount=incentive_amount,
            ) + SARA_BETA_BATT * (SARA_RANGE_PER_PCT_KM * max(0.0, battery_offer))
            p_offer, p_base, p_out = _softmax([v_offer, v_base, v_out])
            return {"offer": p_offer, "base": p_base, "opt_out": p_out}

        p_base, p_out = _softmax([v_base, v_out])
        return {"offer": 0.0, "base": p_base, "opt_out": p_out}

    def decide_trip_action(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> str:
        """Sample one action from the single-layer choice distribution."""
        probs = self.choice_probabilities(
            has_offer=has_offer,
            incentive_amount=incentive_amount,
            walk_offer_min=walk_offer_min,
            walk_base_min=walk_base_min,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        actions = self.ACTIONS_WITH_OFFER if has_offer else self.ACTIONS_NO_OFFER
        x = self.rng.random()
        csum = 0.0
        for a in actions:
            csum += float(probs[a])
            if x <= csum:
                return a
        return actions[-1]

    # ── Backward-compatible helpers ──────────────────────────────────────────

    def participation_probabilities(
        self,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_high: float = 0.0,
        battery_low: float = 0.0,
        user_type: str = "normal",
    ) -> Dict[str, float]:
        """Legacy helper: derive {'ride','opt_out'} from no-offer mode."""
        _ = battery_high
        probs = self.choice_probabilities(
            has_offer=False,
            walk_base_min=walk_base,
            rt_base_min=trip_duration,
            battery_base=battery_low,
            user_type=user_type,
        )
        return {"ride": float(probs["base"]), "opt_out": float(probs["opt_out"])}

    def decide_participation(
        self,
        walk_base: float = 0.0,
        trip_duration: float = 0.0,
        battery_high: float = 0.0,
        battery_low: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """Legacy helper: True iff action is not opt_out in no-offer mode."""
        probs = self.participation_probabilities(
            walk_base=walk_base,
            trip_duration=trip_duration,
            battery_high=battery_high,
            battery_low=battery_low,
            user_type=user_type,
        )
        return self.rng.random() < probs["ride"]

    def acceptance_probabilities(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> Dict[str, float]:
        """Legacy helper: conditional {'offer','base'} probabilities."""
        _ = user_type
        if not has_offer:
            return {"offer": 0.0, "base": 1.0}
        probs = self.choice_probabilities(
            has_offer=True,
            incentive_amount=incentive_amount,
            walk_offer_min=walk_offer_min,
            walk_base_min=walk_base_min,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        denom = max(1e-12, float(probs["offer"]) + float(probs["base"]))
        return {
            "offer": float(probs["offer"]) / denom,
            "base": float(probs["base"]) / denom,
        }

    def decide_offer_acceptance(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """Legacy helper: True iff offer accepted."""
        probs = self.acceptance_probabilities(
            has_offer=has_offer,
            incentive_amount=incentive_amount,
            walk_offer_min=walk_offer_min,
            walk_base_min=walk_base_min,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
        if not has_offer:
            return False
        if self.acceptance_mode == "deterministic":
            return probs["offer"] >= probs["base"]
        return self.rng.random() < probs["offer"]

    def choose_relocation_action(
        self,
        has_offer: bool,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rho0_offer: float = 0.0,
        rho0_base: float = 0.0,
        rho1_offer: float = 0.0,
        rho1_base: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> str:
        """Backward-compatible adapter returning offer/base/opt_out."""
        _ = (rho0_offer, rho0_base, rho1_offer, rho1_base)
        return self.decide_trip_action(
            has_offer=has_offer,
            incentive_amount=incentive_amount,
            walk_offer_min=walk_offer_min,
            walk_base_min=walk_base_min,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )

    def accept_relocation(
        self,
        incentive_amount: float = RELOCATION_INCENTIVE,
        walk_offer_min: float = 0.0,
        walk_base_min: float = 0.0,
        rho0_offer: float = 0.0,
        rho0_base: float = 0.0,
        rho1_offer: float = 0.0,
        rho1_base: float = 0.0,
        rt_offer_min: float = 0.0,
        rt_base_min: float = 0.0,
        battery_offer: float = 0.0,
        battery_base: float = 0.0,
        user_type: str = "normal",
    ) -> bool:
        """Backward-compatible helper: True iff offer accepted."""
        _ = (rho0_offer, rho0_base, rho1_offer, rho1_base)
        return self.decide_offer_acceptance(
            has_offer=True,
            incentive_amount=incentive_amount,
            walk_offer_min=walk_offer_min,
            walk_base_min=walk_base_min,
            rt_offer_min=rt_offer_min,
            rt_base_min=rt_base_min,
            battery_offer=battery_offer,
            battery_base=battery_base,
            user_type=user_type,
        )
