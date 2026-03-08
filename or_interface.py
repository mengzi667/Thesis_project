# or_interface.py
# OR model interface — placeholder implementation.
#
# Defines the relocation opportunity structure U_odit and the query interface
# used by the simulation engine.  The placeholder generates synthetic
# recommendations so the simulation loop can be tested before the real OR
# model is available.
#
# Integration contract (must remain stable):
#   ORInterface.query(origin, destination, request_time) -> Optional[RelocationOpportunity]
#
# When the real OR model is ready, replace only the body of `query()` (or
# subclass ORInterface) — the simulation loop requires no changes.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import RELOCATION_OFFER_PROB, RELOCATION_INCENTIVE


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RelocationOpportunity:
    """
    Represents a single OR recommendation:

        U_odit — if a trip (origin → original_dest) arrives at time_indicator,
                 the operator recommends the user drop off at recommended_dest.
    """
    origin:           int
    original_dest:    int
    recommended_dest: int
    time_indicator:   float
    incentive_amount: float = field(default=RELOCATION_INCENTIVE)

    def __repr__(self) -> str:
        return (
            f"RelocOpp(o={self.origin}→d={self.original_dest}"
            f"→i={self.recommended_dest}, t={self.time_indicator:.1f}, "
            f"inc={self.incentive_amount})"
        )


# ── Interface ─────────────────────────────────────────────────────────────────

class ORInterface:
    """
    Interface between the simulation engine and OR model outputs.

    Placeholder behaviour: returns a synthetic recommendation with probability
    *offer_probability* for each incoming trip, choosing the recommended zone
    uniformly at random from zones other than the original destination.
    """

    def __init__(
        self,
        zone_ids: List[int],
        offer_probability: float = RELOCATION_OFFER_PROB,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.zone_ids = list(zone_ids)
        self.offer_probability = offer_probability
        self.rng = rng or random.Random()

    def query(
        self,
        origin: int,
        destination: int,
        request_time: float,
    ) -> Optional[RelocationOpportunity]:
        """
        Ask the OR model whether a relocation recommendation exists for this trip.

        Returns
        -------
        RelocationOpportunity if a recommendation is available, else None.

        This method signature is the stable contract with the simulation engine.
        """
        if self.rng.random() > self.offer_probability:
            return None

        candidates = [z for z in self.zone_ids if z != destination]
        if not candidates:
            return None

        recommended = self.rng.choice(candidates)
        return RelocationOpportunity(
            origin=origin,
            original_dest=destination,
            recommended_dest=recommended,
            time_indicator=request_time,
        )
