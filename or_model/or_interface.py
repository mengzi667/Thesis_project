# or_interface.py
# OR model interface — structured external input implementation.
#
# PRD requirement: OR outputs are treated as *external structured inputs*
# injected through a predefined interface (U_odit table).  The simulation
# engine only calls query(); it never sees how the table was built or loaded.
#
# Stable integration contract (simulation engine must not change):
#   ORInterface.query(origin, destination, request_time) -> Optional[RelocationOpportunity]
#
# Switching from placeholder to real OR outputs:
#   Build or load a real UoditTable and pass it to ORInterface() — done.
#   No changes required anywhere else.

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from config import PLANNING_PERIOD, RELOCATION_INCENTIVE, RELOCATION_OFFER_PROB


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RelocationOpportunity:
    """
    One row of the U_odit table.

    Semantics: if a trip (origin → original_dest) arrives during
    planning interval *time_slot*, recommend the user to drop off at
    recommended_dest with the offered incentive.
    """
    origin:           int
    original_dest:    int
    recommended_dest: int
    time_indicator:   float          # planning-interval start time (minutes)
    incentive_amount: float = field(default=RELOCATION_INCENTIVE)

    def __repr__(self) -> str:
        return (
            f"RelocOpp(o={self.origin}→d={self.original_dest}"
            f"→i={self.recommended_dest}, t={self.time_indicator:.1f}, "
            f"inc={self.incentive_amount})"
        )


# Key type: (origin, original_dest, time_slot_index)
UoditKey   = Tuple[int, int, int]
UoditTable = Dict[UoditKey, RelocationOpportunity]


# ── Interface ─────────────────────────────────────────────────────────────────

class ORInterface:
    """
    Looks up relocation recommendations from a pre-loaded UoditTable.

    The table is a *structured external input* — it is built once
    (synthetically, from a file, or from the real OR model) and injected
    at construction time.  The query() method performs a pure table look-up
    with no internal randomness.

    Parameters
    ----------
    u_odit_table     : Pre-built mapping (origin, dest, slot) → RelocationOpportunity.
    planning_interval: Width of each OR planning interval in minutes.
                       Used to convert continuous request_time to slot index.
    """

    def __init__(
        self,
        u_odit_table: UoditTable,
        planning_interval: float = PLANNING_PERIOD,
    ) -> None:
        self._table: UoditTable = u_odit_table
        self.planning_interval = planning_interval

    # ── Stable query contract ─────────────────────────────────────────────────

    def query(
        self,
        origin: int,
        destination: int,
        request_time: float,
    ) -> Optional[RelocationOpportunity]:
        """
        Look up the U_odit table for a matching relocation recommendation.

        Returns a RelocationOpportunity if one exists for (origin, destination,
        time_slot), otherwise None.  No randomness is involved here.
        """
        slot = int(request_time // self.planning_interval)
        return self._table.get((origin, destination, slot))

    # ── Inspection helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of (o, d, slot) entries in the loaded table."""
        return len(self._table)

    # ── Loaders (structured external input sources) ───────────────────────────

    @classmethod
    def load_from_dict(
        cls,
        records: Iterable[dict],
        planning_interval: float = PLANNING_PERIOD,
    ) -> "ORInterface":
        """
        Build an ORInterface from a list of dicts, e.g.:

            [{"origin": 0, "original_dest": 3, "recommended_dest": 7,
              "time_slot": 2, "incentive_amount": 1.5}, ...]
        """
        table: UoditTable = {}
        for r in records:
            key: UoditKey = (int(r["origin"]), int(r["original_dest"]), int(r["time_slot"]))
            table[key] = RelocationOpportunity(
                origin=int(r["origin"]),
                original_dest=int(r["original_dest"]),
                recommended_dest=int(r["recommended_dest"]),
                time_indicator=float(r["time_slot"]) * planning_interval,
                incentive_amount=float(r.get("incentive_amount", RELOCATION_INCENTIVE)),
            )
        return cls(table, planning_interval)

    @classmethod
    def load_from_csv(
        cls,
        filepath: str,
        planning_interval: float = PLANNING_PERIOD,
    ) -> "ORInterface":
        """
        Load U_odit table from a CSV file.

        Expected columns: origin, original_dest, recommended_dest,
                          time_slot, incentive_amount (optional).
        """
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return cls.load_from_dict(reader, planning_interval)

    @classmethod
    def load_from_json(
        cls,
        filepath: str,
        planning_interval: float = PLANNING_PERIOD,
    ) -> "ORInterface":
        """
        Load U_odit table from a JSON file (array of record objects).
        """
        with open(filepath, encoding="utf-8") as f:
            records = json.load(f)
        return cls.load_from_dict(records, planning_interval)


# ── Placeholder table generator ───────────────────────────────────────────────
# Randomness lives *only* here — outside the interface.  Call this once,
# inspect / persist the result, then inject it into ORInterface.

def generate_synthetic_table(
    zone_ids: List[int],
    sim_duration: float,
    planning_interval: float = PLANNING_PERIOD,
    offer_probability: float = RELOCATION_OFFER_PROB,
    incentive_amount: float = RELOCATION_INCENTIVE,
    rng: Optional[random.Random] = None,
) -> UoditTable:
    """
    Generate a synthetic U_odit table covering all (origin, dest, slot)
    combinations for the given simulation window.

    For each combination a recommendation is created with probability
    *offer_probability*; the recommended zone is drawn uniformly from
    zones other than the original destination.

    Returns
    -------
    UoditTable — a plain dict that can be inspected, serialised, or
    passed directly to ORInterface().
    """
    if rng is None:
        rng = random.Random()

    num_slots = max(1, int(sim_duration // planning_interval) + 1)
    table: UoditTable = {}

    for origin in zone_ids:
        for dest in zone_ids:
            for slot in range(num_slots):
                if rng.random() > offer_probability:
                    continue
                candidates = [z for z in zone_ids if z != dest]
                if not candidates:
                    continue
                recommended = rng.choice(candidates)
                key: UoditKey = (origin, dest, slot)
                table[key] = RelocationOpportunity(
                    origin=origin,
                    original_dest=dest,
                    recommended_dest=recommended,
                    time_indicator=slot * planning_interval,
                    incentive_amount=incentive_amount,
                )
    return table


# ── Demand-informed table builder ──────────────────────────────────────────────────

def build_demand_informed_table(
    demand_profile,              # DemandProfile (from trip_generator)
    zone_ids: List[int],
    sim_duration: float,
    planning_period: float = PLANNING_PERIOD,
    incentive_amount: float = RELOCATION_INCENTIVE,
    rng: Optional[random.Random] = None,
) -> UoditTable:
    """
    Build a structured U_odit table driven by a DemandProfile.

    For each planning slot d the algorithm:
      1. Computes expected net flow per zone:
            net_flow(z) = arrivals_at_z(d) - departures_from_z(d)
         Positive -> surplus (supply accumulates); negative -> deficit.
      2. For each (origin, surplus_dest) pair with non-trivial OD mass,
         adds a recommendation (origin, surplus_dest -> deficit_zone).

    This approximates an OR plan that re-balances supply from over-served
    destinations to zones where demand exceeds incoming supply.

    Switching from placeholder to real OR output:
      Replace this function with ORInterface.load_from_csv("or_outputs.csv").
    """
    if rng is None:
        rng = random.Random()

    num_slots = max(1, int(sim_duration // planning_period) + 1)
    table: UoditTable = {}

    for slot in range(num_slots):
        # ── Step 1: net flow per zone ─────────────────────────────────────────
        net_flow: Dict[int, float] = {
            z: demand_profile.dest_arrival_rate(z, slot, zone_ids)
               - demand_profile.rate_for(z, slot)
            for z in zone_ids
        }

        deficit_zones = sorted(
            [z for z in zone_ids if net_flow[z] < 0], key=lambda z: net_flow[z]
        )
        surplus_zones = [z for z in zone_ids if net_flow[z] > 0]

        if not deficit_zones or not surplus_zones:
            continue  # balanced slot — no relocation needed

        # Top half of deficit zones are valid redirect targets
        top_deficit = deficit_zones[: max(1, len(deficit_zones) // 2 + 1)]

        # ── Step 2: generate recommendations ─────────────────────────────────
        # For each origin, precompute denominator of OD probability
        od_denom = {
            o: max(
                sum(
                    max(demand_profile.od_weights.get((o, d2), {}).get(slot, 1.0), 0.0)
                    for d2 in zone_ids
                ),
                1e-9,
            )
            for o in zone_ids
        }

        for orig_dest in surplus_zones:
            for origin in zone_ids:
                od_prob = (
                    max(demand_profile.od_weights.get((origin, orig_dest), {}).get(slot, 1.0), 0.0)
                    / od_denom[origin]
                )
                # Only recommend for OD pairs with material flow
                if od_prob < 1.0 / (2 * len(zone_ids)):
                    continue

                candidates = [j for j in top_deficit if j != orig_dest]
                if not candidates:
                    continue
                redirect = rng.choice(candidates)

                key: UoditKey = (origin, orig_dest, slot)
                if key not in table:
                    table[key] = RelocationOpportunity(
                        origin=origin,
                        original_dest=orig_dest,
                        recommended_dest=redirect,
                        time_indicator=slot * planning_period,
                        incentive_amount=incentive_amount,
                    )

    return table
