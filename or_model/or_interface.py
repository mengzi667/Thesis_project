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
from typing import Dict, Iterable, List, Optional, Tuple, Union

from config import (
    OR_FIXED_INCENTIVE_EUR,
    OR_FORCE_FIXED_INCENTIVE,
    OR_QUOTA_CONSUME_POLICY,
    OR_VALIDATION_MODE,
    PLANNING_PERIOD,
    RELOCATION_INCENTIVE,
    RELOCATION_OFFER_PROB,
)


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
    quota:            int = field(default=1)
    quota_remaining:  int = field(default=1)

    def __repr__(self) -> str:
        return (
            f"RelocOpp(o={self.origin}→d={self.original_dest}"
            f"→i={self.recommended_dest}, t={self.time_indicator:.1f}, "
            f"inc={self.incentive_amount}, q={self.quota_remaining}/{self.quota})"
        )


# Key type: (origin, original_dest, time_slot_index)
UoditKey   = Tuple[int, int, int]
UoditTable = Dict[UoditKey, List[RelocationOpportunity]]


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
        u_odit_table: Union[UoditTable, Dict[UoditKey, RelocationOpportunity]],
        planning_interval: float = PLANNING_PERIOD,
        quota_consume_policy: str = OR_QUOTA_CONSUME_POLICY,
    ) -> None:
        self._table: UoditTable = {}
        for key, value in u_odit_table.items():
            if isinstance(value, RelocationOpportunity):
                self._table[key] = [value]
            else:
                self._table[key] = list(value)
        self.planning_interval = planning_interval
        self.quota_consume_policy = quota_consume_policy

        # OR injection layer metrics
        self.or_rows_loaded = sum(len(rows) for rows in self._table.values())
        self.or_offer_attempts = 0
        self.or_offer_blocked_by_quota = 0
        self.or_quota_consumed = 0
        self.or_incentive_overridden_count = 0

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
        candidates = self._table.get((origin, destination, slot))
        if not candidates:
            return None

        self.or_offer_attempts += 1
        eligible = [row for row in candidates if row.quota_remaining > 0]
        if not eligible:
            self.or_offer_blocked_by_quota += 1
            return None

        # Deterministic tie-break: prioritize highest remaining quota, then smaller zone id.
        row = sorted(
            eligible,
            key=lambda x: (-x.quota_remaining, x.recommended_dest),
        )[0]

        if self.quota_consume_policy == "consume_on_offer":
            self._consume_one(row)
        return row

    def consume_after_decision(
        self,
        opportunity: Optional[RelocationOpportunity],
        accepted: bool,
    ) -> None:
        """
        Finalize quota consumption based on configured policy.

        - consume_on_accept: consume only when accepted is True
        - consume_on_offer : already consumed in query()
        """
        if opportunity is None:
            return
        if self.quota_consume_policy == "consume_on_accept" and accepted:
            self._consume_one(opportunity)

    def _consume_one(self, opportunity: RelocationOpportunity) -> None:
        if opportunity.quota_remaining <= 0:
            return
        opportunity.quota_remaining -= 1
        self.or_quota_consumed += 1

    def stats(self) -> Dict[str, int]:
        active_rows = sum(
            1
            for rows in self._table.values()
            for row in rows
            if row.quota_remaining > 0
        )
        quota_remaining = sum(
            max(0, row.quota_remaining)
            for rows in self._table.values()
            for row in rows
        )
        return {
            "or_rows_loaded": self.or_rows_loaded,
            "or_rows_active_in_horizon": active_rows,
            "or_offer_attempts": self.or_offer_attempts,
            "or_offer_blocked_by_quota": self.or_offer_blocked_by_quota,
            "or_quota_consumed": self.or_quota_consumed,
            "or_quota_remaining_end": quota_remaining,
            "or_incentive_overridden_count": self.or_incentive_overridden_count,
        }

    # ── Inspection helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of (o, d, slot) entries in the loaded table."""
        return sum(len(rows) for rows in self._table.values())

    # ── Loaders (structured external input sources) ───────────────────────────

    @classmethod
    def load_from_dict(
        cls,
        records: Iterable[dict],
        planning_interval: float = PLANNING_PERIOD,
        quota_consume_policy: str = OR_QUOTA_CONSUME_POLICY,
        force_fixed_incentive: bool = OR_FORCE_FIXED_INCENTIVE,
        fixed_incentive_eur: float = OR_FIXED_INCENTIVE_EUR,
        validation_mode: str = OR_VALIDATION_MODE,
        valid_zone_ids: Optional[set] = None,
        max_time_slot: Optional[int] = None,
    ) -> "ORInterface":
        """
        Build an ORInterface from a list of dicts, e.g.:

            [{"origin": 0, "original_dest": 3, "recommended_dest": 7,
              "time_slot": 2, "incentive_amount": 1.5}, ...]
        """
        table: UoditTable = {}
        overridden_count = 0

        strict = str(validation_mode).strip().lower() == "strict"

        def _invalid(msg: str) -> None:
            if strict:
                raise ValueError(msg)
            print(f"[ORInterface] warning: {msg}")

        for r in records:
            try:
                origin = int(r["origin"])
                original_dest = int(r["original_dest"])
                recommended_dest = int(r["recommended_dest"])
                time_slot = int(r["time_slot"])
                quota = int(float(r.get("quota", 1)))
                raw_incentive = float(r.get("incentive_amount", RELOCATION_INCENTIVE))
            except Exception as exc:
                _invalid(f"row parse failed: {exc}; row={r}")
                continue

            if quota < 0:
                _invalid(f"negative quota found, clipped to 0; row={r}")
                quota = 0

            if recommended_dest == original_dest:
                _invalid(f"recommended_dest == original_dest not allowed; row={r}")
                continue

            if valid_zone_ids is not None:
                if origin not in valid_zone_ids or original_dest not in valid_zone_ids or recommended_dest not in valid_zone_ids:
                    _invalid(f"zone id out of range; row={r}")
                    continue

            if max_time_slot is not None and (time_slot < 0 or time_slot > max_time_slot):
                _invalid(f"time_slot out of simulation horizon; row={r}")
                continue

            if force_fixed_incentive:
                if abs(raw_incentive - fixed_incentive_eur) > 1e-9:
                    overridden_count += 1
                incentive = fixed_incentive_eur
            else:
                incentive = raw_incentive

            if quota == 0:
                continue

            key: UoditKey = (origin, original_dest, time_slot)
            if key not in table:
                table[key] = []

            # Merge same target-i rows; keep different i rows as separate candidates.
            merged = False
            for opp in table[key]:
                if opp.recommended_dest == recommended_dest:
                    opp.quota += quota
                    opp.quota_remaining += quota
                    merged = True
                    break
            if not merged:
                table[key].append(
                    RelocationOpportunity(
                        origin=origin,
                        original_dest=original_dest,
                        recommended_dest=recommended_dest,
                        time_indicator=float(time_slot) * planning_interval,
                        incentive_amount=incentive,
                        quota=quota,
                        quota_remaining=quota,
                    )
                )
        obj = cls(table, planning_interval, quota_consume_policy=quota_consume_policy)
        obj.or_incentive_overridden_count = overridden_count
        return obj

    @classmethod
    def load_from_csv(
        cls,
        filepath: str,
        planning_interval: float = PLANNING_PERIOD,
        quota_consume_policy: str = OR_QUOTA_CONSUME_POLICY,
        force_fixed_incentive: bool = OR_FORCE_FIXED_INCENTIVE,
        fixed_incentive_eur: float = OR_FIXED_INCENTIVE_EUR,
        validation_mode: str = OR_VALIDATION_MODE,
        valid_zone_ids: Optional[set] = None,
        max_time_slot: Optional[int] = None,
    ) -> "ORInterface":
        """
        Load U_odit table from a CSV file.

        Expected columns: origin, original_dest, recommended_dest,
                          time_slot, incentive_amount (optional).
        """
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return cls.load_from_dict(
                reader,
                planning_interval,
                quota_consume_policy=quota_consume_policy,
                force_fixed_incentive=force_fixed_incentive,
                fixed_incentive_eur=fixed_incentive_eur,
                validation_mode=validation_mode,
                valid_zone_ids=valid_zone_ids,
                max_time_slot=max_time_slot,
            )

    @classmethod
    def load_from_json(
        cls,
        filepath: str,
        planning_interval: float = PLANNING_PERIOD,
        quota_consume_policy: str = OR_QUOTA_CONSUME_POLICY,
        force_fixed_incentive: bool = OR_FORCE_FIXED_INCENTIVE,
        fixed_incentive_eur: float = OR_FIXED_INCENTIVE_EUR,
        validation_mode: str = OR_VALIDATION_MODE,
        valid_zone_ids: Optional[set] = None,
        max_time_slot: Optional[int] = None,
    ) -> "ORInterface":
        """
        Load U_odit table from a JSON file (array of record objects).
        """
        with open(filepath, encoding="utf-8") as f:
            records = json.load(f)
        return cls.load_from_dict(
            records,
            planning_interval,
            quota_consume_policy=quota_consume_policy,
            force_fixed_incentive=force_fixed_incentive,
            fixed_incentive_eur=fixed_incentive_eur,
            validation_mode=validation_mode,
            valid_zone_ids=valid_zone_ids,
            max_time_slot=max_time_slot,
        )


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
                table[key] = [RelocationOpportunity(
                    origin=origin,
                    original_dest=dest,
                    recommended_dest=recommended,
                    time_indicator=slot * planning_interval,
                    incentive_amount=incentive_amount,
                )]
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
                    table[key] = [RelocationOpportunity(
                        origin=origin,
                        original_dest=orig_dest,
                        recommended_dest=redirect,
                        time_indicator=slot * planning_period,
                        incentive_amount=incentive_amount,
                    )]

    return table
