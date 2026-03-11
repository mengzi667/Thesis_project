# generate_or_output.py
# Simulated OR model output generator.
#
# Runs a rolling-horizon flow-balance planning model over the same demand
# profile used by the simulation, then writes a structured U_odit table that
# ORInterface can load directly via load_from_csv() or load_from_json().
#
# Algorithm (per re-planning step):
#   1. Project expected inventory forward over the look-ahead horizon using
#      the demand profile rates (piece-wise Poisson superposition).
#   2. Identify zones with projected SURPLUS (supply accumulates) and zones
#      with projected DEFICIT (demand exceeds incoming supply).
#   3. For each (origin, surplus_dest, slot) with material OD flow,
#      generate a recommendation to redirect to the highest-deficit zone
#      reachable with a reasonable detour.
#   4. Incentive = f(extra walking distance); longer detour → higher incentive.
#
# Files produced:
#   or_output.csv    — loadable by ORInterface.load_from_csv()
#   or_output.json   — loadable by ORInterface.load_from_json()
#   or_report.txt    — human-readable period-by-period planning report
#
# Usage (run from project root):
#   python -m or_model.generate_or_output
#
# To use the output in the simulation, edit main.py:
#   or_interface = ORInterface.load_from_csv("or_output.csv")

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from config import (
    FLEET_SIZE,
    GRID_SPACING,
    LOOKAHEAD_HORIZON,
    NUM_ZONES,
    PLANNING_PERIOD,
    RANDOM_SEED,
    RELOCATION_INCENTIVE,
    SIM_DURATION,
    TRIP_ARRIVAL_RATE,
    WALKING_THRESHOLD,
)
from simulation.trip_generator import build_synthetic_demand_profile, DemandProfile


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _build_coords(zone_ids: List[int], spacing: float) -> Dict[int, Tuple[float, float]]:
    side = math.ceil(math.sqrt(len(zone_ids)))
    return {
        zid: (col * spacing, row * spacing)
        for zid, (row, col) in (
            (zone_ids[idx], divmod(idx, side)) for idx in range(len(zone_ids))
        )
    }


def _dist_m(coords: Dict[int, Tuple[float, float]], a: int, b: int) -> float:
    ax, ay = coords[a]
    bx, by = coords[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _incentive(detour_m: float) -> float:
    """
    Incentive schedule:
      ≤ 500 m  (adjacent zone)   : 1.20  — short detour
      ≤ 1000 m (one zone gap)    : 1.50  — medium detour
      > 1000 m (two+ zones away) : 2.00  — long detour, high reward
    """
    if detour_m <= WALKING_THRESHOLD:
        return 1.20
    if detour_m <= 2 * WALKING_THRESHOLD:
        return 1.50
    return 2.00


# ── OR decision record ────────────────────────────────────────────────────────

@dataclass
class ORDecision:
    # Required by ORInterface.load_from_csv / load_from_json
    origin:           int
    original_dest:    int
    recommended_dest: int
    time_slot:        int
    incentive_amount: float
    # Verification / analysis fields (ignored by ORInterface, useful for review)
    period_start_min: float
    expected_flow:    float   # expected trips on (origin→original_dest) in this slot
    surplus_at_dest:  float   # projected excess supply at original_dest in look-ahead
    deficit_at_redir: float   # projected supply shortfall at recommended_dest
    detour_m:         float   # extra walking distance (original_dest → recommended_dest)


# ── Rolling-horizon flow-balance planner ──────────────────────────────────────

def run_or_model(
    zone_ids: List[int],
    demand_profile: DemandProfile,
    sim_duration: float       = SIM_DURATION,
    planning_period: float    = PLANNING_PERIOD,
    lookahead_horizon: float  = LOOKAHEAD_HORIZON,
    fleet_size: int           = FLEET_SIZE,
    grid_spacing: float       = GRID_SPACING,
    inv_alert_threshold: int  = 1,   # trigger relocation if projected inv < this
    min_od_flow: float        = 0.08, # minimum expected trips to justify a recommendation
) -> List[ORDecision]:
    """
    Rolling-horizon OR planner.

    For every planning slot (re-planning cadence = *planning_period*):
      - Projects inventory over *lookahead_horizon* minutes.
      - Identifies surplus destinations and deficit zones within the look-ahead.
      - For the CURRENT slot, recommends redirecting high-flow trips from
        surplus destinations to the most-deficit zone.

    Returns a list of ORDecision records covering the entire simulation window.
    """
    coords     = _build_coords(zone_ids, grid_spacing)
    num_slots  = int(sim_duration // planning_period) + 1
    la_slots   = int(lookahead_horizon // planning_period)

    # Inventory state (evolves period-by-period)
    inventory: Dict[int, float] = {z: fleet_size / len(zone_ids) for z in zone_ids}

    decisions: List[ORDecision] = []
    seen: Set[Tuple[int, int, int]] = set()

    for slot in range(num_slots):
        slot_t = slot * planning_period

        # ── 1. Forecast inventory over look-ahead window ──────────────────────
        inv_fwd: List[Dict[int, float]] = [dict(inventory)]
        for k in range(1, la_slots + 1):
            future_slot = slot + k - 1
            prev = inv_fwd[k - 1]
            nxt: Dict[int, float] = {}
            for z in zone_ids:
                dep = demand_profile.rate_for(z, future_slot) * planning_period
                arr = demand_profile.dest_arrival_rate(z, future_slot, zone_ids) * planning_period
                nxt[z] = max(0.0, prev[z] - dep + arr)
            inv_fwd.append(nxt)

        # ── 2. Aggregate surplus / deficit over near-term look-ahead ──────────
        # Weight nearer periods more heavily (1/k decay).
        n_near = min(la_slots, 4)
        surplus: Dict[int, float] = {z: 0.0 for z in zone_ids}
        deficit: Dict[int, float] = {z: 0.0 for z in zone_ids}
        for k in range(1, n_near + 1):
            w   = 1.0 / k
            prj = inv_fwd[k]
            for z in zone_ids:
                gap = prj[z] - inv_alert_threshold
                if gap > 0:
                    surplus[z] += gap * w
                else:
                    deficit[z] -= gap * w   # positive value = shortfall

        surplus_dests  = [z for z in zone_ids if surplus[z] > 0.25]
        deficit_zones  = [z for z in zone_ids if deficit[z] > 0.25]

        if not surplus_dests or not deficit_zones:
            inventory = inv_fwd[1]
            continue

        # ── 3. OD probability denominators for this slot ──────────────────────
        od_denom: Dict[int, float] = {
            o: max(
                sum(
                    max(demand_profile.od_weights.get((o, d2), {}).get(slot, 1.0), 0.0)
                    for d2 in zone_ids
                ),
                1e-9,
            )
            for o in zone_ids
        }

        # ── 4. Generate recommendations ───────────────────────────────────────
        for orig_dest in surplus_dests:
            # Best redirect: highest deficit, then shortest detour
            candidates = sorted(
                [z for z in deficit_zones if z != orig_dest],
                key=lambda z: (-deficit[z], _dist_m(coords, orig_dest, z)),
            )
            if not candidates:
                continue
            redirect = candidates[0]
            detour   = _dist_m(coords, orig_dest, redirect)

            for origin in zone_ids:
                od_w = max(
                    demand_profile.od_weights.get((origin, orig_dest), {}).get(slot, 1.0), 0.0
                )
                od_prob = od_w / od_denom[origin]
                exp_flow = demand_profile.rate_for(origin, slot) * planning_period * od_prob

                if exp_flow < min_od_flow:
                    continue

                key = (origin, orig_dest, slot)
                if key in seen:
                    continue
                seen.add(key)

                decisions.append(ORDecision(
                    origin=origin,
                    original_dest=orig_dest,
                    recommended_dest=redirect,
                    time_slot=slot,
                    incentive_amount=_incentive(detour),
                    period_start_min=slot_t,
                    expected_flow=round(exp_flow, 3),
                    surplus_at_dest=round(surplus[orig_dest], 2),
                    deficit_at_redir=round(deficit[redirect], 2),
                    detour_m=round(detour, 1),
                ))

        inventory = inv_fwd[1]

    return decisions


# ── Output writers ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "origin", "original_dest", "recommended_dest", "time_slot", "incentive_amount",
    "period_start_min", "expected_flow", "surplus_at_dest", "deficit_at_redir", "detour_m",
]

def write_csv(decisions: List[ORDecision], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for d in decisions:
            writer.writerow({
                "origin":           d.origin,
                "original_dest":    d.original_dest,
                "recommended_dest": d.recommended_dest,
                "time_slot":        d.time_slot,
                "incentive_amount": d.incentive_amount,
                "period_start_min": d.period_start_min,
                "expected_flow":    d.expected_flow,
                "surplus_at_dest":  d.surplus_at_dest,
                "deficit_at_redir": d.deficit_at_redir,
                "detour_m":         d.detour_m,
            })

def write_json(decisions: List[ORDecision], path: str) -> None:
    records = [
        {
            "origin":           d.origin,
            "original_dest":    d.original_dest,
            "recommended_dest": d.recommended_dest,
            "time_slot":        d.time_slot,
            "incentive_amount": d.incentive_amount,
            "period_start_min": d.period_start_min,
            "expected_flow":    d.expected_flow,
            "surplus_at_dest":  d.surplus_at_dest,
            "deficit_at_redir": d.deficit_at_redir,
            "detour_m":         d.detour_m,
        }
        for d in decisions
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ── Human-readable report ─────────────────────────────────────────────────────

def write_report(
    decisions: List[ORDecision],
    zone_ids: List[int],
    demand_profile: DemandProfile,
    path: str,
) -> None:
    W = 72
    SEP  = "=" * W
    SEP2 = "-" * W

    # Group decisions by slot
    from collections import defaultdict
    by_slot: Dict[int, List[ORDecision]] = defaultdict(list)
    for d in decisions:
        by_slot[d.time_slot].append(d)

    num_slots = int(SIM_DURATION // PLANNING_PERIOD) + 1

    with open(path, "w", encoding="utf-8") as f:
        def p(line: str = "") -> None:
            f.write(line + "\n")

        p(SEP)
        p("  OR MODEL PLANNING REPORT".center(W))
        p(SEP)
        p(f"  Simulation duration  : {SIM_DURATION:.0f} min")
        p(f"  Planning period      : {PLANNING_PERIOD:.0f} min / period  "
          f"({num_slots} total periods)")
        p(f"  Look-ahead horizon   : {LOOKAHEAD_HORIZON:.0f} min  "
          f"({int(LOOKAHEAD_HORIZON / PLANNING_PERIOD)} periods)")
        p(f"  Total recommendations: {len(decisions)}")
        p(f"  Periods with plan    : {len(by_slot)} / {num_slots}")
        p(SEP)
        p()

        for slot in range(num_slots):
            slot_t    = slot * PLANNING_PERIOD
            slot_recs = by_slot.get(slot, [])

            total_rate = demand_profile.total_rate(slot)
            zone_rates = {
                z: demand_profile.rate_for(z, slot) for z in zone_ids
            }

            p(f"  PERIOD d={slot:02d}  |  t = {slot_t:.0f} – {slot_t + PLANNING_PERIOD:.0f} min"
              f"  |  total λ = {total_rate:.3f} trips/min")

            # Zone net-flow table
            p(f"  {'Zone':>4}  {'Depart rate':>12}  {'Arrive rate':>12}  {'Net flow':>10}")
            p(f"  {'':─<52}")
            for z in sorted(zone_ids):
                dep = demand_profile.rate_for(z, slot)
                arr = demand_profile.dest_arrival_rate(z, slot, zone_ids)
                net = arr - dep
                tag = "  ← SURPLUS" if net > 0.02 else ("  ← DEFICIT" if net < -0.02 else "")
                p(f"  {z:>4}  {dep:>12.4f}  {arr:>12.4f}  {net:>+10.4f}{tag}")

            if slot_recs:
                p()
                p(f"  Recommendations ({len(slot_recs)} rows):")
                p(f"  {'Orig':>4}  {'From':>5}  {'To':>5}  {'Incentive':>9}"
                  f"  {'ExpFlow':>8}  {'Surplus':>8}  {'Deficit':>8}  {'Detour(m)':>10}")
                p(f"  {'':─<70}")
                for r in sorted(slot_recs, key=lambda x: (x.original_dest, x.origin)):
                    p(
                        f"  Z{r.origin:>2} → Z{r.original_dest:>2}  "
                        f"⟹ Z{r.recommended_dest:>2}  "
                        f"  ${r.incentive_amount:>6.2f}  "
                        f"  {r.expected_flow:>8.3f}  "
                        f"  {r.surplus_at_dest:>8.2f}  "
                        f"  {r.deficit_at_redir:>8.2f}  "
                        f"  {r.detour_m:>10.1f}"
                    )
            else:
                p("  (no recommendations — demand balanced)")

            p()

        # ── Global statistics ─────────────────────────────────────────────────
        p(SEP)
        p("  SUMMARY STATISTICS".center(W))
        p(SEP)

        inc_vals = sorted(set(d.incentive_amount for d in decisions))
        for iv in inc_vals:
            cnt = sum(1 for d in decisions if d.incentive_amount == iv)
            p(f"  Incentive ${iv:.2f}  :  {cnt:>4} recommendations")
        p(SEP2)

        # Per-zone redirect counts
        from collections import Counter
        redir_to = Counter(d.recommended_dest for d in decisions)
        redir_from = Counter(d.original_dest for d in decisions)
        p(f"  {'Zone':>4}  {'Redirected FROM':>16}  {'Redirected TO':>16}")
        p(f"  {'':─<42}")
        for z in sorted(zone_ids):
            p(f"  {z:>4}  {redir_from.get(z, 0):>16}  {redir_to.get(z, 0):>16}")
        p(SEP)
        p()
        p("  Loading in simulation:")
        p("    or_interface = ORInterface.load_from_csv(\"or_output.csv\")")
        p("  or:")
        p("    or_interface = ORInterface.load_from_json(\"or_output.json\")")
        p()
        p(SEP)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    W = 62
    print("=" * W)
    print("  OR Model Output Generator".center(W))
    print("=" * W)

    rng = random.Random(RANDOM_SEED)

    # Use the same spatial layout as the simulation (0..NUM_ZONES-1)
    zone_ids = list(range(NUM_ZONES))
    n_grp    = max(1, NUM_ZONES // 3)
    generators = zone_ids[:n_grp]
    neutrals   = zone_ids[n_grp: NUM_ZONES - n_grp]
    attractors = zone_ids[NUM_ZONES - n_grp:]

    print(f"  Zones     : {NUM_ZONES}  |  Fleet : {FLEET_SIZE}")
    print(f"  Generators: {generators}")
    print(f"  Neutrals  : {neutrals}")
    print(f"  Attractors: {attractors}")
    print(f"  Period    : {PLANNING_PERIOD:.0f} min  |  Look-ahead: {LOOKAHEAD_HORIZON:.0f} min")
    print()

    # Build demand profile (identical to simulation)
    print("  Building demand profile ...")
    demand_profile = build_synthetic_demand_profile(
        zone_ids=zone_ids,
        sim_duration=SIM_DURATION,
        planning_period=PLANNING_PERIOD,
        total_rate=TRIP_ARRIVAL_RATE,
        rng=rng,
    )

    # Run rolling-horizon OR model
    print("  Running rolling-horizon OR planner ...")
    decisions = run_or_model(
        zone_ids=zone_ids,
        demand_profile=demand_profile,
        sim_duration=SIM_DURATION,
        planning_period=PLANNING_PERIOD,
        lookahead_horizon=LOOKAHEAD_HORIZON,
        fleet_size=FLEET_SIZE,
        grid_spacing=GRID_SPACING,
    )

    num_slots    = int(SIM_DURATION // PLANNING_PERIOD) + 1
    slots_w_plan = len(set(d.time_slot for d in decisions))
    print(f"  Done: {len(decisions)} recommendations over "
          f"{slots_w_plan}/{num_slots} planning periods")
    print()

    # Incentive breakdown
    from collections import Counter
    inc_counts = Counter(d.incentive_amount for d in decisions)
    for iv in sorted(inc_counts):
        print(f"    Incentive ${iv:.2f}  → {inc_counts[iv]:>3} recommendations")
    print()

    # Write outputs
    write_csv(decisions, "data/or_output.csv")
    print("  Wrote: data/or_output.csv")

    write_json(decisions, "data/or_output.json")
    print("  Wrote: data/or_output.json")

    write_report(decisions, zone_ids, demand_profile, "data/or_report.txt")
    print("  Wrote: data/or_report.txt")
    print()
    print("  To use in simulation, change main.py:")
    print("    or_interface = ORInterface.load_from_csv(\"data/or_output.csv\")")
    print("=" * W)


if __name__ == "__main__":
    main()
