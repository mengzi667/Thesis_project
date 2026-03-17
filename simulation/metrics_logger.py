# metrics_logger.py
# KPI collection and summary reporting.
#
# Records per-trip events and periodic inventory snapshots.
# Provides aggregated statistics for system, behavioural, and operational metrics.
# Preserves sufficient detail for future EDL / RL reward computation.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Per-trip record ───────────────────────────────────────────────────────────

# Possible values for TripRecord.unserved_reason
UNSERVED_NO_SUPPLY = "no_supply"    # no rentable scooter at origin zone (supply shortage)
UNSERVED_OPT_OUT   = "user_opt_out" # scooters available but user chose not to rent


@dataclass
class TripRecord:
    """Stores all relevant attributes of a single processed trip request."""
    request_id:          int
    origin_zone:         int
    effective_dest:      int      # actual drop-off (may differ from requested dest)
    request_time:        float
    trip_duration:       float
    trip_distance:       float
    user_type:           str
    served:              bool
    relocation_offered:  bool
    relocation_accepted: bool
    scooter_id:          Optional[int]
    user_choice:         Optional[str] = None  # offer / base / opt_out
    unserved_reason:     Optional[str] = None  # None when served; see UNSERVED_* constants


# ── MetricsLogger ─────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Collects per-trip TripRecords and periodic inventory snapshots, then
    produces aggregated KPI summaries.
    """

    def __init__(self) -> None:
        self.trip_records: List[TripRecord] = []
        self.inventory_snapshots: List[Dict[int, Tuple[int, int, int]]] = []
        self.snapshot_times: List[float] = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def log_trip(self, record: TripRecord) -> None:
        self.trip_records.append(record)

    def snapshot_inventories(
        self,
        time: float,
        zone_state: Dict[int, Tuple[int, int, int]],
    ) -> None:
        """
        Store a copy of zone-level inventory state at the given simulation time.
        zone_state maps zone_id → (inactive, low, high).
        """
        self.snapshot_times.append(time)
        self.inventory_snapshots.append(dict(zone_state))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a flat dictionary of aggregated KPIs."""
        total   = len(self.trip_records)
        served  = sum(1 for r in self.trip_records if r.served)
        # PRD §14: only "no rentable scooter" counts as true demand loss (EDL-relevant).
        # User opt-out is a behavioural outcome, not a supply shortage.
        unserved_no_supply = sum(
            1 for r in self.trip_records if r.unserved_reason == UNSERVED_NO_SUPPLY
        )
        unserved_opt_out = sum(
            1 for r in self.trip_records if r.unserved_reason == UNSERVED_OPT_OUT
        )
        unserved_total = unserved_no_supply + unserved_opt_out

        reloc_offered  = sum(1 for r in self.trip_records if r.relocation_offered)
        reloc_accepted = sum(1 for r in self.trip_records if r.relocation_accepted)
        reloc_rejected = reloc_offered - reloc_accepted

        choice_offer = sum(1 for r in self.trip_records if r.user_choice == "offer")
        choice_base = sum(1 for r in self.trip_records if r.user_choice == "base")
        choice_opt_out = sum(1 for r in self.trip_records if r.user_choice == "opt_out")
        choice_total = choice_offer + choice_base + choice_opt_out

        # Fleet utilisation: fraction of trips where a scooter was actually assigned
        utilisation = served / total if total > 0 else 0.0

        # Battery composition averaged over inventory snapshots
        avg_inactive, avg_low, avg_high = self._avg_battery_composition()

        return {
            # System metrics
            "total_requests":         total,
            "served_trips":           served,
            "unserved_trips":         unserved_total,
            "unserved_no_supply":     unserved_no_supply,   # EDL-relevant
            "unserved_user_opt_out":  unserved_opt_out,     # behavioural
            "service_rate":           served / total if total > 0 else 0.0,
            # Behavioural metrics
            "relocation_offers":        reloc_offered,
            "relocation_accepted":      reloc_accepted,
            "relocation_rejected":      reloc_rejected,
            "relocation_acceptance_rate": (
                reloc_accepted / reloc_offered if reloc_offered > 0 else 0.0
            ),
            "choice_offer_count":       choice_offer,
            "choice_base_count":        choice_base,
            "choice_opt_out_count":     choice_opt_out,
            "choice_offer_rate": (
                choice_offer / choice_total if choice_total > 0 else 0.0
            ),
            "choice_base_rate": (
                choice_base / choice_total if choice_total > 0 else 0.0
            ),
            "choice_opt_out_rate": (
                choice_opt_out / choice_total if choice_total > 0 else 0.0
            ),
            # Operational metrics
            "fleet_utilisation":      utilisation,
            "avg_inventory_inactive": avg_inactive,
            "avg_inventory_low":      avg_low,
            "avg_inventory_high":     avg_high,
            "num_snapshots":          len(self.snapshot_times),
        }

    def _avg_battery_composition(self) -> Tuple[float, float, float]:
        """Average (inactive, low, high) counts per snapshot, summed across zones."""
        if not self.inventory_snapshots:
            return 0.0, 0.0, 0.0
        totals = [0, 0, 0]
        for snap in self.inventory_snapshots:
            for inactive, low, high in snap.values():
                totals[0] += inactive
                totals[1] += low
                totals[2] += high
        n = len(self.inventory_snapshots)
        return totals[0] / n, totals[1] / n, totals[2] / n

    def running_stats(self) -> Dict:
        """Return live-access counts without full aggregation — used by snapshot printer."""
        total    = len(self.trip_records)
        served   = sum(1 for r in self.trip_records if r.served)
        no_sup   = sum(1 for r in self.trip_records if r.unserved_reason == UNSERVED_NO_SUPPLY)
        opt_out  = sum(1 for r in self.trip_records if r.unserved_reason == UNSERVED_OPT_OUT)
        reloc_ok = sum(1 for r in self.trip_records if r.relocation_accepted)
        return {
            "total":   total,
            "served":  served,
            "no_supply": no_sup,
            "opt_out":   opt_out,
            "reloc_accepted": reloc_ok,
        }

    def print_summary(self) -> None:
        s = self.summary()
        W = 60
        SEP  = "=" * W
        SEP2 = "-" * W
        print()
        print(SEP)
        print("  SIMULATION COMPLETE — KPI Summary".center(W))
        print(SEP)

        # ── Service metrics ────────────────────────────────────────────────
        print(f"  {'Metric':<32}  {'Value':>10}")
        print(SEP2)
        print(f"  {'Total trip requests':<32}  {s['total_requests']:>10,}")
        print(f"  {'Served trips':<32}  {s['served_trips']:>10,}")
        print(f"  {'Unserved — no supply (EDL)':<32}  {s['unserved_no_supply']:>10,}")
        print(f"  {'Unserved — user opt-out':<32}  {s['unserved_user_opt_out']:>10,}")
        print(f"  {'Service rate':<32}  {s['service_rate']:>9.1%}")
        print(SEP2)

        # ── Relocation metrics ─────────────────────────────────────────────
        print(f"  {'Relocation offers made':<32}  {s['relocation_offers']:>10,}")
        print(f"  {'Relocation accepted':<32}  {s['relocation_accepted']:>10,}")
        print(f"  {'Relocation rejected':<32}  {s['relocation_rejected']:>10,}")
        print(f"  {'Relocation acceptance rate':<32}  {s['relocation_acceptance_rate']:>9.1%}")
        print(f"  {'User choice — offer':<32}  {s['choice_offer_count']:>10,} ({s['choice_offer_rate']:>5.1%})")
        print(f"  {'User choice — base':<32}  {s['choice_base_count']:>10,} ({s['choice_base_rate']:>5.1%})")
        print(f"  {'User choice — opt-out':<32}  {s['choice_opt_out_count']:>10,} ({s['choice_opt_out_rate']:>5.1%})")
        print(SEP2)

        # ── Fleet metrics ──────────────────────────────────────────────────
        print(f"  {'Fleet utilisation':<32}  {s['fleet_utilisation']:>9.1%}")
        print(f"  {'Avg inactive inventory / snap':<32}  {s['avg_inventory_inactive']:>10.1f}")
        print(f"  {'Avg low inventory / snap':<32}  {s['avg_inventory_low']:>10.1f}")
        print(f"  {'Avg high inventory / snap':<32}  {s['avg_inventory_high']:>10.1f}")
        print(f"  {'Inventory snapshots recorded':<32}  {s['num_snapshots']:>10,}")
        print(SEP)
