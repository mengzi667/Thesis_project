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
        unserved = total - served

        reloc_offered  = sum(1 for r in self.trip_records if r.relocation_offered)
        reloc_accepted = sum(1 for r in self.trip_records if r.relocation_accepted)
        reloc_rejected = reloc_offered - reloc_accepted

        # Fleet utilisation: fraction of recorded trips where a scooter was assigned
        utilisation = served / total if total > 0 else 0.0

        # Battery composition averaged over inventory snapshots
        avg_inactive, avg_low, avg_high = self._avg_battery_composition()

        return {
            # System metrics
            "total_requests":      total,
            "served_trips":        served,
            "unserved_trips":      unserved,
            "service_rate":        served / total if total > 0 else 0.0,
            # Behavioural metrics
            "relocation_offers":        reloc_offered,
            "relocation_accepted":      reloc_accepted,
            "relocation_rejected":      reloc_rejected,
            "relocation_acceptance_rate": (
                reloc_accepted / reloc_offered if reloc_offered > 0 else 0.0
            ),
            # Operational metrics
            "fleet_utilisation":   utilisation,
            "avg_inventory_inactive": avg_inactive,
            "avg_inventory_low":      avg_low,
            "avg_inventory_high":     avg_high,
            "num_snapshots":       len(self.snapshot_times),
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

    def print_summary(self) -> None:
        s = self.summary()
        sep = "=" * 52
        print(sep)
        print("  Simulation KPI Summary")
        print(sep)
        print(f"  Total trip requests    : {s['total_requests']}")
        print(f"  Served trips           : {s['served_trips']}")
        print(f"  Unserved trips         : {s['unserved_trips']}")
        print(f"  Service rate           : {s['service_rate']:.1%}")
        print(sep)
        print(f"  Relocation offers      : {s['relocation_offers']}")
        print(f"  Relocation accepted    : {s['relocation_accepted']}")
        print(f"  Relocation rejected    : {s['relocation_rejected']}")
        print(f"  Relocation acc. rate   : {s['relocation_acceptance_rate']:.1%}")
        print(sep)
        print(f"  Fleet utilisation      : {s['fleet_utilisation']:.1%}")
        print(f"  Avg inactive inventory : {s['avg_inventory_inactive']:.1f}")
        print(f"  Avg low      inventory : {s['avg_inventory_low']:.1f}")
        print(f"  Avg high     inventory : {s['avg_inventory_high']:.1f}")
        print(sep)
