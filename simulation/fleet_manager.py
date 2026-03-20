# fleet_manager.py
# Individual scooter state and zone-level inventory management.
#
# FleetManager keeps scooter-level records as the single source of truth;
# zone-level inventory counters (X_i^n, X_i^l, X_i^h) are derived from these
# records and updated incrementally on every pickup / drop-off event.

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from config import (
    FLEET_SIZE,
    BATTERY_HIGH_TO_INACTIVE_POLICY,
    BATTERY_INACTIVE_THRESHOLD,
    BATTERY_LOW_THRESHOLD,
    BATTERY_MIN_PRIMARY_N_FROM,
    BATTERY_TRANSITION_CSV,
    PHI_HL,
    PHI_LN,
    SIM_IS_WEEKEND,
    SIM_START_HOUR,
    INITIAL_BATTERY_MEAN,
    INITIAL_BATTERY_STD,
)
from simulation.spatial_system import SpatialSystem
from simulation.battery_transition import BatteryTransitionModel, TransitionContext


# ── Enumerations (plain strings for readability in logs) ─────────────────────

class BatteryCategory:
    INACTIVE = "inactive"   # battery < 10 %  — not rentable
    LOW      = "low"        # 10 % – 25 %     — rentable
    HIGH     = "high"       # > 25 %           — rentable (preferred)


class ScooterStatus:
    IDLE        = "idle"
    IN_TRIP     = "in_trip"
    UNAVAILABLE = "unavailable"   # battery depleted after a trip


# ── Scooter ───────────────────────────────────────────────────────────────────

class Scooter:
    """Represents a single e-scooter with full individual-level state."""

    def __init__(self, scooter_id: int, current_zone: int, battery_level: float) -> None:
        self.scooter_id: int = scooter_id
        self.current_zone: int = current_zone
        self.battery_level: float = max(0.0, min(1.0, battery_level))
        self.battery_category: str = _classify_battery(self.battery_level)
        self.status: str = ScooterStatus.IDLE
        self.available_time: float = 0.0   # earliest time at which scooter can be rented

    def is_rentable(self) -> bool:
        """A scooter is rentable when idle and battery category is low or high."""
        return (
            self.status == ScooterStatus.IDLE
            and self.battery_category in (BatteryCategory.LOW, BatteryCategory.HIGH)
        )

    def consume_battery(
        self,
        arrival_time: float,
        transition_model: Optional[BatteryTransitionModel],
        rng: random.Random,
        sim_is_weekend: int,
        sim_start_hour: int,
    ) -> None:
        """
        Sara-consistent Markov battery transition at trip completion.

        Current implementation uses CSV-based time-conditioned Markov rows:
          P(next_state | init_state, is_weekend, hour)
        Fallback keeps legacy fixed-rate Markov when CSV cannot be loaded.
        """

        old_category = self.battery_category

        if transition_model is not None:
            hour = int((sim_start_hour + (arrival_time // 60)) % 24)
            ctx = TransitionContext(is_weekend=int(sim_is_weekend), hour=hour)
            self.battery_category = transition_model.sample_next_state(
                init_state=old_category,
                context=ctx,
                rng=rng,
            )
        else:
            # Compatibility fallback to fixed-rate Markov if CSV model is unavailable.
            if old_category == BatteryCategory.HIGH:
                self.battery_category = (
                    BatteryCategory.LOW if rng.random() < PHI_HL else BatteryCategory.HIGH
                )
            elif old_category == BatteryCategory.LOW:
                self.battery_category = (
                    BatteryCategory.INACTIVE if rng.random() < PHI_LN else BatteryCategory.LOW
                )
            else:
                self.battery_category = BatteryCategory.INACTIVE

        # Keep SOC display consistent with the discrete category after transition.
        self.battery_level = _representative_level(self.battery_category)

        if self.battery_category == BatteryCategory.INACTIVE:
            self.status = ScooterStatus.UNAVAILABLE

    def __repr__(self) -> str:
        return (
            f"Scooter(id={self.scooter_id}, zone={self.current_zone}, "
            f"bat={self.battery_level:.1%}, cat={self.battery_category}, "
            f"status={self.status})"
        )


# ── FleetManager ─────────────────────────────────────────────────────────────

class FleetManager:
    """
    Maintains the complete scooter fleet and keeps zone-level inventory
    counters consistent with individual scooter records.
    """

    def __init__(
        self,
        spatial_system: SpatialSystem,
        rng: Optional[random.Random] = None,
        sim_is_weekend: int = SIM_IS_WEEKEND,
        sim_start_hour: int = SIM_START_HOUR,
    ) -> None:
        self.spatial = spatial_system
        self.scooters: Dict[int, Scooter] = {}
        self.rng = rng or random.Random()
        self.sim_is_weekend = int(sim_is_weekend)
        self.sim_start_hour = int(sim_start_hour)
        self.battery_transition_model: Optional[BatteryTransitionModel] = None

        try:
            self.battery_transition_model = BatteryTransitionModel.from_csv(
                csv_path=BATTERY_TRANSITION_CSV,
                high_to_inactive_policy=BATTERY_HIGH_TO_INACTIVE_POLICY,
                min_primary_n_from=BATTERY_MIN_PRIMARY_N_FROM,
            )
            self.battery_transition_model.validate()
        except Exception as exc:
            print(
                "[FleetManager] warning: failed to load battery transition CSV; "
                f"falling back to fixed Markov rates. detail={exc}"
            )

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize_fleet(
        self,
        fleet_size: int = FLEET_SIZE,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Distribute *fleet_size* scooters across all zones (round-robin) with
        randomly sampled battery levels, then rebuild zone inventory counters.
        """
        if rng is None:
            rng = self.rng
        else:
            self.rng = rng
        zone_ids = self.spatial.all_zone_ids()
        self.scooters.clear()

        for sid in range(fleet_size):
            zone_id = zone_ids[sid % len(zone_ids)]
            battery = float(
                max(0.0, min(1.0, rng.gauss(INITIAL_BATTERY_MEAN, INITIAL_BATTERY_STD)))
            )
            scooter = Scooter(scooter_id=sid, current_zone=zone_id, battery_level=battery)
            self.scooters[sid] = scooter

        self._rebuild_zone_inventories()

    def initialize_fleet_from_zone_state(
        self,
        zone_state: Dict[int, Tuple[int, int, int]],
    ) -> None:
        """
        Deterministically initialize scooters from zone-level counts:
          zone_state[z] = (inactive_count, low_count, high_count)
        """
        self.scooters.clear()
        sid = 0

        for zone_id in sorted(zone_state.keys()):
            n_cnt, l_cnt, h_cnt = zone_state[zone_id]
            for _ in range(max(0, int(n_cnt))):
                s = Scooter(scooter_id=sid, current_zone=zone_id, battery_level=_representative_level(BatteryCategory.INACTIVE))
                s.status = ScooterStatus.UNAVAILABLE
                self.scooters[sid] = s
                sid += 1
            for _ in range(max(0, int(l_cnt))):
                s = Scooter(scooter_id=sid, current_zone=zone_id, battery_level=_representative_level(BatteryCategory.LOW))
                s.status = ScooterStatus.IDLE
                self.scooters[sid] = s
                sid += 1
            for _ in range(max(0, int(h_cnt))):
                s = Scooter(scooter_id=sid, current_zone=zone_id, battery_level=_representative_level(BatteryCategory.HIGH))
                s.status = ScooterStatus.IDLE
                self.scooters[sid] = s
                sid += 1

        self._rebuild_zone_inventories()

    # ── Inventory helpers ─────────────────────────────────────────────────────

    def _rebuild_zone_inventories(self) -> None:
        """Full recompute of all zone inventory counters from scooter records."""
        for zone in self.spatial.zones.values():
            zone.inventory_inactive = 0
            zone.inventory_low = 0
            zone.inventory_high = 0
        for scooter in self.scooters.values():
            if scooter.status != ScooterStatus.IN_TRIP:
                _delta_zone_inventory(
                    self.spatial.get_zone(scooter.current_zone),
                    scooter.battery_category,
                    delta=+1,
                )

    def _update_inventory(self, zone_id: int, category: str, delta: int) -> None:
        """Incremental single-scooter update to a zone's inventory counter."""
        _delta_zone_inventory(self.spatial.get_zone(zone_id), category, delta)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_available_scooters(
        self, zone_id: int, current_time: float = 0.0
    ) -> List[Scooter]:
        """
        Return all rentable scooters at *zone_id* whose available_time <= current_time.
        Results are sorted: high-battery scooters first, then descending battery level.
        """
        candidates = [
            s for s in self.scooters.values()
            if (
                s.current_zone == zone_id
                and s.is_rentable()
                and s.available_time <= current_time
            )
        ]
        candidates.sort(
            key=lambda s: (
                0 if s.battery_category == BatteryCategory.HIGH else 1,
                -s.battery_level,
            )
        )
        return candidates

    # ── Trip execution ────────────────────────────────────────────────────────

    def pickup_scooter(self, scooter: Scooter) -> None:
        """
        Mark the scooter as in-trip and decrement the origin zone's inventory.
        """
        self._update_inventory(scooter.current_zone, scooter.battery_category, delta=-1)
        scooter.status = ScooterStatus.IN_TRIP

    def dropoff_scooter(
        self,
        scooter: Scooter,
        dest_zone: int,
        arrival_time: float,
    ) -> None:
        """
        Complete the trip:
          1. Apply battery consumption and update battery category.
          2. Relocate scooter to dest_zone.
          3. Update scooter status (idle or unavailable if battery depleted).
          4. Increment the destination zone's inventory.
        """
        scooter.consume_battery(
            arrival_time=arrival_time,
            transition_model=self.battery_transition_model,
            rng=self.rng,
            sim_is_weekend=self.sim_is_weekend,
            sim_start_hour=self.sim_start_hour,
        )
        scooter.current_zone = dest_zone
        scooter.available_time = arrival_time

        # Status is either already UNAVAILABLE (set by consume_battery) or restored to IDLE
        if scooter.status == ScooterStatus.IN_TRIP:
            scooter.status = ScooterStatus.IDLE

        self._update_inventory(dest_zone, scooter.battery_category, delta=+1)


# ── Module-level helper ───────────────────────────────────────────────────────

def _classify_battery(level: float) -> str:
    if level < BATTERY_INACTIVE_THRESHOLD:
        return BatteryCategory.INACTIVE
    if level <= BATTERY_LOW_THRESHOLD:
        return BatteryCategory.LOW
    return BatteryCategory.HIGH


def _representative_level(category: str) -> float:
    if category == BatteryCategory.INACTIVE:
        return BATTERY_INACTIVE_THRESHOLD * 0.5
    if category == BatteryCategory.LOW:
        return (BATTERY_INACTIVE_THRESHOLD + BATTERY_LOW_THRESHOLD) / 2.0
    return min(1.0, max(BATTERY_LOW_THRESHOLD + 0.05, 0.6))


def _delta_zone_inventory(zone, category: str, delta: int) -> None:
    if category == BatteryCategory.INACTIVE:
        zone.inventory_inactive += delta
    elif category == BatteryCategory.LOW:
        zone.inventory_low += delta
    elif category == BatteryCategory.HIGH:
        zone.inventory_high += delta
