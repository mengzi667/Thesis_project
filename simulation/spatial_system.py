# spatial_system.py
# Zone structure and adjacency management.
#
# Each Zone represents a geo-fenced parking area.
# SpatialSystem manages the full set of zones and build adjacency from coordinates.

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from config import ZONE_CAPACITY, WALKING_THRESHOLD, NUM_ZONES, GRID_SPACING


class Zone:
    """A geo-fenced parking zone where scooters can be picked up or dropped off."""

    def __init__(
        self,
        zone_id: int,
        capacity: int = ZONE_CAPACITY,
        coordinates: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.zone_id: int = zone_id
        self.capacity: int = capacity
        self.coordinates: Optional[Tuple[float, float]] = coordinates   # (x, y) metres
        self.neighbor_zones: List[int] = []  # zone_ids within walking threshold

        # Zone-level inventory counters — always derived from scooter-level state.
        # These are updated incrementally by FleetManager on every pickup/dropoff.
        self.inventory_inactive: int = 0
        self.inventory_low: int = 0
        self.inventory_high: int = 0

    # ── Inventory helpers ──────────────────────────────────────────────────────

    @property
    def total_inventory(self) -> int:
        return self.inventory_inactive + self.inventory_low + self.inventory_high

    @property
    def rentable_count(self) -> int:
        return self.inventory_low + self.inventory_high

    def satisfies_capacity_constraint(self) -> bool:
        """Check X_i^n + X_i^l + X_i^h <= C_i."""
        return self.total_inventory <= self.capacity

    def state_tuple(self) -> Tuple[int, int, int]:
        """Return (inactive, low, high) for compact state representation X_i^t."""
        return (self.inventory_inactive, self.inventory_low, self.inventory_high)

    def __repr__(self) -> str:
        return (
            f"Zone(id={self.zone_id}, cap={self.capacity}, "
            f"n={self.inventory_inactive}, l={self.inventory_low}, h={self.inventory_high})"
        )


class SpatialSystem:
    """Manages all zones and their walking-distance adjacency relationships."""

    def __init__(
        self,
        zones: List[Zone],
        walking_threshold: float = WALKING_THRESHOLD,
    ) -> None:
        self.zones: Dict[int, Zone] = {z.zone_id: z for z in zones}
        self.walking_threshold = walking_threshold
        self._build_adjacency()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_adjacency(self) -> None:
        """
        Populate neighbor_zones for every zone pair whose centres are within
        walking_threshold metres of each other.
        Zones without coordinates are skipped.
        """
        zone_list = list(self.zones.values())
        for i in range(len(zone_list)):
            for j in range(i + 1, len(zone_list)):
                zi, zj = zone_list[i], zone_list[j]
                if zi.coordinates is None or zj.coordinates is None:
                    continue
                dist = _euclidean(zi.coordinates, zj.coordinates)
                if dist <= self.walking_threshold:
                    zi.neighbor_zones.append(zj.zone_id)
                    zj.neighbor_zones.append(zi.zone_id)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_zone(self, zone_id: int) -> Zone:
        return self.zones[zone_id]

    def get_neighbors(self, zone_id: int) -> List[int]:
        return self.zones[zone_id].neighbor_zones

    def all_zone_ids(self) -> List[int]:
        return list(self.zones.keys())

    def get_state_snapshot(self) -> Dict[int, Tuple[int, int, int]]:
        """Return {zone_id: (inactive, low, high)} for every zone."""
        return {zid: z.state_tuple() for zid, z in self.zones.items()}

    def distance_between(self, zone_a: int, zone_b: int) -> float:
        """
        Return Euclidean distance (metres) between two zones.
        Returns 0.0 if coordinates are unavailable.
        """
        za = self.zones.get(zone_a)
        zb = self.zones.get(zone_b)
        if za and zb and za.coordinates and zb.coordinates:
            return _euclidean(za.coordinates, zb.coordinates)
        return 0.0

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create_grid(
        cls,
        num_zones: int = NUM_ZONES,
        capacity: int = ZONE_CAPACITY,
        grid_spacing: float = GRID_SPACING,
        walking_threshold: float = WALKING_THRESHOLD,
    ) -> "SpatialSystem":
        """
        Build a synthetic rectangular grid of zones.
        Zone centres are spaced *grid_spacing* metres apart.
        """
        side = math.ceil(math.sqrt(num_zones))
        zones: List[Zone] = []
        for idx in range(num_zones):
            row, col = divmod(idx, side)
            coords = (col * grid_spacing, row * grid_spacing)
            zones.append(Zone(zone_id=idx, capacity=capacity, coordinates=coords))
        return cls(zones, walking_threshold=walking_threshold)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
