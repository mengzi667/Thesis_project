from __future__ import annotations

import math
from typing import Dict, List, Tuple

import h3
import pandas as pd

from simulation.spatial_system import SpatialSystem, Zone
from simulation.trip_generator import DemandProfile


def _path(data_dir: str, name: str) -> str:
    return f"{data_dir.rstrip('/\\')}/{name}"


def build_sara_spatial_system(
    data_dir: str,
    zone_capacity: int,
    walking_threshold: float,
) -> SpatialSystem:
    """
    Build zones from Sara's H3 station map.

    Zone IDs follow Sara's convention: 1..N in the row order of h3_station_map.csv.
    Coordinates are converted to a local meter-scale plane relative to first station.
    """
    station_map = pd.read_csv(_path(data_dir, "h3_station_map.csv"))
    if "h3_index" not in station_map.columns:
        raise ValueError("h3_station_map.csv must contain column 'h3_index'")

    latlon: List[Tuple[float, float]] = [h3.h3_to_geo(h) for h in station_map["h3_index"]]
    lat0, lon0 = latlon[0]

    # Local tangent-plane approximation around first station.
    # Adequate for adjacency thresholding at city scale.
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.1, abs(math.cos(math.radians(lat0))))

    zones: List[Zone] = []
    for idx, (lat, lon) in enumerate(latlon, start=1):
        x = (lon - lon0) * m_per_deg_lon
        y = (lat - lat0) * m_per_deg_lat
        zones.append(
            Zone(
                zone_id=idx,
                capacity=zone_capacity,
                coordinates=(float(x), float(y)),
            )
        )

    return SpatialSystem(zones=zones, walking_threshold=walking_threshold)


def build_sara_demand_profile(
    data_dir: str,
    zone_ids: List[int],
    sim_duration: float,
    planning_period: float,
    is_weekend: int,
    slot0_hour: int,
) -> DemandProfile:
    """
    Build DemandProfile from Sara's pickup rates and OD omega table.
    """
    num_slots = max(1, int(sim_duration // planning_period) + 1)
    zone_set = set(zone_ids)

    # -- pickup rates --
    pickup = pd.read_csv(_path(data_dir, "30sep-df_pickup_rates.csv"))
    for c in list(pickup.columns):
        if str(c).startswith("Unnamed:"):
            pickup = pickup.drop(columns=[c])
    pickup = pickup[pickup["is_weekend"].astype(int) == int(is_weekend)]

    hour_cols = [c for c in pickup.columns if str(c).isdigit() and 0 <= int(c) <= 23]
    pickup_by_station_hour: Dict[int, Dict[int, float]] = {}
    for _, row in pickup.iterrows():
        s = int(row["start_station"])
        if s not in zone_set:
            continue
        pickup_by_station_hour[s] = {int(h): float(row[h]) for h in hour_cols}

    zone_rates: Dict[int, Dict[int, float]] = {z: {} for z in zone_ids}
    for z in zone_ids:
        hourly = pickup_by_station_hour.get(z, {})
        for slot in range(num_slots):
            hour = int((slot0_hour + int((slot * planning_period) // 60)) % 24)
            # Sara pickup rates are trips/hour; generator expects trips/min.
            zone_rates[z][slot] = max(0.0, float(hourly.get(hour, 0.0)) / 60.0)

    # -- OD weights from omega_h --
    omega = pd.read_csv(_path(data_dir, "30sep-omega_h.csv"))
    for c in list(omega.columns):
        if str(c).startswith("Unnamed:"):
            omega = omega.drop(columns=[c])
    omega = omega[omega["is_weekend"].astype(int) == int(is_weekend)]

    od_hour_weight: Dict[Tuple[int, int, int], float] = {}
    for _, row in omega.iterrows():
        o = int(row["start_station"])
        d = int(row["end_station"])
        h = int(row["hour"])
        if o in zone_set and d in zone_set:
            od_hour_weight[(o, d, h)] = max(0.0, float(row["omega"]))

    od_weights: Dict[Tuple[int, int], Dict[int, float]] = {}
    for o in zone_ids:
        for d in zone_ids:
            slot_weights: Dict[int, float] = {}
            for slot in range(num_slots):
                hour = int((slot0_hour + int((slot * planning_period) // 60)) % 24)
                slot_weights[slot] = od_hour_weight.get((o, d, hour), 0.0)
            od_weights[(o, d)] = slot_weights

    return DemandProfile(
        planning_period=planning_period,
        zone_rates=zone_rates,
        od_weights=od_weights,
    )


def build_uniform_zone_state(
    zone_ids: List[int],
    n_count: int,
    l_count: int,
    h_count: int,
) -> Dict[int, Tuple[int, int, int]]:
    return {z: (int(n_count), int(l_count), int(h_count)) for z in zone_ids}


def load_zone_state_from_csv(
    csv_path: str,
    zone_ids: List[int],
    default_state: Tuple[int, int, int],
) -> Dict[int, Tuple[int, int, int]]:
    """
    CSV schema: station,n,l,h
    """
    df = pd.read_csv(csv_path)
    need = {"station", "n", "l", "h"}
    if not need.issubset(df.columns):
        raise ValueError(f"initial inventory csv must include columns: {sorted(need)}")

    zone_set = set(zone_ids)
    out = {z: tuple(map(int, default_state)) for z in zone_ids}
    for _, row in df.iterrows():
        z = int(row["station"])
        if z not in zone_set:
            continue
        out[z] = (int(row["n"]), int(row["l"]), int(row["h"]))
    return out
