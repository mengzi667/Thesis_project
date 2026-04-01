from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MasterTrip:
    request_id: int
    origin_zone: int
    destination_zone: int
    time_slot: int
    request_time: float
    trip_duration: float
    trip_distance: float
    user_type: str
    origin_h3: str
    destination_h3: str


def _slots_per_hour(slot_minutes: int) -> int:
    if slot_minutes <= 0 or 60 % slot_minutes != 0:
        raise ValueError("slot_minutes must be a positive divisor of 60")
    return 60 // slot_minutes


def _resolve_dow_set(is_weekend: int, day_of_week: List[int] | None) -> List[int]:
    if day_of_week:
        bad = [d for d in day_of_week if d < 0 or d > 6]
        if bad:
            raise ValueError(f"day_of_week must be in [0..6], got {bad}")
        return sorted(set(int(d) for d in day_of_week))
    return [5, 6] if int(is_weekend) == 1 else [0, 1, 2, 3, 4]


def _build_zone_mapping(df: pd.DataFrame) -> Dict[str, int]:
    h3_ids = sorted(
        set(df["start_h3"].astype(str).unique()).union(set(df["end_h3"].astype(str).unique()))
    )
    return {h3_id: idx + 1 for idx, h3_id in enumerate(h3_ids)}


def _build_hourly_omega_from_avg_od(df: pd.DataFrame) -> Dict[Tuple[str, str, int], float]:
    g = (
        df.groupby(["start_h3", "end_h3", "hour"], as_index=False)["avg_OD_demand"]
        .mean()
        .rename(columns={"avg_OD_demand": "omega"})
    )
    return {
        (str(r["start_h3"]), str(r["end_h3"]), int(r["hour"])): max(0.0, float(r["omega"]))
        for _, r in g.iterrows()
    }


def build_master_trips_from_avg_od(
    avg_od_csv: str,
    output_master_csv: str,
    *,
    output_mapping_csv: str | None,
    is_weekend: int,
    day_of_week: List[int] | None,
    slot_minutes: int,
    slot0_hour: int,
    t_begin: int,
    t_end: int,
    seed: int,
    sample_mode: str,
    user_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
    trip_duration_mean: float = 15.0,
    trip_duration_std: float = 5.0,
    trip_distance_mean: float = 2.0,
    trip_distance_std: float = 0.5,
) -> int:
    if t_end < t_begin:
        raise ValueError("t_end must be >= t_begin")

    s_per_h = _slots_per_hour(slot_minutes)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    df = pd.read_csv(avg_od_csv)
    keep_cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    df = df[keep_cols]

    required = {"start_h3", "end_h3", "hour", "day_of_week", "avg_OD_demand"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"avg_od_csv missing required columns: {sorted(required - set(df.columns))}")

    dow_set = _resolve_dow_set(is_weekend=is_weekend, day_of_week=day_of_week)
    df = df[df["day_of_week"].astype(int).isin(dow_set)].copy()
    df["start_h3"] = df["start_h3"].astype(str)
    df["end_h3"] = df["end_h3"].astype(str)
    df["hour"] = df["hour"].astype(int)
    df["avg_OD_demand"] = pd.to_numeric(df["avg_OD_demand"], errors="coerce").fillna(0.0)

    zone_map = _build_zone_mapping(df)
    omega_h = _build_hourly_omega_from_avg_od(df)
    od_pairs = sorted({(o, d) for o, d, _ in omega_h.keys()})

    trips: List[MasterTrip] = []
    req_id = 0
    for slot in range(t_begin, t_end + 1):
        hour = int((slot0_hour + (slot // s_per_h)) % 24)
        slot_start = float(slot * slot_minutes)
        for o_h3, d_h3 in od_pairs:
            lam_hour = omega_h.get((o_h3, d_h3, hour), 0.0)
            if lam_hour <= 0.0:
                continue
            lam_slot = lam_hour / float(s_per_h)
            if sample_mode == "poisson":
                n = int(np_rng.poisson(lam_slot))
            elif sample_mode == "round":
                n = int(round(lam_slot))
            else:
                raise ValueError("sample_mode must be one of: poisson, round")
            if n <= 0:
                continue
            o = zone_map[o_h3]
            d = zone_map[d_h3]
            for _ in range(n):
                req_id += 1
                trips.append(
                    MasterTrip(
                        request_id=req_id,
                        origin_zone=o,
                        destination_zone=d,
                        time_slot=slot,
                        request_time=slot_start + rng.uniform(0.0, float(slot_minutes)),
                        trip_duration=max(1.0, rng.gauss(trip_duration_mean, trip_duration_std)),
                        trip_distance=max(0.1, rng.gauss(trip_distance_mean, trip_distance_std)),
                        user_type=rng.choices(
                            ["price_sensitive", "time_sensitive", "normal"],
                            weights=list(user_weights),
                            k=1,
                        )[0],
                        origin_h3=o_h3,
                        destination_h3=d_h3,
                    )
                )

    trips.sort(key=lambda x: (x.request_time, x.request_id))
    for i, t in enumerate(trips, start=1):
        t.request_id = i

    out_dir = os.path.dirname(output_master_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "request_id",
                "origin_zone",
                "destination_zone",
                "time_slot",
                "request_time",
                "trip_duration",
                "trip_distance",
                "user_type",
                "origin_h3",
                "destination_h3",
            ]
        )
        for t in trips:
            w.writerow(
                [
                    t.request_id,
                    t.origin_zone,
                    t.destination_zone,
                    t.time_slot,
                    round(t.request_time, 3),
                    round(t.trip_duration, 3),
                    round(t.trip_distance, 5),
                    t.user_type,
                    t.origin_h3,
                    t.destination_h3,
                ]
            )

    if output_mapping_csv:
        map_dir = os.path.dirname(output_mapping_csv)
        if map_dir:
            os.makedirs(map_dir, exist_ok=True)
        map_df = (
            pd.DataFrame(
                [{"zone_id": zone_id, "h3_id": h3_id} for h3_id, zone_id in zone_map.items()]
            )
            .sort_values("zone_id")
            .reset_index(drop=True)
        )
        map_df.to_csv(output_mapping_csv, index=False)

    return len(trips)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build master trip sample from avg OD demand (H3, hour, day_of_week)."
    )
    p.add_argument(
        "--avg-od-csv",
        default="data/Yumeng Data Files/avg_OD_demand_day_hour_within_region.csv",
    )
    p.add_argument("--output-master", required=True)
    p.add_argument("--output-mapping", default="")
    p.add_argument("--is-weekend", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--day-of-week",
        type=int,
        nargs="*",
        default=None,
        help="optional explicit day-of-week set [0..6], overrides --is-weekend",
    )
    p.add_argument("--slot-minutes", type=int, default=15)
    p.add_argument("--slot0-hour", type=int, default=6)
    p.add_argument("--t-begin", type=int, default=0)
    p.add_argument("--t-end", type=int, default=56)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-mode", choices=["poisson", "round"], default="poisson")
    args = p.parse_args()

    n = build_master_trips_from_avg_od(
        avg_od_csv=args.avg_od_csv,
        output_master_csv=args.output_master,
        output_mapping_csv=(args.output_mapping or None),
        is_weekend=args.is_weekend,
        day_of_week=args.day_of_week,
        slot_minutes=args.slot_minutes,
        slot0_hour=args.slot0_hour,
        t_begin=args.t_begin,
        t_end=args.t_end,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )
    dow_label = (
        ",".join(str(x) for x in args.day_of_week) if args.day_of_week else ("5,6" if args.is_weekend else "0,1,2,3,4")
    )
    print(
        f"master_trips generated: {args.output_master} | rows={n} | "
        f"slot={args.t_begin}..{args.t_end} | mode={args.sample_mode} | "
        f"seed={args.seed} | dow={dow_label}"
    )
    if args.output_mapping:
        print(f"zone mapping generated: {args.output_mapping}")


if __name__ == "__main__":
    main()
