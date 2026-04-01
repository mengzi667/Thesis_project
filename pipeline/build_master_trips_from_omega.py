from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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


def _slots_per_hour(slot_minutes: int) -> int:
    if slot_minutes <= 0 or 60 % slot_minutes != 0:
        raise ValueError("slot_minutes must be a positive divisor of 60")
    return 60 // slot_minutes


def _load_omega(
    omega_csv: str,
    is_weekend: int,
    zone_ids: set[int] | None = None,
) -> List[Tuple[int, int, int, float]]:
    df = pd.read_csv(omega_csv)
    keep_cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    df = df[keep_cols]
    df = df[df["is_weekend"].astype(int) == int(is_weekend)]

    rows: List[Tuple[int, int, int, float]] = []
    for _, r in df.iterrows():
        o = int(r["start_station"])
        d = int(r["end_station"])
        h = int(r["hour"])
        w = max(0.0, float(r["omega"]))
        if zone_ids is not None and (o not in zone_ids or d not in zone_ids):
            continue
        rows.append((o, d, h, w))
    return rows


def build_master_trips(
    omega_csv: str,
    output_csv: str,
    *,
    is_weekend: int,
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

    omega_rows = _load_omega(omega_csv, is_weekend=is_weekend)
    # Hourly omega map for quick lookup
    omega_h: Dict[Tuple[int, int, int], float] = {(o, d, h): w for o, d, h, w in omega_rows}
    od_pairs = sorted({(o, d) for o, d, _, _ in omega_rows})

    trips: List[MasterTrip] = []
    req_id = 0
    for slot in range(t_begin, t_end + 1):
        hour = int((slot0_hour + (slot // s_per_h)) % 24)
        slot_start = float(slot * slot_minutes)
        for o, d in od_pairs:
            lam_hour = omega_h.get((o, d, hour), 0.0)
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
            for _ in range(n):
                req_id += 1
                t_req = slot_start + rng.uniform(0.0, float(slot_minutes))
                dur = max(1.0, rng.gauss(trip_duration_mean, trip_duration_std))
                dist = max(0.1, rng.gauss(trip_distance_mean, trip_distance_std))
                user_type = rng.choices(
                    ["price_sensitive", "time_sensitive", "normal"],
                    weights=list(user_weights),
                    k=1,
                )[0]
                trips.append(
                    MasterTrip(
                        request_id=req_id,
                        origin_zone=o,
                        destination_zone=d,
                        time_slot=slot,
                        request_time=t_req,
                        trip_duration=dur,
                        trip_distance=dist,
                        user_type=user_type,
                    )
                )

    trips.sort(key=lambda x: (x.request_time, x.request_id))
    for i, t in enumerate(trips, start=1):
        t.request_id = i

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
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
                ]
            )
    return len(trips)


def main() -> None:
    p = argparse.ArgumentParser(description="Build deterministic master trip sample from omega_h.")
    p.add_argument("--omega-csv", default="sara_repo/data/30sep-omega_h.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--is-weekend", type=int, default=1, choices=[0, 1])
    p.add_argument("--slot-minutes", type=int, default=15)
    p.add_argument("--slot0-hour", type=int, default=6)
    p.add_argument("--t-begin", type=int, default=0)
    p.add_argument("--t-end", type=int, default=56)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-mode", choices=["poisson", "round"], default="poisson")
    args = p.parse_args()

    n = build_master_trips(
        omega_csv=args.omega_csv,
        output_csv=args.output,
        is_weekend=args.is_weekend,
        slot_minutes=args.slot_minutes,
        slot0_hour=args.slot0_hour,
        t_begin=args.t_begin,
        t_end=args.t_end,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )
    print(
        f"master_trips generated: {args.output} | rows={n} | "
        f"slot={args.t_begin}..{args.t_end} | mode={args.sample_mode} | seed={args.seed}"
    )


if __name__ == "__main__":
    main()

