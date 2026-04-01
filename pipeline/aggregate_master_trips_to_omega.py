from __future__ import annotations

import argparse
import os

import pandas as pd


def aggregate_master_to_omega(
    master_csv: str,
    output_csv: str,
    *,
    is_weekend: int,
    slot_minutes: int,
    slot0_hour: int,
) -> int:
    if slot_minutes <= 0 or 60 % slot_minutes != 0:
        raise ValueError("slot_minutes must be a positive divisor of 60")
    slots_per_hour = 60 // slot_minutes

    df = pd.read_csv(master_csv)
    need = {"origin_zone", "destination_zone"}
    if not need.issubset(df.columns):
        raise ValueError(f"master trips must include columns: {sorted(need)}")

    if "time_slot" not in df.columns:
        if "request_time" not in df.columns:
            raise ValueError("master trips must include time_slot or request_time")
        df["time_slot"] = (df["request_time"].astype(float) // float(slot_minutes)).astype(int)

    df["hour"] = ((slot0_hour + (df["time_slot"].astype(int) // slots_per_hour)) % 24).astype(int)
    g = (
        df.groupby(["origin_zone", "destination_zone", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )

    out = pd.DataFrame(
        {
            "is_weekend": int(is_weekend),
            "start_station": g["origin_zone"].astype(int),
            "end_station": g["destination_zone"].astype(int),
            "hour": g["hour"].astype(int),
            "omega": g["trip_count"].astype(float),
        }
    )

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return len(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate master trip sample to Sara omega_h format.")
    p.add_argument("--master", required=True, help="master_trips.csv path")
    p.add_argument("--output", required=True, help="output omega csv path")
    p.add_argument("--is-weekend", type=int, default=1, choices=[0, 1])
    p.add_argument("--slot-minutes", type=int, default=15)
    p.add_argument("--slot0-hour", type=int, default=6)
    args = p.parse_args()

    n = aggregate_master_to_omega(
        master_csv=args.master,
        output_csv=args.output,
        is_weekend=args.is_weekend,
        slot_minutes=args.slot_minutes,
        slot0_hour=args.slot0_hour,
    )
    print(f"omega aggregated: {args.output} | rows={n}")


if __name__ == "__main__":
    main()
