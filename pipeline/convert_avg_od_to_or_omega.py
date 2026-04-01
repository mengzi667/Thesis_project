from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    return df[keep].copy()


def _build_zone_map(df: pd.DataFrame) -> Dict[str, int]:
    h3_ids = sorted(
        set(df["start_h3"].astype(str).unique()).union(set(df["end_h3"].astype(str).unique()))
    )
    return {h3_id: idx + 1 for idx, h3_id in enumerate(h3_ids)}


def convert_avg_od_to_or_omega(
    avg_od_csv: str,
    output_omega_csv: str,
    *,
    output_mapping_csv: str | None = None,
    slot0_hour: int = 6,
    total_hours: int = 14,
) -> int:
    df = pd.read_csv(avg_od_csv)
    df = _clean_columns(df)

    required = {"start_h3", "end_h3", "hour", "day_of_week", "avg_OD_demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"avg OD file missing required columns: {sorted(missing)}")

    df["start_h3"] = df["start_h3"].astype(str)
    df["end_h3"] = df["end_h3"].astype(str)
    # Input hour is clock hour [0..23]. Sara expects operational-hour index [0..13].
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["avg_OD_demand"] = pd.to_numeric(df["avg_OD_demand"], errors="coerce").fillna(0.0)

    df["op_hour"] = df["hour"] - int(slot0_hour)
    df = df[(df["op_hour"] >= 0) & (df["op_hour"] < int(total_hours))].copy()

    zone_map = _build_zone_map(df)
    df["start_station"] = df["start_h3"].map(zone_map).astype(int)
    df["end_station"] = df["end_h3"].map(zone_map).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Aggregate to expected demand per (weekend_flag, OD, op_hour).
    out = (
        df.groupby(["is_weekend", "start_station", "end_station", "op_hour"], as_index=False)[
            "avg_OD_demand"
        ]
        .mean()
        .rename(columns={"avg_OD_demand": "omega", "op_hour": "hour"})
    )

    out = out[out["omega"] > 0.0].copy()
    out = out.sort_values(["is_weekend", "hour", "start_station", "end_station"]).reset_index(
        drop=True
    )

    out_dir = os.path.dirname(output_omega_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_omega_csv, index=False)

    if output_mapping_csv:
        mdir = os.path.dirname(output_mapping_csv)
        if mdir:
            os.makedirs(mdir, exist_ok=True)
        mapping = (
            pd.DataFrame(
                [{"zone_id": zid, "h3_id": h3} for h3, zid in zone_map.items()],
            )
            .sort_values("zone_id")
            .reset_index(drop=True)
        )
        mapping.to_csv(output_mapping_csv, index=False)

    return len(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert aggregated avg OD demand to OR omega format."
    )
    p.add_argument(
        "--avg-od-csv",
        default="data/Yumeng Data Files/avg_OD_demand_day_hour_within_region.csv",
    )
    p.add_argument("--output-omega", required=True)
    p.add_argument("--output-mapping", default="")
    p.add_argument("--slot0-hour", type=int, default=6)
    p.add_argument("--total-hours", type=int, default=14)
    args = p.parse_args()

    n = convert_avg_od_to_or_omega(
        avg_od_csv=args.avg_od_csv,
        output_omega_csv=args.output_omega,
        output_mapping_csv=(args.output_mapping or None),
        slot0_hour=args.slot0_hour,
        total_hours=args.total_hours,
    )
    print(f"omega generated: {args.output_omega} | rows={n}")
    if args.output_mapping:
        print(f"zone mapping generated: {args.output_mapping}")


if __name__ == "__main__":
    main()
