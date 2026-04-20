from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def build_hourly_phi_from_avg_od(
    avg_od_csv: str,
    phi_min: float,
    phi_max: float,
    a: float = 2.0,
    b: float = 0.1,
) -> pd.DataFrame:
    df = pd.read_csv(avg_od_csv)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    req = {"day_of_week", "hour", "avg_OD_demand"}
    if not req.issubset(df.columns):
        raise ValueError(f"missing required columns in avg OD file: {sorted(req)}")

    # Network total demand by (day, hour), based on average OD table.
    net_dh = (
        df.groupby(["day_of_week", "hour"], as_index=False)["avg_OD_demand"]
        .sum()
        .rename(columns={"avg_OD_demand": "network_total"})
    )

    rows = []
    for hour, g in net_dh.groupby("hour"):
        x = g["network_total"].astype(float).to_numpy()
        mu = float(np.mean(x)) if len(x) else 0.0
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        cv = (sd / mu) if mu > 1e-12 else 0.0
        phi = _clip(a * cv + b, phi_min, phi_max)
        rows.append(
            {
                "phi_mode_key": "hourly_cv",
                "is_weekend": -1,
                "hour": int(hour),
                "phi": float(phi),
                "cv": float(cv),
                "network_total_mean": float(mu),
                "network_total_std": float(sd),
            }
        )

    out = pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NB phi profile from aggregated OD averages.")
    p.add_argument(
        "--avg-od-csv",
        type=str,
        default="data/Yumeng Data Files/avg_OD_demand_day_hour_within_region.csv",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="data/input/nb_phi_profile.csv",
    )
    p.add_argument("--phi-min", type=float, default=0.05)
    p.add_argument("--phi-max", type=float, default=5.0)
    p.add_argument("--a", type=float, default=2.0, help="phi = clip(a*CV + b, phi_min, phi_max)")
    p.add_argument("--b", type=float, default=0.1, help="phi = clip(a*CV + b, phi_min, phi_max)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_hourly_phi_from_avg_od(
        avg_od_csv=args.avg_od_csv,
        phi_min=float(args.phi_min),
        phi_max=float(args.phi_max),
        a=float(args.a),
        b=float(args.b),
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"phi profile written: {out_path}")
    print(
        "phi(min/p50/max)="
        f"{out['phi'].min():.4f}/{out['phi'].median():.4f}/{out['phi'].max():.4f}"
    )


if __name__ == "__main__":
    main()

