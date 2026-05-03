from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def build_trip_summary_tables(df: pd.DataFrame, planning_period_min: float = 15.0) -> dict[str, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame()
        return {
            "overall": empty,
            "by_slot": empty,
            "by_origin": empty,
            "by_od": empty,
            "top_od": empty,
        }

    d = df.copy()
    d["served_int"] = d["served"].astype(int)
    d["offered_int"] = d["relocation_offered"].astype(int)
    d["accepted_int"] = d["relocation_accepted"].astype(int)
    d["request_time"] = pd.to_numeric(d["request_time"], errors="coerce").fillna(0.0)
    d["slot"] = np.floor(d["request_time"] / float(planning_period_min)).astype(int)

    total = float(len(d))
    served = float(d["served_int"].sum())
    no_supply = float((d["unserved_reason"] == "no_supply").sum())
    opt_out = float((d["unserved_reason"] == "user_opt_out").sum())
    offers = float(d["offered_int"].sum())
    accepts = float(d["accepted_int"].sum())
    rejects = float(offers - accepts)

    overall = pd.DataFrame(
        [
            {
                "total_trips": int(total),
                "served_trips": int(served),
                "unserved_trips": int(total - served),
                "service_rate": _safe_rate(served, total),
                "unserved_no_supply": int(no_supply),
                "unserved_opt_out": int(opt_out),
                "relocation_offers": int(offers),
                "relocation_accepted": int(accepts),
                "relocation_rejected": int(rejects),
                "relocation_acceptance_rate": _safe_rate(accepts, offers),
            }
        ]
    )

    by_slot = (
        d.groupby("slot", dropna=False)
        .agg(
            trip_count=("request_id", "count"),
            served_trips=("served_int", "sum"),
            offers=("offered_int", "sum"),
            accepts=("accepted_int", "sum"),
        )
        .reset_index()
    )
    by_slot["service_rate"] = by_slot.apply(lambda r: _safe_rate(float(r["served_trips"]), float(r["trip_count"])), axis=1)
    by_slot["accept_rate"] = by_slot.apply(lambda r: _safe_rate(float(r["accepts"]), float(r["offers"])), axis=1)

    by_origin = (
        d.groupby("origin_zone", dropna=False)
        .agg(
            trip_count=("request_id", "count"),
            served_trips=("served_int", "sum"),
            offers=("offered_int", "sum"),
            accepts=("accepted_int", "sum"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )
    by_origin["service_rate"] = by_origin.apply(
        lambda r: _safe_rate(float(r["served_trips"]), float(r["trip_count"])), axis=1
    )
    by_origin["accept_rate"] = by_origin.apply(lambda r: _safe_rate(float(r["accepts"]), float(r["offers"])), axis=1)

    by_od = (
        d.groupby(["origin_zone", "effective_dest"], dropna=False)
        .agg(
            trip_count=("request_id", "count"),
            served_trips=("served_int", "sum"),
            offers=("offered_int", "sum"),
            accepts=("accepted_int", "sum"),
        )
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )
    by_od["service_rate"] = by_od.apply(lambda r: _safe_rate(float(r["served_trips"]), float(r["trip_count"])), axis=1)
    by_od["accept_rate"] = by_od.apply(lambda r: _safe_rate(float(r["accepts"]), float(r["offers"])), axis=1)
    top_od = by_od.head(20).copy()

    return {
        "overall": overall,
        "by_slot": by_slot,
        "by_origin": by_origin,
        "by_od": by_od,
        "top_od": top_od,
    }


def write_trip_run_report(
    trip_rows: Iterable[dict],
    output_dir: Path,
    planning_period_min: float = 15.0,
    file_prefix: str = "trip",
) -> None:
    out_metrics = Path(output_dir) / "metrics"
    out_metrics.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(list(trip_rows))
    if df.empty:
        (out_metrics / f"{file_prefix}_run_summary.md").write_text(
            "# Trip Run Summary\n\nNo trip records available.\n",
            encoding="utf-8",
        )
        return

    tables = build_trip_summary_tables(df=df, planning_period_min=planning_period_min)
    df.to_csv(out_metrics / f"{file_prefix}_records.csv", index=False)
    tables["overall"].to_csv(out_metrics / f"{file_prefix}_summary_overall.csv", index=False)
    tables["by_slot"].to_csv(out_metrics / f"{file_prefix}_summary_by_slot.csv", index=False)
    tables["by_origin"].to_csv(out_metrics / f"{file_prefix}_summary_by_origin.csv", index=False)
    tables["by_od"].to_csv(out_metrics / f"{file_prefix}_summary_by_od.csv", index=False)
    tables["top_od"].to_csv(out_metrics / f"{file_prefix}_summary_top_od.csv", index=False)

    o = tables["overall"].iloc[0].to_dict()
    md_lines = [
        "# Trip Run Summary",
        "",
        "## Overall",
        f"- Total trips: {int(o['total_trips'])}",
        f"- Served trips: {int(o['served_trips'])}",
        f"- Unserved trips: {int(o['unserved_trips'])}",
        f"- Service rate: {float(o['service_rate']):.4f}",
        f"- Unserved (no supply): {int(o['unserved_no_supply'])}",
        f"- Unserved (opt-out): {int(o['unserved_opt_out'])}",
        f"- Relocation offers: {int(o['relocation_offers'])}",
        f"- Relocation accepted: {int(o['relocation_accepted'])}",
        f"- Relocation rejected: {int(o['relocation_rejected'])}",
        f"- Relocation acceptance rate: {float(o['relocation_acceptance_rate']):.4f}",
        "",
        "## Outputs",
        f"- `{file_prefix}_records.csv`",
        f"- `{file_prefix}_summary_overall.csv`",
        f"- `{file_prefix}_summary_by_slot.csv`",
        f"- `{file_prefix}_summary_by_origin.csv`",
        f"- `{file_prefix}_summary_by_od.csv`",
        f"- `{file_prefix}_summary_top_od.csv`",
    ]
    (out_metrics / f"{file_prefix}_run_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
