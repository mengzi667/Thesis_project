from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List, Tuple


def _pick_first(row: dict, names: List[str], default=None):
    for n in names:
        if n in row and row[n] is not None and str(row[n]).strip() != "":
            return row[n]
    return default


def _parse_standard_row(row: dict, slot_minutes: int) -> Tuple[int, int, int, int, float, int]:
    """
    Parse one raw row from Sara-like output into standard tuple:
      (origin, original_dest, recommended_dest, time_slot, incentive_amount, quota)

    Supported source patterns:
      1) Standard U_odit fields (origin, original_dest, recommended_dest, time_slot, incentive_amount, quota)
      2) Sara incentive export fields (origin, orig_dest, new_dest, depart_slot, redirected_trips, ride_cost_euro)
      3) z-style fields (o,d,i,t,z)
    """
    origin = int(_pick_first(row, ["origin", "o", "start_station"]))
    original_dest = int(_pick_first(row, ["original_dest", "orig_dest", "destination", "d", "end_station"]))
    recommended_dest = int(_pick_first(row, ["recommended_dest", "new_dest", "i", "alt_station"]))

    slot_raw = _pick_first(row, ["time_slot", "depart_slot", "t", "slot"])
    if slot_raw is None:
        minute_raw = _pick_first(row, ["minute", "request_minute", "time_min"])
        if minute_raw is None:
            raise ValueError("missing time_slot/depart_slot and minute fields")
        time_slot = int(float(minute_raw) // float(slot_minutes))
    else:
        time_slot = int(float(slot_raw))

    z_val = _pick_first(row, ["quota", "redirected_trips", "z", "flow", "value"], default=1)
    quota = int(round(float(z_val)))
    if quota < 0:
        quota = 0

    # ride_cost_euro is not an incentive field and must not be mapped as such.
    incentive_raw = _pick_first(row, ["incentive_amount", "incentive"], default=1.0)
    incentive_amount = float(incentive_raw)
    return origin, original_dest, recommended_dest, time_slot, incentive_amount, quota


def _aggregate(rows: Iterable[dict], slot_minutes: int) -> List[dict]:
    agg: Dict[Tuple[int, int, int, int], dict] = {}

    for row in rows:
        origin, original_dest, recommended_dest, time_slot, incentive, quota = _parse_standard_row(row, slot_minutes)
        if quota <= 0:
            continue

        key = (origin, original_dest, recommended_dest, time_slot)
        if key not in agg:
            agg[key] = {
                "origin": origin,
                "original_dest": original_dest,
                "recommended_dest": recommended_dest,
                "time_slot": time_slot,
                "incentive_amount": incentive,
                "quota": quota,
            }
        else:
            agg[key]["quota"] += quota

    return list(agg.values())


def load_sara_rows(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records")
        return data
    raise ValueError(f"unsupported adapter input extension: {ext}")


def convert_sara_output_to_uodit(
    input_path: str,
    output_path: str,
    slot_minutes: int = 15,
) -> List[dict]:
    rows = load_sara_rows(input_path)
    mapped = _aggregate(rows, slot_minutes=slot_minutes)

    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == ".csv":
        fields = ["origin", "original_dest", "recommended_dest", "time_slot", "incentive_amount", "quota"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in mapped:
                writer.writerow(r)
    elif out_ext == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapped, f, indent=2)
    else:
        raise ValueError("output_path must be .csv or .json")

    return mapped
