from __future__ import annotations

import argparse
import os

import pandas as pd


def _infer_uodit(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"origin", "original_dest", "time_slot"}
    if not need.issubset(df.columns):
        raise ValueError(f"u_odit missing columns: {sorted(need - set(df.columns))}")
    return df


def _infer_omega(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"is_weekend", "start_station", "end_station", "hour", "omega"}
    if not need.issubset(df.columns):
        raise ValueError(f"omega missing columns: {sorted(need - set(df.columns))}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check whether u_odit keys exist in input omega under slot->hour mapping."
    )
    p.add_argument("--uodit", required=True)
    p.add_argument("--omega", required=True)
    p.add_argument("--slot0-hour", type=int, default=6)
    p.add_argument("--slot-minutes", type=int, default=15)
    p.add_argument("--is-weekend", type=int, default=1, choices=[0, 1])
    p.add_argument("--output", default="")
    args = p.parse_args()

    if args.slot_minutes <= 0 or 60 % args.slot_minutes != 0:
        raise ValueError("slot-minutes must be a positive divisor of 60")
    slots_per_hour = 60 // args.slot_minutes

    u = _infer_uodit(args.uodit).copy()
    o = _infer_omega(args.omega).copy()

    u["origin"] = u["origin"].astype(int)
    u["original_dest"] = u["original_dest"].astype(int)
    u["time_slot"] = u["time_slot"].astype(int)
    u["hour"] = ((args.slot0_hour + (u["time_slot"] // slots_per_hour)) % 24).astype(int)

    o["is_weekend"] = o["is_weekend"].astype(int)
    o["start_station"] = o["start_station"].astype(int)
    o["end_station"] = o["end_station"].astype(int)
    o["hour"] = o["hour"].astype(int)
    o["omega"] = pd.to_numeric(o["omega"], errors="coerce").fillna(0.0)

    keyset = {
        (int(r.start_station), int(r.end_station), int(r.hour), int(r.is_weekend))
        for r in o.itertuples(index=False)
        if float(r.omega) > 0.0
    }

    rows = []
    for r in u.itertuples(index=False):
        k = (int(r.origin), int(r.original_dest), int(r.hour), int(args.is_weekend))
        rows.append(
            {
                "origin": int(r.origin),
                "original_dest": int(r.original_dest),
                "time_slot": int(r.time_slot),
                "hour": int(r.hour),
                "exists_in_omega": int(k in keyset),
            }
        )

    chk = pd.DataFrame(rows)
    total = len(chk)
    found = int(chk["exists_in_omega"].sum())
    miss = total - found

    print(f"u_odit rows          : {total}")
    print(f"found in omega       : {found}")
    print(f"missing from omega   : {miss}")
    print(f"coverage ratio       : {found/total if total else 0.0:.3f}")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        chk.to_csv(args.output, index=False)
        print(f"detail output        : {args.output}")

    if miss > 0:
        print("\nSample missing keys (first 15):")
        sample = chk[chk["exists_in_omega"] == 0].head(15)
        if len(sample) > 0:
            print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
