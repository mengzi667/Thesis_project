from __future__ import annotations

import argparse

from config import OR_SLOT_MINUTES
from or_model.sara_adapter import convert_sara_output_to_uodit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Sara OR output to standard U_odit (csv/json)."
    )
    parser.add_argument("--input", required=True, help="Sara output path (.csv/.json)")
    parser.add_argument("--output", required=True, help="U_odit output path (.csv/.json)")
    parser.add_argument("--slot-minutes", type=int, default=OR_SLOT_MINUTES)
    args = parser.parse_args()

    rows = convert_sara_output_to_uodit(
        input_path=args.input,
        output_path=args.output,
        slot_minutes=args.slot_minutes,
    )
    print(
        f"U_odit generated: {args.output} | rows={len(rows)} | slot_minutes={args.slot_minutes}"
    )


if __name__ == "__main__":
    main()
