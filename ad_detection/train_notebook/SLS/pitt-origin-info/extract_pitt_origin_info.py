#!/usr/bin/env python3
"""Extract demographics from Pitt-origin CHAT transcripts into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INFO_DIR = Path(__file__).resolve().parent / "Pitt-origin-info"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pitt_origin_info.csv"


def parse_age(age_text: str) -> str:
    """Return the integer year component from CHAT age fields such as 74;00."""
    match = re.match(r"^(\d+)", age_text.strip())
    return match.group(1) if match else ""


def parse_participant_id(chat_file: Path) -> dict[str, str]:
    """Parse the participant PAR @ID line from a .cha file."""
    par_id = ""
    with chat_file.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("@ID:"):
                continue

            raw_id = line.split("\t", 1)[-1].strip()
            parts = raw_id.split("|")
            if len(parts) > 2 and parts[2] == "PAR":
                par_id = raw_id
                break

    parts = par_id.split("|") if par_id else []
    stem = chat_file.stem
    stem_parts = stem.split("-", 1)

    return {
        "name": stem,
        "participant_id": stem_parts[0],
        "age": parse_age(parts[3]) if len(parts) > 3 else "",
        "sex": parts[4].strip().lower() if len(parts) > 4 else "",
        "transcript_path": str(chat_file.resolve()),
    }


def collect_rows(info_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in ("Control", "Dementia"):
        group_dir = info_dir / label
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Missing transcript directory: {group_dir}")

        for chat_file in sorted(group_dir.glob("*.cha")):
            row = parse_participant_id(chat_file)
            row["label"] = label
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "label",
        "sex",
        "age",
        "participant_id",
        "transcript_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sex, age, and group labels from Pitt-origin transcript @ID lines."
    )
    parser.add_argument(
        "--info-dir",
        type=Path,
        default=DEFAULT_INFO_DIR,
        help=f"Transcript info directory. Default: {DEFAULT_INFO_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    rows = collect_rows(args.info_dir)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
