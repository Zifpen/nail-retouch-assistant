#!/usr/bin/env python3
"""
Build a filtered dataset subset from a processed dataset and a review CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered subset from processed pair data.")
    parser.add_argument(
        "--review",
        type=Path,
        required=True,
        help="CSV review file with id/split/status columns.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Processed dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output subset root.",
    )
    parser.add_argument(
        "--status",
        nargs="+",
        default=["keep"],
        help="Statuses to include, e.g. keep maybe.",
    )
    return parser.parse_args()


def load_review(path: Path, allowed_statuses: set[str]) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {"train": set(), "val": set()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"].strip() not in allowed_statuses:
                continue
            split = row["split"].strip()
            if split not in selected:
                raise ValueError(f"Unsupported split in review file: {split}")
            selected[split].add(row["id"].strip())
    return selected


def rebuild_split(source_dir: Path, output_dir: Path, split: str, selected_ids: set[str]) -> int:
    metadata_in = source_dir / split / "metadata.jsonl"
    images_in = source_dir / split / "images"
    images_out = output_dir / split / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict[str, object]] = []
    with metadata_in.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in selected_ids:
                continue
            kept_rows.append(row)
            for key in ("source", "target"):
                src = source_dir / split / row[key]
                dst = output_dir / split / row[key]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    metadata_out = output_dir / split / "metadata.jsonl"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept_rows)


def main() -> None:
    args = parse_args()
    allowed_statuses = {status.strip() for status in args.status}
    selected = load_review(args.review, allowed_statuses)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val"):
        summary[split] = rebuild_split(args.source_dir, args.output_dir, split, selected[split])

    summary["statuses"] = sorted(allowed_statuses)
    summary["review"] = str(args.review)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
