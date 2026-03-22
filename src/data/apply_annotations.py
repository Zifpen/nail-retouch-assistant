#!/usr/bin/env python3
"""
Merge structured manicure tags into processed metadata files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, glossy nail surface, natural skin texture, realistic hand photo"
)
DEFAULT_NEGATIVE_PROMPT = (
    "deformed fingers, extra nails, over-smoothed skin, plastic texture, "
    "unrealistic shine, distorted hand, blurred nail edges"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply structured annotations to processed metadata.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/annotations/pair_tags.csv"),
        help="CSV file with pair-level tags.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Processed dataset directory containing train/val metadata.jsonl files.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    annotations: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair_id = row["id"].strip()
            annotations[pair_id] = {
                "shape": row.get("shape", "").strip(),
                "finish": row.get("finish", "").strip(),
                "color_family": row.get("color_family", "").strip(),
            }
    return annotations


def build_prompt(tags: dict[str, str]) -> str:
    parts = [BASE_PROMPT]
    for key in ("shape", "finish", "color_family"):
        value = tags.get(key, "")
        if value:
            parts.append(value.replace("_", " "))
    return ", ".join(parts)


def update_metadata_file(path: Path, annotations: dict[str, dict[str, str]]) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    rows = []
    updated = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tags = annotations.get(row["id"], {})
            if tags:
                row["shape"] = tags["shape"]
                row["finish"] = tags["finish"]
                row["color_family"] = tags["color_family"]
                row["prompt"] = build_prompt(tags)
                row["negative_prompt"] = row.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
                updated += 1
            rows.append(row)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)

    summary = {}
    for split in ("train", "val"):
        path = args.processed_dir / split / "metadata.jsonl"
        summary[split] = update_metadata_file(path, annotations)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
