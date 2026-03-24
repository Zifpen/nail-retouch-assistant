#!/usr/bin/env python3
"""
Build a paired image-edit dataset for pix2pix-turbo style training.

Expected source layout:
  dataset/processed_strict_plus/
    train/
      images/
      metadata.jsonl
    val/
      images/
      metadata.jsonl

Output layout:
  dataset/paired_edit_strict_plus/
    train_A/
    train_B/
    test_A/
    test_B/
    train_prompts.json
    test_prompts.json
    train_metadata.jsonl
    test_metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, preserve original nail design, natural skin texture, realistic hand photo"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired-edit dataset from processed pairs.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset/processed_strict_plus"),
        help="Processed source dataset with train/val metadata.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/paired_edit_strict_plus"),
        help="Destination dataset folder in pix2pix-turbo format.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_split(source_dir: Path, output_dir: Path, split: str, prefix: str) -> int:
    rows = load_rows(source_dir / split / "metadata.jsonl")
    side_a = output_dir / f"{prefix}_A"
    side_b = output_dir / f"{prefix}_B"
    side_a.mkdir(parents=True, exist_ok=True)
    side_b.mkdir(parents=True, exist_ok=True)

    prompts: dict[str, str] = {}
    exported_rows: list[dict[str, str]] = []

    for row in rows:
        file_name = f"{row['id']}.png"
        src_input = source_dir / split / row["source"]
        src_target = source_dir / split / row["target"]

        shutil.copy2(src_input, side_a / file_name)
        shutil.copy2(src_target, side_b / file_name)

        prompt = str(row.get("prompt") or DEFAULT_PROMPT)
        prompts[file_name] = prompt
        exported_rows.append(
            {
                "id": str(row["id"]),
                "input": f"{prefix}_A/{file_name}",
                "target": f"{prefix}_B/{file_name}",
                "prompt": prompt,
            }
        )

    (output_dir / f"{prefix}_prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{prefix}_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in exported_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(exported_rows)


def main() -> None:
    args = parse_args()
    ensure_clean_dir(args.output_dir)

    summary = {
        "train": export_split(args.source_dir, args.output_dir, "train", "train"),
        "test": export_split(args.source_dir, args.output_dir, "val", "test"),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
