#!/usr/bin/env python3
"""
Build a paired manicure retouch dataset from raw before/after folders.

Expected raw layout:
  raw/
    pair_0001/
      before.jpg
      after.jpg

Output layout:
  dataset/processed/
    train/
      images/
      metadata.jsonl
    val/
      images/
      metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, glossy nail surface, natural skin texture, realistic hand photo"
)
DEFAULT_NEGATIVE_PROMPT = (
    "deformed fingers, extra nails, over-smoothed skin, plastic texture, "
    "unrealistic shine, distorted hand, blurred nail edges"
)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    before_path: Path
    after_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess manicure before/after pairs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--size", type=int, default=768, help="Square export size.")
    parser.add_argument("--val-ratio", type=float, default=0.125, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Export format for processed images.",
    )
    parser.add_argument(
        "--resample",
        choices=["lanczos", "bicubic"],
        default="lanczos",
        help="Resize filter for final export.",
    )
    return parser.parse_args()


def find_image(base: Path, stem: str) -> Path | None:
    for ext in VALID_EXTENSIONS:
        candidate = base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_pairs(raw_dir: Path) -> list[PairRecord]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    pairs: list[PairRecord] = []
    for pair_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")):
        before = find_image(pair_dir, "before")
        after = find_image(pair_dir, "after")
        if before is None or after is None:
            raise FileNotFoundError(f"Missing before/after image in {pair_dir}")
        pairs.append(PairRecord(pair_id=pair_dir.name, before_path=before, after_path=after))

    if not pairs:
        raise ValueError(f"No pair_* directories found in {raw_dir}")
    return pairs


def center_crop_shared_area(before: Image.Image, after: Image.Image) -> tuple[Image.Image, Image.Image]:
    shared_width = min(before.width, after.width)
    shared_height = min(before.height, after.height)
    return crop_center(before, shared_width, shared_height), crop_center(after, shared_width, shared_height)


def crop_center(image: Image.Image, width: int, height: int) -> Image.Image:
    left = max((image.width - width) // 2, 0)
    top = max((image.height - height) // 2, 0)
    return image.crop((left, top, left + width, top + height))


def fit_to_square(image: Image.Image, size: int, resample: int) -> Image.Image:
    return ImageOps.fit(image, (size, size), method=resample, centering=(0.5, 0.5))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_pair(
    record: PairRecord,
    split_dir: Path,
    size: int,
    image_format: str,
    resample: int,
) -> tuple[str, str]:
    ensure_dir(split_dir / "images")

    before = ImageOps.exif_transpose(Image.open(record.before_path)).convert("RGB")
    after = ImageOps.exif_transpose(Image.open(record.after_path)).convert("RGB")

    before, after = center_crop_shared_area(before, after)
    before = fit_to_square(before, size=size, resample=resample)
    after = fit_to_square(after, size=size, resample=resample)

    suffix = ".png" if image_format == "png" else ".jpg"
    before_name = f"{record.pair_id}_before{suffix}"
    after_name = f"{record.pair_id}_after{suffix}"

    before_out = split_dir / "images" / before_name
    after_out = split_dir / "images" / after_name

    save_kwargs = {"format": image_format.upper()}
    if image_format == "jpg":
        save_kwargs["quality"] = 95

    before.save(before_out, **save_kwargs)
    after.save(after_out, **save_kwargs)

    return before_name, after_name


def build_metadata_entry(pair_id: str, before_name: str, after_name: str) -> dict[str, str]:
    return {
        "id": pair_id,
        "source": f"images/{before_name}",
        "target": f"images/{after_name}",
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir

    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
    }
    resample = resample_map[args.resample]

    pairs = collect_pairs(raw_dir)
    rng = random.Random(args.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * args.val_ratio))
    val_ids = {record.pair_id for record in shuffled[:val_count]}

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    split_counts = {"train": 0, "val": 0}

    for record in pairs:
        split = "val" if record.pair_id in val_ids else "train"
        split_dir = output_dir / split
        before_name, after_name = export_pair(
            record=record,
            split_dir=split_dir,
            size=args.size,
            image_format=args.image_format,
            resample=resample,
        )
        split_rows[split].append(build_metadata_entry(record.pair_id, before_name, after_name))
        split_counts[split] += 1

    ensure_dir(output_dir / "train")
    ensure_dir(output_dir / "val")
    write_jsonl(output_dir / "train" / "metadata.jsonl", split_rows["train"])
    write_jsonl(output_dir / "val" / "metadata.jsonl", split_rows["val"])

    summary = {
        "raw_pairs": len(pairs),
        "train_pairs": split_counts["train"],
        "val_pairs": split_counts["val"],
        "export_size": args.size,
        "image_format": args.image_format,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
