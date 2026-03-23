#!/usr/bin/env python3
"""
Create overview sheets for before/after pairs to help manual dataset curation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CELL_W = 256
CELL_H = 256
LABEL_H = 28
PAD = 8
BG = (20, 20, 22)
FG = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pair overview sheets.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        nargs="+",
        default=[Path("dataset/processed/train/images")],
        help="One or more directories containing pair_xxxx_before/after images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/comparisons/train_pair_overview.png"),
        help="Output sheet path.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of pair groups per row.",
    )
    parser.add_argument(
        "--pairs-per-page",
        type=int,
        default=18,
        help="Maximum number of pairs per output page.",
    )
    return parser.parse_args()


def discover_pairs(images_dirs: list[Path]) -> list[tuple[str, Path, Path]]:
    pair_map: dict[str, tuple[Path, Path]] = {}
    for images_dir in images_dirs:
        before_paths = {
            path.name.removesuffix("_before.png"): path
            for path in images_dir.glob("pair_*_before.png")
        }
        after_paths = {
            path.name.removesuffix("_after.png"): path
            for path in images_dir.glob("pair_*_after.png")
        }
        for pair_id, before_path in sorted(before_paths.items()):
            after_path = after_paths.get(pair_id)
            if after_path is None:
                raise FileNotFoundError(f"Missing after image for {pair_id} in {images_dir}")
            pair_map[pair_id] = (before_path, after_path)

    if not pair_map:
        joined = ", ".join(str(path) for path in images_dirs)
        raise FileNotFoundError(f"No pair images found in: {joined}")
    return [(pair_id, before_path, after_path) for pair_id, (before_path, after_path) in sorted(pair_map.items())]


def load_resized(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((CELL_W, CELL_H), Image.Resampling.LANCZOS)


def chunk_pairs(pairs: list[tuple[str, Path, Path]], size: int) -> list[list[tuple[str, Path, Path]]]:
    return [pairs[index : index + size] for index in range(0, len(pairs), size)]


def output_path_for_page(base_output: Path, page_index: int, total_pages: int) -> Path:
    if total_pages == 1:
        return base_output
    return base_output.with_name(f"{base_output.stem}_{page_index:02d}{base_output.suffix}")


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.images_dir)
    pages = chunk_pairs(pairs, args.pairs_per_page)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    group_w = (CELL_W * 2) + (PAD * 3)
    group_h = CELL_H + LABEL_H + (PAD * 2)
    font = ImageFont.load_default()

    for page_index, page_pairs in enumerate(pages, start=1):
        rows = (len(page_pairs) + args.columns - 1) // args.columns
        sheet = Image.new(
            "RGB",
            (group_w * args.columns, group_h * rows),
            BG,
        )
        draw = ImageDraw.Draw(sheet)

        for index, (pair_id, before_path, after_path) in enumerate(page_pairs):
            row = index // args.columns
            col = index % args.columns
            x0 = col * group_w
            y0 = row * group_h

            before = load_resized(before_path)
            after = load_resized(after_path)

            draw.text((x0 + PAD, y0 + PAD), f"{pair_id}  before | after", fill=FG, font=font)
            sheet.paste(before, (x0 + PAD, y0 + PAD + LABEL_H))
            sheet.paste(after, (x0 + (PAD * 2) + CELL_W, y0 + PAD + LABEL_H))

        output_path = output_path_for_page(args.output, page_index, len(pages))
        sheet.save(output_path)
        print(f"Saved overview sheet: {output_path}")


if __name__ == "__main__":
    main()
