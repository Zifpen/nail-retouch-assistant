#!/usr/bin/env python3
"""
Create a simple input / output / target comparison sheet for paired-edit experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (18, 18, 20)
LABEL_COLOR = (245, 245, 245)
LABEL_HEIGHT = 42
PADDING = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an input/output/target sheet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(images: list[tuple[str, Image.Image]]) -> tuple[list[tuple[str, Image.Image]], int, int]:
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    resized = []
    for label, image in images:
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append((label, image))
    return resized, width, height


def main() -> None:
    args = parse_args()
    columns, width, height = fit(
        [
            ("input", load(args.input)),
            ("output", load(args.output)),
            ("target", load(args.target)),
        ]
    )
    sheet = Image.new(
        "RGB",
        ((width * 3) + (PADDING * 4), height + LABEL_HEIGHT + (PADDING * 2)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = PADDING
    for label, image in columns:
        sheet.paste(image, (x, PADDING + LABEL_HEIGHT))
        draw.text((x, PADDING), label, fill=LABEL_COLOR, font=font)
        x += width + PADDING

    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.sheet)
    print(f"Saved paired-edit sheet: {args.sheet}")


if __name__ == "__main__":
    main()
