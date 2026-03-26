#!/usr/bin/env python3
"""
Build a curated paired-edit dataset from raw before/after image folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


DEFAULT_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, preserve original nail design, natural skin texture, realistic hand photo"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated paired-edit dataset from raw pairs.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Directory containing pair_xxxx/{before,after}.jpg folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest describing train/val ids and optional prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination dataset folder in pix2pix-turbo format.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_image(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="PNG")


def export_split(
    *,
    raw_dir: Path,
    output_dir: Path,
    ids: list[str],
    prefix: str,
    prompts: dict[str, str],
    default_prompt: str,
) -> int:
    side_a = output_dir / f"{prefix}_A"
    side_b = output_dir / f"{prefix}_B"
    side_a.mkdir(parents=True, exist_ok=True)
    side_b.mkdir(parents=True, exist_ok=True)

    prompt_map: dict[str, str] = {}
    rows: list[dict[str, str]] = []

    for pair_id in ids:
        pair_dir = raw_dir / pair_id
        before_path = pair_dir / "before.jpg"
        after_path = pair_dir / "after.jpg"
        if not before_path.exists() or not after_path.exists():
            raise FileNotFoundError(f"Missing before/after image for {pair_id} in {pair_dir}")

        file_name = f"{pair_id}.png"
        export_image(before_path, side_a / file_name)
        export_image(after_path, side_b / file_name)

        prompt = prompts.get(pair_id, default_prompt)
        prompt_map[file_name] = prompt
        rows.append(
            {
                "id": pair_id,
                "input": f"{prefix}_A/{file_name}",
                "target": f"{prefix}_B/{file_name}",
                "prompt": prompt,
            }
        )

    (output_dir / f"{prefix}_prompts.json").write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{prefix}_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    prompts = manifest.get("prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError("Manifest field 'prompts' must be an object mapping pair ids to prompt strings.")

    train_ids = manifest.get("train_ids", [])
    val_ids = manifest.get("val_ids", [])
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Manifest fields 'train_ids' and 'val_ids' must be lists.")

    default_prompt = str(manifest.get("default_prompt") or DEFAULT_PROMPT)

    ensure_clean_dir(args.output_dir)
    summary = {
        "train": export_split(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            ids=[str(pair_id) for pair_id in train_ids],
            prefix="train",
            prompts={str(key): str(value) for key, value in prompts.items()},
            default_prompt=default_prompt,
        ),
        "test": export_split(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            ids=[str(pair_id) for pair_id in val_ids],
            prefix="test",
            prompts={str(key): str(value) for key, value in prompts.items()},
            default_prompt=default_prompt,
        ),
        "manifest": str(args.manifest),
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
