#!/usr/bin/env python3
"""
Run local paired-edit validation for pix2pix-turbo checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from pix2pix_runtime import (
    DEFAULT_UPSTREAM_DIR,
    apply_runtime_checkpoint_patch,
    clear_device_cache,
    latest_checkpoint,
    load_image_tensor,
    pick_device,
    prepare_runtime_upstream,
    resolve_default_path,
    tensor_to_image,
)
from shared_config import DEFAULT_BASELINE_PAIR_IDS, DEFAULT_CORE_DATASET_DIR, PAIRED_EDIT_PROMPT


DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")
DEFAULT_DATASET_DIR = DEFAULT_CORE_DATASET_DIR
DEFAULT_OUTPUT_ROOT = Path("outputs/paired_edit_validation")
DEFAULT_PAIR_IDS = DEFAULT_BASELINE_PAIR_IDS
BACKGROUND = (18, 18, 20)
LABEL_COLOR = (245, 245, 245)
LABEL_HEIGHT = 42
PADDING = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired-edit pix2pix-turbo checkpoints locally.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Specific model_*.pkl checkpoint. Defaults to the latest file in outputs/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing model_*.pkl checkpoints.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Paired-edit dataset directory containing test_A/test_B and metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated validation outputs.",
    )
    parser.add_argument(
        "--mirror-output-root",
        type=Path,
        default=None,
        help="Optional second directory that receives a copy of the validation outputs.",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="Local clone of the upstream img2img-turbo repository.",
    )
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Specific pair id. Can be passed multiple times. Defaults to the 2-train/2-val baseline set.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Runtime device.",
    )
    parser.add_argument(
        "--prompt",
        default=PAIRED_EDIT_PROMPT,
        help="Prompt used for validation inference.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Optional max input side used before inference.",
    )
    return parser.parse_args()


def load_metadata(dataset_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for metadata_name in ("train_metadata.jsonl", "test_metadata.jsonl"):
        metadata_path = dataset_dir / metadata_name
        if not metadata_path.exists():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def build_sheet(input_path: Path, output_path: Path, target_path: Path, sheet_path: Path) -> None:
    images = [
        ("input", Image.open(input_path).convert("RGB")),
        ("output", Image.open(output_path).convert("RGB")),
        ("target", Image.open(target_path).convert("RGB")),
    ]
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    fitted = []
    for label, image in images:
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        fitted.append((label, image))

    sheet = Image.new(
        "RGB",
        ((width * 3) + (PADDING * 4), height + LABEL_HEIGHT + (PADDING * 2)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = PADDING
    for label, image in fitted:
        sheet.paste(image, (x, PADDING + LABEL_HEIGHT))
        draw.text((x, PADDING), label, fill=LABEL_COLOR, font=font)
        x += width + PADDING

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def validate_prompt_consistency(metadata: dict[str, dict], pair_ids: list[str], prompt: str) -> None:
    for pair_id in pair_ids:
        row = metadata[pair_id]
        row_prompt = row.get("prompt")
        if row_prompt and row_prompt != prompt:
            raise ValueError(
                f"Prompt mismatch for {pair_id}: dataset={row_prompt!r} validation={prompt!r}"
            )


def mirror_output_dir(source_dir: Path, mirror_root: Path | None) -> Path | None:
    if mirror_root is None:
        return None
    mirror_dir = mirror_root / source_dir.name
    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)
    shutil.copytree(source_dir, mirror_dir)
    return mirror_dir


def main() -> None:
    args = parse_args()
    args.upstream_dir = resolve_default_path(args.upstream_dir, Path("/tmp/img2img-turbo-local"))
    args.dataset_dir = resolve_default_path(
        args.dataset_dir,
        Path("/content/nail-retouch-assistant/dataset/paired_edit_core_v1"),
    )
    args.checkpoint_dir = resolve_default_path(
        args.checkpoint_dir,
        Path("/content/drive/MyDrive/nail-retouch-paired-core-v1-outputs/checkpoints"),
    )
    if args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = resolve_default_path(
            args.output_root,
            Path("/content/workdir/paired_edit/validation"),
        )
    if args.mirror_output_root is None and str(args.output_root).startswith("/content/drive/MyDrive/"):
        args.mirror_output_root = Path("/content/workdir/paired_edit/validation")
    device = pick_device(args.device)
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    pair_ids = args.pair_id or DEFAULT_PAIR_IDS

    runtime_root = prepare_runtime_upstream(args.upstream_dir, device)
    apply_runtime_checkpoint_patch(runtime_root)
    sys.path.insert(0, str(runtime_root / "src"))
    import os

    os.environ["PIX2PIX_TURBO_DEVICE"] = device
    from pix2pix_turbo import Pix2Pix_Turbo

    metadata = load_metadata(args.dataset_dir)
    missing_pair_ids = [pair_id for pair_id in pair_ids if pair_id not in metadata]
    if missing_pair_ids:
        raise KeyError(f"Missing pair ids in dataset metadata: {missing_pair_ids}")
    validate_prompt_consistency(metadata, pair_ids, args.prompt)
    output_dir = args.output_root / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Prompt: {args.prompt}")
    clear_device_cache(device)
    model = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
    model.set_eval()

    if device == "cuda":
        model.half()
    elif device == "cpu":
        model = model.to(torch.float32)

    for pair_id in pair_ids:
        row = metadata[pair_id]
        input_path = args.dataset_dir / row["input"]
        target_path = args.dataset_dir / row["target"]
        output_path = output_dir / f"{pair_id}_output.png"
        sheet_path = output_dir / f"{pair_id}_sheet.png"
        metadata_path = output_dir / f"{pair_id}_metadata.json"

        clear_device_cache(device)
        x_src = load_image_tensor(input_path, device, max_side=args.max_side)
        if device == "cuda":
            x_src = x_src.half()
        with torch.no_grad():
            x_out = model(x_src, prompt=args.prompt, deterministic=True)

        image = tensor_to_image(x_out)
        image.save(output_path)
        build_sheet(input_path, output_path, target_path, sheet_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "pair_id": pair_id,
                    "checkpoint": str(checkpoint),
                    "input": str(input_path),
                    "target": str(target_path),
                    "output": str(output_path),
                    "sheet": str(sheet_path),
                    "prompt": args.prompt,
                    "device": device,
                    "max_side": args.max_side,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved output: {output_path}")
        print(f"Saved sheet: {sheet_path}")
        print(f"Saved metadata: {metadata_path}")
        clear_device_cache(device)

    summary = {
        "checkpoint": checkpoint.name,
        "device": device,
        "prompt": args.prompt,
        "pairs": pair_ids,
        "output_dir": str(output_dir),
        "max_side": args.max_side,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {output_dir / 'validation_summary.json'}")
    mirror_dir = mirror_output_dir(output_dir, args.mirror_output_root)
    if mirror_dir is not None:
        print(f"Mirrored outputs: {mirror_dir}")


if __name__ == "__main__":
    main()
