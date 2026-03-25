#!/usr/bin/env python3
"""
Run paired-edit pix2pix-turbo inference on a single image.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch

from pix2pix_runtime import (
    DEFAULT_UPSTREAM_DIR,
    apply_runtime_checkpoint_patch,
    latest_checkpoint,
    load_image_tensor,
    pick_device,
    prepare_runtime_upstream,
    resolve_default_path,
    tensor_to_image,
)


DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")
DEFAULT_OUTPUT_DIR = Path("outputs/paired_edit_inference")
DEFAULT_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, glossy nail surface, natural skin texture, realistic hand photo"
)
BACKGROUND = (18, 18, 20)
LABEL_COLOR = (245, 245, 245)
LABEL_HEIGHT = 42
PADDING = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired-edit pix2pix-turbo inference on a single image.")
    parser.add_argument("--input", type=Path, required=True, help="Input image to retouch.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Specific model_*.pkl checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing model_*.pkl checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where output images will be saved.",
    )
    parser.add_argument("--output-name", default=None, help="Optional output filename.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for retouch inference.")
    parser.add_argument("--target", type=Path, default=None, help="Optional target image for comparison sheets.")
    parser.add_argument("--sheet", type=Path, default=None, help="Optional explicit path for the comparison sheet.")
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="Local clone of the upstream img2img-turbo repository.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Runtime device.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Resize the input so its longest side is at most this many pixels before inference.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Apply more aggressive CUDA memory-saving behavior.",
    )
    return parser.parse_args()


def build_sheet(input_path: Path, output_path: Path, target_path: Path, sheet_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

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


def main() -> None:
    args = parse_args()
    args.upstream_dir = resolve_default_path(args.upstream_dir, Path("/tmp/img2img-turbo-local"))
    args.checkpoint_dir = resolve_default_path(
        args.checkpoint_dir,
        Path("/content/drive/MyDrive/nail-retouch-paired-outputs/checkpoints"),
    )
    if args.output_dir == DEFAULT_OUTPUT_DIR and Path("/content").exists():
        args.output_dir = Path("/content/workdir/paired_edit/inference")

    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    device = pick_device(args.device)
    if args.max_side is None and device == "cuda":
        args.max_side = 1024
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    runtime_root = prepare_runtime_upstream(args.upstream_dir, device)
    apply_runtime_checkpoint_patch(runtime_root)
    sys.path.insert(0, str(runtime_root / "src"))
    os.environ["PIX2PIX_TURBO_DEVICE"] = device
    from pix2pix_turbo import Pix2Pix_Turbo

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.input.stem}_{checkpoint.stem}.png"
    output_path = args.output_dir / output_name

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    if args.max_side is not None:
        print(f"Max input side: {args.max_side}")
    if args.low_vram:
        print("Low-VRAM mode: enabled")

    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    model = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
    model.set_eval()
    if device == "cuda":
        model.half()
    elif device == "cpu":
        model = model.to(torch.float32)

    if device == "cuda" and args.low_vram:
        if hasattr(model.vae, "enable_slicing"):
            model.vae.enable_slicing()
        if hasattr(model.vae, "enable_tiling"):
            model.vae.enable_tiling()

    x_src = load_image_tensor(args.input, device, max_side=args.max_side)
    if device == "cuda":
        x_src = x_src.half()

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_context:
        x_out = model(x_src, prompt=args.prompt, deterministic=True)

    image = tensor_to_image(x_out)
    image.save(output_path)
    if device == "cuda":
        torch.cuda.empty_cache()

    metadata = {
        "input": str(args.input),
        "checkpoint": str(checkpoint),
        "prompt": args.prompt,
        "device": device,
        "output": str(output_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved output: {output_path}")
    print(f"Saved metadata: {metadata_path}")

    if args.target is not None:
        if not args.target.exists():
            raise FileNotFoundError(f"Target image not found: {args.target}")
        sheet_path = args.sheet or output_path.with_name(f"{output_path.stem}_sheet.png")
        build_sheet(args.input, output_path, args.target, sheet_path)
        print(f"Saved sheet: {sheet_path}")


if __name__ == "__main__":
    main()
