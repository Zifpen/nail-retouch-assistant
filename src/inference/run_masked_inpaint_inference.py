#!/usr/bin/env python3
"""
Run masked local-retouch inference with a Stable Diffusion inpainting pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.masked_inpaint_utils import (
    build_sheet,
    dilate_mask_image,
    load_mask_image,
    load_rgb_image,
    mask_ratio,
    overlay_mask,
    resolve_latest_lora_path,
    run_inpaint,
)
from paired_edit.shared_config import PAIRED_EDIT_PROMPT


DEFAULT_OUTPUT_DIR = Path("outputs/masked_inpaint_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run masked inpainting inference for local nail retouch.")
    parser.add_argument("--input", type=Path, required=True, help="Input image to retouch.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary edit mask for the input image.")
    parser.add_argument("--target", type=Path, default=None, help="Optional target image for comparison sheets.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Base inpainting checkpoint.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="LoRA checkpoint file, or a directory containing .safetensors checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--output-name", default=None, help="Optional final composite filename.")
    parser.add_argument("--prompt", default=PAIRED_EDIT_PROMPT, help="Prompt used for retouch inference.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--strength", type=float, default=0.28, help="Inpainting denoise strength.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--steps", type=int, default=25, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--mask-dilate", type=int, default=5, help="Optional mask dilation before inference.")
    parser.add_argument("--crop-padding", type=int, default=48, help="Extra padding around the masked ROI crop.")
    parser.add_argument("--disable-roi-crop", action="store_true", help="Run on the full image instead of a mask ROI.")
    parser.add_argument(
        "--preserve-unmasked-exact",
        action="store_true",
        help="Composite original pixels back outside the mask after generation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Runtime device.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for the diffusion pipeline.",
    )
    return parser.parse_args()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask image not found: {args.mask}")
    if args.mask_dilate < 1 or args.mask_dilate % 2 == 0:
        raise ValueError("--mask-dilate must be a positive odd integer.")

    from diffusers import StableDiffusionInpaintPipeline

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.torch_dtype)
    lora_path = resolve_latest_lora_path(args.lora_path)

    input_image = load_rgb_image(args.input)
    mask_image = load_mask_image(args.mask, size=input_image.size)
    if args.mask_dilate > 1:
        mask_image = dilate_mask_image(mask_image, args.mask_dilate)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    if lora_path is not None:
        if lora_path.is_file():
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        else:
            pipe.load_lora_weights(str(lora_path))
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(args.seed)
    result = run_inpaint(
        pipe=pipe,
        input_image=input_image,
        mask_image=mask_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        crop_padding=args.crop_padding,
        use_roi_crop=not args.disable_roi_crop,
        preserve_unmasked_exact=args.preserve_unmasked_exact,
        generator=generator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.input.stem}_masked_inpaint.png"
    composite_path = args.output_dir / output_name
    generated_path = composite_path.with_name(f"{composite_path.stem}_generated.png")
    overlay_path = composite_path.with_name(f"{composite_path.stem}_mask_overlay.png")
    sheet_path = composite_path.with_name(f"{composite_path.stem}_sheet.png")
    metadata_path = composite_path.with_suffix(".json")

    result.composite_full.save(composite_path)
    result.generated_full.save(generated_path)
    overlay_mask(input_image, mask_image).save(overlay_path)

    columns = [
        ("input", input_image),
        ("mask", overlay_mask(input_image, mask_image)),
        ("generated", result.generated_full),
        ("composite", result.composite_full),
    ]
    if args.target is not None:
        if not args.target.exists():
            raise FileNotFoundError(f"Target image not found: {args.target}")
        columns.append(("target", load_rgb_image(args.target)))
    build_sheet(columns, sheet_path)

    metadata = {
        "input": str(args.input),
        "mask": str(args.mask),
        "target": str(args.target) if args.target is not None else None,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "lora_path": str(lora_path) if lora_path is not None else None,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "seed": args.seed,
        "mask_dilate": args.mask_dilate,
        "crop_padding": args.crop_padding,
        "disable_roi_crop": args.disable_roi_crop,
        "preserve_unmasked_exact": args.preserve_unmasked_exact,
        "crop_box": list(result.crop_box),
        "mask_ratio": mask_ratio(mask_image),
        "device": device,
        "dtype": str(dtype),
        "generated_path": str(generated_path),
        "composite_path": str(composite_path),
        "sheet_path": str(sheet_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved composite: {composite_path}")
    print(f"Saved raw generated output: {generated_path}")
    print(f"Saved mask overlay: {overlay_path}")
    print(f"Saved sheet: {sheet_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
