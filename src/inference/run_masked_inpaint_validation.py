#!/usr/bin/env python3
"""
Run validation for a masked inpainting LoRA on a masked dataset split.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.masked_inpaint_utils import (
    boundary_ring,
    build_sheet,
    dilate_mask_image,
    load_mask_image,
    load_rgb_image,
    mask_ratio,
    mask_to_bool,
    masked_delta_e,
    masked_l1,
    overlay_mask,
    pil_to_np01,
    resolve_latest_lora_path,
    run_inpaint,
)
from paired_edit.shared_config import DEFAULT_BASELINE_PAIR_IDS, PAIRED_EDIT_PROMPT


DEFAULT_OUTPUT_ROOT = Path("outputs/masked_inpaint_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a masked inpainting LoRA on a masked dataset split.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Masked dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to validate.")
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Validation output root.")
    parser.add_argument("--pair-id", action="append", default=[], help="Specific pair id. Can be passed multiple times.")
    parser.add_argument("--max-samples", type=int, default=4, help="Fallback sample count when no pair ids are set.")
    parser.add_argument("--prompt", default=PAIRED_EDIT_PROMPT, help="Fallback prompt used when metadata omits one.")
    parser.add_argument("--allow-dataset-prompt-variants", action="store_true", help="Use per-sample prompts from metadata.")
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
    parser.add_argument("--border-width", type=int, default=5, help="Boundary ring width used for seam metrics.")
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


def load_rows(dataset_dir: Path, split: str) -> dict[str, dict]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    rows: dict[str, dict] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["id"])] = row
    return rows


def select_pair_ids(rows: dict[str, dict], requested_pair_ids: list[str], max_samples: int) -> list[str]:
    if requested_pair_ids:
        missing = [pair_id for pair_id in requested_pair_ids if pair_id not in rows]
        if missing:
            raise KeyError(f"Missing pair ids in dataset metadata: {missing}")
        return requested_pair_ids

    defaults = [pair_id for pair_id in DEFAULT_BASELINE_PAIR_IDS if pair_id in rows]
    if defaults:
        return defaults
    return list(sorted(rows.keys()))[:max_samples]


def summarize_metrics(metrics_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_rows:
        return {"count": 0.0}
    summary: dict[str, float] = {"count": float(len(metrics_rows))}
    for key in (
        "mask_ratio",
        "masked_l1_to_target",
        "masked_delta_e_to_target",
        "unmasked_l1_to_input",
        "unmasked_delta_e_to_input",
        "border_l1_to_target",
    ):
        summary[f"mean_{key}"] = float(statistics.fmean(row[key] for row in metrics_rows))
    return summary


def main() -> None:
    args = parse_args()
    if args.mask_dilate < 1 or args.mask_dilate % 2 == 0:
        raise ValueError("--mask-dilate must be a positive odd integer.")
    if args.border_width < 0:
        raise ValueError("--border-width must be non-negative.")

    from diffusers import StableDiffusionInpaintPipeline

    rows = load_rows(args.dataset_dir, args.split)
    pair_ids = select_pair_ids(rows, args.pair_id, args.max_samples)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.torch_dtype)
    lora_path = resolve_latest_lora_path(args.lora_path)

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

    run_name = lora_path.stem if lora_path is not None else "base_inpaint"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float]] = []

    for pair_id in pair_ids:
        row = rows[pair_id]
        prompt = row.get("prompt") if args.allow_dataset_prompt_variants else args.prompt
        prompt = str(prompt or args.prompt)
        input_path = args.dataset_dir / str(row["input"])
        mask_path = args.dataset_dir / str(row["mask"])
        target_path = args.dataset_dir / str(row["target"])
        input_image = load_rgb_image(input_path)
        target_image = load_rgb_image(target_path)
        mask_image = load_mask_image(mask_path, size=input_image.size)
        if args.mask_dilate > 1:
            mask_image = dilate_mask_image(mask_image, args.mask_dilate)

        generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(args.seed)
        result = run_inpaint(
            pipe=pipe,
            input_image=input_image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=None,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            crop_padding=args.crop_padding,
            use_roi_crop=not args.disable_roi_crop,
            preserve_unmasked_exact=args.preserve_unmasked_exact,
            generator=generator,
        )

        pred_np = pil_to_np01(result.composite_full)
        target_np = pil_to_np01(target_image)
        input_np = pil_to_np01(input_image)
        mask_bool = mask_to_bool(mask_image)
        unmasked_bool = ~mask_bool
        border_bool = mask_to_bool(boundary_ring(mask_image, args.border_width))

        metrics = {
            "pair_id": pair_id,
            "mask_ratio": mask_ratio(mask_image),
            "masked_l1_to_target": masked_l1(pred_np, target_np, mask_bool),
            "masked_delta_e_to_target": masked_delta_e(pred_np, target_np, mask_bool),
            "unmasked_l1_to_input": masked_l1(pred_np, input_np, unmasked_bool),
            "unmasked_delta_e_to_input": masked_delta_e(pred_np, input_np, unmasked_bool),
            "border_l1_to_target": masked_l1(pred_np, target_np, border_bool),
        }
        metrics_rows.append(metrics)

        pair_output_path = output_dir / f"{pair_id}_output.png"
        pair_sheet_path = output_dir / f"{pair_id}_sheet.png"
        pair_metadata_path = output_dir / f"{pair_id}_metrics.json"
        result.composite_full.save(pair_output_path)
        build_sheet(
            [
                ("input", input_image),
                ("mask", overlay_mask(input_image, mask_image)),
                ("output", result.composite_full),
                ("target", target_image),
            ],
            pair_sheet_path,
        )
        pair_metadata = {
            "pair_id": pair_id,
            "input": str(input_path),
            "mask": str(mask_path),
            "target": str(target_path),
            "output": str(pair_output_path),
            "sheet": str(pair_sheet_path),
            "prompt": prompt,
            "crop_box": list(result.crop_box),
            "strength": args.strength,
            "guidance_scale": args.guidance_scale,
            "steps": args.steps,
            "seed": args.seed,
            "mask_dilate": args.mask_dilate,
            "crop_padding": args.crop_padding,
            "disable_roi_crop": args.disable_roi_crop,
            "preserve_unmasked_exact": args.preserve_unmasked_exact,
            "border_width": args.border_width,
            "metrics": metrics,
        }
        pair_metadata_path.write_text(json.dumps(pair_metadata, indent=2), encoding="utf-8")
        print(f"Saved validation output: {pair_output_path}")
        print(f"Saved validation sheet: {pair_sheet_path}")
        print(f"Saved metrics: {pair_metadata_path}")

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in metrics_rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "lora_path": str(lora_path) if lora_path is not None else None,
        "device": device,
        "dtype": str(dtype),
        "pair_ids": pair_ids,
        "summary": summarize_metrics(metrics_rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved aggregate metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
