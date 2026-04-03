#!/usr/bin/env python3
"""
Build a masked inpainting dataset for local retouch supervision.

Expected raw layout:
  raw/
    pair_0001/
      before.jpg
      after.jpg
      mask.png            # optional when --mask-mode explicit

Output layout:
  dataset/masked_inpaint_core_v1/
    train/
      images/
      masks/
      targets/
      metadata.jsonl
    val/
      images/
      masks/
      targets/
      metadata.jsonl
    build_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paired_edit.shared_config import PAIRED_EDIT_PROMPT


DEFAULT_MASK_NAMES = ("mask.png", "mask.jpg", "mask.jpeg", "edit_mask.png", "edit_mask.jpg")
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class PairBuildStats:
    pair_id: str
    split: str
    prompt: str
    task: str
    mask_mode: str
    mask_ratio: float
    unmasked_ratio: float
    alignment_pixels: int
    raw_luma_delta: float
    global_aligned_luma_delta: float
    final_luma_delta: float
    source_input: str
    source_target: str
    source_mask: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a masked inpainting dataset from raw before/after pairs.")
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
        help="JSON manifest describing train/val ids and optional prompts/tasks/masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination dataset folder in masked inpainting format.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["explicit", "diff"],
        default="explicit",
        help="Use explicit masks when available, or bootstrap masks from before/after differences.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-pair masks named <pair_id>.png.",
    )
    parser.add_argument(
        "--default-task",
        default="cuticle_cleanup",
        help="Fallback task label used when the manifest does not define per-pair tasks.",
    )
    parser.add_argument(
        "--color-align-mode",
        choices=["none", "pairwise_stats"],
        default="pairwise_stats",
        help="Apply per-pair color alignment before writing target_local.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.10,
        help="Threshold in [0,1] used for bootstrap diff masks.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=5,
        help="Odd dilation kernel used for bootstrap diff masks.",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.005,
        help="Minimum allowed fraction of edited pixels.",
    )
    parser.add_argument(
        "--max-mask-ratio",
        type=float,
        default=0.60,
        help="Maximum allowed fraction of edited pixels.",
    )
    parser.add_argument(
        "--min-alignment-pixels",
        type=int,
        default=4096,
        help="Minimum number of unmasked pixels required for unmasked-region color alignment.",
    )
    parser.add_argument(
        "--save-aligned-targets",
        action="store_true",
        help="Also export the aligned full target image for inspection.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(pair_dir: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = pair_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem} image in {pair_dir}")


def resolve_mask_path(
    *,
    pair_id: str,
    pair_dir: Path,
    mask_dir: Path | None,
    manifest_masks: dict[str, str],
    manifest_dir: Path,
) -> Path:
    manifest_value = manifest_masks.get(pair_id)
    if manifest_value:
        candidate = Path(manifest_value)
        if not candidate.is_absolute():
            candidate = manifest_dir / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Mask path from manifest does not exist for {pair_id}: {candidate}")

    if mask_dir is not None:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = mask_dir / f"{pair_id}{suffix}"
            if candidate.exists():
                return candidate

    for name in DEFAULT_MASK_NAMES:
        candidate = pair_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find an explicit mask for {pair_id}. "
        "Provide raw/<pair_id>/mask.png, --mask-dir/<pair_id>.png, or manifest['masks'][pair_id]."
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8))
    image.save(path, format="PNG")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8))
    image.save(path, format="PNG")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8))
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.float32) / 255.0) > 0.5


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.float32) / 255.0) > 0.5


def crop_center_array(array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    height, width = array.shape[:2]
    if target_height > height or target_width > width:
        raise ValueError(
            f"Cannot crop array of size {(height, width)} to larger target {(target_height, target_width)}"
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    if array.ndim == 2:
        return array[top : top + target_height, left : left + target_width]
    return array[top : top + target_height, left : left + target_width, :]


def crop_pair_to_shared_area(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_height = min(before.shape[0], after.shape[0])
    target_width = min(before.shape[1], after.shape[1])
    return (
        crop_center_array(before, target_height, target_width),
        crop_center_array(after, target_height, target_width),
    )


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        raise ValueError("--mask-dilate must be a positive odd integer.")
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    dilated = image.filter(ImageFilter.MaxFilter(size=kernel_size))
    return (np.asarray(dilated, dtype=np.float32) / 255.0) > 0.5


def compute_diff_mask(before: np.ndarray, after: np.ndarray, *, threshold: float, dilate: int) -> np.ndarray:
    diff = np.abs(after - before).max(axis=2)
    mask = diff > threshold
    return dilate_mask(mask, dilate)


def channel_stats(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
    if mask is not None and mask.any():
        pixels = image[mask]
    else:
        pixels = image.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, std, int(pixels.shape[0])


def align_image_pairwise(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    stats_mask: np.ndarray | None,
    min_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    mean_ref, std_ref, pixel_count = channel_stats(reference, stats_mask)
    mean_src, std_src, _ = channel_stats(source, stats_mask)
    if pixel_count < min_pixels:
        mean_ref, std_ref, pixel_count = channel_stats(reference, None)
        mean_src, std_src, _ = channel_stats(source, None)
        used_unmasked_region = False
    else:
        used_unmasked_region = stats_mask is not None

    scale = std_ref / np.clip(std_src, 1e-4, None)
    aligned = np.clip((source - mean_src) * scale + mean_ref, 0.0, 1.0)
    stats = {
        "reference_mean": mean_ref.tolist(),
        "reference_std": std_ref.tolist(),
        "source_mean": mean_src.tolist(),
        "source_std": std_src.tolist(),
        "scale": scale.tolist(),
        "pixel_count": pixel_count,
        "used_unmasked_region": used_unmasked_region,
    }
    return aligned, stats


def mean_luma_delta(before: np.ndarray, after: np.ndarray) -> float:
    before_luma = before @ LUMA_WEIGHTS
    after_luma = after @ LUMA_WEIGHTS
    return float(after_luma.mean() - before_luma.mean())


def build_target_local(before: np.ndarray, aligned_after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_3 = mask[..., None].astype(np.float32)
    return before * (1.0 - mask_3) + aligned_after * mask_3


def validate_mask(mask: np.ndarray, *, pair_id: str, min_ratio: float, max_ratio: float) -> tuple[float, float]:
    mask_ratio = float(mask.mean())
    unmasked_ratio = 1.0 - mask_ratio
    if mask_ratio <= min_ratio:
        raise ValueError(
            f"Mask for {pair_id} is too small: ratio={mask_ratio:.4f}, min_allowed={min_ratio:.4f}"
        )
    if mask_ratio >= max_ratio:
        raise ValueError(
            f"Mask for {pair_id} is too large: ratio={mask_ratio:.4f}, max_allowed={max_ratio:.4f}"
        )
    return mask_ratio, unmasked_ratio


def write_split_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_split(
    *,
    raw_dir: Path,
    manifest_dir: Path,
    output_dir: Path,
    split_name: str,
    pair_ids: list[str],
    prompts: dict[str, str],
    tasks: dict[str, str],
    masks: dict[str, str],
    default_prompt: str,
    default_task: str,
    args: argparse.Namespace,
) -> tuple[int, list[PairBuildStats]]:
    split_dir = output_dir / split_name
    image_dir = split_dir / "images"
    mask_out_dir = split_dir / "masks"
    target_dir = split_dir / "targets"
    aligned_dir = split_dir / "aligned_targets"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    if args.save_aligned_targets:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    stats_rows: list[PairBuildStats] = []

    for pair_id in pair_ids:
        pair_dir = raw_dir / pair_id
        before_path = resolve_image_path(pair_dir, "before")
        after_path = resolve_image_path(pair_dir, "after")

        before_raw = load_rgb(before_path)
        after_raw = load_rgb(after_path)
        before, after = crop_pair_to_shared_area(before_raw, after_raw)
        raw_luma_delta = mean_luma_delta(before, after)

        coarse_after = after
        coarse_alignment = {
            "reference_mean": [],
            "reference_std": [],
            "source_mean": [],
            "source_std": [],
            "scale": [],
            "pixel_count": before.shape[0] * before.shape[1],
            "used_unmasked_region": False,
        }
        if args.color_align_mode == "pairwise_stats":
            coarse_after, coarse_alignment = align_image_pairwise(
                after,
                before,
                stats_mask=None,
                min_pixels=args.min_alignment_pixels,
            )

        if args.mask_mode == "explicit":
            mask_path = resolve_mask_path(
                pair_id=pair_id,
                pair_dir=pair_dir,
                mask_dir=args.mask_dir,
                manifest_masks=masks,
                manifest_dir=manifest_dir,
            )
            mask = load_mask(mask_path, size=(before_raw.shape[1], before_raw.shape[0]))
            if mask.shape != before.shape[:2]:
                mask = crop_center_array(mask, before.shape[0], before.shape[1])
        else:
            mask_path = None
            mask = compute_diff_mask(
                before,
                coarse_after,
                threshold=args.diff_threshold,
                dilate=args.mask_dilate,
            )

        alignment_mask = ~mask
        aligned_after = coarse_after
        alignment_stats = coarse_alignment
        if args.color_align_mode == "pairwise_stats":
            aligned_after, alignment_stats = align_image_pairwise(
                after,
                before,
                stats_mask=alignment_mask,
                min_pixels=args.min_alignment_pixels,
            )
        if args.mask_mode == "diff":
            mask = compute_diff_mask(
                before,
                aligned_after,
                threshold=args.diff_threshold,
                dilate=args.mask_dilate,
            )

        mask_ratio, unmasked_ratio = validate_mask(
            mask,
            pair_id=pair_id,
            min_ratio=args.min_mask_ratio,
            max_ratio=args.max_mask_ratio,
        )

        target_local = build_target_local(before, aligned_after, mask)
        file_name = f"{pair_id}.png"
        input_out = image_dir / file_name
        mask_out = mask_out_dir / file_name
        target_out = target_dir / file_name

        save_rgb(input_out, before)
        save_mask(mask_out, mask)
        save_rgb(target_out, target_local)
        if args.save_aligned_targets:
            save_rgb(aligned_dir / file_name, aligned_after)

        prompt = str(prompts.get(pair_id) or default_prompt)
        task = str(tasks.get(pair_id) or default_task)
        stats_row = PairBuildStats(
            pair_id=pair_id,
            split=split_name,
            prompt=prompt,
            task=task,
            mask_mode=args.mask_mode,
            mask_ratio=mask_ratio,
            unmasked_ratio=unmasked_ratio,
            alignment_pixels=int(alignment_stats["pixel_count"]),
            raw_luma_delta=raw_luma_delta,
            global_aligned_luma_delta=mean_luma_delta(before, coarse_after),
            final_luma_delta=mean_luma_delta(before, aligned_after),
            source_input=str(before_path),
            source_target=str(after_path),
            source_mask=str(mask_path) if mask_path is not None else None,
        )
        stats_rows.append(stats_row)
        metadata_rows.append(
            {
                "id": pair_id,
                "input": str(input_out.relative_to(output_dir)),
                "mask": str(mask_out.relative_to(output_dir)),
                "target": str(target_out.relative_to(output_dir)),
                "prompt": prompt,
                "task": task,
                "mask_mode": args.mask_mode,
                "mask_ratio": mask_ratio,
                "unmasked_ratio": unmasked_ratio,
                "source_input": str(before_path),
                "source_target": str(after_path),
                "source_mask": str(mask_path) if mask_path is not None else None,
                "alignment": alignment_stats,
                "raw_luma_delta": stats_row.raw_luma_delta,
                "global_aligned_luma_delta": stats_row.global_aligned_luma_delta,
                "final_luma_delta": stats_row.final_luma_delta,
            }
        )

    write_split_metadata(split_dir / "metadata.jsonl", metadata_rows)
    return len(metadata_rows), stats_rows


def summarize_stats(rows: list[PairBuildStats]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "mean_mask_ratio": 0.0,
            "mean_unmasked_ratio": 0.0,
            "mean_coarse_luma_delta": 0.0,
            "mean_aligned_luma_delta": 0.0,
        }
    return {
        "count": float(len(rows)),
        "mean_mask_ratio": float(sum(row.mask_ratio for row in rows) / len(rows)),
        "mean_unmasked_ratio": float(sum(row.unmasked_ratio for row in rows) / len(rows)),
        "mean_raw_luma_delta": float(sum(row.raw_luma_delta for row in rows) / len(rows)),
        "mean_global_aligned_luma_delta": float(
            sum(row.global_aligned_luma_delta for row in rows) / len(rows)
        ),
        "mean_final_luma_delta": float(sum(row.final_luma_delta for row in rows) / len(rows)),
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    prompts = manifest.get("prompts", {})
    tasks = manifest.get("tasks", {})
    masks = manifest.get("masks", {})
    train_ids = manifest.get("train_ids", [])
    val_ids = manifest.get("val_ids", [])
    if not isinstance(prompts, dict) or not isinstance(tasks, dict) or not isinstance(masks, dict):
        raise ValueError("Manifest fields 'prompts', 'tasks', and 'masks' must be JSON objects when present.")
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Manifest fields 'train_ids' and 'val_ids' must be lists.")
    if args.diff_threshold <= 0 or args.diff_threshold >= 1:
        raise ValueError("--diff-threshold must be in the open interval (0, 1).")

    default_prompt = str(manifest.get("default_prompt") or PAIRED_EDIT_PROMPT)
    default_task = str(manifest.get("default_task") or args.default_task)

    ensure_clean_dir(args.output_dir)
    train_count, train_stats = export_split(
        raw_dir=args.raw_dir,
        manifest_dir=args.manifest.parent,
        output_dir=args.output_dir,
        split_name="train",
        pair_ids=[str(pair_id) for pair_id in train_ids],
        prompts={str(key): str(value) for key, value in prompts.items()},
        tasks={str(key): str(value) for key, value in tasks.items()},
        masks={str(key): str(value) for key, value in masks.items()},
        default_prompt=default_prompt,
        default_task=default_task,
        args=args,
    )
    val_count, val_stats = export_split(
        raw_dir=args.raw_dir,
        manifest_dir=args.manifest.parent,
        output_dir=args.output_dir,
        split_name="val",
        pair_ids=[str(pair_id) for pair_id in val_ids],
        prompts={str(key): str(value) for key, value in prompts.items()},
        tasks={str(key): str(value) for key, value in tasks.items()},
        masks={str(key): str(value) for key, value in masks.items()},
        default_prompt=default_prompt,
        default_task=default_task,
        args=args,
    )

    summary = {
        "raw_dir": str(args.raw_dir),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "mask_mode": args.mask_mode,
        "color_align_mode": args.color_align_mode,
        "train_count": train_count,
        "val_count": val_count,
        "train_summary": summarize_stats(train_stats),
        "val_summary": summarize_stats(val_stats),
        "train_pairs": [asdict(row) for row in train_stats],
        "val_pairs": [asdict(row) for row in val_stats],
    }
    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
