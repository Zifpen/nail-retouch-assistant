#!/usr/bin/env python3
"""
Utilities shared by masked inpainting inference and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


LABEL_HEIGHT = 42
PADDING = 12
BACKGROUND = (18, 18, 20)
LABEL_COLOR = (245, 245, 245)
MASK_OVERLAY_COLOR = np.array([255.0, 64.0, 64.0], dtype=np.float32)


@dataclass
class InpaintResult:
    crop_box: tuple[int, int, int, int]
    input_crop: Image.Image
    mask_crop: Image.Image
    generated_crop: Image.Image
    composite_crop: Image.Image
    generated_full: Image.Image
    composite_full: Image.Image


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_image(path: Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return binarize_mask(image)


def binarize_mask(mask_image: Image.Image) -> Image.Image:
    array = np.where(np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def dilate_mask_image(mask_image: Image.Image, kernel_size: int) -> Image.Image:
    if kernel_size <= 1:
        return binarize_mask(mask_image)
    if kernel_size % 2 == 0:
        raise ValueError("Mask dilation kernel must be a positive odd integer.")
    return binarize_mask(mask_image).filter(ImageFilter.MaxFilter(size=kernel_size))


def mask_to_bool(mask_image: Image.Image) -> np.ndarray:
    return np.asarray(binarize_mask(mask_image), dtype=np.uint8) >= 128


def mask_ratio(mask_image: Image.Image) -> float:
    return float(mask_to_bool(mask_image).mean())


def compute_mask_bbox(mask_image: Image.Image) -> tuple[int, int, int, int] | None:
    bbox = binarize_mask(mask_image).getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    if left == right or top == bottom:
        return None
    return left, top, right, bottom


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def composite_exact(base_image: Image.Image, edited_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    if base_image.size != edited_image.size or base_image.size != mask_image.size:
        raise ValueError("Base image, edited image, and mask must have identical sizes for compositing.")
    base = pil_to_np01(base_image)
    edited = pil_to_np01(edited_image)
    mask = mask_to_bool(mask_image)[..., None].astype(np.float32)
    composite = base * (1.0 - mask) + edited * mask
    return np01_to_pil(composite)


def overlay_mask(
    base_image: Image.Image,
    mask_image: Image.Image,
    *,
    alpha: float = 0.35,
) -> Image.Image:
    if base_image.size != mask_image.size:
        raise ValueError("Base image and mask must have identical sizes for overlay.")
    base = pil_to_np01(base_image) * 255.0
    mask = mask_to_bool(mask_image)[..., None].astype(np.float32)
    overlay = base * (1.0 - (alpha * mask)) + MASK_OVERLAY_COLOR.reshape(1, 1, 3) * (alpha * mask)
    return Image.fromarray(np.clip(np.round(overlay), 0, 255).astype(np.uint8), mode="RGB")


def pil_to_np01(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def np01_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8), mode="RGB")


def srgb_to_linear_np(rgb: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055) ** 2.4
    return np.where(rgb <= threshold, low, high)


def rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    linear = srgb_to_linear_np(rgb)
    r = linear[..., 0]
    g = linear[..., 1]
    b = linear[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    epsilon = 216 / 24389
    kappa = 24389 / 27

    def f(value: np.ndarray) -> np.ndarray:
        return np.where(value > epsilon, np.cbrt(value), (kappa * value + 16.0) / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)
    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([l, a, b], axis=-1)


def masked_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError("Expected a 2D boolean mask.")
    if not mask.any():
        return 0.0
    return float(np.abs(prediction[mask] - target[mask]).mean())


def masked_delta_e(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    pred_lab = rgb_to_lab_np(prediction)
    target_lab = rgb_to_lab_np(target)
    delta = pred_lab - target_lab
    distances = np.sqrt((delta ** 2).sum(axis=-1))
    return float(distances[mask].mean())


def boundary_ring(mask_image: Image.Image, width: int) -> Image.Image:
    if width <= 0:
        return Image.fromarray(np.zeros((mask_image.height, mask_image.width), dtype=np.uint8), mode="L")
    expanded = dilate_mask_image(mask_image, (width * 2) + 1)
    contracted = binarize_mask(mask_image).filter(ImageFilter.MinFilter(size=(width * 2) + 1))
    ring = np.asarray(expanded, dtype=np.uint8) - np.asarray(contracted, dtype=np.uint8)
    return Image.fromarray(np.where(ring > 0, 255, 0).astype(np.uint8), mode="L")


def fit_columns(columns: list[tuple[str, Image.Image]]) -> tuple[list[tuple[str, Image.Image]], int, int]:
    width = max(image.width for _, image in columns)
    height = max(image.height for _, image in columns)
    resized: list[tuple[str, Image.Image]] = []
    for label, image in columns:
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append((label, image))
    return resized, width, height


def build_sheet(columns: list[tuple[str, Image.Image]], output_path: Path) -> None:
    columns, width, height = fit_columns(columns)
    canvas = Image.new(
        "RGB",
        ((width * len(columns)) + (PADDING * (len(columns) + 1)), height + LABEL_HEIGHT + (PADDING * 2)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = PADDING
    for label, image in columns:
        canvas.paste(image, (x, PADDING + LABEL_HEIGHT))
        draw.text((x, PADDING), label, fill=LABEL_COLOR, font=font)
        x += width + PADDING

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def resolve_latest_lora_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors LoRA weights found in {path}")
    return candidates[-1]


def run_inpaint(
    *,
    pipe,
    input_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    negative_prompt: str | None,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    crop_padding: int,
    use_roi_crop: bool,
    preserve_unmasked_exact: bool,
    generator,
) -> InpaintResult:
    mask_image = binarize_mask(mask_image)
    crop_box = (0, 0, input_image.width, input_image.height)
    if use_roi_crop:
        bbox = compute_mask_bbox(mask_image)
        if bbox is not None:
            crop_box = expand_bbox(bbox, input_image.size, crop_padding)

    input_crop = input_image.crop(crop_box)
    mask_crop = mask_image.crop(crop_box)
    generated_crop = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_crop,
        mask_image=mask_crop,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0].convert("RGB")
    composite_crop = (
        composite_exact(input_crop, generated_crop, mask_crop) if preserve_unmasked_exact else generated_crop
    )

    generated_full = input_image.copy()
    generated_full.paste(generated_crop, crop_box[:2])
    composite_full = input_image.copy()
    composite_full.paste(composite_crop, crop_box[:2])
    return InpaintResult(
        crop_box=crop_box,
        input_crop=input_crop,
        mask_crop=mask_crop,
        generated_crop=generated_crop,
        composite_crop=composite_crop,
        generated_full=generated_full,
        composite_full=composite_full,
    )
