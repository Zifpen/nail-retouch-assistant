#!/usr/bin/env python3
"""
Train a masked local-retouch LoRA on top of a Stable Diffusion inpainting checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paired_edit.shared_config import PAIRED_EDIT_PROMPT


@dataclass
class Example:
    pair_id: str
    input_path: Path
    mask_path: Path
    target_path: Path
    prompt: str
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked inpainting LoRA for local retouch.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Inpainting checkpoint used as the base model.",
    )
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Masked dataset root directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where checkpoints are written.")
    parser.add_argument(
        "--train_split",
        default="train",
        choices=["train", "val"],
        help="Dataset split used for training.",
    )
    parser.add_argument(
        "--val_split",
        default="val",
        choices=["train", "val"],
        help="Dataset split used for lightweight evaluation previews.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--image_prep",
        choices=["center_crop", "resize"],
        default="center_crop",
        help="How to fit the dataset samples to the square training resolution.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=40)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--preview_steps", type=int, default=250)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--prompt", default=PAIRED_EDIT_PROMPT, help="Fallback prompt when metadata omits one.")
    parser.add_argument("--allow_dataset_prompt_variants", action="store_true")
    parser.add_argument("--lambda_diffusion", type=float, default=1.0)
    parser.add_argument("--lambda_mask_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_identity", type=float, default=5.0)
    parser.add_argument("--lambda_color", type=float, default=0.5)
    parser.add_argument("--mask_dilate", type=int, default=3, help="Optional dilation applied to masks during loss.")
    parser.add_argument("--max_preview_images", type=int, default=4)
    parser.add_argument("--validation_samples", type=int, default=4)
    return parser.parse_args()


def read_examples(dataset_dir: Path, split: str, fallback_prompt: str) -> list[Example]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    examples: list[Example] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            Example(
                pair_id=str(row["id"]),
                input_path=dataset_dir / str(row["input"]),
                mask_path=dataset_dir / str(row["mask"]),
                target_path=dataset_dir / str(row["target"]),
                prompt=str(row.get("prompt") or fallback_prompt),
                task=str(row.get("task") or "cuticle_cleanup"),
            )
        )
    if not examples:
        raise ValueError(f"No samples found in {metadata_path}")
    return examples


def fit_image(image: Image.Image, resolution: int, mode: str, *, is_mask: bool) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    if mode == "resize":
        return image.resize((resolution, resolution), resample)
    if mode != "center_crop":
        raise ValueError(f"Unsupported image_prep mode: {mode}")

    width, height = image.size
    if width == height:
        cropped = image
    elif width > height:
        left = (width - height) // 2
        cropped = image.crop((left, 0, left + height, height))
    else:
        top = (height - width) // 2
        cropped = image.crop((0, top, width, top + width))
    return cropped.resize((resolution, resolution), resample)


def pil_to_tensor(image: Image.Image, *, is_mask: bool) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if is_mask:
        if array.ndim == 3:
            array = array[..., 0]
        tensor = torch.from_numpy(array / 255.0).unsqueeze(0)
        return (tensor > 0.5).float()
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    tensor = torch.from_numpy(array / 255.0).permute(2, 0, 1)
    return tensor.clamp(0.0, 1.0)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        raise ValueError("--mask_dilate must be a positive odd integer.")
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


class MaskedInpaintDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_dir: Path,
        split: str,
        tokenizer: CLIPTokenizer,
        resolution: int,
        image_prep: str,
        fallback_prompt: str,
        allow_prompt_variants: bool,
        mask_dilate: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.examples = read_examples(dataset_dir, split, fallback_prompt)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.image_prep = image_prep
        self.fallback_prompt = fallback_prompt
        self.allow_prompt_variants = allow_prompt_variants
        self.mask_dilate = mask_dilate

        prompts = sorted({example.prompt for example in self.examples})
        if not allow_prompt_variants and prompts != [fallback_prompt]:
            raise ValueError(
                f"Prompt mismatch in {split} split: expected only {fallback_prompt!r}, found {prompts!r}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        input_image = fit_image(Image.open(example.input_path).convert("RGB"), self.resolution, self.image_prep, is_mask=False)
        target_image = fit_image(
            Image.open(example.target_path).convert("RGB"), self.resolution, self.image_prep, is_mask=False
        )
        mask_image = fit_image(Image.open(example.mask_path).convert("L"), self.resolution, self.image_prep, is_mask=True)

        input_tensor = pil_to_tensor(input_image, is_mask=False)
        target_tensor = pil_to_tensor(target_image, is_mask=False)
        mask_tensor = pil_to_tensor(mask_image, is_mask=True)
        if self.mask_dilate > 1:
            mask_tensor = dilate_mask(mask_tensor.unsqueeze(0), self.mask_dilate).squeeze(0)
        tokenized = self.tokenizer(
            example.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pair_id": example.pair_id,
            "prompt": example.prompt,
            "task": example.task,
            "input_pixel_values": input_tensor,
            "target_pixel_values": target_tensor,
            "mask_pixel_values": mask_tensor,
            "input_ids": tokenized.input_ids.squeeze(0),
        }


def collate_examples(examples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pair_id": [example["pair_id"] for example in examples],
        "prompt": [example["prompt"] for example in examples],
        "task": [example["task"] for example in examples],
        "input_pixel_values": torch.stack([example["input_pixel_values"] for example in examples]),
        "target_pixel_values": torch.stack([example["target_pixel_values"] for example in examples]),
        "mask_pixel_values": torch.stack([example["mask_pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
    }


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand(-1, value.shape[1], -1, -1)
    denom = expanded_mask.sum().clamp_min(1.0)
    return (value * expanded_mask).sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, mask)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    threshold = 0.04045
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055) ** 2.4
    return torch.where(rgb <= threshold, low, high)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    linear = srgb_to_linear(rgb)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    epsilon = 216 / 24389
    kappa = 24389 / 27

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > epsilon, torch.pow(t, 1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([l, a, b], dim=1)


def encode_latents(vae: AutoencoderKL, image_01: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    image = image_01.to(dtype=dtype) * 2.0 - 1.0
    latents = vae.encode(image).latent_dist.sample()
    return latents * vae.config.scaling_factor


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def predict_x0(
    scheduler: DDPMScheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    target_noise: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma_t = (1.0 - alpha_t).sqrt()
    sqrt_alpha_t = alpha_t.sqrt()

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        epsilon = model_pred
    elif prediction_type == "v_prediction":
        epsilon = sqrt_alpha_t * model_pred + sigma_t * noisy_latents
    elif prediction_type == "sample":
        return model_pred
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return (noisy_latents - sigma_t * epsilon) / sqrt_alpha_t.clamp_min(1e-6)


def make_preview_grid(
    *,
    pair_ids: list[str],
    inputs: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    path: Path,
) -> None:
    rows = min(len(pair_ids), inputs.shape[0])
    tile_width = inputs.shape[-1]
    tile_height = inputs.shape[-2]
    canvas = Image.new("RGB", (tile_width * 4 + 50, rows * (tile_height + 40) + 10), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    labels = ["input", "mask", "pred", "target"]
    for row_idx in range(rows):
        y = 10 + row_idx * (tile_height + 40)
        draw.text((10, y), pair_ids[row_idx], fill=(245, 245, 245))
        images = [
            tensor_to_pil(inputs[row_idx]),
            tensor_to_pil(masks[row_idx].repeat(3, 1, 1)),
            tensor_to_pil(preds[row_idx]),
            tensor_to_pil(targets[row_idx]),
        ]
        for col_idx, (label, image) in enumerate(zip(labels, images)):
            x = 10 + col_idx * (tile_width + 10)
            draw.text((x, y + 16), label, fill=(220, 220, 220))
            canvas.paste(image, (x, y + 32))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8))


def save_lora_weights(unet: UNet2DConditionModel, output_dir: Path, step: int, is_main_process: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_state = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state,
        is_main_process=is_main_process,
        weight_name=f"pytorch_lora_weights_step_{step:06d}.safetensors",
        safe_serialization=True,
    )


def run_preview(
    *,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    preview_dir: Path,
    global_step: int,
    max_batches: int,
) -> None:
    unet.eval()
    vae.eval()
    text_encoder.eval()
    batches_seen = 0
    preview_inputs: list[torch.Tensor] = []
    preview_masks: list[torch.Tensor] = []
    preview_preds: list[torch.Tensor] = []
    preview_targets: list[torch.Tensor] = []
    preview_ids: list[str] = []
    for batch in dataloader:
        if batches_seen >= max_batches:
            break
        inputs = batch["input_pixel_values"].to(accelerator.device)
        targets = batch["target_pixel_values"].to(accelerator.device)
        masks = batch["mask_pixel_values"].to(accelerator.device)
        masked_inputs = inputs * (1.0 - masks)

        with torch.no_grad():
            target_latents = encode_latents(vae, targets, dtype=vae.dtype)
            masked_input_latents = encode_latents(vae, masked_inputs, dtype=vae.dtype)
            mask_latents = F.interpolate(masks, size=target_latents.shape[-2:], mode="nearest")
            timesteps = torch.full(
                (inputs.shape[0],),
                fill_value=min(50, scheduler.config.num_train_timesteps - 1),
                device=accelerator.device,
                dtype=torch.long,
            )
            noise = torch.zeros_like(target_latents)
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
            model_input = torch.cat([noisy_latents, mask_latents, masked_input_latents], dim=1)
            model_pred = unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            pred_x0 = predict_x0(scheduler, noisy_latents, model_pred, timesteps, noise)
            pred_images = decode_latents(vae, pred_x0)

        preview_inputs.append(inputs.cpu())
        preview_masks.append(masks.cpu())
        preview_preds.append(pred_images.cpu())
        preview_targets.append(targets.cpu())
        preview_ids.extend(batch["pair_id"])
        batches_seen += 1

    if preview_inputs and accelerator.is_main_process:
        inputs = torch.cat(preview_inputs, dim=0)
        masks = torch.cat(preview_masks, dim=0)
        preds = torch.cat(preview_preds, dim=0)
        targets = torch.cat(preview_targets, dim=0)
        make_preview_grid(
            pair_ids=preview_ids,
            inputs=inputs,
            masks=masks,
            preds=preds,
            targets=targets,
            path=preview_dir / f"preview_step_{global_step:06d}.png",
        )
    unet.train()


def main() -> None:
    args = parse_args()
    if args.rank <= 0:
        raise ValueError("--rank must be positive.")
    if args.mask_dilate < 1 or args.mask_dilate % 2 == 0:
        raise ValueError("--mask_dilate must be a positive odd integer.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if getattr(unet.config, "in_channels", None) != 9:
        raise ValueError(
            f"Expected an inpainting UNet with 9 input channels, found {getattr(unet.config, 'in_channels', None)}."
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xFormers is not available, but --enable_xformers_memory_efficient_attention was set.")
        unet.enable_xformers_memory_efficient_attention()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha or args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in unet.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = MaskedInpaintDataset(
        dataset_dir=args.dataset_dir,
        split=args.train_split,
        tokenizer=tokenizer,
        resolution=args.resolution,
        image_prep=args.image_prep,
        fallback_prompt=args.prompt,
        allow_prompt_variants=args.allow_dataset_prompt_variants,
        mask_dilate=args.mask_dilate,
    )
    val_dataset = MaskedInpaintDataset(
        dataset_dir=args.dataset_dir,
        split=args.val_split,
        tokenizer=tokenizer,
        resolution=args.resolution,
        image_prep=args.image_prep,
        fallback_prompt=args.prompt,
        allow_prompt_variants=True,
        mask_dilate=args.mask_dilate,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_examples,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_examples,
    )

    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    if args.max_train_steps is None or args.max_train_steps <= 0:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    checkpoints_dir = args.output_dir / "lora_checkpoints"
    metrics_path = args.output_dir / "metrics.jsonl"
    config_path = args.output_dir / "training_config.json"
    if accelerator.is_main_process:
        config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                input_images = batch["input_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                target_images = batch["target_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                masks = batch["mask_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                masked_inputs = input_images * (1.0 - masks)

                with torch.no_grad():
                    target_latents = encode_latents(vae, target_images, dtype=torch.float32)
                    masked_input_latents = encode_latents(vae, masked_inputs, dtype=torch.float32)
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                    dtype=torch.long,
                )
                noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
                mask_latents = F.interpolate(masks, size=target_latents.shape[-2:], mode="nearest")
                model_input = torch.cat([noisy_latents, mask_latents, masked_input_latents], dim=1)
                model_pred = unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                if scheduler.config.prediction_type == "epsilon":
                    diffusion_target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    diffusion_target = scheduler.get_velocity(target_latents, noise, timesteps)
                elif scheduler.config.prediction_type == "sample":
                    diffusion_target = target_latents
                else:
                    raise ValueError(f"Unsupported scheduler prediction type: {scheduler.config.prediction_type}")

                loss_diffusion = masked_mse(model_pred, diffusion_target, mask_latents)
                pred_x0 = predict_x0(scheduler, noisy_latents, model_pred, timesteps, noise)
                pred_images = decode_latents(vae, pred_x0)
                loss_mask_rgb = masked_l1(pred_images, target_images, masks)
                loss_identity = masked_l1(pred_images, input_images, 1.0 - masks)
                loss_color = masked_l1(rgb_to_lab(pred_images), rgb_to_lab(target_images), masks)

                loss = (
                    args.lambda_diffusion * loss_diffusion
                    + args.lambda_mask_rgb * loss_mask_rgb
                    + args.lambda_identity * loss_identity
                    + args.lambda_color * loss_color
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "loss_diffusion": float(loss_diffusion.detach().item()),
                    "loss_mask_rgb": float(loss_mask_rgb.detach().item()),
                    "loss_identity": float(loss_identity.detach().item()),
                    "loss_color": float(loss_color.detach().item()),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "mean_mask_ratio": float(masks.mean().item()),
                }
                progress_bar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    ident=f"{metrics['loss_identity']:.4f}",
                    mask=f"{metrics['loss_mask_rgb']:.4f}",
                )
                if accelerator.is_main_process:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")

                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_lora_weights(accelerator.unwrap_model(unet), checkpoints_dir, global_step, True)
                if global_step % args.preview_steps == 0:
                    accelerator.wait_for_everyone()
                    run_preview(
                        accelerator=accelerator,
                        vae=vae,
                        text_encoder=text_encoder,
                        unet=accelerator.unwrap_model(unet),
                        scheduler=scheduler,
                        dataloader=val_dataloader,
                        preview_dir=previews_dir,
                        global_step=global_step,
                        max_batches=args.validation_samples,
                    )
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(accelerator.unwrap_model(unet), checkpoints_dir, global_step, True)


if __name__ == "__main__":
    main()
