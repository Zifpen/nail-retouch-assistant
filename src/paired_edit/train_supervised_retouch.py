#!/usr/bin/env python3
"""
Train a structure-preserving paired retouch model with supervised losses only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from pix2pix_runtime import apply_runtime_checkpoint_patch, prepare_runtime_upstream
from shared_config import PAIRED_EDIT_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised paired retouch model.")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--upstream_src_dir", default=None)
    parser.add_argument("--paired_prompt", default=PAIRED_EDIT_PROMPT)
    parser.add_argument("--allow_dataset_prompt_variants", action="store_true")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_image_prep", default="resize_256")
    parser.add_argument("--test_image_prep", default="resize_256")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=120)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_rank_unet", type=int, default=8)
    parser.add_argument("--lora_rank_vae", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    parser.add_argument("--lambda_full_l1", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=2.0)
    parser.add_argument("--lambda_edit", type=float, default=2.0)
    parser.add_argument("--lambda_l2", type=float, default=0.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    parser.add_argument("--lambda_gan", type=float, default=0.0)
    parser.add_argument("--lambda_clipsim", type=float, default=0.0)
    parser.add_argument("--change_mask_threshold", type=float, default=0.12)
    parser.add_argument("--change_mask_dilate", type=int, default=3)
    return parser.parse_args()


def add_upstream_src_to_path(upstream_src_dir: str | None) -> Path:
    if upstream_src_dir:
        src_dir = Path(upstream_src_dir)
    else:
        src_dir = Path.cwd() / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Upstream img2img-turbo src not found: {src_dir}")
    sys.path.insert(0, str(src_dir))
    return src_dir


def prepare_training_runtime(upstream_src_dir: str | None, device: str) -> Path:
    src_dir = add_upstream_src_to_path(upstream_src_dir)
    upstream_root = src_dir.parent if src_dir.name == "src" else src_dir
    runtime_root = prepare_runtime_upstream(upstream_root, device)
    apply_runtime_checkpoint_patch(runtime_root)
    runtime_src = runtime_root / "src"
    sys.path.insert(0, str(runtime_src))
    os.environ["PIX2PIX_TURBO_DEVICE"] = device
    return runtime_src


def resolve_resume_model_path(
    *,
    resume_state_path: str | None,
    resume_pkl: str | None,
    output_dir: Path,
    global_step_hint: int | None = None,
) -> str | None:
    candidates: list[Path] = []
    if resume_pkl:
        candidates.append(Path(resume_pkl))
    if resume_state_path:
        state_path = Path(resume_state_path)
        if global_step_hint is not None:
            candidates.append(state_path.with_name(f"model_{global_step_hint}.pkl"))
        state_name = state_path.name
        if state_name.startswith("training_state_") and state_name.endswith(".pt"):
            suffix = state_name[len("training_state_") : -len(".pt")]
            candidates.append(state_path.with_name(f"model_{suffix}.pkl"))
    if global_step_hint is not None:
        candidates.append(output_dir / "checkpoints" / f"model_{global_step_hint}.pkl")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)
    return resume_pkl


def save_image_tensor(tensor: torch.Tensor, path: Path, *, source: bool) -> None:
    tensor = tensor.detach().cpu().squeeze(0)
    if source:
        tensor = tensor.clamp(0, 1)
    else:
        tensor = tensor.clamp(-1, 1) * 0.5 + 0.5
    image = transforms.ToPILImage()(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_triptych(source: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    images = [
        ("input", transforms.ToPILImage()(source.detach().cpu().squeeze(0).clamp(0, 1))),
        ("output", transforms.ToPILImage()((pred.detach().cpu().squeeze(0).clamp(-1, 1) * 0.5 + 0.5))),
        ("target", transforms.ToPILImage()((target.detach().cpu().squeeze(0).clamp(-1, 1) * 0.5 + 0.5))),
    ]
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    canvas = Image.new("RGB", (width * 3 + 40, height + 40), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(images):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        x = 10 + idx * (width + 10)
        canvas.paste(image, (x, 24))
        draw.text((x, 5), label, fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def ensure_dataset_prompt_consistency(dataset_folder: Path, paired_prompt: str) -> None:
    for prompt_name in ("train_prompts.json", "test_prompts.json"):
        prompt_path = dataset_folder / prompt_name
        if not prompt_path.exists():
            continue
        prompt_map = json.loads(prompt_path.read_text(encoding="utf-8"))
        unique_prompts = {str(value) for value in prompt_map.values()}
        if not unique_prompts:
            continue
        if unique_prompts != {paired_prompt}:
            raise ValueError(
                f"Prompt mismatch in {prompt_path}: expected only {paired_prompt!r}, found {sorted(unique_prompts)!r}"
            )


def summarize_dataset_prompts(dataset_folder: Path) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for prompt_name in ("train_prompts.json", "test_prompts.json"):
        prompt_path = dataset_folder / prompt_name
        if not prompt_path.exists():
            continue
        prompt_map = json.loads(prompt_path.read_text(encoding="utf-8"))
        summary[prompt_name] = sorted({str(value) for value in prompt_map.values()})
    return summary


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = expanded_mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * expanded_mask).sum() / denom


def build_change_masks(
    x_src_01: torch.Tensor,
    x_tgt_norm: torch.Tensor,
    *,
    threshold: float,
    dilate_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_src_01 = x_src_01.float()
    x_tgt_01 = x_tgt_norm.float() * 0.5 + 0.5
    diff = (x_tgt_01 - x_src_01).abs().amax(dim=1, keepdim=True)
    change_mask = (diff > threshold).float()
    if dilate_kernel > 1:
        change_mask = F.max_pool2d(
            change_mask,
            kernel_size=dilate_kernel,
            stride=1,
            padding=dilate_kernel // 2,
        )
    preserve_mask = 1.0 - change_mask
    return change_mask, preserve_mask


def mask_ratio(mask: torch.Tensor) -> float:
    return float(mask.detach().float().mean().item())


def validate_mask_sanity(change_mask: torch.Tensor, preserve_mask: torch.Tensor) -> dict[str, float]:
    if change_mask.shape != preserve_mask.shape:
        raise ValueError(f"Mask shape mismatch: {change_mask.shape} vs {preserve_mask.shape}")
    if not torch.allclose(change_mask + preserve_mask, torch.ones_like(change_mask)):
        raise ValueError("Preserve mask is not complementary to change mask.")

    change_ratio = change_mask.detach().float().mean(dim=(1, 2, 3))
    if torch.any(change_ratio <= 1e-4):
        raise ValueError(
            f"Change mask is empty for at least one sample: ratios={change_ratio.tolist()}"
        )
    if torch.any(change_ratio >= 0.60):
        raise ValueError(
            f"Change mask covers too much of the image: ratios={change_ratio.tolist()}"
        )
    preserve_ratio = preserve_mask.detach().float().mean(dim=(1, 2, 3))
    return {
        "change_ratio_min": float(change_ratio.min().item()),
        "change_ratio_max": float(change_ratio.max().item()),
        "change_ratio_mean": float(change_ratio.mean().item()),
        "preserve_ratio_min": float(preserve_ratio.min().item()),
        "preserve_ratio_max": float(preserve_ratio.max().item()),
        "preserve_ratio_mean": float(preserve_ratio.mean().item()),
    }


def evaluate(model, dataloader, net_lpips, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    full_l1_vals: list[float] = []
    preserve_l1_vals: list[float] = []
    edit_l1_vals: list[float] = []
    l2_vals: list[float] = []
    lpips_vals: list[float] = []
    change_ratio_vals: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            x_src = batch["conditioning_pixel_values"].to(device)
            x_tgt = batch["output_pixel_values"].to(device)
            x_pred = model(x_src, prompt_tokens=batch["input_ids"].to(device), deterministic=True)
            x_pred_f = x_pred.float()
            x_tgt_f = x_tgt.float()
            src_norm = x_src.float() * 2.0 - 1.0
            change_mask, preserve_mask = build_change_masks(
                x_src,
                x_tgt,
                threshold=args.change_mask_threshold,
                dilate_kernel=args.change_mask_dilate,
            )
            validate_mask_sanity(change_mask, preserve_mask)
            full_l1_vals.append(F.l1_loss(x_pred_f, x_tgt_f, reduction="mean").item())
            preserve_l1_vals.append(masked_l1(x_pred_f, src_norm, preserve_mask).item())
            edit_l1_vals.append(masked_l1(x_pred_f, x_tgt_f, change_mask).item())
            l2_vals.append(F.mse_loss(x_pred_f, x_tgt_f, reduction="mean").item())
            lpips_vals.append(net_lpips(x_pred_f, x_tgt_f).mean().item())
            change_ratio_vals.append(mask_ratio(change_mask))
    model.train()
    return {
        "val/full_l1": float(sum(full_l1_vals) / max(1, len(full_l1_vals))),
        "val/preserve_l1": float(sum(preserve_l1_vals) / max(1, len(preserve_l1_vals))),
        "val/edit_l1": float(sum(edit_l1_vals) / max(1, len(edit_l1_vals))),
        "val/l2": float(sum(l2_vals) / max(1, len(l2_vals))),
        "val/lpips": float(sum(lpips_vals) / max(1, len(lpips_vals))),
        "val/change_ratio": float(sum(change_ratio_vals) / max(1, len(change_ratio_vals))),
    }


def main(args: argparse.Namespace) -> None:
    if args.change_mask_dilate < 1 or args.change_mask_dilate % 2 == 0:
        raise ValueError("--change_mask_dilate must be a positive odd integer.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    runtime_device = accelerator.device.type
    prepare_training_runtime(args.upstream_src_dir, runtime_device)

    from my_utils.training_utils import PairedDataset
    from pix2pix_turbo import Pix2Pix_Turbo

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    eval_dir = output_dir / "eval"
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    dataset_folder = Path(args.dataset_folder)
    if args.allow_dataset_prompt_variants:
        if accelerator.is_main_process:
            print("Dataset prompt variants allowed:")
            print(json.dumps(summarize_dataset_prompts(dataset_folder), indent=2, ensure_ascii=False))
    else:
        ensure_dataset_prompt_consistency(dataset_folder, args.paired_prompt)

    resume_state = None
    resume_pkl = os.environ.get("IMG2IMG_TURBO_RESUME_PKL")
    resume_state_path = os.environ.get("IMG2IMG_TURBO_RESUME_STATE")
    if resume_state_path:
        resume_state = torch.load(resume_state_path, map_location="cpu")
        resume_pkl = resume_state.get("model_path", resume_pkl)
        resume_pkl = resolve_resume_model_path(
            resume_state_path=resume_state_path,
            resume_pkl=resume_pkl,
            output_dir=output_dir,
            global_step_hint=resume_state.get("global_step"),
        )
        if accelerator.is_main_process:
            print(f"Resuming supervised retouch full state from: {resume_state_path}")
            print(f"Resolved supervised retouch model weights: {resume_pkl}")
    elif resume_pkl and accelerator.is_main_process:
        print(f"Resuming supervised retouch weights from: {resume_pkl}")

    if resume_pkl:
        model = Pix2Pix_Turbo(
            pretrained_path=resume_pkl,
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
        )
    elif args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        model = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
    else:
        raise ValueError(f"Unsupported base model: {args.pretrained_model_name_or_path}")
    model.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available, please install it first.")
        model.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    train_dataset = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        tokenizer=model.tokenizer,
    )
    val_dataset = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="test",
        tokenizer=model.tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    params_to_opt = []
    for name, param in model.unet.named_parameters():
        if "lora" in name:
            params_to_opt.append(param)
    params_to_opt += list(model.unet.conv_in.parameters())
    for name, param in model.vae.named_parameters():
        if "lora" in name and "vae_skip" in name:
            params_to_opt.append(param)
    params_to_opt += list(model.vae.decoder.skip_conv_1.parameters())
    params_to_opt += list(model.vae.decoder.skip_conv_2.parameters())
    params_to_opt += list(model.vae.decoder.skip_conv_3.parameters())
    params_to_opt += list(model.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model.to(dtype=weight_dtype)

    net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    global_step = 0
    starting_epoch = 0
    resume_step_in_epoch = -1
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
        global_step = int(resume_state.get("global_step", 0))
        starting_epoch = int(resume_state.get("epoch", 0))
        resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))
        if accelerator.is_main_process:
            print(
                "Resumed optimizer/scheduler state at "
                f"global_step={global_step}, epoch={starting_epoch}, step_in_epoch={resume_step_in_epoch}"
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    model.train()
    mask_sanity_logged = False
    for epoch in range(starting_epoch, args.num_training_epochs):
        if global_step >= args.max_train_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if epoch == starting_epoch and step <= resume_step_in_epoch:
                continue
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(model):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                src_norm = x_src.float() * 2.0 - 1.0
                change_mask, preserve_mask = build_change_masks(
                    x_src,
                    x_tgt,
                    threshold=args.change_mask_threshold,
                    dilate_kernel=args.change_mask_dilate,
                )
                if not mask_sanity_logged:
                    sanity_stats = validate_mask_sanity(change_mask, preserve_mask)
                    if accelerator.is_main_process:
                        print(f"Mask sanity: {json.dumps(sanity_stats, indent=2)}")
                    mask_sanity_logged = True

                x_pred = model(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                x_pred_f = x_pred.float()
                x_tgt_f = x_tgt.float()

                loss_full_l1 = F.l1_loss(x_pred_f, x_tgt_f, reduction="mean") * args.lambda_full_l1
                loss_preserve = masked_l1(x_pred_f, src_norm, preserve_mask) * args.lambda_preserve
                loss_edit = masked_l1(x_pred_f, x_tgt_f, change_mask) * args.lambda_edit
                loss_lpips = net_lpips(x_pred_f, x_tgt_f).mean() * args.lambda_lpips
                loss_l2 = F.mse_loss(x_pred_f, x_tgt_f, reduction="mean") * args.lambda_l2
                loss = loss_full_l1 + loss_preserve + loss_edit + loss_lpips + loss_l2

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if not accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "loss_full_l1": loss_full_l1.detach().item(),
                "loss_preserve": loss_preserve.detach().item(),
                "loss_edit": loss_edit.detach().item(),
                "loss_l2": loss_l2.detach().item(),
                "loss_lpips": loss_lpips.detach().item(),
                "change_ratio": mask_ratio(change_mask),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process and (
                global_step == 1 or global_step % args.viz_freq == 0
            ):
                sample_path = sample_dir / f"train_step_{global_step:06d}.png"
                save_triptych(x_src[0], x_pred[0], x_tgt[0], sample_path)
                print(f"Saved sample: {sample_path}")

            if accelerator.is_main_process and (
                global_step == 1 or global_step % args.eval_freq == 0
            ):
                eval_logs = evaluate(
                    accelerator.unwrap_model(model),
                    val_dataloader,
                    net_lpips,
                    accelerator.device,
                    args,
                )
                summary_path = eval_dir / f"metrics_{global_step:06d}.json"
                summary_path.write_text(json.dumps(eval_logs, indent=2), encoding="utf-8")
                print(f"Saved eval metrics: {summary_path}")
                print(json.dumps(eval_logs, indent=2))

            if accelerator.is_main_process and (
                global_step == 1 or global_step % args.checkpointing_steps == 0
            ):
                model_path = ckpt_dir / f"model_{global_step}.pkl"
                accelerator.unwrap_model(model).save_model(str(model_path))
                state_path = ckpt_dir / f"training_state_{global_step}.pt"
                accelerator.save(
                    {
                        "model_path": str(model_path),
                        "global_step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    state_path,
                )
                print(f"Saved checkpoint: {model_path}")
                print(f"Saved training state: {state_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(parse_args())
