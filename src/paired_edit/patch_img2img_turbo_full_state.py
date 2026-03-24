#!/usr/bin/env python3
"""Patch img2img-turbo training to save and resume full training state."""

from __future__ import annotations

import re
import sys
from pathlib import Path


OUTPUT_DIR_NEEDLE = """    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
"""

OUTPUT_DIR_INSERT = """    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    resume_state = None
    resume_state_path = os.environ.get("IMG2IMG_TURBO_RESUME_STATE")
    resume_pkl = os.environ.get("IMG2IMG_TURBO_RESUME_PKL")
    if resume_state_path:
        print(f"Resuming pix2pix-turbo full training state from: {resume_state_path}")
        resume_state = torch.load(resume_state_path, map_location="cpu")
        resume_pkl = resume_state.get("model_path", resume_pkl)
"""

PREPARE_NEEDLE = """    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
"""

PREPARE_INSERT = """    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    global_step = 0
    starting_epoch = 0
    resume_step_in_epoch = -1
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        optimizer_disc.load_state_dict(resume_state["optimizer_disc"])
        lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
        lr_scheduler_disc.load_state_dict(resume_state["lr_scheduler_disc"])
        global_step = int(resume_state.get("global_step", 0))
        starting_epoch = int(resume_state.get("epoch", 0))
        resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))
        print(
            f"Resumed optimizer/scheduler state at "
            f"global_step={global_step}, epoch={starting_epoch}, step_in_epoch={resume_step_in_epoch}"
        )
"""

PROGRESS_NEEDLE = """    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
"""

PROGRESS_INSERT = """    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
"""

LOOP_NEEDLE = """    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
"""

LOOP_INSERT = """    # start the training loop
    for epoch in range(starting_epoch, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if epoch == starting_epoch and step <= resume_step_in_epoch:
                continue
"""

CHECKPOINT_NEEDLE = """                            if global_step % args.checkpointing_steps == 1:
                                outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                                accelerator.unwrap_model(net_pix2pix).save_model(outf)
"""

CHECKPOINT_INSERT = """                            if global_step % args.checkpointing_steps == 1:
                                outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                                accelerator.unwrap_model(net_pix2pix).save_model(outf)
                                state_outf = os.path.join(
                                    args.output_dir,
                                    "checkpoints",
                                    f"training_state_{global_step}.pt",
                                )
                                accelerator.save(
                                    {
                                        "model_path": outf,
                                        "global_step": global_step,
                                        "epoch": epoch,
                                        "step_in_epoch": step,
                                        "optimizer": optimizer.state_dict(),
                                        "optimizer_disc": optimizer_disc.state_dict(),
                                        "lr_scheduler": lr_scheduler.state_dict(),
                                        "lr_scheduler_disc": lr_scheduler_disc.state_dict(),
                                    },
                                    state_outf,
                                )
"""

BREAK_NEEDLE = """                accelerator.log(logs, step=global_step)
"""

BREAK_INSERT = """                accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
"""


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    if replacement in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find expected block for {label}.")
    return text.replace(needle, replacement, 1)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_img2img_turbo_full_state.py /path/to/train_pix2pix_turbo.py", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    text = target.read_text(encoding="utf-8")

    text = replace_once(text, OUTPUT_DIR_NEEDLE, OUTPUT_DIR_INSERT, "resume state bootstrap")
    text = replace_once(text, PREPARE_NEEDLE, PREPARE_INSERT, "optimizer state restore")
    text = replace_once(text, PROGRESS_NEEDLE, PROGRESS_INSERT, "progress bar init")
    text = replace_once(text, LOOP_NEEDLE, LOOP_INSERT, "training loop resume")
    text = replace_once(text, CHECKPOINT_NEEDLE, CHECKPOINT_INSERT, "checkpoint save")
    text = replace_once(text, BREAK_NEEDLE, BREAK_INSERT, "max_train_steps break")

    target.write_text(text, encoding="utf-8")
    print(f"Patched full-state save/resume into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
