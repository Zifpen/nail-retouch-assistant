#!/usr/bin/env python3
"""Patch img2img-turbo training to save and resume full training state."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def replace_regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    if replacement in text:
        return text
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)
    patched, count = compiled.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find expected block for {label}.")
    return patched


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_img2img_turbo_full_state.py /path/to/train_pix2pix_turbo.py", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    text = target.read_text(encoding="utf-8")

    text = replace_regex_once(
        text,
        r'''(?ms)^    if accelerator\.is_main_process:\n        os\.makedirs\(os\.path\.join\(args\.output_dir, "checkpoints"\), exist_ok=True\)\n        os\.makedirs\(os\.path\.join\(args\.output_dir, "eval"\), exist_ok=True\)\n''',
        '''    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    resume_state = None
    resume_state_path = os.environ.get("IMG2IMG_TURBO_RESUME_STATE")
    resume_pkl = os.environ.get("IMG2IMG_TURBO_RESUME_PKL")
    if resume_state_path:
        print(f"Resuming pix2pix-turbo full training state from: {resume_state_path}")
        resume_state = torch.load(resume_state_path, map_location="cpu")
        resume_pkl = resume_state.get("model_path", resume_pkl)
''',
        "resume state bootstrap",
    )

    text = replace_regex_once(
        text,
        r'''(?ms)^    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator\.prepare\(\n        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc\n    \)\n    net_clip, net_lpips = accelerator\.prepare\(net_clip, net_lpips\)\n''',
        '''    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
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
''',
        "optimizer state restore",
    )

    text = replace_regex_once(
        text,
        r'''(?m)^    progress_bar = tqdm\(range\(0, args\.max_train_steps\), initial=0, desc="Steps", disable=not accelerator\.is_local_main_process,\)\n''',
        '''    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
''',
        "progress bar init",
    )

    text = replace_regex_once(
        text,
        r'''(?ms)^    # start the training loop\n    global_step = 0\n    for epoch in range\(0, args\.num_training_epochs\):\n        for step, batch in enumerate\(dl_train\):\n''',
        '''    # start the training loop
    for epoch in range(starting_epoch, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if epoch == starting_epoch and step <= resume_step_in_epoch:
                continue
''',
        "training loop resume",
    )

    text = replace_regex_once(
        text,
        r'''(?ms)^ {28}if global_step % args\.checkpointing_steps == 1:\n {32}outf = os\.path\.join\(args\.output_dir, "checkpoints", f"model_\{global_step\}\.pkl"\)\n {32}accelerator\.unwrap_model\(net_pix2pix\)\.save_model\(outf\)\n''',
        '''                            if global_step % args.checkpointing_steps == 1:
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
''',
        "checkpoint save",
    )

    text = replace_regex_once(
        text,
        r'''(?m)^ {16}accelerator\.log\(logs, step=global_step\)\n''',
        '''                accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
''',
        "max_train_steps break",
    )

    target.write_text(text, encoding="utf-8")
    print(f"Patched full-state save/resume into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
