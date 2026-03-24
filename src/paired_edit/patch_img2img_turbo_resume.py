#!/usr/bin/env python3
"""Patch img2img-turbo training to resume from a saved .pkl via env var."""

from __future__ import annotations

import re
import sys
from pathlib import Path


INSERT_SNIPPET = """        resume_pkl = os.environ.get("IMG2IMG_TURBO_RESUME_PKL")
        if resume_pkl:
            print(f"Resuming pix2pix-turbo weights from: {resume_pkl}")
            net_pix2pix = Pix2Pix_Turbo(
                pretrained_path=resume_pkl,
                lora_rank_unet=args.lora_rank_unet,
                lora_rank_vae=args.lora_rank_vae,
            )
        else:
            net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_img2img_turbo_resume.py /path/to/train_pix2pix_turbo.py", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    text = target.read_text(encoding="utf-8")

    if 'IMG2IMG_TURBO_RESUME_PKL' in text:
        print(f"Already patched: {target}")
        return 0

    pattern = re.compile(
        r'(?ms)^([ \t]*)if args\.pretrained_model_name_or_path == "stabilityai/sd-turbo":\n'
        r'[ \t]+net_pix2pix = Pix2Pix_Turbo\(lora_rank_unet=args\.lora_rank_unet, lora_rank_vae=args\.lora_rank_vae\)\n'
    )
    replacement = '    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":\\n' + INSERT_SNIPPET

    patched_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not find the expected Pix2Pix_Turbo init block to patch.")

    target.write_text(patched_text, encoding="utf-8")
    print(f"Patched resume support into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
