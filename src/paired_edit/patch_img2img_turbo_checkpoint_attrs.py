#!/usr/bin/env python3
"""Patch img2img-turbo checkpoint branches to restore save_model attrs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERN = re.compile(
    r'(?m)^(?P<indent>[ \t]*)vae_lora_config = LoraConfig\(r=sd\["rank_vae"\], init_lora_weights="gaussian", target_modules=sd\["vae_lora_target_modules"\]\)\n'
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_img2img_turbo_checkpoint_attrs.py /path/to/pix2pix_turbo.py", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    text = target.read_text(encoding="utf-8")

    if 'self.target_modules_unet = sd["unet_lora_target_modules"]' in text:
        print(f"Already patched: {target}")
        return 0

    matches = list(PATTERN.finditer(text))
    count = len(matches)
    if count < 3:
        raise RuntimeError(f"Expected at least 3 checkpoint branches to patch, found {count}.")

    def replacer(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f'{indent}vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", '
            'target_modules=sd["vae_lora_target_modules"])\n'
            f'{indent}self.lora_rank_unet = sd["rank_unet"]\n'
            f'{indent}self.lora_rank_vae = sd["rank_vae"]\n'
            f'{indent}self.target_modules_vae = sd["vae_lora_target_modules"]\n'
            f'{indent}self.target_modules_unet = sd["unet_lora_target_modules"]\n'
        )

    patched, replaced = PATTERN.subn(replacer, text, count=3)
    if replaced != 3:
        raise RuntimeError(f"Expected to patch 3 checkpoint branches, patched {replaced}.")

    target.write_text(patched, encoding="utf-8")
    print(f"Patched checkpoint attrs into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
