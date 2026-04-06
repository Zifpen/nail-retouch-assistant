#!/usr/bin/env python3
"""
Shared runtime helpers for paired-edit pix2pix-turbo inference.
"""

from __future__ import annotations

import shutil
import subprocess
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_UPSTREAM_DIR = Path("/content/img2img-turbo")


def resolve_default_path(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    return fallback


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        checkpoint_dir.glob("model_*.pkl"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No model_*.pkl checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


def prepare_runtime_upstream(upstream_dir: Path, device: str) -> Path:
    upstream_src = upstream_dir / "src"
    if not upstream_src.exists() or not upstream_src.is_dir():
        raise FileNotFoundError(
            f"Upstream repo src not found at {upstream_src}. Clone https://github.com/GaParmar/img2img-turbo there first."
        )

    runtime_root = Path("/tmp/img2img-turbo-runtime") / device
    runtime_parent = runtime_root.parent
    runtime_parent.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_parent / f".{runtime_root.name}.lock"
    staging_root = runtime_parent / f"{runtime_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    _remove_path(staging_root)

    import fcntl

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            _remove_path(staging_root)
            shutil.copytree(upstream_src, staging_root / "src")

            replacements = {
                ".cuda()": ".to(DEVICE)",
                'device="cuda"': "device=DEVICE",
                "device='cuda'": "device=DEVICE",
                '.to("cuda")': ".to(DEVICE)",
                ".to('cuda')": ".to(DEVICE)",
            }
            for rel in ("src/pix2pix_turbo.py", "src/model.py"):
                path = staging_root / rel
                text = path.read_text(encoding="utf-8")
                if 'DEVICE = os.environ.get("PIX2PIX_TURBO_DEVICE", "cuda")' not in text:
                    if "import torch\n" in text:
                        text = text.replace(
                            "import torch\n",
                            'import torch\nDEVICE = os.environ.get("PIX2PIX_TURBO_DEVICE", "cuda")\n',
                            1,
                        )
                    else:
                        text = text.replace(
                            "from tqdm import tqdm\n",
                            'from tqdm import tqdm\nDEVICE = os.environ.get("PIX2PIX_TURBO_DEVICE", "cuda")\n',
                            1,
                        )
                for old, new in replacements.items():
                    text = text.replace(old, new)
                path.write_text(text, encoding="utf-8")

            _remove_path(runtime_root)
            staging_root.rename(runtime_root)
        finally:
            if staging_root.exists():
                _remove_path(staging_root)

    return runtime_root


def apply_runtime_checkpoint_patch(runtime_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    patch_script = repo_root / "src" / "paired_edit" / "patch_img2img_turbo_checkpoint_attrs.py"
    target = runtime_root / "src" / "pix2pix_turbo.py"
    subprocess.run([sys.executable, str(patch_script), str(target)], check=True)


def load_image_tensor(path: Path, device: str, max_side: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        resized = (
            max(8, int(round(image.width * scale))),
            max(8, int(round(image.height * scale))),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)
    width = image.width - image.width % 8
    height = image.height - image.height % 8
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0).clamp(-1, 1)
    tensor = ((tensor + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)
