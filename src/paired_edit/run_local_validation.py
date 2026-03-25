#!/usr/bin/env python3
"""
Run local paired-edit validation for pix2pix-turbo checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")
DEFAULT_DATASET_DIR = Path("dataset/paired_edit_strict_plus")
DEFAULT_OUTPUT_ROOT = Path("outputs/paired_edit_validation")
DEFAULT_UPSTREAM_DIR = Path("/tmp/img2img-turbo-local")
DEFAULT_PAIR_IDS = ["pair_0009", "pair_0040", "pair_0047", "pair_0050", "pair_0064"]
BACKGROUND = (18, 18, 20)
LABEL_COLOR = (245, 245, 245)
LABEL_HEIGHT = 42
PADDING = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired-edit pix2pix-turbo checkpoints locally.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Specific model_*.pkl checkpoint. Defaults to the latest file in outputs/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing model_*.pkl checkpoints.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Paired-edit dataset directory containing test_A/test_B and metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated validation outputs.",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="Local clone of the upstream img2img-turbo repository.",
    )
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Specific test pair id. Can be passed multiple times. Defaults to the 5 held-out pairs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Runtime device.",
    )
    return parser.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        checkpoint_dir.glob("model_*.pkl"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No model_*.pkl checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def prepare_runtime_upstream(upstream_dir: Path, device: str) -> Path:
    if not upstream_dir.exists():
        raise FileNotFoundError(
            f"Upstream repo not found at {upstream_dir}. Clone https://github.com/GaParmar/img2img-turbo there first."
        )

    runtime_root = Path("/tmp/img2img-turbo-runtime") / device
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    shutil.copytree(upstream_dir / "src", runtime_root / "src")

    replacements = {
        '.cuda()': '.to(DEVICE)',
        'device="cuda"': 'device=DEVICE',
        "device='cuda'": "device=DEVICE",
        '.to("cuda")': '.to(DEVICE)',
        ".to('cuda')": ".to(DEVICE)",
    }
    for rel in ("src/pix2pix_turbo.py", "src/model.py"):
        path = runtime_root / rel
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

    return runtime_root


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def load_image_tensor(path: Path, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
    tensor = array.permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0).clamp(-1, 1)
    tensor = ((tensor + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def build_sheet(input_path: Path, output_path: Path, target_path: Path, sheet_path: Path) -> None:
    images = [
        ("input", Image.open(input_path).convert("RGB")),
        ("output", Image.open(output_path).convert("RGB")),
        ("target", Image.open(target_path).convert("RGB")),
    ]
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    fitted = []
    for label, image in images:
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        fitted.append((label, image))

    sheet = Image.new(
        "RGB",
        ((width * 3) + (PADDING * 4), height + LABEL_HEIGHT + (PADDING * 2)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = PADDING
    for label, image in fitted:
        sheet.paste(image, (x, PADDING + LABEL_HEIGHT))
        draw.text((x, PADDING), label, fill=LABEL_COLOR, font=font)
        x += width + PADDING

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    pair_ids = args.pair_id or DEFAULT_PAIR_IDS

    runtime_root = prepare_runtime_upstream(args.upstream_dir, device)
    sys.path.insert(0, str(runtime_root / "src"))
    import os

    os.environ["PIX2PIX_TURBO_DEVICE"] = device
    from pix2pix_turbo import Pix2Pix_Turbo

    metadata = load_metadata(args.dataset_dir / "test_metadata.jsonl")
    output_dir = args.output_root / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    model = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
    model.set_eval()

    if device == "cpu":
        model = model.to(torch.float32)

    for pair_id in pair_ids:
        row = metadata[pair_id]
        input_path = args.dataset_dir / row["input"]
        target_path = args.dataset_dir / row["target"]
        output_path = output_dir / f"{pair_id}_output.png"
        sheet_path = output_dir / f"{pair_id}_sheet.png"

        x_src = load_image_tensor(input_path, device)
        with torch.no_grad():
            x_out = model(x_src, prompt=row["prompt"], deterministic=True)

        image = tensor_to_image(x_out)
        image.save(output_path)
        build_sheet(input_path, output_path, target_path, sheet_path)
        print(f"Saved output: {output_path}")
        print(f"Saved sheet: {sheet_path}")

    summary = {
        "checkpoint": checkpoint.name,
        "device": device,
        "pairs": pair_ids,
        "output_dir": str(output_dir),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {output_dir / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
