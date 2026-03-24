# Nail Retouch Assistant

This repository prepares paired manicure retouch data for LoRA training and future inference services.

## Raw dataset

Expected raw layout:

```text
raw/
  pair_0001/
    before.jpg
    after.jpg
```

## Preprocess dataset

Install dependencies:

```bash
python3 -m pip install pillow
```

Build processed train/val splits:

```bash
python3 src/data/preprocess_pairs.py --raw-dir raw --output-dir dataset/processed --size 768
```

Apply structured tags to metadata:

```bash
python3 src/data/apply_annotations.py
```

Outputs:

```text
dataset/processed/
  train/
    images/
    metadata.jsonl
  val/
    images/
    metadata.jsonl
```

Each metadata row contains:

- paired source image path
- paired target image path
- default prompt
- default negative prompt
- optional structured tags: `shape`, `finish`, `color_family`

## Paired Edit Dataset

Build a paired `before -> after` dataset for structure-preserving retouch training:

```bash
python3 src/data/build_paired_edit_subset.py \
  --source-dir dataset/processed_strict_plus \
  --output-dir dataset/paired_edit_strict_plus
```

This writes:

```text
dataset/paired_edit_strict_plus/
  train_A/
  train_B/
  test_A/
  test_B/
  train_prompts.json
  test_prompts.json
  train_metadata.jsonl
  test_metadata.jsonl
```

`train_A/test_A` contain the `before` images, and `train_B/test_B` contain the paired `after` images.

The repository also includes a Colab entrypoint for this route:

```text
colab/paired_edit_config.yaml
colab/train_paired_edit_v1.ipynb
```

Use this notebook when you want to train a paired image-edit model instead of the target-only LoRA baseline.

## Local checkpoint inference

Install local inference dependencies:

```bash
python3 -m pip install -r configs/inference-requirements.txt
```

Use an SSD-backed Hugging Face cache to avoid storing large base models in the default home cache:

```bash
mkdir -p /Volumes/DevSSD/hf-cache
export HF_HOME=/Volumes/DevSSD/hf-cache
```

Your trained SD 1.5 LoRA checkpoints can live in:

```text
outputs/checkpoints/
  nail_retouch_sd15_v1-000004.safetensors
  nail_retouch_sd15_v1-000008.safetensors
  nail_retouch_sd15_v1.safetensors
```

Run img2img inference against every checkpoint in that folder:

```bash
HF_HOME=/Volumes/DevSSD/hf-cache python3 src/inference/run_img2img.py \
  --input path/to/test_hand.jpg \
  --checkpoint-dir outputs/checkpoints \
  --output-dir outputs/samples
```

Run a single checkpoint with custom prompt settings:

```bash
HF_HOME=/Volumes/DevSSD/hf-cache python3 src/inference/run_img2img.py \
  --input path/to/test_hand.jpg \
  --checkpoint outputs/checkpoints/nail_retouch_sd15_v1-000008.safetensors \
  --strength 0.28 \
  --guidance-scale 6.5 \
  --steps 30 \
  --prompt "professional manicure photo retouch, clean cuticles, clean sidewalls, refined nail shape, glossy nail surface, natural skin texture, realistic hand photo, square, nude" \
  --output-dir outputs/samples
```

Notes:

- The script uses `runwayml/stable-diffusion-v1-5` as the default base model.
- If `HF_HOME` is set, the SD 1.5 base model cache will be stored under that SSD path instead of `~/.cache/huggingface`.
- Input images are resized to a multiple of 8 for SD 1.5 compatibility.
- Each output image gets a matching `.json` file with the exact inference settings used.
- For your current checkpoints, start by comparing `000004`, `000008`, and the final checkpoint on the same source image.

Create a single comparison sheet for `before / outputs / after`:

```bash
python3 src/inference/make_comparison_sheet.py --pair-id pair_0006
```

This writes a combined image to:

```text
outputs/comparisons/pair_0006_comparison.png
```
