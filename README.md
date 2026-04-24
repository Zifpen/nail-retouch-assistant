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

Audit a paired dataset for whole-image lift, color drift, and oversized change masks before training:

```bash
python3 src/data/audit_paired_drift.py \
  --dataset-dir dataset/paired_edit_phase1_expand_batch1_pruned \
  --top-k 12
```

Use this check before expanding the training pool. High `luma_delta`, `channel_delta_std`, or `change_ratio` values usually mean the pair is teaching global relighting or color drift instead of local retouch cleanup.

## Masked Inpaint Dataset

Build a masked local-retouch dataset for inpainting supervision:

```bash
python3 src/data/build_masked_inpaint_dataset.py \
  --raw-dir raw \
  --manifest dataset/annotations/paired_edit_core_v1_manifest.json \
  --output-dir dataset/masked_inpaint_core_v1 \
  --mask-mode explicit
```

For early prototyping, the builder can bootstrap masks from the `before/after` difference:

```bash
python3 src/data/build_masked_inpaint_dataset.py \
  --raw-dir raw \
  --manifest dataset/annotations/paired_edit_core_v1_manifest.json \
  --output-dir dataset/masked_inpaint_core_v1_bootstrap \
  --mask-mode diff
```

The masked dataset writes:

```text
dataset/masked_inpaint_core_v1/
  train/
    images/
    masks/
    targets/
    metadata.jsonl
  val/
    images/
    masks/
    targets/
    metadata.jsonl
  build_summary.json
```

Notes:

- `targets/` are written as `target_local`: outside the mask they are copied from the original input, not the edited target.
- When color alignment is enabled, the builder first normalizes the target image toward the input statistics and then composites the aligned edit only inside the mask.
- `--mask-mode diff` is only a bootstrap path. Use explicit artist-approved masks for the real training set whenever possible.

Prepare the first explicit-mask annotation subset and scaffold its manifest:

```bash
python3 src/data/prepare_explicit_mask_subset.py
```

This writes:

```text
dataset/annotation_packs/masked_cuticle_cleanup_v1/
  README.md
  summary.json
  bootstrap_masks/
  bootstrap_overlays/
  pairs/
dataset/annotations/masked_cuticle_cleanup_v1_manifest.json
dataset/annotations/masks/masked_cuticle_cleanup_v1/
```

Use the generated `bootstrap_masks/` only as rough drafts. Final approved binary masks should be saved under `dataset/annotations/masks/masked_cuticle_cleanup_v1/`.

Current shape-refinement pilot handoff:

- merged manifest: `dataset/annotations/masked_shape_refinement_v4_approved_subset_manifest.json`
- built dataset: `dataset/masked_inpaint_shape_refinement_v4`
- Colab config: `colab/masked_inpaint_shape_refinement_v4_pilot.yaml`
- notebook entrypoint: `colab/train_masked_inpaint_full12_v1.ipynb`

## Masked Inpaint Training

Train a UNet LoRA on top of an inpainting checkpoint using the masked dataset:

```bash
python3 src/training/train_masked_inpaint_lora.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-inpainting \
  --dataset_dir dataset/masked_inpaint_core_v1 \
  --output_dir outputs/masked_inpaint_lora_core_v1 \
  --resolution 512 \
  --rank 4 \
  --learning_rate 1e-5 \
  --lambda_identity 5.0 \
  --lambda_color 0.5
```

This training route:

- reads the `train/val` masked dataset format
- trains UNet LoRA adapters only
- uses masked diffusion supervision plus explicit outside-mask identity loss
- writes LoRA checkpoints under `output_dir/lora_checkpoints`

For a cheap local plumbing smoke on CPU, use the dedicated wrapper:

```bash
bash scripts/run_masked_inpaint_local_smoke.sh
```

This preset intentionally trades fidelity for speed:

- dataset default: `dataset/masked_inpaint_cuticle_cleanup_v1_smoke`
- output default: `outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke`
- model default: local cached inpainting snapshot when present, otherwise `stable-diffusion-v1-5/stable-diffusion-inpainting`
- resolution: `256`
- steps: `4`
- previews: `1` validation sample at the final step only

When the wrapper finds a cached inpainting snapshot under the local Hugging Face cache, it switches to that snapshot and enables offline mode automatically. Use it only to verify that the masked training route still launches, steps, checkpoints, and writes previews after code changes. Do not treat it as a quality benchmark or as a replacement for the planned 10-step dry-run on faster hardware.

Run masked inpainting inference with exact outside-mask compositing:

```bash
python3 src/inference/run_masked_inpaint_inference.py \
  --input path/to/input.png \
  --mask path/to/mask.png \
  --lora-path outputs/masked_inpaint_lora_core_v1/lora_checkpoints \
  --preserve-unmasked-exact
```

Run masked validation and report edit quality vs preservation separately:

```bash
python3 src/inference/run_masked_inpaint_validation.py \
  --dataset-dir dataset/masked_inpaint_core_v1 \
  --lora-path outputs/masked_inpaint_lora_core_v1/lora_checkpoints \
  --preserve-unmasked-exact
```

The validation script writes per-pair sheets plus metrics for:

- masked L1 to target
- masked DeltaE to target
- unmasked L1 to input
- unmasked DeltaE to input
- border-ring L1 to target

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
