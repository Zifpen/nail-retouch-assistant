# Project State

Last updated: 2026-04-16

## Current Dataset Status

- `dataset/paired_edit_core_v1`: 29 train / 5 val. This is the current guarded training dataset referenced by [`colab/paired_edit_core_v1_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_core_v1_config.yaml).
- `dataset/paired_edit_core_v2`: 25 train / 5 val. This removes the four worst drift pairs from `core_v1` (`pair_0154`, `pair_0153`, `pair_0073`, `pair_0069`) while keeping prompt and split structure unchanged.
- `dataset/paired_edit_core_v2_cleanval`: 25 train / 4 val. This is the guarded `core_v2` retrain split that moves `pair_0050` out of the clean validation holdout.
- `dataset/paired_edit_core_v2_hardval`: 25 train / 1 val. This isolates `pair_0050` as a harder validation bucket instead of mixing it into the clean local-edit holdout.
- `dataset/paired_edit_core_v3`: 23 train / 5 val. This removes `pair_0022` and `pair_0066` from the default training pool as the first `core_v3` candidate split.
- `dataset/paired_edit_core_v3_cleanval`: 23 train / 4 val. This is the clean validation version of `core_v3`, keeping the same clean holdout policy as `core_v2_cleanval`.
- `dataset/paired_edit_core_v3_hardval`: 23 train / 1 val. This preserves `pair_0050` as the harder validation bucket for any future `core_v3` legacy comparison.
- `dataset/paired_edit_core_v3_secondary`: 2 train / 0 val. This is the temporary secondary bucket holding `pair_0022` and `pair_0066` outside the clean baseline.
- `dataset/paired_edit_phase1_expand_batch1`: 41 train / 5 val. This is the older expansion batch that produced visible collapse artifacts.
- `dataset/paired_edit_phase1_expand_batch1_pruned`: 36 train / 5 val. This drops the most obvious whole-image shift samples from the batch1 expansion.
- `dataset/paired_edit_strict_plus`: 29 train / 5 val. Older prompt-variant dataset with shape/finish/color tags.
- `dataset/annotation_packs/masked_cuticle_cleanup_v1`: explicit-mask annotation scaffold for the first masked inpainting subset. Despite the folder name, the subset is now understood as `proximal_nail_boundary_refinement`: local cleanup plus small posterior-edge improvement bands.
- Training pool plan currently separates the raw pairs into `phase1_core_train` (29), `phase1_expand_train` (35), `phase2_secondary_train` (73), `hard_val_optional` (3), and `exclude_for_now`.

## Current Training Setup

Latest guarded route in [`colab/paired_edit_core_v1_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_core_v1_config.yaml):

- Dataset: `dataset/paired_edit_core_v1`
- Base model: `stabilityai/sd-turbo`
- Training script: [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py)
- Resolution: `256`
- Prompt: `professional manicure photo retouch, clean cuticles, clean sidewalls, refined nail shape, preserve original nail design, natural skin texture, realistic hand photo`
- LoRA config: `lora_rank_unet=8`, `lora_rank_vae=4`, no explicit alpha flag exposed in the current script
- Batch size: `1`
- Learning rate: `5e-6`
- Epochs / max steps: `120 / 1500`
- Loss weights: `lambda_full_l1=1.0`, `lambda_preserve=2.0`, `lambda_edit=2.0`, `lambda_lpips=0.5`, `lambda_l2=0.0`
- Change-mask settings: `threshold=0.12`, `dilate=3`
- Checkpoint seen in workspace: `outputs/checkpoints/model_1401.pkl`

Historical failing route in [`colab/paired_edit_phase1_expand_batch1_oldroute_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_phase1_expand_batch1_oldroute_config.yaml):

- Dataset: `dataset/paired_edit_phase1_expand_batch1`
- Train script: upstream `src/train_pix2pix_turbo.py`
- Resolution: `256`
- Batch size: `1`
- Max steps: `5000`
- Checkpointing: every `250`
- Visible outputs captured in workspace: `model_251` and `model_501`

## Observed Output Problems

- Historical inference sheet for `pair_0009` at `model_251` shows near-total white collapse.
- Historical inference sheet for `pair_0009` at `model_501` shows strong pink/orange-magenta color cast, severe blur, and texture loss.
- Manual pruning notes in [`dataset/annotations/phase1_expand_batch1_pruned.csv`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/phase1_expand_batch1_pruned.csv) repeatedly cite `whole-image lift`, `warmth drift`, and `global brightening`.
- Even the guarded `core_v1` dataset still contains several high-drift pairs, so bias is not limited to the expansion batch.

## Source Diagnosis

Primary issue: dataset bias.

- Many before/after pairs contain global brightness and blue-channel lift, so the model is rewarded for whole-image tonal change instead of localized retouch.
- The current supervised loss stack then amplifies that bias because full-image reconstruction losses make global relighting cheap to learn.
- Inference is a secondary contributor. The old-route artifacts are already visible under deterministic prompt usage, so prompt wording is not the main root cause.

## Experiment Log

### Experiment 2026-03-30A - Paired Drift Audit Baseline

Hypothesis:
Adding a repeatable paired-image drift audit will identify the same risky samples that manual review was trying to remove, and it will reveal whether the current core dataset is already biased toward whole-image lift.

Change made:

- Added [`src/data/audit_paired_drift.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/audit_paired_drift.py)
- Added README usage for the audit step in [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md)
- Ran the audit on `paired_edit_core_v1`, `paired_edit_phase1_expand_batch1`, `paired_edit_phase1_expand_batch1_pruned`, and `paired_edit_strict_plus`

Result:

- `paired_edit_phase1_expand_batch1` mean drift score: `0.2163`
- `paired_edit_phase1_expand_batch1_pruned` mean drift score: `0.1896`
- `paired_edit_core_v1` mean drift score: `0.2121`
- `paired_edit_strict_plus` mean drift score: `0.1863`
- `core_v1` still has a positive mean luma delta of `+0.0419` and mean change ratio of `0.2884`
- Repeated high-drift pairs include `pair_0154`, `pair_0153`, `pair_0073`, `pair_0069`, `pair_0022`, and `pair_0066`
- Pruning the batch1 expansion helped, but it did not remove the deeper bias already present in the core teaching set

Conclusion:

- The core problem is confirmed to be data-driven first, not just training-step count or inference prompting.
- Dataset gating needs to happen before the next retrain.
- The next clean experiment should change only the training dataset while keeping the guarded training config fixed.

### Experiment 2026-03-30B - Masked Inpaint Dataset Builder Scaffold

Hypothesis:
If the repository can export `input + mask + target_local` with pairwise color alignment on the unmasked region, we can remove full-image supervision leakage before touching the training loop.

Change made:

- Added [`src/data/build_masked_inpaint_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_masked_inpaint_dataset.py)
- Added masked inpaint dataset documentation in [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md)
- Supported two mask modes:
  - `explicit`: production path using provided masks
  - `diff`: bootstrap path using aligned before/after differences
- Added `target_local` export, per-pair task labels, and color-alignment statistics in `metadata.jsonl`
- Ran a smoke test on `pair_0005`, `pair_0015`, `pair_0009`, and `pair_0040`

Result:

- The new builder produced the intended dataset structure under `/tmp/masked_inpaint_smoke`
- On the smoke test, pairwise color alignment reduced raw validation luma drift substantially:
  - `pair_0009`: `raw_luma_delta +0.0387 -> final_luma_delta +0.0022`
  - `pair_0040`: `raw_luma_delta +0.0514 -> final_luma_delta -0.0021`
- Bootstrap diff masks were acceptable on the cleaner validation pairs, around `0.093 - 0.095` mask ratio
- Bootstrap diff masks were too broad on at least one training sample:
  - `pair_0005`: `mask_ratio 0.4120`

Conclusion:

- The masked local-target route is practical in the current repository and can now be built deterministically.
- Pairwise color alignment is effective enough to keep as a permanent preprocessing stage.
- Diff masks are useful only for prototyping and smoke tests; explicit masks are still required for the real training dataset.

### Experiment 2026-03-30C - Masked Inpaint Training Entrypoint Scaffold

Hypothesis:
We can add a clean inpainting-specific LoRA training entrypoint that consumes the new masked dataset format and enforces outside-mask preservation without mutating the legacy full-image training route.

Change made:

- Added [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py)
- The new script:
  - reads `train/val` masked dataset metadata
  - loads a Stable Diffusion inpainting checkpoint
  - adds LoRA adapters to the UNet only
  - optimizes masked diffusion loss, masked RGB reconstruction, outside-mask identity loss, and masked color consistency loss
  - saves preview grids and LoRA checkpoints
- Added a minimal launch example in [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md)

Result:

- `py_compile` passed for the new training script
- `--help` executed successfully, confirming the CLI entrypoint and argument surface
- No full training run was launched in this session

Conclusion:

- The repository now has a dedicated masked inpainting training path separate from the legacy paired-edit script.
- The next blocker is not code structure anymore; it is obtaining the first explicit-mask training subset and running a real dry-run against the inpainting base checkpoint.

### Experiment 2026-03-31A - Masked Inpaint Inference And Validation Closure

Hypothesis:
If the masked route has its own inference and validation entrypoints with exact outside-mask compositing and split metrics, we can measure whether the model is preserving unmasked pixels instead of relying on the old full-image pix2pix validation loop.

Change made:

- Added [`src/inference/masked_inpaint_utils.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/masked_inpaint_utils.py)
- Added [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py)
- Added [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py)
- Updated [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md) with new inference and validation commands

Result:

- `py_compile` passed for all new scripts
- `--help` executed successfully for the new inference and validation CLIs
- The masked validation route now reports:
  - `masked_l1_to_target`
  - `masked_delta_e_to_target`
  - `unmasked_l1_to_input`
  - `unmasked_delta_e_to_input`
  - `border_l1_to_target`
- No real inference run was launched in this session because the manual explicit-mask step still blocks the first meaningful masked dataset build

Conclusion:

- The masked route is now structurally end-to-end: data builder, trainer, inference entrypoint, and validation entrypoint all exist.
- The next missing variable is no longer code plumbing; it is real explicit mask supervision.

### Experiment 2026-03-31B - Explicit Mask Annotation Pack Scaffold

Hypothesis:
If we package a small seed subset with pair sheets, bootstrap draft masks, and a ready-to-fill manifest, we can reduce manual labeling overhead and keep the first masked experiment narrow enough to interpret.

Change made:

- Added [`src/data/prepare_explicit_mask_subset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/prepare_explicit_mask_subset.py)
- Generated [`dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md)
- Generated [`dataset/annotation_packs/masked_cuticle_cleanup_v1/summary.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/summary.json)
- Generated [`dataset/annotations/masked_cuticle_cleanup_v1_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_manifest.json)
- Created the expected final mask root `dataset/annotations/masks/masked_cuticle_cleanup_v1/`

Result:

- The first seed subset is now fixed at:
  - train: `pair_0005`, `pair_0015`, `pair_0018`, `pair_0032`, `pair_0054`, `pair_0057`, `pair_0063`, `pair_0070`
  - val: `pair_0009`, `pair_0040`, `pair_0047`, `pair_0050`
- Bootstrap draft masks surfaced a clear difficulty split:
  - safer drafts: `pair_0015` (`0.0728`), `pair_0018` (`0.0517`), `pair_0009` (`0.0929`), `pair_0040` (`0.0946`)
  - broad drafts that will need manual redraw or careful shrinking: `pair_0005` (`0.4016`), `pair_0047` (`0.3933`), `pair_0063` (`0.5152`), `pair_0070` (`0.4224`), `pair_0050` (`0.6095`)
- Pairwise global alignment again reduced luma drift to near zero before the diff-mask bootstrap step, so the remaining issue is mask locality rather than color alignment

Conclusion:

- Manual work is now tightly scoped: refine or redraw explicit masks into the generated target directory.
- Bootstrap masks are useful as draft guidance, but not trustworthy enough to become final supervision on the broader-mask samples.

### Experiment 2026-03-31C - Core V2 Dataset-Only Legacy Baseline

Hypothesis:
If we remove only the four worst-drift train pairs from `core_v1` and keep everything else unchanged, the paired-edit dataset should become materially safer without confounding the next legacy retrain.

Change made:

- Added [`dataset/annotations/paired_edit_core_v2_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v2_manifest.json)
- Built [`dataset/paired_edit_core_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v2) with [`src/data/build_curated_paired_edit_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_curated_paired_edit_dataset.py)
- Re-ran [`src/data/audit_paired_drift.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/audit_paired_drift.py) on the new dataset

Result:

- `core_v2` size: `25 train / 5 val`
- Mean drift score improved from `0.2121` to `0.1377`
- Mean luma delta improved from `+0.0419` to `+0.0311`
- Mean change ratio improved from `0.2884` to `0.2178`
- Max change ratio improved from `0.9667` to `0.4519`
- New top drift pairs are now `pair_0022`, `pair_0066`, `pair_0035`, `pair_0050`, `pair_0118`, and `pair_0120`

Conclusion:

- The dataset-only hypothesis is strengthened again: removing the top four outliers materially lowers drift without touching training config.
- If we run another legacy paired-edit retrain, `core_v2` should replace `core_v1` as the next baseline.

### Experiment 2026-04-01A - Task Redefinition For The First Explicit-Mask Subset

Hypothesis:
If we relabel the first masked subset to match the real supervision signal, mask authoring and later training analysis will become more consistent and less misleading.

Change made:

- Updated [`dataset/annotations/masked_cuticle_cleanup_v1_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_manifest.json) so the task label is `proximal_nail_boundary_refinement`
- Updated [`dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md) to explicitly allow a narrow posterior-edge movement band
- Updated [`src/data/prepare_explicit_mask_subset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/prepare_explicit_mask_subset.py) so future regenerated packs use the same task definition

Result:

- The annotation instructions now match the actual dataset behavior: many valid samples involve both local dead-skin cleanup and local posterior-edge refinement.
- The first reviewed explicit masks (`pair_0015`, `pair_0018`) can now be interpreted correctly instead of being judged against an unrealistically strict cleanup-only rule.

Conclusion:

- The first masked experiment should be analyzed as a local boundary-refinement model, not as a pure cuticle cleaner.
- This does not justify broad masks; it only legitimizes a narrow edge-transition band when the `after` image clearly refines the posterior edge.

### Experiment 2026-04-01B - Four-Mask Explicit Smoke Subset Build

Hypothesis:
If we limit the first explicit dataset build to the four already-approved masks, we can validate the explicit masked data path before waiting on the rest of the annotation pack.

Change made:

- Added [`dataset/annotations/masked_cuticle_cleanup_v1_smoke_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_smoke_manifest.json)
- Built [`dataset/masked_inpaint_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1_smoke) with explicit masks for:
  - train: `pair_0015`, `pair_0018`
  - val: `pair_0009`, `pair_0040`

Result:

- The explicit masked dataset build succeeded without any missing-file or mask-shape errors
- Train summary:
  - mean mask ratio: `0.0391`
  - mean final luma delta after alignment: `-0.00007`
- Val summary:
  - mean mask ratio: `0.0292`
  - mean final luma delta after alignment: `+0.00030`
- Approved explicit masks now in place:
  - `pair_0015`
  - `pair_0018`
  - `pair_0009`
  - `pair_0040`

Conclusion:

- The explicit-mask data route is now verified on real human-reviewed masks.
- Current mask coverage is local and conservative enough for a first masked smoke run.

### Experiment 2026-04-01C - Masked Training Dry-Run Attempt On Explicit Smoke Subset

Hypothesis:
A 10-step masked LoRA dry-run on the four-mask subset should verify end-to-end training, checkpoint writing, and preview generation.

Change made:

- Launched [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py) against [`dataset/masked_inpaint_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1_smoke)
- Output directory: [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke)

Result:

- The run successfully passed:
  - explicit dataset loading
  - Hugging Face model download / model init
  - optimizer / scheduler setup
  - actual training-step execution
- At least two training steps completed and wrote metrics to [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke/metrics.jsonl):
  - step 1: `loss=2.5480`, `loss_mask_rgb=0.0799`, `loss_identity=0.0195`, `loss_color=4.0780`
  - step 2: `loss=6.4124`, `loss_mask_rgb=0.2519`, `loss_identity=0.0088`, `loss_color=12.0220`
- `training_config.json` was written successfully
- No checkpoint or preview was produced yet because CPU runtime at `512` resolution was too slow to reasonably wait for all 10 steps in this session

Conclusion:

- The masked training route is not blocked by data formatting or immediate runtime errors anymore.
- The main remaining limitation is runtime speed on the current local CPU path, not pipeline correctness.
- Future smoke runs should prefer a smaller local test setting or remote / faster hardware when possible.

### Experiment 2026-04-02A - Local Masked Smoke Preset

Hypothesis:
If we add a cheap, explicit local smoke wrapper for the masked route, we can verify training plumbing after code changes without waiting on a slow 512-resolution CPU dry-run.

Change made:

- Added [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh)
- Updated [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md) with a dedicated local smoke command and scope notes
- Fixed the preset defaults to:
  - dataset: `dataset/masked_inpaint_cuticle_cleanup_v1_smoke`
  - output: `outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke`
  - resolution: `256`
  - max steps: `4`
  - final-step checkpoint and single-sample preview only

Result:

- The repository now has a single-command local smoke entrypoint for the masked route
- The preset keeps model, losses, and dataset type unchanged while reducing only runtime cost
- The README now explicitly warns that this preset is for plumbing verification, not for judging retouch quality or replacing the planned 10-step run on faster hardware

Conclusion:

- A low-cost local smoke preset is a useful permanent support tool for the masked migration
- This reduces iteration cost without introducing a new training-route variable, as long as experiment notes continue to distinguish smoke runs from real quality evaluations

### Experiment 2026-04-02B - Local Masked Smoke Wrapper End-To-End Run

Hypothesis:
If the new local masked smoke wrapper can complete a full 4-step run and write its expected artifacts, we can treat it as a reliable plumbing check for future masked-route code changes.

Change made:

- Checked the current explicit mask directory and confirmed there were still no new masks beyond:
  - `pair_0015`
  - `pair_0018`
  - `pair_0009`
  - `pair_0040`
- Reused the existing four-mask smoke dataset [`dataset/masked_inpaint_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1_smoke)
- Ran [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh)
- After the default model identifier hit a local network-resolution failure against `huggingface.co`, reran the same wrapper against the cached inpainting snapshot with offline environment flags

Result:

- The default wrapper invocation did not complete because the environment could not resolve `huggingface.co`
- The offline rerun completed all `4/4` steps successfully
- Training-segment wall-clock was about `49s`
- The run wrote the expected artifacts under [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke):
  - [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke/metrics.jsonl)
  - [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke/training_config.json)
  - [`pytorch_lora_weights_step_000004.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke/lora_checkpoints/pytorch_lora_weights_step_000004.safetensors)
  - [`preview_step_000004.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke/previews/preview_step_000004.png)
- Recorded losses:
  - step 1: `loss=2.6325`, `loss_mask_rgb=0.0670`, `loss_identity=0.0233`, `loss_color=3.7570`
  - step 2: `loss=7.0977`, `loss_mask_rgb=0.2743`, `loss_identity=0.0179`, `loss_color=12.9865`
  - step 3: `loss=7.7632`, `loss_mask_rgb=0.2945`, `loss_identity=0.0320`, `loss_color=14.3488`
  - step 4: `loss=8.5466`, `loss_mask_rgb=0.3208`, `loss_identity=0.0387`, `loss_color=16.0197`
- The preview image is non-empty and confirms the preview-writing path worked; this run still does not qualify as a quality benchmark

Conclusion:

- The local smoke wrapper is now verified end-to-end as a plumbing check in this repository
- The main blocker surfaced by this run is environment-level model availability, not masked data formatting or trainer logic
- This run confirms startup, stepping, checkpoint writing, and preview generation at the reduced-cost preset
- This run does not provide evidence about final edit quality, color stability, or whether the current task split needs to change

### Experiment 2026-04-02C - Auto Offline Cache Fallback For Local Smoke Wrapper

Hypothesis:
If the local smoke wrapper auto-detects a cached inpainting snapshot and switches to offline mode when that cache is present, the default smoke command should become reliable in network-restricted local sessions without changing training behavior.

Change made:

- Updated [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh) to:
  - keep `stable-diffusion-v1-5/stable-diffusion-inpainting` as the logical default
  - auto-detect a cached snapshot under the local Hugging Face cache when available
  - set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` automatically when the cached snapshot is used
- Updated [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md) to document the cached-snapshot fallback behavior
- Re-ran the default wrapper command with only `OUTPUT_DIR` overridden to a temporary verification directory:
  - `/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify`

Result:

- The patched default wrapper invocation completed successfully without manually overriding `PRETRAINED_MODEL`
- The verification run completed all `4/4` steps and wrote:
  - [`metrics.jsonl`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify/metrics.jsonl)
  - [`training_config.json`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify/training_config.json)
  - [`pytorch_lora_weights_step_000004.safetensors`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify/lora_checkpoints/pytorch_lora_weights_step_000004.safetensors)
  - [`preview_step_000004.png`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify/previews/preview_step_000004.png)
- Step-level losses matched the earlier successful offline rerun:
  - step 1: `loss=2.6325`, `loss_mask_rgb=0.0670`, `loss_identity=0.0233`, `loss_color=3.7570`
  - step 2: `loss=7.0977`, `loss_mask_rgb=0.2743`, `loss_identity=0.0179`, `loss_color=12.9865`
  - step 3: `loss=7.7632`, `loss_mask_rgb=0.2945`, `loss_identity=0.0320`, `loss_color=14.3488`
  - step 4: `loss=8.5466`, `loss_mask_rgb=0.3208`, `loss_identity=0.0387`, `loss_color=16.0197`

Conclusion:

- The local smoke wrapper now behaves like a true single-command plumbing check in this environment
- The fix is infrastructural only; it does not change the masked task definition, dataset, or training-loss interpretation
- Future local smoke failures after this patch are more likely to indicate an actual pipeline regression than a simple remote model-resolution issue

### Experiment 2026-04-02D - QA Review Of Four Newly Drawn Explicit Masks

Hypothesis:
If the newly drawn masks for `pair_0005`, `pair_0032`, `pair_0054`, and `pair_0057` stay local to the proximal boundary cleanup / refinement band, we should be able to promote at least part of the next masked training subset without waiting on the rest of the annotation pack.

Change made:

- Reviewed the newly added masks under [`dataset/annotations/masks/masked_cuticle_cleanup_v1/`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/)
- Checked:
  - image size match
  - near-binary values
  - mask ratio
  - rough locality against the pair sheets in [`dataset/annotation_packs/masked_cuticle_cleanup_v1/pairs/`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/pairs/)
- Reviewed the masks against the current `proximal_nail_boundary_refinement` definition

Result:

- New mask files confirmed:
  - `pair_0005`
  - `pair_0032`
  - `pair_0054`
  - `pair_0057`
- All four masks are near-binary with values `[0, 254, 255]`
- Measured mask ratios:
  - `pair_0005`: `0.0612`
  - `pair_0032`: `0.1101`
  - `pair_0054`: `0.1067`
  - `pair_0057`: `0.0807`
- QA decisions:
  - `pair_0032`: pass
  - `pair_0054`: pass
  - `pair_0005`: needs minor touch-up
  - `pair_0057`: needs minor touch-up
- Main issues on the two touch-up samples:
  - detached tiny mask islands
  - a few bands that widen slightly too far into surrounding skin
- No sample in this batch needs a full redraw

Conclusion:

- The new annotation batch is directionally correct and much more local than the earlier bootstrap drafts
- Two samples (`pair_0032`, `pair_0054`) are ready for dataset promotion immediately
- Two samples (`pair_0005`, `pair_0057`) only need small cleanup edits, not re-annotation
- The masked route is no longer blocked on this entire four-sample tranche; it is now blocked only on two small mask refinements plus any additional samples we choose to label next

### Experiment 2026-04-02E - Re-Review Of Fixed `pair_0005` And `pair_0057`

Hypothesis:
If the touch-up pass removes the detached islands and tightens the over-wide bands on `pair_0005` and `pair_0057`, both masks should become promotable without another redraw cycle.

Change made:

- Re-reviewed the updated masks:
  - [`pair_0005.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0005.png)
  - [`pair_0057.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0057.png)
- Recomputed mask statistics after the edits
- Compared the revised masks against the prior QA findings and pair sheets

Result:

- Updated mask ratios:
  - `pair_0005`: `0.0522` (down from `0.0612`)
  - `pair_0057`: `0.0658` (down from `0.0807`)
- Both masks now pass semantic QA for `proximal_nail_boundary_refinement`
- The prior detached-island and overly wide band issues are no longer blockers
- The revised files are anti-aliased grayscale masks rather than near-binary masks
- This is not a QA failure, but the masks must be normalized to strict binary during dataset promotion

Conclusion:

- `pair_0005` and `pair_0057` are now approved
- The newly approved tranche beyond the original four-mask smoke subset is now:
  - `pair_0005`
  - `pair_0032`
  - `pair_0054`
  - `pair_0057`
- The next masked-route blocker is no longer manual touch-up on this batch; it is rebuilding the masked dataset with deterministic binary normalization for the authored masks

### Experiment 2026-04-02F - Approved-Subset Masked Dataset Rebuild

Hypothesis:
If we rebuild the masked dataset using only the currently approved explicit masks, we should get the first real `proximal_nail_boundary_refinement` dataset without waiting for the four still-missing masks in the original 12-sample manifest.

Change made:

- First attempted to build from [`dataset/annotations/masked_cuticle_cleanup_v1_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_manifest.json)
- Confirmed that the full manifest is still blocked by missing mask files for:
  - `pair_0063`
  - `pair_0070`
  - `pair_0047`
  - `pair_0050`
- Added [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json) containing only the currently approved 8 samples
- Rebuilt [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1) from the approved manifest

Result:

- The approved-subset build completed successfully with:
  - `train_count = 6`
  - `val_count = 2`
- The exported dataset masks are strict binary values `[0, 255]` even for the antialiased author inputs
- Build summary:
  - train mean mask ratio: `0.0692`
  - train mean raw luma delta: `-0.0026`
  - train mean final luma delta: `+0.0017`
  - val mean mask ratio: `0.0292`
  - val mean raw luma delta: `+0.0450`
  - val mean final luma delta: `+0.0003`
- Pair-level highlights:
  - `pair_0005`: `mask_ratio 0.0524`, `final_luma_delta +0.0024`
  - `pair_0032`: `mask_ratio 0.1101`, `final_luma_delta +0.0057`
  - `pair_0054`: `mask_ratio 0.1067`, `final_luma_delta +0.0024`
  - `pair_0057`: `mask_ratio 0.0676`, `final_luma_delta +0.0001`

Conclusion:

- The repository now has its first real approved explicit masked dataset beyond the original four-mask smoke subset
- Build-time mask normalization is already sufficient to turn anti-aliased author masks into strict binary training masks
- The current data-layer blocker is no longer formatting; it is only the missing masks for the remaining 4 samples in the original 12-sample seed pack

### Experiment 2026-04-02G - 8-Sample Approved Dataset Local 4-Step Smoke

Hypothesis:
If we swap only the dataset input from the old 4-mask smoke subset to the new 8-sample approved masked dataset, the low-cost local smoke run should still launch, step, checkpoint, and preview successfully.

Change made:

- Reused [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh)
- Overrode only:
  - `DATASET_DIR=/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1`
  - `OUTPUT_DIR=/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_local_smoke`

Result:

- The 4-step local smoke completed successfully on the approved 8-sample dataset
- It wrote:
  - [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_local_smoke/metrics.jsonl)
  - [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_local_smoke/training_config.json)
  - [`pytorch_lora_weights_step_000004.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_local_smoke/lora_checkpoints/pytorch_lora_weights_step_000004.safetensors)
  - [`preview_step_000004.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_local_smoke/previews/preview_step_000004.png)
- Metrics showed larger per-step variation than the earlier 4-mask smoke, but no dataset-format failures:
  - step 1 loss: `2.4205`
  - step 2 loss: `5.9036`
  - step 3 loss: `12.9324`
  - step 4 loss: `3.6542`

Conclusion:

- The approved 8-sample dataset is structurally valid as the next minimal training input
- Enlarging the explicit dataset introduces harder batches, but not a training-plumbing regression
- This run remains plumbing evidence only, not model-quality evidence

### Experiment 2026-04-02H - 8-Sample Approved Dataset Local 10-Step Dry-Run

Hypothesis:
If the 8-sample approved dataset is stable beyond a trivial smoke, a low-cost 10-step dry-run at `256` resolution should complete locally and give us the first longer runtime signal on the new explicit subset.

Change made:

- Reused the same approved dataset:
  - [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1)
- Reused the same low-cost smoke route, changing only run length:
  - `max_train_steps=10`
  - `checkpointing_steps=10`
  - `preview_steps=10`
- Output directory:
  - [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local)

Result:

- The local 10-step dry-run completed successfully in about `2m12s`
- It wrote:
  - [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local/metrics.jsonl)
  - [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local/training_config.json)
  - [`pytorch_lora_weights_step_000010.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local/lora_checkpoints/pytorch_lora_weights_step_000010.safetensors)
  - [`preview_step_000010.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_approved_step10_local/previews/preview_step_000010.png)
- The first 4 steps matched the earlier approved-dataset smoke exactly, then continued through step 10 without loader or mask failures
- Later-step losses were noisy but finite:
  - step 5: `9.1572`
  - step 6: `7.1828`
  - step 7: `5.5109`
  - step 8: `2.4367`
  - step 9: `2.1243`
  - step 10: `3.2730`

Conclusion:

- The 8-sample approved dataset is not just buildable; it is stable enough for a complete local 10-step dry-run
- The masked route remains limited by CPU runtime and sample count, not by immediate data-format or trainer failures on this subset
- The next meaningful masked-route upgrade is no longer “can this dataset run at all”; it is either expanding approved masks further or using faster hardware for a more informative run

Hypothesis:
If the revised versions of `pair_0005` and `pair_0057` remove the detached islands and tighten the overly broad bands, both masks should become promotable into the next masked dataset build.

Change made:

- Re-reviewed the revised masks:
  - [`pair_0005.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0005.png)
  - [`pair_0057.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0057.png)
- Re-checked:
  - mask ratios
  - connectivity / obvious stray islands
  - locality against the pair sheets
  - whether the remaining issue is semantic or only export-format related

Result:

- `pair_0005` mask ratio improved from `0.0612` to `0.0522`
- `pair_0057` mask ratio improved from `0.0807` to `0.0658`
- Both masks are now semantically local enough to fit `proximal_nail_boundary_refinement`
- The earlier visible detached-island problem is no longer a meaningful QA blocker on either sample
- `pair_0005` still has one tiny residual 1-pixel component after thresholding, but this is build-time cleanup territory, not redraw territory
- New export detail:
  - both revised masks are antialiased grayscale masks rather than near-binary masks
  - this is not a semantic QA failure, but they should be normalized to strict binary values before or during dataset build

Conclusion:

- `pair_0005` now passes
- `pair_0057` now passes
- The previously reviewed four-sample tranche is now fully approved:
  - `pair_0005`
  - `pair_0032`
  - `pair_0054`
  - `pair_0057`
- The next masked-route bottleneck has shifted from mask QA to dataset promotion and rebuild

## 2026-04-03A - QA And Promotion Of `pair_0063` And `pair_0070`

Hypothesis:
If the newly authored `pair_0063` and `pair_0070` masks are now local enough to fit `proximal_nail_boundary_refinement`, both can be promoted into the approved explicit subset and the masked dataset can grow without changing any training variable.

Change made:

- Re-read project memory before the review to keep the current masked-route priorities and task definition fixed.
- Re-reviewed:
  - [`pair_0063.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0063.png)
  - [`pair_0070.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0070.png)
- Checked:
  - locality against the corresponding pair sheets
  - binary / near-binary export state
  - mask ratio and bounding-box spread
  - whether either sample drifted into whole-finger or broad skin retouch territory
- Promoted both passing samples into [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json).
- Rebuilt [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1) from the updated approved manifest.

Result:

- `pair_0063` now passes semantic QA
  - mask ratio: `0.0497` at authored-mask inspection
- `pair_0070` now passes semantic QA
  - mask ratio: `0.1097` at authored-mask inspection
- The approved manifest now contains 10 samples total:
  - train:
    - `pair_0005`
    - `pair_0015`
    - `pair_0018`
    - `pair_0032`
    - `pair_0054`
    - `pair_0057`
    - `pair_0063`
    - `pair_0070`
  - val:
    - `pair_0009`
    - `pair_0040`
- The rebuilt approved dataset completed successfully:
  - train count: `8`
  - val count: `2`
  - train mean mask ratio: `0.0720`
  - train mean final luma delta: `+0.0009`
  - val mean final luma delta: `+0.0003`
- Per-sample rebuilt stats stayed acceptable:
  - `pair_0063`: final luma delta `-0.0006`
  - `pair_0070`: final luma delta `-0.0028`

Conclusion:

- `pair_0063` and `pair_0070` are both approved for the current `proximal_nail_boundary_refinement` subset.
- The approved explicit masked dataset has now grown from the earlier 8-sample total to a 10-sample total (`8` train / `2` val) without introducing obvious data-layer regressions.
- The masked-route expansion is no longer blocked by `pair_0063` or `pair_0070`.
- The only seed-pack expansion candidates still not promoted are the intentionally deferred higher-risk samples:
  - `pair_0047`
  - `pair_0050`

## 2026-04-03B - 10-Sample Approved Dataset Local 10-Step Dry-Run

Hypothesis:
If the newly expanded 10-sample approved explicit subset is already stable enough to serve as the first real masked training set, a low-cost local 10-step dry-run should still complete cleanly and look like a harder-but-healthy continuation of the earlier 8-sample run rather than a new trainer or data-format regression.

Change made:

- Re-read project memory before launching the run so the experiment stayed single-variable.
- Reused the same approved explicit dataset path:
  - [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1)
- Reused the same low-cost local masked training recipe as the earlier 8-sample dry-run:
  - resolution `256`
  - batch size `1`
  - rank `4`
  - learning rate `1e-5`
  - `max_train_steps=10`
  - `checkpointing_steps=10`
  - `preview_steps=10`
- Wrote this run to a new output directory:
  - [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local)

Result:

- The local 10-step dry-run completed successfully in about `2m23s`
- It wrote:
  - [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local/metrics.jsonl)
  - [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local/training_config.json)
  - [`pytorch_lora_weights_step_000010.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local/lora_checkpoints/pytorch_lora_weights_step_000010.safetensors)
  - [`preview_step_000010.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local/previews/preview_step_000010.png)
- The run stayed finite across all 10 steps:
  - step 1 loss: `4.8047`
  - step 2 loss: `7.0540`
  - step 3 loss: `12.9196`
  - step 4 loss: `3.1515`
  - step 5 loss: `8.2478`
  - step 6 loss: `6.8296`
  - step 7 loss: `3.6081`
  - step 8 loss: `9.2267`
  - step 9 loss: `2.1221`
  - step 10 loss: `5.1951`
- Aggregate training signals versus the earlier 8-sample step-10 run:
  - 10-sample mean loss: `6.3159`
  - earlier 8-sample mean loss: `5.4596`
  - 10-sample mean identity loss: `0.0248`
  - earlier 8-sample mean identity loss: `0.0231`
- Preview inspection on the sampled validation example did not reveal a new catastrophic color or structure failure relative to the prior dry-run; the run mainly looks noisier because the subset is broader.

Conclusion:

- The expanded 10-sample approved explicit subset is stable enough to act as the first real masked training set in this repository.
- Adding `pair_0063` and `pair_0070` increased batch difficulty slightly, but did not introduce a new data-format, loader, checkpoint, or preview regression.
- The next masked-route bottleneck is no longer subset validity; it is either:
  - moving this same clean subset onto faster hardware for a more informative run
  - or manually authoring the optional harder cases `pair_0047` / `pair_0050`

## 2026-04-03C - QA Review Of `pair_0047` And `pair_0050`

Hypothesis:
If the newly authored `pair_0047` and `pair_0050` masks stay narrow around the proximal boundary / cuticle band and avoid absorbing decorations or broad lighting-driven appearance changes, both can pass semantic QA even though they were previously treated as the highest-risk remaining seed-pack samples.

Change made:

- Re-read project memory before review so the current task definition stayed fixed.
- Delegated the first-pass audit to the active Mask QA subagent with explicit scope and non-goals.
- Collected local sanity stats on the new masks:
  - `pair_0047`: mask ratio `0.0467`
  - `pair_0050`: mask ratio `0.0590`
- Performed a final main-agent visual spot check against:
  - [`pair_0047_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/pairs/pair_0047/pair_0047_sheet.png)
  - [`pair_0050_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/pairs/pair_0050/pair_0050_sheet.png)
  - [`pair_0047.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0047.png)
  - [`pair_0050.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v1/pair_0050.png)

Result:

- `pair_0047` passes semantic QA
  - the mask stays on the local proximal-boundary / sidewall transition bands
  - it does not absorb the gem / decoration as the edit subject
  - export format is already strict binary
- `pair_0050` passes semantic QA
  - the mask remains local to the proximal boundary / cuticle bands across the nails
  - it does not expand into whole-finger smoothing or whole-nail repaint
  - export format is already strict binary
- Neither sample requires redraw or micro-fix before promotion review

Conclusion:

- `pair_0047` now passes
- `pair_0050` now passes
- The original 12-sample seed pack is now fully labeled and semantically approved for `proximal_nail_boundary_refinement`
- The masked route is no longer blocked by outstanding manual mask work on this first seed pack
- The next step is no longer QA; it is dataset promotion:
  - update the approved manifest
  - rebuild the approved masked dataset with the full 12-sample subset
  - then decide whether the higher-risk val additions should remain in the default approved set long-term

## 2026-04-03D - Promote Full 12-Sample Approved Manifest And Rebuild Dataset

Hypothesis:
If `pair_0047` and `pair_0050` are promoted into the approved manifest while keeping the original seed-pack split intact, the full 12-sample masked dataset should still build cleanly and keep unmasked-region color alignment under control.

Change made:

- Reused the existing approved manifest as the promotion target:
  - [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json)
- Added:
  - `pair_0047` to `val_ids`
  - `pair_0050` to `val_ids`
- Rebuilt:
  - [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1)

Result:

- The approved manifest now matches the full first seed pack:
  - train:
    - `pair_0005`
    - `pair_0015`
    - `pair_0018`
    - `pair_0032`
    - `pair_0054`
    - `pair_0057`
    - `pair_0063`
    - `pair_0070`
  - val:
    - `pair_0009`
    - `pair_0040`
    - `pair_0047`
    - `pair_0050`
- The rebuilt dataset completed successfully:
  - train count: `8`
  - val count: `4`
  - train mean mask ratio: `0.0720`
  - train mean final luma delta: `+0.0009`
  - val mean mask ratio: `0.0421`
  - val mean final luma delta: `+0.0016`
- New harder validation samples remained acceptable at the data layer:
  - `pair_0047`: `mask_ratio 0.0508`, `final_luma_delta +0.0007`
  - `pair_0050`: `mask_ratio 0.0590`, `final_luma_delta +0.0052`

Conclusion:

- The full 12-sample approved manifest is now the current masked promotion target.
- Adding `pair_0047` and `pair_0050` did not introduce a dataset-format or color-alignment regression large enough to block training.
- The masked route is now operating on a full label-complete first seed pack rather than an intentionally incomplete approved subset.

## 2026-04-03E - Full 12-Sample Local Smoke And Dry-Run Verification

Hypothesis:
If the newly promoted full 12-sample masked dataset is still training-safe, the local smoke route should continue to launch, step, checkpoint, and preview successfully without showing a new early-training collapse after adding `pair_0047` and `pair_0050`.

Change made:

- Ran a 4-step local smoke on the full 12-sample dataset into:
  - [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4)
- Ran a 10-step local dry-run on the same dataset into:
  - [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10)
- Extended the same single-variable test to `25` steps, changing only run length, into:
  - [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25)

Result:

- 4-step smoke completed and wrote:
  - [`metrics.jsonl`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4/metrics.jsonl)
  - [`training_config.json`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4/training_config.json)
  - [`pytorch_lora_weights_step_000004.safetensors`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4/lora_checkpoints/pytorch_lora_weights_step_000004.safetensors)
  - [`preview_step_000004.png`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4/previews/preview_step_000004.png)
- 10-step dry-run completed and wrote:
  - [`metrics.jsonl`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10/metrics.jsonl)
  - [`training_config.json`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10/training_config.json)
  - [`pytorch_lora_weights_step_000010.safetensors`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10/lora_checkpoints/pytorch_lora_weights_step_000010.safetensors)
  - [`preview_step_000010.png`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10/previews/preview_step_000010.png)
- 25-step dry-run completed and wrote:
  - [`metrics.jsonl`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25/metrics.jsonl)
  - [`training_config.json`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25/training_config.json)
  - [`pytorch_lora_weights_step_000025.safetensors`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25/lora_checkpoints/pytorch_lora_weights_step_000025.safetensors)
  - [`preview_step_000025.png`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25/previews/preview_step_000025.png)
- The full-12 step-25 run stayed finite through all steps:
  - min loss: `2.1221`
  - max loss: `12.9196`
  - step 25 loss: `8.1130`
- Preview behavior on the sampled validation example did not reveal a new early catastrophic color or structural failure relative to the earlier 10-sample approved runs.

Conclusion:

- The full 12-sample masked dataset is locally training-stable for at least `4`, `10`, and `25` step smoke-scale checks.
- The compute boundary has not become a hard blocker yet on this machine; the real limit is efficiency, not correctness.
- The next masked question is no longer “can full-12 train locally at all”; it is whether the next more meaningful run should stay local or move to GitHub/Colab for a larger compute budget.

## 2026-04-03F - Colab Entry Point For Full 12-Sample Masked Training

Hypothesis:
If the masked route gets a Colab notebook and YAML config that mirror the structure of the historical paired-edit notebooks, longer GPU-side runs can start immediately without re-inventing the notebook workflow each time.

Change made:

- Added:
  - [`colab/masked_inpaint_full12_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_config.yaml)
  - [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb)
- The new notebook follows the historical notebook pattern:
  - mount Drive
  - load YAML config
  - rebuild or copy the masked dataset
  - construct the masked training command
  - periodically sync outputs back to Drive

Result:

- The new notebook JSON parses cleanly and the config YAML loads successfully.
- The config defaults point at the full approved 12-sample masked dataset and a more meaningful GPU-side training regime:
  - resolution `512`
  - max train steps `200`
  - checkpointing every `50` steps
  - preview every `50` steps
- This does not change the local experiment result; it prepares the next GitHub/Colab handoff path only.

Conclusion:

- The repository now has a masked-route Colab training entrypoint that matches the style of the historical paired-edit notebooks.
- If longer runs become inefficient locally, the masked route can now be handed off to GitHub/Colab without notebook scaffolding debt.

## 2026-04-03G - Harden Colab Dataset Preparation For The Full-12 Masked Notebook

Hypothesis:
If the masked Colab notebook handles missing Drive raw paths more gracefully, handoff failures will move from a brittle fixed-path assumption to a clearer data-availability check, which reduces false alarms during Colab startup.

Change made:

- Updated [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb) so dataset preparation now follows a fallback chain:
  - use configured `drive_raw_dir` when present
  - otherwise use configured `drive_dataset_dir` when present
  - otherwise auto-discover a plausible raw pair root under `/content/drive/MyDrive`
  - otherwise raise a clearer error showing the checked paths
- Pushed the notebook update to the existing GitHub handoff branch:
  - [`codex/masked-full12-colab`](https://github.com/Zifpen/nail-retouch-assistant/tree/codex/masked-full12-colab)

Result:

- The notebook no longer fails immediately just because `/content/drive/MyDrive/nail-retouch-raw` is missing.
- The handoff path now tolerates two common Colab situations:
  - the raw pairs live in a different Drive folder than the default config assumed
  - a built masked dataset already exists in Drive and can be copied directly
- When neither path exists, the error now tells the user what was checked instead of only echoing one fixed missing directory.

Conclusion:

- This was a Colab notebook robustness issue, not a regression in the masked dataset or training route.
- The current masked Colab branch is now more resilient to Drive layout differences and should be retried from a fresh clone of the same branch before assuming manual path surgery is needed.

## 2026-04-04A - Archive Full-12 Colab Training Outputs And Push Validation To The Local Boundary

Hypothesis:
If the first real full-12 Colab run is archived locally with its metrics, previews, and checkpoints intact, we can distinguish model-quality evidence from environment blockers and decide whether local validation is still worth pursuing before another Colab pass.

Change made:

- Archived the user-provided Colab output zip into:
  - [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs)
- Confirmed the archive includes:
  - [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/metrics.jsonl)
  - [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/training_config.json)
  - four LoRA checkpoints at `step50 / step100 / step150 / step200`
  - four previews at `step50 / step100 / step150 / step200`
- Reused the existing Evaluation Agent to analyze the archived run, compare `step150` vs `step200`, and attempt to push masked validation as far as the local environment allowed.

Result:

- The archived Colab run used the intended full-12 masked config:
  - base model: `stable-diffusion-v1-5/stable-diffusion-inpainting`
  - dataset: `dataset/masked_inpaint_cuticle_cleanup_v1`
  - resolution: `512`
  - rank: `4`
  - learning rate: `1e-5`
  - max train steps: `200`
  - checkpointing / preview cadence: every `50` steps
- `metrics.jsonl` contains all `200` training steps and does not show divergence:
  - mean loss `step 1-50`: `4.6625`
  - mean loss `step 51-100`: `3.5326`
  - mean loss `step 101-150`: `2.6857`
  - mean loss `step 151-200`: `2.2155`
  - `loss_color` and `loss_mask_rgb` also trended down
  - `loss_identity` stayed roughly stable around `~0.018`
- Preview-based comparison between `step150` and `step200` was directionally favorable to `step200` on all four validation previews, but the improvement was small:
  - mean preview output-to-target proxy distance at `step150`: `0.054433`
  - mean preview output-to-target proxy distance at `step200`: `0.054307`
- The attempt to continue with real local masked validation hit an environment blocker:
  - the local machine does not currently expose a complete offline-loadable inpainting base snapshot for `StableDiffusionInpaintPipeline`
  - the blocker is specific to inference-time base-model availability, not to the LoRA checkpoints or the dataset format

Conclusion:

- The first real full-12 Colab run is now locally archived and project-traceable.
- Training evidence currently supports `step200` as the provisional best checkpoint, with `step150` worth keeping as a rollback candidate.
- The next blocker is no longer “did training run successfully”; it is “where can we run true masked validation with a complete inpainting base available.”
- This is an environment / artifact-availability problem first, not an obvious dataset or optimization failure.

## 2026-04-04B - Restore Local Inpainting Base And Run First Real Masked Validation

Hypothesis:
If the missing local inpainting base snapshot is restored, the archived full-12 Colab checkpoints can be validated locally on at least one real masked sample, which should clarify whether checkpoint selection is limited by training quality or by inference-time artifacts.

Change made:

- Confirmed the previous local validation blocker was a missing `model_index.json` under the cached inpainting snapshot.
- Restored the full local base model snapshot for:
  - [`stable-diffusion-v1-5/stable-diffusion-inpainting`](/Users/simon/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)
- Reused the Evaluation Agent to run real masked validation against the archived Colab checkpoints on the local machine.

Result:

- Local real masked validation now works at least for one true validation sample.
- The following real artifacts were written for `pair_0009`:
  - [`step150 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_nocomposite/pytorch_lora_weights_step_000150/pair_0009_sheet.png)
  - [`step150 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_nocomposite/pytorch_lora_weights_step_000150/pair_0009_metrics.json)
  - [`step200 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0009_sheet.png)
  - [`step200 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0009_metrics.json)
- On `pair_0009`, `step150` was slightly better than `step200`, but the gap was very small:
  - `step150`:
    - `masked_l1_to_target`: `0.09784`
    - `masked_delta_e_to_target`: `11.0908`
    - `unmasked_l1_to_input`: `0.02183`
    - `border_l1_to_target`: `0.08894`
  - `step200`:
    - `masked_l1_to_target`: `0.09807`
    - `masked_delta_e_to_target`: `11.1122`
    - `unmasked_l1_to_input`: `0.02185`
    - `border_l1_to_target`: `0.08912`
- A second validation attempt on `pair_0040` with `step200` produced a black output because the diffusers safety checker triggered, so that sample currently does not provide trustworthy quality evidence:
  - [`pair_0040 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0040_metrics.json)

Conclusion:

- The local masked validation path is now genuinely restored; the project is no longer blocked on missing inpainting-base files.
- For the first real sample (`pair_0009`), `step150` is the slightly stronger checkpoint and should now be treated as the provisional preferred candidate.
- The next blocker has shifted again: it is no longer missing base-model artifacts, but inference-time safety-checker interference on at least one validation sample.
- The next single-variable validation move should keep checkpoint / dataset / prompt fixed and only change the sample or seed enough to recover a second trustworthy validation point.

## 2026-04-04C - Recover The Second Trustworthy Local Validation Point

Hypothesis:
If `pair_0040` is only failing because of sample-level safety-checker interference, then a narrow seed retry on the same checkpoint should either recover that sample or prove that the next clean validation point must come from another existing val pair.

Change made:

- Reused the Evaluation Agent and kept the validation route single-variable.
- Held checkpoint fixed at:
  - [`pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors)
- Tried `pair_0040` with seed retries:
  - `2026`
  - `7`
  - `42`
- After those retries still blacked out, moved to the next val sample without changing the checkpoint or other validation settings:
  - `pair_0047` with `seed=2026`

Result:

- `pair_0040` remained unusable as a ranking sample under all three retry seeds because the safety checker still blacked out the generated image.
- `pair_0047` succeeded on the first try and produced a second trustworthy local validation artifact:
  - [`pair_0047 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047/pytorch_lora_weights_step_000150/pair_0047_sheet.png)
  - [`pair_0047 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047/pytorch_lora_weights_step_000150/pair_0047_metrics.json)
- `pair_0047` metrics for `step150`:
  - `masked_l1_to_target`: `0.1310`
  - `masked_delta_e_to_target`: `17.1773`
  - `unmasked_l1_to_input`: `0.0450`
  - `unmasked_delta_e_to_input`: `5.6020`
  - `border_l1_to_target`: `0.0956`

Conclusion:

- The project now has two trustworthy local validation points for the archived full-12 masked run:
  - `pair_0009`
  - `pair_0047`
- `step150` remains the current preferred checkpoint because it has now succeeded on two real local validation samples without evidence of catastrophic drift or structural collapse.
- `pair_0040` should currently be treated as a sample-specific safety-checker outlier, not as evidence that the checkpoint or validation route is globally broken.
- The next masked validation step is no longer “recover a second usable sample”; it is either a direct `step150 vs step200` comparison on `pair_0047` or a decision about whether current two-point evidence is already enough for the next modeling move.

## 2026-04-04D - Complete The `pair_0047` Step150 vs Step200 Checkpoint Comparison

Hypothesis:
If `step150` was only winning on `pair_0009` by chance, then `pair_0047` should be able to overturn that ranking once the same validation settings are rerun with `step200`.

Change made:

- Reused the Evaluation Agent to run the missing direct comparison on `pair_0047`.
- Kept the validation setup fixed:
  - sample: `pair_0047`
  - seed: `2026`
  - same local inpainting base and validation route
- Changed only the checkpoint:
  - from `step150`
  - to `step200`

Result:

- `pair_0047 step200` produced a trustworthy artifact on the first try:
  - [`pair_0047 step200 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047_step200/pytorch_lora_weights_step_000200/pair_0047_sheet.png)
  - [`pair_0047 step200 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047_step200/pytorch_lora_weights_step_000200/pair_0047_metrics.json)
- Direct `pair_0047` comparison:
  - `step150`
    - `masked_l1_to_target`: `0.131037`
    - `masked_delta_e_to_target`: `17.177265`
    - `unmasked_l1_to_input`: `0.044957`
    - `unmasked_delta_e_to_input`: `5.601954`
    - `border_l1_to_target`: `0.095576`
  - `step200`
    - `masked_l1_to_target`: `0.131108`
    - `masked_delta_e_to_target`: `17.178013`
    - `unmasked_l1_to_input`: `0.044952`
    - `unmasked_delta_e_to_input`: `5.600216`
    - `border_l1_to_target`: `0.095658`
- The differences are extremely small and remain in the same local neighborhood as the earlier `pair_0009` result.

Conclusion:

- `step150` remains the current preferred checkpoint after two clean local ranking samples:
  - `pair_0009`
  - `pair_0047`
- `step200` is still close enough to preserve as an archived comparison point, but it no longer has evidence of being the better default.
- The current masked-validation question is no longer “which of these two checkpoints wins”; that ranking is now stable enough for near-term use.
- The next blocker is not checkpoint ambiguity, but deciding whether the current two-sample local evidence is sufficient to move on or whether wider validation coverage is worth the extra runtime.

## 2026-04-04E - Finish The Remaining Full-12 Val Sample Classification

Hypothesis:
If the last unrun validation sample `pair_0050` behaves like `pair_0040`, then the real limitation of the current local validation set is sample-level safety instability rather than missing validation effort.

Change made:

- Reused the Evaluation Agent to classify the remaining unrun validation sample.
- Held validation settings fixed at the current default checkpoint:
  - [`pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors)
- Ran:
  - `pair_0050` with `seed=2026`
  - `pair_0050` with one retry at `seed=7`

Result:

- Both `pair_0050` attempts completed inference but were blacked out by the safety checker.
- The resulting artifacts were written, but they are not trustworthy quality evidence:
  - [`pair_0050 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/final_pair0050/pytorch_lora_weights_step_000150/pair_0050_sheet.png)
  - [`pair_0050 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/final_pair0050/pytorch_lora_weights_step_000150/pair_0050_metrics.json)
- Distorted black-image metrics for `pair_0050`:
  - `masked_l1_to_target`: `0.3090`
  - `masked_delta_e_to_target`: `34.4014`
  - `unmasked_l1_to_input`: `0.1780`
  - `unmasked_delta_e_to_input`: `19.9375`
  - `border_l1_to_target`: `0.2631`
- Full current full-12 validation classification:
  - trustworthy local ranking samples:
    - `pair_0009`
    - `pair_0047`
  - safety-unstable local validation samples:
    - `pair_0040`
    - `pair_0050`

Conclusion:

- The current local validation coverage for the archived full-12 run is now fully classified.
- The useful local ranking evidence comes from `pair_0009` and `pair_0047`.
- `pair_0040` and `pair_0050` should currently be treated as sample-level safety-checker outliers rather than as quality evidence for or against the checkpoint.
- No further local validation retries are required on the current val set unless a later targeted safety-checker investigation becomes important.

## 2026-04-04F - Launch The Legacy `core_v2` Dataset-Only Retrain And Classify The Local Blocker

Hypothesis:
If the guarded legacy route is rerun with only the dataset changed from `core_v1` to `core_v2`, the project can get a low-ambiguity answer about how much dataset cleanup alone improves whitening, drift, and texture loss.

Change made:

- Reused a Training Agent to launch a real local `core_v2` retrain while holding the guarded training variables fixed.
- Kept unchanged:
  - model: `stabilityai/sd-turbo`
  - prompt
  - resolution `256`
  - learning rate `5e-6`
  - batch size `1`
  - rank settings
  - loss weights
  - change-mask settings
- Changed only the dataset input:
  - from `dataset/paired_edit_core_v1`
  - to [`dataset/paired_edit_core_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v2)

Result:

- The retrain path reached real initialization, which confirms the new dataset itself is structurally usable by the legacy training route.
- The run failed before `step 1`, so no checkpoint, viz, or metrics were written.
- The local output root only contains:
  - [`/tmp/paired_edit_core_v2_retrain_local/run_config.json`](/tmp/paired_edit_core_v2_retrain_local/run_config.json)
- The concrete blocking error was:
  - `ValueError: xformers is not available, please install it first.`

Conclusion:

- The next legacy experiment is not blocked by dataset formatting or by a broken training command.
- It is blocked locally by environment capability: the guarded config still expects `xformers`, and this machine does not currently satisfy that requirement.
- To preserve the clean dataset-only experiment definition, the right next move is not to quietly change the config, but to continue the same retrain in a Colab / GPU environment that already supports the guarded route.

## 2026-04-04G - Prepare A Dedicated Colab Handoff For The Legacy `core_v2` Retrain

Hypothesis:
If the `core_v2` dataset-only retrain gets its own Colab config and notebook entrypoint, the local `xformers` blocker can be bypassed without contaminating the experiment definition.

Change made:

- Reused the Training Agent to prepare a dedicated Colab handoff for the `core_v2` retrain.
- Added:
  - [`colab/paired_edit_core_v2_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_core_v2_config.yaml)
  - [`colab/train_paired_edit_core_v2_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_paired_edit_core_v2_v1.ipynb)
- Relative to `core_v1`, the handoff changed only:
  - dataset path
  - project name
  - prepared workdir / output naming
  - Drive dataset/output locations

Result:

- The repository now has a dedicated `core_v2` Colab training entrypoint for the dataset-only retrain.
- The guarded training variables remain unchanged in the `core_v2` config, including `enable_xformers_memory_efficient_attention: true`.
- This preserves the experiment definition: local failure is treated as an environment boundary, not as permission to silently weaken the guarded route.

Conclusion:

- The legacy `core_v2` dataset-only retrain is now ready for GitHub / Colab handoff.
- The next meaningful artifact for this line is not another local launch attempt, but the first real GPU-side `core_v2` checkpoint from the dedicated Colab entrypoint.

## 2026-04-05A - Fix The Colab Handoff / Trainer CLI Mismatch For The Legacy `core_v2` Retrain

Hypothesis:
If the Colab handoff is already emitting the intended guarded arguments, then the `unrecognized arguments` failure means the branch-visible trainer script is older than the local working trainer and needs to be synced to GitHub.

Change made:

- Verified locally that the current workspace version of:
  - [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py)
  does support:
  - `--paired_prompt`
  - `--lambda_full_l1`
  - `--lambda_preserve`
  - `--lambda_edit`
  - `--change_mask_threshold`
  - `--change_mask_dilate`
- Verified that the checked-in branch version previously visible to Colab did not expose those arguments.
- Pushed the updated trainer script to the active handoff branch:
  - [`codex/masked-full12-colab`](https://github.com/Zifpen/nail-retouch-assistant/tree/codex/masked-full12-colab)

Result:

- The `core_v2` Colab notebook and the branch-visible trainer CLI are now aligned again.
- The failure mode has been reclassified:
  - not a Drive dataset layout problem
  - not a notebook command-construction problem
  - a branch sync problem between the Colab handoff and the trainer script
- The correct retry path is now to fresh-clone the updated branch and rerun the same `core_v2` notebook.

Conclusion:

- The current blocker has been reduced from “command line mismatch” to “needs a fresh Colab clone of the updated branch.”
- No experiment-definition variables changed in this fix; the repair only synchronized the handoff branch with the intended trainer CLI.

## 2026-04-05B - Identify The First-Eval Outlier Blocking The Legacy `core_v2` Retrain

Hypothesis:
If the guarded `core_v2` Colab retrain now launches but dies in the first `evaluate()` call, then at least one validation sample is violating the trainer's local-edit change-mask sanity guard, and we need to identify that sample before deciding whether the experiment should continue unchanged.

Change made:

- Re-read the project memory and kept the round scoped to a single goal: classify the new `core_v2` Colab blocker without changing training variables.
- Reused the Training Agent to judge whether the failure should be treated as a data problem, an eval-policy problem, or a trainer bug.
- Recomputed the effective eval-time change-mask ratios on the `core_v2` test split under the trainer's own assumptions:
  - `resize_256`
  - `threshold=0.12`
  - `dilate=3`
- Cross-checked the result against the hard guard in [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py), where `change_ratio >= 0.60` raises a `ValueError`.

Result:

- The `core_v2` Colab retrain is now a real training run, not just a launch smoke:
  - it initialized fully
  - it completed `step 1`
  - it wrote a first sample preview before failing
- The new blocking error is:
  - `ValueError: Change mask covers too much of the image: ratios=[0.657501220703125]`
- The overflowing validation sample is:
  - `pair_0050`
  - split: `test`
- Under the trainer's actual eval-time preprocessing, the current `core_v2` test split produces:
  - `pair_0009`: `0.1562`
  - `pair_0040`: `0.2229`
  - `pair_0047`: `0.3716`
  - `pair_0050`: `0.6575`
  - `pair_0064`: `0.1799`
- This means the retrain is no longer blocked by:
  - Drive layout
  - CLI mismatch
  - missing `xformers`
  - dataset formatting
- It is now blocked by a validation outlier that violates the trainer's locality contract.

Conclusion:

- The current `core_v2` dataset-only retrain should be interpreted as:
  - training path works
  - first eval is blocked by an oversized-change validation sample
- This is primarily a data / split-fit problem exposed by the eval guard, not a trainer bug.
- The right next step is not to silently relax the guard, disable eval, or change loss / prompt / resolution.
- The next decision should stay on the data side:
  - either keep `pair_0050` out of the clean local-edit validation split
  - or explicitly reclassify it into a harder validation bucket instead of pretending it belongs to the same local-retouch holdout set.

## 2026-04-06A - Archive And Read The Legacy `core_v2` Cleanval Colab Retrain

Hypothesis:
If the clean-val split successfully removes the oversized validation outlier without changing guarded training variables, then the resulting `core_v2` retrain should complete end-to-end and show whether dataset filtering alone reduces the historical white-collapse / tint-drift failure mode.

Change made:

- Archived the user-provided Colab zip:
  - [`/Volumes/DevSSD/Download/nail-retouch-paired-core-v2-cleanval-outputs-20260406T045549Z-3-001.zip`](/Volumes/DevSSD/Download/nail-retouch-paired-core-v2-cleanval-outputs-20260406T045549Z-3-001.zip)
  into:
  - [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs)
- Reused the Evaluation Agent to summarize the archived run relative to the historical legacy failure modes.
- Attempted to continue the standard local paired-edit validation protocol against:
  - [`outputs/checkpoints/model_1401.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/checkpoints/model_1401.pkl)
  - [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl)
- Recovered a complete local upstream clone at:
  - [`/tmp/img2img-turbo-local-full`](/tmp/img2img-turbo-local-full)
  after confirming the previous local upstream stub was incomplete.

Result:

- The clean-val retrain is a real, complete guarded run:
  - reached `1500` steps
  - wrote checkpoints through `model_1500.pkl`
  - wrote eval metrics through `metrics_001500.json`
  - wrote training samples through `train_step_001500.png`
- The archived config confirms this remained a dataset-only retrain:
  - dataset: `paired_edit_core_v2_cleanval`
  - prompt, resolution, loss weights, and change-mask settings stayed on the guarded values
- Eval metrics improved monotonically without late collapse:
  - `val/full_l1`: `0.8378 -> 0.1339`
  - `val/preserve_l1`: `0.8903 -> 0.0866`
  - `val/edit_l1`: `0.8616 -> 0.2201`
  - `val/lpips`: `0.7239 -> 0.3397`
  - `val/change_ratio`: fixed at `0.2326`
- Qualitative read from the archived training samples:
  - the earliest step still resembles the old white-collapse regime
  - by the mid / late checkpoints, the run no longer shows catastrophic full-image whitening
  - strong magenta / orange tint drift is reduced relative to the older bad samples
  - blur / texture softness remain visible
- The local checkpoint-to-checkpoint validation protocol did not complete in this round because the local legacy validation environment is still fragile:
  - the original `/tmp/img2img-turbo-local/src` clone was incomplete
  - after recovering a full clone, the validation helper still remained environment-fragile around temporary runtime preparation

Conclusion:

- The clean-val `core_v2` retrain materially improved the legacy route's worst failure mode:
  - less catastrophic whitening
  - less obvious global color drift
- It did not fully solve the legacy route:
  - blur / texture loss remain visible enough that the route should not yet be treated as production-safe
- The best current interpretation is:
  - dataset filtering moved the model from a collapse regime into a softer but still blurry regime
- The highest-value next legacy step remains a strict local validation comparison:
  - `model_1401.pkl` vs clean-val `model_1500.pkl`
  - fixed baseline pair set
  - no new training-variable changes

## 2026-04-06B - Finish The Strict Legacy Baseline Comparison (`model_1401` vs `core_v2 cleanval model_1500`)

Hypothesis:
If the clean-val `core_v2` retrain really improved the legacy route rather than only looking better in isolated train samples, then the same fixed baseline pair set should look less whitened and less color-shifted under `model_1500.pkl` than under `model_1401.pkl`, even if texture softness remains.

Change made:

- Stabilized the local paired-edit validation helper by making runtime preparation in [`src/paired_edit/pix2pix_runtime.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/pix2pix_runtime.py) more robust against stale half-built `/tmp/img2img-turbo-runtime/<device>` directories.
- Recovered a complete upstream clone at:
  - [`/tmp/img2img-turbo-local-full`](/tmp/img2img-turbo-local-full)
- Ran the strict baseline comparison protocol on the same four fixed pairs:
  - `pair_0005`
  - `pair_0015`
  - `pair_0009`
  - `pair_0040`
- Old baseline outputs:
  - [`outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401)
- New clean-val outputs:
  - [`outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500)

Result:

- The local validation protocol now completes successfully again on CPU.
- Old baseline `model_1401` remains in the historical failure regime on the fixed comparison set:
  - strong whitening
  - obvious pink / magenta drift
  - severe collapse on `pair_0005`, `pair_0009`, `pair_0015`, and `pair_0040`
- Clean-val `model_1500` is clearly better than `model_1401` on the same pairs:
  - much less catastrophic whole-image whitening
  - much less obvious magenta / red color cast
  - outputs remain recognizably aligned to the input / target semantics
- Representative sheets:
  - [`pair_0009 baseline1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401/pair_0009_sheet.png)
  - [`pair_0009 model1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500/pair_0009_sheet.png)
  - [`pair_0040 baseline1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401/pair_0040_sheet.png)
  - [`pair_0040 model1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500/pair_0040_sheet.png)
  - [`pair_0005 baseline1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401/pair_0005_sheet.png)
  - [`pair_0005 model1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500/pair_0005_sheet.png)
  - [`pair_0015 baseline1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401/pair_0015_sheet.png)
  - [`pair_0015 model1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500/pair_0015_sheet.png)
- Remaining weakness:
  - `model_1500` is still visibly soft / blurry
  - detail and edge fidelity remain below the paired target
  - this is an improvement over collapse, not a full-quality recovery

Conclusion:

- The dataset-only clean-val intervention is now supported by a direct checkpoint-to-checkpoint comparison:
  - `model_1500` is meaningfully better than `model_1401`
  - especially on whitening and color drift
- The remaining legacy problem is no longer catastrophic collapse first; it is blur / texture loss.
- The next legacy filtering decision should focus on whether second-tier drift pairs like `pair_0022` / `pair_0066` are now the dominant source of residual softness and preserve-region instability.

## 2026-04-06C - Split `pair_0022` / `pair_0066` Out Of The Legacy Clean Baseline And Build `core_v3`

Hypothesis:
If `pair_0022` and `pair_0066` are now the strongest remaining residual drift pairs inside the legacy clean baseline, then removing only those two from the default training pool should produce a cleaner `core_v3` candidate dataset without changing prompts, losses, or resolution.

Change made:

- Reused the Evaluation Agent to review whether `pair_0022` and `pair_0066` should remain in `core_v2` after the clean-val retrain established that collapse is no longer the main failure mode.
- Re-ran the paired drift audit and reviewed direct before/after sheets for:
  - [`/tmp/core_v2_risk_sheets/pair_0022_sheet.png`](/tmp/core_v2_risk_sheets/pair_0022_sheet.png)
  - [`/tmp/core_v2_risk_sheets/pair_0066_sheet.png`](/tmp/core_v2_risk_sheets/pair_0066_sheet.png)
- Promoted the data-split decision into new manifests and built datasets:
  - [`dataset/annotations/paired_edit_core_v3_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_secondary_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_secondary_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_cleanval_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_cleanval_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_hardval_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_hardval_manifest.json)
  - [`dataset/paired_edit_core_v3`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3)
  - [`dataset/paired_edit_core_v3_secondary`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_secondary)
  - [`dataset/paired_edit_core_v3_cleanval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_cleanval)
  - [`dataset/paired_edit_core_v3_hardval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_hardval)

Result:

- `pair_0022` and `pair_0066` remain the top two train-side drift pairs in `core_v2`:
  - `pair_0022`: `score=0.3616`, `luma=+0.0846`, `change=0.4504`, `preserve=0.0719`
  - `pair_0066`: `score=0.3412`, `luma=+0.0780`, `change=0.4508`, `preserve=0.0487`
- Visual review matches the audit readout:
  - both pairs teach substantial whole-hand brightening / smoothing rather than only local retouch cleanup
  - `pair_0022` is the more aggressive of the two
- Built dataset summaries:
  - `core_v3`: `23 train / 5 val`
  - `core_v3_secondary`: `2 train / 0 val`
  - `core_v3_cleanval`: `23 train / 4 val`
  - `core_v3_hardval`: `23 train / 1 val`
- Audit improvements versus `core_v2`:
  - `core_v2` mean drift score: `0.1377`
  - `core_v3` mean drift score: `0.1224`
  - `core_v3_cleanval` mean drift score: `0.1162`
- `core_v3_cleanval` top remaining risky samples are now:
  - `pair_0035`
  - `pair_0118`
  - `pair_0120`
  - `pair_0070`

Conclusion:

- The evidence is now strong enough to treat `pair_0022` and `pair_0066` as secondary-set members rather than part of the clean legacy baseline.
- This is not proof that they were the only source of residual softness, but it is a stable enough data-side decision to justify the next single-variable experiment.
- The next legacy experiment should now be:
  - a guarded `core_v3 cleanval` retrain
  - with training variables unchanged
  - so we can test whether removing these second-tier drift pairs reduces residual blur / preserve-region instability further.

## 2026-04-06D - Archive And Judge The `core_v3 cleanval` Retrain Against `core_v2 cleanval`

Hypothesis:
If `pair_0022` and `pair_0066` were still major residual drift drivers inside the legacy clean baseline, then the guarded `core_v3 cleanval` retrain should outperform `core_v2 cleanval` on clean-val metrics and on the same strict fixed-pair local validation set.

Change made:

- Archived the user-provided Colab zip:
  - [`/Volumes/DevSSD/Download/nail-retouch-paired-core-v3-cleanval-outputs-20260406T170817Z-3-001.zip`](/Volumes/DevSSD/Download/nail-retouch-paired-core-v3-cleanval-outputs-20260406T170817Z-3-001.zip)
  into:
  - [`outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs)
- Reused the Evaluation Agent to judge whether `core_v3 cleanval` meaningfully improved the legacy dataset-only control line over `core_v2 cleanval`.
- Reused the Training Agent to define the same strict fixed-pair local validation protocol used for `core_v2 cleanval`, then completed that protocol locally against:
  - [`outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs/checkpoints/model_1500.pkl)
- New strict local validation outputs:
  - [`outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500)

Result:

- The archived `core_v3 cleanval` run is a valid guarded dataset-only retrain:
  - dataset: `paired_edit_core_v3_cleanval`
  - prompt, resolution, loss weights, and change-mask settings stayed unchanged
  - run completed to `1500` steps with checkpoints, eval metrics, and training samples
- At `step1500`, `core_v3 cleanval` did not outperform `core_v2 cleanval`; it was slightly worse on every key eval metric:
  - `val/full_l1`: `0.1339 -> 0.1386`
  - `val/preserve_l1`: `0.0866 -> 0.0900`
  - `val/edit_l1`: `0.2201 -> 0.2266`
  - `val/l2`: `0.0410 -> 0.0427`
  - `val/lpips`: `0.3397 -> 0.3667`
- Representative training samples also continue to show softness / blur rather than a new qualitative recovery:
  - [`core_v2 step1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/samples/train_step_001500.png)
  - [`core_v3 step1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs/samples/train_step_001500.png)
- The same strict fixed-pair local validation protocol now also completes for `core_v3 cleanval model_1500`:
  - pairs: `pair_0005`, `pair_0015`, `pair_0009`, `pair_0040`
  - outputs: [`outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500)
- Fixed-pair readout is consistent with the eval metrics:
  - `core_v3` outputs are not identical to `core_v2`, but they do not show a new quality tier
  - the differences look like small shifts within the same soft / blurry regime, not a clear recovery jump

Conclusion:

- The `core_v3 cleanval` ablation is useful closing evidence, but it does not justify continuing deeper into `core_v4`-style dataset-only pruning as a mainline effort.
- The legacy dataset-only line has now answered its main question:
  - cleaning the dataset can rescue the route from catastrophic collapse
  - further small data-only pruning has diminishing returns and did not beat `core_v2 cleanval`
- The current best interpretation is:
  - keep `core_v2 cleanval model_1500` as the legacy paired-edit reference checkpoint
  - treat `core_v3 cleanval` as evidence that the legacy line is near its useful dataset-only ceiling
  - move the main project priority back to the masked route rather than continuing to deepen the legacy control line.

## 2026-04-06E - Preserve User-Provided Result Zips Inside The Workspace

Hypothesis:
If all externally supplied result zips are copied into a stable project-side archive with checksums and run mappings, then clearing the system `Download` folder will not break experiment traceability.

Change made:

- Created:
  - [`archive/2026-04-06_user_result_zips`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips)
- Copied these user-provided zip files into that archive:
  - `drive-download-20260404T040859Z-3-001.zip`
  - `nail-retouch-masked-full12-outputs-20260404T041131Z-3-001.zip`
  - `nail-retouch-paired-core-v2-cleanval-outputs-20260406T045549Z-3-001.zip`
  - `nail-retouch-paired-core-v3-cleanval-outputs-20260406T170817Z-3-001.zip`
- Added an archive manifest:
  - [`archive/2026-04-06_user_result_zips/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/README.md)

Result:

- All user-supplied external result bundles now exist inside the workspace archive, independent of `/Volumes/DevSSD/Download`.
- SHA1 checksums were recorded for all four zip files.
- The archive manifest also maps the three experiment-result zips to their already-ingested run directories under `outputs/`.

Conclusion:

- The project no longer depends on the `Download` folder for these previously shared result artifacts.
- It is now safe to clear those source files from `Download` without losing the project-side copies or their provenance.

## 2026-04-06F - Recover Full Masked Validation Coverage By Making The Safety Checker Optional

Hypothesis:
If the masked validation blackouts on `pair_0040` and `pair_0050` are primarily caused by the diffusion safety checker rather than by the LoRA itself, then a local-eval-only opt-in bypass should recover real outputs for those samples and let us judge `step150` vs `step200` on the full 4-sample validation split.

Change made:

- Reused the Evaluation Agent to judge whether masked-route validation coverage, not a fresh training run, was the current highest-value blocked question.
- Reused the Training Agent to isolate the blackout source and propose the smallest safe implementation boundary.
- Added an opt-in `--disable-safety-checker` flag to:
  - [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py)
  - [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py)
- Added [`build_inpaint_pipeline(...)`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/masked_inpaint_utils.py) so the safety checker is disabled only when that flag is explicitly passed; the default inference path stays unchanged.
- Re-ran the previously safety-unstable validation samples on the archived full-12 masked Colab checkpoints using the restored local inpainting base snapshot:
  - `step150` outputs:
    - [`pair_0040`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety/pytorch_lora_weights_step_000150/pair_0040_sheet.png)
    - [`pair_0050`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety/pytorch_lora_weights_step_000150/pair_0050_sheet.png)
  - `step200` outputs:
    - [`pair_0040`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety/pytorch_lora_weights_step_000200/pair_0040_sheet.png)
    - [`pair_0050`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety/pytorch_lora_weights_step_000200/pair_0050_sheet.png)

Result:

- The new outputs for `pair_0040` and `pair_0050` are no longer blacked out, which confirms that the earlier local-validation blocker was the safety checker rather than an inability to render those samples at all.
- Full 4-sample masked validation coverage is now available for the archived run:
  - `pair_0009`
  - `pair_0047`
  - `pair_0040`
  - `pair_0050`
- Across those four samples, `step150` remains slightly better than `step200` on every tracked mean metric:
  - mean `masked_l1_to_target`: `0.1148` vs `0.1150`
  - mean `masked_delta_e_to_target`: `14.0831` vs `14.0949`
  - mean `unmasked_l1_to_input`: `0.0357` vs `0.0357` (tie within rounding, with `step150` still marginally lower in raw values)
  - mean `unmasked_delta_e_to_input`: `4.1891` vs `4.1922`
  - mean `border_l1_to_target`: `0.0938` vs `0.0939`
- The recovered hard-case metrics are directionally consistent with the earlier clean-case readout:
  - `pair_0040`: `step150` is slightly better than `step200`
  - `pair_0050`: `step150` is slightly better than `step200`

Conclusion:

- The masked-route validation blind spot is now repaired well enough for local checkpoint ranking.
- The earlier blackouts on `pair_0040` / `pair_0050` should be treated as inference-time safety-checker artifacts, not as evidence that those validation samples were unusable forever.
- `step150` should remain the default archived full-12 masked checkpoint, with `step200` kept only as a near-neighbor reference.
- The next masked question should no longer be “can we trust the checkpoint ranking?” but “what is the next single training variable worth changing now that full-val coverage is back?”

## 2026-04-06G - Complete The Archived Full-12 Budget Curve Before Opening A Fresh Masked Retrain

Hypothesis:
If the next masked training variable is really training budget rather than dataset/task/loss, then the already archived `step050` and `step100` checkpoints should clarify whether the current optimum sits earlier than `step150` without requiring a fresh run.

Change made:

- Reused the Training Agent to rank candidate masked training variables and propose the smallest next experiment boundary.
- Reused the Evaluation Agent to decide which variable is worth testing first and why other candidates should wait.
- Both agents converged on the same answer:
  - the next single variable should be training budget / early stopping position
  - not dataset membership
  - not task split
  - not loss stack
  - not rank or resolution yet
- Before opening a new masked retrain, completed the same 4-sample local validation protocol for the two earlier archived checkpoints:
  - [`step050`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step050_disable_safety/pytorch_lora_weights_step_000050/summary.json)
  - [`step100`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step100_disable_safety/pytorch_lora_weights_step_000100/summary.json)
- Compared the full 4-point budget curve on:
  - `step050`
  - `step100`
  - `step150`
  - `step200`
  using the same validation pair set:
  - `pair_0009`
  - `pair_0040`
  - `pair_0047`
  - `pair_0050`

Result:

- The budget curve is now complete on the restored 4-sample local validation split.
- Mean metrics by checkpoint:
  - `step050`
    - `masked_l1_to_target=0.1147`
    - `masked_delta_e_to_target=14.0787`
    - `unmasked_l1_to_input=0.0356`
    - `unmasked_delta_e_to_input=4.1853`
    - `border_l1_to_target=0.0938`
  - `step100`
    - `masked_l1_to_target=0.1147`
    - `masked_delta_e_to_target=14.0758`
    - `unmasked_l1_to_input=0.0356`
    - `unmasked_delta_e_to_input=4.1833`
    - `border_l1_to_target=0.0938`
  - `step150`
    - `masked_l1_to_target=0.1148`
    - `masked_delta_e_to_target=14.0831`
    - `unmasked_l1_to_input=0.0357`
    - `unmasked_delta_e_to_input=4.1891`
    - `border_l1_to_target=0.0938`
  - `step200`
    - `masked_l1_to_target=0.1150`
    - `masked_delta_e_to_target=14.0949`
    - `unmasked_l1_to_input=0.0357`
    - `unmasked_delta_e_to_input=4.1922`
    - `border_l1_to_target=0.0939`
- This is not a dramatic separation, but it is directionally consistent:
  - the earlier checkpoints (`50` / `100`) are slightly better than `150` / `200`
  - `step100` is the best overall point among the four archived checkpoints
- The result supports an early-stop interpretation more strongly than a “keep training longer” interpretation.

Conclusion:

- The current best archived full-12 masked checkpoint is now `step100`, not `step150`.
- The project no longer needs to guess whether training budget is the most important next variable; we now have project-internal evidence that the optimum likely sits closer to `100` steps than to `150` or `200`.
- The next masked experiment should be a narrower early-stop / budget experiment on faster hardware, not a dataset/task/rank/resolution change.
- A dedicated Colab config now exists for that next step:
  - [`colab/masked_inpaint_full12_earlystop_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_earlystop_config.yaml)
  - it keeps dataset, task, prompt, rank, resolution, and losses fixed
  - it only tightens the budget window to `max_train_steps=150` with `checkpointing_steps=25` / `preview_steps=25`
- The existing Colab notebook now defaults to that early-stop config:
  - [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb)
  - `CONFIG_FILE` now points to `masked_inpaint_full12_earlystop_config.yaml`, so a fresh clone can run the budget-only refinement without manual notebook edits.
- The dataset-prepare cell in that notebook now also validates cached Drive datasets explicitly via:
  - `build_summary.json`
  - `train/metadata.jsonl`
  - `val/metadata.jsonl`
  so the Colab handoff prefers a correctly uploaded cached full-12 dataset more robustly and prints clearer diagnostics when the Drive copy is malformed.

## Visual Artifacts

- Whitening / blowout: confirmed in historical `model_251` output
- Pink / orange / magenta cast: confirmed in historical `model_501` output
- Texture loss / blur: confirmed in historical `model_501` output
- Whole-image lift: supported by both pruning notes and paired drift audit statistics

## Lessons Learned

- A small curated set can still be biased if most targets are globally brighter than sources.
- `phase1_expand_batch1_pruned` is safer than the full batch1 expansion, but it does not solve the bias already present in `core_v1`.
- `strict_plus` currently has the best overall drift stats among the built datasets in the workspace, but it still contains several high-drift pairs and mixes prompt variants with the newer `preserve original nail design` route.
- Do not spend the next iteration on more steps, higher rank, or prompt tweaks until the core dataset drift is reduced.
- The next experiment should isolate the data variable: build a `core_v2` manifest that removes the highest-drift core pairs and retrain with the exact guarded config unchanged.
- Pairwise color alignment on the unmasked region can remove most of the accidental global luma drift before training even starts.
- Difference-derived masks are not reliable enough for final supervision on harder train samples; they can still absorb large hand regions when the pair has broad appearance changes.
- A separate inpainting-specific training entrypoint is cleaner than continuing to extend the legacy pix2pix/full-image route.
- The masked route needs split validation metrics because a visually decent output can still be failing the real requirement by nudging unmasked skin tone.
- A small explicit-mask seed pack is enough to unblock the first real masked experiment; we do not need the whole dataset labeled before running a dry-run.
- Bootstrap mask ratio is a useful manual-priority signal: very broad drafts usually indicate the pair needs hand redraw, not just light cleanup.
- `core_v2` is strong enough to act as the next legacy paired baseline while we wait on the explicit-mask subset.
- A large portion of the dataset is not pure cuticle cleanup; many target edits perform local posterior-edge beautification as part of the manicure retouch.
- For this first masked subset, the honest task framing is `proximal_nail_boundary_refinement`.
- The right mask shape is often a narrow transition band between the old and new posterior edge, not just a trace of visible dead skin.
- A four-mask explicit smoke subset is enough to verify the explicit data path and training startup behavior.
- Local CPU execution at `512` resolution is too slow for convenient 10-step smoke runs, even though the pipeline now initializes and trains correctly.
- A named low-cost smoke preset helps keep local verification repeatable without pretending that a 256-resolution 4-step run says anything meaningful about final retouch quality.
- For this environment, the local smoke wrapper should prefer a cached inpainting snapshot automatically; otherwise network/DNS availability adds noise that has nothing to do with masked-route correctness.
- Near-binary masks with `[0, 254, 255]` are acceptable for QA, but the dataset build path should still normalize them to strict binary values before promotion.
- Anti-aliased grayscale masks can also pass QA when the region semantics are correct, but they must be binarized before dataset build so masked supervision stays deterministic.
- Among the four still-unlabeled seed-pack samples, `pair_0070` and `pair_0063` are the next best manual mask targets; `pair_0047` and especially `pair_0050` look higher-risk and should be deferred unless we explicitly want harder decorated / lighting-shift cases.
- An approved-subset manifest is a valid intermediate promotion step when the broader seed manifest still contains unlabeled samples; waiting for all 12 masks is not required to keep the masked route moving.
- Antialiased grayscale masks can still pass semantic QA when the shape is correct, but build-time binarization must be treated as mandatory before promotion into the training dataset.
- In the current local environment, default remote model resolution can fail even when the training stack is healthy; treat that as an environment/input-availability issue, not as a masked-route regression.
- The smoke preset is now proven to finish in under a minute once the base inpainting model is available locally, which is good enough for routine post-change plumbing checks.
- Once `pair_0063` and `pair_0070` are manually tightened enough to stay local, even previously broad bootstrap drafts can become valid `proximal_nail_boundary_refinement` supervision; bootstrap width alone should not force permanent exclusion.
- After promoting `pair_0063` and `pair_0070`, `pair_0047` and `pair_0050` become optional harder expansion cases, not blockers for the next masked-dataset iteration.
- A 10-sample approved subset can already support repeatable low-cost masked dry-runs locally; that is enough evidence to treat it as the default first masked training set until we deliberately add harder cases.
- Even the previously highest-risk remaining seed-pack samples can still fit `proximal_nail_boundary_refinement` if the authored masks stay genuinely local and avoid absorbing decoration or lighting-driven appearance changes.
- Once `pair_0047` and `pair_0050` are promoted, the full first seed pack still behaves like a training-safe masked dataset in local smoke-scale runs; the new question becomes compute budget, not whether those samples were a mistake to include.
- For Colab handoff, notebook brittleness around Drive paths can masquerade as a training blocker; keep the data-prep fallback chain robust so environment quirks do not get confused with modeling regressions.
- Externally produced result bundles should be archived immediately in both raw and extracted form; otherwise later cleanup of `Download` can silently remove reproducibility-critical artifacts.

### Experiment 2026-04-06H - Archive The Full-12 Early-Stop Masked Colab Result Before Any Download Cleanup

Hypothesis:
If each user-provided external result bundle is copied into the workspace archive and extracted into a stable run directory as soon as it arrives, we can safely clear transient download folders without losing later evaluation context.

Change made:

- Copied [`nail-retouch-masked-full12-earlystop-outputs-20260406T191151Z-3-001.zip`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/nail-retouch-masked-full12-earlystop-outputs-20260406T191151Z-3-001.zip) into the workspace archive
- Extracted the run into [`outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs)
- Added the bundle-to-run mapping and checksum to [`archive/2026-04-06_user_result_zips/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/README.md)

Result:

- The raw zip is now preserved in-project with SHA256 `670b06fc5efd120ff020268a1ff51c306bb1b3755c840640e56b8e778c74ce58`
- The extracted run now has stable local paths for:
  - `metrics.jsonl`
  - `training_config.json`
  - LoRA checkpoints at `25 / 50 / 75 / 100 / 125 / 150`
  - preview sheets at `25 / 50 / 75 / 100 / 125 / 150`
- The archived config confirms this is the intended budget-only masked refinement:
  - full-12 masked dataset
  - `resolution=512`
  - `rank=4`
  - `learning_rate=1e-5`
  - `max_train_steps=150`
  - `checkpointing_steps=25`

Conclusion:

- The masked early-stop Colab result is now preserved in both raw and extracted form.
- Clearing `/Volumes/DevSSD/Download` will not remove project access to this run.
- Result ingestion should be treated as mandatory project memory hygiene before any later evaluation or cleanup step.

### Experiment 2026-04-06I - Realign Masked Local Validation Protocol And Rank The Archived Early-Stop Run

Hypothesis:
If the ROI inpaint validation helper is made robust to crop-size mismatches and all early-stop checkpoints are re-evaluated under one consistent local validation protocol, we can make a trustworthy checkpoint-ranking decision for the new archived early-stop run.

Change made:

- Patched [`src/inference/masked_inpaint_utils.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/masked_inpaint_utils.py) so ROI-generated crops are resized back to the source crop size before exact outside-mask compositing when the diffusion pipeline returns an off-size crop
- Re-ran local masked validation with `--disable-safety-checker --preserve-unmasked-exact` for the early-stop archived checkpoints:
  - `step025`
  - `step050`
  - `step075`
  - `step100`
  - `step125`
  - `step150`
- Re-ran the old archived full-12 reference checkpoint `step100` under the same patched validation protocol for apples-to-apples comparison

Result:

- The earlier failure on the early-stop run was confirmed to be a validation-tool robustness problem, not a model-training or dataset problem
- Under the patched protocol, the early-stop run validates cleanly on all 4 standard local samples:
  - `pair_0009`
  - `pair_0040`
  - `pair_0047`
  - `pair_0050`
- On the patched protocol, exact outside-mask preservation is now correctly reflected as zero preserve error for these runs:
  - `mean_unmasked_l1_to_input = 0.0`
  - `mean_unmasked_delta_e_to_input = 0.0`
- Early-stop run mean metrics improve monotonically across the budget curve:
  - `step025`: `masked_l1 0.06630`, `masked_delta_e 8.74055`, `border_l1 0.02840`
  - `step050`: `masked_l1 0.06606`, `masked_delta_e 8.69487`, `border_l1 0.02835`
  - `step075`: `masked_l1 0.06582`, `masked_delta_e 8.64918`, `border_l1 0.02831`
  - `step100`: `masked_l1 0.06560`, `masked_delta_e 8.60186`, `border_l1 0.02828`
  - `step125`: `masked_l1 0.06543`, `masked_delta_e 8.56099`, `border_l1 0.02829`
  - `step150`: `masked_l1 0.06538`, `masked_delta_e 8.53859`, `border_l1 0.02836`
- Old archived full-12 `step100`, re-run under the same patched protocol, lands at:
  - `masked_l1 0.06558`
  - `masked_delta_e 8.60673`
  - `border_l1 0.02829`
- Relative to that old archived `step100`, the new early-stop `step150` is slightly better on masked edit metrics:
  - `masked_l1`: `-0.000205`
  - `masked_delta_e`: `-0.06815`
  - while `border_l1` is slightly worse by `+0.000069`

Conclusion:

- The evaluation discrepancy was protocol-alignment noise, not evidence of a sudden model-regime jump.
- Under one consistent patched validation protocol, the archived early-stop run is trustworthy and trends later than the previously favored archived `step100`.
- The best point in the early-stop run is now `step150`, with `step125` as the nearest low-risk neighbor.
- The new early-stop `step150` slightly outperforms the old archived `step100` on masked edit quality, but the gain is modest rather than dramatic.

### Experiment 2026-04-09A - Close The Patched Masked Budget Curve By Re-Running Old Archived `step150 / step200`

Hypothesis:
If the older archived full-12 checkpoints `step150` and `step200` are also re-run under the same patched local validation protocol, we can close the masked budget question without opening another training run.

Change made:

- Re-ran old archived full-12 `step150` under the patched exact-composite validation protocol into [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety_rerun`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety_rerun)
- Re-ran old archived full-12 `step200` under the same protocol into [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety_rerun`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety_rerun)
- Compared those results against:
  - old archived `step100` re-run
  - new early-stop run `step125`
  - new early-stop run `step150`

Result:

- Old archived `step150` re-run:
  - `masked_l1 0.065375`
  - `masked_delta_e 8.537905`
  - `border_l1 0.028373`
- Old archived `step200` re-run:
  - `masked_l1 0.065640`
  - `masked_delta_e 8.552402`
  - `border_l1 0.028603`
- Combined patched-protocol ranking across the strongest candidate points:
  - `old150`: best masked edit metrics by a hair
  - `new150`: essentially tied with `old150`
  - `new125`: slightly behind the two `step150` checkpoints
  - `old100`: clearly a little worse than the `150` points
  - `old200`: worse than both `150` points
- The gap is extremely small:
  - `old150` beats `new150` on `masked_l1` by about `0.000003`
  - `old150` beats `new150` on `masked_delta_e` by about `0.00068`
  - `new150` slightly beats `old150` on `border_l1` by about `0.000015`

Conclusion:

- The budget question is now effectively closed for the current full-12 masked setup.
- `150` is the stable best-step region; `200` is already beyond the useful optimum, and `100` is slightly early.
- The two `step150` checkpoints are practically tied, so the project can use the dedicated early-stop run as the cleaner current reference without losing meaningful quality.
- The next masked experiment should move on to a new single variable rather than reopening budget yet again.

### Experiment 2026-04-09B - Select The Next Masked Single Variable And Prepare The Colab Handoff

Hypothesis:
If the next masked experiment changes only `lambda_color` while keeping the current full-12 dataset, `step150` budget, rank, resolution, prompt, and validation protocol fixed, the result will be more interpretable than changing rank or resolution next.

Change made:

- Re-read project memory and locked the next-round goal to selecting exactly one new masked training variable.
- Reused the long-lived Evaluation Agent and Training Agent to independently rank the next safest masked single variable after budget closure.
- Added a dedicated Colab config at [`colab/masked_inpaint_full12_lambda_color_v1.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_lambda_color_v1.yaml)
- Updated [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb) so a fresh clone now defaults to that new config.

Result:

- Both reused agents ranked `lambda_color` as the best next single variable.
- The agreed reasoning was:
  - budget is already closed for the current full-12 setup
  - rank and resolution would introduce larger compute and interpretation shifts
  - `lambda_identity` is cheaper than rank / resolution, but current patched validation already shows exact outside-mask preservation
  - the remaining likely improvement target is mask-inside color stability and local color consistency
- The new Colab config changes only one training value relative to the current early-stop baseline:
  - `lambda_color: 0.5 -> 1.0`
- Everything else is held fixed:
  - dataset: `dataset/masked_inpaint_cuticle_cleanup_v1`
  - manifest: `dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`
  - budget: `max_train_steps=150`, `checkpointing_steps=25`, `preview_steps=25`
  - resolution: `512`
  - rank: `4`
  - `lambda_identity: 5.0`
  - `learning_rate: 1e-5`

Conclusion:

- The next masked experiment should be a single-variable `lambda_color` test, not another budget run and not a rank / resolution jump.
- The current best handoff is now ready for direct Colab use after a fresh clone.
- This keeps the next result interpretable against the existing `step150` masked reference checkpoint.

### Experiment 2026-04-09C - Run The Full-12 `lambda_color=1.0` Ablation And Compare It Against The Current Masked Reference

Hypothesis:
If the current masked route is slightly under-regularized on local color consistency, then raising `lambda_color` from `0.5` to `1.0` should improve masked edit metrics on the patched 4-sample validation protocol without damaging exact outside-mask preservation.

Change made:

- Archived the user-provided Colab output zip:
  - raw zip: [`archive/2026-04-06_user_result_zips/nail-retouch-masked-full12-lambda-color-outputs-20260409T193544Z-3-001.zip`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/nail-retouch-masked-full12-lambda-color-outputs-20260409T193544Z-3-001.zip)
  - extracted run: [`outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs)
- Verified that the run is a clean single-variable ablation:
  - dataset unchanged
  - `max_train_steps=150`
  - `resolution=512`
  - `rank=4`
  - only `lambda_color: 0.5 -> 1.0`
- Ran local masked validation for candidate checkpoints `100 / 125 / 150` and reconciled the output with the current patched 4-sample protocol over:
  - `pair_0009`
  - `pair_0040`
  - `pair_0047`
  - `pair_0050`

Result:

- Training-side run integrity is clean:
  - checkpoints at `25 / 50 / 75 / 100 / 125 / 150`
  - previews at the same steps
  - no sign of collapse or late-stage blowout
- Training-side trend remains late-improving:
  - `step100 loss 3.8658`
  - `step125 loss 3.7661`
  - `step150 loss 2.5106`
- Patched 4-sample validation means for the new run:
  - `lambda100`: `masked_l1 0.0655837`, `masked_delta_e 8.59608`, `border_l1 0.0282721`
  - `lambda125`: `masked_l1 0.0653976`, `masked_delta_e 8.55251`, `border_l1 0.0282762`
  - `lambda150`: `masked_l1 0.0653309`, `masked_delta_e 8.52669`, `border_l1 0.0283312`
- Current reference means:
  - `ref125`: `masked_l1 0.0654257`, `masked_delta_e 8.56099`, `border_l1 0.0282929`
  - `ref150`: `masked_l1 0.0653785`, `masked_delta_e 8.53859`, `border_l1 0.0283582`
- Relative to the current masked reference `ref150`, the new `lambda150` is slightly better on all primary tracked means:
  - `masked_l1`: improved by about `0.0000476`
  - `masked_delta_e`: improved by about `0.01190`
  - `border_l1`: improved by about `0.0000270`
- Exact outside-mask preservation remains intact:
  - `mean_unmasked_l1_to_input = 0.0`
  - `mean_unmasked_delta_e_to_input = 0.0`

Conclusion:

- `lambda_color=1.0` is a small but real positive move, not a regression.
- Within this run, `step150` is again the best checkpoint region, with `step125` as the nearest earlier neighbor.
- This strengthens confidence that the current masked route is stable enough to justify moving attention from training-budget uncertainty toward dataset expansion and more mask coverage.

### Experiment 2026-04-09D - Open The Next Conservative Explicit-Mask Seed Batch

Hypothesis:
If the masked route has already stabilized on the first full-12 subset, the next most valuable single step is to expand annotation coverage with a small second seed pack instead of opening another training-variable sweep.

Change made:

- Re-read the current memory after the `lambda_color=1.0` promotion.
- Reused the evaluation role to answer whether the route is ready for the next explicit-mask expansion.
- Reused the dataset role to draft [`dataset/annotations/masked_cuticle_cleanup_v2_seed_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v2_seed_manifest.json) as a conservative next annotation batch.

Result:

- The masked route is now considered ready for controlled expansion:
  - full-12 build, training, and validation are stable
  - the budget region is already narrowed
  - the first loss-side ablation was positive rather than destabilizing
- The v2 seed batch is intentionally small: `5 train + 1 val`
  - train: `pair_0118`, `pair_0122`, `pair_0153`, `pair_0154`, `pair_0190`
  - val: `pair_0064`
- The manifest keeps the task fixed at `proximal_nail_boundary_refinement`.
- This batch is meant to extend coverage conservatively, not to begin large-scale unconstrained annotation.

Conclusion:

- The main bottleneck has shifted from trainer uncertainty to annotation coverage.
- The project is now close enough to route stability that a small second seed pack is justified.
- The next required manual step is to author masks for the six IDs in the new v2 seed manifest, then resume the standard `Mask QA -> Dataset build -> Training -> Evaluation` loop.

### Experiment 2026-04-09E - Generate A Human-Usable V2 Annotation Pack

Hypothesis:
If the new v2 seed batch is going to be hand-labeled, the project should provide a ready-made annotation pack with per-pair `before` images and side-by-side context sheets so manual mask work does not depend on browsing `raw/` directly.

Change made:

- Reused the dataset role to generate a second annotation pack from the v2 seed manifest.
- Reused the existing pack-generation flow instead of inventing a new layout.
- Generated the pack under [`dataset/annotation_packs/masked_cuticle_cleanup_v2_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v2_seed).

Result:

- The v2 annotation pack now exists and is ready for manual use.
- Each pair has:
  - `before.png`
  - `after.png`
  - `pair_<id>_sheet.png` as a 3-panel `before / after / bootstrap overlay` sheet
- The pack also includes:
  - `bootstrap_masks/`
  - `bootstrap_overlays/`
  - `README.md`
  - `summary.json`
- A matching pack-specific manifest was also written to [`dataset/annotations/masked_cuticle_cleanup_v2_seed_pack_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v2_seed_pack_manifest.json).

Conclusion:

- The next manual annotation step no longer requires opening `raw/` directly.
- The user can now take `before` images and 3-panel sheets directly from the v2 pack and draw the final masks into the v2 mask directory.

### Experiment 2026-04-13A - QA Review The First V2 Seed Masks

Hypothesis:
If the first v2 seed masks stay within the same `proximal_nail_boundary_refinement` locality rules as the v1 full-12 set, then most of the batch should be promotable with only limited correction instead of broad redraws.

Change made:

- Re-read the project memory and confirmed that the current single goal is `Mask QA` for the newly uploaded v2 seed masks.
- Reused a scoped mask-QA pass on:
  - `pair_0064`
  - `pair_0118`
  - `pair_0122`
  - `pair_0153`
  - `pair_0154`
  - `pair_0190`
- Checked each sample against:
  - the authored mask in [`dataset/annotations/masks/masked_cuticle_cleanup_v2_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v2_seed)
  - the raw `before/after` pair
  - the v2 annotation-pack sheets

Result:

- `4 / 6` masks are already promotable:
  - `pair_0118`
  - `pair_0122`
  - `pair_0153`
  - `pair_0190`
- `2 / 6` masks need only small tightening rather than redraw:
  - `pair_0064`: thumb-side band is slightly too wide and eats into more nail plate than needed
  - `pair_0154`: thumb-side / lower-edge coverage is slightly too broad and should be tightened toward the true local posterior-edge band
- No sample needs a full redraw.
- Objective authored-mask ratios remain local:
  - `pair_0064`: `0.0386`
  - `pair_0118`: `0.0509`
  - `pair_0122`: `0.0403`
  - `pair_0153`: `0.0261`
  - `pair_0154`: `0.0546`
  - `pair_0190`: `0.0259`

Conclusion:

- The v2 seed batch is healthy overall; the new blocker is not broad semantic failure, only two narrow mask-tightening fixes.
- Dataset promotion should wait for the small fixes on `pair_0064` and `pair_0154` so the v2 batch enters with a clean, consistent locality standard.

### Experiment 2026-04-14A - Re-Review The Two V2 Seed Micro-Fix Masks

Hypothesis:
If `pair_0064` and `pair_0154` are tightened around the previously flagged thumb-side / lower-edge spill, the whole v2 seed batch should become promotable without any redraws.

Change made:

- Re-read project memory before reopening QA.
- Re-checked the updated authored masks for:
  - `pair_0064`
  - `pair_0154`
- Compared the revised masks against the v2 pack `before / after / bootstrap` sheets and direct overlay-on-before visualizations.

Result:

- `pair_0064` now passes:
  - authored-mask ratio tightened from `0.0386` to `0.0274`
  - thumb-side band now reads as a local proximal-boundary band rather than a broader nail-plate carve-out
- `pair_0154` now passes:
  - authored-mask ratio tightened from `0.0546` to `0.0417`
  - thumb-side / lower-edge coverage is now narrow enough to fit the intended local posterior-edge refinement task
- No remaining v2 seed sample needs redraw or additional micro-adjustment.

Conclusion:

- The full v2 seed batch now passes semantic QA.
- The next blocker is no longer mask authoring; it is promoting these six masks into the next approved manifest and rebuilding the masked dataset.

### Experiment 2026-04-14B - Promote The Full V2 Seed Batch And Rebuild The Masked Dataset

Hypothesis:
If the full v2 seed batch is promoted without changing task definition or training hyperparameters, the resulting masked dataset should remain local and color-aligned enough to enter the next training-preparation step.

Change made:

- Reused the dataset role to merge the passed v2 seed masks into a new approved manifest at [`dataset/annotations/masked_cuticle_cleanup_v2_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v2_approved_manifest.json).
- Rebuilt the masked dataset into [`dataset/masked_inpaint_cuticle_cleanup_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v2).
- Reused the evaluation role to compare `v1 -> v2` build summaries before changing anything training-side.

Result:

- The promoted v2 dataset now contains:
  - train: `13`
  - val: `5`
- Data-layer summary remains controlled:
  - train `mean_mask_ratio`: `0.0587`
  - val `mean_mask_ratio`: `0.0391`
  - train `mean_final_luma_delta`: `0.00165`
  - val `mean_final_luma_delta`: `0.00142`
- Compared with `v1`, the new dataset is a moderate expansion rather than a distribution break:
  - masks are still local
  - color alignment still suppresses most global luma drift
  - the main yellow-flag samples are higher raw-luma pairs like `pair_0153`, `pair_0154`, and `pair_0118`, but they do not become red flags after alignment

Conclusion:

- The v2 data layer is safe enough to enter the next training-preparation step.
- The next useful question is no longer “can it build,” but “does the trainer still behave normally when the dataset variable alone changes from `v1` to `v2`?”

### Experiment 2026-04-14C - Run A First Local Smoke On The Rebuilt V2 Dataset

Hypothesis:
If the v2 expansion is truly data-safe, the current masked trainer should still launch and complete a low-cost local smoke run without any code or data-format regressions.

Change made:

- Reused the training role to run the existing local smoke entrypoint against [`dataset/masked_inpaint_cuticle_cleanup_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v2).

Result:

- The v2 smoke run completed `4/4` steps under:
  - [`/tmp/masked_inpaint_lora_cuticle_cleanup_v2_local_smoke`](/tmp/masked_inpaint_lora_cuticle_cleanup_v2_local_smoke)
- It wrote all expected artifacts:
  - `metrics.jsonl`
  - `training_config.json`
  - `lora_checkpoints/pytorch_lora_weights_step_000004.safetensors`
  - `previews/preview_step_000004.png`
- The observed losses were finite and trainer behavior was normal.
- The blocker remains local CPU speed, not dataset validity or trainer correctness.

Conclusion:

- Changing only the dataset from `v1` to `v2` does not break masked training startup.
- The route has now passed data promotion plus smoke-scale training validation on the new v2 expansion.

### Experiment 2026-04-14D - Run A Short 10-Step Local Dry-Run On The Rebuilt V2 Dataset

Hypothesis:
If the rebuilt v2 dataset is truly stable and not just smoke-safe, it should also complete a short 10-step local dry-run with normal artifact writing and finite losses under the unchanged masked training setup.

Change made:

- Reused the training role to run a `10-step` low-cost local dry-run on [`dataset/masked_inpaint_cuticle_cleanup_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v2) without changing the main training variables.

Result:

- The short dry-run completed `10/10` steps under:
  - [`/tmp/masked_inpaint_lora_cuticle_cleanup_v2_step10_local`](/tmp/masked_inpaint_lora_cuticle_cleanup_v2_step10_local)
- It wrote all expected artifacts:
  - `metrics.jsonl`
  - `training_config.json`
  - `lora_checkpoints/pytorch_lora_weights_step_000010.safetensors`
  - `previews/preview_step_000010.png`
- Losses remained finite throughout the run.
- The main limitation is still local CPU speed, not a dataset or trainer regression.

Conclusion:

- The v2 dataset has now cleared both smoke-scale and short dry-run training checks.
- The next useful move is no longer local correctness validation; it is preparing or running a faster-hardware dataset-only continuation on the v2 dataset.

### Experiment 2026-04-15A - Archive And Inspect The V2 Dataset-Only Colab Run

Hypothesis:
If the v2-expanded masked dataset is a real improvement over the current masked reference, the first full Colab run on that dataset should at least remain training-stable and produce checkpoints worth comparing under the same local validation protocol.

Change made:

- Archived the user-provided Colab zip:
  - raw zip: [`archive/2026-04-06_user_result_zips/nail-retouch-masked-cuticle-cleanup-v2-dataset-only-outputs-20260415T011155Z-3-001.zip`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/nail-retouch-masked-cuticle-cleanup-v2-dataset-only-outputs-20260415T011155Z-3-001.zip)
  - extracted run: [`outputs/masked_inpaint_colab_runs/v2_dataset_only_run_2026-04-14_step150/nail-retouch-masked-cuticle-cleanup-v2-dataset-only-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/v2_dataset_only_run_2026-04-14_step150/nail-retouch-masked-cuticle-cleanup-v2-dataset-only-outputs)
- Reused the evaluation role to analyze run integrity, loss trends, and candidate checkpoints before reopening local validation.

Result:

- The run is structurally complete:
  - checkpoints at `25 / 50 / 75 / 100 / 125 / 150`
  - previews at the same steps
  - `metrics.jsonl` and `training_config.json` are present
- The run is a true dataset-only continuation:
  - dataset changed to `dataset/masked_inpaint_cuticle_cleanup_v2`
  - core masked training variables stayed aligned with the current masked reference
- Training-side interpretation is currently:
  - healthy and non-collapsed
  - but not obviously stronger than the current `lambda_color=1.0` reference from loss trends alone
- Most relevant candidate checkpoints for direct validation:
  - `step150`
  - `step100`
  - `step125`

Conclusion:

- The v2 dataset-only run is worth validating, but training-side evidence alone does not justify promoting it yet.
- The next required evidence is same-protocol local validation against the current masked reference.

### Experiment 2026-04-15B - Compare The Archived V2 Dataset-Only Run Against The Current Masked Reference

Hypothesis:
If the promoted v2 dataset is already teaching meaningfully better local edits, then at least one of the archived `step100 / step125 / step150` checkpoints should beat the current masked reference on the same patched 4-sample local validation protocol.

Change made:

- Reused the training role to finish same-protocol local validation on the archived v2 dataset-only checkpoints:
  - `step100`
  - `step125`
  - `step150`
- Reused the evaluation role to compare those summaries directly against the current masked reference:
  - [`outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors)
- Filled a small validation gap by re-running `pair_0050` for v2 `step150`, so the `step150` comparison also covers the same 4 validation pairs as `step100`, `step125`, and the current reference.

Result:

- Current masked reference (`v1 full12 lambda_color step150`) on the patched 4-sample validation protocol:
  - `mean_masked_l1_to_target = 0.0653309`
  - `mean_masked_delta_e_to_target = 8.52669`
  - `mean_unmasked_l1_to_input = 0.00560649`
  - `mean_unmasked_delta_e_to_input = 1.13071`
  - `mean_border_l1_to_target = 0.0361802`
- Archived v2 dataset-only checkpoints:
  - `step100`
    - `mean_masked_l1_to_target = 0.0656145`
    - `mean_masked_delta_e_to_target = 8.60553`
    - `mean_unmasked_l1_to_input = 0.00565653`
    - `mean_unmasked_delta_e_to_input = 1.13780`
    - `mean_border_l1_to_target = 0.0361691`
  - `step125`
    - `mean_masked_l1_to_target = 0.0654315`
    - `mean_masked_delta_e_to_target = 8.55559`
    - `mean_unmasked_l1_to_input = 0.00563620`
    - `mean_unmasked_delta_e_to_input = 1.13344`
    - `mean_border_l1_to_target = 0.0361127`
  - `step150`
    - `mean_masked_l1_to_target = 0.0653598`
    - `mean_masked_delta_e_to_target = 8.52575`
    - `mean_unmasked_l1_to_input = 0.00561866`
    - `mean_unmasked_delta_e_to_input = 1.13017`
    - `mean_border_l1_to_target = 0.0361426`
- Readout from the reused Evaluation Agent:
  - `v2 dataset-only` is `flat`, not a clear forward or backward move
  - `step150` is the best v2 checkpoint, but only ties the current reference within noise
  - `step100` and `step125` both remain slightly behind `step150`

Conclusion:

- Expanding the masked dataset from `v1` to `v2` looks safe, but this first dataset-only continuation does not produce a strong enough gain to replace the current masked reference.
- Keep the current masked reference checkpoint unchanged.
- The next useful project lever is broader annotation coverage, not another near-neighbor dataset-only continuation on the same setup.

### Experiment 2026-04-15C - Prepare The Next Conservative V3 Explicit-Mask Seed Pack

Hypothesis:
If the first `v2 dataset-only` continuation is only flat, the most useful next step is to expand annotation coverage again with another conservative seed pack instead of reopening near-neighbor training tweaks.

Change made:

- Reused the dataset role to propose and then materialize the next explicit-mask batch:
  - manifest: [`dataset/annotations/masked_cuticle_cleanup_v3_seed_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v3_seed_manifest.json)
  - pack manifest: [`dataset/annotations/masked_cuticle_cleanup_v3_seed_pack_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v3_seed_pack_manifest.json)
  - annotation pack: [`dataset/annotation_packs/masked_cuticle_cleanup_v3_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed)
  - final mask scaffold: [`dataset/annotations/masks/masked_cuticle_cleanup_v3_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v3_seed)
- Kept the task fixed at `proximal_nail_boundary_refinement` rather than splitting taxonomy.

Result:

- The new conservative v3 seed batch is:
  - train:
    - `pair_0022`
    - `pair_0028`
    - `pair_0035`
    - `pair_0043`
    - `pair_0066`
    - `pair_0071`
    - `pair_0073`
  - val:
    - `pair_0120`
    - `pair_0208`
- The human-usable annotation pack now includes, for every pair:
  - `before.png`
  - `after.png`
  - `<pair_id>_sheet.png`
- The generated bootstrap stats show that color alignment remains controlled, but several draft masks are visibly broader than the safer v2 tranche:
  - `pair_0043`: `bootstrap_mask_ratio 0.3633`
  - `pair_0066`: `0.3258`
  - `pair_0071`: `0.3136`
  - `pair_0073`: `0.2777`
  - `pair_0120`: `0.3851`
  - `pair_0208`: `0.4796`

Conclusion:

- The next annotation batch is ready for human labeling, but the bootstrap masks should be treated strictly as drafts and will need careful narrowing during authoring.
- The project is now at a real manual boundary again: the next useful step is to draw the v3 seed masks and then resume `Mask QA -> approved-manifest promotion -> dataset rebuild`.

### Experiment 2026-04-15D - Inspect `pair_0120` For Pairwise Geometry Risk Before V3 Annotation

Hypothesis:
If `pair_0120` carries noticeable whole-nail / whole-finger geometric offset between `before` and `after`, then it should not be treated like a normal local `proximal_nail_boundary_refinement` annotation target inside the v3 seed batch.

Change made:

- Manually reviewed [`dataset/annotation_packs/masked_cuticle_cleanup_v3_seed/pairs/pair_0120/pair_0120_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed/pairs/pair_0120/pair_0120_sheet.png) against the current v3 seed criteria.

Result:

- `pair_0120` shows visible whole-nail / whole-finger positional offset between `before` and `after`, not just local cuticle or posterior-edge cleanup.
- The bootstrap overlay spreads broadly across nail plates and finger regions in a way that is consistent with pairwise geometric mismatch, not only with benign local cleanup.
- This makes `pair_0120` materially riskier than a normal v3 seed sample for local-mask authoring.

Conclusion:

- Keep `pair_0120` in the v3 pack only as a high-risk candidate, not as a routine first-pass annotation target.
- If it is annotated at all, the mask should be restricted to the small stable overlap region around the true proximal boundary; do not chase the shifted nail silhouette.
- Operationally, it should be drawn after the more stable v3 samples, or deferred if a narrow local mask cannot be authored cleanly.

### Experiment 2026-04-15E - Prioritize The V3 Seed Pack For Human Annotation

Hypothesis:
If the v3 seed batch is split into a stable-first drawing order before annotation starts, manual effort will stay focused on clean local-boundary samples and avoid early time loss on geometry-mismatch pairs.

Change made:

- Reviewed the v3 pack structure and bootstrap statistics.
- Manually spot-checked representative v3 sheets, including:
  - [`pair_0120_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed/pairs/pair_0120/pair_0120_sheet.png)
  - [`pair_0028_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed/pairs/pair_0028/pair_0028_sheet.png)
  - [`pair_0208_sheet.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed/pairs/pair_0208/pair_0208_sheet.png)
- Combined those observations with the v3 bootstrap-mask ratios to set a practical annotation order.

Result:

- Recommended `v3` annotation order now splits into three tiers:
  - `prioritize first`
    - `pair_0028`
    - `pair_0035`
    - `pair_0022`
  - `draw after the first stable tranche`
    - `pair_0043`
    - `pair_0066`
    - `pair_0071`
    - `pair_0073`
  - `high-risk / defer to last`
    - `pair_0120`
    - `pair_0208`
- Operational readout:
  - `pair_0028` and `pair_0035` look like the cleanest local-boundary candidates in the pack.
  - `pair_0043`, `pair_0066`, `pair_0071`, and `pair_0073` appear annotatable, but their bootstrap drafts are broad enough that they should not be used as-is.
  - `pair_0120` shows visible before/after geometric offset.
  - `pair_0208` also looks riskier than the main body of the batch because of broad drift-like draft coverage plus decoration, so it should be treated as a last-pass candidate rather than an early one.

Conclusion:

- The v3 seed pack is still usable, but it should be annotated in a staged order instead of treated as one uniform batch.
- The next manual pass should start with `pair_0028`, `pair_0035`, and `pair_0022`, then move outward only if those masks stay clean under QA.

### Experiment 2026-04-16A - Run The First Mask QA Pass On The Uploaded V3 Seed Masks

Hypothesis:
If the stable-first v3 annotation order was chosen correctly, then the first uploaded v3 masks should mostly pass QA, with only the riskier tail of the batch needing tightening or deferral.

Change made:

- Reused the dataset / mask-QA role to review the uploaded final masks under [`dataset/annotations/masks/masked_cuticle_cleanup_v3_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v3_seed).
- Confirmed which files are actually present before judging the batch:
  - `pair_0022`
  - `pair_0028`
  - `pair_0035`
  - `pair_0043`
  - `pair_0066`
  - `pair_0071`
  - `pair_0073`
- Also checked basic file-format stats:
  - all uploaded masks are near-binary with pixel values `[0, 254, 255]`
  - ratios remain local-scale, roughly `0.0247 - 0.0810`

Result:

- First-pass `v3` QA results:
  - pass:
    - `pair_0022`
    - `pair_0028`
    - `pair_0035`
    - `pair_0043`
    - `pair_0071`
  - needs micro-adjust:
    - `pair_0066`
  - high-risk / defer:
    - `pair_0073`
- `pair_0066` issue:
  - some multi-finger coverage bands connect too broadly and should be tightened into narrower local posterior-edge / cuticle bands
- `pair_0073` issue:
  - coverage reads too broad and too close to a stronger shape-edit / large contour edit, beyond a comfortable first-pass local-boundary sample
- `pair_0120` and `pair_0208` still have no final uploaded masks in this round, and both remain high-risk based on earlier pack review

Conclusion:

- The stable-first ordering mostly worked: five of the first seven uploaded v3 masks already pass.
- The next blocker is not a broad redraw of the batch; it is a focused cleanup of `pair_0066` plus a decision on whether `pair_0073` should be narrowed aggressively or postponed.

### Experiment 2026-04-16B - Re-Review `pair_0066` And Finalize `pair_0073` Status

Hypothesis:
If `pair_0066` is tightened back into narrow local posterior-edge / cuticle bands, it should join the approved v3 tranche, while `pair_0073` can be cleanly deferred if its underlying edit semantics are too shape-driven to narrow honestly.

Change made:

- Re-reviewed the updated authored mask at [`dataset/annotations/masks/masked_cuticle_cleanup_v3_seed/pair_0066.png`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_cuticle_cleanup_v3_seed/pair_0066.png).
- Used the user-provided context for `pair_0073`: the retouch itself stretches the posterior edge to improve the nail shape, so the sample cannot be honestly narrowed into the current local-boundary task without discarding the true edit.

Result:

- `pair_0066` now passes:
  - authored mask ratio tightened to `0.0399`
  - the previous over-connected multi-finger bands have been reduced back into local boundary-shaped regions
- `pair_0073` should be deferred from the current v3 promotion tranche:
  - its underlying edit is not just cleanup; it materially reshapes the posterior edge to improve nail shape
  - that makes it a poor fit for the current conservative `proximal_nail_boundary_refinement` tranche

Conclusion:

- Add `pair_0066` to the pass set for the current v3 batch.
- Do not keep trying to squeeze `pair_0073` into the current tranche; defer it instead of teaching a mislabeled broader shape-edit under the local-boundary task.

### Experiment 2026-04-16C - Promote The Current Passed V3 Subset Into A Buildable Dataset

Hypothesis:
If the currently passed v3 masks are promoted as a partial approved subset instead of waiting for every high-risk sample, the resulting masked dataset should stay local and clean enough to enter the next training-preparation step.

Change made:

- Reused the dataset role to create a partial approved-subset manifest:
  - [`dataset/annotations/masked_cuticle_cleanup_v3_approved_subset_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v3_approved_subset_manifest.json)
- Rebuilt the masked dataset into:
  - [`dataset/masked_inpaint_cuticle_cleanup_v3`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v3)
- Promoted only the currently passed v3 samples:
  - train:
    - `pair_0022`
    - `pair_0028`
    - `pair_0035`
    - `pair_0066`
    - `pair_0071`
  - val:
    - `pair_0043`

Result:

- The partial v3 dataset now contains:
  - `train_count = 5`
  - `val_count = 1`
- Data-layer summary remains controlled:
  - train `mean_mask_ratio = 0.0373`
  - val `mean_mask_ratio = 0.0810`
  - train `mean_raw_luma_delta = 0.0646`
  - val `mean_raw_luma_delta = 0.00119`
  - train `mean_global_aligned_luma_delta = 0.001105`
  - val `mean_global_aligned_luma_delta = 0.00000995`
  - train `mean_final_luma_delta = 0.00165`
  - val `mean_final_luma_delta = -0.000719`

Conclusion:

- The current passed v3 subset is already coherent enough to function as a real masked dataset, even before the high-risk tail is resolved.
- The next useful question is no longer whether the subset can build; it is whether the trainer still behaves normally on this partial v3 expansion.

### Experiment 2026-04-16D - Run The Minimal Training Closure On The Partial V3 Dataset

Hypothesis:
If the partial v3 approved subset is genuinely data-safe, then the current masked trainer should still complete both a low-cost smoke and a short 10-step dry-run without any new regressions.

Change made:

- Reused the training role to run a low-cost local smoke against [`dataset/masked_inpaint_cuticle_cleanup_v3`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v3):
  - output: [`/tmp/masked_inpaint_lora_cuticle_cleanup_v3_local_smoke`](/tmp/masked_inpaint_lora_cuticle_cleanup_v3_local_smoke)
- After the smoke passed, re-used the training role again to run a 10-step short dry-run:
  - output: [`/tmp/masked_inpaint_lora_cuticle_cleanup_v3_step10_local`](/tmp/masked_inpaint_lora_cuticle_cleanup_v3_step10_local)

Result:

- Local smoke completed `4/4` steps and wrote:
  - `metrics.jsonl`
  - `training_config.json`
  - `lora_checkpoints/pytorch_lora_weights_step_000004.safetensors`
  - `previews/preview_step_000004.png`
- Short dry-run completed `10/10` steps and wrote:
  - `metrics.jsonl`
  - `training_config.json`
  - `lora_checkpoints/pytorch_lora_weights_step_000010.safetensors`
  - `previews/preview_step_000010.png`
- Losses stayed finite throughout both runs.
- The effective blocker is still not trainer correctness; it is experiment budget / hardware.

Conclusion:

- The current passed v3 subset has now cleared both data-layer promotion and minimal training closure.
- This partial subset is ready for a clearer Colab handoff if we want to continue before the remaining high-risk masks are resolved.

### Experiment 2026-04-16E - Prepare A Dataset-Only Colab Handoff For The Partial V3 Subset

Hypothesis:
If the current passed v3 subset is already data-safe and trainer-safe locally, then the next useful automation step is a dataset-only Colab handoff that swaps only the masked training input to the partial v3 dataset.

Change made:

- Reused the training role to prepare a dedicated Colab config:
  - [`colab/masked_inpaint_cuticle_cleanup_v3_dataset_only_v1.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_cuticle_cleanup_v3_dataset_only_v1.yaml)
- Updated the existing masked notebook entrypoint:
  - [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb)
- Kept the handoff single-variable by changing only dataset-related fields:
  - `dataset_dir -> dataset/masked_inpaint_cuticle_cleanup_v3`
  - `manifest_path -> dataset/annotations/masked_cuticle_cleanup_v3_approved_subset_manifest.json`
  - `drive_dataset_dir -> /content/drive/MyDrive/masked_inpaint_cuticle_cleanup_v3`

Result:

- The partial v3 subset now has a dedicated Colab handoff.
- Notebook default now points at the v3 dataset-only config instead of the previous masked handoff.
- The intended Drive dataset path for the handoff is:
  - `/content/drive/MyDrive/masked_inpaint_cuticle_cleanup_v3`
- Core training variables remain unchanged:
  - `resolution=512`
  - `max_train_steps=150`
  - `checkpointing_steps=25`
  - `preview_steps=25`
  - `rank=4`
  - `lambda_identity=5.0`
  - `lambda_color=1.0`

Conclusion:

- The current passed v3 subset is now ready for a more informative GPU-side dataset-only continuation without waiting for the full v3 tail to resolve.
- The project is at a clean manual boundary again: either run the v3 partial-subset Colab handoff, or continue manual mask work on the deferred high-risk samples.
