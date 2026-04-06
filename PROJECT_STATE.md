# Project State

Last updated: 2026-04-06

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
