# Handoff

Last updated: 2026-04-06

## What Was Tested

- Read the existing repo state and reconstructed the missing persistent memory files.
- Reviewed the current guarded training config and the historical old-route expansion config.
- Inspected historical validation sheets for `pair_0009`.
- Added and ran [`src/data/audit_paired_drift.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/audit_paired_drift.py) across the built paired datasets.
- Added and smoke-tested [`src/data/build_masked_inpaint_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_masked_inpaint_dataset.py).
- Added and statically verified [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py).
- Added and statically verified masked-route inference and validation entrypoints.
- Generated the first explicit-mask annotation pack and scaffold manifest for `cuticle_cleanup`.
- Reframed the first explicit-mask subset as `proximal_nail_boundary_refinement` after reviewing how often the posterior edge is locally reshaped in the targets.
- Built and re-audited the dataset-only filtered legacy baseline `dataset/paired_edit_core_v2`.
- Built the four-mask explicit smoke subset and launched a real masked training dry-run on it.
- Added a dedicated local masked smoke wrapper so future CPU sanity checks do not need to reuse the slower 512-resolution command.
- Ran the local masked smoke wrapper end-to-end, confirmed that it completes in offline mode against the cached inpainting snapshot, and recorded the produced artifacts plus wall-clock.
- Updated the local smoke wrapper to auto-detect a cached inpainting snapshot and reran the default command successfully without manually overriding `PRETRAINED_MODEL`.
- Reviewed the newly drawn masks for `pair_0005`, `pair_0032`, `pair_0054`, and `pair_0057`.
- Re-reviewed the fixed `pair_0005` and `pair_0057` masks after the requested cleanup pass.
- Built the first real approved-subset masked dataset and ran both a 4-step smoke and a 10-step local dry-run on it.
- Reviewed the newly drawn masks for `pair_0063` and `pair_0070`, approved both, updated the approved manifest, and rebuilt the masked dataset from the expanded approved subset.
- Ran a new 10-step local masked dry-run on the expanded 10-sample approved subset and compared its metrics against the earlier 8-sample dry-run.
- Reviewed the newly drawn masks for `pair_0047` and `pair_0050`; both now pass semantic QA for `proximal_nail_boundary_refinement`.
- Promoted `pair_0047` and `pair_0050` into the approved manifest, rebuilt the full 12-sample masked dataset, and verified local training at `4`, `10`, and `25` steps.
- Added a masked Colab training notebook/config for the full approved 12-sample dataset.
- Hardened the masked Colab notebook so missing Drive raw paths now fall back to a cached dataset or auto-discovered raw-pair root before failing.
- Archived the first full-12 masked Colab output zip locally and analyzed its metrics, previews, and checkpoint progression.
- Restored the local inpainting base snapshot and advanced masked validation to the first real local validation artifacts.
- Recovered the second trustworthy local validation point by abandoning repeated safety-blackout retries on `pair_0040` and moving to `pair_0047`.
- Completed the direct `step150` vs `step200` comparison on the clean `pair_0047` validation sample.
- Finished classification of the last remaining full-12 val sample `pair_0050`.
- Launched the local `legacy core_v2` retrain far enough to classify the blocker, then prepared a dedicated Colab handoff for that same dataset-only experiment.
- Fixed a follow-up Colab failure where the handoff notebook was newer than the branch-visible paired trainer script.

## What Worked

- The audit script reliably surfaced the same failure mode the manual pruning notes describe: global lift and color drift.
- Pruning `phase1_expand_batch1` down to `phase1_expand_batch1_pruned` improved the mean drift score from `0.2163` to `0.1896`.
- The audit made it clear that `core_v1` itself is still biased, which explains why simply keeping the dataset small has not fully solved the problem.
- The new masked dataset builder successfully exports `input`, `mask`, and `target_local`.
- Pairwise color alignment materially reduced luma drift on smoke-test validation pairs before training:
  - `pair_0009`: `+0.0387 -> +0.0022`
  - `pair_0040`: `+0.0514 -> -0.0021`
- The repository now has a dedicated masked inpainting training entrypoint instead of only the legacy full-image trainer.
- The repository now also has:
  - [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py)
  - [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py)
- The explicit-mask annotation pack exists under [`dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v1/README.md).
- The first approved masks now have a more accurate interpretation: they are local posterior-edge refinement masks, not pure dead-skin cleanup masks.
- Four explicit masks are now approved and usable:
  - `pair_0015`
  - `pair_0018`
  - `pair_0009`
  - `pair_0040`
- The explicit smoke dataset exists under [`dataset/masked_inpaint_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1_smoke).
- The masked training route successfully initialized and executed real training steps on the explicit smoke dataset, writing [`metrics.jsonl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke/metrics.jsonl) and [`training_config.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke/training_config.json).
- The legacy paired dataset-only filter worked as intended:
  - `core_v1` mean drift score: `0.2121`
  - `core_v2` mean drift score: `0.1377`
- The repository now has a single-command local masked smoke entrypoint in [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh).
- The local smoke wrapper now has one confirmed complete run under [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke):
  - `4/4` steps completed
  - wall-clock about `49s`
  - checkpoint written at step `4`
  - preview written at step `4`
- After the cache-fallback patch, the default smoke command also completed successfully into [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_local_smoke_verify) without a manual model override.
- The next explicit-mask batch is partially usable already:
  - approved: `pair_0005`, `pair_0032`, `pair_0054`, `pair_0057`
- The next explicit-mask expansion tranche has now also been promoted:
  - approved: `pair_0063`, `pair_0070`
- The current approved-subset promotion target now exists as [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json) and builds into [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1).
- The approved-subset masked dataset has now completed:
  - a 4-step low-cost local smoke
  - a 10-step low-cost local dry-run
- The approved explicit masked dataset now contains 10 samples total:
  - train: `pair_0005`, `pair_0015`, `pair_0018`, `pair_0032`, `pair_0054`, `pair_0057`, `pair_0063`, `pair_0070`
  - val: `pair_0009`, `pair_0040`
- The rebuilt 10-sample dataset still looks healthy at the data layer:
  - train mean mask ratio: `0.0720`
  - train mean final luma delta: `+0.0009`
  - val mean final luma delta: `+0.0003`
- The current approved masked dataset now extends to the full first seed pack:
  - train: `pair_0005`, `pair_0015`, `pair_0018`, `pair_0032`, `pair_0054`, `pair_0057`, `pair_0063`, `pair_0070`
  - val: `pair_0009`, `pair_0040`, `pair_0047`, `pair_0050`
- The rebuilt full-12 dataset still looks healthy at the data layer:
  - train mean mask ratio: `0.0720`
  - train mean final luma delta: `+0.0009`
  - val mean mask ratio: `0.0421`
  - val mean final luma delta: `+0.0016`
- The full-12 dataset now has three local training checks:
  - 4-step smoke: [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_smoke4)
  - 10-step dry-run: [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step10)
  - 25-step dry-run: [`/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25`](/tmp/masked_inpaint_lora_cuticle_cleanup_v1_full12_step25)
- The repository now also has a masked Colab entrypoint for longer GPU-side runs:
- The repository now also has a masked Colab entrypoint for longer GPU-side runs:
  - [`colab/masked_inpaint_full12_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_config.yaml)
  - [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb)
- That notebook now prepares data with a fallback chain:
  - configured `drive_raw_dir`
  - configured `drive_dataset_dir`
  - auto-discovered raw pair root under `/content/drive/MyDrive`
  - clear checked-path error if none of the above exist
- The expanded approved subset has now also completed its own dedicated 10-step local dry-run under [`outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_lora_cuticle_cleanup_v1_10sample_step10_local):
  - runtime about `2m23s`
  - checkpoint written at step `10`
  - preview written at step `10`
  - mean loss `6.3159`
  - mean identity loss `0.0248`
- Compared with the earlier 8-sample step-10 run, the 10-sample run is slightly noisier but still clearly healthy:
  - earlier 8-sample mean loss: `5.4596`
  - earlier 8-sample mean identity loss: `0.0231`

## What Failed

- Historical `phase1_expand_batch1` training outputs collapsed badly:
  - `model_251`: white blowout
  - `model_501`: magenta/orange blur with heavy texture loss
- The assumption that `core_v1` is already safe was incorrect. It still has notable positive luma drift and several high-risk pairs.
- Difference-derived masks are too broad on at least some train samples:
  - `pair_0005`: `mask_ratio 0.4120`
- That means the bootstrap diff-mask path is not strong enough to become the final supervision source.
- The generated explicit-mask pack confirmed the same issue at subset level:
  - very broad draft masks include `pair_0050` (`0.6095`), `pair_0063` (`0.5152`), `pair_0070` (`0.4224`), `pair_0005` (`0.4016`), and `pair_0047` (`0.3933`)
- The original `cuticle_cleanup` label was too narrow for the actual dataset. Many samples legitimately include a narrow posterior-edge movement band.
- A full 10-step local dry-run at `512` resolution is very slow on CPU. The route is no longer blocked by initialization or formatting, but by runtime cost.
- The first unpatched default local smoke invocation failed on `huggingface.co` resolution before training started; this was an environment issue and has now been mitigated in the wrapper.
- The fixed `pair_0005` and `pair_0057` masks are now semantically acceptable, but the revised exports are antialiased grayscale masks rather than strict binary masks.
- The original “missing masks” blocker has been reduced further:
  - promoted: `pair_0063`, `pair_0070`
  - newly approved and ready for promotion review: `pair_0047`, `pair_0050`
- Manual mask work is no longer the blocker on the first seed pack. The next masked step is no longer promotion either; it is deciding whether to keep full-12 as the default long-run subset and where to run the next larger experiment.
- The first real full-12 Colab training result is now locally archived, but local true validation still stops at inference-time base-model availability rather than model quality evidence.
- Local true validation no longer stops at missing base-model files; it now stops at validation coverage quality because one sample (`pair_0040`) is being blacked out by the safety checker.
- Local true validation now has two trustworthy anchor samples, and the remaining limitation is sample coverage / ranking confidence rather than route viability.

## Next Best Experiment

Run a dataset-only retrain on `dataset/paired_edit_core_v2` or validate the archived full-12 masked Colab checkpoints in an environment with a complete inpainting base.

Hypothesis:
Two narrow next moves are now available:

- Legacy control: if the highest-drift core pairs are removed while the guarded training config stays fixed, whole-image lift and color cast should decrease without making the result interpretation ambiguous.
- Masked route: if the first explicit masks are kept local and the dry-run completes cleanly, the new route should let us measure mask-inside edit quality separately from mask-outside preservation for local posterior-edge refinement.
- Mask QA: the next reviewed tranche is now fully approved, so the masked subset can grow beyond the four-mask smoke pool without waiting on more redraws for this batch.
- Approved-subset runtime: the 8-sample explicit subset is now strong enough for both local smoke and local 10-step dry-run checks, so the next masked bottleneck is subset size and hardware budget rather than immediate pipeline correctness.
- Approved-subset expansion: the explicit subset now supports an `8`-train / `2`-val approved dataset and completes its own dedicated 10-step dry-run, so it is now strong enough to serve as the first real masked training set without waiting on `pair_0047` / `pair_0050`.
- Full-seed-pack promotion: `pair_0047` and `pair_0050` are now in the approved manifest, the full 12-sample dataset rebuilds cleanly, and local smoke-scale training remains healthy after promotion.
- Colab handoff readiness: the repository now has a masked notebook/config in the same style as the historical paired-edit Colab files, so longer GPU-side runs no longer need notebook scaffolding work first.
- Colab handoff robustness: a missing fixed raw Drive path no longer immediately blocks startup; retry with the updated branch before manually rewriting notebook paths.
- Local smoke infrastructure: now that the wrapper has produced a checkpoint and preview in about `49s` and the default command auto-prefers the cached local base model, future masked code changes can be verified locally without blocking on the slower 512-resolution path.
- Archived masked baseline run: [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs) now contains:
  - `metrics.jsonl`
  - `training_config.json`
  - checkpoints at `50 / 100 / 150 / 200`
  - previews at `50 / 100 / 150 / 200`
- Archived run readout:
  - loss means improved by quarter: `4.6625 -> 3.5326 -> 2.6857 -> 2.2155`
  - `loss_identity` remained roughly stable around `~0.018`
  - `step200` is the current provisional best checkpoint
  - `step150` should be retained as the rollback / comparison candidate
- Local validation blocker after archiving:
  - no longer missing LoRA checkpoints
  - no longer missing masked dataset
  - no longer missing the local inpainting base snapshot
  - current blocker is validation coverage distortion from safety-checker blackouts on at least one sample
- First real local masked validation artifacts now exist:
  - [`step150 pair_0009 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_nocomposite/pytorch_lora_weights_step_000150/pair_0009_sheet.png)
  - [`step150 pair_0009 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_nocomposite/pytorch_lora_weights_step_000150/pair_0009_metrics.json)
  - [`step200 pair_0009 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0009_sheet.png)
  - [`step200 pair_0009 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0009_metrics.json)
- Current checkpoint readout after first real local validation:
  - `step150` slightly outperformed `step200` on `pair_0009`
  - the gap is very small, so both checkpoints should still be kept
  - `step150` is now the provisional preferred checkpoint
- New validation artifact caveat:
  - [`pair_0040 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_nocomposite/pytorch_lora_weights_step_000200/pair_0040_metrics.json) was contaminated by a safety-checker-triggered black output and should not be used for checkpoint ranking
- Seed-retry follow-up on `pair_0040`:
  - tried `step150` with seeds `2026`, `7`, and `42`
  - all three still blacked out via safety checker
  - treat `pair_0040` as safety-unstable for now
- Second trustworthy local validation point:
  - [`pair_0047 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047/pytorch_lora_weights_step_000150/pair_0047_sheet.png)
  - [`pair_0047 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047/pytorch_lora_weights_step_000150/pair_0047_metrics.json)
  - key metrics:
    - `masked_l1_to_target`: `0.1310`
    - `masked_delta_e_to_target`: `17.1773`
    - `unmasked_l1_to_input`: `0.0450`
    - `unmasked_delta_e_to_input`: `5.6020`
    - `border_l1_to_target`: `0.0956`
- Current trustworthy local validation anchors:
  - `pair_0009`
  - `pair_0047`
- Current checkpoint confidence:
  - `step150` is the preferred candidate
  - confidence is stronger than before because it now has two clean local validation points
  - the `pair_0047` direct checkpoint comparison is now complete
- `pair_0047` direct checkpoint comparison:
  - [`step200 pair_0047 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047_step200/pytorch_lora_weights_step_000200/pair_0047_sheet.png)
  - [`step200 pair_0047 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/seed_retry_0047_step200/pytorch_lora_weights_step_000200/pair_0047_metrics.json)
  - `step150` remained slightly ahead on `pair_0047`, though the gap stayed extremely small
- Current ranking summary from clean local samples:
  - `pair_0009`: `step150` slightly better than `step200`
  - `pair_0047`: `step150` slightly better than `step200`
  - practical conclusion: use `step150` as the default checkpoint from this archived run
- `pair_0050` final classification:
  - [`pair_0050 sheet`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/final_pair0050/pytorch_lora_weights_step_000150/pair_0050_sheet.png)
  - [`pair_0050 metrics`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/final_pair0050/pytorch_lora_weights_step_000150/pair_0050_metrics.json)
  - `step150` with seeds `2026` and `7` both blacked out via safety checker
  - operationally classify `pair_0050` as safety-unstable / unusable for local ranking
- Final local full-12 val-set classification:
  - trustworthy local ranking samples:
    - `pair_0009`
    - `pair_0047`
  - safety-unstable local validation samples:
    - `pair_0040`
    - `pair_0050`

Expected effect:
Less whitening and less blue/pink cast on easy validation pairs.

Potential side effect:
Removing too many core pairs may reduce shape diversity and underfit edge-cleanup behavior, so keep the first pass conservative.

## Approved Migration Direction

The repository should transition in phases:

1. Keep the immediate next legacy experiment narrow: `core_v2` retrain with no hyperparameter changes.
2. Treat the full 12-sample approved subset as the current masked promotion target, and plan the first more meaningful GPU-side masked runtime check on Colab or faster hardware.
3. Do not merge those two variables into the same experiment.
4. For the archived Colab result, keep the next masked validation single-variable: recover a second clean comparison point while leaving checkpoint, dataset, and prompt settings otherwise unchanged.

Concrete code migration targets:

- [`src/data/build_curated_paired_edit_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_curated_paired_edit_dataset.py)
  Add a new export mode for `mask` and `target_local`.
- [`src/data/build_masked_inpaint_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_masked_inpaint_dataset.py)
  This now exists as the dedicated masked-data route and should be the base for explicit-mask dataset creation.
- [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py)
  Treat as the old full-image route; do not keep layering complexity into it indefinitely.
- [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py)
  This now exists as the dedicated masked inpainting LoRA trainer.
- [`src/paired_edit/run_paired_edit_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/run_paired_edit_inference.py)
  Legacy baseline only. Prefer the new masked route for future local-edit experiments.
- [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py)
  New inpainting inference entrypoint with exact outside-mask compositing.
- [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py)
  New validation entrypoint with masked/unmasked split metrics.
- [`src/data/prepare_explicit_mask_subset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/prepare_explicit_mask_subset.py)
  Generates the seed annotation pack and manifest scaffold.

## Suggested Commands

Build and retrain the next legacy paired dataset:

```bash
python3 src/data/build_curated_paired_edit_dataset.py \
  --raw-dir raw \
  --manifest dataset/annotations/paired_edit_core_v2_manifest.json \
  --output-dir dataset/paired_edit_core_v2
```

Then run the first explicit-mask prep if you need to regenerate the annotation pack:

```bash
python3 src/data/prepare_explicit_mask_subset.py
```

Then, after masks exist, build the masked dataset and run a tiny dry-run:

```bash
python3 src/data/build_masked_inpaint_dataset.py \
  --raw-dir raw \
  --manifest dataset/annotations/masked_cuticle_cleanup_v1_smoke_manifest.json \
  --output-dir dataset/masked_inpaint_cuticle_cleanup_v1_smoke \
  --mask-mode explicit

python3 src/training/train_masked_inpaint_lora.py \
  --dataset_dir dataset/masked_inpaint_cuticle_cleanup_v1_smoke \
  --output_dir outputs/masked_inpaint_lora_cuticle_cleanup_v1_smoke \
  --resolution 512 \
  --rank 4 \
  --learning_rate 1e-5 \
  --max_train_steps 10 \
  --checkpointing_steps 10 \
  --preview_steps 10
```

## Notes For The Next Session

- Do not start by changing learning rate, rank, resolution, or loss weights.
- `core_v2` already exists; no more manifest work is needed before the next legacy retrain.
- Reuse the same baseline validation pairs so the before/after comparison stays interpretable.
- For the masked route, the current approved dataset is now the full first seed pack:
  - train: `pair_0005`, `pair_0015`, `pair_0018`, `pair_0032`, `pair_0054`, `pair_0057`, `pair_0063`, `pair_0070`
  - val: `pair_0009`, `pair_0040`, `pair_0047`, `pair_0050`
- The local machine can currently handle smoke-scale checks through `25` steps on this dataset, but longer or more quality-focused runs should start moving to Colab / faster hardware.
- Use the new masked Colab notebook/config when that handoff becomes worthwhile.
- The current archived masked baseline is now local and traceable; do not ask the user for the same output zip again unless a later run supersedes it.
- A cheap local smoke command now exists, has completed successfully, and now auto-prefers the cached inpainting snapshot in this environment; it is still strictly a plumbing check and should stay separate from any quality conclusions.
- If the next session wants real masked validation rather than another training run, prefer an environment that definitely has the full inpainting base available and reuse the archived `step150` / `step200` checkpoints first.
- The local environment now has the required inpainting base snapshot, so the next session can continue local masked validation without re-downloading the base model.
- Do not spend the next session re-trying `pair_0040` first; use it only if there is a specific safety-checker investigation. The normal next move is checkpoint comparison on `pair_0047` or expansion to another clean val sample.
- Do not spend the next session re-ranking `step150` and `step200` on the same existing clean points again. The checkpoint choice is stable enough; the real next decision is whether to expand validation coverage or move to the next experiment.
- Do not spend the next session retrying the current local val split blindly. That split is now fully classified; any further validation work should be a deliberate broader-clean-sample pass or a targeted safety-checker investigation.
- Local `core_v2` retrain status:
  - real launch path confirmed
  - blocked before `step 1`
  - blocker: missing `xformers` support on this machine, not dataset formatting
  - local output root only wrote [`/tmp/paired_edit_core_v2_retrain_local/run_config.json`](/tmp/paired_edit_core_v2_retrain_local/run_config.json)
- Dedicated `core_v2` Colab handoff files now exist:
  - [`colab/paired_edit_core_v2_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_core_v2_config.yaml)
  - [`colab/train_paired_edit_core_v2_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_paired_edit_core_v2_v1.ipynb)
- `core_v2` handoff principle:
  - only dataset / naming / output paths changed versus `core_v1`
  - guarded training variables stayed fixed
  - do not “fix” the local blocker by quietly changing `enable_xformers_memory_efficient_attention`
- Colab CLI mismatch postmortem:
  - the notebook/config were already emitting the intended guarded arguments
  - Colab rejected them because the active branch still exposed an older `src/paired_edit/train_supervised_retouch.py`
  - the fix was to push the updated trainer script to [`codex/masked-full12-colab`](https://github.com/Zifpen/nail-retouch-assistant/tree/codex/masked-full12-colab), not to weaken the notebook command
- Correct retry instruction after the trainer sync:
  - fresh-clone the branch again before rerunning the notebook:
    `cd /content && rm -rf nail-retouch-assistant && git clone -b codex/masked-full12-colab https://github.com/Zifpen/nail-retouch-assistant.git`
- New legacy `core_v2` blocker after the fresh clone and real Colab launch:
  - the run now gets through real initialization and completes `step 1`
  - it fails in the first `evaluate()` call, not at startup
  - the blocking guard is in [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py):
    - `change_ratio >= 0.60` raises `ValueError: Change mask covers too much of the image`
  - the offending eval sample is `pair_0050`
  - under trainer-equivalent `resize_256 + threshold=0.12 + dilate=3`, current `core_v2` test ratios are:
    - `pair_0009`: `0.1562`
    - `pair_0040`: `0.2229`
    - `pair_0047`: `0.3716`
    - `pair_0050`: `0.6575`
    - `pair_0064`: `0.1799`
- Interpretation of that failure:
  - not a notebook bug
  - not a CLI mismatch anymore
  - not an `xformers` problem
  - not a dataset-format problem
  - it is a validation-split locality problem exposed by the guard
- Do not “fix” this by:
  - increasing the allowed max change ratio
  - disabling eval
  - changing prompt / loss / resolution / threshold / dilation
- The next session should keep the experiment dataset-only and decide one data-side action:
  - remove `pair_0050` from the clean local-edit validation split for `core_v2`, or
  - move it to a separate harder validation bucket
- That split decision has now been executed:
  - clean validation dataset:
    - [`dataset/paired_edit_core_v2_cleanval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v2_cleanval)
    - val ids: `pair_0009`, `pair_0040`, `pair_0047`, `pair_0064`
  - harder validation dataset:
    - [`dataset/paired_edit_core_v2_hardval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v2_hardval)
    - val ids: `pair_0050`
- The clean-val Colab retrain has now completed and been archived locally at:
  - [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs)
- Clean-val run readout:
  - reached `1500` steps
  - wrote checkpoints through `model_1500.pkl`
  - wrote eval metrics through `metrics_001500.json`
  - wrote training samples through `train_step_001500.png`
  - stayed on the guarded config except for the dataset path / split artifact
- Current interpretation of that archived run:
  - definitely better than the old catastrophic white-collapse regime
  - also better on obvious global red / magenta drift
  - still visibly soft / blurry, so not a complete recovery of the legacy route
- Current local validation blocker for the next legacy comparison:
  - not the clean-val checkpoint itself
  - the local paired-edit validation environment
  - specifically, the old local upstream stub at `/tmp/img2img-turbo-local/src` was incomplete
  - a full clone now exists at [`/tmp/img2img-turbo-local-full`](/tmp/img2img-turbo-local-full), but the temporary runtime preparation remained environment-fragile in this round
- Highest-value next legacy task:
  - finish the strict side-by-side local validation comparison:
    - [`outputs/checkpoints/model_1401.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/checkpoints/model_1401.pkl)
    - vs [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl)
  - fixed pair set:
    - `pair_0005`
    - `pair_0015`
    - `pair_0009`
    - `pair_0040`
- That strict comparison has now completed locally after repairing the runtime helper.
- Runtime/tooling note:
  - [`src/paired_edit/pix2pix_runtime.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/pix2pix_runtime.py) was hardened so local validation no longer depends on fragile half-built `/tmp/img2img-turbo-runtime/<device>` directories.
  - a complete upstream clone is now available at [`/tmp/img2img-turbo-local-full`](/tmp/img2img-turbo-local-full)
- Comparison output roots:
  - old baseline:
    - [`outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_baseline1401/model_1401)
  - clean-val checkpoint:
    - [`outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v2_cleanval_model1500/model_1500)
- Comparison conclusion:
  - `model_1500` is clearly better than `model_1401` on the fixed pair set
  - the biggest gains are:
    - less catastrophic whitening
    - less magenta / red global drift
  - the main unresolved weakness is:
    - blur / texture softness
- Practical legacy baseline update:
  - treat [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl) as the current best legacy paired-edit checkpoint
  - keep [`outputs/checkpoints/model_1401.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/checkpoints/model_1401.pkl) only as the old baseline reference
- The next legacy decision is now narrower:
  - decide whether `pair_0022` and `pair_0066` should remain in `core_v2`
  - rationale: collapse is no longer the primary problem, so the next likely dominant issue is residual drift / softness from second-tier hard pairs
- That decision has now been made:
  - move `pair_0022` and `pair_0066` out of the legacy clean baseline
  - keep them in a secondary bucket instead of deleting them from the project
- New legacy candidate datasets now exist:
  - clean baseline candidate:
    - [`dataset/paired_edit_core_v3_cleanval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_cleanval)
    - `23 train / 4 val`
    - val ids: `pair_0009`, `pair_0040`, `pair_0047`, `pair_0064`
  - hard validation bucket:
    - [`dataset/paired_edit_core_v3_hardval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_hardval)
    - `23 train / 1 val`
    - val id: `pair_0050`
  - secondary drift bucket:
    - [`dataset/paired_edit_core_v3_secondary`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_secondary)
    - train ids: `pair_0022`, `pair_0066`
- Supporting manifests:
  - [`dataset/annotations/paired_edit_core_v3_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_secondary_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_secondary_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_cleanval_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_cleanval_manifest.json)
  - [`dataset/annotations/paired_edit_core_v3_hardval_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v3_hardval_manifest.json)
- Colab handoff entry for the next retrain:
  - notebook:
    - [`colab/train_paired_edit_core_v3_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_paired_edit_core_v3_v1.ipynb)
  - cleanval config:
    - [`colab/paired_edit_core_v3_cleanval_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/paired_edit_core_v3_cleanval_config.yaml)
  - expected Drive dataset path:
    - `/content/drive/MyDrive/paired_edit_core_v3_cleanval`
  - expected Drive output path:
    - `/content/drive/MyDrive/nail-retouch-paired-core-v3-cleanval-outputs`
- Audit readout after the split:
  - `core_v2` mean drift score: `0.1377`
  - `core_v3` mean drift score: `0.1224`
  - `core_v3_cleanval` mean drift score: `0.1162`
- Why `pair_0022` / `pair_0066` were split:
  - they remained the top two train-side residual drift pairs after the `core_v2 cleanval` retrain
  - both show broad whole-hand lift / smoothing, not just local retouch behavior
  - `pair_0022` is the riskier one of the two
- Current interpretation:
  - this is a justified data-side tightening step, not yet proof that these two were the only cause of residual softness
  - the next real answer must come from a controlled retrain, not from more eyeballing alone
- Highest-value next legacy task is now updated:
  - run the guarded dataset-only retrain on [`dataset/paired_edit_core_v3_cleanval`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_cleanval)
  - then compare its checkpoint against the current `core_v2 cleanval model_1500` baseline on the same fixed validation pairs
- That `core_v3 cleanval` retrain has now completed and been archived locally at:
  - [`outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs)
- `core_v3 cleanval` readout:
  - reached `1500` steps
  - wrote checkpoints through `model_1500.pkl`
  - wrote eval metrics through `metrics_001500.json`
  - preserved the guarded config while only swapping the dataset
- Comparison against `core_v2 cleanval`:
  - `core_v3` did not beat `core_v2`
  - its step-1500 metrics are slightly worse on full, preserve, edit, and LPIPS terms
  - strict local fixed-pair validation also stayed in the same soft / blurry regime
- New strict local validation outputs for `core_v3 cleanval`:
  - [`outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_validation_core_v3_cleanval_model1500/model_1500)
- Practical interpretation now:
  - `core_v2 cleanval model_1500` remains the best legacy paired-edit reference
  - `core_v3 cleanval` is useful closing evidence, but not a new primary baseline
  - the legacy dataset-only line appears to be near its useful ceiling
- Updated project priority:
  - stop treating deeper `core_v4`-style legacy dataset pruning as the main next move
  - shift the main experiment priority back to the masked route
- Reopened the masked line with a narrow single goal:
  - repair masked validation coverage before deciding on any fresh masked training run
- Reused the Evaluation Agent to confirm that validation coverage, not new training, was the most informative next masked question.
- Reused the Training Agent to isolate the local blackout source and keep the code change minimal.
- A local-eval-only opt-in safety-checker bypass now exists in:
  - [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py)
  - [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py)
  - shared pipeline builder:
    - [`src/inference/masked_inpaint_utils.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/masked_inpaint_utils.py)
- Default behavior did not change:
  - the safety checker stays enabled unless `--disable-safety-checker` is explicitly passed for local evaluation/debugging
- Recovered masked validation artifacts for the two previously safety-unstable val samples on both archived checkpoints:
  - `step150`:
    - [`pair_0040`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety/pytorch_lora_weights_step_000150/pair_0040_sheet.png)
    - [`pair_0050`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step150_disable_safety/pytorch_lora_weights_step_000150/pair_0050_sheet.png)
  - `step200`:
    - [`pair_0040`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety/pytorch_lora_weights_step_000200/pair_0040_sheet.png)
    - [`pair_0050`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/local_validation_step200_disable_safety/pytorch_lora_weights_step_000200/pair_0050_sheet.png)
- Current masked validation interpretation after restoring full 4-sample coverage:
  - `pair_0009`, `pair_0047`, `pair_0040`, and `pair_0050` are now all locally rankable
  - `step150` remains slightly better than `step200` across the restored 4-sample mean metrics
  - the previous blackouts on `pair_0040` / `pair_0050` were safety-checker artifacts, not proof that those samples were unusable
- Practical masked checkpoint update:
  - initial post-repair conclusion had kept `step150` as the archived default
  - that conclusion has now been superseded after completing the full budget curve
- User-provided external zips are now preserved inside the workspace at:
  - [`archive/2026-04-06_user_result_zips`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips)
- That archive includes the original zipped inputs for:
  - masked full-12 Colab outputs
  - paired `core_v2 cleanval` Colab outputs
  - paired `core_v3 cleanval` Colab outputs
  - one additional Drive bundle referenced in-thread
- Archive manifest with checksums and output-run mappings:
  - [`archive/2026-04-06_user_result_zips/README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips/README.md)
- Practical implication:
  - the system `Download` folder is no longer the only copy of those user-supplied artifacts
  - clearing `Download` will not remove the project-side archived copies
- Highest-value next masked task is now updated:
  - decide the next single masked training variable to change, now that legacy has been de-prioritized and checkpoint ranking on the archived full-12 run no longer depends on only two clean validation samples
- That decision has now also been made:
  - the next masked single variable should be training budget / early stopping position
  - not dataset
  - not task split
  - not rank
  - not resolution
- To avoid opening a fresh run prematurely, the archived full-12 budget curve was completed locally on the restored 4-sample validation split using:
  - `step050`
  - `step100`
  - `step150`
  - `step200`
- Budget-curve conclusion:
  - `step100` is now the best archived checkpoint on the 4-sample local validation split
  - `step050` is also slightly better than `step150` / `step200`
  - `step150` is still a reasonable near-neighbor fallback
  - `step200` is no longer the right default and should be treated as a later-training comparison point only
- New practical masked checkpoint update:
  - default archived full-12 checkpoint:
    - [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000100.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000100.safetensors)
  - near-neighbor fallback:
    - [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors)
- Highest-value next masked experiment is now narrower:
  - run a budget-only early-stop refinement on faster hardware / Colab
  - keep the full-12 dataset, task framing, prompt, loss stack, rank, and resolution fixed
  - only tighten the useful training region around `step100`
- Dedicated config for that next run:
  - [`colab/masked_inpaint_full12_earlystop_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_earlystop_config.yaml)
  - intended settings:
    - `max_train_steps=150`
    - `checkpointing_steps=25`
    - `preview_steps=25`
  - purpose:
    - refine the early-stop curve around the new archived best point without changing any other training variable
- Colab handoff is now one step simpler:
  - [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb) now defaults to `masked_inpaint_full12_earlystop_config.yaml`
  - after a fresh clone, the user no longer needs to edit `CONFIG_FILE` manually before running the notebook
- The same notebook now also treats a correctly uploaded cached dataset as a first-class input:
  - it explicitly validates `build_summary.json`, `train/metadata.jsonl`, and `val/metadata.jsonl` under `/content/drive/MyDrive/masked_inpaint_cuticle_cleanup_v1_full12`
  - it prints Drive existence / child diagnostics before attempting raw-pair fallback
  - this should prevent a valid cached full-12 dataset from being mistaken for a missing-input situation
- Legacy comparison protocol is prepared:
  - baseline pair ids: `pair_0005`, `pair_0015`, `pair_0009`, `pair_0040`
  - compare new `core_v2` checkpoint against [`outputs/checkpoints/model_1401.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/checkpoints/model_1401.pkl)
  - use historical bad `pair_0009` artifacts at `model_251` / `model_501` as negative references for whitening / color drift / blur
- Interpret those masks as `proximal_nail_boundary_refinement`, not pure cuticle cleanup.
- Allow a narrow posterior-edge transition band when the target clearly improves the proximal nail boundary.
- If mask budget is tight, start with the safer draft pairs: `pair_0015`, `pair_0018`, `pair_0009`, `pair_0040`.
- The tiny dry-run is now verified to start correctly; the next masked priority is to expand the approved subset beyond 8 samples or move the same subset onto faster hardware for a more informative run.
