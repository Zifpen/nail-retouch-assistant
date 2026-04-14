# Tasks

Last updated: 2026-04-14

## Completed

- [x] Bootstrap persistent project memory files for ongoing experiment tracking.
- [x] Add paired drift audit script in [`src/data/audit_paired_drift.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/audit_paired_drift.py).
- [x] Audit `paired_edit_core_v1`, `paired_edit_phase1_expand_batch1`, `paired_edit_phase1_expand_batch1_pruned`, and `paired_edit_strict_plus`.
- [x] Confirm that the failure mode is dataset-first: core data still teaches positive whole-image lift.
- [x] Add [`src/data/build_masked_inpaint_dataset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/build_masked_inpaint_dataset.py) for `input + mask + target_local` export.
- [x] Add pairwise unmasked-region color alignment and per-pair alignment statistics to the masked dataset metadata.
- [x] Smoke-test the masked dataset builder on a 4-pair subset using bootstrap diff masks.
- [x] Add [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py) as the new masked inpainting training entrypoint.
- [x] Add explicit masked diffusion, outside-mask identity, and masked color losses to the new training route.
- [x] Add a dedicated masked inpainting inference entrypoint in [`src/inference/run_masked_inpaint_inference.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_inference.py).
- [x] Add split masked-route validation metrics in [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py).
- [x] Scaffold the first explicit-mask annotation subset with [`src/data/prepare_explicit_mask_subset.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/prepare_explicit_mask_subset.py).
- [x] Create [`dataset/annotations/paired_edit_core_v2_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/paired_edit_core_v2_manifest.json), build `dataset/paired_edit_core_v2`, and re-audit the filtered dataset.
- [x] Approve the first four explicit masks: `pair_0015`, `pair_0018`, `pair_0009`, `pair_0040`.
- [x] Build the four-mask explicit smoke dataset [`dataset/masked_inpaint_cuticle_cleanup_v1_smoke`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1_smoke).
- [x] Confirm that the masked training route can initialize and execute real steps on the explicit smoke subset.
- [x] Add a lower-cost local masked smoke entrypoint in [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh) and document its scope in [`README.md`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/README.md).
- [x] Run the new local smoke wrapper end-to-end, record its runtime, and confirm that it writes metrics, a checkpoint, and a preview artifact.
- [x] Make the local smoke wrapper auto-use a cached inpainting snapshot in offline mode when present, and verify that the default command completes without a manual model override.
- [x] QA-review newly drawn masks for `pair_0005`, `pair_0032`, `pair_0054`, and `pair_0057`.
- [x] Approve `pair_0032` and `pair_0054` for the next masked dataset build.
- [x] Re-review fixed `pair_0005` and `pair_0057`, and approve both for dataset promotion.
- [x] Build the first real approved-subset masked dataset in [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1).
- [x] Run a 4-step local masked smoke on the approved 8-sample dataset.
- [x] Finish a 10-step low-cost dry-run on the approved 8-sample masked dataset.
- [x] QA-review newly drawn masks for `pair_0063` and `pair_0070`, approve both, and promote them into the approved explicit manifest.
- [x] Rebuild [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1) from the expanded 10-sample approved manifest.
- [x] Run a fresh 10-step local masked dry-run on the expanded 10-sample approved subset and confirm it still writes metrics, a checkpoint, and a preview artifact.
- [x] QA-review newly drawn masks for `pair_0047` and `pair_0050`, and confirm both pass semantic review for `proximal_nail_boundary_refinement`.
- [x] Update the approved explicit manifest to include `pair_0047` and `pair_0050`, then rebuild the masked dataset from the fully approved 12-sample seed pack.
- [x] Run full-12 local masked smoke / dry-run checks at `4`, `10`, and `25` steps and confirm that the training route still writes metrics, checkpoints, and previews.
- [x] Add a masked Colab training notebook and YAML config for the full 12-sample approved dataset, following the historical paired-edit notebook format.
- [x] Harden the masked full-12 Colab notebook so it can fall back from a missing `drive_raw_dir` to a cached dataset or auto-discovered raw pair root.
- [x] Archive the first full-12 masked Colab training outputs locally under [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200).
- [x] Analyze the archived full-12 Colab run through metrics, previews, and checkpoint comparison, and provisionally select `step200` with `step150` as fallback.
- [x] Restore the local inpainting base snapshot and run the first real local masked validation artifacts for `step150` and `step200` on `pair_0009`.
- [x] Recover a second trustworthy local validation point for `step150` by switching from safety-unstable `pair_0040` to clean val sample `pair_0047`.
- [x] Complete the direct `step150` vs `step200` comparison on `pair_0047`.
- [x] Classify the last remaining full-12 val sample `pair_0050` as safety-unstable after two blacked-out local validation attempts.
- [x] Launch the local `legacy core_v2` dataset-only retrain far enough to classify the blocker as missing `xformers`, not dataset or command failure.
- [x] Add a dedicated Colab handoff for the guarded `legacy core_v2` retrain with dataset-only changes.
- [x] Sync the GitHub branch-visible paired trainer CLI with the `core_v2` Colab handoff after Colab exposed a stale-script mismatch.
- [x] Identify the exact `core_v2` validation outlier blocking the first real Colab retrain and confirm that the failure is `pair_0050` at eval-time `change_ratio=0.6575`.
- [x] Split `pair_0050` out of the clean `core_v2` validation bucket, build `paired_edit_core_v2_cleanval` / `paired_edit_core_v2_hardval`, and push the matching Colab handoff.
- [x] Archive the completed `core_v2 cleanval` Colab run locally and confirm that it reached `1500` steps with checkpoints, eval metrics, and sample artifacts.
- [x] Repair the local paired-edit validation runtime enough to rerun strict fixed-pair checkpoint comparisons.
- [x] Run strict local validation on `pair_0005`, `pair_0015`, `pair_0009`, and `pair_0040` for `model_1401.pkl` vs clean-val `model_1500.pkl`.
- [x] Confirm that `model_1500.pkl` improves whitening and color drift relative to `model_1401.pkl`, while still leaving blur / texture loss unresolved.
- [x] Decide that `pair_0022` and `pair_0066` should leave the legacy clean baseline and move into a temporary secondary set.
- [x] Build and audit `paired_edit_core_v3`, `paired_edit_core_v3_secondary`, `paired_edit_core_v3_cleanval`, and `paired_edit_core_v3_hardval`.
- [x] Prepare a dedicated `core_v3 cleanval` Colab config so the next guarded legacy retrain can start from the cleaner split directly.
- [x] Archive the completed `core_v3 cleanval` Colab run locally and compare it against the `core_v2 cleanval` baseline.
- [x] Run strict local validation on the same fixed baseline set for `core_v3 cleanval model_1500`.
- [x] Copy all user-provided external result zips into a workspace archive before clearing the system `Download` folder.
- [x] Recover full 4-sample masked validation coverage by making the local safety-checker bypass explicitly opt-in, then re-compare `step150` vs `step200` on `pair_0040` and `pair_0050`.
- [x] Archive the masked full-12 early-stop Colab output zip locally in both raw and extracted form before clearing transient downloads.
- [x] Patch masked ROI validation compositing so off-size generated crops are normalized before exact outside-mask compositing.
- [x] Re-run the archived masked early-stop checkpoints on one consistent patched local validation protocol.
- [x] Re-run the older archived full-12 `step100` checkpoint under the same patched local validation protocol for direct comparison.
- [x] Re-run the older archived full-12 `step150` and `step200` checkpoints under the same patched local validation protocol to close the current masked budget curve.
- [x] Decide the next masked single-variable experiment now that the budget-only question is closed and `step150` is the stable best-step region for the current full-12 setup.
- [x] Prepare a direct Colab handoff for the next masked `lambda_color` ablation.
- [x] Archive the `lambda_color=1.0` masked Colab output zip locally in both raw and extracted form.
- [x] Validate `lambda_color=1.0` checkpoints against the same patched 4-sample local protocol used for the current masked reference.
- [x] Decide whether the `lambda_color=1.0` run is a regression, tie, or small improvement relative to the current masked reference.

## Next Experiments

- [x] Decide whether `pair_0050` should leave the clean `core_v2` validation split or move into a separate harder validation bucket for the dataset-only legacy baseline.
- [x] Retry the guarded `core_v2` retrain after the validation-split decision, with training variables still unchanged.
- [x] Run local validation on the same fixed baseline set after the retrain: `pair_0005`, `pair_0015`, `pair_0009`, `pair_0040`.
- [x] Compare whether whitening, blue-channel lift, and texture loss fall relative to the old-route artifacts.
- [x] Decide whether `pair_0022` and `pair_0066` should stay in `core_v2` or move to a secondary set if preserve-region drift stays high.
- [x] Run the next guarded legacy retrain on `dataset/paired_edit_core_v3_cleanval` with no training-variable changes.
- [x] Compare the future `core_v3 cleanval` checkpoint against the current `core_v2 cleanval model_1500` baseline on the same fixed validation pair set.
- [x] Decide the next masked-route training experiment now that:
  - the legacy dataset-only control line has effectively reached its useful endpoint
  - and the archived full-12 masked checkpoint ranking has been revalidated on the full 4-sample local validation split.

## Migration Plan

- [ ] Wire notebooks / calling scripts to prefer the new inpainting inference entrypoint over the legacy paired-edit CLI.
- [ ] Introduce or curate task labels/manifests that split at least `proximal_nail_boundary_refinement` and stronger `shape_refinement`.
- [x] Keep the current 10-sample approved explicit subset as the first real masked training set instead of waiting on deferred `pair_0047` / `pair_0050`.
- [ ] Decide whether the full 12-sample approved masked dataset should now replace the previous 10-sample subset as the default long-run training set in project messaging and downstream configs.
- [ ] Move the current full 12-sample approved masked subset onto faster hardware or Colab for the first more informative GPU-side masked run beyond local CPU dry-runs.
- [ ] Run a budget-only masked early-stop refinement on faster hardware, keeping the full-12 dataset and all other training variables fixed while tightening the useful checkpoint region around `step100`.
- [ ] Evaluate the archived masked full-12 early-stop run on the same 4-sample local validation protocol used for the earlier `step050 / 100 / 150 / 200` ranking, and decide whether it sharpens or overturns the current `step100` preference.
- [x] Evaluate the archived masked full-12 early-stop run on one internally consistent 4-sample local validation protocol, and decide whether it sharpens or overturns the current `step100` preference.
- [x] Run the next masked Colab experiment with only `lambda_color` changed from `0.5` to `1.0`, keeping the full-12 dataset and current `step150` budget fixed.
- [x] Compare the resulting checkpoints against the current masked reference on the same patched 4-sample local validation protocol.
- [x] Define the next post-`lambda_color` masked expansion step: either the next annotation pack for more masks, or one final taxonomy split before broader annotation.
- [x] Decide whether the route is ready for a second explicit-mask seed batch after promoting the `lambda_color=1.0` run.
- [x] Draft a conservative v2 explicit-mask seed manifest for the next annotation tranche.
- [x] Generate a human-usable v2 annotation pack with per-pair `before` images and 3-panel sheets for manual mask authoring.
- [x] Run the first Mask QA pass on the uploaded v2 seed masks.
- [x] Re-review the two v2 micro-fix masks after the requested tightening pass.
- [x] Promote the full v2 seed batch into a new approved manifest and rebuild the masked dataset.
- [x] Confirm the rebuilt v2 dataset still completes a local smoke run.
- [x] Confirm the rebuilt v2 dataset also completes a short 10-step local dry-run.
- [x] Prepare a dedicated Colab config for the full-12 masked early-stop refinement experiment.
- [x] Update the existing masked full-12 Colab notebook so a fresh clone defaults to the early-stop config without manual notebook edits.
- [x] Harden the masked full-12 early-stop Colab notebook so it explicitly accepts a correctly uploaded cached Drive dataset before falling back to raw-pair discovery.
- [x] Run real masked validation on the archived full-12 Colab checkpoints in an environment with a complete inpainting base, comparing `step150` vs `step200` with checkpoint step as the only changed variable.
- [x] Decide whether the earlier two-sample local validation evidence (`pair_0009`, `pair_0047`) was sufficient to move on, or whether broader coverage was worth the runtime.
- [x] Push the dedicated `core_v2` Colab handoff files to GitHub and use that entrypoint for the first guarded GPU-side `core_v2` retrain.

## Backlog

- [ ] Evaluate whether mild color normalization of targets is needed after dataset-only filtering.
- [ ] Consider splitting `proximal_nail_boundary_refinement` and stronger shape-refinement into separate training routes if a filtered dataset still causes coupled failures.
- [ ] Add a small hard validation set from `hard_val_optional` after the core route stops collapsing on easy pairs.
- [x] Consider a `core_v3` legacy manifest if `pair_0022`, `pair_0066`, and `pair_0035` still dominate preserve-region drift after the `core_v2` retrain.
- [ ] Revisit whether `pair_0035` should join a later secondary legacy bucket if `core_v3 cleanval` still looks too soft.
- [x] Micro-adjust the two v2 seed masks that failed first-pass QA:
  - `pair_0064`
  - `pair_0154`
- [x] Confirm the following v2 seed masks already pass first-pass QA:
  - `pair_0118`
  - `pair_0122`
  - `pair_0153`
  - `pair_0190`
- [x] Confirm the repaired v2 seed masks now also pass QA:
  - `pair_0064`
  - `pair_0154`
- [ ] Use [`dataset/annotation_packs/masked_cuticle_cleanup_v2_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v2_seed) as the working annotation pack instead of browsing `raw/` directly.
- [x] Run `Mask QA -> approved-manifest promotion -> masked dataset rebuild` once the v2 seed masks arrive.
- [ ] Run a slightly more informative short dry-run on `dataset/masked_inpaint_cuticle_cleanup_v2` after the smoke pass, keeping all training variables fixed.
- [x] Run a slightly more informative short dry-run on `dataset/masked_inpaint_cuticle_cleanup_v2` after the smoke pass, keeping all training variables fixed.
- [ ] Prepare a dataset-only Colab handoff that swaps the masked training input from `v1` to `v2` without changing the current masked training variables.
