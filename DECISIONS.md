# Decisions

Last updated: 2026-04-22

## 2026-03-30 - Treat Dataset Drift As The Primary Failure Mode

Decision:
Prioritize dataset drift control before changing training steps, rank, or inference prompts.

Why:

- Historical outputs already show collapse and color cast under deterministic inference.
- Manual pruning notes and the new audit script both point to global brightness/color drift inside the training pairs.
- Full-image supervised losses will keep amplifying that bias if the data remains unchanged.

Implication:

- Future experiments should first change dataset membership or pair normalization, not stack multiple training tweaks at once.

## 2026-03-30 - Run Paired Drift Audit Before Promoting Any Dataset

Decision:
Use [`src/data/audit_paired_drift.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/data/audit_paired_drift.py) as a standard gate before training on a new curated dataset.

Why:

- It catches the same classes of issues that manual pruning flagged: whole-image lift, warmth drift, and oversized change masks.
- It also revealed that `core_v1` is not drift-free, which was easy to miss by looking only at the expansion batch.

Implication:

- Do not promote `phase1_expand` or future raw-pair subsets without first checking mean luma delta, channel drift, change ratio, and worst-pair rankings.

## 2026-03-30 - Next Retrain Must Be A Dataset-Only Experiment

Decision:
The next training run should keep the guarded training config fixed and only change the dataset.

Why:

- We now have enough evidence that the data variable is confounded with the current failure mode.
- Changing resolution, loss weights, and dataset at the same time would make the next result hard to interpret.

Implication:

- Hold `resolution=256`, `learning_rate=5e-6`, `batch_size=1`, `lora_rank_unet=8`, `lora_rank_vae=4`, and the current loss weights constant for the next retrain.

## 2026-03-30 - Migrate From Full-Image Paired Editing To Explicit Masked Inpainting

Decision:
The long-term training and inference route should move away from full-image `before -> after` paired editing and toward explicit `input + mask + target_local` inpainting supervision.

Why:

- The current dataset format teaches whole-image brightness and color drift whenever the target differs globally from the input.
- The current training route derives masks from image differences, which is useful as a guardrail but still leaves the model supervised on full-frame target statistics.
- The current inference route does not enforce exact pixel preservation outside the intended edit region.

Implication:

- Future dataset builders should export masks and local-only targets.
- Future training should optimize masked reconstruction plus strong outside-mask identity preservation.
- Future inference should use an inpainting pipeline and explicitly composite original pixels back outside the mask.

## 2026-03-30 - Treat Diff Masks As Bootstrap Only, Not Final Supervision

Decision:
Use explicit masks for the real masked inpainting dataset. Keep difference-derived masks only as a temporary bootstrap path for smoke tests and rapid prototyping.

Why:

- The new masked dataset builder worked, but the smoke test showed that `pair_0005` still produced an overly broad diff mask (`mask_ratio 0.4120`), covering much more than the intended retouch region.
- Broad diff masks would leak hand-wide appearance changes back into supervision and weaken the identity-preservation goal.

Implication:

- Use `--mask-mode diff` only to test pipeline plumbing.
- Do not train the final production route on diff masks when explicit masks can be authored or reviewed.

## 2026-03-30 - Keep The New Masked Route Separate From The Legacy Full-Image Trainer

Decision:
Implement the masked inpainting LoRA route as a separate training script instead of continuing to evolve the existing full-image paired-edit trainer.

Why:

- The legacy trainer is built around full-image pix2pix-style supervision and derived change masks.
- The new route depends on explicit `input + mask + target_local` data, an inpainting checkpoint, and explicit outside-mask identity loss.
- Keeping them separate preserves comparability and avoids hiding two different modeling assumptions behind one script.

Implication:

- [`src/training/train_masked_inpaint_lora.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/training/train_masked_inpaint_lora.py) is the new masked route.
- [`src/paired_edit/train_supervised_retouch.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/paired_edit/train_supervised_retouch.py) remains the legacy full-image baseline.

## 2026-03-31 - Use Core V2 For The Next Legacy Paired-Edit Retrain

Decision:
If we run another legacy paired-edit retrain before the masked route is fully labeled, use `dataset/paired_edit_core_v2` instead of `dataset/paired_edit_core_v1`.

Why:

- Removing only `pair_0154`, `pair_0153`, `pair_0073`, and `pair_0069` dropped the audit mean drift score from `0.2121` to `0.1377`.
- The max change ratio also fell sharply from `0.9667` to `0.4519`, which is exactly the kind of outlier suppression the next narrow experiment was supposed to test.

Implication:

- The next legacy training comparison should hold all guarded training hyperparameters fixed and swap only the dataset folder to `dataset/paired_edit_core_v2`.
- `pair_0022`, `pair_0066`, and `pair_0035` become the next candidates for secondary filtering only if `core_v2` retraining still shows preserve-region drift.

## 2026-03-31 - Start Explicit Mask Labeling From A Small Seed Pack

Decision:
Use the scaffolded `masked_cuticle_cleanup_v1` subset as the first explicit-mask dataset instead of waiting for broad full-dataset labeling.

Why:

- The repository now has a generated annotation pack, a manifest scaffold, and a dedicated final mask directory for a 12-pair cuticle-cleanup seed subset.
- This keeps the first masked dry-run narrow enough to interpret and avoids mixing multiple task types before we know the cleanup route is stable.

Implication:

- Save final masks under `dataset/annotations/masks/masked_cuticle_cleanup_v1/`.
- After the first few masks are ready, build the masked dataset and run a tiny dry-run instead of waiting for a fully labeled large subset.

## 2026-03-31 - Treat Bootstrap Masks As Annotation Drafts, Not Just Training Bootstrap

Decision:
Use bootstrap diff masks as human annotation drafts only. Do not point the final training dataset at them directly.

Why:

- The scaffold pack showed a split between safe draft masks (`pair_0015`, `pair_0018`, `pair_0009`, `pair_0040`) and very broad drafts (`pair_0050`, `pair_0063`, `pair_0070`, `pair_0005`, `pair_0047`).
- Broad drafts are still informative for an annotator, but they are not safe to reuse as final supervision.

Implication:

- Start manual refinement from the safer seed pairs if label budget is tight.
- Redraw or defer the broad-mask samples instead of treating the draft masks as nearly final.

## 2026-03-31 - Validate Masked Models On Edit And Preserve Metrics Separately

Decision:
Use masked-route validation that reports inside-mask edit quality and outside-mask preservation separately.

Why:

- The product requirement is local retouch, not general beautification.
- A model can appear visually plausible while still failing by shifting unmasked skin tone or border color.

Implication:

- Use [`src/inference/run_masked_inpaint_validation.py`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/src/inference/run_masked_inpaint_validation.py) for masked-route checks.
- Track at least masked L1 / DeltaE, unmasked L1 / DeltaE, and boundary-ring error before promoting a checkpoint.

## 2026-04-01 - Reframe The First Masked Task As Proximal Nail Boundary Refinement

Decision:
Treat the first explicit-mask subset as `proximal_nail_boundary_refinement`, not pure `cuticle_cleanup`.

Why:

- Manual review of the actual before/after pairs showed that many edits do more than remove dead skin; they also improve the posterior nail edge shape.
- In much of the dataset, the `after` image reveals or slightly redraws the proximal boundary so the nail looks cleaner and more aesthetically balanced.
- Calling these samples `cuticle_cleanup` would understate the true supervision signal and create confusion during mask authoring and evaluation.

Implication:

- Masks for this subset may include a narrow local movement band between the old and new posterior edge.
- The task still remains local; broad finger-wide edits are not allowed.
- Update manifests, annotation guidance, and future experiment notes to use the more accurate task label.

## 2026-04-01 - Allow Local Posterior-Edge Movement Inside The Mask

Decision:
For the first masked subset, allow the mask to cover a narrow local band where the posterior edge shifts between `before` and `after`.

Why:

- If the mask only traces the old edge or only traces the new edge, the model does not get a clean region in which to perform the intended boundary refinement.
- The real product target values posterior-edge neatness, so forbidding all edge movement would throw away a large portion of the useful supervision.

Implication:

- Author masks as a tight transition band around the posterior edge when needed.
- Do not let that band widen into general skin cleanup or global nail redesign.
- Samples with extreme posterior-edge relocation should still be considered for a later stronger shape-refinement subset instead of being mixed blindly into this first masked run.

## 2026-04-02 - Keep A Separate Low-Cost Local Smoke Preset For Masked Training

Decision:
Maintain a dedicated low-cost local smoke entrypoint for the masked inpainting trainer, separate from any real quality-evaluation run.

Why:

- The masked route is already structurally working, but 512-resolution CPU dry-runs are too slow for frequent sanity checks.
- We still need a repeatable way to verify launch, stepping, checkpoint writing, and preview generation after code changes.
- Folding smoke-only runtime shortcuts into the main experiment description would blur the boundary between plumbing checks and real training evidence.

Implication:

- Use [`scripts/run_masked_inpaint_local_smoke.sh`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/scripts/run_masked_inpaint_local_smoke.sh) for cheap local verification only.
- Do not use the local smoke preset as evidence for model quality, color stability, or boundary-refinement success.
- Continue to treat the planned 10-step masked run on faster hardware as the first meaningful training-runtime check.

## 2026-04-02 - Treat Local Smoke Model Fetch Failures As Environment Issues, Not Trainer Regressions

Decision:
When the local smoke wrapper fails at resolving `stable-diffusion-v1-5/stable-diffusion-inpainting`, interpret that as model-availability / network environment failure unless the same run also fails against a known-good cached snapshot.

Why:

- The default local smoke invocation failed before training because the environment could not resolve `huggingface.co`.
- The same wrapper completed `4/4` steps, wrote a checkpoint, and wrote a preview as soon as it was pointed at the cached local inpainting snapshot with offline flags.
- That isolates the failure to model fetch availability rather than the masked dataset, trainer, or preview pipeline.

Implication:

- For local plumbing checks in this environment, prefer a cached inpainting snapshot when available.
- Do not log remote model-resolution failures as evidence that the masked training route is broken.

## 2026-04-17 - Keep The Full12 Lambda-Color Reference As The Masked Default After V3 Dataset-Only

Decision:
Do not replace the current masked reference with the partial `v3 dataset-only` run. Keep [`full12 lambda_color=1.0 step150`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the default checkpoint.

Why:

- The `v3 dataset-only` run was stable and slightly forward on some metrics, but the deltas were extremely small and did not form a decisive upgrade.
- `step150` in the v3 run landed in a near-tie regime:
  - better on `masked_delta_e`, `unmasked_delta_e`, `border_l1`
  - slightly worse on `masked_l1`, `unmasked_l1`
- This matches the earlier `v2 dataset-only` readout: safe continuation, but not enough gain to justify a default checkpoint promotion.

Implication:

- Treat dataset-only masked expansion as useful for coverage and future taxonomy growth, but not as the best next optimization lever.
- Keep the current full12 lambda-color reference as the comparison anchor for future masked experiments.

## 2026-04-17 - Move The Next Masked Single-Variable Experiment To Lambda-Identity

Decision:
After `v2` and `v3` both landed in the near-tie / diminishing-returns regime, the next masked single-variable experiment should change `lambda_identity`, not dataset membership again.

Why:

- The current masked route is no longer bottlenecked by obvious dataset corruption on the promoted conservative subsets.
- The dominant remaining question is whether we can improve preserve-region behavior and edge discipline by retuning the outside-mask identity pressure.
- Continuing to probe small dataset-only additions would likely spend more annotation and GPU budget for similarly ambiguous gains.

Implication:

- Keep the current masked reference config fixed except for `lambda_identity`.
- The next Colab handoff should define one narrow `lambda_identity` experiment around the current reference route.

## 2026-04-18 - Do Not Promote The Lambda-Identity-7p5 Run Over The Current Masked Reference

Decision:
Keep [`full12 lambda_color=1.0 step150`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the masked default. Do not promote the archived `lambda_identity=7.5` run.

Why:

- The `lambda_identity=7.5` run is stable and near-tied with the current reference, but not clearly better.
- It improves preservation only by a negligible amount:
  - `unmasked_l1 0.0056065 -> 0.0056021`
  - `unmasked_delta_e 1.1307144 -> 1.1304536`
- At the same time, edit-side fit becomes negligibly worse:
  - `masked_l1 0.0653309 -> 0.0653356`
  - `masked_delta_e 8.5266851 -> 8.5277605`
- Border error is effectively unchanged.

Implication:

- `lambda_identity=7.5` should be treated as a safe ablation, not a new default.
- The route remains in a diminishing-returns regime on this variable scale, so the next masked experiment should move to a different single variable rather than immediately adopting stronger identity pressure as the new baseline.

## 2026-04-18 - Use A Small Lambda-Color Increase As The Next Masked Single Variable

Decision:
After the flat `v3 dataset-only` and `lambda_identity=7.5` results, make the next masked single-variable experiment a small `lambda_color` increase from `1.0` to `1.25`.

Why:

- `lambda_color` is the only recent variable family that has already produced a meaningful frontier improvement inside this project.
- The latest two ablations suggest that both dataset-only expansion and stronger identity pressure are now in a diminishing-returns regime.
- A small `1.0 -> 1.25` move is easier to interpret than jumping directly to a new capacity regime like `rank=8`.

Implication:

- Keep the current full12 reference recipe fixed except for `lambda_color`.
- Use [`colab/masked_inpaint_full12_lambda_color_1p25_v1.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_lambda_color_1p25_v1.yaml) as the next Colab handoff.

## 2026-04-18 - Do Not Promote The Lambda-Color-1p25 Run Over The Current Masked Reference

Decision:
Keep [`full12 lambda_color=1.0 step150`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the masked default. Do not promote the archived `lambda_color=1.25` run.

Why:

- The `lambda_color=1.25` run is stable but flat relative to the current reference.
- It is microscopically worse on both edit-side and preserve-side means:
  - `masked_l1 0.0653309 -> 0.0653402`
  - `masked_delta_e 8.5266851 -> 8.5272422`
  - `unmasked_l1 0.0056065 -> 0.0056094`
  - `unmasked_delta_e 1.1307144 -> 1.1307668`
- Border behavior is effectively unchanged.

Implication:

- A further small upward move in `lambda_color` no longer looks like the best immediate lever.
- The route has now gone flat on three recent near-neighbor ablations: `v3 dataset-only`, `lambda_identity=7.5`, and `lambda_color=1.25`.

## 2026-04-18 - Use Rank-8 As The Next Masked Single Variable

Decision:
After three flat near-neighbor ablations, make the next masked single-variable experiment a capacity test: `rank 4 -> 8`.

Why:

- Both evaluation and training roles ranked `rank 4 -> 8` above another conservative dataset/taxonomy expansion and above a `512 -> 768` resolution jump.
- `dataset-only` has now gone flat twice, and the recent `lambda_identity` / `lambda_color` nudges were also flat.
- A `rank 4 -> 8` test changes capacity while preserving the current winning recipe, making it the cleanest next probe of whether the model is now bottlenecked by LoRA capacity rather than by loss weights.

Implication:

- Keep the current full12 masked reference recipe fixed except for `rank`.
- Use [`colab/masked_inpaint_full12_rank8_v1.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_rank8_v1.yaml) as the next Colab handoff.
- Keep the modeling roadmap unchanged; this is an infrastructure detail, not a reason to revisit the masked task definition or loss stack.

## 2026-04-18 - Do Not Promote The Rank8 Run Over The Current Masked Reference

Decision:
Keep [`full12 lambda_color=1.0 step150`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the masked default. Do not promote the archived `rank8` run.

Why:

- The `rank8` run is worse than the current reference on every tracked patched-validation mean:
  - `masked_l1 0.0653309 -> 0.0656602`
  - `masked_delta_e 8.5266851 -> 8.5940570`
  - `unmasked_l1 0.0056065 -> 0.0056190`
  - `unmasked_delta_e 1.1307144 -> 1.1351714`
  - `border_l1 0.0361802 -> 0.0362734`
- The regression is moderate rather than catastrophic, but the direction is fully consistent.

Implication:

- Increasing LoRA capacity is not the missing lever on the current full12 masked setup.
- The next masked experiment should not spend another round on nearby rank changes.

## 2026-04-18 - Split Shape-Refinement Out As A Tiny Side Route

Decision:
After the `rank8` run came back backward, shift the next highest-information masked experiment from nearby hyperparameter tuning to task-structure isolation: create a tiny explicit `shape_refinement` side seed separate from `proximal_nail_boundary_refinement`.

Why:

- Recent evidence now rules out several nearby explanations:
  - `v3 dataset-only`: flat
  - `lambda_identity=7.5`: flat
  - `lambda_color=1.25`: flat
  - `rank8`: backward
- That pattern is more consistent with a task-mixture bottleneck than with an untested nearby scalar hyperparameter.
- The project has already documented several samples that do not fit the conservative local-boundary task cleanly, especially `pair_0073` and `pair_0120`.

Implication:

- Keep `proximal_nail_boundary_refinement` as the masked mainline default task.
- Test `shape_refinement` as a tiny separate side route instead of continuing to mix broader contour edits into the mainline.

## 2026-04-19 - Exclude Pair_0120 From The First Shape-Refinement Pass

Decision:
Do not use `pair_0120` as an active member of the first `shape_refinement` side-seed pass.

Why:

- The current user review is that `pair_0120` has severe before/after offset.
- That makes it too likely to test pairwise alignment noise rather than the intended `shape_refinement` task itself.
- The first side-route pass should maximize task clarity, not difficulty.

Implication:

- Keep `pair_0120` as a later hard-case / follow-up candidate.
- Use `pair_0073` as the first validation anchor for the initial `shape_refinement` pass instead.

## 2026-04-19 - Hold The First Shape-Refinement Promotion Until Distal Coverage Is Real

Decision:
Keep `pair_0074`, `pair_0100`, and `pair_0209` in micro-adjust status until their masks clearly cover the intended distal sidewall / tip-shape corridor, even if they are already local and visually tidy.

Why:

- The second QA pass showed that all three revisions remained clean and local, but they still read too conservatively for a genuine `shape_refinement` teaching signal.
- Promoting semantically under-scoped masks would blur the distinction between the new `shape_refinement` side route and the main `proximal_nail_boundary_refinement` task.

Implication:

- `pair_0073` remains the only current pass in the first `shape_refinement` seed.
- Do one more targeted refinement pass on `pair_0074`, `pair_0100`, and `pair_0209` before building an approved subset or training scaffold for this side route.

## 2026-04-19 - Keep Pair_0074 As The Only Remaining Shape-Refinement Blocker

Decision:
After the latest single-pair fixes, treat `pair_0209` and `pair_0100` as passed, and keep only `pair_0074` in micro-adjust status before promotion.

Why:

- `pair_0209` and `pair_0100` now cover enough distal silhouette / tip-transition corridor to function as real `shape_refinement` masks.
- `pair_0074` improved again, but the long french nails still leave the distal sidewall / tip corridor slightly too narrow, especially on the right-side visible nail.

Implication:

- Current promotable pass set is now `pair_0073`, `pair_0100`, and `pair_0209`.
- Do not build the first approved `shape_refinement` subset until `pair_0074` also passes.

## 2026-04-19 - Treat The First Shape-Refinement Subset As A Real Side Route

Decision:
After `pair_0074` passed final QA, promote the first four-sample `shape_refinement` seed into its own approved subset and keep it separate from the `proximal_nail_boundary_refinement` mainline.

Why:

- The final QA pass now clears all four first-pass samples:
  - `pair_0073`
  - `pair_0074`
  - `pair_0100`
  - `pair_0209`
- The rebuilt dataset stayed local and color-aligned, and both the smoke run and short dry-run completed cleanly.
- That is enough evidence to treat `shape_refinement` as a real trainable side route rather than only an annotation thought experiment.

Implication:

- Use [`dataset/annotations/masked_shape_refinement_v1_approved_subset_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_shape_refinement_v1_approved_subset_manifest.json) as the current approved manifest for the first `shape_refinement` subset.
- Use [`dataset/masked_inpaint_shape_refinement_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_shape_refinement_v1) as the current built dataset for any future smoke, short-run, or GPU-side continuation of the side route.

## 2026-04-20 - Use A 100-Step Pilot For The First GPU-Side Shape-Refinement Run

Decision:
For the first GPU-side `shape_refinement` run, keep the main masked training variables fixed and use a modest `100`-step pilot budget with checkpoints and previews every `25` steps.

Why:

- The approved side-route dataset is currently only `3 train / 1 val`, so the first GPU run should be treated as a readability check rather than a hard optimization attempt.
- `100` steps is enough runway to see whether the route learns a coherent shape signal, while keeping overfitting risk and Colab spend lower than a full `150`-step run.
- Nearby variables like rank, resolution, and loss weights do not need to move yet because this pilot is about route viability, not hyperparameter exploration.

Implication:

- Use [`colab/masked_inpaint_shape_refinement_v1_pilot.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_shape_refinement_v1_pilot.yaml) as the first Colab handoff for the side route.
- Keep `resolution=512`, `rank=4`, `learning_rate=1e-5`, `lambda_identity=5.0`, and `lambda_color=1.0` unchanged for the first GPU-side readout.

## 2026-04-20 - Treat The First Shape-Refinement Pilot As A Data-Limited Success

Decision:
Interpret the first GPU-side `shape_refinement` pilot as a successful route-validation run whose next bottleneck is more data, not more steps.

Why:

- The pilot trained stably and produced a usable checkpoint ladder with no collapse.
- `step75` reads better than `step100`, which is more consistent with tiny-set wobble than with undertraining.
- The current subset is only `3 train / 1 val`, so additional training on the same tiny pool is less informative than adding a few more shape-focused examples.

Implication:

- Keep the current first pilot as proof that the side route is viable.
- Prioritize a second `shape_refinement` annotation batch before any longer or more aggressive GPU run on this side route.

## 2026-04-20 - Use A Conservative Second Shape-Refinement Seed That Still Includes One Hard Geometric Probe

Decision:
Open the next `shape_refinement` seed as a conservative 4-pair batch:
- train: `pair_0120`, `pair_0071`, `pair_0066`
- val: `pair_0043`

Why:

- The route now needs more examples, but not yet the noisiest possible batch.
- `pair_0120` is still high-value enough that continuing to defer it would reduce the information gain of the side route.
- `pair_0071`, `pair_0066`, and `pair_0043` add cleaner same-family contour supervision around that harder probe.
- `pair_0208` remains noisier than needed for this second, still-conservative seed.

Implication:

- Use [`dataset/annotation_packs/masked_shape_refinement_v2_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_shape_refinement_v2_seed) as the next manual annotation target for the side route.
- Keep `pair_0208` out of this batch and revisit it only after the second seed is QA-reviewed.

## 2026-04-22 - Promote Passing Shape-Refinement V2 Masks Immediately Instead Of Waiting For A Perfect Batch

Decision:
When a new `shape_refinement` seed batch contains a mix of clearly passing and still-noisy masks, promote the passing subset immediately into a merged approved manifest and keep only the failing masks in the manual queue.

Why:

- The v2 seed produced two clean passes (`pair_0071`, `pair_0066`) and two masks that still need revision (`pair_0120`, `pair_0043`).
- Waiting for the whole batch to become perfect would stall side-route growth even though the clean half is already trainable.
- The merged subset rebuild and local smoke / short dry-run both completed successfully, so there is no technical reason to delay the clean promotions.

Implication:

- Keep [`dataset/annotations/masked_shape_refinement_v2_approved_subset_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_shape_refinement_v2_approved_subset_manifest.json) as the active merged side-route manifest.
- Use [`dataset/masked_inpaint_shape_refinement_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_shape_refinement_v2) as the current shape-refinement training dataset.
- Narrow the remaining manual work to `pair_0120` and `pair_0043` instead of treating the whole v2 seed as blocked.

## 2026-04-22 - Stop Spending Annotation Budget On `pair_0120` / `pair_0043`

Decision:
Retire `pair_0120` and `pair_0043` from the active shape-refinement queue and replace them with a new tiny top-off pack instead of continuing to repair them.

Why:

- User review now confirms that both pairs have obvious before/after offset and should not be used.
- `pair_0120` was already the hardest geometric probe in the side route, and `pair_0043` still read more like a broad envelope than a clean corridor.
- The side route already has a healthy merged approved subset, so additional annotation time should now favor cleaner new signal rather than trying to rescue noisy tails.

Implication:

- Keep the current merged approved subset unchanged:
  - [`dataset/annotations/masked_shape_refinement_v2_approved_subset_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_shape_refinement_v2_approved_subset_manifest.json)
  - [`dataset/masked_inpaint_shape_refinement_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_shape_refinement_v2)
- Use the new replacement pack as the active manual target:
  - [`dataset/annotation_packs/masked_shape_refinement_v3_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_shape_refinement_v3_seed)
- Current replacement split:
  - train: `pair_0208`
  - val: `pair_0022`

## 2026-04-22 - Promote The `shape_refinement_v3` Top-Off Into The Active Side Route

Decision:
Promote the passed `shape_refinement_v3` replacement masks (`pair_0208`, `pair_0022`) into a new merged approved subset and use that merged subset as the active shape-refinement training dataset.

Why:

- Both replacement masks passed QA without inheriting the obvious offset problems that retired `pair_0120` and `pair_0043`.
- The rebuilt dataset stays in the same healthy local-mask regime while expanding the side route from `5 train / 1 val` to `6 train / 2 val`.
- The rebuilt dataset completed both a local smoke run and a short 10-step dry-run, so the promotion is not just annotation-complete but training-ready.

Implication:

- Replace the active merged side-route manifest with:
  - [`dataset/annotations/masked_shape_refinement_v3_approved_subset_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_shape_refinement_v3_approved_subset_manifest.json)
- Replace the active merged side-route dataset with:
  - [`dataset/masked_inpaint_shape_refinement_v3`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_shape_refinement_v3)
- Use the new Colab pilot handoff for the next GPU-side readout:
  - [`colab/masked_inpaint_shape_refinement_v3_pilot.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_shape_refinement_v3_pilot.yaml)

## 2026-04-02 - Make The Local Smoke Wrapper Auto-Prefer Cached Inpainting Weights

Decision:
The local smoke wrapper should automatically switch to a cached inpainting snapshot and offline mode when that cache exists, instead of requiring a manual `PRETRAINED_MODEL` override.

Why:

- The first end-to-end smoke run only failed because the environment could not resolve `huggingface.co`.
- The same command succeeded immediately once it used the local cached snapshot, which proved the training path itself was sound.
- Requiring a manual override every time would make local plumbing checks brittle and easy to misinterpret.

Implication:

- Keep `stable-diffusion-v1-5/stable-diffusion-inpainting` as the logical default for documentation and portable runs.
- For this repository's local smoke helper, auto-detect the local Hugging Face cache and enable offline mode when a cached snapshot is present.
- Treat the wrapper as a faster, lower-noise plumbing check, not as a quality benchmark.

## 2026-04-02 - Accept Anti-Aliased Authored Masks At QA Time Only If Build Binarizes Them

Decision:
Allow artist-authored masks with anti-aliased grayscale edges to pass QA when their region semantics are correct, but require the dataset promotion step to convert them to strict binary masks before training use.

Why:

- The repaired `pair_0005` and `pair_0057` masks became semantically correct after the touch-up pass even though their saved pixel values were no longer binary.
- Rejecting those masks at QA time would confuse a file-format artifact with the actual region-definition quality.
- The training route still benefits from deterministic binary masks for reproducible masked supervision and clean mask statistics.

Implication:

- Keep QA focused on locality and task semantics first.
- Normalize anti-aliased author masks during dataset build or an explicit pre-build cleanup step.
- Do not request a redraw only because an otherwise correct mask was exported with grayscale edges.

## 2026-04-02 - Use An Approved-Subset Manifest To Keep The Masked Route Moving

Decision:
When the broader seed manifest still includes unlabeled or unapproved samples, use a separate approved-only manifest to promote the currently valid explicit masks into a real masked dataset instead of waiting for the whole pack to finish.

Why:

- The original 12-sample seed manifest is still blocked by missing masks for `pair_0063`, `pair_0070`, `pair_0047`, and `pair_0050`.
- The approved 8-sample subset already builds cleanly, binarizes masks correctly, and completes both 4-step and 10-step local dry-runs.
- Waiting for all remaining labels would slow down runtime validation without improving interpretability.

Implication:

- Maintain [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json) as the current promotion target for masked-route experiments.
- Keep the full seed manifest as the longer-term labeling target.
- Treat future masked experiments on this subset as approved-subset experiments, not as evidence about the unfinished 12-sample pack.

## 2026-04-03 - Treat `pair_0047` And `pair_0050` As Optional Hard Cases, Not Immediate Blockers

Decision:
After promoting `pair_0063` and `pair_0070`, keep `pair_0047` and `pair_0050` deferred by default and do not treat them as required blockers for the next masked-route iteration.

Why:

- The approved explicit subset now builds cleanly as a 10-sample dataset (`8` train / `2` val), with healthy mask-locality and near-zero final luma drift after alignment.
- `pair_0063` and `pair_0070` were the two most useful remaining expansion targets because they still fit the current `proximal_nail_boundary_refinement` framing after manual tightening.
- `pair_0047` and `pair_0050` remain higher-risk because decoration and stronger lighting / appearance shifts make them more likely to reintroduce ambiguity about what should be preserved.

Implication:

- The masked route can continue using the current approved subset without waiting on more labeling.
- Only bring `pair_0047` or `pair_0050` back into the active queue if we explicitly want a harder expansion pass.
- Keep the next masked experiment single-variable: train or validate on the clean approved subset before adding these harder cases.

## 2026-04-03 - Use The Current 10-Sample Approved Subset As The First Real Masked Training Set

Decision:
Treat the current approved explicit subset (`8` train / `2` val) as the default first real masked training set instead of waiting on more mask expansion before running the next meaningful masked experiment.

Why:

- The approved explicit dataset now builds cleanly with near-zero final luma drift after alignment.
- A dedicated local 10-step dry-run on the expanded subset completed successfully, wrote metrics, a checkpoint, and a preview, and did not show a new catastrophic regression after adding `pair_0063` and `pair_0070`.
- The only remaining seed-pack samples are the intentionally deferred higher-risk cases `pair_0047` and `pair_0050`, so waiting on them would add ambiguity faster than it would add clean signal.

Implication:

- Use [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1) as the default masked training input for the next substantive experiment.
- Treat future additions of `pair_0047` or `pair_0050` as explicit harder-case expansion experiments, not as prerequisites.
- The next masked-route upgrade should change compute budget or runtime scale, not the subset membership.

## 2026-04-03 - Do Not Reject Decorated Or Lighting-Shift Pairs Solely By Historical Risk Label

Decision:
Do not reject `pair_0047`-style decorated samples or `pair_0050`-style stronger appearance-shift samples solely because they were previously triaged as high risk; judge them by the final authored mask locality and task semantics.

Why:

- Both `pair_0047` and `pair_0050` were initially deferred because their bootstrap drafts were broad and visually risky.
- After manual redraw, both passed semantic QA: the masks stayed local to the proximal boundary / cuticle bands and did not absorb decorations, whole-finger smoothing, or whole-nail repaint.
- That means the real gate is not pair difficulty by itself; it is whether the final explicit mask keeps the supervision local enough for `proximal_nail_boundary_refinement`.

Implication:

- Keep using bootstrap-draft risk only as annotation triage, not as an irreversible exclusion rule.
- Promotion into the default approved dataset should still remain a separate decision, because harder samples can be semantically valid while still making the default subset noisier.
- Future harder samples should be reviewed against the final authored mask, not pre-rejected from the task category just because the raw pair looked risky.

## 2026-04-03 - Promote The Full 12-Sample Seed Pack Into The Approved Masked Dataset

Decision:
Use the full first seed pack, including `pair_0047` and `pair_0050`, as the current approved explicit masked dataset rather than keeping those two samples outside the default approved manifest.

Why:

- Both samples passed semantic QA with local masks that stayed inside the intended `proximal_nail_boundary_refinement` region.
- After promotion, the rebuilt full-12 dataset still showed near-zero final luma drift after alignment at the dataset-summary level.
- Local smoke-scale training at `4`, `10`, and `25` steps remained stable after adding these samples, so the inclusion did not immediately dirty the route in a way that blocked training.

Implication:

- [`dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v1_approved_manifest.json) is now the full 12-sample promotion target.
- [`dataset/masked_inpaint_cuticle_cleanup_v1`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v1) should be treated as the current default masked dataset.
- Future experiments that remove `pair_0047` or `pair_0050` again should be logged as explicit subset-pruning experiments, not as a return to the previous default.

## 2026-04-03 - Keep Longer Full-12 Masked Runs Ready For Colab Even Though Local Smoke Still Works

Decision:
Keep using the local machine for smoke-scale masked checks on the full 12-sample dataset, but prepare and prefer a Colab notebook/config for longer GPU-side runs.

Why:

- The local machine still completed `4`, `10`, and `25` step runs on the full-12 dataset without correctness failures.
- That means local compute is still suitable for pipeline sanity checks and short early-training comparisons.
- Past that point, the real issue becomes efficiency rather than correctness, so notebook scaffolding should not be the reason a larger run gets delayed.

Implication:

- Use local smoke runs for short masked plumbing / stability checks.
- Use [`colab/train_masked_inpaint_full12_v1.ipynb`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/train_masked_inpaint_full12_v1.ipynb) and [`colab/masked_inpaint_full12_config.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_config.yaml) when the next run needs a meaningfully larger compute budget.
- Do not interpret “local still works” as a reason to avoid preparing the Colab path.

## 2026-04-03 - Make The Masked Colab Notebook Tolerant To Drive Layout Differences

Decision:
The masked full-12 Colab notebook should not assume one hard-coded Drive raw-data path; it should fall back to a cached built dataset or auto-discovered raw pair root before failing.

Why:

- The first real Colab handoff failed on a missing `/content/drive/MyDrive/nail-retouch-raw` directory even though the training route itself was fine.
- In practice, Drive layouts vary more than the model/data logic does during handoff.
- A brittle fixed-path assumption creates noisy false blockers that look like route failures even when the real issue is only folder placement.

Implication:

- Keep the notebook data-prep stage environment-aware and explicit about what it checked.
- Treat Drive-path failures as notebook handoff issues first, not as evidence that the full-12 masked route is unstable.
- Retry with the updated notebook branch before asking the user to manually rewrite notebook paths.

## 2026-04-04 - Treat The First Full-12 Colab Run As The New Masked Baseline Artifact

Decision:
Use the archived full-12 Colab run as the current masked baseline artifact, keep `step200` as the provisional default checkpoint, and keep `step150` as the explicit rollback comparison point.

Why:

- The archived run is the first full masked training result that includes metrics, previews, and intermediate checkpoints at a meaningful `512`-resolution GPU setting.
- Training metrics improved steadily through `200` steps without signs of divergence, and preview-based proxy comparisons were slightly but consistently better at `step200` than at `step150`.
- The remaining blocker to true masked validation is local inpainting-base availability, not evidence that later checkpoints are collapsing.

Implication:

- Refer to [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs) as the current masked baseline run.
- Default to `pytorch_lora_weights_step_000200.safetensors` for the next validation attempt, but keep `step150` in the comparison set so the next single-variable check is still “checkpoint step only.”
- Do not interpret the current local validation stop as a model failure until the same checkpoints have been tested in an environment with a complete inpainting base available.

## 2026-04-04 - Prefer `step150` As The Current Local Validation Candidate

Decision:
After the first real local masked validation sample, prefer `step150` over `step200` as the current candidate checkpoint, while still keeping both archived.

Why:

- Once the local inpainting base was restored, both checkpoints could be evaluated on `pair_0009` under the same masked validation path.
- `step150` was slightly better than `step200` on all key reported metrics for that sample, including masked L1, masked DeltaE, unmasked L1, and boundary-ring error.
- The margin is small enough that `step200` is still useful as a nearby comparison point, but it is no longer the best-supported default.

Implication:

- Use [`pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the current masked validation reference checkpoint.
- Keep `step200` in the same comparison set and avoid deleting or de-emphasizing it until a second trustworthy validation sample confirms the ranking.
- Treat the current checkpoint preference as provisional because it is still based on one cleanly evaluated sample.

## 2026-04-04 - Treat Safety-Checker Blackouts As Validation Artifacts, Not Immediate Model Failure

Decision:
When a masked validation sample is replaced by a black image after the diffusers safety checker triggers, treat that output as a validation artifact and not as direct evidence that the checkpoint itself failed.

Why:

- `pair_0040` under local validation produced a black output after safety filtering, which inflated its error metrics in a way that no longer reflects the model's actual edit behavior.
- Another validation sample (`pair_0009`) completed normally in the same environment, so the route itself is functioning.
- Promoting safety-filtered blackouts to model-quality evidence would confound checkpoint ranking with inference-time post-processing.

Implication:

- Exclude safety-blackout samples from checkpoint ranking until the same sample is rerun with a different seed or another trustworthy sample is available.
- Keep the next validation experiment single-variable: recover a second usable validation point without changing dataset membership or training settings.
- Log safety-filtered validation failures explicitly as inference-time artifacts.

## 2026-04-04 - Treat `pair_0040` As A Safety-Unstable Validation Sample For Now

Decision:
Stop using `pair_0040` as the immediate next ranking sample after three seed retries on `step150` all produced safety-blackout outputs, and prefer the next clean val sample instead.

Why:

- `pair_0040` remained blacked out under seeds `2026`, `7`, and `42` while the same local validation route was otherwise functioning.
- `pair_0047` succeeded immediately under the same checkpoint and validation settings once only the sample changed.
- Continuing to spend retries on `pair_0040` would add more inference noise without increasing checkpoint confidence proportionally.

Implication:

- Treat `pair_0040` as a sample-specific validation outlier unless a later targeted experiment revisits it.
- Use `pair_0009` and `pair_0047` as the current trustworthy local validation anchors for the archived full-12 run.
- If more coverage is needed, prefer new clean samples before adding more `pair_0040` seed retries.

## 2026-04-04 - Keep `step150` As The Default Masked Checkpoint After Two Clean Ranking Samples

Decision:
Use `step150` as the default checkpoint for the archived full-12 masked run.

Why:

- `step150` already slightly outperformed `step200` on `pair_0009`.
- The direct follow-up comparison on `pair_0047` also left `step150` slightly ahead on the masked and border metrics, while `step200` only matched it within tiny noise-level differences.
- The two checkpoints are very close, but the available clean ranking samples now point in the same direction twice.

Implication:

- Treat [`pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the default masked validation / inference candidate from this archived run.
- Keep `step200` archived as a nearby fallback and comparison point, not as the primary candidate.
- Future validation should focus on broader sample coverage or next-step modeling choices rather than continuing to micro-rank `step150` and `step200` on the same narrow sample set.

## 2026-04-04 - Treat `pair_0050` As A Safety-Unstable Validation Sample For Now

Decision:
Do not use `pair_0050` as a local ranking sample in the current masked-validation loop.

Why:

- `pair_0050` blacked out under the current default checkpoint `step150` at both tried seeds (`2026` and `7`).
- That behavior now matches `pair_0040`, which means the issue is not “this sample just needed one more retry.”
- The validation set already has two clean ranking anchors, so spending more retries on `pair_0050` would add little confidence and a lot of noise.

Implication:

- Treat `pair_0050` as safety-unstable in the same operational bucket as `pair_0040`.
- Use only `pair_0009` and `pair_0047` for current local checkpoint ranking.
- Revisit `pair_0050` only if there is a targeted need to study safety-checker behavior on harder samples.

## 2026-04-04 - Preserve The Legacy `core_v2` Experiment Definition By Moving The Guarded Retrain To Colab

Decision:
Do not weaken the guarded legacy config locally just to bypass the missing `xformers` dependency. Keep the experiment dataset-only and move the retrain to a Colab / GPU environment instead.

Why:

- The local `core_v2` retrain did reach real model initialization, so the dataset swap itself is valid and the command path is basically sound.
- The run failed before `step 1` because the environment lacks `xformers`, not because the new dataset or training script is invalid.
- Disabling `xformers` locally would change the effective training setup and muddy the interpretation of a dataset-only experiment.

Implication:

- Treat the local machine as a launch-validation environment for this legacy route, not as the authoritative place to run the guarded `core_v2` retrain.
- Run the actual `core_v2` dataset-only experiment from a dedicated Colab entrypoint that preserves the guarded training settings.
- Keep any future comparison with `model_1401.pkl` framed as a dataset-only result, not as a config-changed result.

## 2026-04-05 - Fix Colab Legacy Handoffs By Syncing The Trainer, Not By Weakening The Notebook

Decision:
When the Colab handoff emits arguments that the local trainer clearly supports but Colab rejects, fix the branch-visible trainer script first instead of stripping guarded arguments out of the notebook.

Why:

- The local workspace version of `train_supervised_retouch.py` already supported the guarded prompt, loss, and change-mask arguments.
- The Colab traceback showed the branch-visible trainer did not, which means the real problem was branch drift between handoff files and trainer implementation.
- Removing those arguments from the notebook would have silently weakened the guarded experiment definition and turned a sync bug into a config change.

Implication:

- Keep the `core_v2` Colab notebook/config aligned to the intended guarded trainer interface.
- When Colab and local behavior disagree, check whether the active GitHub branch is missing paired trainer updates before editing the notebook semantics.
- Require a fresh clone of the updated branch after trainer-sync fixes so Colab does not keep running stale code.

## 2026-04-05 - Treat The `core_v2` First-Eval Failure As A Validation-Split Problem, Not A Reason To Weaken The Guard

Decision:
Do not relax the `change_ratio < 0.60` sanity guard, disable evaluation, or modify the guarded training config just because the first `core_v2` Colab retrain failed on `pair_0050` during validation.

Why:

- The retrain now launches for real, reaches `step 1`, and writes a first train sample, so the experiment is no longer blocked by notebook wiring, CLI mismatch, or environment setup.
- The failure happens because `pair_0050` reaches `change_ratio=0.6575` under the trainer's actual eval-time preprocessing, which violates the locality assumptions of this legacy baseline.
- Silently weakening the guard would turn a useful data-split diagnosis into an untracked experiment-definition change.

Implication:

- Treat `pair_0050` as incompatible with the current clean local-edit validation split for the guarded `core_v2` experiment.
- Keep the dataset-only interpretation intact by handling this on the split / evaluation side, not by changing prompt, loss, threshold, dilation, or other training variables.
- The next legacy decision should be whether to:
  - remove `pair_0050` from the clean validation holdout for this experiment, or
  - move it into a separate harder validation bucket.

## 2026-04-06 - Treat The `core_v2` Cleanval Retrain As A Partial Legacy Recovery, Not A Full Fix

Decision:
Interpret the archived `core_v2 cleanval` retrain as evidence that dataset filtering helps the legacy route, but do not treat it as proof that the legacy paired-edit baseline is now fully acceptable.

Why:

- The clean-val run completed the full guarded schedule to `1500` steps without the earlier eval-split blocker.
- Its archived metrics and sample trajectory show clear improvement over the historical collapse regime:
  - much less whole-image whitening
  - less severe red / magenta color drift
- However, the later training samples still show softness / blur and incomplete texture recovery, so the route has improved but not fully recovered.

Implication:

- Record the clean-val split as a useful legacy control improvement, not as an endpoint.
- Keep the next legacy comparison single-variable:
  - compare `model_1401.pkl` against clean-val `model_1500.pkl` on the fixed baseline pairs
- Do not respond to the remaining blur by immediately changing prompt, loss, or resolution in the same round.

## 2026-04-06 - Accept `core_v2 cleanval model_1500` As The New Legacy Baseline Reference

Decision:
Use `core_v2 cleanval model_1500.pkl` as the current legacy paired-edit reference checkpoint instead of `model_1401.pkl`.

Why:

- The strict fixed-pair comparison now completed locally on `pair_0005`, `pair_0015`, `pair_0009`, and `pair_0040`.
- Across that shared comparison set, `model_1500.pkl` is consistently less whitened and less magenta-shifted than `model_1401.pkl`.
- The new checkpoint is still blurry, but it is no longer in the same catastrophic collapse regime as the old baseline.

Implication:

- When discussing the current best-case legacy route, reference:
  - [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl)
- Treat the next legacy question as:
  - whether secondary drift pairs such as `pair_0022` and `pair_0066` are now the main remaining reason the route still looks soft
- Do not reopen the old `model_1401` checkpoint as the preferred legacy baseline unless a later controlled comparison reverses this result.

## 2026-04-06 - Move `pair_0022` And `pair_0066` Out Of The Legacy Clean Baseline

Decision:
Treat `pair_0022` and `pair_0066` as secondary-set legacy samples instead of keeping them inside the default clean training pool.

Why:

- After the `core_v2 cleanval` retrain, collapse is no longer the dominant legacy failure mode; residual blur / softness is now the main issue.
- `pair_0022` and `pair_0066` remained the top two train-side drift pairs in `core_v2` with large positive luma lift and large change-area ratios.
- Their before/after sheets also show broad hand-wide brightening / smoothing behavior that does not match the cleanest local-retouch baseline we want the next dataset-only experiment to represent.
- Removing just these two lowers the mean drift score from `0.1377` in `core_v2` to `0.1224` in `core_v3`, and to `0.1162` in `core_v3_cleanval`.

Implication:

- Use `core_v3_cleanval` as the next legacy clean-baseline retrain candidate, not `core_v2_cleanval`.
- Keep `pair_0022` and `pair_0066` available in [`dataset/paired_edit_core_v3_secondary`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/paired_edit_core_v3_secondary) instead of pretending they were invalid forever.
- Keep the next experiment single-variable:
  - guarded retrain on `core_v3_cleanval`
  - compare against the existing `core_v2 cleanval model_1500` baseline
  - do not simultaneously change prompt, loss, rank, or resolution.

## 2026-04-06 - Treat `core_v2 cleanval model_1500` As The Legacy Dataset-Only Endpoint Unless New Evidence Appears

Decision:
Do not prioritize a `core_v4`-style continuation of the legacy dataset-only pruning line after `core_v3 cleanval`, unless a new, clearly justified outlier diagnosis emerges.

Why:

- The `core_v3 cleanval` retrain was a clean single-variable test of whether removing `pair_0022` and `pair_0066` would materially improve the legacy route.
- It did not outperform `core_v2 cleanval`; its step-1500 eval metrics were slightly worse across full, preserve, edit, and LPIPS terms.
- Strict fixed-pair local validation also failed to show a new quality tier; the route stayed in the same soft / blurry regime.

Implication:

- Keep [`outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v2_cleanval_run_2026-04-05_step1500/nail-retouch-paired-core-v2-cleanval-outputs/checkpoints/model_1500.pkl) as the default legacy paired-edit reference.
- Treat [`outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs/checkpoints/model_1500.pkl`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/paired_edit_colab_runs/core_v3_cleanval_run_2026-04-06_step1500/nail-retouch-paired-core-v3-cleanval-outputs/checkpoints/model_1500.pkl) as a closing ablation, not a new primary baseline.
- Shift the main project priority back toward masked-route training / validation instead of continuing to spend primary budget on legacy dataset-only pruning.

## 2026-04-06 - Make The Masked Safety-Checker Bypass Explicitly Opt-In And Keep `step150` As The Archived Full-12 Default

Decision:
Add a local-eval-only `--disable-safety-checker` switch to the masked inference / validation entrypoints, and keep `pytorch_lora_weights_step_000150.safetensors` as the default archived checkpoint for the full-12 masked Colab run.

Why:

- The previous local blackouts on `pair_0040` and `pair_0050` were caused by the diffusion safety checker, which prevented full validation coverage even though the pipeline could otherwise render those samples.
- We need a way to recover trustworthy local evaluation on already-approved retouch images without silently changing default inference behavior for normal usage.
- Once the safety-checker bypass was enabled only for local validation, both `pair_0040` and `pair_0050` produced real outputs and the full 4-sample validation split became comparable again.
- With those recovered points included, `step150` still remained slightly better than `step200` across masked, unmasked, and border metrics; the ranking did not reverse after the harder cases were restored.

Implication:

- Keep the default masked inference path safety-checked unless the operator explicitly passes `--disable-safety-checker` for local debugging / validation.
- Treat safety-checker blackouts as an inference-time artifact class, not as automatic evidence that a masked validation sample or checkpoint is invalid.
- Keep [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the archived full-12 default checkpoint.
- Move the next masked planning discussion from checkpoint ranking back to the next single training variable worth testing.

## 2026-04-06 - Treat Training Budget As The Next Masked Single Variable And Promote `step100` To The Archived Full-12 Default

Decision:
Use training budget / early stopping position as the next masked training variable, and promote `pytorch_lora_weights_step_000100.safetensors` to the current archived full-12 default checkpoint.

Why:

- Both the Training Agent and the Evaluation Agent independently converged on the same masked-route priority:
  - do not change dataset, task split, loss stack, rank, or resolution yet
  - first answer whether the current route is simply training past its best point
- The archived full-12 run already contained `step050` and `step100`, so we could finish the budget curve without opening a new training run.
- On the restored 4-sample local validation split, the ranking is directionally consistent:
  - `step100` is slightly better than `step150`
  - `step150` is slightly better than `step200`
  - `step050` is also slightly better than `step150` / `step200`, but `step100` is the best overall point
- This makes “early stop around ~100” a better-supported interpretation than “train longer.”

Implication:

- Treat [`outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000100.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_run_2026-04-03_step200/nail-retouch-masked-full12-outputs/lora_checkpoints/pytorch_lora_weights_step_000100.safetensors) as the current archived full-12 masked reference checkpoint.
- Keep `step150` as the nearest fallback / comparison point rather than the primary candidate.
- Define the next masked training experiment as a budget-only early-stop refinement on faster hardware, with dataset, task, prompt, loss, rank, and resolution unchanged.
- Do not spend the next round changing rank or resolution until the early-stop question is deliberately tightened or falsified.

## 2026-04-06 - Archive Every User-Provided External Result Bundle Before Any Cleanup

Decision:
Whenever the user provides an external result zip from Colab or another machine, first archive the raw bundle inside the workspace and then extract it into a stable run directory before relying on it for evaluation or allowing local cleanup.

Why:

- The project now depends on multiple externally produced training bundles for both masked and legacy comparisons.
- System download folders are transient and easy to clear.
- Keeping both the raw zip and the extracted run preserves reproducibility, traceability, and future re-analysis.

Implication:

- Use [`archive/2026-04-06_user_result_zips`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/archive/2026-04-06_user_result_zips) as the default raw landing zone for user-provided result bundles.
- Also map each archived bundle to a stable extracted directory under `outputs/`.
- Treat result archiving as part of project memory hygiene, not as an optional convenience step.

## 2026-04-06 - Treat The Patched Exact-Composite Validation Protocol As The Current Masked Local Ranking Baseline

Decision:
Use the patched ROI exact-composite validation protocol as the current masked local ranking baseline, and do not mix pre-patch and post-patch masked validation metrics in one checkpoint ranking table.

Why:

- The early-stop run exposed a validation-tool failure where ROI inpaint crops could come back at a size different from the source crop, causing exact compositing to fail.
- After normalizing generated ROI size before compositing, the same checkpoints validated cleanly and outside-mask preservation correctly collapsed to exact-zero error.
- That means the pre-patch and post-patch preserve metrics are not directly comparable; mixing them would confuse evaluation-implementation differences with real checkpoint quality differences.

Implication:

- When comparing masked checkpoints going forward, prefer the patched local validation outputs.
- Re-run older checkpoint references under the patched protocol before using them as direct baselines against newly evaluated runs.
- Treat the current exact-preserve local validation outputs as the authoritative masked ranking evidence.

## 2026-04-06 - Promote Early-Stop `step150` To The Current Masked Reference, With `step125` As The Near Neighbor

Decision:
Promote the archived early-stop run's `step150` checkpoint to the current masked reference checkpoint, and keep `step125` as the nearest comparison / rollback neighbor.

Why:

- Within the patched and internally consistent early-stop budget curve, edit-side metrics improve monotonically from `step025` through `step150`.
- `step150` is the best point on `masked_l1_to_target` and `masked_delta_e_to_target`.
- Compared against the older archived full-12 `step100` re-run under the same patched validation protocol, early-stop `step150` is still slightly better on masked edit quality, even though the margin is modest.

Implication:

- Treat [`outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the current masked reference checkpoint.
- Keep [`outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs/lora_checkpoints/pytorch_lora_weights_step_000125.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_earlystop_run_2026-04-06_step150/nail-retouch-masked-full12-earlystop-outputs/lora_checkpoints/pytorch_lora_weights_step_000125.safetensors) as the nearest low-risk neighbor.
- The next masked experiment should move to a new single variable instead of continuing to re-open the budget-only question on this same 12-sample set.

## 2026-04-09 - Close The Current Masked Budget Question And Move To A New Single Variable

Decision:
Treat the current full-12 masked budget question as closed: the useful optimum is at `step150`, with `step125` as the nearest earlier neighbor and `step200` already beyond the optimum.

Why:

- After re-running old archived `step100`, `step150`, and `step200` under the same patched validation protocol, the ranking is now consistent across both the old archived run and the dedicated early-stop run.
- `step150` remains the best region across both runs, while `step200` is slightly worse and `step100` is slightly early.
- The two `step150` checkpoints are effectively tied, so further budget-only iteration on this same 12-sample setup is unlikely to produce a meaningfully more informative answer.

Implication:

- Keep the dedicated early-stop run `step150` as the practical masked reference checkpoint.
- Keep early-stop `step125` as the nearest rollback / comparison point.
- The next masked experiment must change a different single variable, not budget.

## 2026-04-09 - Make `lambda_color` The Next Masked Single-Variable Experiment

Decision:
After closing the full-12 masked budget question, make `lambda_color` the next masked single variable, using a dedicated `lambda_color=1.0` Colab handoff while keeping all other training and validation settings fixed.

Why:

- The budget question is already closed well enough on the current full-12 setup, so reopening steps would add little information.
- Both the reused Evaluation Agent and Training Agent independently ranked `lambda_color` above `lambda_identity`, `rank`, and `resolution`.
- `lambda_color` is the cheapest next probe that still targets a product-relevant failure mode: color drift, red/purple tinting, and local color inconsistency inside the edit region.
- Changing rank or resolution next would confound compute, convergence, and capacity at the same time.

Implication:

- Use [`colab/masked_inpaint_full12_lambda_color_v1.yaml`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/colab/masked_inpaint_full12_lambda_color_v1.yaml) as the next masked Colab config.
- Keep dataset, manifest, prompt behavior, rank, resolution, learning rate, and `step150` budget fixed.
- Treat the result as a color-loss ablation against the current masked `step150` reference, not as a new broad training regime.

## 2026-04-09 - Promote `lambda_color=1.0` `step150` To The Current Masked Reference

Decision:
Promote the new full-12 `lambda_color=1.0` run at `step150` to the current masked reference checkpoint, with `lambda_color=1.0` `step125` as the nearest rollback / comparison point.

Why:

- The run was a clean single-variable ablation relative to the previous masked reference: only `lambda_color` changed from `0.5` to `1.0`.
- On the same patched 4-sample validation protocol, `lambda150` slightly outperformed the previous `ref150` on:
  - `masked_l1_to_target`
  - `masked_delta_e_to_target`
  - `border_l1_to_target`
- Exact outside-mask preservation remained perfect under the same protocol.

Implication:

- Treat [`outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the current masked reference checkpoint.
- Keep [`outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000125.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000125.safetensors) as the nearest lower-risk neighbor.
- The project can now shift its main uncertainty away from the current loss stack and toward the next stage of mask/data expansion.

## 2026-04-09 - Start The Next Masked Expansion With A Conservative V2 Seed Pack

Decision:
Open a second explicit-mask seed batch now, but keep it small and still constrained to `proximal_nail_boundary_refinement`.

Why:

- The masked route has already answered the immediate stability questions on the first full-12 subset:
  - dataset build is stable
  - Colab training is repeatable
  - local validation is restored
  - the budget region is already narrowed
  - `lambda_color=1.0` improved slightly instead of regressing
- That means annotation coverage is now a bigger limitation than another near-neighbor training ablation.
- A conservative seed pack keeps the next step interpretable and avoids prematurely committing to large-scale labeling.

Implication:

- Use [`dataset/annotations/masked_cuticle_cleanup_v2_seed_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v2_seed_manifest.json) as the next manual annotation target.
- Keep the task fixed at `proximal_nail_boundary_refinement`; do not mix in stronger shape-refinement masks yet.
- Treat this as a controlled expansion batch, not as permission to start unrestricted mass annotation.

## 2026-04-14 - Promote The V2 Expanded Dataset As The Next Masked Training Input

Decision:
After QA passes on all six v2 seed masks, promote them into a new approved manifest and use the rebuilt `dataset/masked_inpaint_cuticle_cleanup_v2` as the next masked training input.

Why:

- The entire v2 seed batch now passes semantic QA.
- The rebuilt v2 dataset remains local in mask coverage and well-controlled after color alignment.
- A first smoke run on the rebuilt dataset completed normally, so the dataset-only expansion did not break the masked training route.

Implication:

- Treat [`dataset/annotations/masked_cuticle_cleanup_v2_approved_manifest.json`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_cuticle_cleanup_v2_approved_manifest.json) as the current approved manifest for the next expansion stage.
- Treat [`dataset/masked_inpaint_cuticle_cleanup_v2`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/masked_inpaint_cuticle_cleanup_v2) as the next masked dataset to use for short validation runs and future faster-hardware training.
- Keep the next experiment single-variable: the dataset changed; do not simultaneously change loss, rank, or resolution.

## 2026-04-15 - Treat The First V2 Dataset-Only Run As Safe But Not Yet Promotable

Decision:
Keep the current masked reference checkpoint unchanged after the first `v2 dataset-only` continuation, and classify the run as `flat` rather than a decisive forward move.

Why:

- On the same patched 4-sample local validation protocol, the best v2 checkpoint (`step150`) only ties the current masked reference within tiny metric differences.
- `step150` is slightly better on `masked_delta_e`, `unmasked_delta_e`, and `border_l1`, but slightly worse on `masked_l1` and `unmasked_l1`; the gaps are too small to justify a reference swap.
- Earlier v2 checkpoints (`step100`, `step125`) remain a bit worse than v2 `step150`, so the run does not reveal a stronger hidden optimum elsewhere.

Implication:

- Keep [`outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/outputs/masked_inpaint_colab_runs/full12_lambda_color_run_2026-04-09_step150/nail-retouch-masked-full12-lambda-color-outputs/lora_checkpoints/pytorch_lora_weights_step_000150.safetensors) as the current masked reference checkpoint.
- Treat the promoted `v2` dataset as safe training input, but not yet as evidence that dataset expansion alone has materially improved the route.
- Shift the next masked priority from another near-neighbor dataset-only retrain toward the next conservative annotation expansion batch.

## 2026-04-15 - Keep The Next Annotation Expansion Conservative And Stay On The Same Task

Decision:
Open the next explicit-mask batch as another conservative `proximal_nail_boundary_refinement` seed pack instead of splitting taxonomy or jumping to a broad mass-labeling phase.

Why:

- The first `v2 dataset-only` continuation came back essentially flat, which means the current bottleneck is more likely annotation coverage than another small training tweak.
- The route is already stable enough on build, smoke, Colab training, and patched local validation that it can absorb another moderate expansion.
- The newly prepared v3 candidate pack includes some broader bootstrap drafts, so widening taxonomy right now would add ambiguity instead of clarity.

Implication:

- Keep the next manual annotation batch under the same task label: `proximal_nail_boundary_refinement`.
- Use the new conservative seed artifacts under [`dataset/annotation_packs/masked_cuticle_cleanup_v3_seed`](/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotation_packs/masked_cuticle_cleanup_v3_seed) as the next annotation target.
- Treat the bootstrap overlays as draft guidance only, and require manual narrowing before promotion.
