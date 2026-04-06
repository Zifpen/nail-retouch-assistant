# Decisions

Last updated: 2026-04-06

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
- Keep the modeling roadmap unchanged; this is an infrastructure detail, not a reason to revisit the masked task definition or loss stack.

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
