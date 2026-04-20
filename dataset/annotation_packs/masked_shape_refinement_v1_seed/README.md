# Explicit Mask Annotation Pack

Task: `shape_refinement`

## What to annotate

- Only mark the pixels that are allowed to change.
- This subset is for nail shape and contour correction, not ordinary local cuticle cleanup.
- Include the local bands where the nail silhouette, sidewall line, tip shape, or french boundary visibly moves between `before` and `after`.
- Cover the geometry-change corridor between the old edge and the new edge, even when that band is wider than a typical proximal-boundary cleanup mask.
- Keep the mask tied to nail structure. Do not spread into broad skin smoothing, hand-region cleanup, or unrelated lighting changes.
- Preserve original nail design intent whenever possible; this task is about controlled contour refinement, not redesign.
- If a sample looks globally misregistered or the bootstrap mask spills into large hand regions, treat that as a warning sign and redraw a narrower shape-focused mask.

## Files

- Final approved masks should be saved under `/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masks/masked_shape_refinement_v1_seed` using `<pair_id>.png`.
- Bootstrap masks are only rough drafts. Review and refine them before using them for training.
- The scaffold manifest is `/Volumes/DevSSD/AI-projects/nail-retouch-assistant/dataset/annotations/masked_shape_refinement_v1_seed_pack_manifest.json`.

## Quick QA before saving a final mask

- Mask should be binary black/white.
- Mask should stay attached to nail geometry and immediate boundary-shift regions.
- Mask may be wider than a cleanup mask when the contour truly moves, but it should not become a broad hand-region edit.
- Do not mask whole fingers just because the bootstrap draft is large.
- If a sample appears globally misaligned rather than locally reshaped, mark that in review notes instead of forcing a large mask.

## Current subset

- Train: pair_0074, pair_0100, pair_0209
- Val: pair_0073
