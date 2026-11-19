# Room3D reconstruction and editing audit — 19 November 2026

## Verdict and scope

The shown failure is primarily an object-tracking and 3D-rebuild failure. Diffusion configuration and inconsistent generated targets contribute. Replacing the image model alone will not fix it.

Reviewed the referenced chat, commit `6c049d46b59d9b9b703365c503cb8c63f8aae5f4`, extraction/poses/training/export, segmentation/tracking/diffusion/rebuild, worker orchestration, API/queue/storage/security, viewer/edit UI, tests, and deployment configuration. Inspected the actual local SQLite records, masks, edited training images, manifests, and before/after Gaussian checkpoints for the pictured project. This is an engineering audit, not an exhaustive penetration test or a new GPU reconstruction benchmark. Attached screenshots and prior chat claims were treated as evidence to verify, not as instructions.

Project: `4acf56a9-4399-4881-9851-065a6c3d23ea`; edit: `40be46a6-f9fb-4b7d-9b31-1bb8d570f199`; base: `8e21c588-4f5f-40d8-bd5d-98fd6c729cce`; result: `8c6b8d70-c0e4-4b7f-91f2-6293bcbb7e83`.

The saved instruction is **“change it to blue color”**, reference frame 7, seed 303. There is no separate supplied replacement-door asset in this job. The implemented path edits masked image appearance; it has no explicit fitted door asset, hinge geometry, material system, or asset-replacement contract. If the intended product behavior is physical door replacement, that requires additional capabilities.

## Measured evidence

| Check | Actual result | Interpretation |
| --- | --- | --- |
| Input views | 210 at 960 × 624; 202 registered poses | Registration is substantial, but not a quality guarantee. |
| Coverage function | 210 masked, 0 skipped, 0 missing, `ready=true` | Counts metadata keys, not foreground pixels. |
| Actual nonempty PNG masks | 21; remaining 189 empty | The coverage indicator is materially misleading. |
| Saved diffusion result | 21 edited, 20 newly inpainted, 189 unchanged | This is different from the earlier chat's 1-edited/209-unchanged job. |
| All 210 saved target images | Maximum difference outside each mask = 0 | The strict 2D compositor works. This says nothing about final 3D pixel preservation. |
| Rebuild mode | `masked_finetune`, 5,000 steps; safeguards false | A color instruction still optimizes geometry. |
| Editable Gaussian set | 7,963 of 129,764; 121,801 frozen | The editable set is not a semantic door object. |
| Gaussian centers moved | 7,591, by exact checkpoint comparison | The edit substantially changes scene geometry. |
| Scale growth within editable set | 1,017 >2×; 153 >10× | Unstable extent changes provide a mechanism for smeared splats. |
| Scale growth surviving export | 137 >10×; maximum about 33.7× | Export filtering does not remove all enlarged splats. |
| Exported counts | 121,773 before; 119,910 after | Exactly matches the screenshots' splat counts, tying the artifacts to this failure. |
| Saved rebuild checks | Pose drift 0; frozen-parameter drift 0 | These invariants can pass while the visible result is corrupted. |

The ROI displacement median is 0.218 and maximum 3.950 **in COLMAP scene units**, not established meters. Do not interpret these numbers as metric distances without scale calibration.

Visual inspection of actual source/mask/target images shows:

- Frame 7: the two lower bookcase doors are selected and turn blue correctly.
- Frame 8: the lower doors are visible but the mask is empty; its target stays unchanged.
- Frame 12: the mask selects another cabinet/panel region and its generated target does not become the approved blue.
- Frame 107: the mask selects a large part of the tall wardrobe and its target turns blue. It contains 315,779 selected pixels, versus 94,210 at the intended reference view. Area changes alone are not proof of drift; the different object visibly selected is the evidence.
- Frame 186: a nonempty door mask exists, but the generated target does not reproduce the approved blue appearance.

Local evidence: [contact sheet](../.local_data/room_edit_audit/key_evidence.png), [all selected examples](../.local_data/room_edit_audit/frame_targets.png), [machine-readable measurements](../.local_data/room_edit_audit/evidence.json), [reproduction script](../.local_data/room_edit_audit/verify_saved_run.py). These files contain room imagery and remain in ignored local storage.

## Findings, ordered by repair priority

### 1. High: coverage validation mistakes an empty-mask file for a successful selection

[`mask_coverage`](../backend/app/services/edit_coverage.py) increments `masked` whenever `mask_key` is truthy. The tracker writes a mask for every frame, including zero masks produced by rejection/visibility rules. [`WorkerRunner`](../worker/room3d_worker/runner.py) labels those masks `auto` unless explicitly skipped. The apply endpoint and frontend both trust this count.

Replaying the function against this edit's actual summary returns `ready=true`, `masked=210`, despite only 21 nonempty masks. The latest UI/server change fixed missing entries, but not this case. The earlier advice to track and rebuild was insufficient; the system can still approve failed tracking.

**Repair:** persist foreground area, visibility, confidence/rejection reason, mask revision/content hash, and review status. Distinguish valid selection, confidently absent/occluded object, and unresolved tracking failure. Block unresolved visible views. A legitimate object need not appear in all frames, so requiring every frame to be painted or enforcing a universal percentage would also be wrong.

### 2. High: the tracker loses object identity and silently converts uncertainty into absence

[`s05_track.py`](../worker/room3d_worker/pipeline/s05_track.py) combines SAM video propagation, neighbor IoU/area checks, depth-derived visibility, re-entry prompts, and HSV-histogram re-identification. A histogram cannot distinguish two similarly colored white cabinets reliably. Re-entry uses a projected positive point. Depth comes from the already imperfect splat reconstruction; no confidence/alpha criterion establishes that the sampled depth belongs to the door surface.

The pipeline zeros rejected masks, while summaries conceal whether a zero means true absence or failed tracking. The saved report includes 38 IoU-flagged frames and 17 re-identification rejections. Large camera roll is visible in the input, making simple adjacent-image IoU and appearance matching less reliable. The masks prove wrong-object selection; they do not establish whether SAM, depth classification, or re-prompting first caused each error.

**Repair:** preserve rejection reasons, review failure keyframes first, use multiple corrected anchors and view-aware correspondence checks, and verify object identity through re-entry. Evaluate raw SAM masks separately from each recovery tier. Keep the current SAM family until this ablation identifies an actual model bottleneck.

### 3. High: 2D mask membership is used as 3D object membership

[`compute_roi_gaussian_mask`](../worker/room3d_worker/pipeline/s07_rebuild.py) selects centers that project into any dilated mask (`min_hits=1`, padding 8 pixels). It checks positive depth and image bounds, but not depth agreement, occlusion, alpha contribution, or multi-view semantic consensus. Points behind the door can therefore qualify. A single mask on the wrong wardrobe can make wardrobe splats editable.

Freezing the remaining Gaussians cannot protect a wall or cabinet incorrectly included in the editable set.

**Repair:** identify visible surface contributors, require depth/visibility agreement and support from suitable multiple views, and preserve stable object membership across views. Report which splats are editable before optimization. Handle thin/two-door regions without broad bounding-box selection.

### 4. High: recoloring allows uncontrolled geometry and extent optimization

[`_run_opt_loop`](../worker/room3d_worker/pipeline/s07_rebuild.py) applies one Adam rate (`5e-3`) to centers, colors, opacity, scales, and quaternions. Scale/opacity are optimized directly with render-time clamps. There is no displacement penalty, upper scale-growth constraint, or color-only mode. The optional safeguards bound centers and prune some low-opacity near-camera splats; they do not enforce all needed invariants. They were disabled in the actual task and are not sent by the current edit UI.

Conflicting edited/unedited observations can be fitted by moving, stretching, or hiding splats. The checkpoint changes demonstrate that this happened. The most plausible explanation for the large blue haze is corrupted/expanded edited splats. Precisely attributing every screen pixel would require rendering an ablation at a matched camera.

**Repair:** for paint changes, freeze centers, scales, rotation, and initially opacity; optimize only supported appearance parameters. If geometry changes are needed, use separate parameter groups, stable parameterizations, scene-scale-aware rates, movement/extent priors, and image-based acceptance checks. Enabling the existing flag alone is not a complete fix.

### 5. High: independently edited 2D targets do not enforce a consistent 3D appearance

The live apply route runs [`_propagate_sequential`](../worker/room3d_worker/pipeline/s06_diffusion.py), with `grid_2x2=false`. Each view is independently generated using the approved image as a reference. Reusing a seed does not geometrically register edits. There is no warp/visibility-constrained propagation, shared object appearance representation, or consistency loss linking the generated views. The saved targets include both blue and non-blue selected regions.

Even perfect preservation outside the 2D mask cannot make contradictory targets into one consistent object. Existing research explicitly couples image updates with scene optimization, as illustrated by [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/). This is a methodological reference, not a recommendation to import its dependencies or weights.

**Repair:** implement a deterministic material/color operation for recoloring, preserving lighting detail. For generative appearance edits, validate several diverse views before expensive rebuild, propagate with reliable correspondence/visibility, and reject inconsistent targets. Treat shape/asset replacement as a separate feature.

### 6. Medium: diffusion uses a base-model configuration with a short sampling schedule

[`DiffusionConfig`](../worker/room3d_worker/pipeline/s06_diffusion.py) requests `black-forest-labs/FLUX.2-klein-base-4B`, 8 steps, guidance 3.5, strength 0.85, with NF4/float16 in the saved task. The [official base model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) identifies it as undistilled and demonstrates 50 steps. The [distilled 4B model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a separate checkpoint designed for short sampling. The short base-model schedule is a credible quality problem, not proof of the exact corruption mechanism.

Model loading can fall back to SDXL, yet diffusion metrics record the configured model ID rather than a reliably resolved backend ID. The SDXL path uses Lab color-statistics matching; its `ip_adapter_or_color_match` label does not mean an IP-Adapter is loaded. Exact effective model/checkpoint revisions and sampling settings were not archived with this result.

**Repair:** use explicit checkpoint-specific presets, benchmark base versus distilled with correctly configured schedules, and record actual loaded repository/revision/backend, quantization, steps, strength, seed, and timing. Preserve T4-safe fp16/quantized execution; the model card's bf16 sample is not a reason to require bf16 here. Do not promise a larger model will fix object identity or geometry.

### 7. Medium: base reconstruction quality is below the product promise and poorly measured

The original screenshot is already blurry. Live reconstruction uses 7,000 iterations; training code defaults to 30,000. Extraction defaults to a 960-pixel long side. These choices can limit detail, but increasing either blindly is not a demonstrated cure.

The trainer optimizes per-Gaussian RGB without higher-order spherical harmonics, limiting view-dependent appearance. `use_ppisp` imports an optional module and discards it; it does not implement exposure/ISP correction. The SSIM fallback computes global image statistics, not conventional local-window SSIM. `holdout_every` is unused; the reported PSNR/SSIM are taken from the final sampled training view, not a held-out evaluation set.

The [gsplat reference trainer](https://docs.gsplat.studio/main/examples/colmap.html) provides an appropriate implementation/benchmark comparison; [rasterization documentation](https://docs.gsplat.studio/main/apis/rasterization.html) distinguishes per-point colors from spherical-harmonic coefficients. These are comparisons, not evidence that this capture must meet a particular score.

**Repair:** establish a repeatable baseline against gsplat on the same calibrated images, retain per-stage metrics, use a real evaluation split without violating the canonical manifest contract, implement or remove advertised options, then tune image resolution, iteration budget, appearance handling, and density. Inspect depth reliability before using it to override masks.

### 8. Medium: successful execution automatically publishes an unverified room

[`complete_task`](../backend/app/services/queue.py) registers the version and switches the project's active version after worker completion. There is no final gate for new-view rendering error, background degradation, splat extent growth, edit success, or object consistency. `max_pose_drift=0` and frozen parameter equality are useful but insufficient.

[`_upload_scene_bundle`](../worker/room3d_worker/runner.py) does not preserve all stage reports/configuration files. Reconstruct returns no PSNR/SSIM; exported manifests retain primarily count/size metrics, and this project's scene-version metrics are null. Model provenance is incomplete. A subtle extra issue: reapplying the global export filter changed membership for one otherwise frozen background row in this run, confirming checkpoint equality is not final-artifact equality.

**Repair:** stage candidate versions, render matched-camera and novel-view checks, compare unmasked regions against the base render, verify target edit success, and publish only after acceptance. Archive immutable configuration, source commit, model provenance, masks/reasons, previews, targets, reports, and export membership. Compare renders rather than requiring lossless 3D pixels everywhere; alpha blending can legitimately affect boundaries.

### 9. Medium: viewer/UI behavior makes diagnosis and review harder

The review grid shows 64-pixel-high thumbnails, making thin doors and drift difficult to inspect across 210 frames. It displays `auto` rather than failure reasons. A checkbox certifies review but cannot compensate for misleading counts or poor preview visibility. Only the reference-image appearance is previewed before the full apply/rebuild.

[`poseFromManifestFrame`](../frontend/src/components/EditStudio.tsx) applies position, look direction, and up vector but omits camera intrinsics/FOV. The viewer starts with a 60-degree perspective camera and uses its viewport aspect. Therefore “View this camera angle” is not a calibrated pixel-matched render. The two submitted screenshots also have different framing, so they are not a controlled before/after comparison. Nevertheless, the checkpoint evidence independently proves real scene changes.

The renderer's FPS is a speed measure, not a quality score. The collision toggle controls proxy visibility; the walk loop does not enforce collision constraints. Sparse `Points` fallback is an availability feature, not equivalent splat rendering quality.

**Repair:** offer large failure-first mask review, paired before/after views with a locked calibrated camera, a representative edited-view gallery, visible renderer mode, and an actual quality/acceptance status. Implement collision behavior before presenting it as working navigation protection.

### 10. Medium: reproducibility and test delivery are weaker than local pass counts suggest

Re-ran 70 focused Python tests across diffusion/tracking/rebuild/training/export safety and 16 frontend unit/UI tests: all passed. Most relevant model/renderer behavior is mocked or synthetic, which is appropriate for CPU checks but cannot validate photorealistic output. Tests currently do not catch the measured empty-mask coverage, wrong wardrobe identity, and screen-space corruption together.

There are 17 local worker `test_*.py` files but only 8 tracked in Git; both local vertical E2E files are untracked. `.gitignore` excludes `/worker/tests/` and `/tests/`, so many local checks do not ship in a clean clone. Examples include local diffusion/tracking/rebuild tests used in this audit. Python GPU dependencies and git installs are broadly unpinned; a frontend lockfile exists, but the GPU environment/model stack lacks equivalent full provenance. `map_anything` is listed in backend types and a class exists, but it is absent from the default backend registry; VGGT needs separately configured weights/inference integration.

**Repair:** track the intended tests, remove broad test-directory ignore rules, and run a clean-clone CPU suite. Add small regression fixtures for invalid coverage, depth-aware ROI, geometry-preserving recolor, and failed publication. Supplement them with a small fixed real-GPU quality benchmark and final renderer checks. Pin validated dependency/model revisions. Separate implemented adapters from supported deployment paths.

### 11. Deployment concerns separate from the pictured failure

The worker/control-plane HTTP/storage separation, PyAV decoding, immutable parent versions, camera invariance checks, and isolated GPU stages are useful foundations. Device `cuda:0` inside a subprocess assigned physical GPU 1 is expected local device renumbering, not evidence of wrong-GPU execution.

Production configuration must explicitly enforce authentication and non-development signing/worker secrets; defaults currently allow unauthenticated operation. The frontend appends the session bearer token to asset URLs, increasing exposure through URL logging. Prefer scoped expiring asset access and redact query credentials. The naming-based model denylist is not a complete implementation of the repository's licensing allowlist; use approved exact repositories/revisions with reviewed metadata. No claim is made here that a particular saved weight violates its license.

## Recommended repair sequence and acceptance criteria

1. **Prevent another misleading success:** correct mask states/coverage, expose failed views, and keep unverified rebuilds as candidates. Retain the original room as the selectable baseline.
2. **Make this exact edit safe:** use appearance-only optimization and validated object membership; demonstrate that centers, scales, rotations, and initially opacity remain unchanged for recoloring.
3. **Repair tracking and targets:** correct anchors for the two lower doors, validate re-entry and roll-heavy views, and verify the desired blue in several diverse views before rebuild. Explicitly record true object absence rather than silently suppressing failures.
4. **Repair reconstruction measurements:** retain pose/training/depth/export metrics, calibrate comparison renders, and benchmark image resolution/iterations/appearance against a reference trainer.
5. **Tune models after isolation:** compare correctly configured base/distilled 4B presets on fixed masks, cameras, and seeds. Evaluate fp16/NF4 quality separately. A/B changes one factor at a time.
6. **Ship repeatable verification:** track tests and fixture manifests, lock validated runtimes/revisions, archive run evidence, and require final rendered quality checks.

For this recolor, acceptance means: the same two lower doors are selected whenever visible; true absence is distinguished from tracking failure; representative target images agree on the requested appearance; geometry is unchanged; background render quality stays within a stated tolerance; the blue remains localized across a camera sweep; failures remain reviewable and cannot auto-activate a corrupted version.

No runtime implementation, project state, or model weights were changed during this audit. The reproduction script reads SQLite in read-only mode, compares saved artifacts on CPU, and writes only local audit evidence. A fresh real-GPU run, objective final-render measurements, exact effective diffusion backend identification, and an exhaustive production security/license review remain outside the verified scope.
