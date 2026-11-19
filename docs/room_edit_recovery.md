# Room3D edit recovery and verification

The core repair introduces deterministic paint/recolor, conservative mask evidence and surface membership, reviewable candidate versions, and retained stage timings. It does not establish photorealistic quality or a new dual-T4 runtime benchmark. The original reconstruction defaults remain 960 pixels / 7,000 iterations while a measured reconstruction baseline is developed.

## Restart locally

Stop the existing backend and frontend processes with Ctrl+C in their own terminals. Stop the Kaggle daemon before submitting edits with the new contract. Preserve `.local_data`, the database, all original versions, and model caches.

Backend, from PowerShell:

```powershell
Set-Location E:\Cellula_Technologies\Capastone_Project
$roomPython = 'C:/Users/Mohamed/miniconda3/envs/torch_gpu/python.exe'
& $roomPython -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```powershell
Set-Location E:\Cellula_Technologies\Capastone_Project\frontend
npm ci
npm run dev -- -H 127.0.0.1 -p 3000
```

Open `http://127.0.0.1:3000`. Reuse your existing `WORKER_TOKEN`, database/storage environment settings, and tunnel configuration. If the tunnel URL changes, update `ROOM3D_BACKEND_URL` in Kaggle. No database migration is needed for this change: evidence, quality, and timing fields use existing JSON columns. Deploy/restart backend, frontend, and worker together.

For production, set `ENVIRONMENT=production`, `AUTH_REQUIRED=true`, explicit CORS origins, and distinct non-development `JWT_SECRET` / `WORKER_TOKEN` values of at least 32 characters. Production startup rejects insecure defaults. Browser asset credentials are project-scoped, read-only, and expire after five minutes; session tokens are no longer put into image URLs. Application access logs redact token query values; configure equivalent redaction in any external reverse proxy.

## Synchronize GitHub and Kaggle

No commit, push, remote execution, or deployment was performed by this implementation task. The checkout started on `main`. Review the implementation and intended test files before committing; do not add the unrelated `images/` directory or `.local_data`.

After reviewing the staged changes, use a branch such as:

```powershell
Set-Location E:\Cellula_Technologies\Capastone_Project
git switch -c codex/room-edit-safety
git diff --cached --stat
git diff --stat
# Include the reviewed source changes, new runtime modules, tests, pytest.ini and this guide.
# Keep unrelated images and local room artifacts out of the commit.
git commit -m "Add geometry-preserving recolor and candidate review gates"
git push -u origin codex/room-edit-safety
git rev-parse HEAD
```

If files are not staged in your checkout, stage only the reviewed paths. `git commit` does not include unstaged source changes. The audit document is useful provenance and can be included with this change.

In Kaggle, stop Cell 12. After the code has actually been pushed, synchronize the existing checkout in a separate cell:

```python
import os
import subprocess
import sys
from pathlib import Path

repo = Path('/kaggle/working/room3d')
branch = 'codex/room-edit-safety'
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', branch], check=True)
subprocess.run(['git', '-C', str(repo), 'switch', branch], check=True)
subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only', 'origin', branch], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(repo / 'worker')], check=True)
os.environ['ROOM3D_SOURCE_COMMIT'] = subprocess.check_output(
    ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
print('Worker source:', os.environ['ROOM3D_SOURCE_COMMIT'])
os.chdir(repo)
```

For a newly created remote branch that is not yet local, replace `git switch branch` with `git switch -c branch --track origin/branch`. If Git reports local changes, inspect them and preserve them before resolving the checkout; do not reset or clean the Kaggle directory. Compare the printed commit with your local commit.

Rerun Cell 12 with the existing backend URL, matching worker secret, `ROOM3D_RUN_DAEMON=1`, and dual-T4 accelerator enabled. A worker restart is required; the running daemon cannot discover modified Python code automatically. No regeneration of the notebook or rerun of standalone lab Cells 4–11 is required for the new recolor path. Recolor does not load diffusion weights. Generative editing now needs an installed Diffusers build exposing the official `Flux2KleinPipeline`; a missing integration produces an explicit error rather than silently switching models. Its real T4 execution remains to be validated.

## Recover this particular room

- Project: `4acf56a9-4399-4881-9851-065a6c3d23ea`.
- Use original version **`8e21c588-4f5f-40d8-bd5d-98fd6c729cce`** as the parent of a new edit.
- Preserve failed-looking result **`8c6b8d70-c0e4-4b7f-91f2-6293bcbb7e83`** for comparison. Refreshing it will not rerun processing.
- Old edit **`40be46a6-f9fb-4b7d-9b31-1bb8d570f199`** requires a new tracked/reviewed edit. Its old masks are not trustworthy evidence.

Open the previous “change it to blue color” edit, click **Before**, then **Use Before as active baseline**. This explicitly restores the original active version while preserving both versions. Room-version options show short IDs: the intended original starts `8e21c588`; the failed result starts `8c6b8d70`. Select **Select a different object** to start the new edit from the original.

1. Choose video frame **7**. Add a positive point on **each lower bookcase door**. Add negative points on shelves, handles, or adjacent panels that should remain unchanged. The tall wardrobe is a different object.
2. Track the object. Review unresolved views first in the larger grid. Correct frame **8**, where the doors were visible but previously had an empty mask. Inspect frames **12**, **107**, and **186** carefully. In frame 107, reject a wardrobe selection; mark absence only if the lower doors really are absent/occluded there.
3. Correct additional views from different camera directions and rerun tracking. Corrected masks now become SAM anchors. Empty raw SAM masks with strong depth support for occlusion/out-of-frustum absence can pass automatically; contradictory selections and weak depth remain unresolved. Other empty/rejected results require correction or explicit confirmation of true absence. Do not mark a visible tracking failure as absent merely to unblock progress.
4. Confirm the selection review. Keep **Paint / recolor** selected, choose blue (default `#2563eb`), and adjust strength if desired. The disabled generative prompt is intentionally unused for paint.
5. Click **Preview paint in diverse views**. Inspect both doors and the surrounding furniture in every preview. Confirm the preview-review checkbox, then **Build recolor candidate**.
6. The worker selects surfaces using depth/alpha agreement, two-view support, 80% visible-view consensus, and support from a corrected anchor. If insufficient surfaces survive, correct masks or investigate reconstruction depth; there is no broad-volume fallback.
7. Compare **Before / After** at the same calibrated camera and navigate around the doors. Check for missed door patches, color on other objects, boundary bleed, and background damage. Before/After switching preserves the camera. **I reviewed the camera sweep — accept this room** activates only a candidate with passing GPU checks and unchanged reviewed mask identity.

`quality_failed` leaves the original active. `validation_pending` means real GPU render checks did not run (including injected test backends); it cannot be accepted. Start another corrected candidate rather than trying to activate a failed one. If another room version became active meanwhile, activation is refused until the baseline is reconciled.

## What is measured

The saved audit measured **1,462.3013 seconds** for diffusion apply: 20 new inpaints plus one approved reference, approximately **73 seconds per new generated image**. Rebuild was **77.0158 seconds**. Those numbers do not measure browser upload time. The new recolor path runs **zero diffusion inferences and zero optimization steps**, but still performs tracking, surface rendering, candidate checks, export, and artifact transfer. No new total T4 runtime is established yet.

New runs retain:

- Browser transfer and round-trip time, plus server storage-copy time after multipart parsing. These are separate measurements, not additive stages.
- Extraction, poses, training, export, isolated-stage wall time, and worker wall time. Queue wait includes dependency wait and worker availability.
- Bundle synchronization (inventory/cache/download/unpack/copy), individual fetches, bundle/asset upload, bytes uploaded, and the task report key. Some I/O counters are inclusive; do not sum them blindly with stage wall times.
- Tracking model loading, surface preparation, SAM propagation, raw mask areas, geometry agreement, rejection reasons, and final mask SHA-256 identities. Old wall-clock gaps cannot retrospectively be called GPU compute.
- Actual diffusion repository/revision/pipeline, schedule, dtype/quantization, reference-conditioning method, model load time, generated-image count, and seconds per generated image. Model load time combines cache/network loading and initialization; network-only Hugging Face download profiling remains separate work.
- Training configuration and disjoint held-out frame IDs (every eighth registered view when possible), local-window SSIM, PSNR, geometry invariants, and matched-camera candidate reports.

Open **Processing → Measured stage timings** for task metrics. Worker reports live under `PROJECT/VERSION/reports/ATTEMPT/task_report.json` or `PROJECT/edits/EDIT/reports/ATTEMPT/`. Candidate versions contain `rebuild_metrics.json`, `roi_selection.npz` (the selected base-Gaussian indices), and `quality_previews/*_before.png` / `*_after.png`. Tracking reports archive raw SAM masks alongside the final-mask evidence. Worker source hashes and installed package versions are retained; package versions alone are not a fully locked GPU environment.

## Verification and remaining limits

From the repository root:

```powershell
& 'C:/Users/Mohamed/miniconda3/envs/torch_gpu/python.exe' -m pytest -c pytest.ini -q
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
Set-Location frontend
npx playwright test tests/viewer.spec.ts tests/week2_studio.spec.ts --workers=1
```

Run browser suites sequentially because the test Next server shares `.next-test`. CPU tests use temporary databases, synthetic PyAV videos, and mocked model/rasterizer behavior. They verify contracts and rejection paths, not photorealistic output. The browser tests use SwiftShader. Intended worker and vertical E2E tests are no longer excluded by `.gitignore`.

Verified locally during implementation: **225 Python tests**, **19 frontend unit/UI tests**, **3 standalone headless browser checks**, ESLint, and strict TypeScript. The Python suite includes the synthetic local backend-to-browser vertical test. Native Python `spz` is unavailable in this environment; export tests exercised the built-in SPZ v2 packer. Saved original/result checkpoint hashes still match the audit. Ruff is not installed in this Python environment, so no Ruff result is claimed.

Before claiming commercial visual quality or a speed target, run this corrected-door task on the actual dual T4 worker at a recorded commit. Archive stage metrics, immutable masks, previews, `rebuild_metrics.json`, and a manual novel-view sweep. Acceptance thresholds (background MAE ≤0.015, background p99 ≤0.08 on [0,1] RGB; ≥10% target-error improvement) are conservative starting values requiring real-capture calibration. Depth/alpha and center membership approximate surface contribution; they are not a per-splat semantic or visibility guarantee. Thin, poorly reconstructed doors may fail conservatively.

Higher-order SH training/export, a real PPISP integration, a reference-trainer quality benchmark, true novel-view automated acceptance, and full collision enforcement remain outside this implementation. PPISP now rejects the unsupported enabled flag, and the old collision control is accurately labeled geometry-proxy visibility. MapAnything remains an injected adapter, not a default deployment backend; VGGT needs configured compatible weights/inference. Generative editing remains experimental and has no guaranteed multi-view identity/appearance consistency. New model revisions are pinned but still need GPU qualification; a fully locked GPU dependency environment remains pending.

Sources: [audit](2026-11-19_room_edit_audit.md), [gsplat rasterization/depth/alpha](https://docs.gsplat.studio/main/apis/rasterization.html), [official distilled Klein schedule](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [official base Klein schedule](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B), [official SDXL inpaint model](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), [Diffusers Klein source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py), [OWASP REST security guidance](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html).
