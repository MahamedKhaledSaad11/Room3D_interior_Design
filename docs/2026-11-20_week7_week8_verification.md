# W_7 / W_8 verification and diffusion recovery — 20 November 2026

This review covers [W_7](../W_7.md), [W_8](../W_8.md), and the current repository standard in [AGENTS](../AGENTS.md). Implementation and synthetic tests are evidence of contracts and invariants, not evidence of photorealistic output or a completed dual-T4 benchmark. The older [edit audit](2026-11-19_room_edit_audit.md) concerns a different room and remains useful historical evidence.

## Confirmed failure and measured workload

The supplied worker error says `diffusion_apply` was terminated after exactly 3,600 seconds. The log contains active denoising, including a completed 50-step pass lasting 7m41s and another interrupted at 11/50. Downloading 17 model files took about 76 seconds; download warnings do not explain the entire hour. Heartbeats are independent of GPU work. The dependent rebuild failed because diffusion failed, not because a separate rebuild GPU failure was observed.

Read-only SQLite inspection of task `983c8aa4-2b8c-4972-bb0e-f057b6bb50e6` found 210 mask-summary entries, 194 excluded views, and 16 included nonempty masks. Reference frame 192 already had an approved preview, leaving **15 sequential inference passes**, not 210. At the historical 461 seconds/pass, 15 passes imply approximately **115 minutes of inference**, before setup, upload, and rebuild. This is an extrapolation from one measured pass, not a new benchmark or a completion promise. Four fresh quality previews would add roughly 31 minutes at that same rate; the existing approved reference need not be regenerated when its identity is still valid.

Project `2b489ba3-1e24-4c04-be51-a1086c3e0400` retains original version `1225653e-e2a2-4a05-9a60-2441c7c1a337`. The most recent inspected tracking task, `7b88e9a3-0090-4abd-a62b-e8714af39cc9`, was queued with `Re-queued after worker lease timeout`, last updated 08:42:58 UTC (10:42:58 Cairo). These are a read-only snapshot, not live monitoring. No task was cancelled, retried, or modified by this repair.

The original build's recorded wall time was 1,166.85 seconds: extraction 6.35s, poses 106.50s, training 577.38s, export 1.86s, upload 429.77s (approximately 1.05 GB). Browser upload was about 0.59s. These stages must not be described as slow browser video upload. Per-stage VRAM and visual quality for that build were not established here.

## Repair behavior

- [Stage budgets](../worker/room3d_worker/common/stage_budget.py) distinguish an absolute deadline from a silence deadline. Diffusion gets `min(max_budget, setup_budget + inference_operations × effective_steps × step_budget)`. The production apply path is sequential and counts nonempty included non-reference masks. Preview counts seeds. Fast uses the distilled 4B checkpoint / 4 steps; Quality uses base-4B / 50 steps. Other stages retain their separate configurable absolute budget.
- [Subprocess monitoring](../worker/room3d_worker/common/subprocess_runner.py) drains both output pipes on Windows and Linux. Subprocess output or progress-file growth resets the silence deadline. The worker's HTTP heartbeat does not. A chatty but stuck process still has an absolute deadline. Cancellation terminates the process group/tree. A failed progress/checkpoint relay aborts the stage rather than running invisibly.
- [Checkpoints](../worker/room3d_worker/common/diffusion_checkpoints.py) save a complete PNG and then atomically commit a SHA-256 receipt. The identity includes source image bytes, all mask bytes, mask revision/digest, approved reference bytes, prompt/negative prompt, seed, exclusions, config, repository/revision, pipeline, dtype, and effective quantization. Changing any of these invalidates reuse. The policy deliberately invalidates the entire run on a changed input. Grid groups are resumed only when the whole group is valid.
- The [runner](../worker/room3d_worker/runner.py) uploads completed checkpoint pairs during isolated execution, independently of final edited targets, and downloads them on a later worker/session. A newly created task ID or work directory can reuse matching checkpoints in the same edit. Seed previews also checkpoint. Interrupted denoising steps are recomputed; completed inference outputs can be reused. A sudden machine/session loss can lose the in-flight pass and a checkpoint not yet acknowledged by storage. Backend storage must itself be persistent. `/kaggle/working` alone is not a cross-session guarantee.
- Partial checkpoints live under `<project>/edits/<edit>/diffusion_checkpoints/<identity>/`. They do not create an accepted room version. All target images must complete and upload before diffusion completes; the rebuild prerequisite remains blocked until then. Candidate quality and explicit acceptance gates remain in force.
- [Generation progress](../worker/room3d_worker/common/generation_progress.py) counts actual inference passes and denoising callbacks. It exposes total views needing inference, resumed passes, elapsed time, and ETA extrapolated from measured work. Copied originals do not advance this denominator. Input bundle progress reports received bytes; supported loader stderr reports measured download/load counts. Unsupported loaders show a phase without fabricated percentages. ETA excludes later upload/rebuild and is approximate during text encoding/offload/decoding.
- The official Klein editing path now receives the original image, a visual binary-mask reference, the user prompt, and the approved edited reference as actual conditioning images. Native inpaint adapters receive their explicit mask/reference arguments. The compositor then enforces exact unchanged background pixels. A visual mask supplied to a general editing model is not a claim of native spatial inpainting support or guaranteed object identity. Real image quality must still be inspected.
- SDXL's Lab-statistics transfer remains available as a single-image helper, but is rejected for multi-view propagation: it does not satisfy W_8's edited-reference conditioning requirement. The UI no longer offers it as an equivalent batch fallback. Effective model revisions and schedules remain recorded; no silent switch from Quality to Fast occurs.
- [Session recovery](../frontend/src/components/SessionRecovery.tsx) handles an expired browser token with a login dialog over the existing page. Repeated protected network requests stop until login succeeds, and project polling resumes using the new token. A late 401 from the old token cannot discard a newly authenticated session. Worker credentials and backend token lifetime are unchanged.

## Requirement matrix

| Requirement | Evidence | State / remaining validation |
|---|---|---|
| W_7 login, projects, upload, pull queue, staged progress | [backend routers](../backend/app/routers), [API tests](../backend/tests/test_s12_api.py), [Week 1 vertical test](../tests/e2e/test_s18_week1_vertical.py) | Implemented, local tests; original successful build recorded above. Full new real-video demo still needed. |
| W_7 capped extraction, metadata removal | [PyAV extraction](../worker/room3d_worker/pipeline/s01_extract.py), [tests](../worker/tests/test_extract.py) | Implemented. PyAV takes precedence over the task's ffmpeg suggestion on Windows. |
| W_7 canonical undistorted PINHOLE, mod-16 intrinsics, immutable COLMAP poses | [pose pipeline](../worker/room3d_worker/pipeline/s02_poses.py), [pose tests](../worker/tests/test_poses.py), [handoff](week1_to_week2_handoff.md) | Implemented/tested on fixtures; this repair does not alter coordinates or calibration. |
| W_7 SPZ free camera / sparse Points fallback | [viewer](../frontend/src/components/Viewer3D.tsx), [browser test](../frontend/tests/viewer.spec.ts) | Implemented; headless SwiftShader test passes. Native SPZ library absent locally; built-in SPZ packer tested. Sparse fallback is not equivalent visual quality. |
| W_7 VGGT versus GLOMAP on the SAME real video: registration, runtime, peak VRAM, point-cloud inspection | [benchmark report](week1_benchmark_report.md), [benchmark tooling](../worker/room3d_worker/benchmarks.py) | **Missing comparable real-video measurements.** Synthetic timings and historical preflight data do not fulfill this experiment. Commercially approved VGGT weights/configuration must be selected before a deployment benchmark. |
| W_7 stage runtime / VRAM table, demo, handoff | Build timings above; [handoff](week1_to_week2_handoff.md) | Handoff implemented; full real-video per-stage peak VRAM and recorded demo pending. |
| W_8 2D click, positive/negative corrections, propagation, manual fallback | [tracking](../worker/room3d_worker/pipeline/s05_track.py), [grid](../frontend/src/components/TrackingGrid.tsx), [tracking tests](../worker/tests/test_track.py), [browser flow](../frontend/tests/week2_studio.spec.ts) | Implemented and locally exercised. Real re-entry/occlusion accuracy and warm SAM claim latency need measurement. |
| W_8 approve first seed, then mask/prompt/edited-reference conditioning | [diffusion](../worker/room3d_worker/pipeline/s06_diffusion.py), [recovery tests](../worker/tests/test_diffusion_recovery.py) | Actual argument routing and callbacks tested; real multi-view material/style quality and mask-following pending on T4. Recolor is an extra appearance-only route, not a replacement for this requirement. |
| W_8 mask PNG uint8 0/255 and exact unmasked difference 0 | [diffusion tests](../worker/tests/test_diffusion.py), [recovery tests](../worker/tests/test_diffusion_recovery.py) | Implemented and tested, including resumed outputs. |
| W_8 fixed cameras, rebuild, immutable original, before/after and candidate acceptance | [rebuild](../worker/room3d_worker/pipeline/s07_rebuild.py), [frozen-state tests](../worker/tests/test_rebuild_frozen_state.py), [acceptance tests](../worker/tests/test_edit_acceptance.py), [backend integration](../backend/tests/test_s21_edits.py) | Implemented. Local tests prove prerequisite/retry behavior and activation guards. Final rendered quality still needs real-GPU evaluation. |
| W_8 fine-tune versus scratch rebuild comparison | [rebuild tests](../worker/tests/test_rebuild.py) | Both modes exist; **measured same-scene comparison remains missing**. |
| W_8 dual-T4 memory, quantization, seconds/frame, full demo plus fallback demo | Pinned model registry, T4 dtype tests, runtime metrics, historical failed-run pass timing | GPU 0 diffusion / GPU 1 SAM+3DGS preserved; fp16/NF4/CPU offload retained. **New end-to-end T4 VRAM, quality, runtime and demo evidence pending.** W_8's bf16/FP8 or disallowed model suggestions do not override AGENTS. |

## Restart and recover without uploading/reconstructing again

1. Keep `.local_data`, SQLite, object storage, existing versions and model caches. Stop backend/frontend in their terminals with Ctrl+C. Stop Kaggle Cell 12 before switching code; do not cancel current tasks or reset/clean its working directory automatically.
2. Restart the backend using the same environment/secrets/storage settings:

   ```powershell
   Set-Location E:\Cellula_Technologies\Capastone_Project
   & 'C:/Users/Mohamed/miniconda3/envs/torch_gpu/python.exe' -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```

3. Restart frontend in a second terminal:

   ```powershell
   Set-Location E:\Cellula_Technologies\Capastone_Project\frontend
   npm run dev -- -H 127.0.0.1 -p 3000
   ```

4. Review/commit/push these local changes yourself when ready. This repair did not push or deploy. On Kaggle, preserve local changes and use `git fetch` and `git pull --ff-only` on the branch that actually contains the repair. Verify its commit with `git rev-parse HEAD`, then reinstall editable worker code:

   ```python
   import os, subprocess, sys
   from pathlib import Path
   repo = Path('/kaggle/working/room3d')
   subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
   subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(repo / 'worker')], check=True)
   os.environ['ROOM3D_SOURCE_COMMIT'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
   os.chdir(repo)
   ```

5. Use the updated [notebook Cell 12](../kaggle_bundle/room3d_kaggle_lab.ipynb), or copy its updated daemon cell from the synchronized repository into the already open notebook. A running notebook does not reload cell source after `git pull`. Keep the correct backend URL and existing worker secret, and set `ROOM3D_RUN_DAEMON=1`. Cell 12 now passes a stable work directory and budget settings. Standalone lab Cells 4–11 do not need rerunning to retry an existing UI edit.
6. Open the existing Living room project and choose the original version / **Continue a previous edit**. Inspect the queued tracking task first. After masks/corrections change, review all views again and generate fresh previews: invalidation is intentional. When previews and masks are still valid, reselect the same approved seed and settings and press **Apply to all views & rebuild room** again. The backend creates a fresh diffusion task and its dependent rebuild; matching checkpoint identities are reused automatically. No new video upload or pose reconstruction is needed. A failed prerequisite must have been processed as failed before another apply is allowed.
7. For a simple color change, explicitly choose **Paint / recolor** and review the diverse-view previews. For a generative material/style change, explicitly choose **Generative**, then **Fast / Distilled 4B / 4 steps** or **Quality / Base 4B / 50 steps**. Review the approved reference before batch apply, and inspect the candidate before acceptance. Changing model/preset requires new previews and intentionally invalidates old diffusion outputs. Restoring an edit restores its saved preview settings and seed.

The old failed task predates these checkpoints. The original `/tmp/room3d_work_l5fo1shu/...` directory was on Kaggle and was not accessible here. There is no verified recovery of that hour's partial images. This implementation does not automatically trust unidentified legacy output files.

## Timeout settings

| CLI / environment | Default | Meaning |
|---|---:|---|
| `--stage-timeout-sec` / `ROOM3D_STAGE_TIMEOUT_SEC` | 3600 | Absolute budget for non-diffusion subprocesses. |
| `--stage-stall-timeout-sec` / `ROOM3D_STAGE_STALL_TIMEOUT_SEC` | 900 | Silence deadline for subprocess output/progress, independent of heartbeat. |
| `ROOM3D_DIFFUSION_SETUP_BUDGET_SEC` | 1800 | Setup allowance in the absolute diffusion budget. |
| `ROOM3D_DIFFUSION_STEP_BUDGET_SEC` | 30 | Conservative budget per configured denoising step, not an ETA. |
| `ROOM3D_DIFFUSION_MAX_BUDGET_SEC` | 36000 | Hard upper cap, even with active logs/progress; adjust below actual remaining runtime/quota. This is not a statement of Kaggle's session limit. |
| `--work-dir` / notebook `ROOM3D_WORK_DIR` | notebook: `/kaggle/working/room3d_worker` | Stable local checkpoints across daemon restarts. Uploaded checkpoint pairs survive only as long as backend object storage does. |

All values must be positive. For the saved 15-pass Quality workload, the default absolute budget is 24,300 seconds (6h45m); a stalled process is still stopped after 900 seconds of silence. This replaces a one-size-fits-all cutoff with workload-based limits and a separate liveness check. Select operational budgets from measured throughput and remaining runtime. A budget exhaustion can be retried using completed checkpoints; it is not an instruction to keep increasing the cap indefinitely.

## Verification and sources

Local verification uses the `torch_gpu` Python environment, synthetic/mocked inference, SQLite temporary databases, and browser tests with SwiftShader. Results: **244 Python tests passed**, **22 Vitest tests passed**, **3 headless Playwright tests passed**, and frontend **TypeScript typecheck / ESLint passed**. Python warnings were from the expected built-in SPZ fallback. Python Ruff was unavailable in the selected environment and is not claimed as a passing check. No real model-quality or VRAM benchmark was run during this repair.

- [Recovery, invalidation, reference-routing and restart tests](../worker/tests/test_diffusion_recovery.py), [subprocess deadline/cancellation tests](../worker/tests/test_stage_subprocess.py), [session recovery tests](../frontend/tests/session_recovery.test.tsx).
- [Official distilled FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [base-4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B): different checkpoints and sampling schedules, both multi-reference capable. Hardware examples in those cards are not T4 measurements for this project.
- [Diffusers pipeline callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback) and [Flux2 pipeline API](https://huggingface.co/docs/diffusers/api/pipelines/flux2): actual denoising callbacks and image inputs.
- [FastAPI JWT authentication](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/): expiring bearer credentials remain validated; this repair recovers the browser session through normal login.
