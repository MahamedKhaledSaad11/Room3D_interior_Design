# Slow / interrupted artifact upload repair — 20 November 2026

This follows the [W_7/W_8 verification report](2026-11-20_week7_week8_verification.md). It repairs transfer volume, resumability and phase reporting; diffusion sampling, mask compositing, calibration, poses and candidate acceptance are unchanged.

## Observed failure

The current edit `00ea5717-05f2-4895-bcbc-cdb6dfb12cf2` reached upload in task `576061eb-496f-4820-a1c1-0ab69b318b79`. Read-only checks found 19 completed inference checkpoints and four seed checkpoints, with valid image SHA-256 receipts. The old worker uploaded all 210 PNG targets, including unchanged originals. Its random upload session `71db059f-b8b2-4388-8670-734747cd3291` had two received 8 MiB parts at 14:02:07 and 14:04:04 Cairo time. Cloudflare logged interrupted requests and repeated attempts on individual parts. The worker later entered another attempt of the same task and continued using the old upload path.

The initial `connection refused` errors at 12:41 Cairo indicate that the local backend was not accepting connections on port 8000 at that time. Later interrupted uploads are a separate transport problem. `claim: 204` is an empty poll; `heartbeat: 200` proves the worker can reach the control plane but does not prove artifact transfer is progressing.

HTTPX's read/write timeouts measure waiting for network operations, not total wall time for a transfer. The existing scalar 120-second timeout alone did not prove that a whole-file deadline caused the cancellation. The supplied logs do not identify the precise network component that cancelled each request. See [HTTPX timeout semantics](https://www.python-httpx.org/advanced/timeouts/) and [Cloudflare origin troubleshooting](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/troubleshoot-tunnels/common-errors/).

## Implemented behavior

- Diffusion uploads only non-skipped edited frames, including the approved reference. For the observed selection, that is **20 PNG images instead of 210**. The remaining 190 views reuse original training images already stored with the base version. This is a file-count reduction, not a measured bandwidth or runtime percentage. Rebuild already supports original-image fallback and preserves excluded-from-edit views.
- Artifact parts default to **1 MiB**, with separately configurable connection, write and response-read deadlines. Each part retries independently. Cancellation remains checked during packaging, hashing, transfer transitions and backoff.
- Large upload IDs derive from the destination, payload SHA-256, size and part size. Bundles sort filenames and use stable ZIP metadata. Recreated archives with the same content and compression runtime can therefore find the same upload session after a worker/task retry, even when local file mtimes or input ordering change.
- The worker queries a worker-authenticated, uncached upload-status endpoint and skips a part only when both its size and SHA-256 match. It checks status after an interrupted acknowledgement too, so a successfully received part need not be resent. Missing or corrupt parts are uploaded again.
- Completion validates the entire payload size and SHA-256 before publishing. A saved completion receipt also checks current published object metadata, so reusing an older payload cannot falsely report success after the destination was overwritten. Existing in-flight legacy clients remain supported.
- Completed diffusion checkpoints downloaded from backend storage are recognized as already published, avoiding another upload of unchanged checkpoint images on each restart.
- Upload telemetry replaces inference telemetry with **confirmed bytes, total bytes, received parts, reused bytes, acknowledgement-based rate, approximate ETA and retry/verification phase**. Sending a request does not count its bytes as confirmed. Verification is distinct from transfer. An upload does not complete the task or release the rebuild prerequisite until completion succeeds.
- The final inference event carries the completed counter even when JSONL events are coalesced. The UI suppresses old inference counters/ETA during upload, including messages from an older worker. Explicit empty telemetry can clear stale fields.

## Apply to the existing Kaggle job

Local source edits do not replace an already running Kaggle daemon. No live project state was edited, no task was stopped, and no push/deploy was performed by this repair.

1. Review/commit/synchronize this code to the branch used by Kaggle. Update backend, frontend and worker together. Preserve `.local_data`, object storage, the existing project/version, model caches and `/kaggle/working/room3d_worker`.
2. If the old task is still looping on upload, use **Pause processing** and wait for its interruption acknowledgement before replacing the worker. Avoid **Stop processing**, which cancels tasks. If it already failed, retain the edit and its checkpoints. Do not re-upload the video or reconstruct poses.
3. Restart the backend from its existing terminal using the same secrets and storage settings:

   ```powershell
   Set-Location E:\Cellula_Technologies\Capastone_Project
   & 'C:/Users/Mohamed/miniconda3/envs/torch_gpu/python.exe' -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
   ```

   In another terminal, verify readiness before starting/reusing the tunnel:

   ```powershell
   (Invoke-WebRequest 'http://127.0.0.1:8000/openapi.json').StatusCode
   ```

   Expect `200`. A running tunnel can be reused while the backend restarts. If starting a new quick tunnel changes its URL, update Kaggle's `ROOM3D_BACKEND_URL`. Disabling TLS verification is not a fix for an HTTP origin refusing connections.

4. Restart frontend with `npm run dev -- -H 127.0.0.1 -p 3000` from the `frontend` directory.
5. Stop the old Kaggle daemon cell after processing is paused. In a separate cell, after the repair is actually present on the remote branch:

   ```python
   import os, subprocess, sys
   from pathlib import Path
   repo = Path('/kaggle/working/room3d')
   subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
   subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(repo / 'worker')], check=True)
   os.environ['ROOM3D_SOURCE_COMMIT'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
   os.environ['ROOM3D_UPLOAD_CHUNK_BYTES'] = '1048576'
   os.environ['ROOM3D_UPLOAD_ATTEMPTS'] = '6'
   os.environ['ROOM3D_UPLOAD_BUDGET_SEC'] = '7200'
   os.chdir(repo)
   print('Worker code:', os.environ['ROOM3D_SOURCE_COMMIT'])
   ```

   Inspect and preserve local changes if Git refuses the pull. Do not reset or clean the Kaggle directory. Compare the printed commit with the synchronized repair commit. Use the updated notebook **Cell 12**, or copy its updated source into the open notebook; `git pull` does not rewrite an already loaded cell. Keep the existing worker secret and backend URL, and use `ROOM3D_RUN_DAEMON=1`.
6. Start Cell 12, then **Resume processing**. A paused task keeps its identity. A terminally failed edit can be reapplied after its failed prerequisite is reflected in the UI. Keep the same masks, approved preview, seed, prompt, model and preset to reuse the 19 completed inference outputs. Model setup may still run to resolve the effective backend; matching completed frames do not need denoising again. Changing the selection/config intentionally invalidates them.

The old random 8 MiB upload session is not automatically adopted by the new content-addressed 1 MiB protocol. Its bytes are left intact. This repair preserves and reuses validated generated images, then starts the smaller edited-only transfer. Subsequent failures/restarts under the new protocol reuse verified parts. There is no claim that the old in-flight process was repaired remotely or that actual Cloudflare throughput has been benchmarked.

## Transfer settings

Set these environment variables before starting the daemon. Updated Cell 12 exposes the first three defaults.

| Environment variable | Default | Meaning |
|---|---:|---|
| `ROOM3D_UPLOAD_CHUNK_BYTES` | 1048576 | Size of each artifact part. Changing it changes upload identity/part boundaries. |
| `ROOM3D_UPLOAD_ATTEMPTS` | 6 | Bounded attempts per operation/part; permanent auth/validation failures are not retried. |
| `ROOM3D_UPLOAD_BUDGET_SEC` | 7200 | Transfer budget checked between operations; confirmed server parts remain after exhaustion. In-flight network operations retain their own deadlines. |
| `ROOM3D_UPLOAD_CONNECT_TIMEOUT_SEC` | 15 | Wait for connecting. |
| `ROOM3D_UPLOAD_WRITE_TIMEOUT_SEC` | 60 | Wait for sending a network chunk, not total transfer duration. |
| `ROOM3D_UPLOAD_READ_TIMEOUT_SEC` | 120 | Wait for response data / acknowledgement. |

Values must be finite/positive where applicable. Smaller parts and reduced payload address the observed retry cost; arbitrary increases to timeouts do not establish a network diagnosis.

## Verification

Tests cover acknowledgement loss, a new worker resuming stored parts, corrupt-part replacement, changed destination contents, stable ZIP reconstruction after mtime/order changes, cancellation without publishing an incomplete artifact, split timeout configuration, edited-only frame upload, bit-identical original fallback, final inference counters and legacy UI telemetry. [Backend transfer tests](../backend/tests/test_artifact_uploads.py), [worker recovery tests](../worker/tests/test_diffusion_recovery.py) and [UI progress tests](../frontend/tests/api_and_progress.test.tsx) run locally without Kaggle or Docker. Results: **250 Python tests, 23 Vitest tests and 3 headless Playwright tests passed**, with passing TypeScript typecheck and ESLint. Final focused transfer/recovery tests also passed after the last adjustments. No real-network speedup or new T4 visual-quality benchmark is claimed.
