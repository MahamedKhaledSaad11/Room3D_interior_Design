# Live pull worker

Run the backend locally with SQLite and local storage; no CUDA libraries are needed in FastAPI:

```powershell
$env:WORKER_TOKEN = "your-worker-secret"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
cloudflared tunnel --url http://127.0.0.1:8000
```

In Kaggle, run the installation cells, then set these values before Cell 12:

```python
import os
os.environ["ROOM3D_BACKEND_URL"] = "https://your-tunnel.trycloudflare.com"
os.environ["ROOM3D_WORKER_TOKEN"] = "your-worker-secret"
os.environ["ROOM3D_RUN_DAEMON"] = "1"
```

Use the same worker secret on both machines. Restart the backend after changing its environment.
The notebook's existing `BACKEND_URL` default is overridden by `ROOM3D_BACKEND_URL`.
Cells 4–11 are the standalone lab; they do not have to run before the live daemon.
The daemon does not automatically import an earlier standalone lab scene. Upload a video through
the frontend or use `scripts/import_kaggle_bundle.py` to register an existing bundle first.

The normal daemon starts a separate interactive polling lane (`250 ms` idle interval), preserving
one warm SAM predictor and cached embeddings. Batch stages run in isolated subprocesses:
diffusion on physical GPU 0, tracking and 3DGS on GPU 1. Inside each isolated process the selected
physical GPU is exposed as `cuda:0`. Cold model loading and network latency are additional to the
polling interval. SAM can load the official Hugging Face weights automatically, or use
`ROOM3D_SAM2_CONFIG` and `ROOM3D_SAM2_CHECKPOINT` for an installed checkpoint.

| Logical task | Existing backend task name |
| --- | --- |
| `reconstruct` | `reconstruct_week1` |
| `segment_track` | `track_propagate` |
| `diffusion_preview` | `diffusion_preview` |
| `diffusion_propagate` | `diffusion_apply` |
| `rebuild` | `rebuild_week2` |

Both naming schemes work. Interactive reference clicks use `segment_init` and `segment_refine`.
The backend creates diffusion propagation and rebuild as separate tasks with an explicit
dependency, so rebuild cannot claim work before the edited frames have uploaded.

Progress uses `POST /api/tasks/{id}/progress`: training/rebuild iterations, tracked frames,
generated seeds, edited frames, and individual artifact uploads. The parent relays GPU stage
events from a JSONL file while maintaining the task heartbeat.

Pose subprocesses force `QT_QPA_PLATFORM=offscreen`. When a `colmap` executable is installed,
the worker extracts SIFT features and matches sequential keyframes with four CPU threads each,
then runs `glomap mapper`, or `colmap mapper` when GLOMAP is absent. It detects the installed
COLMAP option names (`SiftExtraction`/`SiftMatching` in older releases,
`FeatureExtraction`/`FeatureMatching` in newer ones). Without the COLMAP executable it uses
PyCOLMAP with the same thread cap. PyCOLMAP is still needed to read and normalize CLI results.
See [COLMAP's CLI](https://colmap.github.io/cli.html) and
[GLOMAP's mapper contract](https://github.com/colmap/glomap#usage).

Failed stages print their command, exit code, and last 3,500 stderr characters in the Kaggle
cell output. Timeouts retain partial stderr too. The backend retains the last 3,500 characters
of the failure message for the UI, with the full error stored on the task attempt.
After pulling an update in Kaggle, stop and rerun Cell 12 so the daemon reloads its modules;
restart the local backend after updating its failure-message handling. Submit a new task for
a failed reconstruction. A passing local regression suite does not replace a live T4 run.

Artifacts go through `StorageDriver` and retain their project-scoped keys. Masks and edited
frames are individual PNGs for direct browser access; `.zip` bundles are also accepted as binary
storage objects without automatic extraction. The storage upload endpoint does not apply the
separate video upload or manual mask limits. Files above 8 MiB use independently retried chunks,
then a size/SHA-256 checked commit. Upload acknowledgements and commits can safely be retried;
local final files are replaced atomically. Small completion receipts remain under `.uploads/`;
successful commits remove their staged parts. Failed uploads retain parts for diagnosis and
should be included in the operator's storage retention cleanup.

Cloudflare still has a per-request limit; FastAPI cannot remove that proxy limit. Chunking keeps
individual worker requests below it. See [Cloudflare upload limits](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/)
and [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/).

In Edit Studio, select an object in 3D or through the 2D frame picker, confirm tracking, and inspect
the stored mask overlays. Use **Mark Object**, **Skip/Include Frame**, or **Upload PNG Mask** to
correct tracking. Generate four seed previews, select one, then apply the edit. Completion loads
the rebuilt version in `Viewer3D` (the repository's splat viewer) and exposes **Before / After**.

Asset transport now uses these routes (repeat `?paths=images&paths=depths` to select components):

- `GET /api/storage/{project_id}/{version_id}/bundle-index` and `bundle.zip`.
- `GET /api/storage/{project_id}/edits/{edit_id}/bundle-index` and `bundle.zip`.
- `POST /api/storage/{project_id}/{version_id}/upload-bundle` and
  `POST /api/storage/{project_id}/edits/{edit_id}/upload-bundle`, with a ZIP `file` form field.

The archive uses lossless DEFLATE level 1; images, masks, float32 depths, and checkpoints are
copied as bytes without decoding or re-encoding. Responses stream from temporary disk files,
keeping memory bounded. Uploads validate names, links, expanded size, and CRCs before publishing.
Existing single-file endpoints remain available. ZIPs above the worker's 8 MiB threshold use the
existing chunk/size/SHA-256 contract and are unpacked at commit; acknowledgement retries are safe.

On Linux the daemon caches assets in `/tmp/room3d_cache/{project_id}/{version_id}` and
`/tmp/room3d_cache/{project_id}/edits/{edit_id}`. `ROOM3D_CACHE_ROOT` overrides the cache root
(Windows defaults to its system temporary directory). Each stage checks one asset inventory,
downloads only components with changed revisions, and copies cached bytes into its own work
directory. Mutable masks are revalidated; conditional ZIP downloads reject a revision race and
refresh the inventory. LRU eviction bounds the cache at 3,000,000,000 bytes, protects active
scene transfers, and bypasses caching when an asset set cannot fit. ZIP staging uses task work
directories, so it does not add unbounded temporary archives to the cache partition. Run one
daemon per cache root. Stage work directories have their own storage lifetime.

New frame/depth/mask outputs use six-digit indices. Legacy five-digit depths and edit PNGs are
mapped to six digits inside the inventory/archive without HTTP fallback probes; original stored
files and older single-file URLs remain available. If both names exist, the canonical six-digit
asset takes precedence. The lossless transport changes do not alter optimizer iterations,
diffusion settings, image pixels, or checkpoint parameters.

Diffusion and rebuild report live 10–95% processing progress, then 95–99% bundle transfer progress;
completion reaches 100%. The subprocess relay coalesces queued events rather than replaying a
long HTTP backlog after GPU work finishes. Edit Studio uses task percentages and stable mask
revision URLs, and review thumbnails load lazily.

Rebuild safeguards are **off by default** to preserve the existing visual behavior. To opt in,
send `rebuild_safeguards: true` and optionally `unmasked_weight: 1.0` (minimum 1.0) to
`POST /api/edits/{edit_id}/apply`, or include those fields in a rebuild stage spec. They require
`masked_finetune`: foreground depth samples define a padded 3D box, only the existing mask ROI
inside it is optimized, and positions are constrained to that box. Background parameters stay
bitwise frozen. After optimization, editable splats with opacity below 0.05 and visible positive
camera depth below 0.35 are pruned. This distance uses reconstruction units, **not guaranteed
meters**. These optional safeguards intentionally change the reconstruction and do not promise
to eliminate all fog artifacts or repair a poorly calibrated base scene.

The local 900-asset transport regression uses four HTTP requests: one upload, one inventory and
ZIP download for the first stage, and one inventory-only check for a second stage. This is a
request-count result, not a measured Kaggle/Cloudflare wall-time speedup. After pulling updates,
restart the backend and the Kaggle Cell 12 daemon to load the new endpoints and cache behavior.

Local verification:

```powershell
$env:PYTHONPATH = "worker;backend"
python -m pytest worker/tests backend/tests
cd frontend
npm ci
npm run typecheck
npm test
npx playwright test tests/week2_studio.spec.ts
```
