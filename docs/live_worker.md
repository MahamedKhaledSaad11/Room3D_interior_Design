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
