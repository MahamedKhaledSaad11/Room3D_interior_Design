from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402
from app.security import hash_password  # noqa: E402


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("JWT_SECRET", "unit-test-secret-with-enough-length")
    monkeypatch.setenv("WORKER_TOKEN", "test-worker-secret")

    import app.config as cfg
    import app.db as db_mod
    import app.models as models_mod

    cfg.get_settings.cache_clear()
    db_mod.settings = cfg.get_settings()
    engine = db_mod.create_engine(
        db_mod.settings.database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    db_mod.engine = engine
    db_mod.SessionLocal.configure(bind=engine)
    models_mod.Base.metadata.drop_all(engine)
    models_mod.Base.metadata.create_all(engine)

    with db_mod.SessionLocal() as session:
        user_a = models_mod.User(email="owner@test.dev", password_hash=hash_password("pw-owner-123"))
        user_b = models_mod.User(email="other@test.dev", password_hash=hash_password("pw-other-123"))
        session.add_all([user_a, user_b])
        session.flush()
        proj = models_mod.Project(id="proj-s21", owner_id=user_a.id, name="Living Room", status="completed", active_version_id="v1")
        manifest_rel = "proj-s21/v1/manifest.json"
        manifest_file = storage_dir / manifest_rel
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.0.0",
            "project_id": "proj-s21",
            "version_id": "v1",
            "coord_convention": "opencv_colmap",
            "T_viewer_from_colmap": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            "camera_model": "PINHOLE",
            "width": 160,
            "height": 128,
            "intrinsics": {"fx": 160.0, "fy": 160.0, "cx": 80.0, "cy": 64.0},
            "test_every": 0,
            "frames": [
                {"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0, "image_relpath": "images/0.png",
                 "colmap_image_id": 1, "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                {"frame_idx": 1, "orig_video_frame": 1, "timestamp_sec": 0.5, "image_relpath": "images/1.png",
                 "colmap_image_id": 2, "cam_from_world": [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 10], [0, 0, 0, 1]]},
            ],
            "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse_points.json"},
        }
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        session.add(proj)
        session.add(models_mod.SceneVersion(id="v1", project_id="proj-s21", label="v1", manifest_relpath=manifest_rel))
        session.commit()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        monkeypatch.undo()
        cfg.get_settings.cache_clear()


def _auth(client: TestClient, email: str, password: str) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _png_bytes(width: int, height: int, mode: str = "L", color: int | tuple[int, int, int] = 0) -> bytes:
    buf = BytesIO()
    if mode == "L":
        arr = np.full((height, width), int(color), dtype=np.uint8)
        arr[: height // 2, : width // 2] = 255
        img = Image.fromarray(arr)
    else:
        img = Image.new(mode, (width, height), color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_edit_ownership_isolation_and_strict_mask_validation(isolated_client: TestClient) -> None:
    owner_h = _auth(isolated_client, "owner@test.dev", "pw-owner-123")
    other_h = _auth(isolated_client, "other@test.dev", "pw-other-123")

    res = isolated_client.post(
        "/api/projects/proj-s21/edits/resolve-click",
        json={"screenX": 80, "screenY": 64, "worldHitColmap": [0.0, 0.0, 4.0], "nearestFrameIdx": 1},
        headers=owner_h,
    )
    assert res.status_code == 201
    data = res.json()
    assert data["ref_frame_idx"] == 0
    assert data["method"] == "projection"
    edit_id = data["edit_id"]

    assert isolated_client.post(f"/api/edits/{edit_id}/refine", json={"points": [{"x": 10, "y": 10, "label": 1}]}, headers=other_h).status_code == 404
    assert isolated_client.get(f"/api/edits/{edit_id}/frames", headers=other_h).status_code == 404

    bad_rgb = _png_bytes(160, 128, mode="RGB", color=(255, 255, 255))
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_rgb, "image/png")}, headers=owner_h).status_code == 422
    bad_gray = _png_bytes(160, 128, mode="L", color=128)
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_gray, "image/png")}, headers=owner_h).status_code == 422
    bad_dim = _png_bytes(64, 64, mode="L", color=0)
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_dim, "image/png")}, headers=owner_h).status_code == 422

    good_png = _png_bytes(160, 128, mode="L", color=0)
    ok = isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", good_png, "image/png")}, headers=owner_h)
    assert ok.status_code == 200
    assert ok.json()["status"] == "manual"


def test_apply_barrier_completion_failure_cascade_and_idempotency(isolated_client: TestClient) -> None:
    owner_h = _auth(isolated_client, "owner@test.dev", "pw-owner-123")
    worker_h = {"X-Worker-Token": "test-worker-secret"}

    res = isolated_client.post("/api/projects/proj-s21/edits/resolve-click", json={"worldHitColmap": [0.0, 0.0, 4.0]}, headers=owner_h)
    edit_id = res.json()["edit_id"]

    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_1"}, headers=owner_h).status_code == 422

    c0 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    isolated_client.post(f"/api/tasks/{c0['task_id']}/complete", json={"attempt_id": c0["attempt_id"], "result_payload": {"ok": True}}, headers=worker_h)
    late = isolated_client.post(f"/api/tasks/{c0['task_id']}/progress", json={"attempt_id": c0["attempt_id"], "stage": "05_track", "progress_pct": 50.0}, headers=worker_h)
    assert late.status_code == 409

    isolated_client.post(f"/api/edits/{edit_id}/preview-seeds", json={"prompt": "blue velvet sofa", "seeds": [11, 22]}, headers=owner_h)
    c1 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    isolated_client.post(f"/api/tasks/{c1['task_id']}/complete", json={"attempt_id": c1["attempt_id"], "result_payload": {"seeds": []}}, headers=worker_h)

    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_11"}, headers=owner_h).status_code == 422

    isolated_client.post(f"/api/edits/{edit_id}/preview-seeds", json={"prompt": "blue velvet sofa", "seeds": [11, 22]}, headers=owner_h)
    c1_ok = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    isolated_client.post(
        f"/api/tasks/{c1_ok['task_id']}/complete",
        json={"attempt_id": c1_ok["attempt_id"], "result_payload": {"preview_seeds": {"11": f"proj-s21/edits/{edit_id}/previews/seed_11.png"}}},
        headers=worker_h,
    )

    applied = isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_11"}, headers=owner_h)
    assert applied.status_code == 200
    rebuild_id = applied.json()["id"]
    diff_id = applied.json()["diffusion_task_id"]

    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_22"}, headers=owner_h).status_code == 409

    c2 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    assert c2["task_id"] == diff_id

    isolated_client.post(
        f"/api/tasks/{diff_id}/fail",
        json={"attempt_id": c2["attempt_id"], "error_code": "OOM", "error_message": "gpu error", "retryable": False},
        headers=worker_h,
    )
    claim_after_fail = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h)
    assert claim_after_fail.status_code == 204

    proj_state = isolated_client.get("/api/projects/proj-s21", headers=owner_h).json()
    rebuild_task = next(t for t in proj_state["tasks"] if t["id"] == rebuild_id)
    assert rebuild_task["status"] == "failed"
    edits = isolated_client.get("/api/projects/proj-s21/edits", headers=owner_h).json()
    assert edits[0]["status"] == "failed"


def test_2d_frame_picker_resolve_and_edit_session_transitions(isolated_client: TestClient) -> None:
    owner_h = _auth(isolated_client, "owner@test.dev", "pw-owner-123")
    worker_h = {"X-Worker-Token": "test-worker-secret"}

    bad = isolated_client.post(
        "/api/projects/proj-s21/edits/resolve-click",
        json={"frame_idx": 99, "u_norm": 0.5, "v_norm": 0.5, "base_version_id": "v1"},
        headers=owner_h,
    )
    assert bad.status_code == 422

    picked = isolated_client.post(
        "/api/projects/proj-s21/edits/resolve-click",
        json={"frame_idx": 1, "u_norm": 0.25, "v_norm": 0.75, "base_version_id": "v1"},
        headers=owner_h,
    )
    assert picked.status_code == 201
    p_data = picked.json()
    assert p_data["method"] == "frame_picker"
    assert p_data["ref_frame_idx"] == 1
    assert p_data["u"] == pytest.approx(40.0)
    assert p_data["v"] == pytest.approx(96.0)
    assert p_data["base_version_id"] == "v1"
    edit_id = p_data["edit_id"]

    # Claim segment_init and verify self-contained payload + transactional state update.
    c0 = isolated_client.post("/api/tasks/claim", json={"worker_id": "sam-1", "capabilities": ["segment_init"]}, headers=worker_h).json()
    assert c0["payload"]["manifest_key"] == "proj-s21/v1/manifest.json"
    assert c0["payload"]["image_relpath"] == "images/1.png"
    assert c0["payload"]["points"][0]["u_norm"] == pytest.approx(0.25)
    isolated_client.post(
        f"/api/tasks/{c0['task_id']}/complete",
        json={"attempt_id": c0["attempt_id"], "result_payload": {"mask_summary": {"1": {"status": "segmented", "mask_key": c0["payload"]["mask_out_key"], "skip_frame": False}}}},
        headers=worker_h,
    )
    assert isolated_client.get(f"/api/projects/proj-s21/edits/{edit_id}", headers=owner_h).json()["status"] == "segmented"

    # Track propagate -> status "tracked"
    isolated_client.post(f"/api/edits/{edit_id}/track", headers=owner_h)
    c1 = isolated_client.post("/api/tasks/claim", json={"worker_id": "sam-1", "capabilities": ["track_propagate"]}, headers=worker_h).json()
    isolated_client.post(
        f"/api/tasks/{c1['task_id']}/complete",
        json={
            "attempt_id": c1["attempt_id"],
            "result_payload": {
                "masks_summary": {
                    "0": {"status": "auto", "mask_key": f"proj-s21/edits/{edit_id}/masks/000000.png", "skip_frame": False},
                    "1": {"status": "auto", "mask_key": f"proj-s21/edits/{edit_id}/masks/000001.png", "skip_frame": False},
                }
            },
        },
        headers=worker_h,
    )
    assert isolated_client.get(f"/api/edits/{edit_id}", headers=owner_h).json()["status"] == "tracked"

    # Diffusion preview -> late completion after mask change must be ignored
    isolated_client.post(
        f"/api/edits/{edit_id}/preview-seeds",
        json={"prompt": "emerald velvet sofa", "seeds": [101, 202], "model_variant": "flux2_klein_4b"},
        headers=owner_h,
    )
    c2_stale = isolated_client.post("/api/tasks/claim", json={"worker_id": "diff-0", "capabilities": ["diffusion_preview"]}, headers=worker_h).json()

    # Per-frame refine targets the requested frame_idx and invalidates stale previews
    refine_f1 = isolated_client.post(
        f"/api/edits/{edit_id}/refine",
        json={"frame_idx": 1, "points": [{"u_norm": 0.4, "v_norm": 0.6, "label": 1}]},
        headers=owner_h,
    )
    assert refine_f1.status_code == 200
    assert isolated_client.get(f"/api/edits/{edit_id}", headers=owner_h).json()["preview_seeds"] == {}

    # Completing c2_stale after mask revision bumped must NOT restore stale previews
    isolated_client.post(
        f"/api/tasks/{c2_stale['task_id']}/complete",
        json={
            "attempt_id": c2_stale["attempt_id"],
            "result_payload": {
                "preview_generation": c2_stale["payload"]["preview_generation"],
                "mask_revision": c2_stale["payload"]["mask_revision"],
                "masks_digest": c2_stale["payload"]["masks_digest"],
                "preview_seeds": {"101": f"proj-s21/edits/{edit_id}/previews/seed_101.png"},
            },
        },
        headers=worker_h,
    )
    stale_check = isolated_client.get(f"/api/edits/{edit_id}", headers=owner_h).json()
    assert stale_check["preview_seeds"] == {}
    assert stale_check["status"] != "preview_ready"

    cr1 = isolated_client.post("/api/tasks/claim", json={"worker_id": "sam-1", "capabilities": ["segment_refine"]}, headers=worker_h).json()
    assert cr1["payload"]["ref_frame_idx"] == 1
    assert cr1["payload"]["mask_out_key"] == f"proj-s21/edits/{edit_id}/masks/000001.png"
    isolated_client.post(
        f"/api/tasks/{cr1['task_id']}/complete",
        json={
            "attempt_id": cr1["attempt_id"],
            "result_payload": {
                "mask_summary": {"frame_idx": 1, "status": "manual", "mask_key": cr1["payload"]["mask_out_key"], "skip_frame": False}
            },
        },
        headers=worker_h,
    )
    isolated_client.post(
        f"/api/edits/{edit_id}/preview-seeds",
        json={
            "prompt": "emerald velvet sofa",
            "negative_prompt": "blurry",
            "seeds": [101, 202],
            "model_variant": "sdxl_inpaint_t4",
        },
        headers=owner_h,
    )
    c2b = isolated_client.post("/api/tasks/claim", json={"worker_id": "diff-0", "capabilities": ["diffusion_preview"]}, headers=worker_h).json()
    isolated_client.post(
        f"/api/tasks/{c2b['task_id']}/complete",
        json={
            "attempt_id": c2b["attempt_id"],
            "result_payload": {
                "preview_generation": c2b["payload"]["preview_generation"],
                "mask_revision": c2b["payload"]["mask_revision"],
                "masks_digest": c2b["payload"]["masks_digest"],
                "preview_seeds": {
                    "101": f"proj-s21/edits/{edit_id}/previews/seed_101.png",
                    "202": f"proj-s21/edits/{edit_id}/previews/seed_202.png",
                },
            },
        },
        headers=worker_h,
    )
    edit_detail = isolated_client.get(f"/api/projects/proj-s21/edits/{edit_id}", headers=owner_h).json()
    assert edit_detail["status"] == "preview_ready"
    assert "101" in edit_detail["preview_seeds"]

    # Unknown seed or mismatched prompt/model_variant/negative_prompt must be rejected with 422
    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "999"}, headers=owner_h).status_code == 422
    assert isolated_client.post(
        f"/api/edits/{edit_id}/apply",
        json={"chosen_seed_id": "101", "prompt": "different chair"},
        headers=owner_h,
    ).status_code == 422
    assert isolated_client.post(
        f"/api/edits/{edit_id}/apply",
        json={"chosen_seed_id": "101", "model_variant": "flux2_klein_4b"},
        headers=owner_h,
    ).status_code == 422

    # Apply omitting model_variant/negative_prompt inherits preview parameters -> diffusion_apply ("Edited") -> rebuild_week2 ("completed")
    applied_res = isolated_client.post(
        f"/api/edits/{edit_id}/apply",
        json={"chosen_seed_id": "101"},
        headers=owner_h,
    )
    assert applied_res.status_code == 200
    applied = applied_res.json()
    c3 = isolated_client.post("/api/tasks/claim", json={"worker_id": "diff-0", "capabilities": ["diffusion_apply"]}, headers=worker_h).json()
    assert c3["payload"]["preview_key"] == f"proj-s21/edits/{edit_id}/previews/seed_101.png"
    assert c3["payload"]["model_variant"] == "sdxl_inpaint_t4"
    assert c3["payload"]["negative_prompt"] == "blurry"
    isolated_client.post(f"/api/tasks/{c3['task_id']}/complete", json={"attempt_id": c3["attempt_id"], "result_payload": {"ok": True}}, headers=worker_h)
    assert isolated_client.get(f"/api/edits/{edit_id}", headers=owner_h).json()["status"] == "Edited"

    c4 = isolated_client.post("/api/tasks/claim", json={"worker_id": "reb-1", "capabilities": ["rebuild_week2"]}, headers=worker_h).json()
    isolated_client.post(
        f"/api/tasks/{c4['task_id']}/complete",
        json={
            "attempt_id": c4["attempt_id"],
            "result_manifest_key": f"proj-s21/{applied['target_version_id']}/manifest.json",
            "result_payload": {"version_id": applied["target_version_id"], "parent_version_id": "v1"},
        },
        headers=worker_h,
    )
    final_edit = isolated_client.get(f"/api/edits/{edit_id}", headers=owner_h).json()
    assert final_edit["status"] == "completed"
    assert final_edit["output_version_id"] == applied["target_version_id"]


def test_malformed_payload_is_isolated_and_does_not_poison_queue(isolated_client: TestClient) -> None:
    import app.db as db_mod
    from app.models import Project, Task

    with db_mod.SessionLocal() as session:
        session.add(Project(id="proj-bad", name="Malformed Payload Test", status="queued"))
        session.add(Task(id="t-bad-json", project_id="proj-bad", task_type="segment_init", payload_json="{not-json"))
        session.add(
            Task(
                id="t-good",
                project_id="proj-bad",
                task_type="segment_init",
                payload_json=json.dumps({"required_capabilities": None}),
            )
        )
        session.commit()

    worker_h = {"X-Worker-Token": "test-worker-secret"}
    res = isolated_client.post(
        "/api/tasks/claim",
        json={"worker_id": "sam-1", "capabilities": ["segment_init"], "interactive_only": True},
        headers=worker_h,
    )
    assert res.status_code == 200
    assert res.json()["task_id"] == "t-good"

    with db_mod.SessionLocal() as session:
        bad = session.get(Task, "t-bad-json")
        assert bad is not None and bad.status == "failed"


def test_legacy_edit_sessions_schema_upgrades_on_create_all_and_alembic(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.orm import Session

    from app.config import get_settings
    from app.db import Base
    from app.models import EditSession

    # 1) Legacy DB missing new EditSession columns is upgraded automatically by Base.metadata.create_all
    legacy_db = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{legacy_db}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY, email VARCHAR(320), password_hash TEXT, created_at DATETIME)"))
        conn.execute(text("CREATE TABLE projects (id VARCHAR(36) PRIMARY KEY, owner_id VARCHAR(36), name VARCHAR(255), status VARCHAR(32), active_version_id VARCHAR(36), video_relpath TEXT, created_at DATETIME, updated_at DATETIME)"))
        conn.execute(text("CREATE TABLE edit_sessions (id VARCHAR(36) PRIMARY KEY, project_id VARCHAR(36) NOT NULL, base_version_id VARCHAR(36), target_version_id VARCHAR(36), ref_frame_idx INTEGER, prompt TEXT, points_json TEXT, masks_summary_json TEXT, chosen_seed_id VARCHAR(64), status VARCHAR(32) NOT NULL, created_at DATETIME, updated_at DATETIME)"))
        conn.execute(text("INSERT INTO projects (id, name, status) VALUES ('p-leg', 'Legacy', 'completed')"))
        conn.execute(text("INSERT INTO edit_sessions (id, project_id, status) VALUES ('e-leg', 'p-leg', 'segmented')"))
    Base.metadata.create_all(engine)
    upgraded_cols = {c["name"] for c in inspect(engine).get_columns("edit_sessions")}
    assert {"preview_seeds_json", "mask_revision", "preview_generation", "preview_meta_json"}.issubset(upgraded_cols)
    with Session(engine) as session:
        leg = session.get(EditSession, "e-leg")
        assert leg is not None
        assert leg.mask_revision == 0
        assert leg.preview_generation == 0
    engine.dispose()

    # 2) Alembic upgrade from 0001_foundation -> head adds the preview generation columns cleanly
    alembic_db = tmp_path / "alembic_upgrade.db"
    backend_dir = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("ROOM3D_DATABASE_URL", f"sqlite:///{alembic_db}")
    get_settings.cache_clear()
    try:
        cfg = Config()
        cfg.set_main_option("script_location", str(backend_dir / "migrations"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{alembic_db}")
        command.upgrade(cfg, "0001_foundation")
        command.upgrade(cfg, "head")
    finally:
        get_settings.cache_clear()
    alembic_engine = create_engine(f"sqlite:///{alembic_db}")
    alembic_cols = {c["name"] for c in inspect(alembic_engine).get_columns("edit_sessions")}
    assert {"preview_seeds_json", "mask_revision", "preview_generation", "preview_meta_json"}.issubset(alembic_cols)
    alembic_engine.dispose()
