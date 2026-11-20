from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Base, get_db
from app.main import app
from app.models import SceneVersion, Task
from app.services.queue import ConflictError, claim_next_task, complete_task, fail_task, update_progress


@pytest.fixture()
def api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.db'}", local_storage_dir=str(tmp_path / "storage"), max_upload_bytes=1024)
    monkeypatch.setattr("app.routers.projects.get_settings", lambda: settings)
    monkeypatch.setattr("app.routers.scenes.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client, session_factory, settings
    app.dependency_overrides.clear()


def _video(size: int = 32) -> bytes:
    return b"\x00\x00\x00\x18ftypmp42" + b"x" * size


def test_register_login_upload_and_task_creation(api) -> None:
    client, session_factory, _ = api
    registered = client.post("/api/auth/register", json={"email": "user@example.test", "password": "correct horse"})
    assert registered.status_code == 201
    assert client.post("/api/auth/login", json={"email": "user@example.test", "password": "correct horse"}).status_code == 200
    project = client.post("/api/projects", json={"name": "Demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    assert response.status_code == 200
    with session_factory() as session:
        task = session.query(Task).filter_by(project_id=project["id"], task_type="reconstruct_week1").one()
        payload = json.loads(task.payload_json or "{}")
        assert payload["project_id"] == project["id"]
        assert len(payload["version_id"]) == 36
        assert payload["source_video_key"].endswith(f"/input/{payload['version_id']}.mp4")
        assert payload["pose_backend"] == "colmap_global"

    project_two = client.post("/api/projects", json={"name": "Demo Two"}).json()
    res_two = client.post(f"/api/projects/{project_two['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    assert res_two.status_code == 200
    assert res_two.json()["version_id"] != response.json()["version_id"]


def test_only_one_competing_worker_claims(api) -> None:
    client, _, _ = api
    project = client.post("/api/projects", json={"name": "Queue"}).json()
    client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})

    def claim(worker: str):
        return client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": worker, "capabilities": []})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["w1", "w2"]))
    assert sum(result.status_code == 200 for result in results) == 1


def test_stale_completion_and_progress_persist(api) -> None:
    client, session_factory, _ = api
    project = client.post("/api/projects", json={"name": "Progress"}).json()
    client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    claimed = client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": "w1"}).json()
    headers = {"X-Worker-Token": "dev-worker-token"}
    assert client.post(f"/api/tasks/{claimed['task_id']}/progress", headers=headers, json={"attempt_id": claimed["attempt_id"], "stage": "01_extract", "progress_pct": 50}).status_code == 200
    with session_factory() as session:
        task = session.get(Task, claimed["task_id"])
        assert task and task.progress_pct == 7.5
    stale = client.post(f"/api/tasks/{claimed['task_id']}/complete", headers=headers, json={"attempt_id": "stale", "result_payload": {}})
    assert stale.status_code == 409
    for telemetry in ({"phase": "inference", "eta_sec": 0}, {}):
        response = client.post(f"/api/tasks/{claimed['task_id']}/progress", headers=headers, json={
            "attempt_id": claimed["attempt_id"], "stage": "01_extract", "progress_pct": 50, "telemetry": telemetry,
        })
        assert response.status_code == 200
    with session_factory() as session:
        task = session.get(Task, claimed["task_id"])
        assert task is not None and json.loads(task.payload_json or "{}")["telemetry"] == {}


def test_worker_ownership_conflict_rejected(api) -> None:
    client, session_factory, _ = api
    project = client.post("/api/projects", json={"name": "Ownership"}).json()
    client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    claimed = client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": "w1"}).json()
    headers = {"X-Worker-Token": "dev-worker-token"}

    wrong_hb = client.post(
        f"/api/tasks/{claimed['task_id']}/heartbeat",
        headers=headers,
        json={"attempt_id": claimed["attempt_id"], "worker_id": "w2"},
    )
    assert wrong_hb.status_code == 409

    with session_factory() as session:
        with pytest.raises(ConflictError, match="own"):
            update_progress(session, claimed["task_id"], claimed["attempt_id"], "01_extract", 10.0, worker_id="w2")
        with pytest.raises(ConflictError, match="own"):
            complete_task(session, claimed["task_id"], claimed["attempt_id"], {}, worker_id="w2")
        with pytest.raises(ConflictError, match="own"):
            fail_task(session, claimed["task_id"], claimed["attempt_id"], "ERR", "bad", False, worker_id="w2")


def test_failure_message_preserves_diagnostic_tail_for_ui(
    api: tuple[TestClient, sessionmaker[Session], Settings],
) -> None:
    client, session_factory, _ = api
    project = client.post("/api/projects", json={"name": "Failure diagnostics"}).json()
    client.post(f"/api/projects/{project['id']}/upload",
                files={"file": ("room.mp4", _video(), "video/mp4")})
    headers = {"X-Worker-Token": "dev-worker-token"}
    claimed = client.post("/api/tasks/claim", headers=headers, json={"worker_id": "w1"}).json()
    diagnostic = "feature_extraction.cc:445 warning\n" + "x" * 5000 + "\nFATAL: allocation failed"
    response = client.post(f"/api/tasks/{claimed['task_id']}/fail", headers=headers, json={
        "attempt_id": claimed["attempt_id"], "error_code": "StageExecutionError",
        "error_message": diagnostic, "retryable": False,
    })
    assert response.status_code == 200
    with session_factory() as session:
        task = session.get(Task, claimed["task_id"])
        assert task is not None and task.status == "failed"
        assert task.message == diagnostic[-3500:]


def test_settings_defaults_resolve_to_repo_local_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ROOM3D_DATABASE_URL", raising=False)
    defaults = Settings()
    expected_root = Path(__file__).resolve().parents[2] / ".local_data"
    assert defaults.storage_root == expected_root / "storage"
    assert (expected_root / "room3d.db").as_posix() in defaults.database_url


def test_storage_traversal_is_rejected(api) -> None:
    client, _, _ = api
    project = client.post("/api/projects", json={"name": "Storage"}).json()
    response = client.get(f"/api/storage/{project['id']}/version/../secret")
    assert response.status_code in {400, 403, 404}


def test_stale_lease_reclaimed_and_blocked_queue_no_starvation(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, session_factory, settings = api
    monkeypatch.setattr("app.services.queue.get_settings", lambda: settings)
    p1 = client.post("/api/projects", json={"name": "P1"}).json()
    p2 = client.post("/api/projects", json={"name": "P2"}).json()
    client.post(f"/api/projects/{p1['id']}/upload", files={"file": ("r1.mp4", _video(), "video/mp4")})
    client.post(f"/api/projects/{p2['id']}/upload", files={"file": ("r2.mp4", _video(), "video/mp4")})

    # Even if two workers complete with colliding version_id="v1", P2 never overwrites P1
    c1 = client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": "w1"}).json()
    c2 = client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": "w2"}).json()
    assert client.post(
        f"/api/tasks/{c1['task_id']}/complete",
        headers={"X-Worker-Token": "dev-worker-token"},
        json={"attempt_id": c1["attempt_id"], "result_manifest_key": f"{p1['id']}/v1/manifest.json", "result_payload": {"version_id": "v1"}},
    ).status_code == 200
    assert client.post(
        f"/api/tasks/{c2['task_id']}/complete",
        headers={"X-Worker-Token": "dev-worker-token"},
        json={"attempt_id": c2["attempt_id"], "result_manifest_key": f"{p2['id']}/v1/manifest.json", "result_payload": {"version_id": "v1"}},
    ).status_code == 200
    with session_factory() as session:
        assert session.get(SceneVersion, "v1").project_id == c1["project_id"]  # type: ignore[union-attr]
        p2_versions = session.query(SceneVersion).filter_by(project_id=c2["project_id"]).all()
        assert len(p2_versions) == 1 and p2_versions[0].id != "v1" and len(p2_versions[0].id) <= 36

    # Runnable task behind >50 blocked dependency tasks is still claimed
    with session_factory() as session:
        dep_root = Task(id="dep-root", project_id=p1["id"], task_type="diffusion_apply", status="queued", payload_json="{}")
        session.add(dep_root)
        session.flush()
        for i in range(55):
            session.add(
                Task(
                    id=f"blocked-{i}",
                    project_id=p1["id"],
                    task_type="rebuild_week2",
                    status="queued",
                    payload_json=json.dumps({"depends_on_task_id": "dep-root"}),
                )
            )
        session.add(Task(id="runnable-tail", project_id=p1["id"], task_type="rebuild_week2", status="queued", payload_json="{}"))
        session.commit()
        claimed = claim_next_task(session, "w-tail", ["rebuild_week2"], interactive_only=False)
        assert claimed is not None and claimed["task_id"] == "runnable-tail"

        # Stale lease timeout reclaims task back to queued
        stale_row = session.get(Task, "runnable-tail")
        assert stale_row is not None
        stale_row.updated_at = datetime.now(timezone.utc) - timedelta(seconds=settings.task_lease_timeout_sec + 30)
        from app.models import TaskAttempt
        att = session.get(TaskAttempt, claimed["attempt_id"])
        assert att is not None
        att.heartbeat_at = stale_row.updated_at
        session.commit()
        reclaimed = claim_next_task(session, "w-fresh", ["rebuild_week2"], interactive_only=False)
        assert reclaimed is not None and reclaimed["task_id"] == "runnable-tail"


def test_storage_query_token_auth_when_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'auth.db'}",
        local_storage_dir=str(tmp_path / "storage"),
        auth_required=True,
        jwt_secret="test-jwt-secret-long-enough-1234",
    )
    for mod in ("app.routers.projects", "app.routers.scenes", "app.security"):
        monkeypatch.setattr(f"{mod}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            reg = client.post("/api/auth/register", json={"email": "tok@example.test", "password": "secret password"})
            assert reg.status_code == 201
            jwt_token = reg.json()["access_token"]
            auth_h = {"Authorization": f"Bearer {jwt_token}"}
            proj = client.post("/api/projects", json={"name": "Authed Room"}, headers=auth_h).json()
            up = client.post(
                f"/api/projects/{proj['id']}/upload",
                files={"file": ("room.mp4", _video(), "video/mp4")},
                headers=auth_h,
            )
            assert up.status_code == 200
            video_key = up.json()["video_key"]

            assert client.get(f"/api/storage/{video_key}").status_code == 401
            assert client.get(f"/api/storage/{video_key}?token={jwt_token}").status_code == 401
            asset_token = client.get(f"/api/projects/{proj['id']}", headers=auth_h).json()["asset_access"]
            res = client.get(f"/api/storage/{video_key}?token={asset_token}")
            assert res.status_code == 200 and len(res.content) > 0
    finally:
        app.dependency_overrides.clear()
