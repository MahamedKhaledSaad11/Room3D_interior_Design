from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from room3d_worker.kaggle.preflight_runner import validate_notebook_syntax
from room3d_worker.kaggle.worker_runner import generate_worker_notebook


def test_generate_worker_notebook_structure_and_cells(tmp_path: Path) -> None:
    nb_path = generate_worker_notebook(
        output_path=tmp_path / "kaggle" / "room3d_kaggle_lab.ipynb",
        backend_url="https://room3d-demo.example.com/",
        worker_token="tok-secret-123",
    )
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    # 12-cell hybrid architecture (Cells 1..8, 9a, 9b, 10, 11, 12 -> 13 code cells).
    assert len(nb["cells"]) == 13
    assert validate_notebook_syntax(nb) == []
    cell_ids = [c.get("id") for c in nb["cells"]]
    assert all(isinstance(cid, str) and re.fullmatch(r"[A-Za-z0-9-_]{1,64}", cid) for cid in cell_ids)
    assert len(set(cell_ids)) == len(nb["cells"])

    cell_sources = ["".join(c["source"]) for c in nb["cells"]]
    # Cell 1: GPU topology & sm_75 check
    assert "Diffusion (GPU 0)" in cell_sources[0]
    assert "SAM 2.1 + 3DGS (GPU 1)" in cell_sources[0]
    assert "TORCH_CUDA_ARCH_LIST" in cell_sources[0]
    assert '"bf16_required": False' in cell_sources[0]

    # Cell 2: Pinned Apache-2.0 / MIT / BSD dependencies + torch/cuda protection + db discovery
    for pkg in ("numpy>=2.1.0,<2.3", "av", "pycolmap", "gsplat", "sam2", "diffusers", "bitsandbytes", "optimum-quanto", "vggt.git", "huggingface_hub", "einops", "omegaconf", "hydra-core", "iopath", "plyfile"):
        assert pkg in cell_sources[1]
    assert "/tmp/kaggle_constraints.txt" in cell_sources[1]
    assert "--no-deps" in cell_sources[1]
    assert "--no-build-isolation" in cell_sources[1]
    assert 'SAM2_BUILD_CUDA"] = "0"' in cell_sources[1]
    assert 'subprocess.run([sys.executable, "-c", VERIFY_CODE], check=True)' in cell_sources[1]
    assert "ROOM3D_VGGT_MODEL" in cell_sources[1]
    assert "Room3D_interior_Design.git" in cell_sources[1]
    assert "playroom" in cell_sources[1] and "drjohnson" in cell_sources[1]

    # Cell 3: Interactive lab config & subprocess stage runner
    assert "precomputed_colmap" in cell_sources[2]
    assert "MAX_LONG_SIDE = 960" in cell_sources[2]
    assert "MAX_EDIT_FRAMES = 16" in cell_sources[2]
    assert "room3d_worker.stage_cli" in cell_sources[2]
    assert "ROOM3D_VGGT_MODEL" in cell_sources[2]

    # Cells 4..10: Subprocess Week 1 & Week 2 stages
    assert "s01_extract" in cell_sources[3]
    assert '".png", ".jpg", ".jpeg"' in cell_sources[3]
    assert "av.open(VIDEO_PATH" in cell_sources[3]
    assert "s02_poses" in cell_sources[4] and "W_7.md" in cell_sources[4]
    assert "extract_index.json" in cell_sources[4] and "vggt_skipped_or_failed" in cell_sources[4]
    assert 'isinstance(sp_raw, list)' in cell_sources[4] and 'sp_raw.get("points", [])' in cell_sources[4]
    assert "s03_train" in cell_sources[5]
    assert "s04_export" in cell_sources[6]
    assert "s05_track" in cell_sources[7] and "sam2.1_hiera_small.pt" in cell_sources[7] and "MAX_EDIT_FRAMES" in cell_sources[7]
    assert "diffusion_preview" in cell_sources[8] and "resolve_t4_dtype" in cell_sources[8]
    assert "diffusion_apply" in cell_sources[9] and "max_unmasked_diff" in cell_sources[9] and "MAX_EDIT_FRAMES" in cell_sources[9]
    assert "s07_rebuild" in cell_sources[10]

    # Cell 11: Zip artifact bundle
    assert "_room3d_bundle.zip" in cell_sources[11]

    # Cell 12: Optional pull-worker daemon launch with backend URL, token, and heartbeat
    assert "room3d_worker" in cell_sources[12]
    assert "--backend-url" in cell_sources[12]
    assert "https://room3d-demo.example.com" in cell_sources[12]
    assert "tok-secret-123" in cell_sources[12]
    assert "--heartbeat-interval" in cell_sources[12]


def test_worker_notebook_cli_args_parse_with_room3d_worker_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import room3d_worker.__main__ as worker_main

    nb_path = generate_worker_notebook(
        output_path=tmp_path / "worker.ipynb",
        backend_url="https://room3d-demo.example.com/",
        worker_token="tok-secret-123",
        worker_id="kaggle-dual-t4",
    )
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    launch_src = "".join(nb["cells"][-1]["source"])

    captured_cmd: list[str] = []

    def _fake_subproc_run(cmd: list[str], env: dict[str, str], check: bool) -> None:
        _ = (env, check)
        captured_cmd.extend(cmd)

    monkeypatch.setenv("ROOM3D_RUN_DAEMON", "1")
    monkeypatch.setattr(subprocess, "run", _fake_subproc_run)
    exec(compile(launch_src, "<kaggle_cell_12>", "exec"), {})
    assert captured_cmd[1:3] == ["-m", "room3d_worker"]
    cli_args = [*captured_cmd[3:], "--once"]

    seen: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, server_url: str, worker_token: str) -> None:
            seen["server_url"] = server_url
            seen["worker_token"] = worker_token

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class _FakeRunner:
        def __init__(self, _client: object, **kwargs: object) -> None:
            seen.update(kwargs)

        def run_once(self) -> None:
            seen["ran_once"] = True

    monkeypatch.setattr(worker_main, "WorkerClient", _FakeClient)
    monkeypatch.setattr(worker_main, "WorkerRunner", _FakeRunner)
    rc = worker_main.main(cli_args)
    assert rc == 0
    assert seen["server_url"] == "https://room3d-demo.example.com"
    assert seen["worker_token"] == "tok-secret-123"
    assert seen["worker_id"] == "kaggle-dual-t4"
    assert seen["heartbeat_interval_sec"] == 15.0
    assert seen["ran_once"] is True


def test_generate_worker_notebook_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend_url"):
        generate_worker_notebook(tmp_path / "bad.ipynb", "ftp://invalid", "tok")
    with pytest.raises(ValueError, match="worker_token"):
        generate_worker_notebook(tmp_path / "bad.ipynb", "https://api.example.com", "   ")


def test_benchmarks_module_runs_end_to_end(tmp_path: Path) -> None:
    from room3d_worker.benchmarks import format_markdown_report, run_benchmarks, write_benchmark_outputs

    results = run_benchmarks(tmp_path / "bench", num_frames=4, num_gaussians=128)
    assert results["poses"]["manifest_mod16_ok"] is True
    assert results["tracking"]["vos_preserved_without_depth"] is True
    assert results["tracking"]["mean_iou"] > 0.9
    assert results["diffusion"]["max_unmasked_diff"] == 0
    assert results["diffusion"]["ref_method"] == "reference_seed"
    assert results["diffusion"]["empty_mask_skipped"] is True
    assert results["export"]["spz_version"] == 2

    json_out, md_out = write_benchmark_outputs(
        results,
        json_path=tmp_path / "out" / "bench.json",
        markdown_path=tmp_path / "out" / "bench.md",
    )
    assert json_out is not None and json_out.exists()
    assert md_out is not None and md_out.exists()
    assert "Room3D Local Benchmark Report" in format_markdown_report(results)


def test_checked_in_kaggle_lab_notebook_is_valid() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    nb_path = repo_root / "kaggle_bundle" / "room3d_kaggle_lab.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert validate_notebook_syntax(nb) == []
    assert len(nb["cells"]) == 13
    cell_ids = [c.get("id") for c in nb["cells"]]
    assert all(isinstance(cid, str) and re.fullmatch(r"[A-Za-z0-9-_]{1,64}", cid) for cid in cell_ids)
    assert len(set(cell_ids)) == len(nb["cells"])
