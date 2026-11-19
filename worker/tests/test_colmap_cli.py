from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from room3d_worker import stage_cli
from room3d_worker.common.errors import StageExecutionError
from room3d_worker.pipeline import s02_poses
from room3d_worker.pipeline.s02_poses import ColmapCliBackend, ColmapFailureError, PoseRequest


@pytest.fixture
def request_spec(tmp_path: Path) -> PoseRequest:
    images = tmp_path / "images"
    images.mkdir()
    return PoseRequest(images, tmp_path / "output", 4, 640, 480,
                       {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0})


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("loop_detection", [False, True])
def test_cli_extracts_then_matches_sequentially_headless(
    request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch, modern: bool, loop_detection: bool,
) -> None:
    from dataclasses import replace

    calls: list[list[str]] = []
    request = replace(request_spec, vocab_tree_path=Path("vocab.bin") if loop_detection else None)

    def run(cmd: list[str], *, env: dict[str, str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert env["QT_QPA_PLATFORM"] == "offscreen"
        calls.append(cmd)
        help_text = "--FeatureExtraction.num_threads --FeatureMatching.num_threads" if modern else ""
        return subprocess.CompletedProcess(cmd, 0, help_text if "--help" in cmd else "", "")

    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setattr(subprocess, "run", run)
    backend = ColmapCliBackend("global", "/usr/bin/colmap")
    database = request.output_dir / "database.db"
    backend._extract(None, database, request)
    commands = [cmd for cmd in calls if "--help" not in cmd]
    assert [cmd[1] for cmd in commands] == ["feature_extractor", "sequential_matcher"]
    extraction, matching = (dict(zip(cmd[2::2], cmd[3::2])) for cmd in commands)
    assert extraction["--database_path"] == matching["--database_path"] == str(database)
    assert extraction["--image_path"] == str(request.image_dir)
    assert extraction["--ImageReader.camera_params"] == "500.0,501.0,320.0,240.0"
    assert extraction["--ImageReader.camera_model"] == "PINHOLE"
    for options, prefix in ((extraction, "FeatureExtraction" if modern else "SiftExtraction"),
                            (matching, "FeatureMatching" if modern else "SiftMatching")):
        assert options[f"--{prefix}.num_threads"] == "4"
        assert options[f"--{prefix}.use_gpu"] == "0"
    assert matching["--SequentialMatching.overlap"] == "12"
    assert matching["--SequentialMatching.loop_detection"] == str(int(loop_detection))
    assert ("--SequentialMatching.vocab_tree_path" in matching) == loop_detection


@pytest.mark.parametrize(("mode", "glomap", "expected"), [
    ("global", "/usr/bin/glomap", "/usr/bin/glomap"),
    ("global", None, "/usr/bin/colmap"),
    ("incremental", "/usr/bin/glomap", "/usr/bin/colmap"),
])
def test_cli_mapping_uses_matches_and_undistorts_largest_model(
    request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch,
    mode: Literal["global", "incremental"], glomap: str | None, expected: str,
) -> None:
    sparse = request_spec.output_dir / "raw_sparse"
    database = request_spec.output_dir / "database.db"
    commands: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        for name in ("0", "1"):
            model = sparse / name
            model.mkdir(parents=True)
            (model / "cameras.bin").touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(s02_poses.shutil, "which", lambda name: glomap)
    backend = ColmapCliBackend(mode, "/usr/bin/colmap")
    pycolmap = SimpleNamespace(Reconstruction=lambda path: SimpleNamespace(images=range(int(Path(path).name) + 2)))
    undistorted: list[Path] = []
    monkeypatch.setattr(backend, "_maybe_undistort", lambda module, request, path, result: undistorted.append(path) or result)
    result = backend._map(pycolmap, database, request_spec, sparse)
    assert commands == [[expected, "mapper", "--database_path", str(database),
                         "--image_path", str(request_spec.image_dir), "--output_path", str(sparse)]]
    assert undistorted == [sparse / "1"]
    assert len(result[0].images) == 3


def test_cli_extraction_failure_stops_before_matching(
    request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd[1])
        return subprocess.CompletedProcess(cmd, 0 if "--help" in cmd else -9, "", "killed")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(StageExecutionError, match="exited -9.*killed"):
        ColmapCliBackend("global", "colmap")._extract(None, Path("database.db"), request_spec)
    assert calls == ["feature_extractor", "feature_extractor"]


def test_cli_mapper_rejects_empty_output(request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(s02_poses.shutil, "which", lambda name: None)
    with pytest.raises(ColmapFailureError, match="no binary reconstruction"):
        ColmapCliBackend("global", "colmap")._map(None, Path("database.db"), request_spec, request_spec.output_dir)


@pytest.mark.parametrize("colmap", [None, "/usr/bin/colmap"])
def test_pose_registry_selects_installed_cli(monkeypatch: pytest.MonkeyPatch, colmap: str | None) -> None:
    monkeypatch.setattr(s02_poses.shutil, "which", lambda name: colmap)
    backends = s02_poses._default_backends()
    assert isinstance(backends["colmap_global"], ColmapCliBackend) == bool(colmap)
    assert isinstance(backends["colmap_incremental"], ColmapCliBackend) == bool(colmap)


def test_pycolmap_global_mapping_receives_pipeline_options_only(
    request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = s02_poses.PyColmapBackend("global")
    database = request_spec.output_dir / "database.db"
    sparse = request_spec.output_dir / "raw_sparse"

    def mapping(db: str, images: str, output: str, *, options: dict[str, int]) -> str:
        assert (db, images, output) == (str(database), str(request_spec.image_dir), str(sparse))
        assert options == {"num_threads": 4}
        return "mapped"

    monkeypatch.setattr(backend, "_maybe_undistort", lambda module, request, path, result: result)
    assert backend._global(SimpleNamespace(global_mapping=mapping), database, request_spec, sparse, None) == "mapped"


def test_pycolmap_limits_threads_and_matches_sequentially(request_spec: PoseRequest) -> None:
    extraction = SimpleNamespace(num_threads=-1, sift=SimpleNamespace(max_num_features=8192))
    matching = SimpleNamespace(num_threads=-1)
    reader = SimpleNamespace(camera_model="SIMPLE_RADIAL", camera_params="")
    pairing = SimpleNamespace(overlap=0, loop_detection=True)
    calls: list[str] = []
    pycolmap = SimpleNamespace(
        ImageReaderOptions=lambda: reader,
        FeatureExtractionOptions=lambda: extraction,
        FeatureMatchingOptions=lambda: matching,
        SequentialPairingOptions=lambda: pairing,
        extract_features=lambda *args, **kwargs: calls.append("extract"),
        match_sequential=lambda *args, **kwargs: calls.append("sequential"),
    )
    s02_poses.PyColmapBackend("global")._extract(pycolmap, Path("database.db"), request_spec)
    assert calls == ["extract", "sequential"]
    assert extraction.num_threads == matching.num_threads == 4
    assert extraction.sift.max_num_features == 4096
    assert reader.camera_model == "PINHOLE"
    assert (pairing.overlap, pairing.loop_detection) == (12, False)


def test_stage_cli_sets_offscreen_for_direct_notebook_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")

    def check_environment(spec: dict[str, object]) -> dict[str, object]:
        assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
        return {}

    monkeypatch.setitem(stage_cli._STAGE_DISPATCH, "poses", check_environment)
    assert stage_cli.main(["poses", "--spec", str(spec)]) == 0


@pytest.mark.parametrize("images_in_scene", [False, True])
def test_colmap_fallback_keeps_raw_frames_and_native_destinations_isolated(
    request_spec: PoseRequest, monkeypatch: pytest.MonkeyPatch, images_in_scene: bool,
) -> None:
    source = request_spec.output_dir / "images" if images_in_scene else request_spec.image_dir
    source.mkdir(parents=True, exist_ok=True)
    request = replace(request_spec, image_dir=source, width=650, height=490, frame_records=[
        {"frame_idx": i, "orig_video_frame": i, "timestamp_sec": float(i),
         "image_relpath": f"images/{i}.png"} for i in range(4)
    ])
    originals: dict[str, bytes] = {}
    for i in range(4):
        path = request.image_dir / f"{i}.png"
        Image.new("RGB", (650, 490), (i * 20, 80, 160)).save(path)
        originals[path.name] = path.read_bytes()
    models: dict[str, SimpleNamespace] = {}
    mapped: list[Path] = []
    undistorted: list[Path] = []

    def mapping(database: str, images: str, output: str, **options: object) -> dict[int, SimpleNamespace]:
        assert {p.name: p.read_bytes() for p in Path(images).iterdir()} == originals
        sparse = Path(output)
        assert not any(sparse.iterdir()), "Mapper output from a previous attempt must never be reused"
        mapped.append(sparse)
        count = 3 if len(mapped) == 1 else 4
        camera = SimpleNamespace(model="PINHOLE", width=650, height=490,
                                 params=[520.0, 521.0, 325.0, 245.0])
        registered: dict[int, SimpleNamespace] = {}
        for i in range(count):
            pose = np.eye(4)
            pose[:3, 3] = [-float(i), -2.0, 0.0]
            registered[i + 1] = SimpleNamespace(image_id=i + 1, name=f"{i}.png",
                                                 camera_id=1, cam_from_world=pose)

        def write_binary(output_path: str) -> None:
            target = Path(output_path)
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                (target / filename).write_bytes(f"{count} {camera.width} {camera.params}".encode())

        rec = SimpleNamespace(cameras={1: camera}, images=registered, points3D={
            i: SimpleNamespace(xyz=np.array(xyz), color=(10, 20, 30))
            for i, xyz in enumerate(((0, 0, 0), (1, 0, 0), (0, 0, 1)))
        }, write_binary=write_binary, compute_mean_reprojection_error=lambda: 0.25)
        models[str(sparse)] = rec
        return {0: rec}

    def undistort(output: str, sparse: str, images: str) -> None:
        target = Path(output)
        undistorted.append(target)
        rec = models[sparse]
        (target / "images").mkdir(parents=True)
        for image in rec.images.values():
            # COLMAP's native copy refuses existing files instead of replacing them.
            with (target / "images" / image.name).open("xb") as handle:
                handle.write((Path(images) / image.name).read_bytes())
        rec.write_binary(str(target / "sparse"))
        models[str(target / "sparse")] = rec

    pycolmap = SimpleNamespace(global_mapping=mapping, incremental_mapping=mapping,
                              undistort_images=undistort, Reconstruction=lambda path: models[path])
    monkeypatch.setattr(s02_poses, "_module", lambda name: pycolmap)
    monkeypatch.setattr(s02_poses.PyColmapBackend, "_extract", lambda *args: None)
    backends: dict[s02_poses.BackendName, s02_poses.PoseBackend] = {
        "colmap_global": s02_poses.PyColmapBackend("global"),
        "colmap_incremental": s02_poses.PyColmapBackend("incremental"),
    }
    manifest, _ = s02_poses.estimate_poses(request, backends)
    assert manifest.pose_backend == "colmap_incremental"
    assert (manifest.width, manifest.height) == (640, 480)
    assert manifest.intrinsics.cx == 320.0 and manifest.intrinsics.cy == 240.0
    assert len(set(mapped)) == len(set(undistorted)) == 2
    assert (request.output_dir / "sparse/0/cameras.bin").read_bytes().startswith(b"4 640")
    for i in range(4):
        with Image.open(request.output_dir / f"images/{i}.png") as published:
            assert published.size == (640, 480)
    if not images_in_scene:
        assert {p.name: p.read_bytes() for p in request.image_dir.iterdir()} == originals
        backends["colmap_incremental"].estimate(request)
        assert len(set(mapped)) == len(set(undistorted)) == 3
        assert {p.name: p.read_bytes() for p in request.image_dir.iterdir()} == originals
