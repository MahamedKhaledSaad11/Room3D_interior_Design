from __future__ import annotations

import subprocess
import sys

import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.subprocess_runner import run_isolated_stage


def test_failure_keeps_cause_and_prints_full_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    stderr = "feature_extraction.cc:445 warning\n" + "x" * 5000 + "\nFATAL: allocation failed\n"
    script = f"import sys; sys.stderr.write({stderr!r}); sys.exit(9)"
    with pytest.raises(StageExecutionError) as error:
        run_isolated_stage([sys.executable, "-c", script], None, None, 5)
    assert "Stage command exited 9" in str(error.value)
    assert "FATAL: allocation failed" in str(error.value)
    assert len(str(error.value)) <= 3200
    assert capsys.readouterr().err.endswith(stderr + "\n")


def test_offscreen_overrides_parent_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    result = run_isolated_stage(
        [sys.executable, "-c", "import os; print(os.environ['QT_QPA_PLATFORM'])"],
        None, {"QT_QPA_PLATFORM": "wayland"}, 5,
    )
    assert result.stdout.strip() == "offscreen"


def test_timeout_preserves_partial_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    stderr = b"x" * 5000 + b"last diagnostic"

    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args[0], 5, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(StageExecutionError) as error:
        run_isolated_stage(["colmap", "feature_extractor"], None, None, 5)
    assert str(error.value).endswith("last diagnostic")
    assert len(str(error.value)) <= 3200
    assert "last diagnostic" in capsys.readouterr().err


def test_native_fatal_survives_interleaved_info_logs_and_long_stack(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fatal = "F20261119 14:59:09.29 undistorters.cc:264] Check failed: File exists"
    info = 'I20261119 14:59:09.29 undistorters.cc:264] Copying already undistorted image\n'
    stderr = info * 100 + fatal + "\n" + info * 100 + "    @ native_stack_frame\n" * 300
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, -6, "", stderr))
    with pytest.raises(StageExecutionError) as error:
        run_isolated_stage(["python", "-m", "room3d_worker.stage_cli", "poses"], None, None, 5)
    message = str(error.value)
    assert "Stage command exited -6" in message
    assert fatal in message
    assert "Copying already undistorted" not in message
    assert len("StageExecutionError: " + message) <= 3500
    assert capsys.readouterr().err.endswith(stderr + "\n")
