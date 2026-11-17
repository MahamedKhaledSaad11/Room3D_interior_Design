from __future__ import annotations

import subprocess
import sys

import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.subprocess_runner import run_isolated_stage


def test_failure_keeps_and_prints_stderr_tail(capsys: pytest.CaptureFixture[str]) -> None:
    stderr = "feature_extraction.cc:445 warning\n" + "x" * 5000 + "\nFATAL: allocation failed\n"
    script = f"import sys; sys.stderr.write({stderr!r}); sys.exit(9)"
    with pytest.raises(StageExecutionError) as error:
        run_isolated_stage([sys.executable, "-c", script], None, None, 5)
    assert str(error.value).split("; stderr tail=", 1)[1] == stderr[-3500:]
    assert capsys.readouterr().err.endswith(stderr[-3500:] + "\n")


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
    assert str(error.value).endswith(stderr[-3500:].decode())
    assert "last diagnostic" in capsys.readouterr().err
