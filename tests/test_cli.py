from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from kuclab.checkpoint_utils import make_test_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpt")
    path = make_test_checkpoint(out)
    return path


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kuclab.cli", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_prints_bare_label(checkpoint):
    result = run_cli(["hello how are you", "--checkpoint", str(checkpoint), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"0", "1"}
    assert len(result.stdout.strip()) == 1


def test_cli_json_mode(checkpoint):
    result = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["label"] in {"0", "1"}
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["unsafe"] == (payload["label"] == "1")


def test_cli_score_mode(checkpoint):
    result = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--score"])
    assert result.returncode == 0, result.stderr
    label, score = result.stdout.split()
    assert label in {"0", "1"}
    assert 0.0 <= float(score) <= 1.0


def test_cli_long_input_is_truncated(checkpoint):
    result = run_cli(["word " * 400, "--checkpoint", str(checkpoint), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"0", "1"}


def test_threshold_is_respected(checkpoint):
    hot = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--threshold", "0.0"])
    cold = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--threshold", "1.01"])
    assert hot.stdout.strip() == "1"
    assert cold.stdout.strip() == "0"
