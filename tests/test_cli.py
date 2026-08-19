"""Test the privrag CLI commands."""

from pathlib import Path

import numpy as np

from privrag.cli import main


def test_attack_runs_without_error(capsys) -> None:
    main(["attack", "patient diabetes medication 12345"])
    out = capsys.readouterr().out
    assert "Confidence reduction" in out


def test_benchmark_runs_and_reports_pass(capsys) -> None:
    main(["benchmark", "--samples", "60", "--epsilon", "1.0"])
    out = capsys.readouterr().out
    assert "PASS" in out or "Recall" in out


def test_protect_reads_and_writes_npy(tmp_path, capsys) -> None:
    raw = np.random.default_rng(1).normal(size=(20, 32))
    src = tmp_path / "raw.npy"
    dst = tmp_path / "safe.npy"
    np.save(str(src), raw)
    main(["protect", "--input", str(src), "--output", str(dst)])
    assert dst.exists()
    protected = np.load(str(dst))
    assert protected.shape == raw.shape
    assert np.allclose(np.linalg.norm(protected, axis=1), 1.0, atol=1e-5)


def test_no_command_prints_help(capsys) -> None:
    main([])
    out = capsys.readouterr().out
    assert "privrag" in out.lower() or "usage" in out.lower()
