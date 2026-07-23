"""Per-job workdir isolation for concurrent RunPod jobs."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.modules.setdefault("runpod", mock.MagicMock())

import handler  # noqa: E402


def test_handler_uses_per_job_subdir_when_workdir_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("WAN_TRAIN_WORKDIR", str(tmp_path / "shared"))
    seen: list[Path] = []

    def fake_run_job(job, *, store, workdir, job_id, on_progress=None):
        seen.append(workdir)
        return {"ok": True}

    monkeypatch.setattr(handler, "run_job", fake_run_job)
    monkeypatch.setattr(handler, "_store", lambda: object())

    handler.handler({"id": "job-a", "input": {"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz"}})
    handler.handler({"id": "job-b", "input": {"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz"}})

    assert len(seen) == 2
    assert seen[0] == tmp_path / "shared" / "job-a"
    assert seen[1] == tmp_path / "shared" / "job-b"
    assert seen[0] != seen[1]
