"""Structured progress channel (R2 + stdout + optional RunPod hook)."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from . import keys
from .redact import redact_error_message

_COUNTED = ("train_done", "train_step", "upload_done", "wan_train_progress")


class ProgressEmitter:
    def __init__(self, store, project: str, job_id: str, *,
                 on_progress: Callable[[dict], Any] | None = None,
                 log: Callable[[str], Any] = print, clock: Callable[[], float] = time.time):
        self.store = store
        self.project = project
        self.job_id = str(job_id)
        self.on_progress = on_progress
        self._log = log
        self._clock = clock
        self._events: list[dict] = []
        self._snapshot: dict[str, Any] = {
            "project": project, "job_id": self.job_id, "status": "running",
            "started_ts": None, "updated_ts": None,
            "counts": {}, "last_event": None, "error": None,
        }

    def emit(self, event: str, **fields) -> None:
        rec = {"ts": round(self._clock(), 3), "event": event, **fields}
        self._events.append(rec)
        self._update_snapshot(rec)
        self._human(rec)
        self._write()
        self._hook()

    def complete(self, **fields) -> None:
        self._snapshot["status"] = "complete"
        self.emit("complete", **fields)

    def error(self, stage: str, message: object) -> None:
        safe = redact_error_message(message)
        self.emit("error", stage=stage, message=safe)

    def _update_snapshot(self, rec: dict) -> None:
        if self._snapshot["started_ts"] is None:
            self._snapshot["started_ts"] = rec["ts"]
        self._snapshot["updated_ts"] = rec["ts"]
        self._snapshot["last_event"] = rec["event"]
        if rec["event"] in _COUNTED:
            self._snapshot["counts"][rec["event"]] = self._snapshot["counts"].get(rec["event"], 0) + 1

    def _human(self, rec: dict) -> None:
        try:
            self._log("@event " + rec["event"] + " " + json.dumps({k: v for k, v in rec.items() if k != "event"}))
        except Exception:
            pass

    def _write(self) -> None:
        try:
            body = "\n".join(json.dumps(e, separators=(",", ":")) for e in self._events) + "\n"
            self.store.put_bytes(body.encode("utf-8"), keys.progress_log_key(self.project, self.job_id),
                                 content_type="application/x-ndjson")
            snap = json.dumps(self._snapshot, separators=(",", ":")).encode("utf-8")
            self.store.put_bytes(snap, keys.progress_snapshot_key(self.project, self.job_id),
                                 content_type="application/json")
        except Exception:
            pass

    def _hook(self) -> None:
        if not self.on_progress:
            return
        try:
            self.on_progress(dict(self._snapshot))
        except Exception:
            pass


class NullEmitter:
    def emit(self, *a, **k): ...
    def complete(self, **k): ...
    def error(self, *a, **k): ...
