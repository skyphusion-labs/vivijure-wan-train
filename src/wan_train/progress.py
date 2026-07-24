"""Structured progress channel (R2 + stdout + optional RunPod hook).

Telemetry MUST NEVER throttle training. The emitter runs the R2 writes on a
background daemon thread: `emit()` only appends to the in-memory log and signals
the writer, so a slow or retrying R2 PUT can never fill the trainer subprocess`s
stdout pipe and stall the GPU (wan-train #12: the previous full-blob-per-line
synchronous write did exactly that -- ~90s stalls every ~40 steps, ~32% util).

Two channels, both written off the hot path, both flushed synchronously at
terminal so the harvested envelope is never racy:
  - snapshot `.json` (small, fixed shape): the live pulse, flushed every
    ``snapshot_interval`` seconds.
  - full `.ndjson` event log (grows with the run): the durable audit trail,
    flushed on the slower ``log_interval`` and always at terminal. Not rewritten
    per line anymore.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from . import keys
from .redact import redact_error_message

_COUNTED = ("train_done", "train_step", "upload_done", "wan_train_progress")

# Live snapshot flush cadence (seconds); the full log flushes on a slower beat.
_SNAPSHOT_INTERVAL = 4.0
_LOG_INTERVAL = 30.0
# Bound the terminal join so a wedged R2 PUT cannot hang the handler`s return.
_CLOSE_JOIN_TIMEOUT = 15.0


class ProgressEmitter:
    def __init__(self, store, project: str, job_id: str, *,
                 on_progress: Callable[[dict], Any] | None = None,
                 log: Callable[[str], Any] = print, clock: Callable[[], float] = time.time,
                 snapshot_interval: float = _SNAPSHOT_INTERVAL,
                 log_interval: float = _LOG_INTERVAL):
        self.store = store
        self.project = project
        self.job_id = str(job_id)
        self.on_progress = on_progress
        self._log = log
        self._clock = clock
        self._snapshot_interval = snapshot_interval
        self._log_interval = log_interval
        self._events: list[dict] = []
        self._snapshot: dict[str, Any] = {
            "project": project, "job_id": self.job_id, "status": "running",
            "started_ts": None, "updated_ts": None,
            "counts": {}, "last_event": None, "error": None,
        }
        # Writer-thread coordination. Everything the writer reads (_events,
        # _snapshot, counters, flags) is touched only under _lock.
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closed = False
        self._n_snap_written = -1     # len(_events) at last snapshot flush
        self._n_log_written = 0       # len(_events) at last log flush
        self._last_log_write = 0.0
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="wan-progress-writer", daemon=True)
        self._writer.start()

    # -- hot path (called from the stdout-drain loop): never blocks on I/O --

    def emit(self, event: str, **fields) -> None:
        rec = {"ts": round(self._clock(), 3), "event": event, **fields}
        with self._lock:
            self._events.append(rec)
            self._update_snapshot(rec)
            self._cv.notify()
        self._human(rec)   # local stdout, cheap, off the lock
        self._hook()       # best-effort RunPod callback, off the lock

    def complete(self, **fields) -> None:
        with self._lock:
            self._snapshot["status"] = "complete"
        self.emit("complete", **fields)
        self.close()       # terminal: synchronous durable flush

    def error(self, stage: str, message: object) -> None:
        safe = redact_error_message(message)
        self.emit("error", stage=stage, message=safe)
        self.close()       # terminal: synchronous durable flush

    def close(self) -> None:
        """Stop the writer thread and force a final synchronous flush of both
        channels. Idempotent; safe to call from complete()/error() and again
        from a finally in the caller."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cv.notify()
        self._writer.join(timeout=_CLOSE_JOIN_TIMEOUT)
        self._flush(force=True)

    # -- writer thread: all R2 I/O happens here, never raising --

    def _writer_loop(self) -> None:
        while True:
            with self._lock:
                if not self._closed and len(self._events) == self._n_snap_written:
                    self._cv.wait(timeout=self._snapshot_interval)
                if self._closed:
                    return  # close() owns the authoritative final flush
            self._flush(force=False)

    def _flush(self, *, force: bool) -> None:
        """Capture a consistent copy under the lock, then PUT outside it."""
        snap_bytes = log_bytes = None
        with self._lock:
            n = len(self._events)
            now = self._clock()
            if force or n != self._n_snap_written:
                snap_bytes = json.dumps(self._snapshot, separators=(",", ":")).encode("utf-8")
                self._n_snap_written = n
            if force or (n != self._n_log_written and now - self._last_log_write >= self._log_interval):
                log_bytes = ("\n".join(json.dumps(e, separators=(",", ":")) for e in self._events)
                             + "\n").encode("utf-8")
                self._n_log_written = n
                self._last_log_write = now
        if snap_bytes is not None:
            self._put(snap_bytes, keys.progress_snapshot_key(self.project, self.job_id),
                      "application/json")
        if log_bytes is not None:
            self._put(log_bytes, keys.progress_log_key(self.project, self.job_id),
                      "application/x-ndjson")

    def _put(self, data: bytes, key: str, content_type: str) -> None:
        try:
            self.store.put_bytes(data, key, content_type=content_type)
        except Exception as exc:
            # K3 #10: progress write failures are logged, never raised.
            self._log(f"progress write failed: {type(exc).__name__}")

    # -- snapshot / human channels --

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

    def _hook(self) -> None:
        if not self.on_progress:
            return
        try:
            with self._lock:
                snap = dict(self._snapshot)
            self.on_progress(snap)
        except Exception:
            pass


class NullEmitter:
    def emit(self, *a, **k): ...
    def complete(self, **k): ...
    def error(self, *a, **k): ...
    def close(self, *a, **k): ...
