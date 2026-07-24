"""ProgressEmitter contract: telemetry never throttles training (wan-train #12).

The trainer runs `emit()` from its stdout-drain loop; a slow or failing R2 write
must never block that call or raise into it. Terminal (complete/error/close)
flushes both channels synchronously so the harvested envelope is never racy.
"""
import json
import threading
import time

from wan_train import keys
from wan_train.progress import ProgressEmitter


class RecStore:
    def __init__(self):
        self.puts: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put_bytes(self, data, key, *, content_type=None):
        with self._lock:
            self.puts[key] = data
        return key


class BoomStore:
    def put_bytes(self, data, key, *, content_type=None):
        raise RuntimeError("r2 unavailable")


def test_emit_never_blocks_on_a_stuck_r2_put():
    """A wedged put_bytes on the writer thread must not slow emit()."""
    release = threading.Event()

    class StuckStore:
        def __init__(self):
            self.puts = []

        def put_bytes(self, data, key, *, content_type=None):
            release.wait(5.0)      # writer thread parks here
            self.puts.append(key)
            return key

    em = ProgressEmitter(StuckStore(), "neon", "job", snapshot_interval=0.01)
    try:
        t0 = time.perf_counter()
        for i in range(200):
            em.emit("wan_train_progress", slot="A", line=str(i))
        dt = time.perf_counter() - t0
        assert dt < 0.5, f"emit blocked on the stuck put ({dt:.2f}s for 200 emits)"
    finally:
        release.set()
        em.close()


def test_terminal_flush_is_synchronous_and_durable():
    """complete() must leave both channels written by the time it returns."""
    store = RecStore()
    em = ProgressEmitter(store, "neon", "job1")
    em.emit("wan_train_progress", slot="A", line="0/2000")
    em.complete(lora_slots=1)   # synchronous terminal flush

    snap_k = keys.progress_snapshot_key("neon", "job1")
    log_k = keys.progress_log_key("neon", "job1")
    assert snap_k in store.puts, "snapshot not durably written at terminal"
    assert log_k in store.puts, "event log not durably written at terminal"
    snap = json.loads(store.puts[snap_k])
    assert snap["status"] == "complete"
    assert snap["counts"]["wan_train_progress"] == 1
    # the full log carries every event, terminal complete included
    lines = store.puts[log_k].decode().strip().splitlines()
    events = [json.loads(x)["event"] for x in lines]
    assert events[-1] == "complete"


def test_write_failure_is_logged_not_raised():
    """K3 #10: an R2 write failure is logged best-effort, never propagated."""
    logs: list[str] = []
    em = ProgressEmitter(BoomStore(), "neon", "job2", log=logs.append)
    em.emit("train_done", slot="A")
    em.complete(lora_slots=1)   # must not raise despite every put failing
    assert any("progress write failed" in m for m in logs), \
        "expected a logged progress-write failure"
