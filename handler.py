"""RunPod serverless handler for Vivijure Wan 2.2 A14B character LoRA training.

Job input (same wire shape the control plane already sends to train_lora):
  {
    "action": "train_lora",
    "project": "<project>",
    "bundle_key": "bundles/<project>/...tar.gz",
    "model_family": "wan",           # optional; defaults to wan on this endpoint
    "pretrained_loras": {},          # optional slot -> existing lora key passthrough
    "render_overrides": {},          # ignored here (this endpoint only trains)
    "train_overrides": {}            # optional, allow-listed: batch_size / steps / resolution (#22).
                                     # Absent = the shipped WanLoraTrainConfig defaults. An unknown
                                     # key or an out-of-range value is REFUSED, never dropped.
  }

Returns: { project, lora: { slot: { lora_id_high, lora_id_low, family: "wan" } }, ... }
Both Wan MoE experts must upload or the slot is not recorded as trained.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import runpod

from wan_train.job import JobError, run_job
from wan_train.r2 import R2, R2Config
from wan_train.redact import redact_error_message


def _store():
    return R2(R2Config.from_env())


def _selftest(inp: dict) -> dict:
    """CPU/GPU readiness without R2. Verifies ai-toolkit seam + baked Wan base paths."""
    from wan_train import wan_lora_train as W
    out = {"ok": False, "selftest": True}
    try:
        out["runtime_ready"] = W.wan_train_runtime_ready()
        out["aitoolkit_python"] = W.aitoolkit_python()
        out["wan_base_path"] = W.default_wan_base_path()
        out["ok"] = bool(out["runtime_ready"])
        if not out["ok"]:
            out["error"] = "wan train runtime not ready (check VIVIJURE_AITOOLKIT_PYTHON and baked weights)"
        return out
    except Exception as e:
        out["error"] = redact_error_message(e)
        return out


def handler(job: dict) -> dict:
    inp = (job or {}).get("input") or {}
    if inp.get("selftest"):
        return _selftest(inp)
    if str(inp.get("action", "train_lora")) == "health":
        from wan_train import wan_lora_train as W
        ready = W.wan_train_runtime_ready()
        return {"ok": ready, "action": "health", "runtime_ready": ready}
    job_id = str((job or {}).get("id") or "local")
    try:
        from wan_train import keys
        keys.check_job_id_slug(job_id, what="job_id")
    except ValueError as e:
        return {"ok": False, "error": redact_error_message(e)}
    base = os.environ.get("WAN_TRAIN_WORKDIR")
    if base:
        work = Path(base) / job_id
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="wan-train-"))
    try:
        return run_job(job, store=_store(), workdir=work, job_id=job_id,
                       on_progress=(job or {}).get("progress_update"))
    except JobError as e:
        return {"ok": False, "error": redact_error_message(e)}
    except Exception as e:
        return {"ok": False, "error": redact_error_message(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
