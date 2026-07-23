"""Train-only job runner: bundle in -> Wan LoRA train -> dual experts out."""
from __future__ import annotations

from pathlib import Path

from . import wan_lora_train as W
from .contract import Bundle, TrainRequest, TrainResult
from . import keys
from .progress import ProgressEmitter


class JobError(RuntimeError):
    """Malformed job or missing bundle/cast data."""


def _job_key(key: str, *, prefixes: tuple[str, ...], what: str) -> str:
    try:
        return keys.check_job_key(key, prefixes=prefixes, what=what)
    except ValueError as e:
        raise JobError(str(e)) from None


def _trained_wan_slots(store, project: str, slots: list[str]) -> set[str]:
    trained: set[str] = set()
    for slot in slots:
        try:
            if (store.exists(keys.wan_lora_key(project, slot, "high"))
                    and store.exists(keys.wan_lora_key(project, slot, "low"))):
                trained.add(slot)
        except Exception:
            pass
    return trained


def _slots_to_train(req: TrainRequest, bundle: Bundle, store) -> list[str]:
    needed = list(bundle.storyboard.use_characters)
    already = _trained_wan_slots(store, req.project, needed) | set(req.pretrained_loras.keys())
    return [s for s in needed if s not in already]


def run_train_job(
    job: dict,
    *,
    store,
    workdir: Path,
    job_id: str = "local",
    on_progress=None,
) -> dict:
    """End-to-end Wan train_lora job. Returns the control-plane result dict."""
    req = TrainRequest.from_dict(job)
    if req.action != "train_lora":
        raise JobError(f"unsupported action {req.action!r}; this endpoint only handles train_lora")
    family = str(req.model_family or "wan").strip().lower()
    if family == "sdxl":
        raise JobError(
            "model_family:'sdxl' is not supported on the Wan train endpoint; "
            "submit SDXL training to the render endpoint")
    if not W.wan_train_runtime_ready():
        raise JobError(
            "Wan train runtime not ready (missing ai-toolkit env or baked Wan base weights)")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    progress = ProgressEmitter(store, req.project, job_id, on_progress=on_progress)
    progress.emit("started", action=req.action, project=req.project)

    try:
        if not req.bundle_key:
            raise JobError("train_lora: bundle_key is required")
        try:
            keys.check_bundle_key_for_project(req.bundle_key, req.project, what="train_lora: bundle_key")
        except ValueError as e:
            raise JobError(str(e)) from None
        tar = store.get_file(req.bundle_key, workdir / "bundle.tar.gz")
        bundle = Bundle.extract(Path(tar), workdir / "project")

        to_train = _slots_to_train(req, bundle, store)
        ref_errs = []
        for slot in to_train:
            char = bundle.cast.characters.get(slot)
            if char is None:
                ref_errs.append(f"use_characters slot {slot!r} has no entry in the cast registry")
            elif not char.ref_paths:
                ref_errs.append(f"character slot {slot!r} has no reference images; LoRA training will fail")
        if ref_errs:
            raise JobError("invalid train job: " + "; ".join(ref_errs))

        result = TrainResult(project=req.project)
        for slot in to_train:
            char = bundle.cast.characters[slot]
            out_dir = workdir / "loras" / slot

            def line_cb(line: str) -> None:
                if "/" in line and ("it/s" in line or "loss" in line):
                    progress.emit("wan_train_progress", slot=slot, line=line[:200])

            trained = W.train_slot_wan(char, out_dir, progress_cb=line_cb)
            key_high = store.put_file(trained.high_path, keys.wan_lora_key(req.project, slot, "high"))
            key_low = store.put_file(trained.low_path, keys.wan_lora_key(req.project, slot, "low"))
            result.lora[slot] = {"lora_id_high": key_high, "lora_id_low": key_low, "family": "wan"}
            progress.emit("train_done", slot=slot, family="wan", high=key_high, low=key_low)

        for slot, lora_id in req.pretrained_loras.items():
            result.lora.setdefault(slot, {"lora_id": lora_id})

        progress.complete(lora_slots=len(result.lora))
        return result.to_dict()
    except Exception as e:
        progress.error("train", e)
        raise


def run_job(job: dict, *, store, workdir: Path, job_id: str = "local", on_progress=None) -> dict:
    """RunPod-facing entry: unwrap input dict and delegate."""
    payload = job.get("input", job) if isinstance(job, dict) else job
    return run_train_job(payload, store=store, workdir=workdir, job_id=job_id, on_progress=on_progress)
