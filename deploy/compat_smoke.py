#!/usr/bin/env python3
"""Optional local GPU dependency/ABI probe for the Wan train image (vivijure-wan-train #29).

NOT CI. NOT a merge gate. The Plane C `compat-smoke.yml` workflow was RETIRED 2026-08-06:
workstation RTX 4000-class cards are not the realistic operating condition for Wan 2.2
A14B. Authoritative readiness is bake (`build-image.yml`) + live train on **RunPod**.

This module remains for hand-debug inside `docker build --target deps` + `docker run
--gpus all` if an operator chooses to poke ABI/import questions locally. Do not treat a
green local tiny-UNet LoRA as evidence for A14B.

Suites (one per conda env, because the two envs carry DIFFERENT dependency sets):
  --suite gputrain  (aitoolkit env)  torch trio ABI + ai-toolkit imports + tiny LoRA train
  --suite hfhub     (vivijure env)   huggingface_hub API surface this repo actually calls
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "compat-smoke"
AITOOLKIT_DIR = Path(os.environ.get("VIVIJURE_AITOOLKIT_DIR", "/opt/ai-toolkit"))

# The ai-toolkit modules the Wan path actually loads. Importing these pulls the whole
# trainer stack (torch + diffusers + transformers + torchaudio), which is precisely the
# surface a bad trio pin breaks.
AITOOLKIT_MODULES = (
    "toolkit.models.wan21.wan21",
    "extensions_built_in.diffusion_models.wan22.wan22_14b_model",
)


class CheckFailed(RuntimeError):
    """A compat check that did not hold. Never soft-degraded: this script is a gate."""


class Results:
    """Ordered check log; any failure makes the process exit non-zero."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def run(self, name: str, fn) -> object:
        try:
            detail = fn()
        except Exception as e:  # noqa: BLE001 -- a gate reports every failure class
            self.checks.append({
                "check": name,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None
        self.checks.append({"check": name, "ok": True, "detail": detail})
        print(f"OK   {name}: {detail}", flush=True)
        return detail

    @property
    def failed(self) -> list[str]:
        return [c["check"] for c in self.checks if not c["ok"]]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CheckFailed(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- hfhub suite


def suite_hfhub(res: Results) -> None:
    """The huggingface_hub surface THIS repo calls, at whatever version is installed.

    The floor in deploy/requirements.txt is unpinned, so a fresh build already installs the
    newest 1.x; #19 only makes that explicit. The interesting question is not the number,
    it is whether the entry points the repo depends on still exist and still behave.
    """
    # The image sets HF_HUB_OFFLINE=1. The network-forbid seam below must be reached at the
    # HTTP layer, not short-circuited by the offline guard, so clear it BEFORE the import.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)

    import huggingface_hub  # noqa: PLC0415 -- must import after the env is cleared

    res.run("hfhub.version", lambda: huggingface_hub.__version__)

    def _snapshot_download_signature() -> str:
        import inspect

        params = inspect.signature(huggingface_hub.snapshot_download).parameters
        if "local_files_only" not in params:
            raise CheckFailed(
                "snapshot_download lost local_files_only; wan_lora_train.resolve_local_hf_snapshot "
                "and deploy/bake_aitoolkit_offline_paths.py both call it")
        return "local_files_only present"

    res.run("hfhub.snapshot_download(local_files_only=)", _snapshot_download_signature)

    def _offline_error() -> str:
        from huggingface_hub.errors import OfflineModeIsEnabled

        return OfflineModeIsEnabled.__name__

    res.run("hfhub.errors.OfflineModeIsEnabled", _offline_error)

    def _forbid_seam_live() -> str:
        """Drive the blocker in deploy/smoke_train_offline.py itself, not a copy of it.

        That build-time smoke proves the baked image never phones home; it works by
        replacing huggingface_hub.utils._http.get_session. If a hub major moves its HTTP
        layer, that block silently stops blocking and the offline proof becomes a lie. So
        assert the block still bites: a real hub call must raise _NetworkForbidden.
        """
        mod = _load_script_module(
            "smoke_train_offline", REPO_ROOT / "deploy" / "smoke_train_offline.py")
        mod._forbid_hf_http()
        try:
            huggingface_hub.HfApi().model_info("hf-internal-testing/tiny-random-gpt2")
        except Exception as e:  # noqa: BLE001
            chain, cur = [], e
            while cur is not None and len(chain) < 12:
                chain.append(type(cur).__name__)
                cur = cur.__cause__ or cur.__context__
            if "_NetworkForbidden" not in chain:
                raise CheckFailed(
                    "get_session patch no longer intercepts hub HTTP; deploy/smoke_train_offline.py "
                    f"can no longer prove offline operation (raised chain: {chain})") from e
            return f"network-forbid seam bites ({chain[0]})"
        raise CheckFailed(
            "hub call SUCCEEDED with the network-forbid patch installed: the offline proof in "
            "deploy/smoke_train_offline.py is no longer proving anything")

    res.run("hfhub.forbid-http seam still bites", _forbid_seam_live)


# ------------------------------------------------------------------------ gputrain suite


def _fixture_pairs() -> list[tuple[Path, str]]:
    images = sorted(FIXTURE_DIR.glob("*.png"))
    if not images:
        raise CheckFailed(f"no fixture images under {FIXTURE_DIR}")
    pairs = []
    for img in images:
        caption = img.with_suffix(".txt")
        if not caption.is_file():
            raise CheckFailed(f"fixture {img.name} has no caption file")
        pairs.append((img, caption.read_text(encoding="utf-8").strip()))
    return pairs


def suite_gputrain(res: Results, steps: int, out_dir: Path) -> None:
    import torch

    res.run("torch.version", lambda: f"{torch.__version__} (cuda={torch.version.cuda})")

    def _cuda() -> str:
        if not torch.cuda.is_available():
            raise CheckFailed("torch.cuda.is_available() is False on a GPU runner")
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        return f"{name}, {total // (1024 ** 2)} MiB total, {free // (1024 ** 2)} MiB free"

    res.run("torch.cuda device", _cuda)

    def _torchvision_ext() -> str:
        import torchvision
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="cuda")
        scores = torch.tensor([0.9, 0.8], device="cuda")
        kept = nms(boxes, scores, 0.3)
        return f"{torchvision.__version__}, nms kept {kept.numel()} box(es) on cuda"

    res.run("torchvision compiled ext on cuda", _torchvision_ext)

    def _torchaudio_ext() -> str:
        import torchaudio

        wav = torch.randn(1, 16000, device="cuda")
        out = torchaudio.functional.resample(wav, 16000, 8000)
        return f"{torchaudio.__version__}, resample -> {tuple(out.shape)}"

    res.run("torchaudio compiled ext on cuda", _torchaudio_ext)

    def _stack_versions() -> str:
        import diffusers
        import peft
        import safetensors
        import transformers

        return (f"diffusers={diffusers.__version__} transformers={transformers.__version__} "
                f"peft={peft.__version__} safetensors={safetensors.__version__}")

    res.run("trainer stack versions", _stack_versions)

    def _aitoolkit_imports() -> str:
        if not AITOOLKIT_DIR.is_dir():
            raise CheckFailed(f"ai-toolkit checkout missing at {AITOOLKIT_DIR}")
        import importlib

        sys.path.insert(0, str(AITOOLKIT_DIR))
        cwd = Path.cwd()
        os.chdir(AITOOLKIT_DIR)
        try:
            for name in AITOOLKIT_MODULES:
                importlib.import_module(name)
        finally:
            os.chdir(cwd)
        return f"imported {len(AITOOLKIT_MODULES)} ai-toolkit Wan module(s)"

    res.run("ai-toolkit Wan trainer imports", _aitoolkit_imports)

    res.run("fixture dataset", lambda: f"{len(_fixture_pairs())} image/caption pair(s)")

    res.run("real LoRA train on GPU", lambda: _train_lora(steps, out_dir))


def _train_lora(steps: int, out_dir: Path) -> str:
    """A REAL LoRA fine-tune: forward, backward, optimizer step, adapter written to disk.

    Small model on purpose (the card cannot hold A14B), but nothing about the run is
    stubbed: the loss is a diffusion denoising loss over the fixture images, the adapter
    weights must actually MOVE, and the artifact must reload from disk with LoRA keys.
    """
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, UNet2DModel
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file, save_file
    from torchvision.io import read_image

    torch.manual_seed(29)
    device = torch.device("cuda")

    pairs = _fixture_pairs()
    batch = torch.stack([read_image(str(p)).float() / 127.5 - 1.0 for p, _ in pairs]).to(device)

    model = UNet2DModel(
        sample_size=batch.shape[-1],
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    model = inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["to_q", "to_k", "to_v"]),
        model,
    )

    lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
    if not lora_params:
        raise CheckFailed("peft injected no LoRA parameters; target_modules did not match")
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    before = {n: p.detach().clone() for n, p in lora_params.items()}

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(list(lora_params.values()), lr=1e-3)

    losses = []
    for step in range(steps):
        noise = torch.randn_like(batch)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=device)
        noisy = scheduler.add_noise(batch, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(noisy, t).sample
            loss = F.mse_loss(pred.float(), noise)
        if not torch.isfinite(loss):
            raise CheckFailed(f"loss went non-finite at step {step}: {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    moved = [n for n, p in lora_params.items() if not torch.equal(p.detach(), before[n])]
    if not moved:
        raise CheckFailed("no LoRA weight changed across the run; the optimizer did nothing")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "compat_smoke_lora.safetensors"
    save_file({n: p.detach().to(torch.float32).cpu().contiguous() for n, p in lora_params.items()},
              str(artifact))
    reloaded = load_file(str(artifact))
    if set(reloaded) != set(lora_params):
        raise CheckFailed("reloaded adapter key set does not match what was saved")
    size = artifact.stat().st_size
    if size <= 0:
        raise CheckFailed(f"adapter artifact is empty: {artifact}")
    return (f"{steps} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
            f"{len(moved)}/{len(lora_params)} LoRA tensors moved, "
            f"artifact {artifact.name} {size} bytes, {len(reloaded)} keys")


# ------------------------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wan train image dependency compat smoke (#29)")
    ap.add_argument("--suite", required=True, choices=("gputrain", "hfhub"))
    ap.add_argument("--steps", type=int, default=20,
                    help="LoRA train steps for the gputrain suite (10-50 is the smoke range)")
    ap.add_argument("--out", default="/out", help="directory for the LoRA artifact + JSON report")
    args = ap.parse_args(argv)

    if args.suite == "gputrain" and not 1 <= args.steps <= 200:
        raise SystemExit("--steps must be between 1 and 200 for a smoke")

    out_dir = Path(args.out)
    res = Results()
    print(f"== compat smoke suite={args.suite} python={sys.version.split()[0]} ==", flush=True)
    if args.suite == "hfhub":
        suite_hfhub(res)
    else:
        suite_gputrain(res, args.steps, out_dir)

    report = {"suite": args.suite, "ok": not res.failed, "checks": res.checks}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"compat-smoke-{args.suite}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"note: could not write report to {out_dir}: {e}", flush=True)

    if res.failed:
        print(f"== compat smoke FAILED: {res.failed} ==", flush=True)
        return 1
    print(f"== compat smoke PASSED ({len(res.checks)} checks) ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
