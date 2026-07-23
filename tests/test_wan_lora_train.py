"""Wan 2.2 A14B character-LoRA training (CPU-only unit tests)."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from wan_train import keys
from wan_train import wan_lora_train as W
from wan_train.contract import Character
from wan_train.job import run_train_job, JobError


def test_wan_lora_key_layout_and_guard():
    assert keys.wan_lora_key("neon", "A", "high") == "loras/neon/A/wan_high_noise.safetensors"
    assert keys.wan_lora_key("neon", "A", "low") == "loras/neon/A/wan_low_noise.safetensors"
    with pytest.raises(ValueError, match="high.*low"):
        keys.wan_lora_key("neon", "A", "middle")


def test_caption_carries_trigger():
    ch = Character(slot="hero", name="chk_detective", prompt="weathered")
    assert W.caption_for(ch, "{name}") == "chk_detective"


def test_build_config_dual_expert_bf16():
    cfg = W.WanLoraTrainConfig(rank=32, steps=1500)
    c = W.build_aitoolkit_config("hero", Path("/ds"), Path("/out"), cfg)
    model = c["config"]["process"][0]["model"]
    assert model["model_kwargs"] == {"train_high_noise": True, "train_low_noise": True}
    assert model["quantize"] is False and model["low_vram"] is False


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


def test_harvest_takes_highest_step(tmp_path):
    run = tmp_path / "output" / "hero"
    for step in ("000000250", "000001000"):
        run.mkdir(parents=True, exist_ok=True)
        (run / f"hero_{step}_high_noise.safetensors").write_bytes(b"h")
        (run / f"hero_{step}_low_noise.safetensors").write_bytes(b"l")
    high, low = W.harvest_experts(run, "hero")
    assert high.name == "hero_000001000_high_noise.safetensors"
    assert low.name == "hero_000001000_low_noise.safetensors"


def _fake_runner(config_path, *, cwd, progress_cb=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    proc = cfg["config"]["process"][0]
    name = cfg["config"]["name"]
    run_dir = Path(proc["training_folder"]) / name
    steps = proc["train"]["steps"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for e in ("high", "low"):
        (run_dir / f"{name}_{steps:09d}_{e}_noise.safetensors").write_bytes(b"LORA")


_WAN_STORYBOARD = {"title": "neon", "use_characters": ["A"],
                   "scenes": [{"id": "s1", "prompt": "A", "character_slots": ["A"]}]}


def _wan_bundle_tar(path: Path) -> Path:
    members = {
        "storyboard.yaml": yaml.safe_dump(_WAN_STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {"A": {"name": "Vesper", "prompt": "teal"}}}).encode(),
        "characters/refs/A/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


class _FakeStore:
    def __init__(self, bundle_tar: Path, existing=None):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.existing = existing or set()

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest)
        return dest

    def exists(self, key):
        return key in self.existing

    def put_file(self, path, key, *, content_type=None, metadata=None):
        self.puts.append(key)
        return key

    def put_bytes(self, data, key, *, content_type=None):
        return key


def _trained_pair(char, out_dir, **_kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    hi, lo = out_dir / "h.safetensors", out_dir / "l.safetensors"
    hi.write_bytes(b"h")
    lo.write_bytes(b"l")
    return W.TrainedWanLora(
        slot=char.slot, high_path=hi, low_path=lo, trigger=char.name,
        steps=100, rank=32, ref_count=1, base_repo="/b",
    )


def test_job_rejects_cross_project_bundle_key(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    store = _FakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"))
    with pytest.raises(JobError, match="bundle_key"):
        run_train_job(
            {"action": "train_lora", "project": "neon", "bundle_key": "bundles/victim.tar.gz"},
            store=store, workdir=tmp_path / "work")


def test_bundle_key_matches_project():
    assert keys.bundle_key_matches_project("bundles/neon.tar.gz", "neon")
    assert not keys.bundle_key_matches_project("bundles/victim.tar.gz", "neon")


def test_check_project_slug_rejects_colliding_display_names():
    assert keys.check_project_slug("neon") == "neon"
    with pytest.raises(ValueError, match="canonical slug"):
        keys.check_project_slug("a b")
    with pytest.raises(ValueError, match="canonical slug"):
        keys.check_project_slug("a  b")


def test_job_rejects_non_canonical_project(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    store = _FakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"))
    with pytest.raises(JobError, match="canonical slug"):
        run_train_job(
            {"action": "train_lora", "project": "a b", "bundle_key": "bundles/a_b.tar.gz"},
            store=store, workdir=tmp_path / "work")


def test_safe_extract_rejects_device_member(tmp_path):
    from wan_train.contract import _safe_extract
    dest = tmp_path / "out"
    dest.mkdir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="dev/null")
        info.type = tarfile.CHRTYPE
        info.size = 0
        tf.addfile(info)
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        with pytest.raises(ValueError, match="special file"):
            _safe_extract(tf, dest)


def test_job_uploads_both_experts(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    monkeypatch.setattr(W, "train_slot_wan", _trained_pair)
    store = _FakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_train_job(
        {"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz"},
        store=store, workdir=tmp_path / "work")
    assert keys.wan_lora_key("neon", "A", "high") in store.puts
    assert keys.wan_lora_key("neon", "A", "low") in store.puts
    assert res["lora"]["A"]["family"] == "wan"


def test_job_skips_fully_trained_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    calls = []
    monkeypatch.setattr(W, "train_slot_wan", lambda *a, **k: calls.append(1))
    existing = {keys.wan_lora_key("neon", "A", "high"), keys.wan_lora_key("neon", "A", "low")}
    store = _FakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"), existing=existing)
    run_train_job({"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz"},
                  store=store, workdir=tmp_path / "work")
    assert calls == []
