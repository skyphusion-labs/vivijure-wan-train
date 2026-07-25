"""Allow-listed per-job training knobs (#22): honored exactly, or refused loudly."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from wan_train import wan_lora_train as W
from wan_train.contract import TrainRequest
from wan_train.job import run_train_job, JobError
from wan_train.knobs import KnobError, effective_knobs, train_config_overrides


_STORYBOARD = {"title": "neon", "use_characters": ["A"],
               "scenes": [{"id": "s1", "prompt": "A", "character_slots": ["A"]}]}


def _bundle_tar(path: Path) -> Path:
    members = {
        "storyboard.yaml": yaml.safe_dump(_STORYBOARD).encode(),
        "characters/registry.json": json.dumps(
            {"characters": {"A": {"name": "Vesper", "prompt": "teal"}}}).encode(),
        "characters/refs/A/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


class _Store:
    def __init__(self, bundle_tar: Path | None = None):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.fetches = 0

    def get_file(self, key, dest):
        self.fetches += 1
        assert self.bundle_tar is not None, "job fetched the bundle when it should have refused first"
        shutil.copy(self.bundle_tar, dest)
        return dest

    def exists(self, key):
        return False

    def put_file(self, path, key, *, content_type=None, metadata=None):
        self.puts.append(key)
        return key

    def put_bytes(self, data, key, *, content_type=None):
        return key


def _trained_pair(char, out_dir, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    hi, lo = out_dir / "h.safetensors", out_dir / "l.safetensors"
    hi.write_bytes(b"h")
    lo.write_bytes(b"l")
    return W.TrainedWanLora(slot=char.slot, high_path=hi, low_path=lo, trigger=char.name,
                            steps=1, rank=32, ref_count=1, base_repo="/b")


# --- the control: absent knobs must reproduce the shipped run exactly ----------------------------

def test_absent_overrides_are_the_shipped_defaults():
    assert train_config_overrides(None) == {}
    assert train_config_overrides({}) == {}
    cfg = W.WanLoraTrainConfig(**train_config_overrides({}))
    assert (cfg.batch_size, cfg.steps, cfg.resolution) == (1, 2000, (512, 768, 1024))
    train = W.build_aitoolkit_config("n", Path("/d"), Path("/o"), cfg)["config"]["process"][0]
    assert train["train"]["batch_size"] == 1
    assert train["train"]["steps"] == 2000
    assert train["datasets"][0]["resolution"] == [512, 768, 1024]


def test_knobs_reach_the_aitoolkit_config():
    cfg = W.WanLoraTrainConfig(
        **train_config_overrides({"batch_size": 2, "steps": 1200, "resolution": [512, 768]}))
    proc = W.build_aitoolkit_config("n", Path("/d"), Path("/o"), cfg)["config"]["process"][0]
    assert proc["train"]["batch_size"] == 2
    assert proc["train"]["steps"] == 1200
    assert proc["datasets"][0]["resolution"] == [512, 768]


def test_resolution_is_normalized_ascending():
    assert train_config_overrides({"resolution": [768, 512]})["resolution"] == (512, 768)


# --- refusal, never a silent drop ---------------------------------------------------------------

def test_unknown_key_is_refused_and_names_the_allow_list():
    with pytest.raises(KnobError, match=r"unsupported key.*low_vram.*allowed.*batch_size"):
        train_config_overrides({"low_vram": True})


@pytest.mark.parametrize("payload, needle", [
    ({"steps": "1200"}, "must be an integer"),
    ({"steps": True}, "must be an integer"),
    ({"steps": 99}, "between 100 and 6000"),
    ({"steps": 6001}, "between 100 and 6000"),
    ({"batch_size": 0}, "between 1 and 8"),
    ({"batch_size": 9}, "between 1 and 8"),
    ({"resolution": 512}, "must be a list of integers"),
    ({"resolution": []}, "must name 1.."),
    ({"resolution": [512, 999]}, "not one of"),
    ({"resolution": [512, 512]}, "repeats bucket"),
    ({"resolution": ["512"]}, "entries must be integers"),
])
def test_malformed_knob_fails_loud(payload, needle):
    with pytest.raises(KnobError, match=needle):
        train_config_overrides(payload)


def test_non_object_payload_is_refused():
    with pytest.raises(KnobError, match="must be an object"):
        train_config_overrides([("steps", 1200)])


# --- the job seam --------------------------------------------------------------------------------

def test_request_passes_the_payload_through_raw():
    # Junk must survive parsing INTACT so the one validation point can refuse it. Coercing it to {}
    # here would run the baseline while the caller believes it ran a variant.
    assert TrainRequest.from_dict({"train_overrides": {"steps": 1200}}).train_overrides == {"steps": 1200}
    assert TrainRequest.from_dict({"train_overrides": "steps=1200"}).train_overrides == "steps=1200"
    assert TrainRequest.from_dict({}).train_overrides is None
    assert train_config_overrides(TrainRequest.from_dict({}).train_overrides) == {}


def test_job_refuses_a_non_object_payload_instead_of_running_the_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    trained = []
    monkeypatch.setattr(W, "train_slot_wan", lambda *a, **k: trained.append(1))
    store = _Store()   # get_file asserts if reached
    with pytest.raises(JobError, match="must be an object"):
        run_train_job({"action": "train_lora", "project": "neon",
                       "bundle_key": "bundles/neon.tar.gz",
                       "train_overrides": "steps=1200"},
                      store=store, workdir=tmp_path / "work")
    assert store.fetches == 0
    assert trained == [], "job trained under the DEFAULTS after junk knobs; that is the baseline wearing the variant label"


def test_job_refuses_bad_knobs_before_touching_the_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    store = _Store()   # get_file asserts if called: a refusal must cost zero download
    with pytest.raises(JobError, match="between 100 and 6000"):
        run_train_job({"action": "train_lora", "project": "neon",
                       "bundle_key": "bundles/neon.tar.gz",
                       "train_overrides": {"steps": 20000}},
                      store=store, workdir=tmp_path / "work")
    assert store.fetches == 0


def test_job_trains_under_the_requested_knobs_and_reports_them(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    seen = {}

    def _capture(char, out_dir, *, config=None, progress_cb=None, runner=None):
        seen["config"] = config
        return _trained_pair(char, out_dir)

    monkeypatch.setattr(W, "train_slot_wan", _capture)
    store = _Store(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_train_job({"action": "train_lora", "project": "neon",
                         "bundle_key": "bundles/neon.tar.gz",
                         "train_overrides": {"batch_size": 2, "steps": 1200,
                                             "resolution": [512, 768]}},
                        store=store, workdir=tmp_path / "work")
    cfg = seen["config"]
    assert cfg is not None, "job ran the trainer with the defaults, not the requested knobs"
    assert (cfg.batch_size, cfg.steps, cfg.resolution) == (2, 1200, (512, 768))
    assert res["train_config"]["batch_size"] == 2
    assert res["train_config"]["steps"] == 1200
    assert res["train_config"]["resolution"] == [512, 768]


def test_default_job_still_reports_the_default_knobs(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    monkeypatch.setattr(W, "train_slot_wan", lambda char, out_dir, **kw: _trained_pair(char, out_dir))
    store = _Store(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_train_job({"action": "train_lora", "project": "neon",
                         "bundle_key": "bundles/neon.tar.gz"},
                        store=store, workdir=tmp_path / "work")
    assert res["train_config"] == {"batch_size": 1, "steps": 2000,
                                   "resolution": [512, 768, 1024], "rank": 32,
                                   "learning_rate": 1e-4}


def test_effective_knobs_reads_the_config_not_the_request():
    assert effective_knobs(W.WanLoraTrainConfig(steps=1500))["steps"] == 1500
