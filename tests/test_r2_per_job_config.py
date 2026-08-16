"""CPU tests for the per-job tenant R2 configuration (pooled-endpoint credential).

A SHARED hosted pool trains many tenants. Each job carries its own tenant
credential in the payload `r2` block, or (when the key is omitted) the
dedicated-endpoint `R2_*` env vars. The load-bearing property is
`test_malformed_block_fails_and_never_degrades_to_env`: a present-but-malformed
block must FAIL the job, never fall back to the environment.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from wan_train.r2 import R2, R2Config

# A complete, VALID environment, present in every test below. Malformed-block
# tests assert a refusal WHILE this is available: a negative test against an
# environment that could not have worked anyway would pass for the wrong reason.
ENV = {
    "R2_ENDPOINT": "https://env.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "env-key-id",
    "R2_SECRET_ACCESS_KEY": "env-secret-value",
    "R2_BUCKET": "vivijure",
}

TENANT_SECRET = "tenant-secret-value"
BLOCK = {
    "endpoint": "https://tenant.r2.cloudflarestorage.com",
    "access_key_id": "tenant-key-id",
    "secret_access_key": TENANT_SECRET,
    "bucket": "tenant-bucket",
}


# ------------------------------------------------------------------ fallback (dedicated endpoint)

def test_absent_block_falls_back_to_env():
    """No block in the payload is the dedicated-endpoint path: the four R2_* env vars."""
    cfg = R2Config.from_payload_or_env({"project": "p", "action": "train_lora"}, ENV)
    assert cfg.bucket == "vivijure"
    assert cfg.access_key_id == "env-key-id"
    assert cfg.session_token is None


def test_absent_block_with_incomplete_env_still_raises_the_env_error():
    """The fallback path keeps its own failure mode; it is not swallowed by the new branch."""
    with pytest.raises(RuntimeError, match="R2 config incomplete; missing env"):
        R2Config.from_payload_or_env({"project": "p"}, {"R2_ENDPOINT": "https://x"})


# ------------------------------------------------------------------- preference (pooled endpoint)

def test_present_block_wins_over_a_fully_valid_env():
    """Every field must come from the block. The env here is complete on purpose."""
    cfg = R2Config.from_payload_or_env({"project": "p", "r2": BLOCK}, ENV)
    assert cfg.endpoint == "https://tenant.r2.cloudflarestorage.com"
    assert cfg.access_key_id == "tenant-key-id"
    assert cfg.secret_access_key == TENANT_SECRET
    assert cfg.bucket == "tenant-bucket"
    assert cfg.bucket != ENV["R2_BUCKET"]
    assert cfg.access_key_id != ENV["R2_ACCESS_KEY_ID"]


def test_block_fields_are_stripped_of_surrounding_whitespace():
    cfg = R2Config.from_payload_or_env({"r2": {**BLOCK, "bucket": "  tenant-bucket  "}}, ENV)
    assert cfg.bucket == "tenant-bucket"


# ------------------------------------------------------------- refusal (the load-bearing property)

@pytest.mark.parametrize("block, why", [
    ({}, "empty object"),
    ({k: v for k, v in BLOCK.items() if k != "bucket"}, "missing bucket"),
    ({k: v for k, v in BLOCK.items() if k != "secret_access_key"}, "missing secret"),
    ({k: v for k, v in BLOCK.items() if k != "endpoint"}, "missing endpoint"),
    ({k: v for k, v in BLOCK.items() if k != "access_key_id"}, "missing access key id"),
    ({**BLOCK, "bucket": ""}, "blank bucket"),
    ({**BLOCK, "bucket": "   "}, "whitespace-only bucket"),
    ({**BLOCK, "access_key_id": None}, "null field"),
    ({**BLOCK, "bucket": 7}, "non-string field"),
    ("not-an-object", "string instead of an object"),
    ([BLOCK], "list instead of an object"),
    (None, "explicit null block"),
])
def test_malformed_block_fails_and_never_degrades_to_env(block, why):
    """A PRESENT but malformed block must fail. It must NOT fall back to env.

    The valid ENV is passed in on purpose: the refusal has to hold when the
    fallback would have worked. `None` is in this list deliberately: an
    explicit `"r2": null` is a producer defect, and a null must not quietly
    select the shared credential.
    """
    with pytest.raises(RuntimeError) as exc:
        R2Config.from_payload_or_env({"project": "p", "r2": block}, ENV)
    assert "job R2 config" in str(exc.value), why


def test_a_valid_block_is_accepted_by_the_same_call_the_malformed_ones_hit():
    """Control for the parametrized refusals: that path CAN succeed."""
    cfg = R2Config.from_payload_or_env({"r2": dict(BLOCK)}, ENV)
    assert cfg.bucket == "tenant-bucket"


# --------------------------------------------------------------------------- secret hygiene

@pytest.mark.parametrize("block", [
    {**BLOCK, "bucket": ""},
    {**BLOCK, "endpoint": None},
    {**BLOCK, "session_token": ""},
])
def test_refusal_messages_name_fields_never_values(block):
    """Refusal messages may name fields; they may never carry a credential."""
    with pytest.raises(RuntimeError) as exc:
        R2Config.from_payload_or_env({"r2": block}, ENV)
    msg = str(exc.value)
    for secret in (TENANT_SECRET, BLOCK["access_key_id"], ENV["R2_SECRET_ACCESS_KEY"]):
        assert secret not in msg, f"a credential value reached the error message: {msg!r}"


# --------------------------------------------------------------------------- session token

def test_session_token_is_optional_and_defaults_to_none():
    assert R2Config.from_payload_or_env({"r2": dict(BLOCK)}, ENV).session_token is None


def test_session_token_is_carried_when_present():
    cfg = R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": "tok"}}, ENV)
    assert cfg.session_token == "tok"


@pytest.mark.parametrize("token", ["", "   ", 7, []])
def test_blank_or_non_string_session_token_is_refused(token):
    with pytest.raises(RuntimeError, match="session_token"):
        R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": token}}, ENV)


def test_session_token_reaches_the_boto3_client(monkeypatch):
    """A carried token that never reaches boto3 would auth as a plain key pair."""
    import types

    seen: dict = {}

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **kw: seen.update(kw) or object()
    fake_botocore = types.ModuleType("botocore")
    fake_config_mod = types.ModuleType("botocore.config")
    fake_config_mod.Config = lambda **kw: kw
    fake_botocore.config = fake_config_mod
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config_mod)

    R2(R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": "tok"}}, ENV))._client()
    assert seen["aws_session_token"] == "tok"
    assert seen["aws_access_key_id"] == "tenant-key-id"

    seen.clear()
    R2(R2Config.from_env(ENV))._client()
    assert seen["aws_session_token"] is None, "the env path must pass no session token"


# --------------------------------------------------------------------------- payload stripping

def test_strip_removes_only_the_credential_block():
    payload = {"project": "p", "action": "train_lora", "bundle_key": "bundles/p.tar.gz", "r2": BLOCK}
    stripped = R2Config.strip_from_payload(payload)
    assert "r2" not in stripped
    assert stripped == {"project": "p", "action": "train_lora", "bundle_key": "bundles/p.tar.gz"}
    assert "r2" in payload


def test_strip_is_a_noop_when_there_is_no_block():
    payload = {"project": "p"}
    assert R2Config.strip_from_payload(payload) == payload


# ------------------------------------------------- the handler boundary (the real effect test)

def _stub_handler_deps(monkeypatch):
    """Neutralize everything in `handler` below the store."""
    sys.modules.setdefault("runpod", mock.MagicMock())
    import handler as handler_mod

    seen_payloads: list[dict] = []

    def fake_run_job(job, *, store, workdir, job_id, on_progress=None):
        payload = job.get("input", job) if isinstance(job, dict) else job
        seen_payloads.append(payload)
        return {"ok": True}

    monkeypatch.setattr(handler_mod, "run_job", fake_run_job)
    return handler_mod, seen_payloads


def test_handler_uses_the_payload_credential_and_strips_it_before_the_pipeline(monkeypatch):
    """Tenant credential builds the store, then does not exist downstream."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    monkeypatch.setattr(R2Config, "from_env",
                        classmethod(lambda cls, env=None: R2Config("no", "no", "no", "no")))

    captured: list[R2Config] = []
    real_init = R2.__init__

    def spy_init(self, config):
        captured.append(config)
        real_init(self, config)

    monkeypatch.setattr(R2, "__init__", spy_init)

    out = handler_mod.handler({"id": "job-1", "input": {
        "project": "p", "action": "train_lora", "bundle_key": "bundles/p.tar.gz", "r2": dict(BLOCK),
    }})

    assert out == {"ok": True}
    assert captured[0].bucket == "tenant-bucket", "the store was not built from the payload block"
    assert len(seen) == 1
    assert "r2" not in seen[0], "the credential block reached the pipeline payload"
    assert seen[0]["project"] == "p", "stripping removed more than the credential block"


def test_handler_refuses_a_malformed_block_rather_than_running_on_env_bucket(monkeypatch):
    """Refusal at the real entry point: malformed block stops the job before any store exists."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)

    out = handler_mod.handler({"id": "job-1", "input": {
        "project": "p", "action": "train_lora", "r2": {"bucket": "tenant-bucket"},
    }})
    assert out["ok"] is False
    assert "job R2 config" in out["error"]
    assert seen == [], "a job ran despite a malformed credential block"
    for secret in (TENANT_SECRET, ENV["R2_SECRET_ACCESS_KEY"]):
        assert secret not in out["error"]


def test_handler_absent_block_uses_env(monkeypatch):
    """Dedicated-endpoint / operator CF studio path: omit the key, use R2_* env."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)

    captured: list[R2Config] = []
    real_init = R2.__init__

    def spy_init(self, config):
        captured.append(config)
        real_init(self, config)

    monkeypatch.setattr(R2, "__init__", spy_init)

    out = handler_mod.handler({"id": "job-1", "input": {
        "project": "p", "action": "train_lora", "bundle_key": "bundles/p.tar.gz",
    }})
    assert out == {"ok": True}
    assert captured[0].bucket == "vivijure"
    assert captured[0].access_key_id == "env-key-id"
    assert len(seen) == 1
    assert "r2" not in seen[0]
