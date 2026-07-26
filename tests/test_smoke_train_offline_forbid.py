"""The HF network-forbid block in deploy/smoke_train_offline.py must actually block (#29).

Found while building the GPU compat smoke: huggingface_hub 1.x moved its HTTP layer from a
requests Session to a shared httpx client, so the old attribute patch on `get_session` stopped
intercepting anything. The build-time offline check that depends on it was passing vacuously.

These are CPU tests of the SEAM SELECTION (which mechanism gets installed for which hub
generation). Proof that the installed block refuses a real hub call is the compat smoke's job,
in-image, on both hub generations: deploy/compat_smoke.py --suite hfhub.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "smoke_train_offline.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_train_offline_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_fake_hub(monkeypatch, *, modern: bool):
    """Stub huggingface_hub.utils._http for both hub generations."""
    http_mod = types.ModuleType("huggingface_hub.utils._http")
    installed: dict[str, object] = {}
    if modern:
        http_mod.set_client_factory = lambda factory: installed.__setitem__("factory", factory)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http_mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", types.ModuleType("huggingface_hub.utils"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    return http_mod, installed


def _install_fake_httpx(monkeypatch):
    httpx = types.ModuleType("httpx")

    class BaseTransport:
        pass

    class Client:
        def __init__(self, transport=None, **kwargs):
            self.transport = transport

    httpx.BaseTransport = BaseTransport
    httpx.Client = Client
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    return httpx


def test_modern_hub_gets_the_client_factory_seam(monkeypatch):
    """hub 1.x: the block must go through set_client_factory, which resets the shared client.

    Patching the module attribute is not enough there: callers that imported get_session by name
    keep the old reference, which is exactly how the check went quiet.
    """
    mod = _load()
    _http, installed = _install_fake_hub(monkeypatch, modern=True)
    httpx = _install_fake_httpx(monkeypatch)

    seam = mod._forbid_hf_http()

    assert seam == "httpx client factory"
    client = installed["factory"]()
    assert isinstance(client.transport, httpx.BaseTransport)
    request = types.SimpleNamespace(method="GET", url="https://huggingface.co/api/models/x")
    with pytest.raises(mod._NetworkForbidden, match="HF HTTP forbidden"):
        client.transport.handle_request(request)


def test_legacy_hub_falls_back_to_the_session_patch(monkeypatch):
    """hub 0.x has no client factory; the requests-Session swap is still the right seam."""
    mod = _load()
    http_mod, _ = _install_fake_hub(monkeypatch, modern=False)

    seam = mod._forbid_hf_http()

    assert seam == "requests session patch"
    session = http_mod.get_session()
    for call in (lambda: session.get("https://huggingface.co"),
                 lambda: session.head("https://huggingface.co"),
                 lambda: session.post("https://huggingface.co")):
        with pytest.raises(mod._NetworkForbidden):
            call()


def test_control_is_wired_into_main_before_the_negative_check():
    """A control that exists but never runs is not a control."""
    source = SCRIPT.read_text(encoding="utf-8")
    body = source.split("def main(")[1]
    control = body.index("check_forbid_block_is_live()")
    negative = body.index("check_local_path_config_no_network(")
    assert control < negative, (
        "the forbid-block control must run BEFORE the check that depends on the block")
