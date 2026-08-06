"""CPU-side guards for the Dockerfile deps/full stage split + optional local smoke harness.

The Plane C card compat-smoke *workflow* is RETIRED (2026-08-06); train gate is RunPod A14B.
CI still enforces the Dockerfile stage split (deps vs weight bake) and that optional local
harness files stay importable/tiny if present.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "deploy" / "Dockerfile"
FIXTURES = REPO / "tests" / "fixtures" / "compat-smoke"

DEPS_STAGE = "deps"
FULL_STAGE = "full"
# Instruction prefixes that install or fetch a dependency. Every one of these must live in
# the deps stage, or the smoke builds an image that is missing the thing under test.
DEP_INSTALL_PATTERNS = (
    re.compile(r"\bpip install\b"),
    re.compile(r"\bconda create\b"),
    re.compile(r"\bconda install\b"),
    re.compile(r"\bapt-get install\b"),
    re.compile(r"\bgit clone\b"),
)


def _dockerfile_lines() -> list[str]:
    return DOCKERFILE.read_text(encoding="utf-8").splitlines()


def _split_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip() == f"FROM {DEPS_STAGE} AS {FULL_STAGE}":
            return i
    pytest.fail(f"deploy/Dockerfile has no `FROM {DEPS_STAGE} AS {FULL_STAGE}` stage split")
    raise AssertionError  # unreachable, keeps type checkers quiet


def test_dockerfile_declares_the_two_stages():
    lines = _dockerfile_lines()
    froms = [ln.strip() for ln in lines if ln.strip().startswith("FROM ")]
    assert len(froms) == 2, f"expected exactly 2 FROM lines, got {froms}"
    assert froms[0].endswith(f" AS {DEPS_STAGE}"), froms[0]
    assert froms[1] == f"FROM {DEPS_STAGE} AS {FULL_STAGE}", froms[1]


def test_stage_split_comment_names_a_test_that_exists():
    """The split comment points the next editor at the enforcing test; a wrong filename defeats it.

    It shipped wrong once (it named tests/test_dockerfile_stages.py, which never existed), so the
    pointer is now checked rather than trusted.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", text))
    assert referenced, "the stage split comment must name the test that enforces it"
    missing = sorted(rel for rel in referenced if not (REPO / rel).is_file())
    assert not missing, f"deploy/Dockerfile points at test file(s) that do not exist: {missing}"
    assert f"tests/{Path(__file__).name}" in referenced, (
        "the split comment should name this file, which is what actually guards the split")


def test_no_dependency_install_below_the_split():
    lines = _dockerfile_lines()
    split = _split_index(lines)
    offenders = [
        f"{i + 1}: {line.strip()}"
        for i, line in enumerate(lines[split + 1:], start=split + 1)
        if any(p.search(line) for p in DEP_INSTALL_PATTERNS)
    ]
    assert not offenders, (
        "dependency install(s) below the stage split; keep installs in "
        f"{DEPS_STAGE} so the weight bake stage stays pure:\n" + "\n".join(offenders))


def test_weight_bake_stays_above_nothing_and_below_the_split():
    lines = _dockerfile_lines()
    split = _split_index(lines)
    weight_copies = [i for i, line in enumerate(lines) if "deploy/train-bins/" in line]
    assert weight_copies, "no train-bins COPY found; did the weight bake move?"
    assert min(weight_copies) > split, (
        "the 55GB weight bake must stay in the full stage below the deps split")


def test_fixture_dataset_is_present_paired_and_tiny():
    images = sorted(FIXTURES.glob("*.png"))
    assert len(images) >= 2, f"optional local smoke fixtures need a batch, found {images}"
    for img in images:
        caption = img.with_suffix(".txt")
        assert caption.is_file(), f"{img.name} has no caption"
        assert caption.read_text(encoding="utf-8").strip(), f"{caption.name} is empty"
        assert img.stat().st_size < 4096, (
            f"{img.name} is {img.stat().st_size} bytes; keep the fixture set tiny, this is a "
            "source repo")


def _load_compat_smoke():
    path = REPO / "deploy" / "compat_smoke.py"
    spec = importlib.util.spec_from_file_location("compat_smoke", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compat_smoke_imports_without_gpu_deps():
    """Module import must stay light: torch and friends are imported inside the suites."""
    mod = _load_compat_smoke()
    assert mod.FIXTURE_DIR == FIXTURES


def test_fixture_pairs_reads_the_dataset():
    mod = _load_compat_smoke()
    pairs = mod._fixture_pairs()
    assert len(pairs) == len(sorted(FIXTURES.glob("*.png")))
    assert all(caption for _, caption in pairs)


def test_fixture_pairs_fails_loud_on_a_missing_caption(tmp_path, monkeypatch):
    """A half-present fixture set must fail, not train on whatever it found."""
    mod = _load_compat_smoke()
    (tmp_path / "orphan.png").write_bytes(b"not really a png")
    monkeypatch.setattr(mod, "FIXTURE_DIR", tmp_path)
    with pytest.raises(mod.CheckFailed, match="no caption"):
        mod._fixture_pairs()


def test_fixture_pairs_fails_loud_on_an_empty_dataset(tmp_path, monkeypatch):
    mod = _load_compat_smoke()
    monkeypatch.setattr(mod, "FIXTURE_DIR", tmp_path)
    with pytest.raises(mod.CheckFailed, match="no fixture images"):
        mod._fixture_pairs()
