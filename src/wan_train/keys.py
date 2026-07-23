"""R2 object-key layout for Wan LoRA training artifacts."""
from __future__ import annotations

import posixpath
import re


def _slug(project: str) -> str:
    return "_".join(str(project).strip().split()).replace("/", "_") or "untitled"


def wan_lora_key(project: str, slot: str, expert: str) -> str:
    e = str(expert).strip().lower()
    if e not in ("high", "low"):
        raise ValueError(f"wan_lora_key: expert must be 'high' or 'low', got {expert!r}")
    return f"loras/{_slug(project)}/{_slug(slot)}/wan_{e}_noise.safetensors"


def check_job_key(key: str, *, prefixes: tuple[str, ...], what: str) -> str:
    k = str(key or "")
    ok = (
        bool(k)
        and k == k.strip()
        and not k.startswith("/")
        and "\\" not in k
        and ".." not in k.split("/")
        and k.startswith(prefixes)
    )
    if not ok:
        raise ValueError(
            f"{what}: R2 key {k!r} must be a plain relative key under "
            f"{' or '.join(prefixes)} (see the render key map)")
    return k


def bundle_key_matches_project(bundle_key: str, project: str) -> bool:
    slug = _slug(project)
    if not bundle_key.startswith("bundles/"):
        return False
    rest = bundle_key[len("bundles/"):]
    if rest.startswith(f"{slug}/"):
        return True
    if rest == f"{slug}.tar.gz":
        return True
    return bool(re.fullmatch(re.escape(slug) + r"-[0-9a-f]{16}\.tar\.gz", rest))


def check_bundle_key_for_project(bundle_key: str, project: str, *, what: str) -> str:
    k = check_job_key(bundle_key, prefixes=("bundles/",), what=what)
    if not bundle_key_matches_project(k, project):
        slug = _slug(project)
        raise ValueError(
            f"{what}: bundle_key {k!r} must belong to project {project!r} "
            f"(expected bundles/{slug}/..., bundles/{slug}.tar.gz, or "
            f"bundles/{slug}-<contenthash>.tar.gz)")
    return k


def progress_log_key(project: str, job_id: str) -> str:
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.ndjson"


def progress_snapshot_key(project: str, job_id: str) -> str:
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.json"


def join(*parts: str) -> str:
    return posixpath.join(*parts)
