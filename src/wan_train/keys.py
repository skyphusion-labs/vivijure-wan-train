"""R2 object-key layout for Wan LoRA training artifacts."""
from __future__ import annotations

import posixpath
import re

# Canonical slug: ASCII alnum, underscore, hyphen only. Blocks slash-collision and
# display-name normalization bypass called out in K3 bundle-ownership findings.
CANONICAL_SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Exact Wan MoE expert filenames the train endpoint emits/consumes.
_WAN_LORA_FILE_RE = re.compile(
    r"^loras/(?P<project>[a-zA-Z0-9_-]+)/(?P<slot>[a-zA-Z0-9_-]+)/wan_(?:high|low)_noise\.safetensors$"
)


def _slug(project: str) -> str:
    return "_".join(str(project).strip().split()).replace("/", "_") or "untitled"


def check_canonical_slug(value: str, *, what: str) -> str:
    """Reject values that cannot be used safely as a single R2 path segment."""
    raw = str(value or "").strip()
    if not raw or not CANONICAL_SLUG_RE.fullmatch(raw):
        raise ValueError(
            f"{what}: {raw!r} must be a canonical slug "
            "(ASCII letters, digits, underscore, hyphen only; no whitespace or '/')")
    return raw


def check_project_slug(project: str, *, what: str = "project") -> str:
    """Reject display names that collapse to the same R2 prefix under _slug."""
    raw = str(project or "").strip()
    slug = _slug(raw)
    if not raw or raw != slug:
        raise ValueError(
            f"{what}: project {raw!r} must be a canonical slug (no whitespace or '/'; "
            f"use {slug!r} instead)")
    return check_canonical_slug(raw, what=what)


def check_job_id_slug(job_id: str, *, what: str = "job_id") -> str:
    """Progress keys use job_id as a path segment; reject traversal/collision shapes."""
    return check_canonical_slug(job_id, what=what)


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
        and "%" not in k
        and "\0" not in k
        and ".." not in k.split("/")
        and k.startswith(prefixes)
    )
    if not ok:
        raise ValueError(
            f"{what}: R2 key {k!r} must be a plain relative key under "
            f"{' or '.join(prefixes)} (see the render key map)")
    return k


def _bundle_project_segment(bundle_key: str) -> str | None:
    """First path segment after bundles/ (the owning project slug)."""
    if not bundle_key.startswith("bundles/"):
        return None
    rest = bundle_key[len("bundles/"):]
    if not rest:
        return None
    if "/" in rest:
        return rest.split("/", 1)[0]
    if rest.endswith(".tar.gz"):
        stem = rest[:-7]
        if re.fullmatch(r"[a-zA-Z0-9_-]+", stem):
            return stem
        m = re.fullmatch(r"([a-zA-Z0-9_-]+)-[0-9a-f]{16}", stem)
        return m.group(1) if m else None
    return None


def bundle_key_matches_project(bundle_key: str, project: str) -> bool:
    slug = check_canonical_slug(_slug(project), what="project")
    if not bundle_key.startswith("bundles/"):
        return False
    rest = bundle_key[len("bundles/"):]
    if not rest:
        return False
    if "/" in rest:
        return rest.split("/", 1)[0] == slug
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


def check_scoped_lora_key(key: str, *, project: str, what: str) -> str:
    """Validate a reused LoRA read key belongs to the job project."""
    k = check_job_key(key, prefixes=("loras/",), what=what)
    slug = check_canonical_slug(_slug(project), what="project")
    if not k.startswith(f"loras/{slug}/"):
        raise ValueError(
            f"{what}: R2 key {k!r} must be under loras/{slug}/ for project {project!r}")
    return k


def check_pretrained_lora_key(key: str, *, project: str, slot: str, what: str) -> str:
    """Passthrough LoRA keys must match the exact Wan expert filename for the slot."""
    k = check_scoped_lora_key(key, project=project, what=what)
    slug = check_canonical_slug(_slug(project), what="project")
    slot_slug = check_canonical_slug(_slug(slot), what=f"{what} slot")
    m = _WAN_LORA_FILE_RE.fullmatch(k)
    if not m or m.group("project") != slug or m.group("slot") != slot_slug:
        raise ValueError(
            f"{what}: R2 key {k!r} must be "
            f"loras/{slug}/{slot_slug}/wan_(high|low)_noise.safetensors")
    return k


def progress_log_key(project: str, job_id: str) -> str:
    slug = check_canonical_slug(_slug(project), what="project")
    jid = check_canonical_slug(_slug(job_id), what="job_id")
    return f"renders/{slug}/progress/{jid}.ndjson"


def progress_snapshot_key(project: str, job_id: str) -> str:
    slug = check_canonical_slug(_slug(project), what="project")
    jid = check_canonical_slug(_slug(job_id), what="job_id")
    return f"renders/{slug}/progress/{jid}.json"


def join(*parts: str) -> str:
    return posixpath.join(*parts)
