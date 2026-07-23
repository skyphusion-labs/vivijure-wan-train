#!/usr/bin/env python3
"""Bake-time Wan weight snapshots. HF_TOKEN read from env only; never printed."""
from __future__ import annotations

import os
import sys

REPOS = (
    "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
    "ai-toolkit/umt5_xxl_encoder",
    "ai-toolkit/wan2.1-vae",
)


def main() -> int:
    # Presence only; GitHub Actions masks secrets in logs when referenced this way.
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN: unset (public repos only)", file=sys.stderr)
    from huggingface_hub import snapshot_download

    for repo in REPOS:
        dest = snapshot_download(repo)
        print(f"snapshot {repo} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
