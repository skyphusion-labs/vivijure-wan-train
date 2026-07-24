# RunPod Hub -- Vivijure Wan Cast Training

This directory configures the [RunPod Hub](https://docs.runpod.io/hub/publishing-guide) listing for
the Vivijure Wan 2.2 A14B character (cast) LoRA training worker.

## Required environment

Hub deployers fill these when they deploy from the listing. They are the same four names the worker
reads at runtime (`src/wan_train/r2.py`, `R2Config.from_env`):

| Env key | Hub UI label | What to put |
| --- | --- | --- |
| `R2_ENDPOINT` | R2 S3 endpoint | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 access key ID | Public half of an R2 API token with read/write on the bucket |
| `R2_SECRET_ACCESS_KEY` | R2 secret access key | Secret half of that token |
| `R2_BUCKET` | R2 bucket | Bucket name shared with Vivijure Studio (preset default: `vivijure`) |

**Preset:** "Standard (shared Studio bucket)" sets `R2_BUCKET=vivijure`. Override if your Studio uses
another bucket name; the Studio and this worker must agree.

**Name check.** This worker reads `R2_ENDPOINT` (no `_URL`), same as `vivijure-backend`. The finish
satellites (`vivijure-musetalk`, `vivijure-upscale`, `vivijure-audio-upscale`) read
`R2_ENDPOINT_URL`. Copy-pasting env from a finish endpoint will miss the bucket.

No `HF_TOKEN` is needed: the Wan base weights are baked into the image and the runtime is offline
(`HF_HUB_OFFLINE=1`).

## Required files (Hub probe)

Hub looks for `handler.py`, `Dockerfile`, `README.md`, `hub.json`, and `tests.json` in `.runpod/`
(precedence) or the repo root. Here:

| File | Where | Note |
| --- | --- | --- |
| `handler.py` | repo root | Real file (not a symlink), so the Hub probe sees it |
| `README.md` | repo root + this file | Root is the listing README |
| `Dockerfile` | `.runpod/Dockerfile` | Thin `FROM` of the published image, see below |
| `hub.json` / `tests.json` | `.runpod/` | Listing metadata + smoke |

### Why `.runpod/Dockerfile` is not `deploy/Dockerfile`

The production build `COPY`s `deploy/train-bins/`, roughly 68 GB of Wan base weights that are staged
at CI time (`scripts/download_wan_weights.py` + `deploy/bake_layers.py`, run by
`.github/workflows/build-image.yml`) and are gitignored. Nothing in the repo tree can reproduce them,
so the Hub builder cannot build `deploy/Dockerfile`. The Hub listing therefore consumes the SAME
artifact production runs: the public GHCR image, pinned by tag AND digest. `CMD`, `ENV`, and both
conda envs are inherited.

`.github/workflows/hub-pin-check.yml` runs `scripts/check_hub_base_pin.sh` on every push and PR: it
resolves the pin against GHCR anonymously (the Hub builder's reach) and fails on an unpinned base, a
tag that no longer exists, a digest that disagrees with the tag, or a package that is not public.

## Hub test

`.runpod/tests.json` sends `{ "selftest": true }` on an H200. That probe runs the handler's
`_selftest`, which checks the ai-toolkit seam and the baked Wan base paths; it needs no R2
credentials. The 15-minute timeout is sized for a cold pull of the ~68 GB image, not for the probe
itself. A real training job still needs the four R2 vars above.

## Operator checklist (listing status)

1. A GitHub **release** exists whose tree contains this directory (Hub indexes releases, not commits).
2. The image tag pinned in `.runpod/Dockerfile` matches the image the production endpoint runs.
3. In the RunPod console Hub page, add the repo and confirm the build/test is green (Pending vs Live).
4. After any Hub-facing change here, or any new `train-<version>` image, repin `.runpod/Dockerfile`
   and cut a new GitHub release so Hub re-indexes (usually within an hour).
