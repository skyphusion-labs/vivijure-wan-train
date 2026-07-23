# Migration: vivijure-backend:train-* → vivijure-wan-train

Conrad ruling 2026-07-23: Wan train is a separate satellite repo; the backend render image keeps SDXL inline train only.

## Current prod (Phase E, 2026-07-23)

| Artifact | Value |
|---|---|
| Image | `ghcr.io/skyphusion-labs/vivijure-backend:train-0.2.2` |
| CF prod EP | `zqb7tougbqfkqa` |
| Local MinIO EP | `8kjcn5sz6k8p1n` (homelab unwired for Wan train) |

## Target state

| Artifact | Value |
|---|---|
| Image | `ghcr.io/skyphusion-labs/vivijure-wan-train:train-0.1.0` (first satellite release) |
| CF prod EP | same `zqb7tougbqfkqa`, template image only |
| Control plane | **unchanged** (`RUNPOD_WAN_TRAIN_ENDPOINT_ID`, job payload, R2 key layout) |

## Migration steps (no prod break)

1. **Merge + release `vivijure-wan-train`** -- CI green, dispatch `build-image.yml` → `:train-0.1.0`.
2. **GPU smoke on staging EP** (or prod EP during low-traffic window):
   - `{"selftest": true}` → `runtime_ready: true`
   - One bare `train_lora` job (no `model_family`) → both experts in R2
3. **Pin CF prod template** to `ghcr.io/skyphusion-labs/vivijure-wan-train:train-0.1.0` via RunPod MCP / `pin-runpod-template.py`.
4. **Retire backend train image build** -- remove `train-image-build.yml` and `deploy/train.Dockerfile` from `vivijure-backend` (render `:version` unchanged).
5. **Document runlog** in `fleet-chezmoi/docs/runlog/`.

## Rollback

Re-pin template to `ghcr.io/skyphusion-labs/vivijure-backend:train-0.2.2` (immutable tag `:train-0.2.2-29980929583` still on GHCR until retention expires).

## What moved vs stayed

| Moved to vivijure-wan-train | Stays in vivijure-backend |
|---|---|
| `wan_lora_train.py`, ai-toolkit wrapper | SDXL `lora_train.py` (inline render train) |
| `deploy/train.Dockerfile`, bake scripts | `deploy/runtime.Dockerfile` (render) |
| `train-image-build.yml` | render image CI |
| RunPod train handler (`handler.py`) | render worker (`vivijure_backend.worker`) |
| | `keys.wan_lora_key` (render loads Wan experts at i2v) |
| | `resolve_lora_family` → always SDXL on render EP |

## Homelab

Do **not** set `RUNPOD_WAN_TRAIN_ENDPOINT_ID` on propagandhi. Local `/train-lora` uses SDXL on the render endpoint.
