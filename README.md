# vivijure-wan-train

[![Runpod](https://api.runpod.io/badge/skyphusion-labs/vivijure-wan-train)](https://console.runpod.io/hub/skyphusion-labs/vivijure-wan-train)

**Wan 2.2 A14B character LoRA training** for [Vivijure](https://github.com/skyphusion-labs/vivijure). A RunPod satellite (like `vivijure-musetalk` / `vivijure-upscale`) that fits the two-expert Wan video character adapter the cloud-i2v cost door needs.

## Where this fits

```
vivijure-cf / vivijure-local (control plane)
    └── train_lora → RUNPOD_WAN_TRAIN_ENDPOINT_ID
            └── vivijure-wan-train (this repo) → R2 loras/<project>/<slot>/wan_{high,low}_noise.safetensors
vivijure-backend (render EP) → SDXL inline train only; consumes Wan loras at i2v time
```

Homelab does **not** wire `RUNPOD_WAN_TRAIN_ENDPOINT_ID` (Conrad ruling 2026-07-23). Wan cast training is CF prod only.

## Job contract

Same payload the control plane already sends:

```json
{
  "action": "train_lora",
  "project": "my-film",
  "bundle_key": "bundles/my-film/....tar.gz",
  "model_family": "wan"
}
```

Returns dual expert keys under `lora[slot].lora_id_high` / `lora_id_low`.

## Image

`ghcr.io/skyphusion-labs/vivijure-wan-train:train-<version>`

Built by `.github/workflows/build-image.yml` (workflow_dispatch on Plane C GPU runner).

## RunPod Hub

Listing config lives in [`.runpod/`](.runpod/README.md): `hub.json` (metadata, GPU pools, R2 env
inputs), `tests.json` (a `{"selftest": true}` smoke that needs no credentials), and a `Dockerfile`
that is a thin `FROM` of the published GHCR image pinned by tag and digest. The production build
(`deploy/Dockerfile`) copies ~68 GB of CI-staged, gitignored weights, so the Hub builder cannot
build it; the listing consumes the same artifact production runs instead.

Hub indexes GitHub **releases**, not commits: after a Hub-facing change or a new `train-<version>`
image, repin `.runpod/Dockerfile` and cut a release. `hub-pin-check.yml` fails CI if that pin is
stale, unpinned, or no longer anonymously pullable.

## Migration from vivijure-backend:train-*

See [docs/migration-from-backend-train-image.md](docs/migration-from-backend-train-image.md).

Prior prod pin (Phase E): `ghcr.io/skyphusion-labs/vivijure-backend:train-0.2.2` on endpoint `zqb7tougbqfkqa`.

## Commands

```bash
PYTHONPATH=src pytest
```

## Related

- Backend render image: `skyphusion-labs/vivijure-backend` (no `:train-*` after decouple)
- Control plane: `RUNPOD_WAN_TRAIN_ENDPOINT_ID` unchanged (worker image swap only)
