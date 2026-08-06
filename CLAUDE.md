# CLAUDE.md

Guidance for Claude Code (and the crew) working in this repo.

## What this is

**vivijure-wan-train: Wan cast LoRA training satellite.** A first-class constellation member (not a
backend image tag). RunPod serverless image that trains Wan 2.2 A14B dual-expert character adapters
from a cast bundle and writes
`loras/<project>/<slot>/wan_{high,low}_noise.safetensors` to the shared R2 bucket for studio i2v.

- Image line: `ghcr.io/skyphusion-labs/vivijure-wan-train:train-<X.Y.Z>`
- Repo line: `v*` tags (Hub re-index / release notes; **no workflow fires on bare repo tags**)
- GPU class: datacenter B200 / H200 class pools (see `.runpod/hub.json` `gpuIds`; operator pin, not
  frozen here)
- Changelog: `CHANGELOG.md` (records why; image pin is the production artifact)

Do not invent current image pins. See `CHANGELOG.md`, latest `train-*` GHCR tags, and
`.runpod/Dockerfile` (tag **and** digest).

## Relation to the constellation

```
vivijure-cf (CF prod; RUNPOD_WAN_TRAIN_ENDPOINT_ID)
    └── train_lora --> this image --> R2 dual Wan experts
vivijure-backend (render EP)
    └── SDXL inline train only; consumes Wan loras at i2v time
vivijure-local / homelab
    └── does NOT wire RUNPOD_WAN_TRAIN_ENDPOINT_ID by default (CF / RunPod path)
```

Migrated off `vivijure-backend:train-*` (see `docs/migration-from-backend-train-image.md`). Backend
no longer ships `:train-*` after decouple.

## Documentation map

- `README.md` -- job contract, image, Hub, commands
- `CHANGELOG.md` -- repo line vs image line (read the header)
- `docs/migration-from-backend-train-image.md`
- `docs/gpu-compat-smoke.md` -- Plane C compat smoke (not the release gate)
- `.runpod/` -- Hub listing; pin check fails CI if digests go stale

## Commands

```bash
PYTHONPATH=src pytest          # CPU unit suite
python -m py_compile handler.py  # when present at root / entry
# Image bake: .github/workflows/build-image.yml (workflow_dispatch, Plane C GPU)
# Compat smoke: .github/workflows/compat-smoke.yml (dispatch; pushes nothing)
```

Selftest: `{"selftest": true}` on a SecurePod / pinned endpoint (spend-gated). **SecurePod smoke
before prod pin.** Never trust a green bake alone.

## Deploy / pin discipline

1. Bake image (`train-*`) via dispatch; confirm pullable GHCR artifact.
2. SecurePod selftest / smoke; verify **artifact** (weights path, dual experts, progress events).
3. Repin `.runpod/Dockerfile` tag+digest; Hub pin check must stay green.
4. Operator repins the RunPod endpoint to the new image (dashboard / API template). **Do not
   hardcode endpoint IDs in docs or this file.**
5. Optional: cut repo `v*` so Hub re-indexes; that tag does not bake.

## Hard rules

- **CSAM bright-line (NON-NEGOTIABLE):** zero tolerance including synthetic.
- **Clean room:** no `wavevryn` code; attribute models in `THIRD_PARTY_MODELS.md`.
- **Fail-loud overrides:** `train_overrides` unknown keys / bad types / out-of-range refuse before
  bundle download; never silently drop a knob.
- **FLUX self-host OUT** of this lane (not this satellite's job).
- **Verify artifact, not pipeline.** A green workflow is not a pullable image and not a proven train.
- **Ignore Cursor `AGENTS.md`** if present.
- **No em-dashes / en-dashes.** Use `--` or commas.
- **Never freeze open sprint boards or specific RunPod endpoint IDs** here.

## Crew + identity

Crew: `sudo -u <member> bash -lc '...'`; commits under `skyphusion-<member>`. Conrad on laptop only
as `Conrad Rockenhaus <conrad@skyphusion.org>`. Conventional Commits. GPU spend is gated; confirm
before non-trivial train runs.

## Release / deploy

**Tag-gated production deploy.** Merges to `main` run CI only; they do not ship production.
Cut an annotated SemVer tag on `main` to release (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`).
Deploy workflows assert the tag commit is an ancestor of `origin/main`.
