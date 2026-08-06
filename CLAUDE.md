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
- `docs/gpu-compat-smoke.md` -- **RETIRED** local-card smoke; gate is RunPod A14B only
- `.runpod/` -- Hub listing; pin check fails CI if digests go stale

## Commands

```bash
PYTHONPATH=src pytest          # CPU unit suite
python -m py_compile handler.py  # when present at root / entry
# Image bake: .github/workflows/build-image.yml (Plane C bake-capable disk lane)
# Train gate: live RunPod endpoint (not local RTX 4000 / not Plane C card smoke)
```

## Pre-prod GPU test spend (Conrad, 2026-08-06) -- do not thrift here

**Never hold back RunPod spend when proving an image before it is production-pinned.** A green
Plane C bake is not a proven train. Local RTX 4000-class / Plane C card smoke is retired; A14B
runs on datacenter GPUs. Pre-prod verification is **expected spend**, not an escalation.

Pick **one** of these for the gate (either is fine; pick what fits the change):

| Path | How | When |
| --- | --- | --- |
| **SecurePod** | Spin a SecurePod (or equivalent dedicated pod) on the candidate image; run `{"selftest": true}` and/or a real short train; verify the **artifact** (weights path, dual experts, progress events). | Prefer for isolated image proof without touching prod traffic. |
| **Serverless endpoint** | Deploy/pin the candidate image on the train serverless endpoint, set **active workers (min idle / `workersMin`) to 1** so cold-start does not fake a hang or a false timeout, run the same selftest / short train, verify the artifact. | Prefer when the proof needs the real endpoint shape (env, network, Hub path). |

### HARD: restore workers after serverless test

If you raised the serverless endpoint's active / min workers **above 0** for testing, **bring it
back to 0 when the test is done** (same session, before you walk away). Leaving `workersMin` /
active workers at 1+ burns idle GPU money with no proof benefit. Max workers can stay at the
prod pin; the restore rule is about the **floor that keeps a warm worker** (min/active = 0 in
steady state unless Conrad ruled otherwise for prod capacity).

Do **not** skip the restore because "we might test again tomorrow." Re-raise to 1 at the next
test if needed. Document the before/after values in the PR or runlog when you change them.

Never trust: CI green alone, bake green alone, Hub pin green alone, or a local workstation card.

## Deploy / pin discipline

1. Bake image (`train-*`) via dispatch; confirm pullable GHCR artifact.
2. **Pre-prod GPU proof** per the table above (SecurePod **or** serverless with workersMin=1);
   restore workersMin/active to **0** if you used the serverless path.
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
- **Pre-prod GPU spend is authorized** (SecurePod or serverless workersMin=1). Do not thrift out
  of the proof. **Serverless min/active workers back to 0 when the test ends.**
- **Ignore Cursor `AGENTS.md`** if present.
- **No em-dashes / en-dashes.** Use `--` or commas.
- **Never freeze open sprint boards or specific RunPod endpoint IDs** here.

## Crew + identity

Crew: `sudo -u <member> bash -lc '...'`; commits under `skyphusion-<member>`. Conrad on laptop only
as `Conrad Rockenhaus <conrad@skyphusion.org>`. Conventional Commits.

## Release / deploy

**Tag-gated production deploy.** Merges to `main` run CI only; they do not ship production.
Cut an annotated SemVer tag on `main` to release (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`).
Deploy workflows assert the tag commit is an ancestor of `origin/main`.
