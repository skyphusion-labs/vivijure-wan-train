# GPU compat smoke (dispatch-gated, Plane C)

Tracking issue: [#29](https://github.com/skyphusion-labs/vivijure-wan-train/issues/29).

## The gap this closes

`ci.yml` runs ruff and the unit suite on a CPU `ubuntu-latest` runner. Nothing in it installs the
training image dependency set, and nothing in it touches a GPU. So a dependency bump can be fully
green in CI and still be:

- unresolvable (the pinned version is not on the cu128 wheel index the Dockerfile uses),
- uninstallable (a constraint fights a pin elsewhere in the image),
- un-importable (a torch / torchvision / torchaudio trio whose compiled extensions disagree on the
  torch ABI dies at `import`, not at `pip install`),
- silently API-broken (a `huggingface_hub` major removing something this repo calls).

`.github/workflows/compat-smoke.yml` answers those four questions on the real hardware, on demand.

## What it does

1. `workflow_dispatch` with a `ref` input (a PR head sha, preferably) and an optional `steps` count.
2. Checks out that ref on Plane C (`[self-hosted, gpu]`, org runner group `vivijure-gpu-builds`).
3. Builds `deploy/Dockerfile --target deps`: the PRODUCTION dependency layers at that ref. Nothing
   is pushed; there is no tag and no registry write.
4. Runs `deploy/compat_smoke.py` twice, once per conda env, because the two envs carry different
   dependency sets:
   - `--suite hfhub` in the `vivijure` env: the `huggingface_hub` API surface this repo actually
     calls, including a live check that the network-forbid seam in `deploy/smoke_train_offline.py`
     still intercepts hub HTTP.
   - `--suite gputrain` in the `aitoolkit` env, with `--gpus all`: torchvision and torchaudio
     compiled ops on the card, the ai-toolkit Wan trainer imports, and a real LoRA fine-tune over
     `tests/fixtures/compat-smoke` that must move adapter weights and write a reloadable
     `.safetensors`.
5. Asserts the adapter artifact exists ON THE HOST (not just per the script), writes a JSON report
   per suite into the run summary, then removes the locally built image and the scratch dir.

## Scope, stated honestly

It PROVES: the dependency set resolves, installs, imports, and completes a real training step on a
real GPU, and that the artifact path works end to end.

It does NOT prove a Wan 2.2 A14B run. Plane C carries a single RTX 4000 SFF Ada (20GB); the proven
Wan recipe is bf16 with both experts resident on an 80GB card. The smoke deliberately trains a tiny
UNet so the question it answers (does this dependency set work) is not confounded by the question it
cannot answer on this hardware (does A14B fit).

**The final gate for any dependency bump is unchanged: bake the image via `build-image.yml` and run
a live train on the prod RunPod endpoint.** The smoke is a cheap upstream filter, plus the paired
changelog review; a green smoke is evidence, not a merge.

## Running it

```bash
gh workflow run compat-smoke.yml --repo skyphusion-labs/vivijure-wan-train \
  -f ref=<40-char-sha> -f steps=20
```

Dispatch a baseline run against `main` first when the harness itself has changed. A suite that
passes on a bump but was never seen passing on a known-good ref proves nothing about the bump.

The ref must carry the harness, so a branch cut before this landed (an older dependabot PR, say)
has to be rebased onto `main` before it can be smoked. That is not an extra hoop: the branch
ruleset requires up-to-date-with-main to merge anyway. The workflow checks for the harness right
after checkout and says exactly that rather than letting `docker build` fail on an unknown target.

## The stage split contract

`deploy/Dockerfile` is two stages: `deps` (apt, both conda envs, the cu128 torch trio,
`deploy/requirements.txt`, the ai-toolkit checkout plus constraints and overrides) and
`full` (`FROM deps`, then the ~55GB weight bake and the worker code). Production builds name no
target, so they still build `full` and are unchanged.

The smoke targets `deps` so it exercises the REAL dependency layers rather than a lookalike
Dockerfile that could drift away from production silently. `tests/test_compat_smoke_contract.py`
enforces the contract in CPU CI: two stages, no dependency install below the split, the weight bake
above nothing and below it, and the workflow still naming `--target deps`.

## Disk hygiene

The workflow removes its own image in an `always()` step rather than relying on the Plane C disk
floor and digest-keyed reaper (fleet-chezmoi#1091). Build cache layers are left to the reaper, which
is what it is for.
