# GPU compat smoke -- RETIRED (RunPod-only gate)

> ## RETIRED 2026-08-06 -- do not dispatch
>
> The Plane C / local-card `compat-smoke.yml` workflow is **deleted**. It was a
> dependency/ABI experiment on RTX 4000 SFF Ada (20GB). That is **not** the realistic
> operating condition for Wan 2.2 A14B (bf16, both experts, ~80GB class), which runs on
> **RunPod**. Keeping a local card job as "evidence" was ceremony that did not match prod
> and did not fully work on the workstation cards.
>
> **Release / dependency-bump gate (authoritative):**
> 1. Bake the train image: `.github/workflows/build-image.yml` on Plane C (`bake-capable`,
>    disk lane -- no card required for the bake itself).
> 2. Live train on the **prod RunPod** Wan train endpoint (Hub listing under `.runpod/`).
>
> Optional local harness files (`deploy/compat_smoke.py`, fixtures) may remain for
> hand-debug on a machine that has a card; they are **not** CI, not a merge gate, and not
> a substitute for RunPod A14B. Prefer RunPod for any claim about train readiness.

Historical design notes (pre-retirement) lived in git history for this file and the
removed workflow. Tracking: [#29](https://github.com/skyphusion-labs/vivijure-wan-train/issues/29).
