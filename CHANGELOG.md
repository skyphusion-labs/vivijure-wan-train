# Changelog -- vivijure-wan-train

**The repo line (`v<X.Y.Z>`) is deliberately distinct from the image line (`train-<X.Y.Z>`).** The
production artifact is a GHCR image pinned by tag AND digest in `.runpod/Dockerfile`; repo tags are
cut so the RunPod Hub re-indexes the listing, and **no workflow fires on them** (image bakes are
dispatch-only). Reading a repo tag as an image version is the mistake this note exists to prevent.

The tag is the version of record for the repo. This file records the why behind each release.
Newest first.

## Unreleased

- **Retire Plane C card `compat-smoke` workflow (2026-08-06).** Wan 2.2 A14B does not run
  realistically on workstation RTX 4000-class cards; the local GPU smoke was ceremony over
  non-prod conditions. Image bake stays on Plane C (`bake-capable` disk lane). Train readiness
  gate is **live RunPod** only. Optional `deploy/compat_smoke.py` / fixtures remain for
  hand-debug, not CI.

## v0.2.0 -- 2026-07-25

Cut so the Hub re-indexes the listing. Production artifact at this point:
`ghcr.io/skyphusion-labs/vivijure-wan-train:train-0.2.1`. **MINOR rather than a patch**, because the
repo line gained a new job-contract seam and changed a shipped default since v0.1.0.

- **`train_overrides`, an allow-listed per-job knob seam (#22 Leg 1, PR #23).** `batch_size`,
  `steps` and `resolution` only, validated and **fail-loud**: an unknown key, a wrong type, or an
  out-of-range value refuses the job before the bundle is downloaded. A knob is never partially
  applied and never silently dropped, because a dropped knob turns an A/B run into a baseline run
  wearing the variant's label. The effective knobs are emitted on the structured progress channel
  and returned as `result.train_config`, read off the constructed config rather than echoed back
  from the request.
- **Default `steps` 2000 -> 1200 (#22, PR #25).** Earned on A/B evidence rather than a hunch: at
  1200 steps identity held on pixels against the 2000-step control (same conditioning portrait,
  same seed, LoRA the only difference), for a 39% cut in wall clock.
- **Dependency floors** raised by dependabot (#18, #20, #21).
  (Backfilled 2026-07-28 from the v0.2.0 GitHub release; this file did not exist at the tag.)

## v0.1.0 -- 2026-07-24

- **First repo-line release**, deliberately distinct from the image line: the production artifact
  was `ghcr.io/skyphusion-labs/vivijure-wan-train:train-0.1.3`, pinned by digest in
  `.runpod/Dockerfile`. Cut so the RunPod Hub could index the listing added in #14 (backend#301).
  No workflow fires on this tag.
  (Backfilled 2026-07-28 from the v0.1.0 GitHub release; this file did not exist at the tag.)
