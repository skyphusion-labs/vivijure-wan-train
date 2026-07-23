# Security false positives (K3 adversarial audit)

Documented K3 repo-mode findings that are accepted under the Vivijure GPU operator trust model, or closed with code guards.

## Missing authentication on train endpoint (`handler.py`)

**Finding:** No HMAC/JWT on RunPod handler; anyone with endpoint URL can submit jobs.

**Disposition:** False positive (operator trust boundary). RunPod serverless accepts jobs only via the RunPod control API (operator API key + endpoint id). The worker URL is not a public anonymous HTTP surface. The control plane (`vivijure-cf`) is the sole job submitter in prod; R2 keys are scoped by `check_project_slug`, `check_bundle_key_for_project`, and `check_pretrained_lora_key`.

**Evidence:** RunPod serverless architecture; dedicated train EP `zqb7tougbqfkqa` with template env credentials, not tenant-facing.

## Pretrained LoRA key passthrough (`job.py`)

**Finding:** Arbitrary R2 key writes via `pretrained_loras`.

**Disposition:** False positive (closed with strict guard). `check_pretrained_lora_key` requires exact `loras/<project>/<slot>/wan_(high|low)_noise.safetensors` filenames for the declared slot; prefix-only checks removed in fix/k3-wan-train-crit-high-closeout.

## Offline ai-toolkit hub ID patch (`wan_lora_train.py`)

**Finding:** Unpatched ai-toolkit hub IDs leave offline training failures.

**Disposition:** False positive. The bake step pins ai-toolkit and applies offline hub rewrites at image build; runtime patch is belt-and-suspenders for the exact quoted IDs ai-toolkit ships in the baked tree.

## Tar bundle extraction (`contract.py`)

**Finding:** Symlink/path traversal via tar members.

**Disposition:** **Fixed** (not deferred). `_safe_extract` rejects absolute paths, `..`, symlinks, hardlinks, devices, and fifos; validates all members before member-by-member extract (no monolithic `extractall`).

## Bundle ownership slash collision (`keys.py`)

**Finding:** Display-name slug normalization bypass for bundle keys.

**Disposition:** **Fixed**. `check_project_slug` + `CANONICAL_SLUG_RE` require canonical slugs; `bundle_key_matches_project` compares the parsed first segment after `bundles/` to the project slug; `%` and encoded slashes rejected in `check_job_key`.

## HF token in CI (`build-image.yml`)

**Finding:** HF_TOKEN exposed in workflow logs/cache.

**Disposition:** **Fixed**. Token passed only to `scripts/download_wan_weights.py` via step env (GitHub secret masking); no inline `python -c` interpolation.

## Config path injection (`wan_lora_train.py`)

**Finding:** Shell metacharacters in model paths written to ai-toolkit YAML.

**Disposition:** **Fixed**. `_assert_safe_model_ref` rejects metacharacters; resolved filesystem paths must fall under `/opt/models`, `/opt/ai-toolkit`, or HF cache roots before entering config.

## Record

| Date | Audit | Finding | Rationale |
| --- | --- | --- | --- |
| 2026-07-23 | K3 verify ~18:04 | Missing endpoint auth | RunPod operator API boundary |
| 2026-07-23 | K3 verify ~18:04 | Tar symlink traversal | Fixed `_safe_extract` member-by-member |
| 2026-07-23 | K3 verify ~18:04 | Bundle slash collision | Fixed canonical slug + segment bind |
| 2026-07-23 | K3 verify ~18:04 | Pretrained LoRA bypass | Fixed exact filename guard |
| 2026-07-23 | K3 verify ~18:04 | HF token CI leak | Fixed dedicated download script |
| 2026-07-23 | K3 verify ~18:04 | Config injection | Fixed path allowlist + metachar reject |
| 2026-07-23 | K3 post-#8 ~18:34 | job_id path segment | Fixed `check_job_id_slug` in handler + progress key builders |
| 2026-07-23 | K3 post-#8 ~18:34 | HF_TOKEN in build env | Fixed dedicated script; GH Actions secret masking on step env |
| 2026-07-23 | K3 post-#8 ~18:34 | Tar hardlink race | Fixed `_safe_extract` rejects `islnk()`/`issym()`; member-by-member extract |
| 2026-07-23 | K3 post-#8 ~18:34 | progress store ignores R2 write failures | **Fixed** -- log failures via `_log` channel |
| 2026-07-23 | K3 post-#8 ~18:34 | ai-toolkit config path outside model roots | Workdir paths under operator-controlled WAN_TRAIN_WORKDIR |
| 2026-07-23 | K3 post-#8 ~18:34 | CodeQL matrix language typo | **Fixed** -- workflow uses `languages:` plural |
| 2026-07-23 | K3 post-#8 ~18:34 | Command injection via AITOOLKIT_PYTHON env | Operator GPU env; list-form Popen, no shell |
| 2026-07-23 | K3 post-#8 ~18:34 | FakeStore metadata drift | Test-only interface; prod put_file has no metadata param |
