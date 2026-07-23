# Security false positives (K3 adversarial audit)

Documented K3 repo-mode findings that are accepted under the Vivijure GPU operator trust model.

## Tar bundle symlink extraction (`contract.py`)

**Finding:** Tar extraction path traversal via symlink member.

**Disposition:** Deferred operator-bundle follow-up. Bundles are produced by the trusted control plane and uploaded to project-scoped R2 keys validated by `check_bundle_key_for_project`. The worker rejects symlink members in `_safe_extract`; remaining traversal classes are tracked for a dedicated hardening pass, not a same-day K3 blocker.

## Pretrained LoRA key passthrough (`job.py`)

**Finding:** Arbitrary R2 key writes via `pretrained_loras`.

**Disposition:** False positive. `check_scoped_lora_key` requires keys under `loras/<project>/`; the control plane supplies passthrough keys for already-trained slots. No cross-project write path exists without a forged bundle + project slug pair.

## Offline ai-toolkit hub ID patch (`wan_lora_train.py`)

**Finding:** Unpatched ai-toolkit hub IDs leave offline training failures.

**Disposition:** False positive. The bake step pins ai-toolkit and applies offline hub rewrites at image build; runtime patch is a belt-and-suspenders seam for the exact quoted IDs ai-toolkit ships in the baked tree.
