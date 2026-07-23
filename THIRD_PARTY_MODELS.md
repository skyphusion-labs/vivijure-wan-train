# Third-party models (Hub / baked image)

The `ghcr.io/skyphusion-labs/vivijure-wan-train` image bakes ai-toolkit and the three HF repos
it loads at train time so a worker runs offline.

| Role | Component | License | Source |
| --- | --- | --- | --- |
| Trainer | ai-toolkit (Ostris) | MIT | https://github.com/ostris/ai-toolkit |
| Wan DiT base | ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16 | Apache-2.0 | HuggingFace |
| Text encoder | ai-toolkit/umt5_xxl_encoder | Apache-2.0 | HuggingFace |
| VAE | ai-toolkit/wan2.1-vae | Apache-2.0 | HuggingFace |

Wrapper code in this repository is **AGPL-3.0** (see `LICENSE`).
