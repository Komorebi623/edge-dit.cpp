# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.214 | 4.70 | 0.116 | 0.580 | 0.025 | — | — | — |
| diffusers | fp8 | no-offload | none | 0.217 | 4.63 | 0.119 | 0.589 | 0.026 | 29.11 | 0.657 | 0.082 |
| edge-dit.cpp | f16 | no-offload | none | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | f16 | te offload | none | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.305 | 5.04 | 0.070 | 0.749 | 0.026 | 25.35 | 0.801 | 0.200 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.313 | 5.16 | 0.073 | 0.765 | 0.027 | 30.93 | 0.912 | 0.065 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.296 | 4.73 | 0.106 | 0.581 | 0.050 | — | — | — |
| diffusers | fp8 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.183 | 4.15 | 0.022 | 0.963 | 0.021 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.327 | 5.62 | 0.046 | 0.949 | 0.032 | 10.79 | 0.392 | 0.761 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | 39.21 | 0.531 | 0.674 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.314 | 5.42 | 0.044 | 0.859 | 0.027 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.319 | 5.41 | 0.037 | 0.885 | 0.024 | 18.14 | 0.571 | 0.287 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.314 | 5.35 | 0.045 | 0.845 | 0.027 | 25.75 | 0.778 | 0.192 |
