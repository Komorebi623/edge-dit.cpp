# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3374.6 | 4460.0 | 20578 | 0.214 | 4.70 | 0.116 | 0.580 | 0.025 | — | — | — |
| diffusers | fp8 | no-offload | none | 4301.8 | 5351.2 | 19628 | 0.217 | 4.63 | 0.119 | 0.589 | 0.026 | 29.11 | 0.657 | 0.082 |
| edge-dit.cpp | f16 | no-offload | none | 5735.9 | 56903.8 | 17100 | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | f16 | te offload | none | 6388.2 | 57661.8 | 16536 | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 6425.6 | 97302.4 | 8826 | 0.305 | 5.04 | 0.070 | 0.749 | 0.026 | 25.35 | 0.801 | 0.200 |
| edge-dit.cpp | q8_0 | no-offload | none | 6399.6 | 26196.3 | 11704 | 0.313 | 5.16 | 0.073 | 0.765 | 0.027 | 30.93 | 0.912 | 0.065 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24062 | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 145450.4 | 159064.2 | 5269 | 0.296 | 4.73 | 0.106 | 0.581 | 0.050 | — | — | — |
| diffusers | fp8 | no-offload | none | — | — | 24076 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 111290.3 | 507998.7 | 19038 | 0.183 | 4.15 | 0.022 | 0.963 | 0.021 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 4739 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 475 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 36660.1 | 317439.9 | 19373 | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 37324.2 | 846612.8 | 16092 | 0.327 | 5.62 | 0.046 | 0.949 | 0.032 | 10.79 | 0.392 | 0.761 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 24052 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 36668.4 | 194480.3 | 19396 | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | 39.21 | 0.531 | 0.674 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 1429.7 | 26619.2 | 17084 | 0.314 | 5.42 | 0.044 | 0.859 | 0.027 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 1662.1 | 113939.4 | 8810 | 0.319 | 5.41 | 0.037 | 0.885 | 0.024 | 18.14 | 0.571 | 0.287 |
| edge-dit.cpp | q8_0 | no-offload | none | 1647.6 | 26241.4 | 11688 | 0.314 | 5.35 | 0.045 | 0.845 | 0.027 | 25.75 | 0.778 | 0.192 |
