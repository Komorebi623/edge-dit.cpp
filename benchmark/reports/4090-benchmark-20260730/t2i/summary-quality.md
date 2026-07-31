# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | fp8 | no-offload | none | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.303 | 6.07 | 1.797 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.299 | 5.98 | 1.834 | 20.14 | 0.815 | 0.223 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.281 | 5.93 | 1.762 | 16.35 | 0.712 | 0.363 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | 0.298 | 5.64 | 1.766 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.317 | 5.46 | 1.881 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.310 | 5.74 | 1.809 | 14.70 | 0.693 | 0.299 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.312 | 5.58 | 1.885 | 28.52 | 0.944 | 0.036 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.327 | 5.49 | 1.842 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.333 | 5.53 | 1.853 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.314 | 5.53 | 1.850 | — | — | — |
| diffusers | fp8 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.327 | 5.26 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.331 | 5.28 | 1.860 | 2.62 | 0.354 | 0.932 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 0.312 | 5.59 | 1.880 | 2.35 | 0.329 | 0.929 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.174 | 4.30 | -1.144 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.344 | 4.92 | 1.802 | — | — | — |
| diffusers | fp8 | no-offload | none | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| edge-dit.cpp | f16 | no-offload | none | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| stable-diffusion.cpp | f16 | no-offload | none | 0.208 | 3.40 | -2.275 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.308 | 5.12 | -2.272 | — | — | — |
| diffusers | fp8 | no-offload | none | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| edge-dit.cpp | f16 | no-offload | none | 0.335 | 5.21 | 1.641 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.179 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.333 | 5.27 | 1.654 | 37.64 | 0.989 | 0.016 |
