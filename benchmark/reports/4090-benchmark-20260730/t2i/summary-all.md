# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 154784.2 | 206320.3 | 24027 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | no-offload | none | — | — | 24023 | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 180799.4 | 186892.6 | 1896 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | fp8 | no-offload | none | 15880.3 | 16999.1 | 23824 | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 44472.0 | 46621.0 | 19513 | 0.303 | 6.07 | 1.797 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23535 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23574 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10835.0 | 11605.7 | 11177 | 0.299 | 5.98 | 1.834 | 20.14 | 0.815 | 0.223 |
| edge-dit.cpp | q8_0 | no-offload | none | 11130.2 | 12372.3 | 19067 | 0.281 | 5.93 | 1.762 | 16.35 | 0.712 | 0.363 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 63760.0 | 73631.1 | 17512 | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 9856 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 100526.7 | 139972.8 | 11003 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 20406.7 | 26522.6 | 18525 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24023 | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | 3199.0 | 4002.1 | 23757 | 0.298 | 5.64 | 1.766 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 9597.9 | 11626.0 | 19491 | 0.317 | 5.46 | 1.881 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23515 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23555 | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2326.2 | 3084.3 | 11160 | 0.310 | 5.74 | 1.809 | 14.70 | 0.693 | 0.299 |
| edge-dit.cpp | q8_0 | no-offload | none | 2297.8 | 3048.0 | 19047 | 0.312 | 5.58 | 1.885 | 28.52 | 0.944 | 0.036 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 56636.7 | 76481.3 | 17404 | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 9858 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 63176.7 | 99532.8 | 10983 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 9083.3 | 15559.6 | 18505 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 14735.4 | 17368.7 | 19391 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 417 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 417 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 3172.1 | 4627.3 | 18644 | 0.327 | 5.49 | 1.842 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2576.5 | 3418.7 | 17911 | 0.333 | 5.53 | 1.853 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21288 | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24058 | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 390028.3 | 422715.5 | 4572 | 0.314 | 5.53 | 1.850 | — | — | — |
| diffusers | fp8 | no-offload | none | — | — | 24022 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 104382.7 | 107728.7 | 19400 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 426 | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 426 | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 17605.1 | 19033.0 | 18635 | 0.327 | 5.26 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 17719.6 | 18771.0 | 17920 | 0.331 | 5.28 | 1.860 | 2.62 | 0.354 | 0.932 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 64377.9 | 67607.7 | 16817 | 0.312 | 5.59 | 1.880 | 2.35 | 0.329 | 0.929 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21300 | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 164373.3 | 194118.7 | 16941 | 0.174 | 4.30 | -1.144 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 13938 | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 211723.3 | 283997.0 | 17688 | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 84083.3 | 103175.0 | 17795 | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | 7655 | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3092.0 | 3784.8 | 20217 | 0.344 | 4.92 | 1.802 | — | — | — |
| diffusers | fp8 | no-offload | none | 4150.4 | 4755.3 | 18431 | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| edge-dit.cpp | f16 | no-offload | none | 3271.3 | 4077.3 | 15873 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 3444.0 | 4236.0 | 5757 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | 3469.9 | 4247.7 | 9263 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| stable-diffusion.cpp | f16 | no-offload | none | 5456.7 | 8814.9 | 15959 | 0.208 | 3.40 | -2.275 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 16553.3 | 59388.3 | 6103 | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 5316.7 | 11633.1 | 9241 | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 548.2 | 1277.4 | 20757 | 0.308 | 5.12 | -2.272 | — | — | — |
| diffusers | fp8 | no-offload | none | 729.9 | 1307.3 | 18723 | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| edge-dit.cpp | f16 | no-offload | none | 1076.4 | 6611.0 | 16817 | 0.335 | 5.21 | 1.641 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 593.6 | 1605.0 | 6372 | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.179 |
| edge-dit.cpp | q8_0 | no-offload | none | 526.7 | 1269.3 | 10015 | 0.333 | 5.27 | 1.654 | 37.64 | 0.989 | 0.016 |
