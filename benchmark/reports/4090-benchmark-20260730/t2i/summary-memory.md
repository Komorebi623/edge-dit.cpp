# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-dev-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 24027 | 9632 | 24027 | 23879 |
| diffusers | bf16 | no-offload | none | 24023 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 1896 | 739 | 1288 | 1896 |
| diffusers | fp8 | no-offload | none | 23824 | 21734 | 22897 | 23699 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19513 | 9904 | 19513 | 1314 |
| edge-dit.cpp | f16 | no-offload | none | 23535 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23574 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11177 | 10243 | 11177 | 10681 |
| edge-dit.cpp | q8_0 | no-offload | none | 19067 | 18137 | 19067 | 18571 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17512 | 9535 | 17512 | 1257 |
| stable-diffusion.cpp | f16 | no-offload | none | 9856 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 11003 | 3661 | 11003 | 10737 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 18525 | 5541 | 18525 | 18259 |

## flux-schnell-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24023 | — | — | — |
| diffusers | fp8 | no-offload | none | 23757 | 21716 | 22875 | 23301 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19491 | 9901 | 19491 | 1313 |
| edge-dit.cpp | f16 | no-offload | none | 23515 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23555 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 11160 | 10224 | 11160 | 10664 |
| edge-dit.cpp | q8_0 | no-offload | none | 19047 | 18111 | 19047 | 18551 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17404 | 9535 | 17404 | 1257 |
| stable-diffusion.cpp | f16 | no-offload | none | 9858 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 10983 | 3661 | 10983 | 10717 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 18505 | 5541 | 18505 | 18239 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19391 | 14033 | 19391 | 1325 |
| edge-dit.cpp | f16 | no-offload | none | 417 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 417 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18644 | 18644 | 12186 | 12022 |
| edge-dit.cpp | q4_k | no-offload | none | 17911 | 17202 | 17911 | 17676 |
| edge-dit.cpp | q8_0 | no-offload | none | 21288 | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24058 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 4572 | 1584 | 1016 | 3789 |
| diffusers | fp8 | no-offload | none | 24022 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19400 | 14041 | 19400 | 1334 |
| edge-dit.cpp | f16 | no-offload | none | 426 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 426 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 18635 | 18635 | 12212 | 12078 |
| edge-dit.cpp | q4_k | no-offload | none | 17920 | 17207 | 17920 | 17646 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 16817 | 7819 | 16817 | 1222 |
| edge-dit.cpp | q8_0 | no-offload | none | 21300 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 16941 | 13961 | 16941 | 1316 |
| stable-diffusion.cpp | f16 | no-offload | none | 13938 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 17688 | 6052 | 17682 | 17688 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 17795 | 7628 | 17795 | 1206 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7655 | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20217 | 15732 | 16217 | 17383 |
| diffusers | fp8 | no-offload | none | 18431 | 13824 | 14431 | 16892 |
| edge-dit.cpp | f16 | no-offload | none | 15873 | 15445 | 15807 | 15873 |
| edge-dit.cpp | q4_k | no-offload | none | 5757 | 5426 | 5691 | 5757 |
| edge-dit.cpp | q8_0 | no-offload | none | 9263 | 8905 | 9197 | 9263 |
| stable-diffusion.cpp | f16 | no-offload | none | 15959 | 11236 | 15647 | 15959 |
| stable-diffusion.cpp | q4_k | no-offload | none | 6103 | 4199 | 5789 | 6103 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 9241 | 6381 | 8927 | 9241 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20757 | 16340 | 16629 | 18612 |
| diffusers | fp8 | no-offload | none | 18723 | 14255 | 14595 | 16274 |
| edge-dit.cpp | f16 | no-offload | none | 16817 | 16358 | 16714 | 16817 |
| edge-dit.cpp | q4_k | no-offload | none | 6372 | 6052 | 6287 | 6372 |
| edge-dit.cpp | q8_0 | no-offload | none | 10015 | 9637 | 9904 | 10015 |
