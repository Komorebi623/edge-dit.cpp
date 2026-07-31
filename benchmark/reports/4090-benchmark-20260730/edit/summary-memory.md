# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## flux-kontext-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 23967 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 2003 | 985 | 1750 | 1936 |
| diffusers | fp8 | no-offload | none | 23776 | 23211 | 23736 | 23580 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 18786 | 9899 | 18786 | 1714 |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23633 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12108 | 10674 | 12108 | 10860 |
| edge-dit.cpp | q8_0 | no-offload | none | 19998 | 18439 | 19998 | 18750 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 17765 | 9681 | 17765 | 1510 |
| stable-diffusion.cpp | q4_k | no-offload | none | 11858 | 3805 | 11858 | 10868 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 19380 | 5685 | 19380 | 18388 |

## kontext-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 18786 | 9900 | 18786 | 1714 |
| edge-dit.cpp | f16 | no-offload | none | 23594 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 23470 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12108 | 10674 | 12108 | 10860 |
| edge-dit.cpp | q8_0 | no-offload | none | 19998 | 18564 | 19998 | 18750 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24044 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 4785 | 2994 | 1756 | 3940 |
| diffusers | fp8 | no-offload | none | 24062 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19088 | 15381 | 19088 | 1694 |
| edge-dit.cpp | f16 | no-offload | none | 412 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 412 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 18666 | 18666 | 13174 | 12223 |
| edge-dit.cpp | q4_k | no-offload | none | 19296 | 18628 | 19296 | 18293 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 16040 | 8822 | 16040 | 1516 |
| edge-dit.cpp | q8_0 | no-offload | none | 21390 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 16991 | 13988 | 16991 | 1302 |
| stable-diffusion.cpp | f16 | no-offload | none | 14068 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 17776 | 6162 | 17774 | 17776 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 17816 | 7654 | 17816 | 1192 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 7759 | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19088 | 15378 | 19088 | 1694 |
| edge-dit.cpp | f16 | no-offload | none | 412 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 412 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 18666 | 18666 | 13175 | 12225 |
| edge-dit.cpp | q4_k | no-offload | none | 19296 | 18628 | 19296 | 18398 |
| edge-dit.cpp | q8_0 | no-offload | none | 21390 | — | — | — |
