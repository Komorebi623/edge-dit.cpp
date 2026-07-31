# VRAM summary (mean, unit MiB)

> budget names the offloaded components (e.g. `te offload`, `full offload`) + `(max-vram Ng)` when --max-vram was set; auto tiers show the engine's real placement + `(auto-fit)`/`(auto-allocate)`. cache is its own column.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 20578 | 14607 | 14964 | 20578 |
| diffusers | fp8 | no-offload | none | 19628 | 13222 | 13574 | 19628 |
| edge-dit.cpp | f16 | no-offload | none | 17100 | 16313 | 16556 | 17100 |
| edge-dit.cpp | f16 | te offload | none | 16536 | 16536 | 3740 | 4284 |
| edge-dit.cpp | q4_k | no-offload | none | 8826 | 8272 | 8282 | 8826 |
| edge-dit.cpp | q8_0 | no-offload | none | 11704 | 11147 | 11160 | 11704 |

## wan2-t2v-14b-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 24062 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 5269 | 2418 | 1389 | 5257 |
| diffusers | fp8 | no-offload | none | 24076 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 19038 | 13844 | 19038 | 2004 |
| edge-dit.cpp | f16 | no-offload | none | 4739 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | 475 | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 19373 | 19373 | 16084 | 16398 |
| edge-dit.cpp | q4_k | no-offload | none | 16092 | 15511 | 15918 | 16092 |
| edge-dit.cpp | q8_0 | no-offload | none | 24052 | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 19396 | 19396 | 16084 | 16398 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 17084 | 16486 | 16540 | 17084 |
| edge-dit.cpp | q4_k | no-offload | none | 8810 | 8145 | 8266 | 8810 |
| edge-dit.cpp | q8_0 | no-offload | none | 11688 | 11124 | 11144 | 11688 |
