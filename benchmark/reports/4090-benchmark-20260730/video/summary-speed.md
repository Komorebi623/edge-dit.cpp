# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## wan2-t2v-1.3b-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3374.6 | 4460.0 | 285.6 | 767.8 |
| diffusers | fp8 | no-offload | none | 4301.8 | 5351.2 | 275.3 | 735.3 |
| edge-dit.cpp | f16 | no-offload | none | 5735.9 | 56903.8 | 422.4 | 1556.2 |
| edge-dit.cpp | f16 | te offload | none | 6388.2 | 57661.8 | 2775.0 | 1437.2 |
| edge-dit.cpp | q4_k | no-offload | none | 6425.6 | 97302.4 | 355.6 | 1549.4 |
| edge-dit.cpp | q8_0 | no-offload | none | 6399.6 | 26196.3 | 383.3 | 2342.9 |

## wan2-t2v-14b-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 145450.4 | 159064.2 | 5633.1 | 3184.8 |
| diffusers | fp8 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 111290.3 | 507998.7 | 1839.9 | 1963.8 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 36660.1 | 317439.9 | 1794.1 | 1706.9 |
| edge-dit.cpp | q4_k | no-offload | none | 37324.2 | 846612.8 | 393.1 | 1401.5 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 36668.4 | 194480.3 | 1704.2 | 1767.3 |

## wan21-t2v-1.3b-distill-text-to-video

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 1429.7 | 26619.2 | 334.7 | 1719.7 |
| edge-dit.cpp | q4_k | no-offload | none | 1662.1 | 113939.4 | 264.7 | 1742.5 |
| edge-dit.cpp | q8_0 | no-offload | none | 1647.6 | 26241.4 | 295.0 | 2056.2 |
