# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## flux-kontext-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 152136.9 | 156335.9 | 3704.5 | 463.9 |
| diffusers | fp8 | no-offload | none | 32838.1 | 35191.1 | 1640.1 | 663.6 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 65595.5 | 182473.4 | 15397.7 | 618.8 |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 25018.1 | 187461.0 | 1173.6 | 583.6 |
| edge-dit.cpp | q8_0 | no-offload | none | 25111.2 | 47030.0 | 868.5 | 721.1 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 165500.0 | 199991.5 | 30756.7 | 2030.0 |
| stable-diffusion.cpp | q4_k | no-offload | none | 137606.7 | 184114.2 | 43716.7 | 1553.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 58496.7 | 71711.0 | 10796.7 | 1210.0 |

## kontext-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 23685.1 | 126528.5 | 1947.1 | 746.5 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10110.1 | 141028.3 | 709.8 | 464.3 |
| edge-dit.cpp | q8_0 | no-offload | none | 9994.7 | 28691.2 | 705.8 | 443.5 |

## qwen-image-edit-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 640826.8 | 716356.7 | 74505.7 | 975.1 |
| diffusers | fp8 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 141185.9 | 353115.8 | 2802.7 | 1362.5 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 44628.8 | 540089.5 | 1806.6 | 553.8 |
| edge-dit.cpp | q4_k | no-offload | none | 44266.0 | 541223.0 | 1280.7 | 556.6 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 86610.2 | 304349.8 | 1645.2 | 671.8 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 127056.7 | 153148.3 | 19866.7 | 4170.0 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 209036.7 | 297296.7 | 84430.0 | 2320.0 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 68560.0 | 91054.7 | 17330.0 | 3313.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-edit-lightning-image-editing

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 34094.1 | 401569.6 | 2561.7 | 1484.1 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 12557.7 | 379631.3 | 1673.7 | 641.8 |
| edge-dit.cpp | q4_k | no-offload | none | 12021.8 | 451947.2 | 1316.7 | 585.1 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
