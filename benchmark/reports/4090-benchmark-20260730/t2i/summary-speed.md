# Speed summary (mean, unit ms)

> For inference speed look at **DiT sampling ms** (reliable); end-to-end includes loading / on-the-fly quantization conversion and is not comparable across systems. sd.cpp quantized tiers include on-the-fly convert in DiT, so it is inflated.


## flux-dev-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 154784.2 | 206320.3 | 23830.3 | 27350.5 |
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 180799.4 | 186892.6 | 4898.4 | 1119.8 |
| diffusers | fp8 | no-offload | none | 15880.3 | 16999.1 | 307.0 | 760.6 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 44472.0 | 46621.0 | 1385.1 | 747.9 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10835.0 | 11605.7 | 272.4 | 492.0 |
| edge-dit.cpp | q8_0 | no-offload | none | 11130.2 | 12372.3 | 611.8 | 621.8 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 63760.0 | 73631.1 | 8456.7 | 1400.0 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 100526.7 | 139972.8 | 38156.7 | 1260.0 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 20406.7 | 26522.6 | 4840.0 | 1253.3 |

## flux-schnell-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | fp8 | no-offload | none | 3199.0 | 4002.1 | 288.7 | 494.6 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 9597.9 | 11626.0 | 1111.5 | 896.2 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 2326.2 | 3084.3 | 263.8 | 488.3 |
| edge-dit.cpp | q8_0 | no-offload | none | 2297.8 | 3048.0 | 253.1 | 491.1 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 56636.7 | 76481.3 | 18246.7 | 1573.3 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 63176.7 | 99532.8 | 35146.7 | 1186.7 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 9083.3 | 15559.6 | 5253.3 | 1206.7 |

## qwen-image-lightning-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 14735.4 | 17368.7 | 1393.9 | 1224.5 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 3172.1 | 4627.3 | 640.0 | 809.4 |
| edge-dit.cpp | q4_k | no-offload | none | 2576.5 | 3418.7 | 257.5 | 578.2 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |

## qwen-image-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 390028.3 | 422715.5 | 26706.1 | 5888.7 |
| diffusers | fp8 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 104382.7 | 107728.7 | 2274.0 | 1062.6 |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 17605.1 | 19033.0 | 625.5 | 796.3 |
| edge-dit.cpp | q4_k | no-offload | none | 17719.6 | 18771.0 | 264.3 | 778.3 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 64377.9 | 67607.7 | 2382.1 | 838.3 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 164373.3 | 194118.7 | 25516.7 | 4210.0 |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 211723.3 | 283997.0 | 69423.3 | 2830.0 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 84083.3 | 103175.0 | 15790.0 | 3290.0 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — |

## sd3-medium-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 3092.0 | 3784.8 | 329.0 | 339.1 |
| diffusers | fp8 | no-offload | none | 4150.4 | 4755.3 | 307.1 | 277.6 |
| edge-dit.cpp | f16 | no-offload | none | 3271.3 | 4077.3 | 384.5 | 414.7 |
| edge-dit.cpp | q4_k | no-offload | none | 3444.0 | 4236.0 | 366.8 | 418.8 |
| edge-dit.cpp | q8_0 | no-offload | none | 3469.9 | 4247.7 | 355.2 | 416.1 |
| stable-diffusion.cpp | f16 | no-offload | none | 5456.7 | 8814.9 | 2170.0 | 1170.0 |
| stable-diffusion.cpp | q4_k | no-offload | none | 16553.3 | 59388.3 | 41636.7 | 1173.3 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 5316.7 | 11633.1 | 5166.7 | 1126.7 |

## sd35-medium-turbo-text-to-image

| system | precision | budget | cache | DiT sampling ms | end-to-end ms | TE_ms | VAE_ms |
|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 548.2 | 1277.4 | 399.5 | 305.7 |
| diffusers | fp8 | no-offload | none | 729.9 | 1307.3 | 280.7 | 279.4 |
| edge-dit.cpp | f16 | no-offload | none | 1076.4 | 6611.0 | 5072.5 | 455.4 |
| edge-dit.cpp | q4_k | no-offload | none | 593.6 | 1605.0 | 416.4 | 581.4 |
| edge-dit.cpp | q8_0 | no-offload | none | 526.7 | 1269.3 | 305.6 | 430.4 |
