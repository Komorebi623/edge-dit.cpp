# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | — | — | 23967 | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 152136.9 | 156335.9 | 2003 | 0.074 | 0.674 | 0.433 | 5.99 | -0.317 | — | — | — |
| diffusers | fp8 | no-offload | none | 32838.1 | 35191.1 | 23776 | 0.069 | 0.666 | 0.443 | 5.98 | -0.314 | 32.95 | 0.952 | 0.021 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 65595.5 | 182473.4 | 18786 | -0.067 | 0.820 | 0.215 | 5.85 | -0.782 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23633 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 25018.1 | 187461.0 | 12108 | -0.068 | 0.837 | 0.191 | 5.88 | -0.806 | 35.52 | 0.962 | 0.023 |
| edge-dit.cpp | q8_0 | no-offload | none | 25111.2 | 47030.0 | 19998 | -0.071 | 0.818 | 0.216 | 5.86 | -0.764 | 49.12 | 0.996 | 0.001 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 165500.0 | 199991.5 | 17765 | -0.089 | 0.592 | 0.479 | 5.67 | -0.618 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 137606.7 | 184114.2 | 11858 | -0.081 | 0.597 | 0.481 | 5.68 | -0.688 | 25.45 | 0.872 | 0.096 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 58496.7 | 71711.0 | 19380 | -0.084 | 0.589 | 0.484 | 5.70 | -0.635 | 31.29 | 0.977 | 0.020 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 23685.1 | 126528.5 | 18786 | -0.071 | 0.823 | 0.204 | 5.84 | -0.578 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 23594 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 23470 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 10110.1 | 141028.3 | 12108 | -0.061 | 0.826 | 0.200 | 5.83 | -0.608 | 40.05 | 0.970 | 0.014 |
| edge-dit.cpp | q8_0 | no-offload | none | 9994.7 | 28691.2 | 19998 | -0.072 | 0.823 | 0.204 | 5.84 | -0.583 | 50.83 | 0.996 | 0.001 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | 24044 | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 640826.8 | 716356.7 | 4785 | 0.060 | 0.576 | 0.588 | 6.01 | -0.547 | — | — | — |
| diffusers | fp8 | no-offload | none | — | — | 24062 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 141185.9 | 353115.8 | 19088 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 412 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 412 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 44628.8 | 540089.5 | 18666 | -0.095 | 0.620 | 0.538 | 5.78 | -0.935 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 44266.0 | 541223.0 | 19296 | -0.131 | 0.756 | 0.320 | 5.53 | -1.189 | 13.49 | 0.680 | 0.578 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 86610.2 | 304349.8 | 16040 | -0.017 | 0.693 | 0.369 | 5.96 | -1.112 | 10.84 | 0.643 | 0.645 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21390 | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 127056.7 | 153148.3 | 16991 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | 14068 | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 209036.7 | 297296.7 | 17776 | -0.115 | 0.721 | 0.382 | 5.42 | -1.171 | 4.64 | 0.537 | 0.820 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 68560.0 | 91054.7 | 17816 | 0.030 | 0.736 | 0.349 | 5.63 | -1.131 | 4.53 | 0.536 | 0.822 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | 7759 | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 34094.1 | 401569.6 | 19088 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 412 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | — | — | 412 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | 12557.7 | 379631.3 | 18666 | -0.086 | 0.598 | 0.520 | 6.08 | -0.578 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 12021.8 | 451947.2 | 19296 | -0.099 | 0.732 | 0.345 | 5.54 | -1.171 | 10.83 | 0.653 | 0.597 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 21390 | — | — | — | — | — | — | — | — |
