# Cross-system comparison matrix (all metrics, one-shot aggregate)

108 runs total | success 66 | failed 42

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-kontext-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 23964 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | full offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 24024 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | full offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 23912 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | flux_kontext_brushed_metal | success | 155080.3 | 159395.2 | net-inference | 3808.4 | 480.2 | 1750 | 838 | 1750 | 1550 | 0.175 | 0.650 | 0.391 | 5.92 | 0.676 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | matrix_edit_watercolor | success | 146899.8 | 151305.5 | net-inference | 3942.6 | 437.2 | 1938 | 1282 | 1750 | 1938 | 0.072 | 0.596 | 0.558 | 6.47 | -0.587 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | qwen_edit_studio_background | success | 154430.5 | 158307.1 | net-inference | 3362.5 | 474.4 | 2320 | 834 | 1750 | 2320 | -0.026 | 0.774 | 0.351 | 5.59 | -1.042 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 152136.9 | 156335.9 |  | 3704.5 | 463.9 | 2003 | 985 | 1750 | 1936 | 0.074 | 0.674 | 0.433 | 5.99 | -0.317 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_kontext_brushed_metal | success | 32848.4 | 35096.5 | net-inference | 1915.7 | 277.8 | 23776 | 23724 | 23736 | 23736 | 0.147 | 0.647 | 0.399 | 5.96 | 0.635 | 33.80 | 0.965 | 0.017 |
| diffusers | fp8 | no-offload | none | matrix_edit_watercolor | success | 32843.3 | 33802.8 | net-inference | 651.0 | 276.5 | 23776 | 22184 | 23736 | 23268 | 0.086 | 0.589 | 0.565 | 6.41 | -0.529 | 31.82 | 0.913 | 0.033 |
| diffusers | fp8 | no-offload | none | qwen_edit_studio_background | success | 32822.5 | 36674.1 | net-inference | 2353.4 | 1436.6 | 23776 | 23726 | 23736 | 23736 | -0.025 | 0.762 | 0.367 | 5.58 | -1.049 | 33.22 | 0.979 | 0.012 |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 32838.1 | 35191.1 |  | 1640.1 | 663.6 | 23776 | 23211 | 23736 | 23580 | 0.069 | 0.666 | 0.443 | 5.98 | -0.314 | 32.95 | 0.952 | 0.021 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 73138.2 | 178255.2 | incl-load+encode | 42211.7 | 642.6 | 18786 | 9898 | 18786 | 1714 | -0.114 | 0.957 | 0.032 | 5.86 | -0.140 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 55511.9 | 171039.1 | incl-load+encode | 1876.3 | 519.5 | 18786 | 9900 | 18786 | 1714 | 0.028 | 0.590 | 0.522 | 5.85 | -1.160 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 68136.5 | 198125.7 | incl-load+encode | 2105.2 | 694.3 | 18786 | 9898 | 18786 | 1714 | -0.116 | 0.911 | 0.091 | 5.83 | -1.047 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 65595.5 | 182473.4 |  | 15397.7 | 618.8 | 18786 | 9899 | 18786 | 1714 | -0.067 | 0.820 | 0.215 | 5.85 | -0.782 | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 23634 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 23632 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 23634 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 25006.3 | 209138.5 | incl-load+encode | 1741.4 | 588.9 | 12108 | 10674 | 12108 | 10860 | -0.110 | 0.958 | 0.029 | 5.89 | -0.131 | 45.15 | 0.996 | 0.001 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 24898.9 | 174095.8 | incl-load+encode | 743.9 | 574.8 | 12108 | 10674 | 12108 | 10860 | 0.017 | 0.599 | 0.511 | 5.87 | -1.109 | 31.08 | 0.945 | 0.029 |
| edge-dit.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 25149.1 | 179148.5 | incl-load+encode | 1035.7 | 587.1 | 12108 | 10674 | 12108 | 10860 | -0.112 | 0.955 | 0.035 | 5.87 | -1.177 | 30.33 | 0.944 | 0.038 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 25018.1 | 187461.0 |  | 1173.6 | 583.6 | 12108 | 10674 | 12108 | 10860 | -0.068 | 0.837 | 0.191 | 5.88 | -0.806 | 35.52 | 0.962 | 0.023 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | success | 24336.7 | 51382.2 | incl-load+encode | 1152.4 | 581.8 | 19998 | 18564 | 19998 | 18750 | -0.131 | 0.957 | 0.032 | 5.86 | -0.126 | 54.53 | 0.998 | 0.000 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | success | 26597.2 | 44434.6 | incl-load+encode | 711.6 | 1034.3 | 19998 | 18190 | 19998 | 18750 | 0.034 | 0.590 | 0.522 | 5.87 | -1.130 | 42.50 | 0.993 | 0.002 |
| edge-dit.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | success | 24399.7 | 45273.3 | incl-load+encode | 741.3 | 547.1 | 19998 | 18564 | 19998 | 18750 | -0.117 | 0.908 | 0.093 | 5.84 | -1.034 | 50.32 | 0.997 | 0.001 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 25111.2 | 47030.0 |  | 868.5 | 721.1 | 19998 | 18439 | 19998 | 18750 | -0.071 | 0.818 | 0.216 | 5.86 | -0.764 | 49.12 | 0.996 | 0.001 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 173080.0 | 204142.4 | net-inference | 27220.0 | 2460.0 | 17800 | 9728 | 17800 | 1510 | -0.185 | 0.730 | 0.348 | 5.78 | -0.165 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 162960.0 | 206172.1 | net-inference | 39430.0 | 1790.0 | 17748 | 9658 | 17748 | 1510 | 0.044 | 0.311 | 0.741 | 5.48 | -0.493 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 160460.0 | 189660.0 | net-inference | 25620.0 | 1840.0 | 17748 | 9658 | 17748 | 1510 | -0.124 | 0.734 | 0.347 | 5.76 | -1.197 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 165500.0 | 199991.5 |  | 30756.7 | 2030.0 | 17765 | 9681 | 17765 | 1510 | -0.089 | 0.592 | 0.479 | 5.67 | -0.618 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 130460.0 | 169643.8 | net-inference | 36560.0 | 1530.0 | 11858 | 3782 | 11858 | 10868 | -0.172 | 0.721 | 0.353 | 5.80 | -0.150 | 28.94 | 0.952 | 0.047 |
| stable-diffusion.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 146380.0 | 197392.1 | net-inference | 48130.0 | 1590.0 | 11858 | 3782 | 11858 | 10868 | 0.044 | 0.338 | 0.734 | 5.51 | -0.750 | 18.63 | 0.716 | 0.187 |
| stable-diffusion.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 135980.0 | 185306.8 | net-inference | 46460.0 | 1540.0 | 11858 | 3852 | 11858 | 10868 | -0.116 | 0.731 | 0.357 | 5.75 | -1.163 | 28.77 | 0.946 | 0.055 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 137606.7 | 184114.2 |  | 43716.7 | 1553.3 | 11858 | 3805 | 11858 | 10868 | -0.081 | 0.597 | 0.481 | 5.68 | -0.688 | 25.45 | 0.872 | 0.096 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | success | 58000.0 | 74974.2 | net-inference | 14430.0 | 1210.0 | 19380 | 5662 | 19380 | 18388 | -0.167 | 0.725 | 0.355 | 5.83 | -0.102 | 33.41 | 0.987 | 0.014 |
| stable-diffusion.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | success | 58900.0 | 73732.4 | net-inference | 12410.0 | 1200.0 | 19380 | 5662 | 19380 | 18388 | 0.045 | 0.309 | 0.743 | 5.53 | -0.592 | 28.78 | 0.962 | 0.027 |
| stable-diffusion.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | success | 58590.0 | 66426.5 | net-inference | 5550.0 | 1220.0 | 19380 | 5732 | 19380 | 18388 | -0.129 | 0.732 | 0.353 | 5.73 | -1.211 | 31.69 | 0.982 | 0.018 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 58496.7 | 71711.0 |  | 10796.7 | 1210.0 | 19380 | 5685 | 19380 | 18388 | -0.084 | 0.589 | 0.484 | 5.70 | -0.635 | 31.29 | 0.977 | 0.020 |

## kontext-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 26359.7 | 169157.4 | incl-load+encode | 1890.9 | 540.4 | 18786 | 9898 | 18786 | 1714 | -0.136 | 0.954 | 0.034 | 5.80 | -0.132 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 21527.2 | 118071.8 | incl-load+encode | 1662.9 | 1014.1 | 18786 | 9898 | 18786 | 1714 | 0.037 | 0.565 | 0.542 | 5.91 | -0.404 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 23168.3 | 92356.1 | incl-load+encode | 2287.5 | 685.0 | 18786 | 9904 | 18786 | 1714 | -0.115 | 0.951 | 0.037 | 5.81 | -1.199 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 23685.1 | 126528.5 |  | 1947.1 | 746.5 | 18786 | 9900 | 18786 | 1714 | -0.071 | 0.823 | 0.204 | 5.84 | -0.578 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 23594 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 23594 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 23594 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 23634 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 23386 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 23390 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 10094.4 | 142371.0 | incl-load+encode | 736.4 | 448.9 | 12108 | 10674 | 12108 | 10860 | -0.110 | 0.955 | 0.030 | 5.83 | -0.145 | 45.44 | 0.996 | 0.002 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 10075.3 | 133368.5 | incl-load+encode | 688.8 | 493.7 | 12108 | 10674 | 12108 | 10860 | 0.032 | 0.572 | 0.534 | 5.84 | -0.474 | 29.40 | 0.919 | 0.040 |
| edge-dit.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 10160.6 | 147345.4 | incl-load+encode | 704.1 | 450.3 | 12108 | 10674 | 12108 | 10860 | -0.103 | 0.952 | 0.035 | 5.83 | -1.206 | 45.32 | 0.996 | 0.001 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 10110.1 | 141028.3 |  | 709.8 | 464.3 | 12108 | 10674 | 12108 | 10860 | -0.061 | 0.826 | 0.200 | 5.83 | -0.608 | 40.05 | 0.970 | 0.014 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | success | 10019.6 | 28007.4 | incl-load+encode | 701.8 | 446.9 | 19998 | 18564 | 19998 | 18750 | -0.138 | 0.954 | 0.034 | 5.79 | -0.142 | 55.29 | 0.999 | 0.000 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | success | 9969.9 | 28596.6 | incl-load+encode | 721.9 | 437.9 | 19998 | 18564 | 19998 | 18750 | 0.042 | 0.564 | 0.542 | 5.90 | -0.398 | 41.76 | 0.991 | 0.003 |
| edge-dit.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | success | 9994.6 | 29469.7 | incl-load+encode | 693.8 | 445.9 | 19998 | 18564 | 19998 | 18750 | -0.121 | 0.951 | 0.037 | 5.82 | -1.209 | 55.43 | 0.999 | 0.000 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 9994.7 | 28691.2 |  | 705.8 | 443.5 | 19998 | 18564 | 19998 | 18750 | -0.072 | 0.823 | 0.204 | 5.84 | -0.583 | 50.83 | 0.996 | 0.001 |

## qwen-image-edit-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 24044 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 24044 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 24044 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | flux_kontext_brushed_metal | success | 1296866.0 | 1408046.7 | net-inference | 110308.2 | 838.7 | 4662 | 2924 | 1756 | 4454 | 0.067 | 0.686 | 0.363 | 6.11 | 0.037 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | matrix_edit_watercolor | success | 303625.7 | 319052.1 | net-inference | 14128.2 | 1210.8 | 5030 | 2938 | 1756 | 5030 | 0.125 | 0.484 | 0.689 | 6.47 | -0.640 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | qwen_edit_studio_background | success | 321988.6 | 421971.1 | net-inference | 99080.7 | 875.9 | 4662 | 3120 | 1756 | 2336 | -0.013 | 0.559 | 0.710 | 5.44 | -1.036 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 640826.8 | 716356.7 |  | 74505.7 | 975.1 | 4785 | 2994 | 1756 | 3940 | 0.060 | 0.576 | 0.588 | 6.01 | -0.547 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 143632.2 | 361900.9 | incl-load+encode | 2775.9 | 1397.9 | 19088 | 15388 | 19088 | 1694 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 137718.1 | 352605.0 | incl-load+encode | 3289.5 | 956.3 | 19088 | 15378 | 19088 | 1694 | 0.018 | 0.533 | 0.837 | 4.30 | -0.985 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 142207.4 | 344841.6 | incl-load+encode | 2342.8 | 1733.4 | 19088 | 15378 | 19088 | 1694 | 0.036 | 0.533 | 0.837 | 4.30 | -1.051 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 141185.9 | 353115.8 |  | 2802.7 | 1362.5 | 19088 | 15381 | 19088 | 1694 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | flux_kontext_brushed_metal | success | 44339.5 | 476458.2 | incl-load+encode | 1507.6 | 605.7 | 18666 | 18666 | 13174 | 12118 | -0.109 | 0.533 | 0.668 | 5.65 | -0.780 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | matrix_edit_watercolor | success | 45196.9 | 552930.7 | incl-load+encode | 1506.0 | 495.7 | 18666 | 18666 | 13174 | 12276 | -0.005 | 0.578 | 0.603 | 6.15 | -0.850 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | qwen_edit_studio_background | success | 44350.0 | 590879.7 | incl-load+encode | 2406.3 | 559.9 | 18666 | 18666 | 13174 | 12276 | -0.171 | 0.748 | 0.341 | 5.54 | -1.175 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 44628.8 | 540089.5 |  | 1806.6 | 553.8 | 18666 | 18666 | 13174 | 12223 | -0.095 | 0.620 | 0.538 | 5.78 | -0.935 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 44164.5 | 493863.7 | incl-load+encode | 1399.1 | 538.1 | 19296 | 18628 | 19296 | 18398 | -0.163 | 0.765 | 0.308 | 5.54 | -0.520 | 4.51 | 0.529 | 0.856 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 44269.1 | 543983.1 | incl-load+encode | 1213.5 | 506.5 | 19296 | 18628 | 19296 | 18240 | -0.060 | 0.752 | 0.328 | 5.51 | -1.797 | 4.64 | 0.538 | 0.854 |
| edge-dit.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 44364.5 | 585822.2 | incl-load+encode | 1229.7 | 625.3 | 19296 | 18628 | 19296 | 18240 | -0.168 | 0.753 | 0.325 | 5.55 | -1.249 | 31.30 | 0.972 | 0.023 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 44266.0 | 541223.0 |  | 1280.7 | 556.6 | 19296 | 18628 | 19296 | 18293 | -0.131 | 0.756 | 0.320 | 5.53 | -1.189 | 13.49 | 0.680 | 0.578 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | flux_kontext_brushed_metal | success | 87444.0 | 386337.2 | incl-load+encode | 1620.4 | 624.4 | 16040 | 8824 | 16040 | 1516 | -0.065 | 0.693 | 0.370 | 5.99 | -0.252 | 4.47 | 0.536 | 0.853 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | matrix_edit_watercolor | success | 85740.1 | 304949.1 | incl-load+encode | 1689.2 | 660.9 | 16040 | 8824 | 16040 | 1516 | 0.033 | 0.692 | 0.368 | 5.92 | -1.835 | 4.55 | 0.541 | 0.854 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | qwen_edit_studio_background | success | 86646.5 | 221763.2 | incl-load+encode | 1625.9 | 730.2 | 16040 | 8818 | 16040 | 1516 | -0.018 | 0.693 | 0.369 | 5.97 | -1.249 | 23.50 | 0.851 | 0.229 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(3)** | 86610.2 | 304349.8 |  | 1645.2 | 671.8 | 16040 | 8822 | 16040 | 1516 | -0.017 | 0.693 | 0.369 | 5.96 | -1.112 | 10.84 | 0.643 | 0.645 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 131930.0 | 153718.1 | net-inference | 16160.0 | 3620.0 | 16958 | 13988 | 16958 | 1302 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 129120.0 | 162053.7 | net-inference | 25560.0 | 5020.0 | 17008 | 13988 | 17008 | 1302 | 0.018 | 0.533 | 0.837 | 4.30 | -0.985 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 120120.0 | 143672.9 | net-inference | 17880.0 | 3870.0 | 17008 | 13988 | 17008 | 1302 | 0.036 | 0.533 | 0.837 | 4.30 | -1.051 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 127056.7 | 153148.3 |  | 19866.7 | 4170.0 | 16991 | 13988 | 16991 | 1302 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 14068 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 14068 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 14068 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 189180.0 | 269032.5 | net-inference | 76030.0 | 2360.0 | 17776 | 6162 | 17774 | 17776 | -0.119 | 0.720 | 0.382 | 5.45 | -0.293 | 4.65 | 0.536 | 0.819 |
| stable-diffusion.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 232490.0 | 347318.7 | net-inference | 110950.0 | 2300.0 | 17776 | 6162 | 17774 | 17776 | -0.062 | 0.717 | 0.388 | 5.42 | -1.831 | 4.65 | 0.536 | 0.820 |
| stable-diffusion.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 205440.0 | 275538.9 | net-inference | 66310.0 | 2300.0 | 17776 | 6162 | 17774 | 17776 | -0.166 | 0.727 | 0.377 | 5.40 | -1.388 | 4.63 | 0.538 | 0.822 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 209036.7 | 297296.7 |  | 84430.0 | 2320.0 | 17776 | 6162 | 17774 | 17776 | -0.115 | 0.721 | 0.382 | 5.42 | -1.171 | 4.64 | 0.537 | 0.820 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 94900.0 | 121393.7 | net-inference | 20750.0 | 3710.0 | 17816 | 7654 | 17816 | 1192 | 0.050 | 0.736 | 0.347 | 5.62 | -0.228 | 4.54 | 0.538 | 0.817 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 53390.0 | 73152.3 | net-inference | 14910.0 | 3090.0 | 17816 | 7654 | 17816 | 1192 | 0.058 | 0.732 | 0.358 | 5.67 | -2.000 | 4.52 | 0.535 | 0.825 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 57390.0 | 78618.2 | net-inference | 16330.0 | 3140.0 | 17816 | 7654 | 17816 | 1192 | -0.017 | 0.741 | 0.343 | 5.61 | -1.166 | 4.53 | 0.536 | 0.823 |
| **stable-diffusion.cpp** | **q8_0** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 68560.0 | 91054.7 |  | 17330.0 | 3313.3 | 17816 | 7654 | 17816 | 1192 | 0.030 | 0.736 | 0.349 | 5.63 | -1.131 | 4.53 | 0.536 | 0.822 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 7746 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 7786 | — | — | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 7746 | — | — | — | — | — | — | — | — | — | — | — |

## qwen-image-edit-lightning-image-editing  (image-editing)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | dir CLIP | keep SSIM | keep LPIPS | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_kontext_brushed_metal | success | 32272.7 | 476304.0 | incl-load+encode | 2384.9 | 1385.4 | 19088 | 15378 | 19088 | 1694 | 0.004 | 0.533 | 0.837 | 4.30 | -0.780 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_edit_watercolor | success | 37730.9 | 366888.3 | incl-load+encode | 2797.3 | 1996.9 | 19088 | 15378 | 19088 | 1694 | 0.018 | 0.533 | 0.837 | 4.30 | -0.985 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | qwen_edit_studio_background | success | 32278.5 | 361516.4 | incl-load+encode | 2502.8 | 1069.9 | 19088 | 15378 | 19088 | 1694 | 0.036 | 0.533 | 0.837 | 4.30 | -1.051 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 34094.1 | 401569.6 |  | 2561.7 | 1484.1 | 19088 | 15378 | 19088 | 1694 | 0.019 | 0.533 | 0.837 | 4.30 | -0.939 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | flux_kontext_brushed_metal | success | 12162.8 | 417371.5 | incl-load+encode | 1531.3 | 493.0 | 18666 | 18666 | 13174 | 12276 | -0.148 | 0.618 | 0.479 | 5.70 | -0.047 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | matrix_edit_watercolor | success | 13385.0 | 339179.3 | incl-load+encode | 2012.0 | 920.6 | 18666 | 18666 | 13176 | 12122 | 0.014 | 0.455 | 0.723 | 6.95 | -0.390 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload (max-vram 20g) (auto-fit) | none | qwen_edit_studio_background | success | 12125.4 | 382343.1 | incl-load+encode | 1477.8 | 511.8 | 18666 | 18666 | 13174 | 12276 | -0.125 | 0.721 | 0.359 | 5.59 | -1.296 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 12557.7 | 379631.3 |  | 1673.7 | 641.8 | 18666 | 18666 | 13175 | 12225 | -0.086 | 0.598 | 0.520 | 6.08 | -0.578 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_kontext_brushed_metal | success | 12117.9 | 521833.8 | incl-load+encode | 1593.4 | 544.9 | 19296 | 18628 | 19296 | 18398 | -0.139 | 0.728 | 0.350 | 5.51 | -0.328 | 4.47 | 0.523 | 0.849 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_edit_watercolor | success | 11948.4 | 428858.1 | incl-load+encode | 1195.5 | 496.2 | 19296 | 18628 | 19296 | 18398 | -0.018 | 0.739 | 0.336 | 5.57 | -1.909 | 4.62 | 0.532 | 0.852 |
| edge-dit.cpp | q4_k | no-offload | none | qwen_edit_studio_background | success | 11999.1 | 405149.6 | incl-load+encode | 1161.2 | 714.1 | 19296 | 18628 | 19296 | 18398 | -0.141 | 0.729 | 0.348 | 5.55 | -1.276 | 23.40 | 0.904 | 0.089 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 12021.8 | 451947.2 |  | 1316.7 | 585.1 | 19296 | 18628 | 19296 | 18398 | -0.099 | 0.732 | 0.345 | 5.54 | -1.171 | 10.83 | 0.653 | 0.597 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_kontext_brushed_metal | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_edit_watercolor | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | qwen_edit_studio_background | failed | — | — | process-level co | — | — | 21390 | — | — | — | — | — | — | — | — | — | — | — |