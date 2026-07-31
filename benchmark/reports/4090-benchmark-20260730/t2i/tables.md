# Cross-system comparison matrix (all metrics, one-shot aggregate)

174 runs total | success 120 | failed 54

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | flux_glass_teapot | success | 229954.1 | 270245.3 | net-inference | 21719.4 | 18352.9 | 24027 | 9653 | 24027 | 23879 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | full offload | none | sd35_glass_teapot | success | 51646.7 | 136651.5 | net-inference | 37807.2 | 46563.2 | 24027 | 9621 | 24027 | 23879 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | full offload | none | sd3_glass_teapot | success | 182751.6 | 212064.0 | net-inference | 11964.4 | 17135.6 | 24027 | 9621 | 24027 | 23879 | 0.306 | 6.06 | 1.861 | — | — | — |
| **diffusers** | **bf16** | **full offload** | **none** | **mean** | **(3)** | 154784.2 | 206320.3 |  | 23830.3 | 27350.5 | 24027 | 9632 | 24027 | 23879 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | flux_glass_teapot | success | 266484.6 | 277245.2 | net-inference | 9473.7 | 1238.8 | 1553 | 739 | 1287 | 1553 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | sd35_glass_teapot | success | 135988.9 | 139409.1 | net-inference | 2538.0 | 775.0 | 2323 | 739 | 1289 | 2323 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | sd3_glass_teapot | success | 139924.8 | 144023.4 | net-inference | 2683.6 | 1345.6 | 1811 | 739 | 1289 | 1811 | 0.306 | 6.06 | 1.861 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 180799.4 | 186892.6 |  | 4898.4 | 1119.8 | 1896 | 739 | 1288 | 1896 | 0.306 | 6.06 | 1.861 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_glass_teapot | success | 15883.9 | 17168.5 | net-inference | 311.0 | 914.3 | 23779 | 21771 | 22897 | 23403 | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| diffusers | fp8 | no-offload | none | sd35_glass_teapot | success | 15877.5 | 16929.4 | net-inference | 273.2 | 729.7 | 23779 | 21713 | 22897 | 23779 | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| diffusers | fp8 | no-offload | none | sd3_glass_teapot | success | 15879.6 | 16899.4 | net-inference | 336.9 | 637.8 | 23915 | 21717 | 22897 | 23915 | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 15880.3 | 16999.1 |  | 307.0 | 760.6 | 23824 | 21734 | 22897 | 23699 | 0.316 | 6.04 | 1.871 | 24.26 | 0.893 | 0.120 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 43866.7 | 46427.0 | net-inference | 1731.6 | 805.2 | 19513 | 9901 | 19513 | 1313 | 0.303 | 6.04 | 1.808 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 40888.4 | 42940.0 | net-inference | 1279.5 | 758.4 | 19513 | 9903 | 19513 | 1315 | 0.302 | 6.12 | 1.789 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 48661.0 | 50496.0 | net-inference | 1144.2 | 680.0 | 19513 | 9907 | 19513 | 1313 | 0.303 | 6.04 | 1.793 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 44472.0 | 46621.0 |  | 1385.1 | 747.9 | 19513 | 9904 | 19513 | 1314 | 0.303 | 6.07 | 1.797 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 23535 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 23535 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 23535 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 23575 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 23575 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 23573 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 10827.6 | 11609.0 | net-inference | 279.2 | 495.3 | 11177 | 10241 | 11177 | 10681 | 0.299 | 5.98 | 1.841 | 19.84 | 0.808 | 0.234 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 10897.1 | 11663.0 | net-inference | 258.8 | 500.9 | 11177 | 10247 | 11177 | 10681 | 0.299 | 5.98 | 1.816 | 20.07 | 0.815 | 0.222 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 10780.1 | 11545.0 | net-inference | 279.3 | 479.8 | 11177 | 10241 | 11177 | 10681 | 0.298 | 5.99 | 1.847 | 20.51 | 0.823 | 0.211 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 10835.0 | 11605.7 |  | 272.4 | 492.0 | 11177 | 10243 | 11177 | 10681 | 0.299 | 5.98 | 1.834 | 20.14 | 0.815 | 0.223 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 11067.2 | 12254.0 | net-inference | 688.7 | 492.5 | 19067 | 18137 | 19067 | 18571 | 0.281 | 5.95 | 1.765 | 16.42 | 0.715 | 0.358 |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 10871.9 | 11996.0 | net-inference | 515.7 | 599.3 | 19067 | 18137 | 19067 | 18571 | 0.281 | 5.95 | 1.765 | 16.33 | 0.710 | 0.366 |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 11451.6 | 12867.0 | net-inference | 631.2 | 773.4 | 19067 | 18137 | 19067 | 18571 | 0.280 | 5.90 | 1.758 | 16.29 | 0.710 | 0.366 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 11130.2 | 12372.3 |  | 611.8 | 621.8 | 19067 | 18137 | 19067 | 18571 | 0.281 | 5.93 | 1.762 | 16.35 | 0.712 | 0.363 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 88210.0 | 98694.3 | net-inference | 8860.0 | 1610.0 | 17495 | 9535 | 17495 | 1257 | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 49780.0 | 59327.3 | net-inference | 8220.0 | 1310.0 | 17495 | 9535 | 17495 | 1257 | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 53290.0 | 62871.8 | net-inference | 8290.0 | 1280.0 | 17547 | 9535 | 17547 | 1257 | 0.304 | 5.83 | 1.649 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 63760.0 | 73631.1 |  | 8456.7 | 1400.0 | 17512 | 9535 | 17512 | 1257 | 0.304 | 5.83 | 1.649 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 9831 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 9831 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 9907 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 90650.0 | 128764.7 | net-inference | 36900.0 | 1190.0 | 11003 | 3661 | 11003 | 10737 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 105330.0 | 149774.9 | net-inference | 43190.0 | 1210.0 | 11003 | 3661 | 11003 | 10737 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 105600.0 | 141378.9 | net-inference | 34380.0 | 1380.0 | 11003 | 3661 | 11003 | 10737 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 100526.7 | 139972.8 |  | 38156.7 | 1260.0 | 11003 | 3661 | 11003 | 10737 | 0.303 | 5.72 | 1.611 | 25.94 | 0.942 | 0.066 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 20780.0 | 26496.1 | net-inference | 4340.0 | 1360.0 | 18525 | 5541 | 18525 | 18259 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 20750.0 | 26889.8 | net-inference | 4920.0 | 1200.0 | 18525 | 5541 | 18525 | 18259 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 19690.0 | 26181.9 | net-inference | 5260.0 | 1200.0 | 18525 | 5541 | 18525 | 18259 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 20406.7 | 26522.6 |  | 4840.0 | 1253.3 | 18525 | 5541 | 18525 | 18259 | 0.301 | 5.90 | 1.677 | 26.46 | 0.954 | 0.052 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 24023 | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | flux_glass_teapot | success | 3203.3 | 4018.0 | net-inference | 318.3 | 477.2 | 23757 | 21751 | 22875 | 23517 | 0.298 | 5.64 | 1.766 | — | — | — |
| diffusers | fp8 | no-offload | none | sd35_glass_teapot | success | 3202.5 | 4044.0 | net-inference | 278.9 | 539.5 | 23757 | 21679 | 22875 | 23005 | 0.298 | 5.64 | 1.766 | — | — | — |
| diffusers | fp8 | no-offload | none | sd3_glass_teapot | success | 3191.1 | 3944.2 | net-inference | 269.0 | 467.1 | 23757 | 21717 | 22875 | 23381 | 0.298 | 5.64 | 1.766 | — | — | — |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 3199.0 | 4002.1 |  | 288.7 | 494.6 | 23757 | 21716 | 22875 | 23301 | 0.298 | 5.64 | 1.766 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 9057.9 | 10759.0 | net-inference | 971.7 | 702.8 | 19491 | 9901 | 19491 | 1313 | 0.317 | 5.47 | 1.880 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 9807.4 | 11989.0 | net-inference | 946.1 | 1214.1 | 19491 | 9901 | 19491 | 1313 | 0.316 | 5.46 | 1.880 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 9928.4 | 12130.0 | net-inference | 1416.7 | 771.5 | 19491 | 9901 | 19491 | 1313 | 0.316 | 5.46 | 1.883 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 9597.9 | 11626.0 |  | 1111.5 | 896.2 | 19491 | 9901 | 19491 | 1313 | 0.317 | 5.46 | 1.881 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 23515 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 23515 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 23515 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 23555 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 23555 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 23555 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 2322.5 | 3109.0 | net-inference | 288.3 | 492.1 | 11157 | 10227 | 11157 | 10661 | 0.310 | 5.72 | 1.812 | 14.71 | 0.693 | 0.299 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 2317.4 | 3064.0 | net-inference | 266.4 | 474.3 | 11166 | 10218 | 11166 | 10670 | 0.309 | 5.78 | 1.804 | 14.69 | 0.692 | 0.298 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 2338.7 | 3080.0 | net-inference | 236.7 | 498.6 | 11157 | 10227 | 11157 | 10661 | 0.310 | 5.72 | 1.812 | 14.70 | 0.693 | 0.298 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 2326.2 | 3084.3 |  | 263.8 | 488.3 | 11160 | 10224 | 11160 | 10664 | 0.310 | 5.74 | 1.809 | 14.70 | 0.693 | 0.299 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 2295.8 | 3014.0 | net-inference | 228.6 | 483.4 | 19047 | 18117 | 19047 | 18551 | 0.312 | 5.58 | 1.886 | 28.59 | 0.944 | 0.036 |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 2297.0 | 3044.0 | net-inference | 261.7 | 479.2 | 19047 | 18099 | 19047 | 18551 | 0.312 | 5.58 | 1.886 | 28.52 | 0.944 | 0.036 |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 2300.7 | 3086.0 | net-inference | 268.8 | 510.5 | 19047 | 18117 | 19047 | 18551 | 0.312 | 5.58 | 1.883 | 28.46 | 0.943 | 0.036 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 2297.8 | 3048.0 |  | 253.1 | 491.1 | 19047 | 18111 | 19047 | 18551 | 0.312 | 5.58 | 1.885 | 28.52 | 0.944 | 0.036 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 95020.0 | 123005.1 | net-inference | 26260.0 | 1700.0 | 17359 | 9535 | 17359 | 1257 | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 31170.0 | 41479.4 | net-inference | 8810.0 | 1480.0 | 17475 | 9535 | 17475 | 1257 | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 43720.0 | 64959.4 | net-inference | 19670.0 | 1540.0 | 17377 | 9535 | 17377 | 1257 | 0.307 | 5.49 | 1.818 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 56636.7 | 76481.3 |  | 18246.7 | 1573.3 | 17404 | 9535 | 17404 | 1257 | 0.307 | 5.49 | 1.818 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 9831 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 9837 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 9907 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 62290.0 | 98343.2 | net-inference | 34850.0 | 1180.0 | 10983 | 3661 | 10983 | 10717 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 67050.0 | 102299.0 | net-inference | 34030.0 | 1200.0 | 10983 | 3661 | 10983 | 10717 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 60190.0 | 97956.2 | net-inference | 36560.0 | 1180.0 | 10983 | 3661 | 10983 | 10717 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 63176.7 | 99532.8 |  | 35146.7 | 1186.7 | 10983 | 3661 | 10983 | 10717 | 0.317 | 5.53 | 1.675 | 12.09 | 0.581 | 0.518 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 8720.0 | 16151.5 | net-inference | 6210.0 | 1210.0 | 18505 | 5541 | 18505 | 18239 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 9460.0 | 15462.0 | net-inference | 4790.0 | 1190.0 | 18505 | 5541 | 18505 | 18239 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 9070.0 | 15065.5 | net-inference | 4760.0 | 1220.0 | 18505 | 5541 | 18505 | 18239 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 9083.3 | 15559.6 |  | 5253.3 | 1206.7 | 18505 | 5541 | 18505 | 18239 | 0.313 | 5.47 | 1.818 | 21.95 | 0.891 | 0.075 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 15787.9 | 18613.0 | net-inference | 1596.5 | 1216.5 | 19391 | 14035 | 19391 | 1325 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 12653.4 | 14552.0 | net-inference | 993.9 | 895.8 | 19391 | 14031 | 19391 | 1325 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 15764.8 | 18941.0 | net-inference | 1591.1 | 1561.2 | 19391 | 14033 | 19391 | 1325 | 0.174 | 4.30 | -1.144 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 14735.4 | 17368.7 |  | 1393.9 | 1224.5 | 19391 | 14033 | 19391 | 1325 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 417 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | flux_glass_teapot | success | 2510.2 | 3913.0 | net-inference | 578.5 | 818.4 | 18663 | 18663 | 12203 | 11929 | 0.328 | 5.49 | 1.843 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | sd35_glass_teapot | success | 4489.4 | 6075.0 | net-inference | 747.2 | 832.5 | 18607 | 18607 | 12151 | 12069 | 0.327 | 5.51 | 1.836 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | sd3_glass_teapot | success | 2516.5 | 3894.0 | net-inference | 594.3 | 777.3 | 18663 | 18663 | 12203 | 12069 | 0.327 | 5.48 | 1.847 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 3172.1 | 4627.3 |  | 640.0 | 809.4 | 18644 | 18644 | 12186 | 12022 | 0.327 | 5.49 | 1.842 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 2501.5 | 3279.0 | net-inference | 230.7 | 540.1 | 17911 | 17195 | 17911 | 17911 | 0.330 | 5.55 | 1.857 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 2604.1 | 3501.0 | net-inference | 263.2 | 626.7 | 17911 | 17205 | 17911 | 17479 | 0.334 | 5.51 | 1.851 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 2623.8 | 3476.0 | net-inference | 278.5 | 567.9 | 17911 | 17205 | 17911 | 17637 | 0.334 | 5.52 | 1.849 | 2.80 | 0.347 | 0.934 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 2576.5 | 3418.7 |  | 257.5 | 578.2 | 17911 | 17202 | 17911 | 17676 | 0.333 | 5.53 | 1.853 | 2.80 | 0.347 | 0.934 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 21287 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 21289 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 21289 | — | — | — | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 24058 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 24058 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 24058 | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | flux_glass_teapot | success | 616589.2 | 662890.9 | net-inference | 41684.9 | 4501.6 | 4572 | 1592 | 1002 | 4572 | 0.314 | 5.53 | 1.850 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | sd35_glass_teapot | success | 292351.8 | 326198.5 | net-inference | 32472.9 | 1295.7 | 4572 | 1580 | 1022 | 3396 | 0.314 | 5.53 | 1.850 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | sd3_glass_teapot | success | 261143.8 | 279057.0 | net-inference | 5960.5 | 11868.9 | 4572 | 1580 | 1024 | 3400 | 0.314 | 5.53 | 1.850 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 390028.3 | 422715.5 |  | 26706.1 | 5888.7 | 4572 | 1584 | 1016 | 3789 | 0.314 | 5.53 | 1.850 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 24022 | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 24022 | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 24022 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 107782.6 | 111771.0 | net-inference | 2741.4 | 1233.2 | 19400 | 14042 | 19400 | 1334 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 108943.3 | 113071.0 | net-inference | 2997.0 | 1122.1 | 19400 | 14040 | 19400 | 1334 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 96422.1 | 98344.0 | net-inference | 1083.7 | 832.5 | 19400 | 14040 | 19400 | 1334 | 0.174 | 4.30 | -1.144 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 104382.7 | 107728.7 |  | 2274.0 | 1062.6 | 19400 | 14041 | 19400 | 1334 | 0.174 | 4.30 | -1.144 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 426 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | flux_glass_teapot | success | 17615.4 | 19019.0 | net-inference | 607.5 | 790.4 | 18616 | 18616 | 12212 | 12078 | 0.329 | 5.27 | 1.865 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | sd35_glass_teapot | success | 17635.9 | 19087.0 | net-inference | 632.9 | 811.3 | 18672 | 18672 | 12212 | 12078 | 0.327 | 5.27 | 1.863 | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | sd3_glass_teapot | success | 17563.9 | 18993.0 | net-inference | 636.1 | 787.1 | 18616 | 18616 | 12212 | 12078 | 0.326 | 5.24 | 1.865 | — | — | — |
| **edge-dit.cpp** | **f16->q4_k(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 17605.1 | 19033.0 |  | 625.5 | 796.3 | 18635 | 18635 | 12212 | 12078 | 0.327 | 5.26 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 17813.3 | 19252.0 | net-inference | 358.3 | 1069.5 | 17920 | 17204 | 17920 | 17646 | 0.333 | 5.29 | 1.857 | 2.62 | 0.355 | 0.932 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 17674.5 | 18454.0 | net-inference | 229.8 | 543.8 | 17920 | 17214 | 17920 | 17646 | 0.331 | 5.29 | 1.865 | 2.62 | 0.353 | 0.933 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 17671.2 | 18607.0 | net-inference | 204.7 | 721.6 | 17920 | 17202 | 17920 | 17646 | 0.329 | 5.26 | 1.858 | 2.62 | 0.355 | 0.931 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 17719.6 | 18771.0 |  | 264.3 | 778.3 | 17920 | 17207 | 17920 | 17646 | 0.331 | 5.28 | 1.860 | 2.62 | 0.354 | 0.932 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | flux_glass_teapot | success | 68810.4 | 70159.0 | net-inference | 785.5 | 558.1 | 16816 | 7856 | 16816 | 1222 | 0.312 | 5.58 | 1.879 | 2.35 | 0.329 | 0.929 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | sd35_glass_teapot | success | 61227.9 | 68142.0 | net-inference | 5703.4 | 1193.6 | 16818 | 7802 | 16818 | 1222 | 0.312 | 5.58 | 1.881 | 2.35 | 0.329 | 0.930 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | sd3_glass_teapot | success | 63095.4 | 64522.0 | net-inference | 657.5 | 763.3 | 16816 | 7800 | 16816 | 1222 | 0.312 | 5.61 | 1.880 | 2.35 | 0.329 | 0.929 |
| **edge-dit.cpp** | **q8_0** | **DiT offload + te offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(3)** | 64377.9 | 67607.7 |  | 2382.1 | 838.3 | 16817 | 7819 | 16817 | 1222 | 0.312 | 5.59 | 1.880 | 2.35 | 0.329 | 0.929 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 21300 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 21300 | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 21300 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 185060.0 | 228895.8 | net-inference | 38320.0 | 5490.0 | 16924 | 13964 | 16924 | 1316 | 0.174 | 4.30 | -1.144 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 141780.0 | 165661.1 | net-inference | 20670.0 | 3200.0 | 16974 | 13964 | 16974 | 1316 | 0.174 | 4.30 | -1.144 | — | — | — |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 166280.0 | 187799.1 | net-inference | 17560.0 | 3940.0 | 16924 | 13954 | 16924 | 1316 | 0.174 | 4.30 | -1.144 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 164373.3 | 194118.7 |  | 25516.7 | 4210.0 | 16941 | 13961 | 16941 | 1316 | 0.174 | 4.30 | -1.144 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 13942 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 13936 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 13936 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 203320.0 | 280236.9 | net-inference | 74500.0 | 2390.0 | 17688 | 6066 | 17682 | 17688 | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 228940.0 | 298681.2 | net-inference | 66100.0 | 3620.0 | 17688 | 6022 | 17682 | 17688 | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 202910.0 | 273073.1 | net-inference | 67670.0 | 2480.0 | 17688 | 6068 | 17682 | 17688 | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 211723.3 | 283997.0 |  | 69423.3 | 2830.0 | 17688 | 6052 | 17682 | 17688 | 0.317 | 5.38 | 1.851 | 2.55 | 0.351 | 0.934 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | flux_glass_teapot | success | 97380.0 | 114929.2 | net-inference | 14260.0 | 3280.0 | 17778 | 7672 | 17778 | 1206 | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | sd35_glass_teapot | success | 76340.0 | 97510.0 | net-inference | 17690.0 | 3460.0 | 17828 | 7606 | 17828 | 1206 | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | sd3_glass_teapot | success | 78530.0 | 97085.8 | net-inference | 15420.0 | 3130.0 | 17778 | 7606 | 17778 | 1206 | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| **stable-diffusion.cpp** | **q8_0** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 84083.3 | 103175.0 |  | 15790.0 | 3290.0 | 17795 | 7628 | 17795 | 1206 | 0.320 | 5.97 | 1.877 | 2.29 | 0.324 | 0.919 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_glass_teapot | failed | — | — | process-level co | — | — | 7650 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd35_glass_teapot | failed | — | — | process-level co | — | — | 7652 | — | — | — | — | — | — | — | — | — |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd3_glass_teapot | failed | — | — | process-level co | — | — | 7662 | — | — | — | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | flux_glass_teapot | success | 3110.7 | 3810.1 | net-inference | 328.5 | 346.1 | 20217 | 15769 | 16217 | 16485 | 0.344 | 4.92 | 1.802 | — | — | — |
| diffusers | bf16 | no-offload | none | sd35_glass_teapot | success | 3078.1 | 3756.9 | net-inference | 317.5 | 336.7 | 20217 | 15717 | 16217 | 16217 | 0.344 | 4.92 | 1.802 | — | — | — |
| diffusers | bf16 | no-offload | none | sd3_glass_teapot | success | 3087.3 | 3787.5 | net-inference | 341.0 | 334.5 | 20217 | 15711 | 16217 | 19447 | 0.344 | 4.92 | 1.802 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 3092.0 | 3784.8 |  | 329.0 | 339.1 | 20217 | 15732 | 16217 | 17383 | 0.344 | 4.92 | 1.802 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_glass_teapot | success | 4153.8 | 4769.5 | net-inference | 309.4 | 282.0 | 18431 | 13855 | 14431 | 16123 | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| diffusers | fp8 | no-offload | none | sd35_glass_teapot | success | 4146.3 | 4734.7 | net-inference | 306.1 | 264.7 | 18431 | 13823 | 14431 | 16123 | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| diffusers | fp8 | no-offload | none | sd3_glass_teapot | success | 4151.1 | 4761.6 | net-inference | 305.7 | 286.0 | 18431 | 13795 | 14431 | 18431 | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 4150.4 | 4755.3 |  | 307.1 | 277.6 | 18431 | 13824 | 14431 | 16892 | 0.306 | 5.34 | -2.170 | 15.82 | 0.620 | 0.623 |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | success | 3267.0 | 4092.0 | net-inference | 401.1 | 417.2 | 15873 | 15445 | 15807 | 15873 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | success | 3270.5 | 4076.0 | net-inference | 386.0 | 413.2 | 15873 | 15445 | 15807 | 15873 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | success | 3276.4 | 4064.0 | net-inference | 366.4 | 413.6 | 15873 | 15445 | 15807 | 15873 | 0.329 | 5.54 | 1.831 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 3271.3 | 4077.3 |  | 384.5 | 414.7 | 15873 | 15445 | 15807 | 15873 | 0.329 | 5.54 | 1.831 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 3470.8 | 4252.0 | net-inference | 358.8 | 416.3 | 5757 | 5403 | 5691 | 5757 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 3437.7 | 4236.0 | net-inference | 369.9 | 421.9 | 5757 | 5473 | 5691 | 5757 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 3423.6 | 4220.0 | net-inference | 371.7 | 418.2 | 5757 | 5403 | 5691 | 5757 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 3444.0 | 4236.0 |  | 366.8 | 418.8 | 5757 | 5426 | 5691 | 5757 | 0.339 | 5.15 | 1.855 | 19.80 | 0.776 | 0.291 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 3449.7 | 4226.0 | net-inference | 355.5 | 414.3 | 9263 | 8909 | 9197 | 9263 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 3502.9 | 4285.0 | net-inference | 356.5 | 418.9 | 9263 | 8903 | 9197 | 9263 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 3457.2 | 4232.0 | net-inference | 353.6 | 415.0 | 9263 | 8903 | 9197 | 9263 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 3469.9 | 4247.7 |  | 355.2 | 416.1 | 9263 | 8905 | 9197 | 9263 | 0.322 | 5.55 | 1.795 | 25.34 | 0.922 | 0.072 |
| stable-diffusion.cpp | f16 | no-offload | none | flux_glass_teapot | success | 5800.0 | 9364.6 | net-inference | 2260.0 | 1290.0 | 15959 | 11231 | 15647 | 15959 | 0.208 | 3.40 | -2.275 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd35_glass_teapot | success | 5300.0 | 8550.2 | net-inference | 2120.0 | 1110.0 | 15959 | 11245 | 15647 | 15959 | 0.208 | 3.40 | -2.275 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | sd3_glass_teapot | success | 5270.0 | 8529.9 | net-inference | 2130.0 | 1110.0 | 15959 | 11233 | 15647 | 15959 | 0.208 | 3.40 | -2.275 | — | — | — |
| **stable-diffusion.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 5456.7 | 8814.9 |  | 2170.0 | 1170.0 | 15959 | 11236 | 15647 | 15959 | 0.208 | 3.40 | -2.275 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 16670.0 | 58757.8 | net-inference | 40890.0 | 1170.0 | 6103 | 4199 | 5789 | 6103 | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 16580.0 | 59613.7 | net-inference | 41840.0 | 1170.0 | 6103 | 4199 | 5789 | 6103 | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| stable-diffusion.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 16410.0 | 59793.4 | net-inference | 42180.0 | 1180.0 | 6103 | 4199 | 5789 | 6103 | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| **stable-diffusion.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 16553.3 | 59388.3 |  | 41636.7 | 1173.3 | 6103 | 4199 | 5789 | 6103 | 0.186 | 3.13 | -2.278 | 19.61 | 0.915 | 0.102 |
| stable-diffusion.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 5300.0 | 11571.9 | net-inference | 5120.0 | 1130.0 | 9241 | 6377 | 8927 | 9241 | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 5340.0 | 11752.0 | net-inference | 5260.0 | 1130.0 | 9241 | 6389 | 8927 | 9241 | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |
| stable-diffusion.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 5310.0 | 11575.2 | net-inference | 5120.0 | 1120.0 | 9241 | 6377 | 8927 | 9241 | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |
| **stable-diffusion.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 5316.7 | 11633.1 |  | 5166.7 | 1126.7 | 9241 | 6381 | 8927 | 9241 | 0.202 | 3.38 | -2.281 | 29.34 | 0.989 | 0.018 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | flux_glass_teapot | success | 548.9 | 1210.4 | net-inference | 362.2 | 283.3 | 20757 | 16337 | 16629 | 16629 | 0.308 | 5.12 | -2.272 | — | — | — |
| diffusers | bf16 | no-offload | none | sd35_glass_teapot | success | 528.8 | 1418.1 | net-inference | 520.2 | 332.6 | 20757 | 16337 | 16629 | 20757 | 0.308 | 5.12 | -2.272 | — | — | — |
| diffusers | bf16 | no-offload | none | sd3_glass_teapot | success | 566.9 | 1203.6 | net-inference | 315.9 | 301.3 | 20757 | 16345 | 16629 | 18449 | 0.308 | 5.12 | -2.272 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 548.2 | 1277.4 |  | 399.5 | 305.7 | 20757 | 16340 | 16629 | 18612 | 0.308 | 5.12 | -2.272 | — | — | — |
| diffusers | fp8 | no-offload | none | flux_glass_teapot | success | 737.2 | 1324.1 | net-inference | 279.0 | 292.5 | 18723 | 14227 | 14595 | 18723 | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| diffusers | fp8 | no-offload | none | sd35_glass_teapot | success | 715.5 | 1267.4 | net-inference | 267.2 | 266.3 | 18723 | 14253 | 14595 | 15503 | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| diffusers | fp8 | no-offload | none | sd3_glass_teapot | success | 737.0 | 1330.3 | net-inference | 295.8 | 279.3 | 18723 | 14285 | 14595 | 14595 | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 729.9 | 1307.3 |  | 280.7 | 279.4 | 18723 | 14255 | 14595 | 16274 | 0.287 | 4.75 | -2.281 | 33.39 | 0.852 | 0.063 |
| edge-dit.cpp | f16 | no-offload | none | flux_glass_teapot | success | 489.6 | 1310.0 | net-inference | 354.9 | 457.8 | 16817 | 16359 | 16697 | 16817 | 0.335 | 5.21 | 1.644 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd35_glass_teapot | success | 447.8 | 1214.0 | net-inference | 331.2 | 429.1 | 16817 | 16359 | 16697 | 16817 | 0.335 | 5.21 | 1.644 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | sd3_glass_teapot | success | 2291.7 | 17309.0 | net-inference | 14531.3 | 479.4 | 16817 | 16357 | 16749 | 16817 | 0.335 | 5.20 | 1.635 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 1076.4 | 6611.0 |  | 5072.5 | 455.4 | 16817 | 16358 | 16714 | 16817 | 0.335 | 5.21 | 1.641 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | flux_glass_teapot | success | 520.5 | 1387.0 | net-inference | 284.5 | 570.6 | 6407 | 6029 | 6287 | 6407 | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.180 |
| edge-dit.cpp | q4_k | no-offload | none | sd35_glass_teapot | success | 562.9 | 1596.0 | net-inference | 442.2 | 576.0 | 6355 | 6029 | 6287 | 6355 | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.180 |
| edge-dit.cpp | q4_k | no-offload | none | sd3_glass_teapot | success | 697.5 | 1832.0 | net-inference | 522.4 | 597.6 | 6355 | 6099 | 6287 | 6355 | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.179 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 593.6 | 1605.0 |  | 416.4 | 581.4 | 6372 | 6052 | 6287 | 6372 | 0.315 | 5.41 | 1.119 | 24.38 | 0.872 | 0.179 |
| edge-dit.cpp | q8_0 | no-offload | none | flux_glass_teapot | success | 530.8 | 1286.0 | net-inference | 302.1 | 445.4 | 10015 | 9637 | 9923 | 10015 | 0.333 | 5.26 | 1.655 | 37.59 | 0.989 | 0.017 |
| edge-dit.cpp | q8_0 | no-offload | none | sd35_glass_teapot | success | 526.1 | 1255.0 | net-inference | 301.3 | 421.9 | 10015 | 9637 | 9895 | 10015 | 0.333 | 5.27 | 1.654 | 37.66 | 0.990 | 0.016 |
| edge-dit.cpp | q8_0 | no-offload | none | sd3_glass_teapot | success | 523.2 | 1267.0 | net-inference | 313.3 | 423.9 | 10015 | 9637 | 9895 | 10015 | 0.333 | 5.27 | 1.654 | 37.66 | 0.989 | 0.016 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 526.7 | 1269.3 |  | 305.6 | 430.4 | 10015 | 9637 | 9904 | 10015 | 0.333 | 5.27 | 1.654 | 37.64 | 0.989 | 0.016 |