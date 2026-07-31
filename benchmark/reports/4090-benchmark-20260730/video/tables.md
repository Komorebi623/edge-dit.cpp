# Cross-system comparison matrix (all metrics, one-shot aggregate)

57 runs total | success 42 | failed 15

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | matrix_video_bicycle_rain | success | 3386.2 | 4527.6 | net-inference | 282.2 | 817.9 | 20578 | 14564 | 14964 | 20578 | 0.218 | 4.71 | 0.134 | 0.390 | 0.029 | — | — | — |
| diffusers | bf16 | no-offload | none | matrix_video_robot_plaza | success | 3335.8 | 4428.1 | net-inference | 297.2 | 766.1 | 20578 | 14790 | 14964 | 20578 | 0.220 | 4.63 | 0.104 | 0.632 | 0.024 | — | — | — |
| diffusers | bf16 | no-offload | none | wan_teapot_rotation | success | 3401.7 | 4424.3 | net-inference | 277.3 | 719.4 | 20578 | 14466 | 14964 | 20578 | 0.205 | 4.75 | 0.109 | 0.720 | 0.022 | — | — | — |
| **diffusers** | **bf16** | **no-offload** | **none** | **mean** | **(3)** | 3374.6 | 4460.0 |  | 285.6 | 767.8 | 20578 | 14607 | 14964 | 20578 | 0.214 | 4.70 | 0.116 | 0.580 | 0.025 | — | — | — |
| diffusers | fp8 | no-offload | none | matrix_video_bicycle_rain | success | 4321.7 | 5435.3 | net-inference | 320.8 | 722.9 | 19628 | 13318 | 13574 | 19628 | 0.219 | 4.66 | 0.141 | 0.404 | 0.035 | 24.75 | 0.500 | 0.101 |
| diffusers | fp8 | no-offload | none | matrix_video_robot_plaza | success | 4259.7 | 5221.8 | net-inference | 234.2 | 711.1 | 19628 | 13126 | 13574 | 19628 | 0.227 | 4.52 | 0.110 | 0.637 | 0.027 | 29.61 | 0.673 | 0.077 |
| diffusers | fp8 | no-offload | none | wan_teapot_rotation | success | 4324.1 | 5396.5 | net-inference | 270.9 | 772.0 | 19628 | — | 13574 | 19628 | 0.205 | 4.70 | 0.107 | 0.726 | 0.017 | 32.99 | 0.798 | 0.068 |
| **diffusers** | **fp8** | **no-offload** | **none** | **mean** | **(3)** | 4301.8 | 5351.2 |  | 275.3 | 735.3 | 19628 | 13222 | 13574 | 19628 | 0.217 | 4.63 | 0.119 | 0.589 | 0.026 | 29.11 | 0.657 | 0.082 |
| edge-dit.cpp | f16 | no-offload | none | matrix_video_bicycle_rain | success | 5727.4 | 67378.2 | incl-load+encode | 463.3 | 1466.9 | 17100 | 16512 | 16556 | 17100 | 0.308 | 5.36 | 0.137 | 0.407 | 0.040 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_video_robot_plaza | success | 5886.9 | 83740.6 | incl-load+encode | 427.9 | 1321.8 | 17100 | 16214 | 16556 | 17100 | 0.292 | 5.10 | 0.032 | 0.950 | 0.019 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | wan_teapot_rotation | success | 5593.3 | 19592.7 | incl-load+encode | 376.0 | 1879.7 | 17100 | 16214 | 16556 | 17100 | 0.309 | 4.79 | 0.045 | 0.941 | 0.020 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 5735.9 | 56903.8 |  | 422.4 | 1556.2 | 17100 | 16313 | 16556 | 17100 | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | f16 | te offload | none | matrix_video_bicycle_rain | success | 5662.7 | 34777.1 | incl-load+encode | 2990.7 | 1329.5 | 16536 | 16536 | 3740 | 4284 | 0.308 | 5.36 | 0.137 | 0.407 | 0.040 | — | — | — |
| edge-dit.cpp | f16 | te offload | none | matrix_video_robot_plaza | success | 5781.6 | 107198.8 | incl-load+encode | 3089.2 | 1604.9 | 16536 | 16536 | 3740 | 4284 | 0.292 | 5.10 | 0.032 | 0.950 | 0.019 | — | — | — |
| edge-dit.cpp | f16 | te offload | none | wan_teapot_rotation | success | 7720.4 | 31009.3 | incl-load+encode | 2245.1 | 1377.3 | 16536 | 16536 | 3740 | 4284 | 0.309 | 4.79 | 0.045 | 0.941 | 0.020 | — | — | — |
| **edge-dit.cpp** | **f16** | **te offload** | **none** | **mean** | **(3)** | 6388.2 | 57661.8 |  | 2775.0 | 1437.2 | 16536 | 16536 | 3740 | 4284 | 0.303 | 5.08 | 0.072 | 0.766 | 0.026 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_bicycle_rain | success | 6227.2 | 114927.4 | incl-load+encode | 334.5 | 1615.1 | 8826 | 8272 | 8282 | 8826 | 0.314 | 5.34 | 0.119 | 0.372 | 0.034 | 20.40 | 0.553 | 0.224 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_robot_plaza | success | 6795.1 | 88630.5 | incl-load+encode | 390.2 | 1715.6 | 8826 | 8272 | 8282 | 8826 | 0.286 | 4.87 | 0.041 | 0.939 | 0.024 | 29.10 | 0.927 | 0.173 |
| edge-dit.cpp | q4_k | no-offload | none | wan_teapot_rotation | success | 6254.5 | 88349.3 | incl-load+encode | 342.0 | 1317.6 | 8826 | 8272 | 8282 | 8826 | 0.313 | 4.91 | 0.049 | 0.937 | 0.021 | 26.54 | 0.923 | 0.203 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 6425.6 | 97302.4 |  | 355.6 | 1549.4 | 8826 | 8272 | 8282 | 8826 | 0.305 | 5.04 | 0.070 | 0.749 | 0.026 | 25.35 | 0.801 | 0.200 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_bicycle_rain | success | 6120.0 | 26786.0 | incl-load+encode | 339.8 | 1349.3 | 11704 | 11150 | 11160 | 11704 | 0.309 | 5.41 | 0.136 | 0.401 | 0.041 | 28.23 | 0.850 | 0.029 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_robot_plaza | success | 6816.8 | 31715.2 | incl-load+encode | 476.3 | 3691.7 | 11704 | 11142 | 11160 | 11704 | 0.330 | 5.19 | 0.037 | 0.953 | 0.022 | 28.45 | 0.930 | 0.112 |
| edge-dit.cpp | q8_0 | no-offload | none | wan_teapot_rotation | success | 6262.2 | 20087.7 | incl-load+encode | 333.8 | 1987.8 | 11704 | 11150 | 11160 | 11704 | 0.299 | 4.87 | 0.047 | 0.940 | 0.019 | 36.12 | 0.956 | 0.053 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 6399.6 | 26196.3 |  | 383.3 | 2342.9 | 11704 | 11147 | 11160 | 11704 | 0.313 | 5.16 | 0.073 | 0.765 | 0.027 | 30.93 | 0.912 | 0.065 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | matrix_video_bicycle_rain | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | matrix_video_robot_plaza | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | no-offload | none | wan_teapot_rotation | failed | — | — | process-level co | — | — | 24062 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | matrix_video_bicycle_rain | success | 135720.4 | 143166.4 | net-inference | 5438.9 | 1878.8 | 5544 | 2418 | 1378 | 5544 | 0.340 | 4.97 | 0.076 | 0.755 | 0.035 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | matrix_video_robot_plaza | success | 150270.2 | 162777.6 | net-inference | 6526.6 | 5763.2 | 4558 | 2418 | 1378 | 4520 | 0.264 | 4.51 | 0.131 | 0.538 | 0.061 | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | wan_teapot_rotation | success | 150360.5 | 171248.6 | net-inference | 4933.9 | 1912.4 | 5706 | 2418 | 1412 | 5706 | 0.286 | 4.71 | 0.112 | 0.451 | 0.054 | — | — | — |
| **diffusers** | **bf16** | **sequential (full offload)** | **none** | **mean** | **(3)** | 145450.4 | 159064.2 |  | 5633.1 | 3184.8 | 5269 | 2418 | 1389 | 5257 | 0.296 | 4.73 | 0.106 | 0.581 | 0.050 | — | — | — |
| diffusers | fp8 | no-offload | none | matrix_video_bicycle_rain | failed | — | — | process-level co | — | — | 24076 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | matrix_video_robot_plaza | failed | — | — | process-level co | — | — | 24076 | — | — | — | — | — | — | — | — | — | — | — |
| diffusers | fp8 | no-offload | none | wan_teapot_rotation | failed | — | — | process-level co | — | — | 24076 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_video_bicycle_rain | success | 106478.6 | 538029.5 | incl-load+encode | 1802.1 | 2337.7 | 19038 | 13844 | 19038 | 2004 | 0.205 | 3.71 | 0.032 | 0.978 | 0.018 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | matrix_video_robot_plaza | success | 111036.0 | 577462.0 | incl-load+encode | 1886.1 | 1808.1 | 19038 | 13844 | 19038 | 2004 | 0.193 | 4.10 | 0.016 | 0.934 | 0.026 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | wan_teapot_rotation | success | 116356.4 | 408504.4 | incl-load+encode | 1831.3 | 1745.5 | 19038 | 13844 | 19038 | 2004 | 0.151 | 4.63 | 0.016 | 0.979 | 0.020 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(3)** | 111290.3 | 507998.7 |  | 1839.9 | 1963.8 | 19038 | 13844 | 19038 | 2004 | 0.183 | 4.15 | 0.022 | 0.963 | 0.021 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_video_bicycle_rain | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_video_robot_plaza | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | wan_teapot_rotation | failed | — | — | process-level co | — | — | 13392 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_video_bicycle_rain | failed | — | — | process-level co | — | — | 436 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | matrix_video_robot_plaza | failed | — | — | process-level co | — | — | 412 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16 | te offload (max-vram 20g) | none | wan_teapot_rotation | failed | — | — | process-level co | — | — | 576 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | matrix_video_bicycle_rain | success | 36640.8 | 308732.2 | incl-load+encode | 2313.7 | 1836.2 | 19326 | 19326 | 16084 | 16398 | 0.339 | 5.58 | 0.021 | 0.957 | 0.014 | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | matrix_video_robot_plaza | success | 36640.5 | 279092.3 | incl-load+encode | 1517.5 | 1592.4 | 19396 | 19396 | 16084 | 16398 | 0.289 | 5.87 | 0.016 | 0.980 | 0.020 | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | wan_teapot_rotation | success | 36698.9 | 364495.1 | incl-load+encode | 1551.1 | 1692.1 | 19396 | 19396 | 16084 | 16398 | 0.312 | 5.21 | 0.073 | 0.944 | 0.042 | — | — | — |
| **edge-dit.cpp** | **f16->q8_0(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(3)** | 36660.1 | 317439.9 |  | 1794.1 | 1706.9 | 19373 | 19373 | 16084 | 16398 | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_bicycle_rain | success | 37252.7 | 702161.2 | incl-load+encode | 342.7 | 1312.1 | 16092 | 15506 | 15918 | 16092 | 0.356 | 5.98 | 0.025 | 0.947 | 0.009 | 14.92 | 0.525 | 0.292 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_robot_plaza | success | 37271.4 | 896501.4 | incl-load+encode | 345.6 | 1408.6 | 16092 | 15514 | 15918 | 16092 | 0.293 | 5.90 | 0.048 | 0.952 | 0.035 | 8.27 | 0.245 | 0.954 |
| edge-dit.cpp | q4_k | no-offload | none | wan_teapot_rotation | success | 37448.7 | 941175.8 | incl-load+encode | 491.0 | 1483.7 | 16092 | 15514 | 15918 | 16092 | 0.331 | 4.99 | 0.066 | 0.946 | 0.052 | 9.18 | 0.404 | 1.036 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 37324.2 | 846612.8 |  | 393.1 | 1401.5 | 16092 | 15511 | 15918 | 16092 | 0.327 | 5.62 | 0.046 | 0.949 | 0.032 | 10.79 | 0.392 | 0.761 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_bicycle_rain | failed | — | — | process-level co | — | — | 24052 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_robot_plaza | failed | — | — | process-level co | — | — | 24052 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | no-offload | none | wan_teapot_rotation | failed | — | — | process-level co | — | — | 24052 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | matrix_video_bicycle_rain | success | 36667.0 | 148599.6 | incl-load+encode | 1980.8 | 1846.1 | 19396 | 19396 | 16084 | 16398 | 0.339 | 5.58 | 0.021 | 0.957 | 0.014 | 100.00 | 1.000 | 0.000 |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | matrix_video_robot_plaza | success | 36762.9 | 206160.1 | incl-load+encode | 1509.7 | 1652.9 | 19396 | 19396 | 16084 | 16398 | 0.289 | 5.87 | 0.016 | 0.980 | 0.020 | 8.28 | 0.208 | 0.976 |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | wan_teapot_rotation | success | 36575.4 | 228681.3 | incl-load+encode | 1622.1 | 1802.9 | 19396 | 19396 | 16084 | 16398 | 0.312 | 5.21 | 0.073 | 0.944 | 0.042 | 9.37 | 0.383 | 1.045 |
| **edge-dit.cpp** | **q8_0** | **te offload + vae offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(3)** | 36668.4 | 194480.3 |  | 1704.2 | 1767.3 | 19396 | 19396 | 16084 | 16398 | 0.314 | 5.55 | 0.036 | 0.960 | 0.025 | 39.21 | 0.531 | 0.674 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | matrix_video_bicycle_rain | success | 1439.5 | 22256.5 | incl-load+encode | 324.5 | 1946.2 | 17084 | 16484 | 16540 | 17084 | 0.318 | 5.30 | 0.063 | 0.696 | 0.035 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | matrix_video_robot_plaza | success | 1563.5 | 15915.1 | incl-load+encode | 404.5 | 1925.2 | 17084 | 16484 | 16540 | 17084 | 0.297 | 5.55 | 0.030 | 0.935 | 0.023 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | wan_teapot_rotation | success | 1286.1 | 41686.0 | incl-load+encode | 275.0 | 1287.6 | 17084 | 16490 | 16540 | 17084 | 0.325 | 5.41 | 0.038 | 0.947 | 0.023 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(3)** | 1429.7 | 26619.2 |  | 334.7 | 1719.7 | 17084 | 16486 | 16540 | 17084 | 0.314 | 5.42 | 0.044 | 0.859 | 0.027 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_bicycle_rain | success | 1906.8 | 105141.7 | incl-load+encode | 304.7 | 2630.2 | 8810 | 8242 | 8266 | 8810 | 0.322 | 5.24 | 0.048 | 0.764 | 0.030 | 15.54 | 0.254 | 0.261 |
| edge-dit.cpp | q4_k | no-offload | none | matrix_video_robot_plaza | success | 1601.9 | 110814.5 | incl-load+encode | 269.2 | 1272.6 | 8810 | 8242 | 8266 | 8810 | 0.299 | 5.55 | 0.026 | 0.946 | 0.020 | 15.99 | 0.632 | 0.410 |
| edge-dit.cpp | q4_k | no-offload | none | wan_teapot_rotation | success | 1477.5 | 125862.0 | incl-load+encode | 220.1 | 1324.7 | 8810 | 7952 | 8266 | 8810 | 0.336 | 5.43 | 0.036 | 0.945 | 0.021 | 22.88 | 0.825 | 0.190 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(3)** | 1662.1 | 113939.4 |  | 264.7 | 1742.5 | 8810 | 8145 | 8266 | 8810 | 0.319 | 5.41 | 0.037 | 0.885 | 0.024 | 18.14 | 0.571 | 0.287 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_bicycle_rain | success | 1639.2 | 23678.3 | incl-load+encode | 262.5 | 2464.7 | 11688 | — | 11144 | 11688 | 0.315 | 5.24 | 0.065 | 0.686 | 0.037 | 24.43 | 0.804 | 0.061 |
| edge-dit.cpp | q8_0 | no-offload | none | matrix_video_robot_plaza | success | 1605.9 | 25414.7 | incl-load+encode | 353.0 | 1543.7 | 11688 | 11128 | 11144 | 11688 | 0.303 | 5.39 | 0.032 | 0.900 | 0.023 | 13.21 | 0.548 | 0.506 |
| edge-dit.cpp | q8_0 | no-offload | none | wan_teapot_rotation | success | 1697.8 | 29631.1 | incl-load+encode | 269.6 | 2160.1 | 11688 | 11120 | 11144 | 11688 | 0.323 | 5.43 | 0.038 | 0.947 | 0.022 | 39.60 | 0.982 | 0.009 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(3)** | 1647.6 | 26241.4 |  | 295.0 | 2056.2 | 11688 | 11124 | 11144 | 11688 | 0.314 | 5.35 | 0.045 | 0.845 | 0.027 | 25.75 | 0.778 | 0.192 |