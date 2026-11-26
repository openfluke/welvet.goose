# Welvet dense benchmarks in Goose, C++ and Rust

Measured on **Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz**, one thread pinned to logical CPU 2. Every timing uses a fresh run on this machine. Smaller values are better. Percentages are runtime changes relative to each reference, calculated as `100 × (Goose time / reference time − 1)`. For example, half the time is 50% faster and twice the time is 100% slower.

Goose owns the tensor layout, loop traversal, training and update logic. The Goose v2026.3 compiler (commit `8511ae34`) emits C, which GCC builds with `-O3 -march=native -ffp-contract=off`. Fixed-size inline C primitives provide AVX2 arithmetic, widening and packed decoding. Goose keeps reduction order, handles tails and determines where those primitives are applied. C++ and Rust also use explicit SIMD in their tuned kernels.

- `gcc (GCC) 16.2.1 20261029 (Red Hat 16.2.1-3)`
- `clang version 22.1.8 (Fedora 22.1.8-4.fc44)` for C++ training calibration.
- `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Welvet Go v1.0.3 generated the snapshots using the seed and shapes in the existing manifest.
- C++ storage uses the repository's release AVX2/FMA/F16C build. Native C++ builds use `-O3 -march=native -ffp-contract=off`. Rust uses native CPU features, release optimization and LTO.
- [Goose v2026.3 compiler source](https://github.com/aardappel/goose/releases/tag/v2026.3).
- In each timing round the runner shuffles dtype and language order. Long fat passes use 15 measured iterations and 3 warmups per round. Small gaps can shift between rounds.
- The local raw files include samples, fixture hashes and training histories. [Storage](build/storage-raw.json), [native](build/native-raw.json), [kernel modes](build/tuned-raw.json), [regression](build/regression-raw.json), [learning rates](build/learning-raw.json).

## Results at a glance

Summary ratios use the geometric mean of the per-cell median latency ratios.

- **Storage forward (micro).** **0.83× C++ latency**, with Goose faster in 27/34 cells and **0.88× Rust latency**, with Goose faster in 27/34 cells.
- **Storage forward (deep).** **0.84× C++ latency**, with Goose faster in 23/34 cells and **1.00× Rust latency**, with Goose faster in 18/34 cells.
- **Storage forward (fat).** **0.83× C++ latency**, with Goose faster in 30/34 cells and **0.94× Rust latency**, with Goose faster in 25/34 cells.
- **Against the faster reference in each storage case.** Goose matched or came within 10% in 89/102 cells, including 62 wins. The geometric-mean latency was 0.98×.
- **Native forward.** **1.34× C++ latency**, with Goose faster in 1/12 cells and **1.29× Rust latency**, with Goose faster in 1/12 cells.
- **Same-dtype forward/backward/update.** **1.76× C++ latency**, with Goose faster in 5/36 cells and **1.95× Rust latency**, with Goose faster in 4/36 cells.
- **Widened-gradient backward/update.** **1.16× C++ latency**, with Goose faster in 5/24 cells and **1.66× Rust latency**, with Goose faster in 7/24 cells.
- **Supervised training across learning rates and buffer settings.** **1.69× C++ latency**, with Goose faster in 18/192 cells and **1.70× Rust latency**, with Goose faster in 0/192 cells.

## Storage-dtype forward

102 dtype/profile cells. For every cell, Goose, C++ and Rust produced the same output hash. Welvet's wire conversion and quantization rules apply here, including float32 layer outputs. The Goose port keeps the timed conversion for packed and affine weights and builds the same float64 wire caches before timing.

Time is the median across per-round medians, in µs per full stack pass. Throughput is the nominal dense operation count `2 × batch × Σ(input × output)` divided by elapsed seconds. Its speed ratios are the inverse of the latency ratios.

### Micro

Batch 8, input 256, output 128, 1 layers. 3 rounds of 200 timed passes after 20 warmups.

| Dtype | Goose (µs) | C++ (µs) | Rust (µs) | Goose vs C++ | Goose vs Rust | Goose GFLOP/s | C++ GFLOP/s | Rust GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float64 | 28.094 | 30.111 | 29.298 | 6.7% faster | 4.1% faster | 18.66 | 17.41 | 17.90 |
| float32 | 24.274 | 24.005 | 26.250 | 1.1% slower | 7.5% faster | 21.60 | 21.84 | 19.97 |
| float16 | 49.240 | 70.159 | 55.538 | 29.8% faster | 11.3% faster | 10.65 | 7.47 | 9.44 |
| bfloat16 | 49.875 | 70.607 | 55.824 | 29.4% faster | 10.7% faster | 10.51 | 7.43 | 9.39 |
| fp8e4m3 | 54.717 | 74.061 | 60.377 | 26.1% faster | 9.4% faster | 9.58 | 7.08 | 8.68 |
| fp8e5m2 | 55.241 | 74.703 | 61.482 | 26.1% faster | 10.2% faster | 9.49 | 7.02 | 8.53 |
| int64 | 28.356 | 30.099 | 30.027 | 5.8% faster | 5.6% faster | 18.49 | 17.42 | 17.46 |
| int32 | 29.064 | 30.824 | 28.879 | 5.7% faster | 0.6% slower | 18.04 | 17.01 | 18.15 |
| int16 | 29.704 | 30.330 | 29.249 | 2.1% faster | 1.6% slower | 17.65 | 17.29 | 17.92 |
| int8 | 20.450 | 20.508 | 18.966 | 0.3% faster | 7.8% slower | 25.64 | 25.57 | 27.64 |
| uint64 | 70.094 | 50.302 | 71.911 | 39.3% slower | 2.5% faster | 7.48 | 10.42 | 7.29 |
| uint32 | 35.912 | 32.470 | 41.426 | 10.6% slower | 13.3% faster | 14.60 | 16.15 | 12.66 |
| uint16 | 30.247 | 30.076 | 39.006 | 0.6% slower | 22.5% faster | 17.33 | 17.43 | 13.44 |
| uint8 | 31.580 | 33.066 | 39.476 | 4.5% faster | 20.0% faster | 16.60 | 15.86 | 13.28 |
| int4 | 24.073 | 25.187 | 24.450 | 4.4% faster | 1.5% faster | 21.78 | 20.82 | 21.44 |
| uint4 | 29.588 | 32.344 | 40.806 | 8.5% faster | 27.5% faster | 17.72 | 16.21 | 12.85 |
| fp4 | 51.688 | 70.789 | 56.605 | 27.0% faster | 8.7% faster | 10.14 | 7.41 | 9.26 |
| int2 | 25.444 | 26.369 | 25.301 | 3.5% faster | 0.6% slower | 20.61 | 19.88 | 20.72 |
| uint2 | 29.899 | 33.423 | 44.662 | 10.5% faster | 33.1% faster | 17.54 | 15.69 | 11.74 |
| ternary | 25.437 | 26.292 | 25.372 | 3.3% faster | 0.3% slower | 20.61 | 19.94 | 20.66 |
| binary | 25.656 | 26.276 | 25.126 | 2.4% faster | 2.1% slower | 20.44 | 19.95 | 20.87 |
| int | 37.749 | 30.318 | 31.074 | 24.5% slower | 21.5% slower | 13.89 | 17.29 | 16.87 |
| uint | 70.093 | 50.641 | 71.358 | 38.4% slower | 1.8% faster | 7.48 | 10.35 | 7.35 |
| uintptr | 69.840 | 50.807 | 70.943 | 37.5% slower | 1.6% faster | 7.51 | 10.32 | 7.39 |
| complex64 | 28.414 | 31.168 | 31.225 | 8.8% faster | 9.0% faster | 18.45 | 16.82 | 16.79 |
| complex128 | 27.321 | 29.150 | 30.449 | 6.3% faster | 10.3% faster | 19.19 | 17.99 | 17.22 |
| nf4 | 30.260 | 31.311 | 39.595 | 3.4% faster | 23.6% faster | 17.33 | 16.74 | 13.24 |
| fp6 | 32.787 | 65.726 | 43.633 | 50.1% faster | 24.9% faster | 15.99 | 7.98 | 12.02 |
| int6 | 32.462 | 65.767 | 43.089 | 50.6% faster | 24.7% faster | 16.15 | 7.97 | 12.17 |
| uint6 | 31.872 | 67.799 | 42.962 | 53.0% faster | 25.8% faster | 16.45 | 7.73 | 12.20 |
| int5 | 32.350 | 66.210 | 43.027 | 51.1% faster | 24.8% faster | 16.21 | 7.92 | 12.19 |
| uint5 | 33.829 | 68.333 | 42.938 | 50.5% faster | 21.2% faster | 15.50 | 7.67 | 12.21 |
| int3 | 31.012 | 65.461 | 42.745 | 52.6% faster | 27.4% faster | 16.91 | 8.01 | 12.27 |
| uint3 | 31.250 | 67.617 | 43.503 | 53.8% faster | 28.2% faster | 16.78 | 7.75 | 12.05 |

### Deep

Batch 8, input 256, output 256, 8 layers. 3 rounds of 100 timed passes after 20 warmups.

| Dtype | Goose (µs) | C++ (µs) | Rust (µs) | Goose vs C++ | Goose vs Rust | Goose GFLOP/s | C++ GFLOP/s | Rust GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float64 | 572.893 | 580.566 | 578.939 | 1.3% faster | 1.0% faster | 14.64 | 14.45 | 14.49 |
| float32 | 439.781 | 419.817 | 447.724 | 4.8% slower | 1.8% faster | 19.07 | 19.98 | 18.74 |
| float16 | 836.936 | 1,179.159 | 789.474 | 29.0% faster | 6.0% slower | 10.02 | 7.11 | 10.63 |
| bfloat16 | 859.091 | 1,188.833 | 815.969 | 27.7% faster | 5.3% slower | 9.76 | 7.06 | 10.28 |
| fp8e4m3 | 919.876 | 1,212.436 | 846.723 | 24.1% faster | 8.6% slower | 9.12 | 6.92 | 9.91 |
| fp8e5m2 | 953.005 | 1,238.955 | 877.993 | 23.1% faster | 8.5% slower | 8.80 | 6.77 | 9.55 |
| int64 | 596.012 | 573.779 | 570.380 | 3.9% slower | 4.5% slower | 14.07 | 14.62 | 14.71 |
| int32 | 572.876 | 572.457 | 552.572 | 0.1% slower | 3.7% slower | 14.64 | 14.65 | 15.18 |
| int16 | 570.301 | 598.114 | 582.251 | 4.7% faster | 2.1% faster | 14.71 | 14.03 | 14.41 |
| int8 | 252.260 | 268.494 | 272.422 | 6.0% faster | 7.4% faster | 33.25 | 31.24 | 30.79 |
| uint64 | 1,193.786 | 918.232 | 1,085.468 | 30.0% slower | 10.0% slower | 7.03 | 9.14 | 7.73 |
| uint32 | 618.351 | 560.232 | 575.468 | 10.4% slower | 7.5% slower | 13.57 | 14.97 | 14.58 |
| uint16 | 525.687 | 518.280 | 522.980 | 1.4% slower | 0.5% slower | 15.96 | 16.19 | 16.04 |
| uint8 | 538.526 | 534.466 | 488.929 | 0.8% slower | 10.1% slower | 15.58 | 15.70 | 17.16 |
| int4 | 297.378 | 332.350 | 360.113 | 10.5% faster | 17.4% faster | 28.21 | 25.24 | 23.29 |
| uint4 | 510.248 | 545.212 | 558.964 | 6.4% faster | 8.7% faster | 16.44 | 15.39 | 15.01 |
| fp4 | 860.422 | 1,168.849 | 786.677 | 26.4% faster | 9.4% slower | 9.75 | 7.18 | 10.66 |
| int2 | 327.645 | 361.797 | 373.516 | 9.4% faster | 12.3% faster | 25.60 | 23.19 | 22.46 |
| uint2 | 520.596 | 545.896 | 582.358 | 4.6% faster | 10.6% faster | 16.11 | 15.37 | 14.40 |
| ternary | 330.731 | 358.766 | 374.825 | 7.8% faster | 11.8% faster | 25.36 | 23.38 | 22.38 |
| binary | 325.865 | 356.910 | 368.262 | 8.7% faster | 11.5% faster | 25.74 | 23.50 | 22.78 |
| int | 583.216 | 627.240 | 571.692 | 7.0% faster | 2.0% slower | 14.38 | 13.37 | 14.67 |
| uint | 1,235.943 | 899.464 | 1,105.278 | 37.4% slower | 11.8% slower | 6.79 | 9.33 | 7.59 |
| uintptr | 1,188.854 | 900.166 | 1,082.625 | 32.1% slower | 9.8% slower | 7.06 | 9.32 | 7.75 |
| complex64 | 595.596 | 583.439 | 581.195 | 2.1% slower | 2.5% slower | 14.08 | 14.38 | 14.43 |
| complex128 | 612.143 | 594.267 | 568.418 | 3.0% slower | 7.7% slower | 13.70 | 14.12 | 14.76 |
| nf4 | 526.307 | 535.608 | 534.917 | 1.7% faster | 1.6% faster | 15.94 | 15.66 | 15.68 |
| fp6 | 565.596 | 1,094.181 | 574.086 | 48.3% faster | 1.5% faster | 14.83 | 7.67 | 14.61 |
| int6 | 559.133 | 1,076.211 | 571.195 | 48.0% faster | 2.1% faster | 15.00 | 7.79 | 14.69 |
| uint6 | 564.089 | 1,108.381 | 575.551 | 49.1% faster | 2.0% faster | 14.87 | 7.57 | 14.57 |
| int5 | 558.819 | 1,099.291 | 573.286 | 49.2% faster | 2.5% faster | 15.01 | 7.63 | 14.63 |
| uint5 | 557.561 | 1,120.374 | 578.913 | 50.2% faster | 3.7% faster | 15.05 | 7.49 | 14.49 |
| int3 | 544.169 | 1,071.009 | 576.257 | 49.2% faster | 5.6% faster | 15.42 | 7.83 | 14.56 |
| uint3 | 535.996 | 1,105.905 | 584.398 | 51.5% faster | 8.3% faster | 15.65 | 7.59 | 14.35 |

### Fat

Batch 16, input 1024, output 1024, 4 layers. 3 rounds of 15 timed passes after 3 warmups.

| Dtype | Goose (µs) | C++ (µs) | Rust (µs) | Goose vs C++ | Goose vs Rust | Goose GFLOP/s | C++ GFLOP/s | Rust GFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float64 | 15,834.622 | 16,257.343 | 15,539.909 | 2.6% faster | 1.9% slower | 8.48 | 8.26 | 8.64 |
| float32 | 8,166.409 | 9,661.148 | 9,184.265 | 15.5% faster | 11.1% faster | 16.44 | 13.89 | 14.61 |
| float16 | 14,161.745 | 19,456.989 | 14,305.797 | 27.2% faster | 1.0% faster | 9.48 | 6.90 | 9.38 |
| bfloat16 | 14,160.426 | 19,951.913 | 13,390.288 | 29.0% faster | 5.8% slower | 9.48 | 6.73 | 10.02 |
| fp8e4m3 | 14,536.592 | 19,612.319 | 14,231.317 | 25.9% faster | 2.1% slower | 9.23 | 6.84 | 9.43 |
| fp8e5m2 | 14,531.690 | 19,704.600 | 14,830.684 | 26.3% faster | 2.0% faster | 9.24 | 6.81 | 9.05 |
| int64 | 16,259.837 | 20,380.033 | 16,266.827 | 20.2% faster | 0.0% | 8.25 | 6.59 | 8.25 |
| int32 | 15,758.054 | 16,726.518 | 16,506.447 | 5.8% faster | 4.5% faster | 8.52 | 8.02 | 8.13 |
| int16 | 15,936.659 | 15,968.436 | 15,400.453 | 0.2% faster | 3.5% slower | 8.42 | 8.41 | 8.72 |
| int8 | 2,908.307 | 3,501.974 | 3,814.516 | 17.0% faster | 23.8% faster | 46.15 | 38.33 | 35.19 |
| uint64 | 13,918.477 | 11,806.165 | 13,392.294 | 17.9% slower | 3.9% slower | 9.64 | 11.37 | 10.02 |
| uint32 | 9,492.269 | 9,066.661 | 8,669.001 | 4.7% slower | 9.5% slower | 14.14 | 14.80 | 15.48 |
| uint16 | 7,985.630 | 8,168.639 | 8,204.673 | 2.2% faster | 2.7% faster | 16.81 | 16.43 | 16.36 |
| uint8 | 8,178.585 | 8,798.502 | 8,206.022 | 7.0% faster | 0.3% faster | 16.41 | 15.25 | 16.36 |
| int4 | 3,039.457 | 3,803.874 | 4,359.764 | 20.1% faster | 30.3% faster | 44.16 | 35.28 | 30.79 |
| uint4 | 8,144.762 | 8,291.956 | 8,711.343 | 1.8% faster | 6.5% faster | 16.48 | 16.19 | 15.41 |
| fp4 | 13,561.894 | 19,425.876 | 13,698.568 | 30.2% faster | 1.0% faster | 9.90 | 6.91 | 9.80 |
| int2 | 3,284.566 | 4,009.575 | 4,470.204 | 18.1% faster | 26.5% faster | 40.86 | 33.47 | 30.02 |
| uint2 | 7,666.007 | 8,383.160 | 8,503.868 | 8.6% faster | 9.9% faster | 17.51 | 16.01 | 15.78 |
| ternary | 3,297.245 | 4,056.876 | 4,607.658 | 18.7% faster | 28.4% faster | 40.71 | 33.08 | 29.13 |
| binary | 3,295.371 | 4,142.602 | 4,427.933 | 20.5% faster | 25.6% faster | 40.73 | 32.40 | 30.31 |
| int | 15,667.376 | 16,068.494 | 16,115.363 | 2.5% faster | 2.8% faster | 8.57 | 8.35 | 8.33 |
| uint | 13,985.512 | 13,029.049 | 13,629.563 | 7.3% slower | 2.6% slower | 9.60 | 10.30 | 9.85 |
| uintptr | 13,884.891 | 11,721.530 | 14,191.592 | 18.5% slower | 2.2% faster | 9.67 | 11.45 | 9.46 |
| complex64 | 15,491.343 | 16,188.515 | 15,614.714 | 4.3% faster | 0.8% faster | 8.66 | 8.29 | 8.60 |
| complex128 | 15,999.369 | 17,142.039 | 16,449.319 | 6.7% faster | 2.7% faster | 8.39 | 7.83 | 8.16 |
| nf4 | 8,885.382 | 8,984.606 | 8,006.717 | 1.1% faster | 11.0% slower | 15.11 | 14.94 | 16.76 |
| fp6 | 8,866.315 | 13,125.719 | 8,579.342 | 32.5% faster | 3.3% slower | 15.14 | 10.23 | 15.64 |
| int6 | 8,535.531 | 12,593.411 | 9,946.362 | 32.2% faster | 14.2% faster | 15.72 | 10.66 | 13.49 |
| uint6 | 8,126.625 | 13,132.671 | 8,503.955 | 38.1% faster | 4.4% faster | 16.52 | 10.22 | 15.78 |
| int5 | 8,150.328 | 13,292.399 | 9,037.130 | 38.7% faster | 9.8% faster | 16.47 | 10.10 | 14.85 |
| uint5 | 8,237.982 | 13,530.506 | 8,254.213 | 39.1% faster | 0.2% faster | 16.29 | 9.92 | 16.26 |
| int3 | 8,081.437 | 13,985.147 | 8,603.913 | 42.2% faster | 6.1% faster | 16.61 | 9.60 | 15.60 |
| uint3 | 8,184.282 | 13,480.116 | 9,238.870 | 39.3% faster | 11.4% faster | 16.40 | 9.96 | 14.53 |

## Native arithmetic

All native stack benchmarks use batch 8 and eight 256×256 layers. The 12 representations use native float arithmetic, integer wraparound, or packed binary/ternary. Widened-gradient backward follows the older widened intermediate and update contract. All three engines agreed on exact output bytes, including gradients, weights and residuals where applicable. The native forward layer hashes also agreed.

### Native forward

Median of 3 rounds, each with 150 operations after 30 warmups.

| Dtype | Metric | Goose (µs/op) | C++ (µs/op) | Rust (µs/op) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | Forward | 514.664 | 416.017 | 392.323 | 23.7% slower | 31.2% slower |
| float64 | Forward | 831.543 | 810.919 | 786.152 | 2.5% slower | 5.8% slower |
| int8 | Forward | 489.146 | 251.842 | 236.378 | 94.2% slower | 106.9% slower |
| uint8 | Forward | 486.548 | 231.651 | 249.675 | 110.0% slower | 94.9% slower |
| int16 | Forward | 526.782 | 280.920 | 306.484 | 87.5% slower | 71.9% slower |
| uint16 | Forward | 521.726 | 271.922 | 296.995 | 91.9% slower | 75.7% slower |
| int32 | Forward | 772.630 | 544.404 | 544.268 | 41.9% slower | 42.0% slower |
| uint32 | Forward | 795.490 | 603.999 | 598.337 | 31.7% slower | 33.0% slower |
| int64 | Forward | 1,809.382 | 1,564.170 | 1,776.313 | 15.7% slower | 1.9% slower |
| uint64 | Forward | 1,809.501 | 1,584.850 | 1,767.715 | 14.2% slower | 2.4% slower |
| binary | Forward | 117.595 | 164.474 | 204.621 | 28.5% faster | 42.5% faster |
| ternary | Forward | 241.769 | 230.450 | 224.339 | 4.9% slower | 7.8% slower |

### Same-dtype forward/backward/update

Median of 3 rounds, each with 30 operations after 5 warmups.

| Dtype | Metric | Goose (µs/op) | C++ (µs/op) | Rust (µs/op) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | Forward | 503.727 | 386.749 | 345.456 | 30.2% slower | 45.8% slower |
| float32 | Backward | 8,050.160 | 4,360.800 | 3,381.554 | 84.6% slower | 138.1% slower |
| float32 | Update | 695.063 | 1,864.640 | 1,603.104 | 62.7% faster | 56.6% faster |
| float64 | Forward | 779.334 | 717.359 | 747.668 | 8.6% slower | 4.2% slower |
| float64 | Backward | 5,809.090 | 5,916.190 | 4,253.296 | 1.8% faster | 36.6% slower |
| float64 | Update | 1,171.596 | 1,705.660 | 1,254.279 | 31.3% faster | 6.6% faster |
| int8 | Forward | 497.072 | 159.245 | 170.235 | 212.1% slower | 192.0% slower |
| int8 | Backward | 8,851.896 | 4,015.490 | 2,419.745 | 120.4% slower | 265.8% slower |
| int8 | Update | 10,477.813 | 4,942.370 | 4,495.887 | 112.0% slower | 133.1% slower |
| uint8 | Forward | 499.799 | 182.624 | 172.472 | 173.7% slower | 189.8% slower |
| uint8 | Backward | 8,452.189 | 3,674.820 | 2,455.328 | 130.0% slower | 244.2% slower |
| uint8 | Update | 9,703.096 | 4,671.850 | 4,854.405 | 107.7% slower | 99.9% slower |
| int16 | Forward | 522.352 | 195.240 | 247.772 | 167.5% slower | 110.8% slower |
| int16 | Backward | 8,730.530 | 4,471.170 | 2,936.069 | 95.3% slower | 197.4% slower |
| int16 | Update | 10,196.892 | 4,586.490 | 4,174.216 | 122.3% slower | 144.3% slower |
| uint16 | Forward | 528.744 | 233.207 | 279.225 | 126.7% slower | 89.4% slower |
| uint16 | Backward | 8,777.675 | 4,273.260 | 2,847.029 | 105.4% slower | 208.3% slower |
| uint16 | Update | 10,284.174 | 4,154.110 | 4,047.534 | 147.6% slower | 154.1% slower |
| int32 | Forward | 852.846 | 455.245 | 728.522 | 87.3% slower | 17.1% slower |
| int32 | Backward | 9,452.521 | 3,308.860 | 3,802.373 | 185.7% slower | 148.6% slower |
| int32 | Update | 10,319.453 | 5,081.960 | 4,479.325 | 103.1% slower | 130.4% slower |
| uint32 | Forward | 767.493 | 440.434 | 683.213 | 74.3% slower | 12.3% slower |
| uint32 | Backward | 9,428.309 | 3,356.530 | 3,811.998 | 180.9% slower | 147.3% slower |
| uint32 | Update | 10,347.946 | 4,939.640 | 4,586.155 | 109.5% slower | 125.6% slower |
| int64 | Forward | 1,834.868 | 1,516.570 | 2,000.329 | 21.0% slower | 8.3% faster |
| int64 | Backward | 10,980.537 | 5,596.960 | 6,023.249 | 96.2% slower | 82.3% slower |
| int64 | Update | 10,396.850 | 4,596.320 | 4,123.088 | 126.2% slower | 152.2% slower |
| uint64 | Forward | 1,971.899 | 1,596.150 | 2,245.778 | 23.5% slower | 12.2% faster |
| uint64 | Backward | 11,507.211 | 5,940.150 | 6,360.742 | 93.7% slower | 80.9% slower |
| uint64 | Update | 12,075.838 | 1,948.960 | 1,814.502 | 519.6% slower | 565.5% slower |
| binary | Forward | 119.188 | 133.326 | 52.657 | 10.6% faster | 126.3% slower |
| binary | Backward | 2,958.246 | 3,412.070 | 1,810.119 | 13.3% faster | 63.4% slower |
| binary | Update | 10,791.945 | 5,429.570 | 6,274.704 | 98.8% slower | 72.0% slower |
| ternary | Forward | 242.397 | 182.773 | 137.702 | 32.6% slower | 76.0% slower |
| ternary | Backward | 6,258.090 | 5,675.440 | 2,056.902 | 10.3% slower | 204.2% slower |
| ternary | Update | 13,627.070 | 7,284.190 | 7,129.904 | 87.1% slower | 91.1% slower |

### Widened-gradient backward/update

Median of 3 rounds, each with 30 operations after 5 warmups.

| Dtype | Metric | Goose (µs/op) | C++ (µs/op) | Rust (µs/op) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | Backward | 1,144.821 | 1,185.550 | 1,478.217 | 3.4% faster | 22.6% faster |
| float32 | Update | 628.053 | 1,782.380 | 1,765.075 | 64.8% faster | 64.4% faster |
| float64 | Backward | 2,283.078 | 2,365.670 | 2,758.096 | 3.5% faster | 17.2% faster |
| float64 | Update | 620.290 | 1,400.930 | 1,087.025 | 55.7% faster | 42.9% faster |
| int8 | Backward | 7,468.602 | 6,935.180 | 7,143.807 | 7.7% slower | 4.5% slower |
| int8 | Update | 9,182.709 | 5,357.010 | 1,742.562 | 71.4% slower | 427.0% slower |
| uint8 | Backward | 7,030.041 | 6,652.200 | 5,652.608 | 5.7% slower | 24.4% slower |
| uint8 | Update | 9,215.618 | 7,429.610 | 2,571.508 | 24.0% slower | 258.4% slower |
| int16 | Backward | 7,325.547 | 7,078.970 | 7,387.448 | 3.5% slower | 0.8% faster |
| int16 | Update | 9,128.489 | 5,714.810 | 1,732.762 | 59.7% slower | 426.8% slower |
| uint16 | Backward | 7,066.269 | 6,591.960 | 5,023.111 | 7.2% slower | 40.7% slower |
| uint16 | Update | 9,271.690 | 7,238.800 | 2,605.383 | 28.1% slower | 255.9% slower |
| int32 | Backward | 7,817.668 | 6,597.160 | 7,270.261 | 18.5% slower | 7.5% slower |
| int32 | Update | 9,441.625 | 5,411.040 | 1,730.038 | 74.5% slower | 445.7% slower |
| uint32 | Backward | 7,102.027 | 6,622.110 | 6,142.404 | 7.2% slower | 15.6% slower |
| uint32 | Update | 9,502.409 | 7,290.090 | 2,608.055 | 30.3% slower | 264.3% slower |
| int64 | Backward | 7,226.126 | 6,950.070 | 7,316.443 | 4.0% slower | 1.2% faster |
| int64 | Update | 9,318.280 | 4,077.960 | 1,867.756 | 128.5% slower | 398.9% slower |
| uint64 | Backward | 7,418.224 | 7,076.440 | 5,515.990 | 4.8% slower | 34.5% slower |
| uint64 | Update | 9,057.560 | 5,565.680 | 2,322.497 | 62.7% slower | 290.0% slower |
| binary | Backward | 7,062.074 | 6,769.740 | 7,545.507 | 4.3% slower | 6.4% faster |
| binary | Update | 11,325.782 | 5,823.960 | 6,015.700 | 94.5% slower | 88.3% slower |
| ternary | Backward | 8,866.513 | 9,111.680 | 8,562.400 | 2.7% faster | 3.6% slower |
| ternary | Update | 13,144.585 | 6,893.690 | 7,005.959 | 90.7% slower | 87.6% slower |

### Same-dtype kernel modes

All 34 dtype/mode combinations are shown. Goose outputs, gradients and update state matched C++ and Rust within each mode. `tile` is an alias of `dot` for binary and ternary in the reference suite. Each result is the median of 3 rounds of 15 operations after 3 warmups. C++ uses GCC in this table.

| Dtype | Kernel | Metric | Goose (µs/op) | C++ (µs/op) | Rust (µs/op) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | dot | Forward | 518.620 | 366.759 | 345.093 | 41.4% slower | 50.3% slower |
| float32 | dot | Backward | 7,750.439 | 4,092.670 | 3,376.457 | 89.4% slower | 129.5% slower |
| float32 | dot | Update | 699.279 | 1,854.490 | 1,523.784 | 62.3% faster | 54.1% faster |
| float32 | tile | Forward | 503.642 | 264.236 | 308.949 | 90.6% slower | 63.0% slower |
| float32 | tile | Backward | 8,961.685 | 3,213.170 | 2,774.059 | 178.9% slower | 223.1% slower |
| float32 | tile | Update | 641.215 | 1,843.030 | 1,612.522 | 65.2% faster | 60.2% faster |
| float32 | outer | Forward | 1,675.519 | 1,952.500 | 2,052.189 | 14.2% faster | 18.4% faster |
| float32 | outer | Backward | 845.675 | 885.006 | 1,344.375 | 4.4% faster | 37.1% faster |
| float32 | outer | Update | 679.831 | 1,869.370 | 1,545.210 | 63.6% faster | 56.0% faster |
| float64 | dot | Forward | 762.607 | 710.138 | 733.767 | 7.4% slower | 3.9% slower |
| float64 | dot | Backward | 5,725.953 | 5,872.490 | 4,278.592 | 2.5% faster | 33.8% slower |
| float64 | dot | Update | 1,105.026 | 1,550.570 | 1,206.275 | 28.7% faster | 8.4% faster |
| float64 | tile | Forward | 905.309 | 561.723 | 622.631 | 61.2% slower | 45.4% slower |
| float64 | tile | Backward | 8,003.525 | 3,728.940 | 3,184.008 | 114.6% slower | 151.4% slower |
| float64 | tile | Update | 966.550 | 1,552.930 | 1,089.342 | 37.8% faster | 11.3% faster |
| float64 | outer | Forward | 2,407.247 | 2,811.980 | 3,331.914 | 14.4% faster | 27.8% faster |
| float64 | outer | Backward | 1,784.413 | 1,775.550 | 2,712.533 | 0.5% slower | 34.2% faster |
| float64 | outer | Update | 963.032 | 1,489.980 | 1,094.308 | 35.4% faster | 12.0% faster |
| int8 | dot | Forward | 506.624 | 175.339 | 165.372 | 188.9% slower | 206.4% slower |
| int8 | dot | Backward | 8,507.834 | 3,630.530 | 2,367.120 | 134.3% slower | 259.4% slower |
| int8 | dot | Update | 9,656.069 | 4,574.010 | 4,640.098 | 111.1% slower | 108.1% slower |
| int8 | tile | Forward | 8,624.281 | 316.026 | 168.840 | 2629.0% slower | 5008.0% slower |
| int8 | tile | Backward | 20,314.366 | 11,597.000 | 2,807.607 | 75.2% slower | 623.5% slower |
| int8 | tile | Update | 10,041.288 | 4,594.380 | 4,408.890 | 118.6% slower | 127.8% slower |
| int8 | outer | Forward | 1,224.444 | 960.665 | 1,019.757 | 27.5% slower | 20.1% slower |
| int8 | outer | Backward | 993.007 | 335.547 | 451.559 | 195.9% slower | 119.9% slower |
| int8 | outer | Update | 9,595.158 | 4,681.730 | 4,379.994 | 104.9% slower | 119.1% slower |
| uint8 | dot | Forward | 482.832 | 154.252 | 162.159 | 213.0% slower | 197.8% slower |
| uint8 | dot | Backward | 8,363.033 | 3,874.930 | 2,416.479 | 115.8% slower | 246.1% slower |
| uint8 | dot | Update | 10,039.873 | 4,514.280 | 4,817.318 | 122.4% slower | 108.4% slower |
| uint8 | tile | Forward | 8,765.935 | 328.799 | 157.224 | 2566.0% slower | 5475.5% slower |
| uint8 | tile | Backward | 20,112.741 | 11,728.600 | 2,820.639 | 71.5% slower | 613.1% slower |
| uint8 | tile | Update | 9,691.624 | 4,449.660 | 4,881.282 | 117.8% slower | 98.5% slower |
| uint8 | outer | Forward | 1,228.970 | 948.036 | 989.411 | 29.6% slower | 24.2% slower |
| uint8 | outer | Backward | 967.819 | 320.564 | 445.733 | 201.9% slower | 117.1% slower |
| uint8 | outer | Update | 9,657.318 | 4,568.630 | 4,905.952 | 111.4% slower | 96.8% slower |
| int16 | dot | Forward | 508.090 | 192.902 | 197.759 | 163.4% slower | 156.9% slower |
| int16 | dot | Backward | 8,550.364 | 4,274.790 | 2,853.302 | 100.0% slower | 199.7% slower |
| int16 | dot | Update | 10,099.396 | 4,358.130 | 4,023.810 | 131.7% slower | 151.0% slower |
| int16 | tile | Forward | 9,539.243 | 195.156 | 154.630 | 4788.0% slower | 6069.1% slower |
| int16 | tile | Backward | 21,361.499 | 6,445.900 | 2,774.409 | 231.4% slower | 669.9% slower |
| int16 | tile | Update | 9,798.696 | 4,372.220 | 4,082.153 | 124.1% slower | 140.0% slower |
| int16 | outer | Forward | 1,504.021 | 1,329.240 | 1,294.971 | 13.1% slower | 16.1% slower |
| int16 | outer | Backward | 1,031.877 | 474.952 | 673.968 | 117.3% slower | 53.1% slower |
| int16 | outer | Update | 9,885.989 | 4,306.860 | 4,086.728 | 129.5% slower | 141.9% slower |
| uint16 | dot | Forward | 549.806 | 189.917 | 199.359 | 189.5% slower | 175.8% slower |
| uint16 | dot | Backward | 8,709.634 | 4,264.620 | 2,789.753 | 104.2% slower | 212.2% slower |
| uint16 | dot | Update | 9,624.410 | 4,168.690 | 4,040.093 | 130.9% slower | 138.2% slower |
| uint16 | tile | Forward | 9,542.837 | 186.682 | 145.029 | 5011.8% slower | 6480.0% slower |
| uint16 | tile | Backward | 21,345.467 | 6,628.810 | 2,733.079 | 222.0% slower | 681.0% slower |
| uint16 | tile | Update | 9,820.169 | 4,096.710 | 4,033.691 | 139.7% slower | 143.5% slower |
| uint16 | outer | Forward | 1,500.842 | 1,251.060 | 1,348.667 | 20.0% slower | 11.3% slower |
| uint16 | outer | Backward | 1,071.297 | 441.742 | 659.535 | 142.5% slower | 62.4% slower |
| uint16 | outer | Update | 9,471.348 | 4,030.270 | 4,046.603 | 135.0% slower | 134.1% slower |
| int32 | dot | Forward | 770.426 | 452.493 | 638.114 | 70.3% slower | 20.7% slower |
| int32 | dot | Backward | 9,359.782 | 3,256.540 | 3,440.362 | 187.4% slower | 172.1% slower |
| int32 | dot | Update | 11,066.104 | 4,961.120 | 4,374.325 | 123.1% slower | 153.0% slower |
| int32 | tile | Forward | 9,524.464 | 334.624 | 326.344 | 2746.3% slower | 2818.5% slower |
| int32 | tile | Backward | 22,218.635 | 3,603.530 | 2,512.047 | 516.6% slower | 784.5% slower |
| int32 | tile | Update | 10,102.987 | 5,025.840 | 4,362.050 | 101.0% slower | 131.6% slower |
| int32 | outer | Forward | 2,126.013 | 1,897.400 | 2,111.057 | 12.0% slower | 0.7% slower |
| int32 | outer | Backward | 1,552.535 | 892.086 | 1,455.353 | 74.0% slower | 6.7% slower |
| int32 | outer | Update | 9,895.675 | 4,975.300 | 4,368.871 | 98.9% slower | 126.5% slower |
| uint32 | dot | Forward | 744.222 | 404.007 | 642.494 | 84.2% slower | 15.8% slower |
| uint32 | dot | Backward | 9,274.060 | 3,262.540 | 3,526.075 | 184.3% slower | 163.0% slower |
| uint32 | dot | Update | 9,908.098 | 4,806.660 | 4,358.197 | 106.1% slower | 127.3% slower |
| uint32 | tile | Forward | 9,775.520 | 348.945 | 328.370 | 2701.5% slower | 2877.0% slower |
| uint32 | tile | Backward | 21,643.108 | 3,561.440 | 2,526.101 | 507.7% slower | 756.8% slower |
| uint32 | tile | Update | 10,528.595 | 4,831.650 | 4,341.800 | 117.9% slower | 142.5% slower |
| uint32 | outer | Forward | 2,135.029 | 1,888.840 | 2,155.715 | 13.0% slower | 1.0% faster |
| uint32 | outer | Backward | 1,550.192 | 987.976 | 1,443.578 | 56.9% slower | 7.4% slower |
| uint32 | outer | Update | 9,977.557 | 4,871.860 | 4,544.321 | 104.8% slower | 119.6% slower |
| int64 | dot | Forward | 1,749.850 | 1,364.590 | 1,898.672 | 28.2% slower | 7.8% faster |
| int64 | dot | Backward | 11,173.157 | 5,296.990 | 6,039.615 | 110.9% slower | 85.0% slower |
| int64 | dot | Update | 10,987.397 | 4,224.880 | 4,206.301 | 160.1% slower | 161.2% slower |
| int64 | tile | Forward | 8,863.085 | 1,019.670 | 1,191.475 | 769.2% slower | 643.9% slower |
| int64 | tile | Backward | 19,889.093 | 4,506.130 | 4,279.894 | 341.4% slower | 364.7% slower |
| int64 | tile | Update | 10,107.634 | 4,240.290 | 4,155.394 | 138.4% slower | 143.2% slower |
| int64 | outer | Forward | 3,294.148 | 3,327.760 | 3,580.565 | 1.0% faster | 8.0% faster |
| int64 | outer | Backward | 3,417.148 | 3,011.930 | 3,859.050 | 13.5% slower | 11.5% faster |
| int64 | outer | Update | 10,271.122 | 4,233.970 | 4,097.258 | 142.6% slower | 150.7% slower |
| uint64 | dot | Forward | 1,669.584 | 1,312.850 | 1,844.464 | 27.2% slower | 9.5% faster |
| uint64 | dot | Backward | 10,983.672 | 5,276.530 | 5,890.558 | 108.2% slower | 86.5% slower |
| uint64 | dot | Update | 8,753.654 | 1,620.310 | 1,679.378 | 440.2% slower | 421.2% slower |
| uint64 | tile | Forward | 9,332.103 | 1,139.220 | 1,169.371 | 719.2% slower | 698.0% slower |
| uint64 | tile | Backward | 20,482.550 | 4,602.470 | 4,369.744 | 345.0% slower | 368.7% slower |
| uint64 | tile | Update | 8,694.940 | 1,689.830 | 1,749.987 | 414.5% slower | 396.9% slower |
| uint64 | outer | Forward | 3,258.437 | 3,305.760 | 3,651.425 | 1.4% faster | 10.8% faster |
| uint64 | outer | Backward | 3,410.146 | 3,067.940 | 3,662.681 | 11.2% slower | 6.9% faster |
| uint64 | outer | Update | 8,510.015 | 1,763.180 | 1,660.509 | 382.7% slower | 412.5% slower |
| binary | dot | Forward | 114.804 | 128.654 | 51.603 | 10.8% faster | 122.5% slower |
| binary | dot | Backward | 3,025.982 | 3,434.520 | 1,775.412 | 11.9% faster | 70.4% slower |
| binary | dot | Update | 10,871.234 | 5,291.690 | 6,332.303 | 105.4% slower | 71.7% slower |
| binary | outer | Forward | 755.883 | 566.139 | 729.348 | 33.5% slower | 3.6% slower |
| binary | outer | Backward | 1,308.054 | 778.201 | 924.291 | 68.1% slower | 41.5% slower |
| binary | outer | Update | 10,678.928 | 5,214.940 | 6,279.220 | 104.8% slower | 70.1% slower |
| ternary | dot | Forward | 265.775 | 204.993 | 108.743 | 29.7% slower | 144.4% slower |
| ternary | dot | Backward | 6,195.438 | 5,589.960 | 2,070.218 | 10.8% slower | 199.3% slower |
| ternary | dot | Update | 13,680.742 | 7,301.320 | 7,177.652 | 87.4% slower | 90.6% slower |
| ternary | outer | Forward | 1,309.634 | 1,257.240 | 1,025.314 | 4.2% slower | 27.7% slower |
| ternary | outer | Backward | 1,993.096 | 1,719.810 | 1,374.949 | 15.9% slower | 45.0% slower |
| ternary | outer | Update | 13,888.862 | 7,378.170 | 7,356.704 | 88.2% slower | 88.8% slower |

### Fastest observed same-dtype mode

For each language and phase, the mode shown has the lowest median in the mode table above. This uses the measured sample for both ranking and timing.

| Dtype | Metric | Goose mode | C++ mode | Rust mode | Goose (µs/op) | C++ (µs/op) | Rust (µs/op) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | Forward | tile | tile | tile | 503.642 | 264.236 | 308.949 | 90.6% slower | 63.0% slower |
| float32 | Backward | outer | outer | outer | 845.675 | 885.006 | 1,344.375 | 4.4% faster | 37.1% faster |
| float64 | Forward | dot | tile | tile | 762.607 | 561.723 | 622.631 | 35.8% slower | 22.5% slower |
| float64 | Backward | outer | outer | outer | 1,784.413 | 1,775.550 | 2,712.533 | 0.5% slower | 34.2% faster |
| int8 | Forward | dot | dot | dot | 506.624 | 175.339 | 165.372 | 188.9% slower | 206.4% slower |
| int8 | Backward | outer | outer | outer | 993.007 | 335.547 | 451.559 | 195.9% slower | 119.9% slower |
| uint8 | Forward | dot | dot | tile | 482.832 | 154.252 | 157.224 | 213.0% slower | 207.1% slower |
| uint8 | Backward | outer | outer | outer | 967.819 | 320.564 | 445.733 | 201.9% slower | 117.1% slower |
| int16 | Forward | dot | dot | tile | 508.090 | 192.902 | 154.630 | 163.4% slower | 228.6% slower |
| int16 | Backward | outer | outer | outer | 1,031.877 | 474.952 | 673.968 | 117.3% slower | 53.1% slower |
| uint16 | Forward | dot | tile | tile | 549.806 | 186.682 | 145.029 | 194.5% slower | 279.1% slower |
| uint16 | Backward | outer | outer | outer | 1,071.297 | 441.742 | 659.535 | 142.5% slower | 62.4% slower |
| int32 | Forward | dot | tile | tile | 770.426 | 334.624 | 326.344 | 130.2% slower | 136.1% slower |
| int32 | Backward | outer | outer | outer | 1,552.535 | 892.086 | 1,455.353 | 74.0% slower | 6.7% slower |
| uint32 | Forward | dot | tile | tile | 744.222 | 348.945 | 328.370 | 113.3% slower | 126.6% slower |
| uint32 | Backward | outer | outer | outer | 1,550.192 | 987.976 | 1,443.578 | 56.9% slower | 7.4% slower |
| int64 | Forward | dot | tile | tile | 1,749.850 | 1,019.670 | 1,191.475 | 71.6% slower | 46.9% slower |
| int64 | Backward | outer | outer | outer | 3,417.148 | 3,011.930 | 3,859.050 | 13.5% slower | 11.5% faster |
| uint64 | Forward | dot | tile | tile | 1,669.584 | 1,139.220 | 1,169.371 | 46.6% slower | 42.8% slower |
| uint64 | Backward | outer | outer | outer | 3,410.146 | 3,067.940 | 3,662.681 | 11.2% slower | 6.9% faster |
| binary | Forward | dot | dot | dot | 114.804 | 128.654 | 51.603 | 10.8% faster | 122.5% slower |
| binary | Backward | outer | outer | outer | 1,308.054 | 778.201 | 924.291 | 68.1% slower | 41.5% slower |
| ternary | Forward | dot | dot | dot | 265.775 | 204.993 | 108.743 | 29.7% slower | 144.4% slower |
| ternary | Backward | outer | outer | outer | 1,993.096 | 1,719.810 | 1,374.949 | 15.9% slower | 45.0% slower |

### Widened-gradient supervised regression control

All three engines produced the same losses and buffered state after 1,024 steps on the 4×4 one-layer problem. This control records convergence and final state.

| Dtype | Initial loss | Buffered final loss | Unbuffered final loss |
| --- | --- | --- | --- |
| float32 | 0.5 | 0 | 4.547e-13 |
| float64 | 0.5 | 2.85e-29 | 2.934e-29 |
| int8 | 0.5 | 0 | 0.5 |
| uint8 | 0.5 | 0 | 0.5 |
| int16 | 0.5 | 0 | 0.5 |
| uint16 | 0.5 | 0 | 0.5 |
| int32 | 0.5 | 0 | 0.5 |
| uint32 | 0.5 | 0 | 0.5 |
| int64 | 0.5 | 0 | 0.5 |
| uint64 | 0.5 | 0 | 0.5 |
| binary | 8 | 0 | 8 |
| ternary | 0.5 | 0 | 0.5 |

## Supervised training and learning-rate sweep

16 inputs and outputs, 256 training samples, 128 validation samples and 256 held-out test samples. 32 epochs, eight learning rates, both buffer settings and three initialization seeds per case. The three engines agreed on all checkpoint errors, test errors, first checkpoint reaching 90%, and final weight and residual bytes. Each engine's forward mode was picked using seed 99. C++ calibration also chose between GCC and Clang. Timings below are medians across seeds 11, 22 and 33.

| Dtype | Goose forward | C++ compiler and forward | Rust forward |
| --- | --- | --- | --- |
| float32 | tile | train-gcc / tile | tile |
| float64 | dot | train-gcc / tile | tile |
| int8 | tile | train-clang / dot | dot |
| uint8 | tile | train-clang / dot | tile |
| int16 | tile | train-gcc / dot | tile |
| uint16 | tile | train-gcc / dot | dot |
| int32 | tile | train-gcc / tile | tile |
| uint32 | tile | train-gcc / dot | tile |
| int64 | tile | train-gcc / dot | tile |
| uint64 | tile | train-gcc / dot | tile |
| binary | dot | train-gcc / dot | dot |
| ternary | dot | train-gcc / dot | dot |

### Full learning-rate timing grid

Training time sums the timed training epochs. Wall time also includes checkpoint evaluation and the final test. Both metrics exclude input loading and output serialization. Validation and test accuracy are shared across the three engines and shown as medians across seeds.

#### Buffered

| Dtype | LR | Metric | Goose (ms) | C++ (ms) | Rust (ms) | Goose vs C++ | Goose vs Rust | Val. accuracy | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | 1/1024 | Train | 7.293 | 3.927 | 3.862 | 85.7% slower | 88.8% slower | 78.56% | 76.90% |
| float32 | 1/1024 | Wall | 8.219 | 4.235 | 4.122 | 94.1% slower | 99.4% slower | — | — |
| float32 | 1/256 | Train | 6.832 | 2.509 | 2.872 | 172.4% slower | 137.9% slower | 90.53% | 90.58% |
| float32 | 1/256 | Wall | 7.817 | 2.780 | 3.112 | 181.2% slower | 151.2% slower | — | — |
| float32 | 1/64 | Train | 6.051 | 2.184 | 2.681 | 177.0% slower | 125.7% slower | 90.97% | 91.46% |
| float32 | 1/64 | Wall | 7.036 | 2.490 | 2.929 | 182.6% slower | 140.3% slower | — | — |
| float32 | 1/32 | Train | 6.035 | 2.123 | 2.624 | 184.2% slower | 130.0% slower | 91.31% | 91.72% |
| float32 | 1/32 | Wall | 6.955 | 2.395 | 2.876 | 190.5% slower | 141.8% slower | — | — |
| float32 | 1/8 | Train | 6.615 | 2.128 | 2.620 | 210.9% slower | 152.4% slower | 90.67% | 91.06% |
| float32 | 1/8 | Wall | 7.656 | 2.414 | 2.871 | 217.2% slower | 166.7% slower | — | — |
| float32 | 1/2 | Train | 6.328 | 2.148 | 2.642 | 194.7% slower | 139.5% slower | 90.92% | 91.33% |
| float32 | 1/2 | Wall | 7.277 | 2.411 | 2.891 | 201.8% slower | 151.7% slower | — | — |
| float32 | 1 | Train | 5.906 | 2.148 | 2.675 | 174.9% slower | 120.8% slower | 90.72% | 91.46% |
| float32 | 1 | Wall | 6.869 | 2.445 | 2.935 | 180.9% slower | 134.0% slower | — | — |
| float32 | 2 | Train | 6.050 | 2.249 | 2.877 | 169.0% slower | 110.3% slower | 89.99% | 90.41% |
| float32 | 2 | Wall | 7.031 | 2.521 | 3.140 | 178.9% slower | 123.9% slower | — | — |
| float64 | 1/1024 | Train | 15.948 | 6.189 | 4.650 | 157.7% slower | 243.0% slower | 78.56% | 76.90% |
| float64 | 1/1024 | Wall | 16.704 | 6.621 | 4.870 | 152.3% slower | 243.0% slower | — | — |
| float64 | 1/256 | Train | 7.503 | 2.976 | 3.499 | 152.1% slower | 114.4% slower | 90.53% | 90.58% |
| float64 | 1/256 | Wall | 7.987 | 3.261 | 3.721 | 144.9% slower | 114.7% slower | — | — |
| float64 | 1/64 | Train | 8.014 | 2.719 | 3.227 | 194.7% slower | 148.4% slower | 90.97% | 91.46% |
| float64 | 1/64 | Wall | 8.531 | 3.007 | 3.467 | 183.7% slower | 146.1% slower | — | — |
| float64 | 1/32 | Train | 7.055 | 2.597 | 3.118 | 171.6% slower | 126.3% slower | 91.31% | 91.72% |
| float64 | 1/32 | Wall | 7.513 | 2.903 | 3.347 | 158.9% slower | 124.5% slower | — | — |
| float64 | 1/8 | Train | 7.601 | 2.663 | 3.114 | 185.4% slower | 144.1% slower | 90.67% | 91.06% |
| float64 | 1/8 | Wall | 8.093 | 2.959 | 3.334 | 173.5% slower | 142.8% slower | — | — |
| float64 | 1/2 | Train | 7.241 | 2.726 | 3.503 | 165.6% slower | 106.7% slower | 90.92% | 91.33% |
| float64 | 1/2 | Wall | 7.685 | 3.018 | 3.770 | 154.6% slower | 103.9% slower | — | — |
| float64 | 1 | Train | 7.723 | 2.808 | 3.185 | 175.0% slower | 142.4% slower | 90.72% | 91.46% |
| float64 | 1 | Wall | 8.209 | 3.122 | 3.399 | 162.9% slower | 141.5% slower | — | — |
| float64 | 2 | Train | 7.177 | 2.744 | 3.287 | 161.6% slower | 118.3% slower | 89.99% | 90.41% |
| float64 | 2 | Wall | 7.617 | 3.044 | 3.512 | 150.3% slower | 116.9% slower | — | — |
| int8 | 1/1024 | Train | 24.248 | 21.790 | 17.019 | 11.3% slower | 42.5% slower | 74.02% | 74.19% |
| int8 | 1/1024 | Wall | 25.051 | 22.412 | 17.748 | 11.8% slower | 41.1% slower | — | — |
| int8 | 1/256 | Train | 15.944 | 11.786 | 8.805 | 35.3% slower | 81.1% slower | 100.00% | 100.00% |
| int8 | 1/256 | Wall | 16.775 | 12.428 | 9.500 | 35.0% slower | 76.6% slower | — | — |
| int8 | 1/64 | Train | 10.172 | 5.883 | 4.704 | 72.9% slower | 116.2% slower | 100.00% | 100.00% |
| int8 | 1/64 | Wall | 10.938 | 6.475 | 5.365 | 68.9% slower | 103.9% slower | — | — |
| int8 | 1/32 | Train | 9.779 | 4.906 | 4.001 | 99.3% slower | 144.5% slower | 100.00% | 100.00% |
| int8 | 1/32 | Wall | 10.527 | 5.489 | 4.716 | 91.8% slower | 123.2% slower | — | — |
| int8 | 1/8 | Train | 9.458 | 4.909 | 4.322 | 92.7% slower | 118.8% slower | 99.02% | 98.78% |
| int8 | 1/8 | Wall | 10.189 | 5.494 | 4.994 | 85.5% slower | 104.0% slower | — | — |
| int8 | 1/2 | Train | 12.100 | 8.739 | 6.804 | 38.5% slower | 77.8% slower | 90.62% | 91.46% |
| int8 | 1/2 | Wall | 12.810 | 9.344 | 7.485 | 37.1% slower | 71.2% slower | — | — |
| int8 | 1 | Train | 14.451 | 12.044 | 9.122 | 20.0% slower | 58.4% slower | 71.19% | 71.46% |
| int8 | 1 | Wall | 15.151 | 12.674 | 9.819 | 19.5% slower | 54.3% slower | — | — |
| int8 | 2 | Train | 26.661 | 29.053 | 20.438 | 8.2% faster | 30.4% slower | 50.39% | 50.59% |
| int8 | 2 | Wall | 27.366 | 29.655 | 21.137 | 7.7% faster | 29.5% slower | — | — |
| uint8 | 1/1024 | Train | 23.673 | 20.884 | 15.693 | 13.4% slower | 50.9% slower | 74.02% | 74.19% |
| uint8 | 1/1024 | Wall | 24.370 | 21.471 | 16.396 | 13.5% slower | 48.6% slower | — | — |
| uint8 | 1/256 | Train | 15.884 | 11.654 | 9.361 | 36.3% slower | 69.7% slower | 100.00% | 100.00% |
| uint8 | 1/256 | Wall | 16.623 | 12.253 | 10.092 | 35.7% slower | 64.7% slower | — | — |
| uint8 | 1/64 | Train | 10.607 | 5.905 | 4.840 | 79.6% slower | 119.2% slower | 100.00% | 100.00% |
| uint8 | 1/64 | Wall | 11.314 | 6.506 | 5.525 | 73.9% slower | 104.8% slower | — | — |
| uint8 | 1/32 | Train | 9.771 | 4.871 | 4.155 | 100.6% slower | 135.2% slower | 100.00% | 100.00% |
| uint8 | 1/32 | Wall | 10.475 | 5.450 | 4.828 | 92.2% slower | 117.0% slower | — | — |
| uint8 | 1/8 | Train | 9.657 | 5.014 | 4.084 | 92.6% slower | 136.5% slower | 99.02% | 98.78% |
| uint8 | 1/8 | Wall | 10.364 | 5.616 | 4.856 | 84.6% slower | 113.4% slower | — | — |
| uint8 | 1/2 | Train | 12.768 | 8.738 | 6.929 | 46.1% slower | 84.3% slower | 90.62% | 91.46% |
| uint8 | 1/2 | Wall | 13.546 | 9.348 | 7.659 | 44.9% slower | 76.9% slower | — | — |
| uint8 | 1 | Train | 14.729 | 12.645 | 9.663 | 16.5% slower | 52.4% slower | 71.19% | 71.46% |
| uint8 | 1 | Wall | 15.486 | 13.262 | 10.367 | 16.8% slower | 49.4% slower | — | — |
| uint8 | 2 | Train | 27.773 | 30.038 | 21.478 | 7.5% faster | 29.3% slower | 50.39% | 50.59% |
| uint8 | 2 | Wall | 28.591 | 30.683 | 22.247 | 6.8% faster | 28.5% slower | — | — |
| int16 | 1/1024 | Train | 22.445 | 18.755 | 15.298 | 19.7% slower | 46.7% slower | 74.02% | 74.19% |
| int16 | 1/1024 | Wall | 23.183 | 19.538 | 15.857 | 18.7% slower | 46.2% slower | — | — |
| int16 | 1/256 | Train | 13.783 | 10.371 | 9.165 | 32.9% slower | 50.4% slower | 100.00% | 100.00% |
| int16 | 1/256 | Wall | 14.598 | 11.202 | 9.734 | 30.3% slower | 50.0% slower | — | — |
| int16 | 1/64 | Train | 8.088 | 4.986 | 4.422 | 62.2% slower | 82.9% slower | 100.00% | 100.00% |
| int16 | 1/64 | Wall | 8.829 | 5.766 | 4.965 | 53.1% slower | 77.8% slower | — | — |
| int16 | 1/32 | Train | 7.139 | 4.182 | 3.762 | 70.7% slower | 89.7% slower | 100.00% | 100.00% |
| int16 | 1/32 | Wall | 7.852 | 4.923 | 4.305 | 59.5% slower | 82.4% slower | — | — |
| int16 | 1/8 | Train | 6.972 | 4.044 | 4.193 | 72.4% slower | 66.3% slower | 99.02% | 98.78% |
| int16 | 1/8 | Wall | 7.717 | 4.797 | 4.771 | 60.9% slower | 61.7% slower | — | — |
| int16 | 1/2 | Train | 10.819 | 7.986 | 6.937 | 35.5% slower | 56.0% slower | 90.62% | 91.46% |
| int16 | 1/2 | Wall | 11.640 | 8.800 | 7.565 | 32.3% slower | 53.9% slower | — | — |
| int16 | 1 | Train | 9.758 | 7.272 | 6.479 | 34.2% slower | 50.6% slower | 90.72% | 91.46% |
| int16 | 1 | Wall | 10.481 | 8.082 | 7.059 | 29.7% slower | 48.5% slower | — | — |
| int16 | 2 | Train | 10.014 | 7.453 | 6.700 | 34.4% slower | 49.4% slower | 89.99% | 90.41% |
| int16 | 2 | Wall | 10.733 | 8.214 | 7.295 | 30.7% slower | 47.1% slower | — | — |
| uint16 | 1/1024 | Train | 23.729 | 19.387 | 16.511 | 22.4% slower | 43.7% slower | 74.02% | 74.19% |
| uint16 | 1/1024 | Wall | 24.456 | 20.225 | 17.138 | 20.9% slower | 42.7% slower | — | — |
| uint16 | 1/256 | Train | 13.285 | 10.006 | 8.747 | 32.8% slower | 51.9% slower | 100.00% | 100.00% |
| uint16 | 1/256 | Wall | 14.000 | 10.757 | 9.357 | 30.1% slower | 49.6% slower | — | — |
| uint16 | 1/64 | Train | 9.421 | 4.758 | 4.945 | 98.0% slower | 90.5% slower | 100.00% | 100.00% |
| uint16 | 1/64 | Wall | 10.249 | 5.520 | 5.599 | 85.7% slower | 83.1% slower | — | — |
| uint16 | 1/32 | Train | 7.055 | 3.951 | 3.928 | 78.6% slower | 79.6% slower | 100.00% | 100.00% |
| uint16 | 1/32 | Wall | 7.775 | 4.706 | 4.525 | 65.2% slower | 71.8% slower | — | — |
| uint16 | 1/8 | Train | 7.088 | 3.977 | 3.844 | 78.2% slower | 84.4% slower | 99.02% | 98.78% |
| uint16 | 1/8 | Wall | 7.799 | 4.766 | 4.447 | 63.7% slower | 75.4% slower | — | — |
| uint16 | 1/2 | Train | 11.071 | 7.845 | 7.794 | 41.1% slower | 42.0% slower | 90.62% | 91.46% |
| uint16 | 1/2 | Wall | 11.971 | 8.668 | 8.496 | 38.1% slower | 40.9% slower | — | — |
| uint16 | 1 | Train | 10.127 | 7.060 | 6.564 | 43.4% slower | 54.3% slower | 90.72% | 91.46% |
| uint16 | 1 | Wall | 10.922 | 7.832 | 7.196 | 39.5% slower | 51.8% slower | — | — |
| uint16 | 2 | Train | 10.051 | 7.500 | 7.786 | 34.0% slower | 29.1% slower | 89.99% | 90.41% |
| uint16 | 2 | Wall | 10.766 | 8.256 | 8.481 | 30.4% slower | 26.9% slower | — | — |
| int32 | 1/1024 | Train | 24.569 | 19.115 | 15.727 | 28.5% slower | 56.2% slower | 74.02% | 74.19% |
| int32 | 1/1024 | Wall | 25.356 | 19.919 | 16.310 | 27.3% slower | 55.5% slower | — | — |
| int32 | 1/256 | Train | 15.181 | 10.568 | 9.781 | 43.7% slower | 55.2% slower | 100.00% | 100.00% |
| int32 | 1/256 | Wall | 15.938 | 11.357 | 10.392 | 40.3% slower | 53.4% slower | — | — |
| int32 | 1/64 | Train | 10.104 | 5.389 | 4.877 | 87.5% slower | 107.2% slower | 100.00% | 100.00% |
| int32 | 1/64 | Wall | 10.886 | 6.222 | 5.535 | 75.0% slower | 96.7% slower | — | — |
| int32 | 1/32 | Train | 9.458 | 4.256 | 4.243 | 122.2% slower | 122.9% slower | 100.00% | 100.00% |
| int32 | 1/32 | Wall | 10.228 | 5.071 | 4.800 | 101.7% slower | 113.1% slower | — | — |
| int32 | 1/8 | Train | 8.782 | 4.227 | 4.656 | 107.8% slower | 88.6% slower | 99.02% | 98.78% |
| int32 | 1/8 | Wall | 9.515 | 5.065 | 5.292 | 87.9% slower | 79.8% slower | — | — |
| int32 | 1/2 | Train | 11.593 | 7.953 | 6.931 | 45.8% slower | 67.3% slower | 90.62% | 91.46% |
| int32 | 1/2 | Wall | 12.339 | 8.787 | 7.574 | 40.4% slower | 62.9% slower | — | — |
| int32 | 1 | Train | 11.173 | 7.393 | 6.793 | 51.1% slower | 64.5% slower | 90.72% | 91.46% |
| int32 | 1 | Wall | 11.913 | 8.210 | 7.378 | 45.1% slower | 61.5% slower | — | — |
| int32 | 2 | Train | 15.435 | 9.008 | 7.252 | 71.3% slower | 112.8% slower | 89.99% | 90.41% |
| int32 | 2 | Wall | 16.453 | 10.012 | 7.851 | 64.3% slower | 109.6% slower | — | — |
| uint32 | 1/1024 | Train | 24.931 | 20.038 | 16.657 | 24.4% slower | 49.7% slower | 74.02% | 74.19% |
| uint32 | 1/1024 | Wall | 25.720 | 20.817 | 17.237 | 23.6% slower | 49.2% slower | — | — |
| uint32 | 1/256 | Train | 15.730 | 10.558 | 8.949 | 49.0% slower | 75.8% slower | 100.00% | 100.00% |
| uint32 | 1/256 | Wall | 16.570 | 11.387 | 9.513 | 45.5% slower | 74.2% slower | — | — |
| uint32 | 1/64 | Train | 9.714 | 5.011 | 4.931 | 93.9% slower | 97.0% slower | 100.00% | 100.00% |
| uint32 | 1/64 | Wall | 10.508 | 5.764 | 5.545 | 82.3% slower | 89.5% slower | — | — |
| uint32 | 1/32 | Train | 9.013 | 4.190 | 4.517 | 115.1% slower | 99.5% slower | 100.00% | 100.00% |
| uint32 | 1/32 | Wall | 9.780 | 4.963 | 5.074 | 97.1% slower | 92.8% slower | — | — |
| uint32 | 1/8 | Train | 8.839 | 4.790 | 4.424 | 84.5% slower | 99.8% slower | 99.02% | 98.78% |
| uint32 | 1/8 | Wall | 9.581 | 5.608 | 5.004 | 70.9% slower | 91.5% slower | — | — |
| uint32 | 1/2 | Train | 11.854 | 7.599 | 7.195 | 56.0% slower | 64.8% slower | 90.62% | 91.46% |
| uint32 | 1/2 | Wall | 12.631 | 8.368 | 7.785 | 50.9% slower | 62.2% slower | — | — |
| uint32 | 1 | Train | 11.465 | 7.235 | 7.157 | 58.5% slower | 60.2% slower | 90.72% | 91.46% |
| uint32 | 1 | Wall | 12.297 | 8.022 | 7.771 | 53.3% slower | 58.2% slower | — | — |
| uint32 | 2 | Train | 11.823 | 7.596 | 7.500 | 55.6% slower | 57.6% slower | 89.99% | 90.41% |
| uint32 | 2 | Wall | 12.609 | 8.389 | 8.128 | 50.3% slower | 55.1% slower | — | — |
| int64 | 1/1024 | Train | 33.206 | 13.201 | 13.223 | 151.5% slower | 151.1% slower | 74.02% | 74.19% |
| int64 | 1/1024 | Wall | 34.093 | 13.797 | 13.692 | 147.1% slower | 149.0% slower | — | — |
| int64 | 1/256 | Train | 20.353 | 7.694 | 8.121 | 164.5% slower | 150.6% slower | 100.00% | 100.00% |
| int64 | 1/256 | Wall | 21.139 | 8.290 | 8.631 | 155.0% slower | 144.9% slower | — | — |
| int64 | 1/64 | Train | 15.620 | 4.655 | 5.720 | 235.6% slower | 173.1% slower | 100.00% | 100.00% |
| int64 | 1/64 | Wall | 16.399 | 5.278 | 6.225 | 210.7% slower | 163.4% slower | — | — |
| int64 | 1/32 | Train | 14.543 | 4.079 | 4.707 | 256.6% slower | 208.9% slower | 100.00% | 100.00% |
| int64 | 1/32 | Wall | 15.361 | 4.670 | 5.214 | 228.9% slower | 194.6% slower | — | — |
| int64 | 1/8 | Train | 14.628 | 4.521 | 4.703 | 223.6% slower | 211.1% slower | 99.02% | 98.78% |
| int64 | 1/8 | Wall | 15.381 | 5.107 | 5.188 | 201.1% slower | 196.5% slower | — | — |
| int64 | 1/2 | Train | 17.614 | 6.210 | 6.527 | 183.6% slower | 169.8% slower | 90.62% | 91.46% |
| int64 | 1/2 | Wall | 18.366 | 6.797 | 6.980 | 170.2% slower | 163.1% slower | — | — |
| int64 | 1 | Train | 17.314 | 6.385 | 6.411 | 171.2% slower | 170.1% slower | 90.72% | 91.46% |
| int64 | 1 | Wall | 18.136 | 6.990 | 6.892 | 159.5% slower | 163.1% slower | — | — |
| int64 | 2 | Train | 18.211 | 6.183 | 7.949 | 194.5% slower | 129.1% slower | 89.99% | 90.41% |
| int64 | 2 | Wall | 19.054 | 6.753 | 8.480 | 182.2% slower | 124.7% slower | — | — |
| uint64 | 1/1024 | Train | 30.281 | 13.359 | 13.776 | 126.7% slower | 119.8% slower | 74.02% | 74.19% |
| uint64 | 1/1024 | Wall | 31.038 | 13.995 | 14.235 | 121.8% slower | 118.0% slower | — | — |
| uint64 | 1/256 | Train | 20.631 | 7.920 | 8.111 | 160.5% slower | 154.4% slower | 100.00% | 100.00% |
| uint64 | 1/256 | Wall | 21.385 | 8.536 | 8.608 | 150.5% slower | 148.4% slower | — | — |
| uint64 | 1/64 | Train | 15.169 | 4.556 | 5.131 | 233.0% slower | 195.6% slower | 100.00% | 100.00% |
| uint64 | 1/64 | Wall | 15.913 | 5.140 | 5.608 | 209.6% slower | 183.7% slower | — | — |
| uint64 | 1/32 | Train | 14.565 | 4.100 | 4.660 | 255.2% slower | 212.5% slower | 100.00% | 100.00% |
| uint64 | 1/32 | Wall | 15.308 | 4.670 | 5.140 | 227.8% slower | 197.8% slower | — | — |
| uint64 | 1/8 | Train | 14.841 | 4.045 | 4.797 | 266.9% slower | 209.4% slower | 99.02% | 98.78% |
| uint64 | 1/8 | Wall | 15.617 | 4.605 | 5.323 | 239.1% slower | 193.4% slower | — | — |
| uint64 | 1/2 | Train | 17.779 | 6.014 | 6.563 | 195.6% slower | 170.9% slower | 90.62% | 91.46% |
| uint64 | 1/2 | Wall | 18.525 | 6.605 | 7.044 | 180.5% slower | 163.0% slower | — | — |
| uint64 | 1 | Train | 16.882 | 6.305 | 6.759 | 167.7% slower | 149.8% slower | 90.72% | 91.46% |
| uint64 | 1 | Wall | 17.629 | 6.885 | 7.240 | 156.1% slower | 143.5% slower | — | — |
| uint64 | 2 | Train | 17.692 | 6.304 | 6.692 | 180.7% slower | 164.4% slower | 89.99% | 90.41% |
| uint64 | 2 | Wall | 18.446 | 6.898 | 7.168 | 167.4% slower | 157.3% slower | — | — |
| binary | 1/1024 | Train | 23.622 | 8.344 | 13.062 | 183.1% slower | 80.9% slower | 96.97% | 96.75% |
| binary | 1/1024 | Wall | 24.206 | 8.853 | 13.362 | 173.4% slower | 81.2% slower | — | — |
| binary | 1/256 | Train | 9.639 | 4.110 | 5.396 | 134.5% slower | 78.6% slower | 100.00% | 100.00% |
| binary | 1/256 | Wall | 10.210 | 4.600 | 5.698 | 122.0% slower | 79.2% slower | — | — |
| binary | 1/64 | Train | 6.053 | 3.011 | 3.448 | 101.0% slower | 75.6% slower | 100.00% | 100.00% |
| binary | 1/64 | Wall | 6.636 | 3.502 | 3.747 | 89.5% slower | 77.1% slower | — | — |
| binary | 1/32 | Train | 5.622 | 2.850 | 3.502 | 97.2% slower | 60.6% slower | 100.00% | 100.00% |
| binary | 1/32 | Wall | 6.210 | 3.350 | 3.823 | 85.4% slower | 62.4% slower | — | — |
| binary | 1/8 | Train | 5.074 | 2.707 | 2.850 | 87.5% slower | 78.0% slower | 100.00% | 100.00% |
| binary | 1/8 | Wall | 5.652 | 3.204 | 3.150 | 76.4% slower | 79.4% slower | — | — |
| binary | 1/2 | Train | 9.359 | 4.973 | 6.033 | 88.2% slower | 55.1% slower | 94.73% | 94.65% |
| binary | 1/2 | Wall | 9.928 | 5.483 | 6.331 | 81.1% slower | 56.8% slower | — | — |
| binary | 1 | Train | 26.249 | 15.421 | 20.025 | 70.2% slower | 31.1% slower | 63.87% | 63.82% |
| binary | 1 | Wall | 26.828 | 15.929 | 20.326 | 68.4% slower | 32.0% slower | — | — |
| binary | 2 | Train | 24.509 | 15.690 | 21.210 | 56.2% slower | 15.6% slower | 58.11% | 58.08% |
| binary | 2 | Wall | 25.086 | 16.119 | 21.542 | 55.6% slower | 16.4% slower | — | — |
| ternary | 1/1024 | Train | 27.287 | 11.761 | 16.466 | 132.0% slower | 65.7% slower | 74.12% | 74.15% |
| ternary | 1/1024 | Wall | 28.310 | 12.681 | 16.966 | 123.2% slower | 66.9% slower | — | — |
| ternary | 1/256 | Train | 15.899 | 8.068 | 10.399 | 97.1% slower | 52.9% slower | 100.00% | 100.00% |
| ternary | 1/256 | Wall | 16.912 | 8.986 | 10.905 | 88.2% slower | 55.1% slower | — | — |
| ternary | 1/64 | Train | 9.119 | 5.331 | 5.830 | 71.1% slower | 56.4% slower | 100.00% | 100.00% |
| ternary | 1/64 | Wall | 10.127 | 6.222 | 6.313 | 62.8% slower | 60.4% slower | — | — |
| ternary | 1/32 | Train | 7.982 | 4.706 | 5.263 | 69.6% slower | 51.7% slower | 100.00% | 100.00% |
| ternary | 1/32 | Wall | 8.997 | 5.565 | 5.756 | 61.7% slower | 56.3% slower | — | — |
| ternary | 1/8 | Train | 7.912 | 4.754 | 5.179 | 66.4% slower | 52.8% slower | 100.00% | 100.00% |
| ternary | 1/8 | Wall | 8.944 | 5.654 | 5.671 | 58.2% slower | 57.7% slower | — | — |
| ternary | 1/2 | Train | 25.452 | 16.581 | 18.994 | 53.5% slower | 34.0% slower | 70.36% | 70.53% |
| ternary | 1/2 | Wall | 26.401 | 17.400 | 19.532 | 51.7% slower | 35.2% slower | — | — |
| ternary | 1 | Train | 29.677 | 18.809 | 21.163 | 57.8% slower | 40.2% slower | 61.13% | 60.84% |
| ternary | 1 | Wall | 30.685 | 19.733 | 21.653 | 55.5% slower | 41.7% slower | — | — |
| ternary | 2 | Train | 28.806 | 18.670 | 25.461 | 54.3% slower | 13.1% slower | 58.11% | 58.08% |
| ternary | 2 | Wall | 29.624 | 19.343 | 25.924 | 53.2% slower | 14.3% slower | — | — |

#### Unbuffered

| Dtype | LR | Metric | Goose (ms) | C++ (ms) | Rust (ms) | Goose vs C++ | Goose vs Rust | Val. accuracy | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | 1/1024 | Train | 7.299 | 3.549 | 3.723 | 105.6% slower | 96.1% slower | 78.56% | 76.90% |
| float32 | 1/1024 | Wall | 8.282 | 3.857 | 3.966 | 114.7% slower | 108.8% slower | — | — |
| float32 | 1/256 | Train | 6.399 | 2.786 | 3.036 | 129.7% slower | 110.8% slower | 90.53% | 90.58% |
| float32 | 1/256 | Wall | 7.327 | 3.107 | 3.286 | 135.8% slower | 123.0% slower | — | — |
| float32 | 1/64 | Train | 6.389 | 2.283 | 2.713 | 179.8% slower | 135.5% slower | 90.97% | 91.46% |
| float32 | 1/64 | Wall | 7.443 | 2.553 | 2.966 | 191.5% slower | 151.0% slower | — | — |
| float32 | 1/32 | Train | 6.418 | 3.022 | 2.676 | 112.4% slower | 139.8% slower | 91.31% | 91.72% |
| float32 | 1/32 | Wall | 7.414 | 3.416 | 2.916 | 117.1% slower | 154.3% slower | — | — |
| float32 | 1/8 | Train | 6.089 | 2.158 | 2.678 | 182.1% slower | 127.3% slower | 90.67% | 91.06% |
| float32 | 1/8 | Wall | 7.039 | 2.487 | 2.927 | 183.0% slower | 140.5% slower | — | — |
| float32 | 1/2 | Train | 6.434 | 2.213 | 2.623 | 190.8% slower | 145.3% slower | 90.92% | 91.33% |
| float32 | 1/2 | Wall | 7.444 | 2.500 | 2.865 | 197.8% slower | 159.8% slower | — | — |
| float32 | 1 | Train | 6.435 | 2.193 | 2.642 | 193.4% slower | 143.6% slower | 90.72% | 91.46% |
| float32 | 1 | Wall | 7.419 | 2.458 | 2.882 | 201.8% slower | 157.5% slower | — | — |
| float32 | 2 | Train | 6.267 | 2.171 | 2.707 | 188.6% slower | 131.5% slower | 89.99% | 90.41% |
| float32 | 2 | Wall | 7.202 | 2.465 | 2.957 | 192.2% slower | 143.6% slower | — | — |
| float64 | 1/1024 | Train | 8.631 | 4.173 | 4.440 | 106.8% slower | 94.4% slower | 78.56% | 76.90% |
| float64 | 1/1024 | Wall | 9.081 | 4.471 | 4.674 | 103.1% slower | 94.3% slower | — | — |
| float64 | 1/256 | Train | 8.458 | 3.031 | 3.384 | 179.1% slower | 149.9% slower | 90.53% | 90.58% |
| float64 | 1/256 | Wall | 8.942 | 3.319 | 3.607 | 169.4% slower | 147.9% slower | — | — |
| float64 | 1/64 | Train | 7.310 | 2.772 | 3.198 | 163.7% slower | 128.6% slower | 90.97% | 91.46% |
| float64 | 1/64 | Wall | 7.751 | 3.054 | 3.419 | 153.8% slower | 126.7% slower | — | — |
| float64 | 1/32 | Train | 7.694 | 2.765 | 3.901 | 178.3% slower | 97.2% slower | 91.31% | 91.72% |
| float64 | 1/32 | Wall | 8.137 | 3.094 | 4.176 | 163.0% slower | 94.9% slower | — | — |
| float64 | 1/8 | Train | 7.213 | 2.717 | 3.238 | 165.5% slower | 122.8% slower | 90.67% | 91.06% |
| float64 | 1/8 | Wall | 7.653 | 3.018 | 3.459 | 153.5% slower | 121.3% slower | — | — |
| float64 | 1/2 | Train | 7.193 | 2.665 | 3.173 | 169.9% slower | 126.7% slower | 90.92% | 91.33% |
| float64 | 1/2 | Wall | 7.632 | 2.949 | 3.390 | 158.8% slower | 125.1% slower | — | — |
| float64 | 1 | Train | 7.165 | 2.693 | 3.286 | 166.1% slower | 118.0% slower | 90.72% | 91.46% |
| float64 | 1 | Wall | 7.627 | 3.011 | 3.509 | 153.3% slower | 117.3% slower | — | — |
| float64 | 2 | Train | 7.208 | 2.717 | 3.179 | 165.3% slower | 126.7% slower | 89.99% | 90.41% |
| float64 | 2 | Wall | 7.647 | 3.006 | 3.420 | 154.4% slower | 123.6% slower | — | — |
| int8 | 1/1024 | Train | 27.719 | 30.181 | 20.991 | 8.2% faster | 32.1% slower | 48.19% | 49.34% |
| int8 | 1/1024 | Wall | 28.417 | 30.781 | 21.672 | 7.7% faster | 31.1% slower | — | — |
| int8 | 1/256 | Train | 28.428 | 31.337 | 21.798 | 9.3% faster | 30.4% slower | 48.19% | 49.34% |
| int8 | 1/256 | Wall | 29.147 | 31.982 | 22.457 | 8.9% faster | 29.8% slower | — | — |
| int8 | 1/64 | Train | 28.696 | 31.941 | 23.413 | 10.2% faster | 22.6% slower | 48.19% | 49.34% |
| int8 | 1/64 | Wall | 29.404 | 32.548 | 24.138 | 9.7% faster | 21.8% slower | — | — |
| int8 | 1/32 | Train | 29.799 | 32.380 | 23.991 | 8.0% faster | 24.2% slower | 48.19% | 49.34% |
| int8 | 1/32 | Wall | 30.521 | 33.050 | 24.694 | 7.7% faster | 23.6% slower | — | — |
| int8 | 1/8 | Train | 27.706 | 30.085 | 21.703 | 7.9% faster | 27.7% slower | 48.19% | 49.34% |
| int8 | 1/8 | Wall | 28.428 | 30.682 | 22.372 | 7.3% faster | 27.1% slower | — | — |
| int8 | 1/2 | Train | 28.580 | 30.785 | 22.336 | 7.2% faster | 28.0% slower | 48.19% | 49.34% |
| int8 | 1/2 | Wall | 29.311 | 31.443 | 23.029 | 6.8% faster | 27.3% slower | — | — |
| int8 | 1 | Train | 14.539 | 12.409 | 12.826 | 17.2% slower | 13.4% slower | 71.19% | 71.46% |
| int8 | 1 | Wall | 15.243 | 13.026 | 13.560 | 17.0% slower | 12.4% slower | — | — |
| int8 | 2 | Train | 26.371 | 28.949 | 19.904 | 8.9% faster | 32.5% slower | 50.39% | 50.59% |
| int8 | 2 | Wall | 27.066 | 29.539 | 20.618 | 8.4% faster | 31.3% slower | — | — |
| uint8 | 1/1024 | Train | 27.763 | 30.363 | 22.717 | 8.6% faster | 22.2% slower | 48.19% | 49.34% |
| uint8 | 1/1024 | Wall | 28.570 | 30.970 | 23.426 | 7.7% faster | 22.0% slower | — | — |
| uint8 | 1/256 | Train | 29.248 | 31.091 | 22.334 | 5.9% faster | 31.0% slower | 48.19% | 49.34% |
| uint8 | 1/256 | Wall | 29.967 | 31.705 | 23.014 | 5.5% faster | 30.2% slower | — | — |
| uint8 | 1/64 | Train | 28.382 | 33.533 | 21.172 | 15.4% faster | 34.1% slower | 48.19% | 49.34% |
| uint8 | 1/64 | Wall | 29.142 | 34.218 | 21.863 | 14.8% faster | 33.3% slower | — | — |
| uint8 | 1/32 | Train | 28.629 | 30.753 | 22.319 | 6.9% faster | 28.3% slower | 48.19% | 49.34% |
| uint8 | 1/32 | Wall | 29.375 | 31.454 | 23.027 | 6.6% faster | 27.6% slower | — | — |
| uint8 | 1/8 | Train | 28.782 | 31.283 | 24.540 | 8.0% faster | 17.3% slower | 48.19% | 49.34% |
| uint8 | 1/8 | Wall | 29.539 | 31.918 | 25.276 | 7.5% faster | 16.9% slower | — | — |
| uint8 | 1/2 | Train | 27.816 | 30.195 | 21.254 | 7.9% faster | 30.9% slower | 48.19% | 49.34% |
| uint8 | 1/2 | Wall | 28.521 | 30.821 | 21.932 | 7.5% faster | 30.0% slower | — | — |
| uint8 | 1 | Train | 14.346 | 12.174 | 9.235 | 17.8% slower | 55.3% slower | 71.19% | 71.46% |
| uint8 | 1 | Wall | 15.042 | 12.765 | 9.958 | 17.8% slower | 51.0% slower | — | — |
| uint8 | 2 | Train | 27.224 | 29.695 | 20.616 | 8.3% faster | 32.1% slower | 50.39% | 50.59% |
| uint8 | 2 | Wall | 27.968 | 30.342 | 21.372 | 7.8% faster | 30.9% slower | — | — |
| int16 | 1/1024 | Train | 26.318 | 26.280 | 22.155 | 0.1% slower | 18.8% slower | 48.19% | 49.34% |
| int16 | 1/1024 | Wall | 27.031 | 27.048 | 22.731 | 0.1% faster | 18.9% slower | — | — |
| int16 | 1/256 | Train | 26.763 | 26.526 | 21.449 | 0.9% slower | 24.8% slower | 48.19% | 49.34% |
| int16 | 1/256 | Wall | 27.541 | 27.325 | 21.996 | 0.8% slower | 25.2% slower | — | — |
| int16 | 1/64 | Train | 26.754 | 26.394 | 21.397 | 1.4% slower | 25.0% slower | 48.19% | 49.34% |
| int16 | 1/64 | Wall | 27.516 | 27.164 | 21.941 | 1.3% slower | 25.4% slower | — | — |
| int16 | 1/32 | Train | 26.492 | 26.302 | 21.933 | 0.7% slower | 20.8% slower | 48.19% | 49.34% |
| int16 | 1/32 | Wall | 27.269 | 27.078 | 22.482 | 0.7% slower | 21.3% slower | — | — |
| int16 | 1/8 | Train | 26.708 | 26.289 | 21.596 | 1.6% slower | 23.7% slower | 48.19% | 49.34% |
| int16 | 1/8 | Wall | 27.506 | 27.041 | 22.149 | 1.7% slower | 24.2% slower | — | — |
| int16 | 1/2 | Train | 27.215 | 27.416 | 21.571 | 0.7% faster | 26.2% slower | 48.19% | 49.34% |
| int16 | 1/2 | Wall | 27.984 | 28.270 | 22.112 | 1.0% faster | 26.6% slower | — | — |
| int16 | 1 | Train | 9.664 | 7.170 | 6.450 | 34.8% slower | 49.8% slower | 90.72% | 91.46% |
| int16 | 1 | Wall | 10.450 | 7.938 | 7.017 | 31.6% slower | 48.9% slower | — | — |
| int16 | 2 | Train | 10.458 | 7.723 | 6.758 | 35.4% slower | 54.8% slower | 89.99% | 90.41% |
| int16 | 2 | Wall | 11.175 | 8.566 | 7.326 | 30.5% slower | 52.5% slower | — | — |
| uint16 | 1/1024 | Train | 29.175 | 27.601 | 22.370 | 5.7% slower | 30.4% slower | 48.19% | 49.34% |
| uint16 | 1/1024 | Wall | 29.920 | 28.369 | 23.015 | 5.5% slower | 30.0% slower | — | — |
| uint16 | 1/256 | Train | 27.855 | 26.453 | 21.496 | 5.3% slower | 29.6% slower | 48.19% | 49.34% |
| uint16 | 1/256 | Wall | 28.577 | 27.270 | 22.119 | 4.8% slower | 29.2% slower | — | — |
| uint16 | 1/64 | Train | 27.188 | 26.278 | 21.342 | 3.5% slower | 27.4% slower | 48.19% | 49.34% |
| uint16 | 1/64 | Wall | 27.978 | 27.070 | 21.957 | 3.4% slower | 27.4% slower | — | — |
| uint16 | 1/32 | Train | 27.731 | 26.608 | 22.016 | 4.2% slower | 26.0% slower | 48.19% | 49.34% |
| uint16 | 1/32 | Wall | 28.473 | 27.390 | 22.662 | 4.0% slower | 25.6% slower | — | — |
| uint16 | 1/8 | Train | 27.531 | 26.900 | 22.551 | 2.3% slower | 22.1% slower | 48.19% | 49.34% |
| uint16 | 1/8 | Wall | 28.267 | 27.698 | 23.169 | 2.1% slower | 22.0% slower | — | — |
| uint16 | 1/2 | Train | 26.765 | 26.966 | 22.382 | 0.7% faster | 19.6% slower | 48.19% | 49.34% |
| uint16 | 1/2 | Wall | 27.483 | 27.767 | 23.049 | 1.0% faster | 19.2% slower | — | — |
| uint16 | 1 | Train | 9.619 | 7.100 | 6.741 | 35.5% slower | 42.7% slower | 90.72% | 91.46% |
| uint16 | 1 | Wall | 10.357 | 7.856 | 7.380 | 31.8% slower | 40.3% slower | — | — |
| uint16 | 2 | Train | 10.145 | 7.787 | 6.773 | 30.3% slower | 49.8% slower | 89.99% | 90.41% |
| uint16 | 2 | Wall | 10.859 | 8.568 | 7.396 | 26.7% slower | 46.8% slower | — | — |
| int32 | 1/1024 | Train | 28.474 | 27.607 | 22.139 | 3.1% slower | 28.6% slower | 48.19% | 49.34% |
| int32 | 1/1024 | Wall | 29.235 | 28.405 | 22.713 | 2.9% slower | 28.7% slower | — | — |
| int32 | 1/256 | Train | 27.788 | 26.308 | 21.959 | 5.6% slower | 26.5% slower | 48.19% | 49.34% |
| int32 | 1/256 | Wall | 28.525 | 27.110 | 22.607 | 5.2% slower | 26.2% slower | — | — |
| int32 | 1/64 | Train | 28.033 | 26.936 | 21.778 | 4.1% slower | 28.7% slower | 48.19% | 49.34% |
| int32 | 1/64 | Wall | 28.772 | 27.772 | 22.361 | 3.6% slower | 28.7% slower | — | — |
| int32 | 1/32 | Train | 28.107 | 26.870 | 22.879 | 4.6% slower | 22.9% slower | 48.19% | 49.34% |
| int32 | 1/32 | Wall | 28.907 | 27.695 | 23.493 | 4.4% slower | 23.0% slower | — | — |
| int32 | 1/8 | Train | 31.363 | 28.483 | 25.043 | 10.1% slower | 25.2% slower | 48.19% | 49.34% |
| int32 | 1/8 | Wall | 32.184 | 29.354 | 25.655 | 9.6% slower | 25.4% slower | — | — |
| int32 | 1/2 | Train | 28.413 | 26.981 | 22.013 | 5.3% slower | 29.1% slower | 48.19% | 49.34% |
| int32 | 1/2 | Wall | 29.165 | 27.865 | 22.595 | 4.7% slower | 29.1% slower | — | — |
| int32 | 1 | Train | 12.219 | 7.490 | 7.668 | 63.1% slower | 59.3% slower | 90.72% | 91.46% |
| int32 | 1 | Wall | 12.975 | 8.294 | 8.342 | 56.4% slower | 55.5% slower | — | — |
| int32 | 2 | Train | 11.883 | 8.026 | 7.158 | 48.1% slower | 66.0% slower | 89.99% | 90.41% |
| int32 | 2 | Wall | 12.702 | 8.850 | 7.760 | 43.5% slower | 63.7% slower | — | — |
| uint32 | 1/1024 | Train | 28.606 | 27.417 | 22.270 | 4.3% slower | 28.5% slower | 48.19% | 49.34% |
| uint32 | 1/1024 | Wall | 29.386 | 28.178 | 22.849 | 4.3% slower | 28.6% slower | — | — |
| uint32 | 1/256 | Train | 29.263 | 28.192 | 22.340 | 3.8% slower | 31.0% slower | 48.19% | 49.34% |
| uint32 | 1/256 | Wall | 30.131 | 29.124 | 22.984 | 3.5% slower | 31.1% slower | — | — |
| uint32 | 1/64 | Train | 27.933 | 26.303 | 22.027 | 6.2% slower | 26.8% slower | 48.19% | 49.34% |
| uint32 | 1/64 | Wall | 28.677 | 27.057 | 22.591 | 6.0% slower | 26.9% slower | — | — |
| uint32 | 1/32 | Train | 27.410 | 26.276 | 21.434 | 4.3% slower | 27.9% slower | 48.19% | 49.34% |
| uint32 | 1/32 | Wall | 28.147 | 27.060 | 22.020 | 4.0% slower | 27.8% slower | — | — |
| uint32 | 1/8 | Train | 29.211 | 26.131 | 22.067 | 11.8% slower | 32.4% slower | 48.19% | 49.34% |
| uint32 | 1/8 | Wall | 30.052 | 26.902 | 22.639 | 11.7% slower | 32.7% slower | — | — |
| uint32 | 1/2 | Train | 27.751 | 26.288 | 21.776 | 5.6% slower | 27.4% slower | 48.19% | 49.34% |
| uint32 | 1/2 | Wall | 28.536 | 27.065 | 22.370 | 5.4% slower | 27.6% slower | — | — |
| uint32 | 1 | Train | 11.293 | 7.525 | 7.902 | 50.1% slower | 42.9% slower | 90.72% | 91.46% |
| uint32 | 1 | Wall | 12.040 | 8.345 | 8.559 | 44.3% slower | 40.7% slower | — | — |
| uint32 | 2 | Train | 11.486 | 7.830 | 7.052 | 46.7% slower | 62.9% slower | 89.99% | 90.41% |
| uint32 | 2 | Wall | 12.222 | 8.637 | 7.648 | 41.5% slower | 59.8% slower | — | — |
| int64 | 1/1024 | Train | 33.351 | 18.954 | 18.520 | 76.0% slower | 80.1% slower | 48.19% | 49.34% |
| int64 | 1/1024 | Wall | 34.180 | 19.643 | 18.997 | 74.0% slower | 79.9% slower | — | — |
| int64 | 1/256 | Train | 33.913 | 20.910 | 18.690 | 62.2% slower | 81.4% slower | 48.19% | 49.34% |
| int64 | 1/256 | Wall | 34.664 | 21.655 | 19.170 | 60.1% slower | 80.8% slower | — | — |
| int64 | 1/64 | Train | 33.477 | 19.047 | 18.619 | 75.8% slower | 79.8% slower | 48.19% | 49.34% |
| int64 | 1/64 | Wall | 34.271 | 19.654 | 19.126 | 74.4% slower | 79.2% slower | — | — |
| int64 | 1/32 | Train | 33.774 | 20.141 | 18.340 | 67.7% slower | 84.2% slower | 48.19% | 49.34% |
| int64 | 1/32 | Wall | 34.535 | 20.721 | 18.798 | 66.7% slower | 83.7% slower | — | — |
| int64 | 1/8 | Train | 36.438 | 18.589 | 17.942 | 96.0% slower | 103.1% slower | 48.19% | 49.34% |
| int64 | 1/8 | Wall | 37.259 | 19.205 | 18.418 | 94.0% slower | 102.3% slower | — | — |
| int64 | 1/2 | Train | 33.638 | 19.662 | 19.045 | 71.1% slower | 76.6% slower | 48.19% | 49.34% |
| int64 | 1/2 | Wall | 34.500 | 20.278 | 19.529 | 70.1% slower | 76.7% slower | — | — |
| int64 | 1 | Train | 16.926 | 6.063 | 7.481 | 179.2% slower | 126.2% slower | 90.72% | 91.46% |
| int64 | 1 | Wall | 17.721 | 6.620 | 8.038 | 167.7% slower | 120.5% slower | — | — |
| int64 | 2 | Train | 17.125 | 6.431 | 6.701 | 166.3% slower | 155.5% slower | 89.99% | 90.41% |
| int64 | 2 | Wall | 17.882 | 7.040 | 7.177 | 154.0% slower | 149.2% slower | — | — |
| uint64 | 1/1024 | Train | 33.667 | 19.345 | 18.187 | 74.0% slower | 85.1% slower | 48.19% | 49.34% |
| uint64 | 1/1024 | Wall | 34.494 | 19.984 | 18.732 | 72.6% slower | 84.1% slower | — | — |
| uint64 | 1/256 | Train | 33.419 | 19.091 | 18.580 | 75.0% slower | 79.9% slower | 48.19% | 49.34% |
| uint64 | 1/256 | Wall | 34.183 | 19.672 | 19.058 | 73.8% slower | 79.4% slower | — | — |
| uint64 | 1/64 | Train | 33.595 | 19.330 | 18.756 | 73.8% slower | 79.1% slower | 48.19% | 49.34% |
| uint64 | 1/64 | Wall | 34.353 | 19.959 | 19.253 | 72.1% slower | 78.4% slower | — | — |
| uint64 | 1/32 | Train | 33.642 | 19.127 | 18.202 | 75.9% slower | 84.8% slower | 48.19% | 49.34% |
| uint64 | 1/32 | Wall | 34.465 | 19.718 | 18.706 | 74.8% slower | 84.2% slower | — | — |
| uint64 | 1/8 | Train | 33.690 | 19.479 | 18.792 | 73.0% slower | 79.3% slower | 48.19% | 49.34% |
| uint64 | 1/8 | Wall | 34.442 | 20.165 | 19.298 | 70.8% slower | 78.5% slower | — | — |
| uint64 | 1/2 | Train | 33.559 | 19.893 | 18.759 | 68.7% slower | 78.9% slower | 48.19% | 49.34% |
| uint64 | 1/2 | Wall | 34.307 | 20.497 | 19.244 | 67.4% slower | 78.3% slower | — | — |
| uint64 | 1 | Train | 17.774 | 6.456 | 6.467 | 175.3% slower | 174.8% slower | 90.72% | 91.46% |
| uint64 | 1 | Wall | 18.619 | 7.100 | 6.941 | 162.2% slower | 168.2% slower | — | — |
| uint64 | 2 | Train | 17.527 | 6.376 | 7.327 | 174.9% slower | 139.2% slower | 89.99% | 90.41% |
| uint64 | 2 | Wall | 18.287 | 6.961 | 7.815 | 162.7% slower | 134.0% slower | — | — |
| binary | 1/1024 | Train | 31.871 | 17.659 | 25.815 | 80.5% slower | 23.5% slower | 48.19% | 49.34% |
| binary | 1/1024 | Wall | 32.478 | 18.199 | 26.123 | 78.5% slower | 24.3% slower | — | — |
| binary | 1/256 | Train | 31.641 | 17.309 | 24.979 | 82.8% slower | 26.7% slower | 48.19% | 49.34% |
| binary | 1/256 | Wall | 32.247 | 17.834 | 25.281 | 80.8% slower | 27.6% slower | — | — |
| binary | 1/64 | Train | 31.932 | 17.653 | 25.604 | 80.9% slower | 24.7% slower | 48.19% | 49.34% |
| binary | 1/64 | Wall | 32.503 | 18.184 | 25.975 | 78.7% slower | 25.1% slower | — | — |
| binary | 1/32 | Train | 31.973 | 17.610 | 25.448 | 81.6% slower | 25.6% slower | 48.19% | 49.34% |
| binary | 1/32 | Wall | 32.543 | 18.120 | 25.747 | 79.6% slower | 26.4% slower | — | — |
| binary | 1/8 | Train | 33.236 | 18.081 | 26.562 | 83.8% slower | 25.1% slower | 48.19% | 49.34% |
| binary | 1/8 | Wall | 33.867 | 18.611 | 26.918 | 82.0% slower | 25.8% slower | — | — |
| binary | 1/2 | Train | 33.109 | 18.832 | 27.103 | 75.8% slower | 22.2% slower | 48.19% | 49.34% |
| binary | 1/2 | Wall | 33.723 | 19.372 | 27.443 | 74.1% slower | 22.9% slower | — | — |
| binary | 1 | Train | 31.994 | 17.337 | 25.615 | 84.5% slower | 24.9% slower | 48.19% | 49.34% |
| binary | 1 | Wall | 32.560 | 17.853 | 25.940 | 82.4% slower | 25.5% slower | — | — |
| binary | 2 | Train | 24.541 | 15.961 | 21.750 | 53.8% slower | 12.8% slower | 58.11% | 58.08% |
| binary | 2 | Wall | 25.165 | 16.414 | 22.074 | 53.3% slower | 14.0% slower | — | — |
| ternary | 1/1024 | Train | 36.377 | 21.081 | 30.491 | 72.6% slower | 19.3% slower | 48.19% | 49.34% |
| ternary | 1/1024 | Wall | 37.396 | 22.081 | 31.028 | 69.4% slower | 20.5% slower | — | — |
| ternary | 1/256 | Train | 36.851 | 21.802 | 30.308 | 69.0% slower | 21.6% slower | 48.19% | 49.34% |
| ternary | 1/256 | Wall | 37.860 | 22.671 | 30.838 | 67.0% slower | 22.8% slower | — | — |
| ternary | 1/64 | Train | 35.798 | 21.425 | 28.325 | 67.1% slower | 26.4% slower | 48.19% | 49.34% |
| ternary | 1/64 | Wall | 36.921 | 22.420 | 28.813 | 64.7% slower | 28.1% slower | — | — |
| ternary | 1/32 | Train | 34.509 | 21.663 | 28.538 | 59.3% slower | 20.9% slower | 48.19% | 49.34% |
| ternary | 1/32 | Wall | 35.524 | 22.639 | 29.033 | 56.9% slower | 22.4% slower | — | — |
| ternary | 1/8 | Train | 36.311 | 20.664 | 29.695 | 75.7% slower | 22.3% slower | 48.19% | 49.34% |
| ternary | 1/8 | Wall | 37.367 | 21.571 | 30.212 | 73.2% slower | 23.7% slower | — | — |
| ternary | 1/2 | Train | 36.405 | 21.700 | 28.830 | 67.8% slower | 26.3% slower | 48.19% | 49.34% |
| ternary | 1/2 | Wall | 37.475 | 22.616 | 29.362 | 65.7% slower | 27.6% slower | — | — |
| ternary | 1 | Train | 29.983 | 18.392 | 22.195 | 63.0% slower | 35.1% slower | 61.13% | 60.84% |
| ternary | 1 | Wall | 30.948 | 19.295 | 22.754 | 60.4% slower | 36.0% slower | — | — |
| ternary | 2 | Train | 27.956 | 18.033 | 23.671 | 55.0% slower | 18.1% slower | 58.11% | 58.08% |
| ternary | 2 | Wall | 28.767 | 18.657 | 24.131 | 54.2% slower | 19.2% slower | — | — |

### Time to 90% validation accuracy

One buffered rate per dtype, chosen by mean final validation error, then mean first epoch reaching 90%, then smaller learning rate. The table uses cumulative training time at the first qualifying checkpoint, taking the median across the seeds that reached it.

| Dtype | Selected LR | Seeds reaching 90% | Epoch (median) | Goose (ms) | C++ (ms) | Rust (ms) | Goose vs C++ | Goose vs Rust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | 1/32 | 3/3 | 4 | 0.825 | 0.324 | 0.368 | 154.7% slower | 124.2% slower |
| float64 | 1/32 | 3/3 | 4 | 0.926 | 0.382 | 0.429 | 142.5% slower | 115.8% slower |
| int8 | 1/32 | 3/3 | 4 | 2.119 | 1.620 | 1.278 | 30.8% slower | 65.9% slower |
| uint8 | 1/32 | 3/3 | 4 | 2.076 | 1.625 | 1.397 | 27.8% slower | 48.6% slower |
| int16 | 1/32 | 3/3 | 4 | 1.820 | 1.633 | 1.262 | 11.5% slower | 44.2% slower |
| uint16 | 1/32 | 3/3 | 4 | 1.876 | 1.436 | 1.220 | 30.6% slower | 53.8% slower |
| int32 | 1/32 | 3/3 | 4 | 2.005 | 1.474 | 1.299 | 36.0% slower | 54.4% slower |
| uint32 | 1/32 | 3/3 | 4 | 2.016 | 1.439 | 1.456 | 40.0% slower | 38.4% slower |
| int64 | 1/32 | 3/3 | 4 | 2.730 | 1.133 | 1.120 | 141.0% slower | 143.7% slower |
| uint64 | 1/32 | 3/3 | 4 | 2.987 | 1.193 | 1.096 | 150.3% slower | 172.5% slower |
| binary | 1/64 | 3/3 | 4 | 1.836 | 0.760 | 1.050 | 141.5% slower | 74.9% slower |
| ternary | 1/32 | 3/3 | 4 | 2.237 | 1.232 | 1.533 | 81.6% slower | 45.9% slower |

### Accuracy at the selected buffered rate

Mean accuracy across the three initialization seeds. Each value applies to all three implementations.

| Dtype | Selected LR | Initial validation | Final train | Final validation | Final test |
| --- | --- | --- | --- | --- | --- |
| float32 | 1/32 | 49.54% | 94.68% | 91.34% | 91.81% |
| float64 | 1/32 | 49.54% | 94.68% | 91.34% | 91.81% |
| int8 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| uint8 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| int16 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| uint16 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| int32 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| uint32 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| int64 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| uint64 | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |
| binary | 1/64 | 49.54% | 100.00% | 100.00% | 100.00% |
| ternary | 1/32 | 49.54% | 100.00% | 100.00% | 100.00% |

### Validation effect of retaining the buffer at LR 1/32

Mean final validation accuracy across seeds. Differences are percentage points.

| Dtype | Buffered | Unbuffered | Buffer effect |
| --- | --- | --- | --- |
| float32 | 91.34% | 91.34% | +0.00 |
| float64 | 91.34% | 91.34% | +0.00 |
| int8 | 100.00% | 49.54% | +50.46 |
| uint8 | 100.00% | 49.54% | +50.46 |
| int16 | 100.00% | 49.54% | +50.46 |
| uint16 | 100.00% | 49.54% | +50.46 |
| int32 | 100.00% | 49.54% | +50.46 |
| uint32 | 100.00% | 49.54% | +50.46 |
| int64 | 100.00% | 49.54% | +50.46 |
| uint64 | 100.00% | 49.54% | +50.46 |
| binary | 100.00% | 49.54% | +50.46 |
| ternary | 100.00% | 49.54% | +50.46 |

## Coverage boundaries

The C++ and Rust dense ports benchmark inference for 34 storage dtypes. Their native training workloads benchmark 12 arithmetic representations. The 24 stack-local training modes, Go CPU/SIMD/WebGPU backends, and Go/Mojo grid and accelerator experiments belong to separate suites. This repository has no matching C++ and Rust results for those modes. These tables cover CPU latency, nominal dense throughput and native training metrics.

## Reproduce

```sh
./welvet_goose/install.sh
./welvet_goose/build_references.sh
./welvet_goose/build_goose.sh bench storage widened regression training
python3 welvet_goose/run.py all --core 2 --rounds 3
python3 welvet_goose/report.py
```

Run individual sections with `storage`, `native`, `tuned`, `regression` or `learning`. The runner targets Linux x86-64 with AVX2, FMA and F16C. Choose an available logical core on your machine.
