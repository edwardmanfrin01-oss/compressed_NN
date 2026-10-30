# Existing compression statistics

IGDs compared with raw RGB/RGBA at 8 bits per channel and with existing PNGs.
Savings = 100 × (1 − IGD bytes / reference bytes). Negative values indicate expansion.
Aggregate savings are calculated from byte totals.

| Group | Images | Aggregate RGB savings | Aggregate PNG savings | Expansions RGB | Expansions PNG |
|---|---:|---:|---:|---:|---:|
| all | 60000 | 24.86% | -248.73% | 2592 | 60000 |
| split/test | 10000 | 24.84% | -248.60% | 425 | 10000 |
| split/train | 50000 | 24.86% | -248.75% | 2167 | 50000 |

"Aggregate RGB savings" means IGD file savings relative to raw RGB images.
"Aggregate PNG savings" means IGD file savings relative to PNG files. Since PNGs are always
smaller in these results, this value is negative.

## all

| Savings (%) | Mean | Population std | Min | P05 | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw RGB | 24.86 | 11.31 | -10.21 | 0.98 | 25.49 | 40.55 | 65.98 | 85.96 |
| PNG | -262.94 | 87.19 | -1738.12 | -433.48 | -240.58 | -171.64 | -151.74 | -109.11 |

Raw bytes: 9,031,680,000; IGD bytes: 6,786,679,971.
Matched, nonempty PNGs: 60,000.

## split/test

| Savings (%) | Mean | Population std | Min | P05 | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw RGB | 24.84 | 11.30 | -8.99 | 1.06 | 25.35 | 40.75 | 65.98 | 83.37 |
| PNG | -262.96 | 88.30 | -1242.20 | -430.95 | -241.12 | -171.81 | -153.32 | -124.28 |

Raw bytes: 1,505,280,000; IGD bytes: 1,131,334,238.
Matched, nonempty PNGs: 10,000.

## split/train

| Savings (%) | Mean | Population std | Min | P05 | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw RGB | 24.86 | 11.32 | -10.21 | 0.97 | 25.52 | 40.49 | 65.98 | 85.96 |
| PNG | -262.94 | 86.97 | -1738.12 | -433.99 | -240.49 | -171.57 | -151.49 | -109.11 |

Raw bytes: 7,526,400,000; IGD bytes: 5,655,345,733.
Matched, nonempty PNGs: 50,000.

Missing PNGs: 0.
Checks: IGD header and payload length; no reconstruction verification.
No recompression or compression timing measurements.
