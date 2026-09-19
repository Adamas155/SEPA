# Seed-0 low-Top-1 diagnostics

Status: running. Updated: 2026-09-12T05:27:45.701150+00:00

All encoders are frozen. Original recipe: SGD, LR 0.05, 50 epochs, batch 128; all training labels.

| Encoder | Top-1 (%) | Top-5 (%) | kNN (%) |
|---|---:|---:|---:|
| k0_student | 15.18 | 36.98 | 11.32 |
| k0_teacher | 15.32 | 37.02 | 10.70 |
| random_tile | 12.40 | 32.16 | 6.92 |
| full_student | 18.62 | 41.74 | 15.04 |
| full_teacher | 18.54 | 41.66 | 14.60 |
| random_full | 14.30 | 35.68 | 7.70 |
| k3_student | 17.26 | 40.32 | 11.76 |
| k3_teacher | 17.46 | 40.28 | 11.58 |
| k6_student | 18.00 | 41.90 | 12.62 |
| k6_teacher | 18.28 | 42.04 | 11.96 |

## Feature health (fixed, unique, class-balanced training images)

| Encoder | Effective rank | Participation rank | Top-PC variance (%) | Mean std |
|---|---:|---:|---:|---:|
| k0_student | 11.97 | 7.16 | 25.58 | 0.118704 |
| k0_teacher | 11.70 | 7.06 | 25.90 | 0.126339 |
| random_tile | 2.91 | 1.93 | 69.87 | 0.382600 |
| full_student | 61.06 | 42.15 | 4.98 | 0.555640 |
| full_teacher | 59.74 | 41.18 | 5.08 | 0.555456 |
| random_full | 2.99 | 1.91 | 70.36 | 0.402205 |
| k3_student | 12.20 | 6.52 | 32.10 | 0.118559 |
| k3_teacher | 11.84 | 6.35 | 32.72 | 0.124471 |
| k6_student | 14.94 | 8.70 | 23.59 | 0.165710 |
| k6_teacher | 14.82 | 8.65 | 24.66 | 0.170102 |

Spectra and cosine values are descriptive; compare each trained encoder with its matched random architecture.

Data audit: 86 training records have image-identical conflicting labels. These image groups are excluded from internal train/dev and feature-health sampling. Full-data baseline and selected-recipe refits retain the original manifest for comparability.

## Classifier convergence

Mean internal-dev Top-1 across k0 student and full student; ties: fewer epochs, then smaller LR.

Selected shared recipe: LR 0.005, 200 epochs. Official validation was excluded from selection.

| LR | Epochs | k0 internal-dev Top-1 (%) | Full internal-dev Top-1 (%) | Mean (%) |
|---|---:|---:|---:|---:|
| 0.005 | 200 | 20.19 | 24.38 | 22.29 |
| 0.005 | 100 | 19.51 | 23.63 | 21.57 |
| 0.005 | 50 | 18.34 | 22.97 | 20.65 |
| 0.05 | 50 | 17.31 | 21.08 | 19.20 |
| 0.05 | 200 | 17.21 | 20.91 | 19.06 |
| 0.05 | 100 | 17.26 | 20.64 | 18.95 |
| 0.1 | 50 | 14.74 | 18.65 | 16.70 |
| 0.1 | 200 | 14.00 | 18.76 | 16.38 |
| 0.1 | 100 | 14.67 | 18.08 | 16.37 |

Single seed only. This diagnostic batch does not establish the roadmap's multi-seed success criteria.
