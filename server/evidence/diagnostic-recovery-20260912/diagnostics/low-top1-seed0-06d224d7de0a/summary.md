# Seed-0 low-Top-1 diagnostics

Status: running. Updated: 2026-09-12T04:22:27.445957+00:00

All encoders are frozen. Original recipe: SGD, LR 0.05, 50 epochs, batch 128; all training labels.

| Encoder | Top-1 (%) | Top-5 (%) | kNN (%) |
|---|---:|---:|---:|
| k0_student | 15.18 | 36.98 | 11.32 |

## Feature health (fixed, unique, class-balanced training images)

| Encoder | Effective rank | Participation rank | Top-PC variance (%) | Mean std |
|---|---:|---:|---:|---:|
| k0_student | 11.97 | 7.16 | 25.58 | 0.118704 |

Spectra and cosine values are descriptive; compare each trained encoder with its matched random architecture.

Data audit: 86 training records have image-identical conflicting labels. These image groups are excluded from internal train/dev and feature-health sampling. Full-data baseline and selected-recipe refits retain the original manifest for comparability.

Single seed only. This diagnostic batch does not establish the roadmap's multi-seed success criteria.
