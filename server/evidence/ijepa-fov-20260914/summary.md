# Frozen I-JEPA receptive-field comparison

Frozen whole-image-pretrained weights. Local evaluation changes the inference distribution; this is not a local-pretraining ablation.

| Branch | Global Top-1 | Local80 Top-1 | Drop (pp) | Global kNN | Local80 kNN |
|---|---:|---:|---:|---:|---:|
| student | 52.92% | 51.54% | 1.38 | 42.42% | 38.88% |
| teacher | 52.86% | 51.50% | 1.36 | 42.36% | 39.04% |
