"""Stateless sample streams. k/arm identity never enters shared streams."""
import hashlib
import json
import random

import torch


def seed_for(seed: int, stream: str, *parts) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw = json.dumps(["sepa-plan-b/v1", seed, stream, *parts], separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (2**63 - 1)


def local_rng(seed, stream, *parts):
    return random.Random(seed_for(seed, stream, *parts))


def generator(seed, stream, *parts):
    return torch.Generator().manual_seed(seed_for(seed, stream, *parts))
