from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import Geometry
from .rng import local_rng

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VALID_K = (0, 2, 3, 4, 5, 6)


def split_tiles(image: torch.Tensor, geometry: Geometry, *, normalize=True):
    """CHW RGB -> 9 canonical tiles. Normalization precedes local padding."""
    if image.shape != (3, geometry.image_size, geometry.image_size):
        raise ValueError(f"Expected 3x{geometry.image_size}x{geometry.image_size}, got {tuple(image.shape)}")
    if geometry.mode == "pad74":
        image = image[:, 1:223, 1:223]
    side = geometry.real_tile_size
    tiles = image.reshape(3, 3, side, 3, side).permute(1, 3, 0, 2, 4).reshape(9, 3, side, side)
    if normalize:
        mean = image.new_tensor(MEAN)[None, :, None, None]
        std = image.new_tensor(STD)[None, :, None, None]
        tiles = (tiles - mean) / std
    if geometry.mode == "pad74":
        tiles = F.pad(tiles, (3, 3, 3, 3), mode=geometry.padding)
    return tiles.contiguous()


@dataclass
class Layout:
    # Sorted observed slots index model inputs. Only loss/data code sees identity.
    visible_slots: torch.Tensor
    canonical_ids: torch.Tensor
    query_slots: torch.Tensor
    piece_to_slot: torch.Tensor
    moved_count: int


def layout_from_mapping(masked, piece_to_slot):
    masked = sorted(int(x) for x in masked)
    mapping = list(map(int, piece_to_slot))
    if len(masked) != 2 or len(set(masked)) != 2 or 4 in masked or not set(masked) <= set(range(9)):
        raise ValueError("Exactly two distinct noncenter canonical slots must be masked")
    if sorted(mapping) != list(range(9)) or mapping[4] != 4 or any(mapping[s] != s for s in masked):
        raise ValueError("Mapping must be a permutation fixing center and masked slots")
    observed = sorted(set(range(9)) - set(masked))
    inverse = [mapping.index(slot) for slot in observed]
    return Layout(torch.tensor(observed), torch.tensor(inverse), torch.tensor(masked),
                  torch.tensor(mapping), sum(i != x for i, x in enumerate(mapping)))


def sample_layout(seed: int, epoch: int, sample_id: str, k: int):
    if k not in VALID_K:
        raise ValueError(f"k must be one of {VALID_K}")
    ring = [i for i in range(9) if i != 4]
    masked = local_rng(seed, "mask", epoch, sample_id).sample(ring, 2)
    available = [i for i in ring if i not in masked]
    rng = local_rng(seed, "permutation", epoch, sample_id, k)
    chosen = rng.sample(available, k)
    targets = chosen.copy()
    rng.shuffle(targets)  # uniform permutation, including fixed points
    mapping = list(range(9))
    for piece, slot in zip(chosen, targets):
        mapping[piece] = slot
    return layout_from_mapping(masked, mapping)
