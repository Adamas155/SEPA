"""Label-side spatial tasks; no position metadata is an encoder/probe input.

``tile_orders[n, container]`` maps a randomized container to its canonical slot.
The mapping is used only to build targets and retrieval queries. Features remain
content-only, and every image contributes every nonself ordered pair.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np


DIRECTION_NAMES = ("down", "up", "right", "left")
_PAIRS = np.asarray(
    [(i, j) for i in range(9) for j in range(9) if i != j], dtype=np.int64
)
_PAIR_INDEX = np.full((9, 9), -1, dtype=np.int64)
_PAIR_INDEX[_PAIRS[:, 0], _PAIRS[:, 1]] = np.arange(72)


def ordered_pairs() -> np.ndarray:
    """Return all 72 ordered nonself pairs of *container*, not canonical, indices."""
    return _PAIRS.copy()


def _validated_tile_orders(tile_orders: np.ndarray) -> np.ndarray:
    orders = np.asarray(tile_orders)
    if orders.ndim != 2 or orders.shape[1] != 9 or orders.shape[0] == 0:
        raise ValueError("tile_orders must have shape (N, 9), with N > 0")
    if orders.dtype.kind not in "iu":
        raise ValueError("tile_orders must contain integer permutations")
    if not np.all(np.sort(orders, axis=1) == np.arange(9)):
        raise ValueError("each tile_orders row must be a permutation of 0 through 8")
    return orders.astype(np.int64, copy=False)


def adjacency_targets(tile_orders: np.ndarray) -> np.ndarray:
    """Return (N, 72, 2) bool targets: i immediately above/left of j.

    Canonical slots are row-major in a 3 by 3 grid. Nonadjacent tiles in the
    same row/column are negative, as are reverse directions and diagonals.
    """
    orders = _validated_tile_orders(tile_orders)
    first = orders[:, _PAIRS[:, 0]]
    second = orders[:, _PAIRS[:, 1]]
    first_row, first_col = first // 3, first % 3
    second_row, second_col = second // 3, second % 3
    vertical = (second_row == first_row + 1) & (second_col == first_col)
    horizontal = (second_col == first_col + 1) & (second_row == first_row)
    return np.stack((vertical, horizontal), axis=-1)


def _image_rng(image_id: str | int, seed: int) -> np.random.Generator:
    # A stable per-image generator makes designs independent of iteration
    # order, model, checkpoint, probe seed, and Python's randomized hash().
    payload = json.dumps(
        [seed, image_id], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.blake2b(
        payload, digest_size=32, person=b"SEPA-spatial-v1"
    ).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype="<u4"))


def make_design(image_ids: Sequence[str | int], seed: int = 0) -> dict:
    """Build a fixed, model-independent pair/query design from opaque image IDs.

    Tile containers, candidate lists, and query order are randomized. Query
    metadata stays on the evaluation side: a model sees only pair features.
    Each query has all eight other tiles, exactly one positive, and a channel
    selecting V or H. For up/left queries the candidate is the first tile in
    the ordered pair, so the same two-output head serves all four directions.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("design seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("design seed must be a nonnegative integer")
    if isinstance(image_ids, (str, bytes)):
        raise ValueError("image_ids must be a sequence of distinct opaque IDs")
    ids = list(image_ids)
    if not ids:
        raise ValueError("image_ids must not be empty")
    normalized_ids: list[str | int] = []
    for image_id in ids:
        if isinstance(image_id, str) and image_id:
            normalized_ids.append(image_id)
        elif isinstance(image_id, (int, np.integer)) and not isinstance(
            image_id, (bool, np.bool_)
        ):
            normalized_ids.append(int(image_id))
        else:
            raise ValueError("each image ID must be a nonempty string or integer")
    keys = [json.dumps(value, ensure_ascii=False) for value in normalized_ids]
    if len(set(keys)) != len(keys):
        raise ValueError("image_ids must be distinct")

    n_images = len(normalized_ids)
    tile_orders = np.empty((n_images, 9), dtype=np.int64)
    query_pairs = np.empty((n_images, 24, 8), dtype=np.int64)
    query_channels = np.empty((n_images, 24), dtype=np.int64)
    query_positive = np.empty((n_images, 24, 8), dtype=bool)
    query_directions = np.empty((n_images, 24), dtype=np.int64)
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for image_index, image_id in enumerate(normalized_ids):
        rng = _image_rng(image_id, seed)
        order = rng.permutation(9)
        tile_orders[image_index] = order
        queries = []
        for anchor in range(9):
            row, col = divmod(int(order[anchor]), 3)
            for direction, (dr, dc) in enumerate(offsets):
                neighbor_row, neighbor_col = row + dr, col + dc
                if not (0 <= neighbor_row < 3 and 0 <= neighbor_col < 3):
                    continue
                candidates = rng.permutation(np.delete(np.arange(9), anchor))
                reverse = direction in (1, 3)
                pair_ids = (
                    _PAIR_INDEX[candidates, anchor]
                    if reverse
                    else _PAIR_INDEX[anchor, candidates]
                )
                positive = order[candidates] == neighbor_row * 3 + neighbor_col
                queries.append((pair_ids, int(direction >= 2), positive, direction))
        if len(queries) != 24:
            raise RuntimeError("a 3 by 3 grid must produce exactly 24 valid queries")
        for query_index, source_index in enumerate(rng.permutation(24)):
            pair_ids, channel, positive, direction = queries[source_index]
            query_pairs[image_index, query_index] = pair_ids
            query_channels[image_index, query_index] = channel
            query_positive[image_index, query_index] = positive
            query_directions[image_index, query_index] = direction

    return {
        "tile_orders": tile_orders,
        "labels": adjacency_targets(tile_orders),
        "query_pairs": query_pairs,
        "query_channels": query_channels,
        "query_positive": query_positive,
        "query_directions": query_directions,
        "direction_names": list(DIRECTION_NAMES),
    }
