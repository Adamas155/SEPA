"""Natural-prevalence adjacency and same-image retrieval measurements.

Average precision groups equal scores as one threshold, matching the standard
noninterpolated AP convention. AUROC awards half credit to positive/negative
ties. Retrieval uses the analytic expectation of uniformly random tie breaking;
candidate order never decides the result. No confidence interval treats pairs
or probe seeds as independent pretrained models.
"""

from __future__ import annotations

import numpy as np

from .tasks import DIRECTION_NAMES, adjacency_targets, ordered_pairs


THRESHOLD_TIE_RULE = (
    "Maximize dev balanced accuracy over observed score thresholds; among "
    "maximizers choose the largest threshold. Predict positive for score >= threshold."
)
RETRIEVAL_TIE_RULE = (
    "Analytic expectation under uniform random ordering within each exact score "
    "tie: fractional Recall@1 and mean reciprocal rank over the tied positions."
)


def _real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scores(value) -> np.ndarray:
    array = _real_array(value, "scores")
    if array.ndim != 3 or array.shape[1:] != (72, 2) or array.shape[0] == 0:
        raise ValueError("scores must have shape (N, 72, 2), with N > 0")
    return array


def _binary_array(value, shape: tuple, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be binary and have shape {shape}")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only zero/one or bool values")
    return array.astype(bool, copy=False)


def _labels(value, shape: tuple) -> np.ndarray:
    array = _binary_array(value, shape, "labels")
    if not np.all(array.sum(axis=1) == 6):
        raise ValueError("each image must retain all 72 pairs and six positives per direction")
    return array


def _grouped_counts(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), len(order) - 1]
    true_positive = np.cumsum(labels[order], dtype=np.int64)[ends]
    false_positive = ends + 1 - true_positive
    return ranked_scores[ends], true_positive, false_positive


def _ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    _, true_positive, false_positive = _grouped_counts(scores, labels)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    previous_tp = np.r_[0, true_positive[:-1]]
    previous_fp = np.r_[0, false_positive[:-1]]
    ap = np.sum(
        (true_positive - previous_tp)
        / n_positive
        * (true_positive / (true_positive + false_positive))
    )
    # Trapezoids at distinct score groups give midrank treatment of ROC ties.
    auroc = np.sum(
        (false_positive - previous_fp)
        / n_negative
        * (true_positive + previous_tp)
        / (2 * n_positive)
    )
    return float(ap), float(auroc)


def select_thresholds(dev_scores, dev_labels) -> list[float]:
    """Choose V/H thresholds exclusively from supplied dev examples.

    The predeclared tie rule is ``THRESHOLD_TIE_RULE``. Observed thresholds
    include predicting everything positive, with balanced accuracy 0.5.
    Predicting nothing also scores 0.5 and cannot improve the maximum, so no
    nonfinite sentinel threshold is needed. Integer counts make ties exact.
    """
    scores = _scores(dev_scores)
    labels = _labels(dev_labels, scores.shape)
    thresholds = []
    for channel in range(2):
        flat_labels = labels[..., channel].ravel()
        values, true_positive, false_positive = _grouped_counts(
            scores[..., channel].ravel(), flat_labels
        )
        n_positive = int(flat_labels.sum())
        n_negative = len(flat_labels) - n_positive
        quality = true_positive * n_negative - false_positive * n_positive
        # Values descend, so the first maximum is the largest tied threshold.
        thresholds.append(float(values[int(np.argmax(quality))]))
    return thresholds


def adjacency_metrics(scores, labels, thresholds=(0.0, 0.0)) -> dict:
    """Measure both binary relations on all pairs at their natural prevalence.

    Default zero-logit thresholds permit AP-only dev monitoring; a final test
    evaluation must explicitly pass thresholds selected on probe-dev.
    """
    scores = _scores(scores)
    labels = _labels(labels, scores.shape)
    thresholds = _real_array(thresholds, "thresholds")
    if thresholds.shape != (2,):
        raise ValueError("thresholds must contain exactly one value for V and H")
    result = {}
    for channel, name in enumerate(("V", "H")):
        flat_labels = labels[..., channel].ravel()
        flat_scores = scores[..., channel].ravel()
        ap, auroc = _ranking_metrics(flat_scores, flat_labels)
        predicted = flat_scores >= thresholds[channel]
        sensitivity = predicted[flat_labels].mean()
        specificity = (~predicted[~flat_labels]).mean()
        result[name] = {
            "ap": ap,
            "auroc": auroc,
            "balanced_accuracy": float((sensitivity + specificity) / 2),
        }
    result["macro"] = {
        name: float((result["V"][name] + result["H"][name]) / 2)
        for name in ("ap", "auroc", "balanced_accuracy")
    }
    result["positive_prevalence"] = labels.mean(axis=(0, 1)).tolist()
    return result


def _index_array(value, shape: tuple, bound: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    if np.any(array < 0) or np.any(array >= bound):
        raise ValueError(f"{name} values must be between 0 and {bound - 1}")
    return array.astype(np.int64, copy=False)


def _validate_retrieval_design(design: dict, n_images: int):
    required = {
        "tile_orders", "labels", "query_pairs", "query_channels",
        "query_positive", "query_directions", "direction_names",
    }
    if not isinstance(design, dict) or not required.issubset(design):
        raise ValueError("design must contain all make_design fields")
    if list(design["direction_names"]) != list(DIRECTION_NAMES):
        raise ValueError("direction_names must be down, up, right, left in that order")
    pair_ids = _index_array(design["query_pairs"], (n_images, 24, 8), 72, "query_pairs")
    channels = _index_array(design["query_channels"], (n_images, 24), 2, "query_channels")
    directions = _index_array(
        design["query_directions"], (n_images, 24), 4, "query_directions"
    )
    positive = _binary_array(
        design["query_positive"], (n_images, 24, 8), "query_positive"
    )
    if not np.all(positive.sum(axis=-1) == 1):
        raise ValueError("each query must contain exactly one positive among eight candidates")
    if not np.all(channels == (directions >= 2)):
        raise ValueError("down/up must use V and right/left must use H")
    for direction in range(4):
        if not np.all((directions == direction).sum(axis=1) == 6):
            raise ValueError("each image must contain six valid queries per direction")

    pairs = ordered_pairs()[pair_ids]
    reverse = (directions == 1) | (directions == 3)
    anchors = np.where(reverse[..., None], pairs[..., 1], pairs[..., 0])
    candidates = np.where(reverse[..., None], pairs[..., 0], pairs[..., 1])
    if not np.all(anchors == anchors[..., :1]):
        raise ValueError("all candidates in a query must share its direction's anchor")
    if not np.all(np.diff(np.sort(candidates, axis=-1), axis=-1) > 0):
        raise ValueError("each query must contain eight distinct nonself candidates")
    query_keys = anchors[..., 0] * 4 + directions
    if not np.all(np.diff(np.sort(query_keys, axis=1), axis=1) > 0):
        raise ValueError("anchor-direction queries must not be repeated")

    labels = _labels(design["labels"], (n_images, 72, 2))
    canonical_labels = adjacency_targets(design["tile_orders"])
    if canonical_labels.shape != labels.shape or not np.array_equal(labels, canonical_labels):
        raise ValueError("design labels must match its container-to-canonical mapping")
    truth = labels[np.arange(n_images)[:, None, None], pair_ids, channels[..., None]]
    if not np.array_equal(truth, positive):
        raise ValueError("query positives must agree with ordered-pair V/H labels")
    return pair_ids, channels, directions, positive


def retrieval_metrics(scores, design: dict) -> dict:
    """Evaluate the same V/H head as four neighbor retrieval directions."""
    scores = _scores(scores)
    n_images = scores.shape[0]
    pair_ids, channels, directions, positive = _validate_retrieval_design(design, n_images)
    values = scores[np.arange(n_images)[:, None, None], pair_ids, channels[..., None]]
    positive_score = np.take_along_axis(values, positive.argmax(axis=-1)[..., None], axis=-1)
    greater = (values > positive_score).sum(axis=-1)
    tied = (values == positive_score).sum(axis=-1)
    recall = np.where(greater == 0, 1.0 / tied, 0.0)
    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, 9))]
    reciprocal_rank = (harmonic[greater + tied] - harmonic[greater]) / tied
    result = {}
    for direction, name in enumerate(DIRECTION_NAMES):
        selected = directions == direction
        result[name] = {
            "recall_at_1": float(recall[selected].mean()),
            "mrr": float(reciprocal_rank[selected].mean()),
            "n_queries": int(selected.sum()),
        }
    result["macro"] = {
        metric: float(np.mean([result[name][metric] for name in DIRECTION_NAMES]))
        for metric in ("recall_at_1", "mrr")
    }
    result["tie_rule"] = RETRIEVAL_TIE_RULE
    return result


def cosine_pair_scores(features) -> np.ndarray:
    """Content cosine baseline, identical for V/H and symmetric under reversal.

    Zero vectors have cosine zero. Max-absolute-value scaling avoids overflow
    when normalizing finite inputs. There is no fitted relation head or
    directional signal in this baseline.
    """
    features = _real_array(features, "features")
    if (
        features.ndim != 3
        or features.shape[0] == 0
        or features.shape[1] != 9
        or features.shape[2] == 0
    ):
        raise ValueError("features must have shape (N, 9, D), with N > 0 and D > 0")
    scale = np.max(np.abs(features), axis=-1, keepdims=True)
    scaled = features / np.where(scale > 0, scale, 1)
    norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
    normalized = scaled / np.where(norm > 0, norm, 1)
    similarities = np.clip(normalized @ normalized.swapaxes(-1, -2), -1.0, 1.0)
    pairs = ordered_pairs()
    selected = similarities[:, pairs[:, 0], pairs[:, 1]]
    return np.repeat(selected[..., None], 2, axis=-1)
