import torch
import torch.nn.functional as F

from .numerics import require_finite


def latent_loss(prediction, target):
    """L2 normalize both vectors, SUM features, MEAN batch and masked slots."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 2:
        raise ValueError("Latent loss requires matching Bx2xD tensors")
    p = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    t = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
    return (p-t).square().sum(-1).mean()


def pair_targets(canonical_ids):
    r, c = canonical_ids // 3, canonical_ids % 3
    dr, dc = r[:, None, :] - r[:, :, None], c[:, None, :] - c[:, :, None]
    valid = ~torch.eye(canonical_ids.shape[1], dtype=torch.bool, device=canonical_ids.device)[None]
    valid = valid.expand(len(canonical_ids), -1, -1)
    adjacent = (dr.abs() + dc.abs()) == 1  # four-neighbor undirected adjacency
    # Eight compass sectors: only sign matters for non-immediate offsets.
    code = (dr.sign()+1)*3 + dc.sign()+1
    directions = code - (code > 4).long()
    return adjacent, directions, valid


def relation_loss(logits, canonical_ids, directed=False):
    adjacent, direction, valid = pair_targets(canonical_ids)
    if directed:
        return F.cross_entropy(logits.float()[valid], direction[valid])
    labels = adjacent[valid].float()
    positives = labels.sum()
    weight = (labels.numel() - positives) / positives.clamp_min(1)
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float()[valid], labels, pos_weight=weight)


def _image_std(x, slots, distinct_rows):
    """Compare distinct images at the same canonical slot, never shuffled rows."""
    if slots is not None and slots.shape != x.shape[:2]:
        raise ValueError("Diagnostic slot labels must match BxV features")
    if len(distinct_rows) < 2:
        return None, 0
    rows = torch.tensor(distinct_rows, device=x.device)
    x = x[rows]
    if slots is None:
        return x.std(0, correction=0).mean().item(), x.shape[1]
    slots = slots.to(x.device)[rows]
    values = []
    for slot in range(9):
        selected = x[slots == slot]
        if len(selected) >= 2:
            values.append(selected.std(0, correction=0).mean())
    return (torch.stack(values).mean().item(), len(values)) if values else (None, 0)


@torch.no_grad()
def diagnostics(prediction, target, encoded, *, canonical_ids=None, query_slots=None, sample_ids=None,
                collapse_ratio=0.05, minimum_std=1e-4, minimum_image_std=1e-4):
    """Heuristic warnings, not a guarantee of healthy or full-rank representations.

    If slot labels are omitted, the caller must supply position-aligned tensors.
    Duplicate image identities count once. A batch without enough comparable
    images has an unknown (null) flag unless an available check detects collapse.
    """
    prediction, target, encoded = [x.detach().float() for x in (prediction, target, encoded)]
    require_finite("collapse diagnostics", prediction=prediction, target=target, encoded=encoded)
    if any(x.ndim != 3 or len(x) != len(encoded) for x in (prediction, target, encoded)):
        raise ValueError("Collapse diagnostics require matching batch dimensions in BxVxD tensors")
    if sample_ids is None:
        rows = list(range(len(encoded)))
    else:
        if len(sample_ids) != len(encoded):
            raise ValueError("Diagnostic image identities must match batch size")
        rows, seen = [], set()
        for i, identity in enumerate(sample_ids):
            if identity not in seen:
                seen.add(identity)
                rows.append(i)
    def std(x):
        return x.flatten(0, 1).std(0, correction=0).mean().item()
    p, t, h = std(prediction), std(target), std(encoded)
    encoder_image_std, encoder_slots = _image_std(encoded, canonical_ids, rows)
    target_image_std, target_slots = _image_std(target, query_slots, rows)
    ratio = p / max(t, 1e-12)
    reasons = [f"low_{name}_std" for name, value in (("prediction", p), ("target", t), ("encoder", h))
               if value < minimum_std]
    if ratio < collapse_ratio:
        reasons.append("low_std_ratio")
    for name, value in (("encoder", encoder_image_std), ("target", target_image_std)):
        if value is not None and value < minimum_image_std:
            reasons.append(f"low_{name}_image_std")
    complete = encoder_image_std is not None and target_image_std is not None
    return {"prediction_std": p, "target_std": t, "encoder_std": h,
            "std_ratio": ratio, "encoder_image_std": encoder_image_std, "target_image_std": target_image_std,
            "encoder_image_std_slots": encoder_slots, "target_image_std_slots": target_slots,
            "distinct_images": len(rows), "collapse_checks_complete": complete,
            "collapse_reasons": reasons, "collapse_flag": True if reasons else False if complete else None}
