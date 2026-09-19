"""Fixed-image diagnostics; values are observations, not scientific pass thresholds."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def position_sensitivity(model, canonical_tiles):
    """Hold content, visible slots and queries fixed; rebind six noncenter tiles.

    Mask corners 0/8 and cyclically move all six visible noncenter contents.
    The same reference image/augmentation is used at each logged training step.
    """
    was_training = model.training
    model.eval()
    try:
        device = canonical_tiles.device
        slots = torch.arange(1, 8, device=device)[None].expand(len(canonical_tiles), -1)
        queries = torch.tensor([[0, 8]], device=device).expand(len(canonical_tiles), -1)
        order = torch.tensor([6, 0, 1, 3, 2, 4, 5], device=device)
        visible = canonical_tiles[:, 1:8]
        baseline = model(visible, slots, queries)[0].float()
        changed = model(visible[:, order], slots, queries)[0].float()
        return {"position_delta_rms": (baseline-changed).square().mean().sqrt().item(),
                "position_delta_normalized": (F.normalize(baseline, dim=-1)-F.normalize(changed, dim=-1))
                                             .square().sum(-1).mean().sqrt().item()}
    finally:
        model.train(was_training)
