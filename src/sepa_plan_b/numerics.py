"""Fail closed before nonfinite features become apparently valid probe scores."""
import torch


def require_finite(context, **values):
    for name, value in values.items():
        if not torch.isfinite(value.detach()).all():
            raise FloatingPointError(f"{context}: {name} contains NaN or infinity")


def checked_probe_step(loss, head, optimizer, context):
    require_finite(context, loss=loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    require_finite(f"{context} gradients", **{name: p.grad for name, p in head.named_parameters() if p.grad is not None})
    optimizer.step()
    require_finite(f"{context} parameters after update", **dict(head.named_parameters()))


def feature_statistics(features, context):
    require_finite(context, features=features)
    mean = features.mean(0)
    std = features.std(0, correction=0).clamp_min(1e-6)
    require_finite(f"{context} statistics", mean=mean, std=std)
    return mean, std
