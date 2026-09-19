"""Causal and resumability checks for the paired SEPA V2 screen."""
from __future__ import annotations

import copy
import json
import math
import time

import ijepa_baseline as j
import train_ijepa_v2 as v

torch = j.torch


def clone_states(models):
    return [{k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            for model in models[:3]]


def assert_states(models, states):
    assert all(torch.equal(x.cpu(), state[k])
               for model, state in zip(models[:3], states)
               for k, x in model.state_dict().items())


def assert_nested_equal(a, b):
    assert type(a) is type(b)
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def reference_local(encoder, images, source):
    result = []
    tile_indices = v.tile_patch_indices(images.device)
    local_pe = encoder.pos_embed[:, tile_indices[0]]
    for row in range(len(images)):
        groups = []
        for indices in source[row].reshape(7, 25):
            tile = (indices[0] // 15 // 5) * 3 + (indices[0] % 15 // 5)
            rr, cc = divmod(int(tile), 3)
            x = encoder.patch_embed(images[row:row+1, :, rr*80:(rr+1)*80, cc*80:(cc+1)*80])
            x = x + local_pe
            for block in encoder.blocks:
                x = block(x)
            groups.append(encoder.norm(x)[0])
        result.append(torch.cat(groups))
    return torch.stack(result)


def smoke():
    cfg, spec, base, train, val, shared = v.context()
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    assert ipe == 1980 and ipe * spec["screen_epochs"] == 49500
    assert sum(step % 4 == 3 for step in range(49500)) == 12375
    assert spec["global_batch_fraction"] == 0.75 and spec["local_batch_fraction"] == 0.25

    # Exactly reproduce the untouched global update, including every state.
    baseline = j.build(cfg, ipe)
    candidate = v.build(cfg, ipe)
    batch0 = next(iter(j.loader(train, {**cfg, "workers": 0}, start=0, stop=1)))
    original_row = j.update(baseline, batch0, cfg, ipe*cfg["epochs"])
    v2_row = v.update(candidate, batch0, cfg, spec, ipe*cfg["epochs"], 3)
    assert v2_row["branch"] == "global"
    for key in ("step", "loss", "lr", "weight_decay", "ema", "grad_norm", "batch_size",
                "context_patches", "mask_resamples", "target_image_std", "batch_mean_target_loss"):
        assert v2_row[key] == original_row[key], key
    assert v2_row["target_patches"] == original_row["target_patches_per_block"]
    assert_states(candidate, clone_states(baseline))
    assert_nested_equal(candidate[3].state_dict(), baseline[3].state_dict())
    del baseline, candidate
    torch.cuda.empty_cache()

    models = v.build(cfg, ipe)
    batch3 = next(iter(j.loader(train, {**cfg, "workers": 0}, start=3, stop=4)))
    images = batch3[0][:3].cuda()
    source0, observed0, target0, moved0, mapping0 = v.layouts(3, 3, 0, images.device)
    source3, observed3, target3, moved3, mapping3 = v.layouts(3, 3, 3, images.device)
    assert torch.equal(target0, target3), "Masking must be paired across k"
    assert torch.all(moved0 == 0) and torch.any(moved3 > 0)
    patches = v.tile_patch_indices(images.device)
    for row in range(3):
        hidden_indices = target0[row].reshape(2, 25)[:, 0]
        hidden = set(((hidden_indices // 15 // 5) * 3 + (hidden_indices % 15 // 5)).tolist())
        assert 4 not in hidden and len(hidden) == 2
        visible_slots = observed3[row].reshape(7, 25)[:, 0]
        visible_slots = ((visible_slots // 15 // 5) * 3 + (visible_slots % 15 // 5)).tolist()
        source_tiles = source3[row].reshape(7, 25)[:, 0]
        source_tiles = ((source_tiles // 15 // 5) * 3 + (source_tiles % 15 // 5)).tolist()
        for piece, slot in zip(source_tiles, visible_slots):
            assert mapping3[row, piece].item() == slot
        assert mapping3[row, 4].item() == 4
        assert all(mapping3[row, h].item() == h for h in hidden)

    actual = v.local_encode(models[0], images, source3)
    reference = reference_local(models[0], images, source3)
    torch.testing.assert_close(actual, reference, rtol=4e-5, atol=4e-5)
    forward_error = (actual-reference).abs().max().item()

    # A canonical piece has the same encoding regardless of its observed slot.
    encoded0 = v.local_encode(models[0], images, source0).reshape(3, 7, 25, 384)
    encoded3 = actual.reshape(3, 7, 25, 384)
    for row in range(3):
        pieces0 = (source0[row].reshape(7, 25)[:, 0] // 15 // 5) * 3 + (source0[row].reshape(7, 25)[:, 0] % 15 // 5)
        pieces3 = (source3[row].reshape(7, 25)[:, 0] // 15 // 5) * 3 + (source3[row].reshape(7, 25)[:, 0] % 15 // 5)
        for piece in pieces0.tolist():
            a = (pieces0 == piece).nonzero().item()
            b = (pieces3 == piece).nonzero().item()
            torch.testing.assert_close(encoded0[row, a], encoded3[row, b], rtol=4e-5, atol=4e-5)

    # Hidden pixels do not enter the local encoder.
    changed = images.clone()
    for row in range(3):
        hidden_indices = target3[row].reshape(2, 25)[:, 0]
        hidden_tiles = (hidden_indices // 15 // 5) * 3 + (hidden_indices % 15 // 5)
        for tile in hidden_tiles.tolist():
            rr, cc = divmod(tile, 3)
            changed[row, :, rr*80:(rr+1)*80, cc*80:(cc+1)*80] = torch.randn_like(
                changed[row, :, rr*80:(rr+1)*80, cc*80:(cc+1)*80])
    torch.testing.assert_close(actual, v.local_encode(models[0], changed, source3), rtol=0, atol=0)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction0 = models[1](encoded0.reshape(3, 175, 384), [observed0], [target0])
        prediction3 = models[1](encoded3.reshape(3, 175, 384), [observed3], [target3])
    position_delta = (prediction0-prediction3).float().abs().mean().item()
    assert position_delta > 1e-4, "Predictor did not respond to observed position corruption"
    assert prediction3.shape == (3, 50, 384)
    del actual, reference, encoded0, encoded3, prediction0, prediction3, images, changed
    torch.cuda.empty_cache()

    # Real BF16 local update and exact checkpoint replay.
    initial = models[0].patch_embed.proj.weight.detach().clone()
    before_teacher = models[2].patch_embed.proj.weight.detach().clone()
    started = time.monotonic()
    row = v.update(models, batch3, cfg, spec, ipe*cfg["epochs"], 3)
    torch.cuda.synchronize()
    local_seconds = time.monotonic()-started
    assert row["branch"] == "local" and row["target_patches"] == 50 and row["context_patches"] == 175
    assert not torch.equal(initial, models[0].patch_embed.proj.weight)
    assert not torch.equal(before_teacher, models[2].patch_embed.proj.weight)
    assert all(p.grad is None and not p.requires_grad for p in models[2].parameters())
    del models
    torch.cuda.empty_cache()
    path = j.ROOT / "server-logs/ijepa-v2-smoke-resume.pt"
    # Save pre-update state, restore, and replay the same local batch.
    fresh = v.build(cfg, ipe)
    j.save(path, fresh, 0, v.context("k3")[-2], 0.0)
    v.update(fresh, batch0, cfg, spec, ipe*cfg["epochs"], 3)
    b1 = next(iter(j.loader(train, {**cfg, "workers": 0}, start=1, stop=2)))
    b2 = next(iter(j.loader(train, {**cfg, "workers": 0}, start=2, stop=3)))
    v.update(fresh, b1, cfg, spec, ipe*cfg["epochs"], 3)
    v.update(fresh, b2, cfg, spec, ipe*cfg["epochs"], 3)
    j.save(path, fresh, 3, v.context("k3")[-2], 0.0)
    replay_expected = v.update(fresh, batch3, cfg, spec, ipe*cfg["epochs"], 3)
    replay_states = clone_states(fresh)
    j.restore(path, fresh, v.context("k3")[-2])
    replay_actual = v.update(fresh, batch3, cfg, spec, ipe*cfg["epochs"], 3)
    assert replay_actual == replay_expected
    assert_states(fresh, replay_states)
    replay_optimizer = copy.deepcopy(fresh[3].state_dict())
    j.restore(path, fresh, v.context("k3")[-2])
    v.update(fresh, batch3, cfg, spec, ipe*cfg["epochs"], 3)
    assert_nested_equal(fresh[3].state_dict(), replay_optimizer)
    path.unlink()

    del fresh, replay_states, replay_optimizer
    torch.cuda.empty_cache()

    last = next(iter(j.loader(train, {**cfg, "workers": 0}, start=49499, stop=49500)))
    assert last[0].shape[0] == 28 and last[4] == 49499 and last[4] % 4 == 3
    last_models = v.build(cfg, ipe)
    last_models[4]._step = last_models[5]._step = 49499
    last_row = v.update(last_models, last, cfg, spec, ipe*cfg["epochs"], 0)
    assert last_row["batch_size"] == 28 and last_row["branch"] == "local"

    receipt = {"status": "passed", "shared": shared, "utc": j.utc(),
        "checks": ["global_update_and_all_states_bitwise_baseline", "paired_k0_k3_masks",
                   "valid_two_hidden_noncenter_tiles_and_fixed_anchor", "mapping_inverse_and_fixed_hidden_slots",
                   "native80_reference", "encoder_has_no_observed_or_canonical_slot_identity",
                   "hidden_pixels_absent", "predictor_responds_to_observed_position_corruption",
                   "50_patch_targets", "real_bf16_local_optimizer_update", "teacher_ema_no_gradient",
                   "bitwise_local_checkpoint_replay", "exact_75_25_branch_ratio",
                   "screen_continues_original_100epoch_schedules", "real_final_batch_28"],
        "native80_forward_max_error": forward_error, "predictor_position_delta": position_delta,
        "local_step_seconds": local_seconds, "last_batch_row": last_row,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    j.write_json(j.ROOT / "server-logs/ijepa_v2_smoke.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
