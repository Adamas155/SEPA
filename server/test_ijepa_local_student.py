"""Causal isolation, sparse-reference gradients, and real CUDA resume checks."""
import json
import math
import time

import ijepa_baseline as j
from train_ijepa_local_student import LocalStudent, build
from evaluate_ijepa_fov import LocalTiles
from sepa_plan_b.data import ImageTiles

torch = j.torch


def sparse_native_reference(encoder, images, masks):
    """Independent reference: patchify actual 80px tiles, then keep visible patches."""
    outputs = []
    for mask in masks:
        for image, indices in zip(images, mask):
            tile_ids = (indices // 15 // 5) * 3 + (indices % 15 // 5)
            chunks, places = [], []
            for tile in range(9):
                place = (tile_ids == tile).nonzero().flatten()
                if len(place) == 0:
                    continue
                r, c = divmod(tile, 3)
                native = image[None, :, r*80:(r+1)*80, c*80:(c+1)*80]
                original = indices[place]
                within = (original // 15 % 5) * 5 + (original % 15 % 5)
                x = encoder.patch_embed(native)[:, within] + encoder.pos_embed[:, original]
                for block in encoder.blocks:
                    x = block(x)
                chunks.append(encoder.norm(x)[0])
                places.append(place)
            outputs.append(torch.cat(chunks)[torch.cat(places).argsort()])
    return torch.stack(outputs)


def smoke(cfg, base, train, val, meta):
    out = j.ROOT / "server-logs/ijepa-local-smoke"
    out.mkdir(parents=True, exist_ok=True)
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    total = ipe * cfg["epochs"]
    original = j.build(cfg, ipe)
    student = LocalStudent(original[0])
    models = (student, *original[1:])
    batches = list(j.loader(train, cfg, stop=12))
    serial = next(iter(j.loader(train, {**cfg, "workers": 0}, start=1, stop=2)))
    assert torch.equal(serial[0], batches[1][0])
    for group in (2, 3):
        assert all(torch.equal(a, b) for a, b in zip(serial[group], batches[1][group]))
    x = batches[0][0][:2].cuda()
    masks = [m[:2].cuda() for m in batches[0][2]]
    # Also exercise multiple context masks, one-token tiles and completely empty tiles.
    irregular = torch.tensor([[0, 4, 5, 6, 17, 100, 224], [16, 0, 15, 1, 2, 3, 4]], device="cuda")
    masks_test = [irregular, irregular.flip(1)]
    global_out = original[0](x, masks)
    torch.testing.assert_close(student(x, masks, restrict=False), global_out, rtol=0, atol=0)
    actual = student(x, masks_test)
    reference = sparse_native_reference(original[0], x, masks_test)
    torch.testing.assert_close(actual, reference, rtol=5e-5, atol=5e-5)
    weight = torch.randn_like(actual) / actual.numel()
    params = [p for p in student.parameters() if p.requires_grad]
    gradients = torch.autograd.grad((actual * weight).sum(), params)
    reference_gradients = torch.autograd.grad((reference * weight).sum(), params)
    gradient_error = 0.0
    for a, b in zip(gradients, reference_gradients):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-6)
        gradient_error = max(gradient_error, (a-b).abs().max().item())
    forward_error = (actual-reference).abs().max().item()
    del actual, reference, global_out, gradients, reference_gradients, params
    with torch.no_grad():
        # All visible student outputs must be invariant to changes in hidden pixels.
        changed = x.clone()
        for b in range(len(x)):
            hidden = set(range(225)) - set(masks[0][b].tolist())
            for index in hidden:
                r, c = divmod(index, 15)
                changed[b, :, r*16:(r+1)*16, c*16:(c+1)*16] = 0
        torch.testing.assert_close(student(x, masks), student(changed, masks), rtol=0, atol=0)
        # Other tiles cannot influence a selected visible token in any encoder layer.
        isolation_mask = [irregular[:1]]
        changed = x[:1].clone()
        changed[:, :, 80:, :] = 0
        changed[:, :, :80, 80:] = 0
        a, b = student(x[:1], isolation_mask), student(changed, isolation_mask)
        within_tile_zero = (irregular[0] // 15 // 5 == 0) & (irregular[0] % 15 // 5 == 0)
        torch.testing.assert_close(a[:, within_tile_zero], b[:, within_tile_zero], rtol=0, atol=0)
        global_change = (models[2](x[:1])[:, :5] - models[2](changed)[:, :5]).abs().max().item()
        assert global_change > 1e-3
        sample = torch.stack([ImageTiles(val, base)[i]["canonical"] for i in (0, 51)]).cuda()
        dense_local = student(j.join_tiles(sample))
        fast_local = LocalTiles(original[0]).cuda().eval().encode(sample)
        torch.testing.assert_close(dense_local, fast_local, rtol=5e-5, atol=5e-5)
        eval_error = (dense_local-fast_local).abs().max().item()
    del original, x, changed, sample, dense_local, fast_local, a, b
    # Reset initialization; the production build must match the untouched baseline exactly.
    del student, models
    torch.cuda.empty_cache()
    models = build(cfg, ipe)
    baseline = j.build(cfg, ipe)
    assert all(torch.equal(v, baseline[i].state_dict()[k]) for i in range(3)
               for k, v in models[i].state_dict().items())
    assert models[3].state_dict() == baseline[3].state_dict()
    del baseline
    initial = models[0].patch_embed.proj.weight.detach().clone()
    rows, times = [], []
    for batch in batches[:2]:
        torch.cuda.synchronize()
        started = time.monotonic()
        rows.append(j.update(models, batch, cfg, total))
        torch.cuda.synchronize()
        times.append(time.monotonic()-started)
        if batch[4] == 0:
            j.save(out / "resume.pt", models, 1, meta, 0)
    assert not torch.equal(initial, models[0].patch_embed.proj.weight)
    assert all(p.grad is None and not p.requires_grad for p in models[2].parameters())
    expected = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models[:3]]
    j.restore(out / "resume.pt", models, meta)
    replay = j.update(models, serial, cfg, total)
    assert replay == rows[1], "Resumed metrics differ"
    assert all(torch.equal(v.cpu(), state[k]) for m, state in zip(models[:3], expected)
               for k, v in m.state_dict().items()), "Resumed model weights differ"
    del expected
    for batch in batches[2:]:
        torch.cuda.synchronize()
        started = time.monotonic()
        rows.append(j.update(models, batch, cfg, total))
        torch.cuda.synchronize()
        times.append(time.monotonic()-started)
    last = next(iter(j.loader(train, cfg, start=ipe-1, stop=ipe)))
    assert last[0].shape[0] == 28
    short_row = j.update(models, last, cfg, total)
    assert short_row["batch_size"] == 28
    for schedule in models[4:]:
        schedule._step = total - 1
    assert abs(models[4].step()-cfg["final_lr"]) < 1e-12
    assert abs(models[5].step()-cfg["final_weight_decay"]) < 1e-12
    receipt = {"status": "passed", "metadata": meta, "utc": j.utc(),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "checks": ["identical_initial_parameters_and_optimizer", "global_forward_bitwise_equal",
                   "sparse_native80_forward_and_parameter_gradients", "empty_and_single_token_tiles",
                   "multiple_context_masks_ordering", "hidden_pixels_do_not_leak",
                   "other_tiles_cannot_influence_local_outputs", "teacher_global_negative_control",
                   "fast_local_evaluation_matches_training_forward", "worker_independent_data",
                   "encoder_updated_and_teacher_no_grad", "bf16_finite_optimizer_steps",
                   "bitwise_resume_metrics_and_all_model_weights", "real_last_batch_28", "schedule_endpoints"],
        "native80_forward_max_error": forward_error, "parameter_gradient_max_error": gradient_error,
        "eval_adapter_max_error": eval_error, "global_control_change": global_change,
        "rows": rows, "last_batch_row": short_row, "step_seconds": times,
        "steady_step_seconds_mean": sum(times[2:])/len(times[2:]),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    j.write_json(j.ROOT / "server-logs/ijepa_local_student_smoke.json", receipt)
    (out / "resume.pt").unlink()
    print(json.dumps(receipt, indent=2), flush=True)
