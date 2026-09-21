"""Synthetic encoder fixtures test implementation only, never trained-model quality."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.spatial_readout.encoders import (  # noqa: E402
    CheckpointError,
    EXPERIMENT_ROOT,
    FIXED_PE_INDICES,
    FrozenLocalEncoder,
    TRAIN_FINGERPRINT,
    UPSTREAM_ARCHIVE_HASH,
    UPSTREAM_COMMIT,
    _official_constructor,
    file_digest,
    load_encoder,
    prepare_upstream,
    state_digest,
)
from sepa_plan_b.config import Config, digest  # noqa: E402
from sepa_plan_b.engine import source_hash  # noqa: E402
from sepa_plan_b.model import TileEncoder  # noqa: E402


def identify(metadata, key="run_id"):
    metadata = copy.deepcopy(metadata)
    metadata.pop(key, None)
    metadata[key] = digest(metadata)
    return metadata


class LocalReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A shallow random unit fixture is intentionally NOT a scientific model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            model = TileEncoder(384, 1, 6, 80)
            cls.tiles = torch.randn(9, 3, 80, 80)
        cls.encoder = FrozenLocalEncoder(model, "v1", {"unit_fixture_only": True})

    def test_nine_tiles_produce_25_tokens_and_384_features_each(self):
        self.assertEqual(self.encoder.tokens(self.tiles).shape, (9, 25, 384))
        self.assertEqual(self.encoder(self.tiles).shape, (9, 384))

    def test_no_coordinates_ids_or_container_metadata_in_interface(self):
        self.assertEqual(list(inspect.signature(self.encoder.forward).parameters), ["tiles"])
        self.assertEqual(list(inspect.signature(self.encoder.tokens).parameters), ["tiles"])
        with self.assertRaises(TypeError):
            self.encoder(self.tiles, canonical_ids=torch.arange(9))
        with self.assertRaises(ValueError):
            self.encoder({"pixels": self.tiles, "coordinates": [(0, 0)] * 9})
        with self.assertRaises(ValueError):
            self.encoder(self.tiles.reshape(1, 9, 3, 80, 80))

    def test_container_reordering_and_metadata_change_only_reorder_features(self):
        order = torch.tensor([8, 2, 5, 0, 7, 3, 1, 6, 4])
        original = [{"pixels": tile, "row": i // 3, "column": i % 3} for i, tile in enumerate(self.tiles)]
        moved = [{"pixels": original[i]["pixels"], "row": 20-j, "column": j+30} for j, i in enumerate(order)]
        first = self.encoder(torch.stack([entry["pixels"] for entry in original]))
        second = self.encoder(torch.stack([entry["pixels"] for entry in moved]))
        torch.testing.assert_close(second, first[order], rtol=1e-5, atol=1e-6)
        head = nn.Linear(768, 2).eval()
        inverse = order.argsort()
        # The exact same ordered pair has the same score after container movement.
        a = head(torch.cat((first[1], first[5])).clone())
        b = head(torch.cat((second[inverse[1]], second[inverse[5]])).clone())
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_other_tiles_cannot_change_the_same_tile_feature(self):
        original = self.encoder(self.tiles)
        changed = self.tiles.clone()
        changed[1:] = 100 * torch.randn_like(changed[1:])
        torch.testing.assert_close(self.encoder(changed)[0], original[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(self.encoder(self.tiles[:1])[0], original[0], rtol=1e-5, atol=1e-6)

    def test_eval_no_grad_and_state_unchanged_after_probe_fit(self):
        before = state_digest(self.encoder)
        self.encoder.train(True)
        self.assertFalse(self.encoder.training)
        self.assertTrue(all(not m.training for m in self.encoder.modules()))
        self.assertTrue(all(not p.requires_grad for p in self.encoder.parameters()))
        pixels = self.tiles.clone().requires_grad_()
        features = self.encoder(pixels).clone()
        self.assertFalse(features.requires_grad)
        head = nn.Linear(768, 2)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        pairs = torch.cat((features[:4], features[4:8]), dim=1)
        for _ in range(2):
            optimizer.zero_grad()
            head(pairs).square().mean().backward()
            optimizer.step()
        self.assertIsNone(pixels.grad)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))
        self.assertEqual(before, state_digest(self.encoder))


class CheckpointLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix=".encoder-unit-fixtures-", dir=EXPERIMENT_ROOT)
        cls.directory = Path(cls.tmp.name)
        cls.path = cls.directory / "synthetic-unit-fixture.pt"
        # Real deserialization is exercised once; malformed metadata tests reuse its
        # in-memory payload to avoid repeatedly writing 85 MB of synthetic weights.
        config = Config().to_dict()
        config["training"]["steps"] = 198000
        metadata = identify({
            "schema": 1, "config": config, "seed": 0, "k": 3, "method": "sepa",
            "train_fingerprint": TRAIN_FINGERPRINT, "val_fingerprint": "unit-fixture-only",
            "source_hash": source_hash(), "unit_fixture_only": True,
        })
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            cls.model = TileEncoder(384, 12, 6, 80)
        cls.checkpoint = {
            "schema": 1, "metadata": metadata, "step": 198000,
            "model": {"encoder." + k: v for k, v in cls.model.state_dict().items()},
        }
        cls.checkpoint["model"]["teacher.unused"] = torch.tensor(float("nan"))
        torch.save(cls.checkpoint, cls.path)
        cls.spec = {
            "name": "SYNTHETIC_UNIT_FIXTURE_NOT_PRETRAINED", "family": "v1", "group": "A",
            "path": str(cls.path), "epochs": 100, "pretraining_seed": 0, "k": 3,
            "steps_per_epoch": 1980,
        }

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.checkpoint
        cls.tmp.cleanup()

    def changed(self, **updates):
        checkpoint = dict(self.checkpoint)
        checkpoint.update(updates)
        return checkpoint

    def invoke_payload(self, payload, spec=None):
        with patch("sepa_plan_b.engine.load_checkpoint", return_value=payload):
            return load_encoder(spec or self.spec)

    def test_actual_v1_student_state_loads_strictly_without_teacher(self):
        result = load_encoder(self.spec)
        self.assertEqual(result.manifest["file_hash"], file_digest(self.path))
        self.assertEqual(result.manifest["state_hash"], state_digest(self.model))
        self.assertEqual(result.manifest["pretraining_seed"], 0)
        self.assertEqual(result.manifest["epochs"], 100)
        self.assertEqual(result.manifest["patch_tokens"], 25)
        self.assertEqual(result.manifest["branch"], "student")
        torch.testing.assert_close(result.encoder.patch_pe, self.model.patch_pe, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in result.parameters()))

    def test_missing_checkpoint_has_no_random_fallback(self):
        with self.assertRaises(FileNotFoundError):
            load_encoder({**self.spec, "path": str(self.directory / "missing.pt")})

    def test_rejects_wrong_group_epochs_seed_k_or_branch(self):
        for updates in (
            {"group": "B"}, {"epochs": 25}, {"pretraining_seed": 8},
            {"k": 2}, {"branch": "teacher"}, {"expected_file_hash": "incorrect"},
        ):
            with self.subTest(updates=updates), self.assertRaises(CheckpointError):
                load_encoder({**self.spec, **updates})

    def test_rejects_checkpoint_seed_k_source_and_data_mismatch(self):
        for updates in (
            {"seed": 5}, {"k": 0}, {"source_hash": "changed"},
            {"train_fingerprint": "unknown-dataset"}, {"method": "full"},
        ):
            metadata = identify({**self.checkpoint["metadata"], **updates})
            with self.subTest(updates=updates), self.assertRaises(CheckpointError):
                self.invoke_payload(self.changed(metadata=metadata))

    def test_step_and_configured_epoch_mismatch_fail_closed(self):
        for step in (49500, 197999, 0):
            with self.subTest(step=step), self.assertRaises(CheckpointError):
                self.invoke_payload(self.changed(step=step))
        with self.assertRaises(CheckpointError):
            self.invoke_payload(self.checkpoint, {**self.spec, "steps_per_epoch": 2000})

    def test_modified_identity_and_missing_metadata_fail_closed(self):
        metadata = {**self.checkpoint["metadata"], "seed": 9}
        with self.assertRaises(CheckpointError):
            self.invoke_payload(self.changed(metadata=metadata))
        with self.assertRaises(CheckpointError):
            self.invoke_payload({"schema": 1, "step": 198000})

    def test_strict_loading_rejects_missing_extra_or_modified_pe_weights(self):
        missing = dict(self.checkpoint["model"])
        missing.pop("encoder.norm.weight")
        extra = {**self.checkpoint["model"], "encoder.unexpected": torch.zeros(1)}
        pe = {**self.checkpoint["model"], "encoder.patch_pe": torch.zeros(25, 384)}
        for weights in (missing, extra, pe):
            with self.subTest(keys=len(weights)), self.assertRaises(CheckpointError):
                self.invoke_payload(self.changed(model=weights))


class OfficialEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix=".official-unit-fixtures-", dir=EXPERIMENT_ROOT)
        cls.directory = Path(cls.tmp.name)
        cls.upstream = prepare_upstream(ROOT / "server/ijepa-upstream-52c1ae9.tar", cls.directory / "upstream")
        # Self-contained synthetic metadata, not a dependency on an old report.
        cls.metadata = identify({
            "unit_fixture_only": True,
            "config": {"model": "vit_small", "image_size": 240, "patch_size": 16,
                       "seed": 0, "epochs": 100, "batch_size": 64,
                       "upstream_commit": UPSTREAM_COMMIT},
            "train_fingerprint": TRAIN_FINGERPRINT,
            "upstream": {"commit": UPSTREAM_COMMIT, "archive_sha256": UPSTREAM_ARCHIVE_HASH,
                         "python_files": {p.relative_to(cls.upstream).as_posix(): file_digest(p)
                                          for p in cls.upstream.rglob("*.py")}},
        })
        constructor = _official_constructor(cls.upstream, cls.metadata["upstream"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            cls.model = constructor(img_size=[240], patch_size=16)
        cls.path = cls.directory / "SYNTHETIC_UNIT_FIXTURE_NOT_PRETRAINED.pt"
        cls.path.write_bytes(b"unit payload injected explicitly by mock; not a scientific checkpoint")
        cls.checkpoint = {
            "metadata": cls.metadata, "step": 49500, "lr_step": 49500, "wd_step": 49500,
            "encoder": cls.model.state_dict(), "teacher": {"invalid": torch.tensor(float("nan"))},
        }
        cls.spec = {
            "name": "SYNTHETIC_UNIT_FIXTURE_NOT_PRETRAINED", "family": "ijepa", "group": "B",
            "path": str(cls.path), "epochs": 25, "pretraining_seed": 0, "steps_per_epoch": 1980,
        }

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.checkpoint
        cls.tmp.cleanup()

    def invoke(self, payload=None, spec=None):
        with patch("experiments.spatial_readout.encoders.torch.load", return_value=payload or self.checkpoint):
            return load_encoder(spec or self.spec, upstream_dir=self.upstream)

    def test_original_official_model_and_fixed_square_pe_window(self):
        result = self.invoke()
        self.assertEqual(list(FIXED_PE_INDICES), [r * 15 + c for r in range(5) for c in range(5)])
        self.assertNotEqual(list(FIXED_PE_INDICES), list(range(25)))
        self.assertEqual(result.manifest["pe_protocol"]["indices"], list(FIXED_PE_INDICES))
        self.assertTrue(result.manifest["pe_protocol"]["input_distribution_shift_for_global_ijepa"])
        tiles = torch.randn(3, 3, 80, 80)
        with torch.inference_mode():
            expected = self.model.patch_embed(tiles) + self.model.pos_embed[:, list(FIXED_PE_INDICES)]
            for block in self.model.blocks:
                expected = block(expected)
            expected = self.model.norm(expected)
        torch.testing.assert_close(result.tokens(tiles), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result(tiles.flip(0)), result(tiles).flip(0), rtol=1e-5, atol=1e-6)
        self.assertEqual(state_digest(self.model), result.manifest["state_hash"])

    def test_v2_nested_provenance_and_same_local_pe_are_checked(self):
        shared = identify({"unit_fixture_only": True, "experiment": "sepa_v2_25epoch_screen",
                           "baseline": self.metadata, "baseline_config": self.metadata["config"],
                           "v2_config": {"seed": 0, "screen_epochs": 25}}, "screen_id")
        metadata = identify({"shared": shared, "arm": "k3", "k": 3})
        payload = {**self.checkpoint, "metadata": metadata}
        spec = {**self.spec, "family": "v2", "k": 3}
        result = self.invoke(payload, spec)
        self.assertTrue(result.manifest["pe_protocol"]["matches_v2_training_local_branch"])
        self.assertEqual(result.manifest["pe_protocol"]["indices"], list(FIXED_PE_INDICES))
        bad = identify({**metadata, "k": 0})
        with self.assertRaises(CheckpointError):
            self.invoke({**payload, "metadata": bad}, spec)

    def test_missing_encoder_schedule_and_wrong_checkpoint_seed_are_rejected(self):
        metadata = identify({**self.metadata, "config": {**self.metadata["config"], "seed": 2}})
        for payload in (
            {**self.checkpoint, "encoder": None},
            {**self.checkpoint, "lr_step": 49501},
            {**self.checkpoint, "metadata": metadata},
        ):
            with self.subTest(keys=list(payload)), self.assertRaises(CheckpointError):
                self.invoke(payload)

    def test_upstream_import_aliases_are_restored(self):
        names = ("src", "src.utils", "src.masks", "src.utils.tensors", "src.masks.utils")
        before = {name: sys.modules.get(name) for name in names}
        _official_constructor(self.upstream, self.metadata["upstream"])
        self.assertEqual(before, {name: sys.modules.get(name) for name in names})

    def test_modified_upstream_source_is_rejected(self):
        source = self.upstream / "src/models/vision_transformer.py"
        content = source.read_bytes()
        try:
            source.write_bytes(content + b"\n# unit-test-only change\n")
            with self.assertRaises(CheckpointError):
                _official_constructor(self.upstream, self.metadata["upstream"])
        finally:
            source.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
