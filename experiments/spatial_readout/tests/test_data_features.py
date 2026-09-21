"""Tiny synthetic image fixtures test engineering contracts, never model quality."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiments.spatial_readout import data, features
from experiments.spatial_readout.common import PREPROCESS, digest, file_hash, read_json, write_json
from experiments.spatial_readout.metrics import retrieval_metrics
from experiments.spatial_readout.probe import predict_scores
from experiments.spatial_readout.tasks import make_design


def synthetic_pixels(seed: int, height=53, width=71) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def image_record(root: Path, relative: str, pixels: np.ndarray, label: int) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    raw = path.read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    return {"path": relative, "label": label, "id": identity,
            "sha256": identity, "bytes": len(raw)}


def manifest(root: Path, role: str, records: list[dict]) -> dict:
    value = {"schema": 1, "role": role, "root": str(root),
             "classes": ["synthetic-class-0", "synthetic-class-1"], "records": records}
    value["fingerprint"] = digest(value)
    return value


class ImageFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_root = self.root / "train-pixels"
        self.test_root = self.root / "test-pixels"
        train_records = []
        for label in range(2):
            for number in range(6):
                train_records.append(image_record(
                    self.train_root, f"{label}/{number}.png",
                    synthetic_pixels(label * 10 + number), label,
                ))
        # One exact byte duplicate and one differently encoded identical image
        # inside the training source. Neither may straddle train and dev.
        byte_copy = copy.deepcopy(train_records[0])
        byte_copy["path"] = "0/byte-copy.png"
        shutil.copyfile(self.train_root / train_records[0]["path"], self.train_root / byte_copy["path"])
        train_records.append(byte_copy)
        train_records.append(image_record(self.train_root, "0/rgb-copy.bmp", synthetic_pixels(1), 0))
        test_records = [
            image_record(self.test_root, "0/novel.png", synthetic_pixels(30), 0),
            image_record(self.test_root, "1/novel.png", synthetic_pixels(40), 1),
            image_record(self.test_root, "0/byte-overlap.png", synthetic_pixels(0), 0),
            image_record(self.test_root, "0/rgb-overlap.bmp", synthetic_pixels(2), 0),
        ]
        self.train = manifest(self.train_root, "train", train_records)
        self.test = manifest(self.test_root, "val", test_records)

    def content_groups(self):
        return data.decoded_groups([self.train, self.test], self.root / "groups.json", workers=1)

    def tiny_dataset(self, condition="clean"):
        source = manifest(self.train_root, "train", self.train["records"][3:5])
        design = make_design(["synthetic-image-a", "synthetic-image-b"], seed=9)
        return data.TileDataset(source, design["tile_orders"], condition=condition), source, design

    def test_center_bicubic_geometry_and_tensor_only_container_reordering(self):
        raw = synthetic_pixels(201, height=251, width=319)
        record = image_record(self.train_root, "geometry.png", raw, 0)
        source = manifest(self.train_root, "train", [record])
        canonical = data.TileDataset(source, np.arange(9)[None])[0]
        self.assertIsInstance(canonical, torch.Tensor)
        self.assertEqual(canonical.dtype, torch.float32)
        self.assertEqual(tuple(canonical.shape), (9, 3, 80, 80))
        self.assertFalse(canonical.requires_grad)
        self.assertEqual(tuple(canonical.unfold(2, 16, 16).unfold(3, 16, 16).shape), (9, 3, 5, 5, 16, 16))
        # Independent reference: odd excess width fixes the floor-offset crop.
        cropped = Image.fromarray(raw).crop((34, 0, 285, 251))
        resized = np.asarray(cropped.resize((240, 240), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        mean, std = np.asarray(PREPROCESS["mean"]), np.asarray(PREPROCESS["std"])
        reference = np.stack([
            ((resized[row * 80:(row + 1) * 80, col * 80:(col + 1) * 80] - mean) / std).transpose(2, 0, 1)
            for row in range(3) for col in range(3)
        ])
        np.testing.assert_allclose(canonical.numpy(), reference, rtol=1e-6, atol=5e-7)
        permutation = np.asarray([8, 0, 4, 2, 7, 3, 6, 1, 5])
        changed_metadata = copy.deepcopy(source)
        changed_metadata["records"][0].update(id="different-label-side-id", label=1, original_row=99, original_col=-30)
        reordered = data.TileDataset(changed_metadata, permutation[None])[0]
        self.assertTrue(torch.equal(reordered, canonical[permutation]))
        self.assertTrue(torch.equal(canonical, data.TileDataset(source, np.arange(9)[None])[0]))

    def test_border_mask_uses_fixed_normalized_zero_and_preserves_interior(self):
        clean, _, design = self.tiny_dataset()
        masked, _, _ = self.tiny_dataset(condition="border_masked")
        original = clean[0]
        modified = masked[0]
        self.assertTrue(torch.equal(modified[..., 8:-8, 8:-8], original[..., 8:-8, 8:-8]))
        border = torch.ones((80, 80), dtype=torch.bool)
        border[8:-8, 8:-8] = False
        self.assertEqual(int(torch.count_nonzero(modified[..., border])), 0)
        raw = modified * torch.tensor(PREPROCESS["std"])[None, :, None, None]
        raw += torch.tensor(PREPROCESS["mean"])[None, :, None, None]
        expected = torch.tensor(PREPROCESS["mean"])[None, :, None].expand(9, -1, int(border.sum()))
        self.assertTrue(torch.equal(raw[..., border], expected))
        self.assertTrue(torch.equal(clean[0], original))
        with self.assertRaises(ValueError):
            data.mask_border(original, width=7)
        with self.assertRaises(ValueError):
            data.TileDataset(clean.images.manifest, design["tile_orders"], condition="random")

    def test_decoded_and_byte_duplicate_groups_never_cross_splits_and_cover_sources(self):
        groups, provenance = self.content_groups()
        records = self.train["records"]
        self.assertEqual(records[0]["sha256"], records[12]["sha256"])
        self.assertNotEqual(records[1]["sha256"], records[13]["sha256"])
        self.assertEqual(groups[records[1]["sha256"]], groups[records[13]["sha256"]])
        self.assertEqual(groups[records[2]["sha256"]], groups[self.test["records"][3]["sha256"]])
        self.assertLess(provenance["unique_decoded_contents"], provenance["unique_file_contents"])
        split = data.make_splits(self.train, self.test, seed=7, dev_fraction=0.25, content_groups=groups)
        data.validate_disjoint(split["manifests"])
        test_ids = {r["id"] for r in split["manifests"]["test"]["records"]}
        allowed = {i for i, record in enumerate(records) if groups[record["sha256"]] not in test_ids}
        selected = {r["source_index"] for role in ("train", "dev") for r in split["manifests"][role]["records"]}
        self.assertEqual(selected, allowed)
        self.assertEqual(split["excluded_training_records"], 3)
        self.assertEqual(len(split["manifests"]["test"]["records"]), 4)
        membership = {r["source_index"]: role for role in ("train", "dev") for r in split["manifests"][role]["records"]}
        self.assertEqual(membership[1], membership[13])

    def test_decoding_rejects_mutated_bytes_and_out_of_root_paths(self):
        first_path = self.train_root / self.train["records"][0]["path"]
        first_path.write_bytes(first_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.content_groups()
        escaped = copy.deepcopy(self.train)
        escaped["records"][0]["path"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "escapes root"):
            data.decoded_groups([escaped, self.test], self.root / "escape-groups.json", workers=1)

    def test_decoded_mapping_cache_checks_source_coverage_and_mapping_digest(self):
        expected_groups, expected_provenance = self.content_groups()
        self.assertEqual(self.content_groups(), (expected_groups, expected_provenance))
        path = self.root / "groups.json"
        original = read_json(path)
        for corruption in ("coverage", "digest", "source"):
            with self.subTest(corruption=corruption):
                changed = copy.deepcopy(original)
                key = next(iter(changed["groups"]))
                if corruption == "coverage":
                    del changed["groups"][key]
                    changed["groups_hash"] = digest(changed["groups"])
                elif corruption == "digest":
                    changed["groups"][key] = "different-content-group"
                else:
                    changed["identity"]["sources"][0] = "different-source"
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    self.content_groups()

    def test_legacy_split_reuse_requires_reproducible_recipe_and_full_coverage(self):
        a, b = data.grouped_split(self.train["records"], seed=3, fraction=0.2)
        old = {"train_indices": a.tolist(), "dev_indices": b.tolist()}
        for role in ("train", "dev"):
            old[role + "_ids_hash"] = digest([self.train["records"][i]["id"] for i in old[role + "_indices"]])
        reused = data.make_splits(self.train, self.test, seed=3, dev_fraction=0.2, existing_split=old)
        self.assertTrue(reused["origin"].startswith("existing grouped split reproduced"))
        regenerated = data.make_splits(self.train, self.test, seed=3, dev_fraction=0.5, existing_split=old)
        self.assertTrue(regenerated["origin"].startswith("new deterministic"))
        self.assertEqual(regenerated["split_seed"], 3)
        self.assertEqual(regenerated["dev_fraction"], 0.5)
        partial = copy.deepcopy(old)
        partial["train_indices"] = partial["train_indices"][1:]
        partial["train_ids_hash"] = digest([self.train["records"][i]["id"] for i in partial["train_indices"]])
        repaired = data.make_splits(self.train, self.test, seed=3, dev_fraction=0.2, existing_split=partial)
        self.assertTrue(repaired["origin"].startswith("new deterministic"))
        included = {r["source_index"] for role in ("train", "dev") for r in repaired["manifests"][role]["records"]}
        test_ids = {r["id"] for r in self.test["records"]}
        self.assertEqual(included, {i for i, r in enumerate(self.train["records"]) if r["id"] not in test_ids})

    def test_smoke_limits_are_complete_positive_integer_triplets_and_deterministic(self):
        for invalid in ((1,), (1, 1), (1, 1, 1, 1), (1, 0, 1), (True, 1, 1), (1.5, 1, 1)):
            with self.subTest(limits=invalid), self.assertRaises(ValueError):
                data.make_splits(self.train, self.test, smoke_limits=invalid)
        first = data.make_splits(self.train, self.test, seed=5, smoke_limits=(2, 1, 2))
        second = data.make_splits(self.train, self.test, seed=5, smoke_limits=(2, 1, 2))
        self.assertTrue(first["engineering_only"])
        self.assertEqual(first, second)
        self.assertEqual([len(first["manifests"][role]["records"]) for role in ("train", "dev", "test")], [2, 1, 2])

    def test_source_loader_enforces_official_test_count_and_explicit_fixture_override(self):
        train_path, test_path = self.root / "train.json", self.root / "val.json"
        write_json(train_path, self.train)
        write_json(test_path, self.test)
        with self.assertRaisesRegex(ValueError, "5000"):
            data.load_sources(train_path, test_path, expected_classes=2)
        with self.assertRaisesRegex(ValueError, "class mappings"):
            data.load_sources(train_path, test_path, expected_test_images=4)
        train, test, provenance = data.load_sources(
            train_path, test_path, expected_classes=2, expected_test_images=4,
        )
        self.assertEqual((len(train["records"]), len(test["records"])), (14, 4))
        self.assertEqual(provenance["test"]["file_hash"], file_hash(test_path))
        self.assertEqual(provenance["test"]["fingerprint"], self.test["fingerprint"])
        with self.assertRaises(FileNotFoundError):
            data.load_sources(train_path, test_path, expected_classes=2, expected_test_images=4, train_root=self.root / "missing")

    def test_extraction_writes_readonly_fp32_memmap_covers_all_tiles_and_reuses_cache(self):
        dataset, source, design = self.tiny_dataset()
        model = features.PixelFeatures("color_lowfreq").train()
        identity = features.cache_identity(model_manifest=model.manifest, split_manifest=source,
                                           design_hash="fixture-design", condition="clean")
        path = self.root / "features.npy"
        observed = []
        hook = model.register_forward_hook(lambda module, args, output: observed.append(
            (torch.is_grad_enabled(), args[0].requires_grad, output.requires_grad, len(args[0]))
        ))
        try:
            array, receipt = features.extract_cached(model, dataset, path, identity, dimension=54,
                                                     batch_images=1, tile_batch_size=5, workers=0)
        finally:
            hook.remove()
        self.assertIsInstance(array, np.memmap)
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(array.shape, (2, 9, 54))
        self.assertFalse(array.flags.writeable)
        self.assertFalse(model.training)
        self.assertEqual(sum(item[3] for item in observed), 18)
        self.assertTrue(all(item[:3] == (False, False, False) for item in observed))
        self.assertEqual(receipt["encoder_state_before"], receipt["encoder_state_after"])
        self.assertFalse(receipt["encoder_gradients"])
        self.assertEqual((receipt["images"], receipt["tiles"]), (2, 18))
        expected = model(torch.stack([dataset[0], dataset[1]]).flatten(0, 1)).numpy().reshape(2, 9, 54)
        np.testing.assert_allclose(array, expected, atol=1e-7)
        before = path.stat().st_mtime_ns
        with patch.object(model, "forward", side_effect=AssertionError("cache should be reused")):
            again, loaded_receipt = features.extract_cached(model, dataset, path, identity, dimension=54)
        np.testing.assert_array_equal(array, again)
        self.assertEqual(receipt, loaded_receipt)
        self.assertEqual(path.stat().st_mtime_ns, before)


class CompactFeaturesAndArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_color_lowfrequency_54_values_match_rgb_population_statistics_and_thumbnail(self):
        raw = np.random.default_rng(33).random((2, 3, 80, 80), dtype=np.float32)
        mean = np.asarray(PREPROCESS["mean"], dtype=np.float32)[None, :, None, None]
        std = np.asarray(PREPROCESS["std"], dtype=np.float32)[None, :, None, None]
        normalized = torch.from_numpy((raw - mean) / std).requires_grad_(True)
        model = features.PixelFeatures("color_lowfreq")
        actual = model(normalized)
        expected = np.concatenate((raw.mean((2, 3)), raw.std((2, 3)),
                                   raw.reshape(2, 3, 4, 20, 4, 20).mean((3, 5)).reshape(2, 48)), axis=1)
        self.assertEqual(tuple(actual.shape), (2, 54))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertFalse(actual.requires_grad)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7, rtol=1e-6)

    def test_border_only_192_values_ignore_interior_and_fill_it_with_fixed_mean(self):
        tiles = torch.from_numpy(np.random.default_rng(66).normal(size=(2, 3, 80, 80)).astype(np.float32))
        changed = tiles.clone()
        changed[..., 8:-8, 8:-8] = 1234
        model = features.PixelFeatures("border_only")
        actual = model(tiles)
        self.assertEqual(tuple(actual.shape), (2, 192))
        self.assertTrue(torch.equal(actual, model(changed)))
        expected_pixels = tiles.numpy().copy()
        expected_pixels[..., 8:-8, 8:-8] = 0
        expected_pixels *= np.asarray(PREPROCESS["std"], dtype=np.float32)[None, :, None, None]
        expected_pixels += np.asarray(PREPROCESS["mean"], dtype=np.float32)[None, :, None, None]
        expected = expected_pixels.reshape(2, 3, 8, 10, 8, 10).mean((3, 5)).reshape(2, 192)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7, rtol=1e-6)
        zeros = model(torch.zeros(1, 3, 80, 80)).numpy().reshape(1, 3, 8, 8)
        np.testing.assert_allclose(zeros, np.broadcast_to(np.asarray(PREPROCESS["mean"])[None, :, None, None], zeros.shape), atol=6e-7)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 79, 80))

    def test_feature_identity_distinguishes_checkpoint_pe_preprocess_condition_split_and_design(self):
        model = {"family": "v1", "checkpoint_hash": "fixture-checkpoint-a", "pe_protocol": "shared-tile-5x5"}
        kwargs = dict(model_manifest=model, split_manifest={"fingerprint": "split-a"}, design_hash="design-a", condition="clean")
        original = digest(features.cache_identity(**kwargs))
        for field in ("checkpoint", "pe", "preprocess", "condition", "split", "design"):
            with self.subTest(field=field):
                changed = copy.deepcopy(kwargs)
                if field == "checkpoint":
                    changed["model_manifest"]["checkpoint_hash"] = "fixture-checkpoint-b"
                elif field == "pe":
                    changed["model_manifest"]["pe_protocol"] = "fixed-top-left-5x5"
                elif field == "condition":
                    changed["condition"] = "border_masked"
                elif field == "split":
                    changed["split_manifest"]["fingerprint"] = "split-b"
                elif field == "design":
                    changed["design_hash"] = "design-b"
                if field == "preprocess":
                    protocol = {**PREPROCESS, "resize": "synthetic different protocol"}
                    with patch.object(features, "PREPROCESS", protocol):
                        actual = digest(features.cache_identity(**changed))
                else:
                    actual = digest(features.cache_identity(**changed))
                self.assertNotEqual(original, actual)

    def test_corrupt_feature_cache_fails_closed_on_identity_bytes_layout_or_finiteness(self):
        path = self.root / "cache.npy"
        identity, shape = {"fixture": "content-cache"}, (2, 9, 54)

        def save(array):
            np.save(path, array, allow_pickle=False)
            write_json(path.with_suffix(".json"), {"identity": identity, "shape": list(shape), "file_hash": file_hash(path)})

        save(np.zeros(shape, dtype=np.float32))
        self.assertIsInstance(features.open_cache(path, identity, shape)[0], np.memmap)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            features.open_cache(path, {"fixture": "wrong"}, shape)
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            features.open_cache(path, identity, shape)
        for invalid in (np.zeros(shape, np.float64), np.zeros((1, 9, 54), np.float32), np.full(shape, np.nan, np.float32)):
            with self.subTest(shape=invalid.shape, dtype=str(invalid.dtype)):
                save(invalid)
                with self.assertRaises(ValueError):
                    features.open_cache(path, identity, shape)

    def test_design_roundtrip_keeps_pair_queries_exact_and_refuses_overwrite(self):
        original = make_design(["synthetic-a", "synthetic-b"], seed=11)
        path = self.root / "design.npz"
        self.assertEqual(features.save_design(path, original), file_hash(path))
        loaded = features.load_design(path)
        for key, value in original.items():
            np.testing.assert_array_equal(loaded[key], value)
        measured = retrieval_metrics(original["labels"].astype(np.float32), loaded)
        self.assertEqual(measured["macro"]["recall_at_1"], 1.0)
        with self.assertRaises(FileExistsError):
            features.save_design(path, original)

    def test_fitted_probe_safe_roundtrip_preserves_metadata_arrays_and_predictions(self):
        fitted = {
            "schema_version": 1, "seed": 3, "head_kind": "linear", "feature_dim": 2, "input_dim": 4,
            "state_dict": {"layers.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                           "layers.bias": torch.tensor([0.2, -0.1])},
            "mean": np.asarray([0.2, 0.3], dtype=np.float32),
            "std": np.asarray([0.5, 0.7], dtype=np.float32),
            "thresholds": np.asarray([0.8, -0.9], dtype=np.float64),
            "pos_weight": np.asarray([11, 11], dtype=np.float32),
            "train_positive_counts": np.asarray([12, 12], dtype=np.int64),
            "history": [{"epoch": 1, "dev_macro_ap": 0.2}],
            "settings": {"learning_rates": [0.01], "epochs": 1},
            "standardization_source": "synthetic fixture train only",
        }
        path = self.root / "fitted.pt"
        features.save_fitted(path, fitted)
        with patch.object(features.torch, "load", wraps=torch.load) as loader:
            loaded = features.load_fitted(path)
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        for key in ("mean", "std", "thresholds", "pos_weight", "train_positive_counts"):
            self.assertIsInstance(loaded[key], np.ndarray)
            np.testing.assert_array_equal(loaded[key], fitted[key])
        for key in ("history", "settings", "standardization_source", "seed"):
            self.assertEqual(loaded[key], fitted[key])
        pixels = np.random.default_rng(55).normal(size=(2, 9, 2)).astype(np.float32)
        np.testing.assert_array_equal(predict_scores(fitted, pixels), predict_scores(loaded, pixels))
        with self.assertRaises(FileExistsError):
            features.save_fitted(path, fitted)


if __name__ == "__main__":
    unittest.main()
