"""Synthetic implementation checks; these are not trained-encoder evidence."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.spatial_readout import probe
from experiments.spatial_readout.tasks import ordered_pairs


def fixture_labels(images: int) -> np.ndarray:
    # Coordinates belong to the independent label fixture, never head inputs.
    pairs = ordered_pairs()
    source, destination = pairs[:, 0], pairs[:, 1]
    vertical = (source // 3 + 1 == destination // 3) & (source % 3 == destination % 3)
    horizontal = (source % 3 + 1 == destination % 3) & (source // 3 == destination // 3)
    return np.broadcast_to(
        np.stack((vertical, horizontal), axis=-1), (images, 72, 2)
    ).copy()


def tiny_features(images: int = 8, dim: int = 4, seed: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(images, 9, dim)).astype(np.float32)


def quick_config(**updates):
    return {"epochs": 2, "patience": 1, "batch_images": 4, **updates}


def fit_fixture(train=None, dev=None, **config):
    train = tiny_features() if train is None else train
    dev = tiny_features(4, train.shape[-1], 41) if dev is None else dev
    return probe.fit_probe(
        train,
        fixture_labels(len(train)),
        dev,
        fixture_labels(len(dev)),
        seed=2,
        config=quick_config(**config),
    )


class HeadContractTests(unittest.TestCase):
    def test_exact_requested_capacities(self):
        for head_kind, expected in (("linear", 1538), ("mlp", 98690)):
            with self.subTest(head_kind=head_kind):
                head = probe.PairHead(384, head_kind)
                self.assertEqual(sum(p.numel() for p in head.parameters()), expected)
                self.assertEqual(probe.parameter_count(384, head_kind), expected)
                self.assertEqual(tuple(head(torch.zeros(2, 768)).shape), (2, 2))

    def test_head_accepts_content_tensor_only(self):
        head = probe.PairHead(4)
        with self.assertRaises(TypeError):
            head({"features": torch.zeros(2, 8), "canonical_ids": [0, 1]})
        with self.assertRaises(TypeError):
            head(torch.zeros(2, 8), coordinates=torch.zeros(2, 4))
        with self.assertRaises(ValueError):
            head(torch.zeros(2, 10))
        with self.assertRaises(TypeError):
            head(np.zeros((2, 8), dtype=np.float32))
        self.assertEqual(list(inspect.signature(head.forward).parameters), ["content_pairs"])

    def test_forward_reverse_logits_follow_ordered_content(self):
        head = probe.PairHead(1)
        with torch.no_grad():
            head.layers.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))
            head.layers.bias.zero_()
        fitted = {
            "state_dict": head.state_dict(),
            "seed": 0,
            "head_kind": "linear",
            "feature_dim": 1,
            "input_dim": 2,
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        }
        features = np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1)
        predictions = probe.predict_scores(fitted, features)
        for pair_index, (i, j) in enumerate(ordered_pairs()):
            self.assertNotEqual(i, j)
            hi, hj = i + 1, j + 1
            np.testing.assert_allclose(
                predictions[0, pair_index], [hi + 2 * hj, 3 * hi + 5 * hj]
            )


class StatisticsAndSelectionTests(unittest.TestCase):
    def test_population_moments_stream_over_train_only(self):
        train = tiny_features(11)
        train[:, :, -1] = 10.0
        mean, std = probe.training_statistics(train, chunk_images=2)
        flattened = train.astype(np.float64).reshape(-1, 4)
        np.testing.assert_allclose(mean, flattened.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            std, np.maximum(flattened.std(axis=0), 1e-6), rtol=1e-6
        )
        self.assertEqual(std[-1], np.float32(1e-6))

    def test_dev_distribution_does_not_change_statistics_or_positive_weights(self):
        train = tiny_features()
        first = fit_fixture(train, epochs=1)
        dev = tiny_features(4, seed=41) * 10 + 50
        dev_labels = fixture_labels(len(dev))
        second = probe.fit_probe(
            train,
            fixture_labels(len(train)),
            dev,
            dev_labels,
            seed=2,
            config=quick_config(epochs=1),
        )
        np.testing.assert_array_equal(first["mean"], second["mean"])
        np.testing.assert_array_equal(first["std"], second["std"])
        np.testing.assert_array_equal(first["pos_weight"], [11, 11])
        np.testing.assert_array_equal(second["pos_weight"], [11, 11])
        np.testing.assert_array_equal(first["train_positive_counts"], [48, 48])
        self.assertEqual(first["train_pair_count"], 8 * 72)

    def test_fit_has_no_test_or_metadata_inputs(self):
        names = list(inspect.signature(probe.fit_probe).parameters)
        self.assertEqual(
            names,
            [
                "train_features", "train_labels", "dev_features", "dev_labels",
                "seed", "config", "device", "progress",
            ],
        )
        with self.assertRaises(TypeError):
            probe.fit_probe(
                tiny_features(), fixture_labels(8), tiny_features(4), fixture_labels(4),
                seed=0, test_features=tiny_features(2),
            )
        with self.assertRaises(ValueError):
            fit_fixture(test_features=tiny_features(2))

    def test_early_stopping_selects_dev_peak_and_thresholds_receive_dev_only(self):
        train, dev = tiny_features(), tiny_features(4, seed=41)
        labels = fixture_labels(4)
        actual_thresholds = probe.select_thresholds
        with (
            patch.object(
                probe, "adjacency_metrics",
                side_effect=[{"macro": {"ap": x}} for x in (0.2, 0.4, 0.3, 0.1)],
            ) as metrics,
            patch.object(probe, "select_thresholds", wraps=actual_thresholds) as thresholds,
        ):
            result = probe.fit_probe(
                train, fixture_labels(8), dev, labels, seed=3,
                config=quick_config(epochs=9, patience=2),
            )
        self.assertEqual(result["selected_epoch"], 2)
        self.assertEqual(result["selected_dev_macro_ap"], 0.4)
        self.assertEqual(len(result["history"]), 4)
        self.assertEqual(metrics.call_count, 4)
        for call in metrics.call_args_list:
            self.assertIs(call.args[1], labels)
            self.assertEqual(call.args[0].shape, (4, 72, 2))
        thresholds.assert_called_once()
        self.assertIs(thresholds.call_args.args[1], labels)
        np.testing.assert_allclose(
            thresholds.call_args.args[0], probe.predict_scores(result, dev), atol=1e-7
        )

    def test_equal_dev_scores_choose_first_learning_rate_and_earliest_epoch(self):
        with patch.object(probe, "adjacency_metrics", return_value={"macro": {"ap": 0.3}}):
            fitted = fit_fixture(learning_rates=[0.01, 0.003], epochs=5, patience=1)
        self.assertEqual(fitted["selected_learning_rate"], 0.01)
        self.assertEqual(fitted["selected_learning_rate_index"], 0)
        self.assertEqual(fitted["selected_epoch"], 1)
        self.assertEqual(len(fitted["history"]), 4)


class FitAndPredictionTests(unittest.TestCase):
    def test_seed_is_deterministic_and_does_not_change_callers_rng(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        first = fit_fixture()
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        second = fit_fixture()
        self.assertEqual(first["history"], second["history"])
        for name in first["state_dict"]:
            self.assertTrue(torch.equal(first["state_dict"][name], second["state_dict"][name]))
        np.testing.assert_array_equal(first["thresholds"], second["thresholds"])

    def test_container_reordering_only_reorders_predictions(self):
        fitted = fit_fixture()
        features = tiny_features(3, seed=42)
        original = probe.predict_scores(fitted, features, batch_images=2)
        permutation = np.array([6, 0, 4, 8, 2, 1, 7, 3, 5])
        reordered = probe.predict_scores(fitted, features[:, permutation], batch_images=2)
        lookup = {tuple(pair): index for index, pair in enumerate(ordered_pairs())}
        for index, (i, j) in enumerate(ordered_pairs()):
            old_index = lookup[(permutation[i], permutation[j])]
            np.testing.assert_allclose(reordered[:, index], original[:, old_index], atol=1e-6)

    def test_frozen_fixture_encoder_hash_and_gradients_unchanged_by_probe_fit(self):
        # This intentionally tiny random module tests isolation only. It is never
        # registered or represented as an available pretrained research model.
        encoder = torch.nn.Linear(3, 4).eval().requires_grad_(False)

        def state_digest():
            digest = hashlib.sha256()
            for name, value in encoder.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().numpy().tobytes())
            return digest.hexdigest()

        before = state_digest()
        inputs = torch.from_numpy(tiny_features(12, dim=3))
        with torch.inference_mode():
            features = encoder(inputs).numpy().copy()
        self.assertTrue(features.flags.owndata)
        fitted = fit_fixture(features[:8], features[8:])
        self.assertEqual(before, state_digest())
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertEqual(fitted["feature_dim"], 4)

    def test_read_only_feature_cache_supported_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-features.npy"
            original = tiny_features(12)
            np.save(path, original)
            features = np.load(path, mmap_mode="r")
            fitted = fit_fixture(features[:8], features[8:])
            predictions = probe.predict_scores(fitted, features[8:])
            self.assertEqual(predictions.shape, (4, 72, 2))
            self.assertEqual(predictions.dtype, np.float32)
            np.testing.assert_array_equal(features, original)

    def test_mlp_is_a_separate_fixed_capacity_probe(self):
        fitted = fit_fixture(head_kind="mlp", epochs=1)
        self.assertEqual(fitted["head_kind"], "mlp")
        self.assertEqual(fitted["parameter_count"], 8 * 128 + 128 + 128 * 2 + 2)
        self.assertEqual(probe.predict_scores(fitted, tiny_features(2)).shape, (2, 72, 2))

    def test_invalid_or_nonfinite_features_are_rejected(self):
        for split in ("train", "dev"):
            with self.subTest(split=split):
                train, dev = tiny_features(), tiny_features(4)
                (train if split == "train" else dev)[0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    fit_fixture(train, dev)
        with self.assertRaises(TypeError):
            probe.fit_probe(
                torch.zeros(8, 9, 4), fixture_labels(8), tiny_features(4), fixture_labels(4),
                seed=0, config=quick_config(),
            )
        invalid = fixture_labels(8).astype(np.float32)
        invalid[0, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            probe.fit_probe(
                tiny_features(), invalid, tiny_features(4), fixture_labels(4),
                seed=0, config=quick_config(),
            )


if __name__ == "__main__":
    unittest.main()
