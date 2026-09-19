"""Regression cases for constant encoders and nonfinite probe results."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch import nn

from sepa_plan_b.config import Config
from sepa_plan_b.evaluation import (analyze, fit_linear, linear_metrics, knn_accuracy,
                                    spatial_fit, position_fit)
from sepa_plan_b.losses import diagnostics


class CollapseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_constant_encoder_is_flagged_even_when_ratio_is_large(self):
        encoded, target = torch.ones(4, 7, 8), torch.ones(4, 2, 8)
        prediction = target.clone()
        prediction[:, 0] = 2
        result = diagnostics(prediction, target, encoded)
        self.assertGreater(result["std_ratio"], 1)
        self.assertIs(result["collapse_flag"], True)
        self.assertIn("low_encoder_std", result["collapse_reasons"])
        self.assertIn("low_target_std", result["collapse_reasons"])

    def test_small_absolute_variance_is_not_hidden_by_healthy_ratio(self):
        target = 1 + 1e-6 * torch.randn(4, 2, 8)
        encoded = 1 + 1e-6 * torch.randn(4, 7, 8)
        result = diagnostics(target, target, encoded)
        self.assertAlmostEqual(result["std_ratio"], 1)
        self.assertIs(result["collapse_flag"], True)

    def test_slot_only_features_are_compared_at_matching_canonical_positions(self):
        # Slot order differs across images. Variance over token index would look healthy.
        ids = torch.stack([torch.tensor([0, 1, 2, 4, 5, 6, 8]).roll(i) for i in range(4)])
        queries = torch.tensor([[3, 7], [7, 3], [3, 7], [7, 3]])
        lookup = torch.arange(9).float()[:, None].expand(-1, 8)
        result = diagnostics(lookup[queries], lookup[queries], lookup[ids],
                             canonical_ids=ids, query_slots=queries, sample_ids=list("abcd"))
        self.assertGreater(result["encoder_std"], 0.1)
        self.assertGreater(result["target_std"], 0.1)
        self.assertEqual(result["encoder_image_std"], 0)
        self.assertEqual(result["target_image_std"], 0)
        self.assertIs(result["collapse_flag"], True)

    def test_healthy_distinct_images_pass_available_checks(self):
        target, encoded = torch.randn(4, 2, 8), torch.randn(4, 7, 8)
        result = diagnostics(target, target, encoded)
        self.assertIs(result["collapse_flag"], False)
        self.assertTrue(result["collapse_checks_complete"])
        self.assertEqual(result["collapse_reasons"], [])

    def test_single_or_duplicate_images_are_unknown_without_positive_evidence(self):
        target, encoded = torch.randn(1, 2, 8), torch.randn(1, 7, 8)
        for repeat in (1, 4):
            result = diagnostics(target.repeat(repeat, 1, 1), target.repeat(repeat, 1, 1),
                                 encoded.repeat(repeat, 1, 1), sample_ids=["same"] * repeat)
            self.assertIsNone(result["collapse_flag"])
            self.assertIsNone(result["encoder_image_std"])
            self.assertFalse(result["collapse_checks_complete"])

    def test_nonoverlapping_queries_do_not_claim_complete_checks(self):
        target, encoded = torch.randn(2, 2, 8), torch.randn(2, 7, 8)
        result = diagnostics(target, target, encoded, query_slots=torch.tensor([[0, 1], [2, 3]]))
        self.assertIsNone(result["target_image_std"])
        self.assertIsNone(result["collapse_flag"])

    def test_nonfinite_diagnostics_fail(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(FloatingPointError):
                diagnostics(torch.full((2, 2, 8), invalid), torch.ones(2, 2, 8), torch.ones(2, 7, 8))

    def test_collapse_thresholds_must_be_positive_and_finite(self):
        for key in ("collapse_min_std", "collapse_min_image_std"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    Config.from_dict({"training": {key: value}})


class ProbeNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.config = Config()
        self.config = replace(self.config, probe=replace(self.config.probe, epochs=1, spatial_steps=1, batch_size=4))
        self.x = torch.tensor([[0., 0.], [1., 0.], [8., 1.], [9., 1.]])
        self.y = torch.tensor([0, 0, 1, 1])
        self.tiles = torch.randn(4, 9, 2)

    def test_invalid_features_rejected_in_every_probe(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            bad, bad_tiles = self.x.clone(), self.tiles.clone()
            bad[0, 0] = bad_tiles[0, 0, 0] = invalid
            calls = [
                lambda: fit_linear(bad, self.y, 2, self.config, 0, "cpu"),
                lambda: linear_metrics(nn.Linear(2, 2), torch.zeros(2), torch.ones(2), bad, self.y, "cpu"),
                lambda: knn_accuracy(bad, self.y, self.x, self.y, 2),
                lambda: knn_accuracy(self.x, self.y, bad, self.y, 2),
                lambda: spatial_fit(bad_tiles, self.tiles, self.config, 0, "adjacency", "cpu"),
                lambda: spatial_fit(self.tiles, bad_tiles, self.config, 0, "direction", "cpu"),
                lambda: position_fit(bad_tiles, self.tiles, self.config, 0, "cpu"),
                lambda: position_fit(self.tiles, bad_tiles, self.config, 0, "cpu"),
            ]
            for i, call in enumerate(calls):
                with self.subTest(invalid=invalid, entry=i), self.assertRaises(FloatingPointError):
                    call()

    def test_invalid_classifier_logits_cannot_become_accuracy(self):
        head = nn.Linear(2, 2)
        with torch.no_grad():
            head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "logits"):
            linear_metrics(head, torch.zeros(2), torch.ones(2), self.x, self.y, "cpu")

    def test_invalid_logits_stop_all_probe_training_heads(self):
        class BadHead(nn.Linear):
            def forward(self, x):
                return super().forward(x) * float("nan")
        with patch("sepa_plan_b.evaluation.nn.Linear", BadHead):
            for call in (lambda: fit_linear(self.x, self.y, 2, self.config, 0, "cpu"),
                         lambda: spatial_fit(self.tiles, self.tiles, self.config, 0, "adjacency", "cpu"),
                         lambda: position_fit(self.tiles, self.tiles, self.config, 0, "cpu")):
                with self.assertRaisesRegex(FloatingPointError, "logits"):
                    call()

    def test_nan_loss_stops_before_optimizer_update(self):
        with patch("sepa_plan_b.evaluation.F.cross_entropy", return_value=torch.tensor(float("nan"))), \
             patch.object(torch.optim.SGD, "step") as step:
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                fit_linear(self.x, self.y, 2, self.config, 0, "cpu")
            step.assert_not_called()

    def test_invalid_gradient_stops_before_optimizer_update(self):
        class BadGradientHead(nn.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        with patch("sepa_plan_b.evaluation.nn.Linear", BadGradientHead), patch.object(torch.optim.SGD, "step") as step:
            with self.assertRaisesRegex(FloatingPointError, "gradients"):
                fit_linear(self.x, self.y, 2, self.config, 0, "cpu")
            step.assert_not_called()

    def test_invalid_final_parameter_update_is_rejected(self):
        original = torch.optim.SGD.step
        def corrupt(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("inf"))
            return result
        with patch.object(torch.optim.SGD, "step", corrupt):
            with self.assertRaisesRegex(FloatingPointError, "parameters after update"):
                fit_linear(self.x, self.y, 2, self.config, 0, "cpu")

    def test_finite_features_with_overflowing_statistics_are_rejected(self):
        x = torch.full((4, 2), 3e38)
        with self.assertRaisesRegex(FloatingPointError, "statistics"):
            fit_linear(x, self.y, 2, self.config, 0, "cpu")

    def test_nonfinite_saved_scores_cannot_enter_stage1_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            for value in (float("nan"), float("inf"), -1, 1.5):
                path.write_text(json.dumps({"scores": {"top1": value}}))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite accuracies"):
                    analyze([path])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
