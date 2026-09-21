from __future__ import annotations

import copy
import unittest

import numpy as np

from experiments.spatial_readout.metrics import (
    adjacency_metrics,
    cosine_pair_scores,
    retrieval_metrics,
    select_thresholds,
)
from experiments.spatial_readout.tasks import (
    DIRECTION_NAMES,
    adjacency_targets,
    make_design,
    ordered_pairs,
)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.pairs = ordered_pairs()
        self.pair_lookup = {tuple(pair): index for index, pair in enumerate(self.pairs)}

    def test_exact_natural_pair_counts_and_immediate_directions(self):
        self.assertEqual(self.pairs.shape, (72, 2))
        self.assertEqual(self.pairs.dtype, np.dtype("int64"))
        self.assertEqual(len(set(map(tuple, self.pairs))), 72)
        self.assertTrue(np.all(self.pairs[:, 0] != self.pairs[:, 1]))
        labels = adjacency_targets(np.arange(9)[None])[0]
        np.testing.assert_array_equal(labels.sum(axis=0), [6, 6])
        self.assertTrue(labels[self.pair_lookup[(0, 3)], 0])
        self.assertTrue(labels[self.pair_lookup[(0, 1)], 1])
        for pair in ((3, 0), (0, 6), (0, 4), (0, 1)):
            self.assertFalse(labels[self.pair_lookup[pair], 0])
        for pair in ((1, 0), (0, 2), (2, 3), (0, 4), (0, 3)):
            self.assertFalse(labels[self.pair_lookup[pair], 1])

    def test_query_count_candidates_and_forward_reverse_score_selection(self):
        design = make_design(["a", "b", "c"], seed=31)
        self.assertEqual(design["query_pairs"].shape, (3, 24, 8))
        np.testing.assert_array_equal(design["query_positive"].sum(axis=-1), 1)
        np.testing.assert_array_equal(design["labels"].sum(axis=1), [[6, 6]] * 3)
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for image_index, order in enumerate(design["tile_orders"]):
            counts = np.bincount(design["query_directions"][image_index], minlength=4)
            np.testing.assert_array_equal(counts, [6, 6, 6, 6])
            seen = set()
            for query in range(24):
                direction = design["query_directions"][image_index, query]
                channel = design["query_channels"][image_index, query]
                self.assertEqual(channel, int(direction >= 2))
                pairs = self.pairs[design["query_pairs"][image_index, query]]
                anchor_column = 1 if direction in (1, 3) else 0
                anchor = pairs[0, anchor_column]
                self.assertTrue(np.all(pairs[:, anchor_column] == anchor))
                candidates = pairs[:, 1 - anchor_column]
                self.assertEqual(set(candidates), set(range(9)) - {anchor})
                self.assertNotIn((anchor, direction), seen)
                seen.add((anchor, direction))
                target = candidates[design["query_positive"][image_index, query]][0]
                anchor_row, anchor_col = divmod(int(order[anchor]), 3)
                target_row, target_col = divmod(int(order[target]), 3)
                self.assertEqual((target_row - anchor_row, target_col - anchor_col), deltas[direction])
                from_labels = design["labels"][
                    image_index, design["query_pairs"][image_index, query], channel
                ]
                np.testing.assert_array_equal(from_labels, design["query_positive"][image_index, query])

    def test_design_is_reproducible_and_independent_of_image_iteration_order(self):
        first = make_design(["opaque-a", "opaque-b"], seed=13)
        repeated = make_design(["opaque-a", "opaque-b"], seed=13)
        reversed_images = make_design(["opaque-b", "opaque-a"], seed=13)
        for key, value in first.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, repeated[key])
                np.testing.assert_array_equal(value, reversed_images[key][::-1])
            else:
                self.assertEqual(value, repeated[key])
        self.assertFalse(np.array_equal(first["tile_orders"], np.tile(np.arange(9), (2, 1))))
        different = make_design(["opaque-a", "opaque-b"], seed=14)
        self.assertFalse(np.array_equal(first["tile_orders"], different["tile_orders"]))

    def test_container_permutation_only_reorders_targets_and_content_scores(self):
        rng = np.random.default_rng(77)
        orders = rng.permutation(9)[None]
        features = rng.normal(size=(1, 9, 11))
        labels = adjacency_targets(orders)
        scores = cosine_pair_scores(features)
        permutation = rng.permutation(9)
        permuted_labels = adjacency_targets(orders[:, permutation])
        permuted_scores = cosine_pair_scores(features[:, permutation])
        lookup = [self.pair_lookup[(int(permutation[i]), int(permutation[j]))] for i, j in self.pairs]
        np.testing.assert_array_equal(permuted_labels, labels[:, lookup])
        np.testing.assert_allclose(permuted_scores, scores[:, lookup], atol=1e-15)
        # A content-only linear pair head has the same property: no coordinate
        # or canonical ID is supplied to the feature/prediction computation.
        weight = rng.normal(size=(22, 2))
        original_logits = np.concatenate(
            (features[:, self.pairs[:, 0]], features[:, self.pairs[:, 1]]), axis=-1
        ) @ weight
        reordered = features[:, permutation]
        permuted_logits = np.concatenate(
            (reordered[:, self.pairs[:, 0]], reordered[:, self.pairs[:, 1]]), axis=-1
        ) @ weight
        np.testing.assert_allclose(permuted_logits, original_logits[:, lookup], atol=1e-14)

    def test_invalid_tile_mappings_and_ids_fail(self):
        for invalid in (np.zeros((1, 9), dtype=int), np.arange(9.0)[None], np.arange(8)[None], np.empty((0, 9), int)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adjacency_targets(invalid)
        for invalid in ([], ["same", "same"], [""], [None], "one-id"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                make_design(invalid)
        for invalid_seed in (-1, 0.5, True):
            with self.subTest(seed=invalid_seed), self.assertRaises(ValueError):
                make_design(["image"], seed=invalid_seed)
        changed = ordered_pairs()
        changed[0] = [0, 0]
        self.assertFalse(np.array_equal(changed, ordered_pairs()))


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.design = make_design(["sample-a", "sample-b", "sample-c"], seed=5)
        self.labels = self.design["labels"]

    def test_all_ties_have_chance_ap_auc_balanced_accuracy_and_retrieval(self):
        scores = np.zeros((3, 72, 2))
        metrics = adjacency_metrics(scores, self.labels, select_thresholds(scores, self.labels))
        for name in ("V", "H", "macro"):
            self.assertAlmostEqual(metrics[name]["ap"], 6 / 72)
            self.assertAlmostEqual(metrics[name]["auroc"], 0.5)
            self.assertAlmostEqual(metrics[name]["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["positive_prevalence"], [6 / 72, 6 / 72])
        retrieval = retrieval_metrics(scores, self.design)
        expected_mrr = sum(1 / rank for rank in range(1, 9)) / 8
        for direction in (*DIRECTION_NAMES, "macro"):
            self.assertAlmostEqual(retrieval[direction]["recall_at_1"], 1 / 8)
            self.assertAlmostEqual(retrieval[direction]["mrr"], expected_mrr)
        for direction in DIRECTION_NAMES:
            self.assertEqual(retrieval[direction]["n_queries"], 18)

    def test_perfect_adjacency_logits_solve_all_four_retrieval_directions(self):
        scores = self.labels.astype(float) * 4 - 2
        metrics = adjacency_metrics(scores, self.labels, select_thresholds(scores, self.labels))
        for metric in metrics["macro"].values():
            self.assertAlmostEqual(metric, 1.0)
        retrieval = retrieval_metrics(scores, self.design)
        for direction in (*DIRECTION_NAMES, "macro"):
            self.assertAlmostEqual(retrieval[direction]["recall_at_1"], 1.0)
            self.assertAlmostEqual(retrieval[direction]["mrr"], 1.0)

    def test_reversing_pairs_changes_directional_predictions(self):
        pairs = ordered_pairs()
        lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
        reverse = [lookup[(j, i)] for i, j in pairs]
        reversed_scores = self.labels[:, reverse].astype(float)
        measured = adjacency_metrics(reversed_scores, self.labels)
        self.assertLess(measured["macro"]["auroc"], 0.5)
        retrieved = retrieval_metrics(reversed_scores, self.design)
        self.assertLess(retrieved["macro"]["recall_at_1"], 1 / 8)

    def test_candidate_and_query_order_do_not_resolve_ties(self):
        scores = np.random.default_rng(9).integers(-1, 2, size=(3, 72, 2)).astype(float)
        expected = retrieval_metrics(scores, self.design)
        changed = copy.deepcopy(self.design)
        candidates = [7, 3, 1, 4, 6, 0, 2, 5]
        queries = np.random.default_rng(8).permutation(24)
        for key in ("query_pairs", "query_positive"):
            changed[key] = changed[key][:, :, candidates]
        for key in ("query_pairs", "query_positive", "query_channels", "query_directions"):
            changed[key] = changed[key][:, queries]
        actual = retrieval_metrics(scores, changed)
        for direction in (*DIRECTION_NAMES, "macro"):
            for key in ("recall_at_1", "mrr"):
                self.assertAlmostEqual(actual[direction][key], expected[direction][key])

    def test_container_relabeling_preserves_retrieval_and_adjacency_metrics(self):
        features = np.random.default_rng(301).normal(size=(3, 9, 6))
        scores = cosine_pair_scores(features)
        expected_retrieval = retrieval_metrics(scores, self.design)
        thresholds = select_thresholds(scores, self.labels)
        expected_adjacency = adjacency_metrics(scores, self.labels, thresholds)
        permutation = np.asarray([3, 8, 1, 7, 5, 0, 2, 6, 4])
        old_to_new = np.argsort(permutation)
        pairs = ordered_pairs()
        lookup = np.full((9, 9), -1, dtype=int)
        lookup[pairs[:, 0], pairs[:, 1]] = np.arange(72)
        old_query_pairs = pairs[self.design["query_pairs"]]
        new_query_pairs = old_to_new[old_query_pairs]
        changed = copy.deepcopy(self.design)
        changed["tile_orders"] = changed["tile_orders"][:, permutation]
        changed["labels"] = adjacency_targets(changed["tile_orders"])
        changed["query_pairs"] = lookup[new_query_pairs[..., 0], new_query_pairs[..., 1]]
        new_scores = cosine_pair_scores(features[:, permutation])
        actual_retrieval = retrieval_metrics(new_scores, changed)
        actual_adjacency = adjacency_metrics(new_scores, changed["labels"], thresholds)
        for direction in (*DIRECTION_NAMES, "macro"):
            for metric in ("recall_at_1", "mrr"):
                self.assertAlmostEqual(actual_retrieval[direction][metric], expected_retrieval[direction][metric])
        for direction in ("V", "H", "macro"):
            for metric in ("ap", "auroc", "balanced_accuracy"):
                self.assertAlmostEqual(actual_adjacency[direction][metric], expected_adjacency[direction][metric])

    def test_threshold_selection_matches_dev_brute_force_and_predeclared_tie(self):
        scores = np.random.default_rng(47).integers(-3, 4, size=self.labels.shape).astype(float)
        thresholds = select_thresholds(scores, self.labels)
        for channel in range(2):
            y = self.labels[..., channel]
            values = scores[..., channel]
            candidates = sorted(np.unique(values), reverse=True)
            qualities = []
            for threshold in candidates:
                predicted = values >= threshold
                qualities.append((predicted[y].mean() + (~predicted[~y]).mean()) / 2)
            best = max(qualities)
            expected = next(value for value, quality in zip(candidates, qualities) if abs(quality - best) < 1e-14)
            self.assertEqual(thresholds[channel], expected)
        labels = np.zeros((1, 72, 2), dtype=bool)
        labels[:, :6] = True
        tied = np.zeros_like(labels, dtype=float)
        tied[:, :3] = 1
        tied[:, 6:39] = 1
        self.assertEqual(select_thresholds(tied, labels), [1.0, 1.0])

    def test_ap_and_auc_match_sklearn_when_available(self):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            self.skipTest("optional sklearn oracle is not installed")
        rng = np.random.default_rng(219)
        for scores in (rng.normal(size=self.labels.shape), rng.integers(-2, 3, size=self.labels.shape)):
            actual = adjacency_metrics(scores, self.labels)
            for channel, name in enumerate(("V", "H")):
                y = self.labels[..., channel].ravel()
                values = scores[..., channel].ravel()
                self.assertAlmostEqual(actual[name]["ap"], average_precision_score(y, values), places=14)
                self.assertAlmostEqual(actual[name]["auroc"], roc_auc_score(y, values), places=14)

    def test_cosine_has_no_direction_and_handles_zero_and_large_vectors(self):
        features = np.random.default_rng(8).normal(size=(3, 9, 7))
        features[0, 0] = 0
        features[1] *= 1e300
        scores = cosine_pair_scores(features)
        self.assertEqual(scores.shape, (3, 72, 2))
        self.assertTrue(np.all(np.isfinite(scores)))
        np.testing.assert_array_equal(scores[..., 0], scores[..., 1])
        pairs = ordered_pairs()
        lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
        reverse = [lookup[(j, i)] for i, j in pairs]
        np.testing.assert_allclose(scores, scores[:, reverse], atol=1e-15)
        np.testing.assert_array_equal(scores[0, (pairs == 0).any(axis=1)], 0)

    def test_strict_shapes_finite_scores_natural_prevalence_and_valid_queries(self):
        valid_scores = np.zeros(self.labels.shape)
        for bad_scores in (np.zeros((3, 71, 2)), np.full(self.labels.shape, np.nan), np.full(self.labels.shape, np.inf)):
            with self.subTest(scores=bad_scores.shape), self.assertRaises(ValueError):
                adjacency_metrics(bad_scores, self.labels)
        with self.assertRaises(ValueError):
            adjacency_metrics(valid_scores, np.zeros_like(self.labels))
        with self.assertRaises(ValueError):
            adjacency_metrics(valid_scores, self.labels, [np.inf, 0])
        with self.assertRaises(ValueError):
            adjacency_metrics(valid_scores, self.labels, [0])
        with self.assertRaises(ValueError):
            cosine_pair_scores(np.ones((3, 8, 7)))
        with self.assertRaises(ValueError):
            cosine_pair_scores(np.full((3, 9, 7), np.inf))
        changed = copy.deepcopy(self.design)
        changed["query_pairs"][0, 0, 0] = changed["query_pairs"][0, 0, 1]
        with self.assertRaises(ValueError):
            retrieval_metrics(valid_scores, changed)
        changed = copy.deepcopy(self.design)
        changed["query_positive"][0, 0] = True
        with self.assertRaises(ValueError):
            retrieval_metrics(valid_scores, changed)
        changed = copy.deepcopy(self.design)
        changed["query_channels"][0, 0] = 1 - changed["query_channels"][0, 0]
        with self.assertRaises(ValueError):
            retrieval_metrics(valid_scores, changed)
        with self.assertRaises(ValueError):
            retrieval_metrics(valid_scores, {})


if __name__ == "__main__":
    unittest.main()
