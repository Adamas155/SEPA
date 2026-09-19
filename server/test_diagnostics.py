"""CPU regression tests for the new diagnostic protocol (no dataset required)."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from diagnostic_lib import (cached_extract, encode_branch, feature_health, fit_trajectory, grouped_split,
                            health_indices, record_audit, save_tensor, select_recipe, validate_features)
from sepa_plan_b.config import Config, Model, Probe
from sepa_plan_b.engine import setup_runtime
from sepa_plan_b.evaluation import fit_linear
from sepa_plan_b.model import SEPA
import torch


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = Config(model=Model(name="tiny", predictor_dim=32, predictor_depth=1, predictor_heads=4),
                            probe=Probe(epochs=3, batch_size=16))
        setup_runtime(cls.config)

    def test_student_teacher_dispatch_and_no_state_mutation(self):
        for method in ("sepa", "full"):
            model = SEPA(self.config, seed=0, method=method).eval()
            with torch.no_grad():
                model.teacher.norm.bias.add_(0.25)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            tiles = torch.randn(2, 9, 3, 80, 80)
            with patch.object(model.predictor, "forward", side_effect=AssertionError("Predictor must not run")):
                student = encode_branch(model, tiles, "student")
                teacher = encode_branch(model, tiles, "teacher")
            with torch.no_grad():
                expected = model.teacher(tiles, torch.arange(9).expand(2, -1)) if method == "full" else model.teacher(tiles)
                self.assertTrue(torch.equal(student, model.encode(tiles)))
                self.assertTrue(torch.equal(teacher, expected))
            self.assertFalse(torch.equal(student, teacher))
            self.assertFalse(student.requires_grad or teacher.requires_grad)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):
            encode_branch(model, tiles, "invalid")

    def test_random_initialization_is_training_initialization(self):
        for method in ("sepa", "full"):
            first = SEPA(self.config, 0, method)
            torch.randn(100)
            second = SEPA(self.config, 0, method)
            self.assertTrue(all(torch.equal(v, second.encoder.state_dict()[k]) for k, v in first.encoder.state_dict().items()))
            self.assertTrue(all(torch.equal(v, first.teacher.state_dict()[k]) for k, v in first.encoder.state_dict().items()))

    def test_instrumented_fit_exactly_matches_original(self):
        rng = torch.Generator().manual_seed(25)
        x = torch.randn(73, 12, generator=rng)
        y = torch.randint(4, (73,), generator=rng)
        original, mean, std = fit_linear(x, y, 4, self.config, 0, "cpu")
        observed, new_mean, new_std, trace = fit_trajectory(x, y, 4, self.config, 0, "cpu", dev=(x[:8], y[:8]), marks=(1, 3))
        self.assertTrue(torch.equal(mean, new_mean) and torch.equal(std, new_std))
        self.assertTrue(all(torch.equal(v, observed.state_dict()[k]) for k, v in original.state_dict().items()))
        self.assertEqual([r["epoch"] for r in trace if "dev_scores" in r], [1, 3])

    def test_dev_features_do_not_fit_normalization_or_head(self):
        x = torch.arange(60).reshape(20, 3).float()/10
        y = torch.arange(20) % 2
        a = fit_trajectory(x, y, 2, self.config, 0, "cpu", dev=(x, y), marks=(1, 3))
        b = fit_trajectory(x, y, 2, self.config, 0, "cpu", dev=(x*1000, y), marks=(1, 3))
        self.assertTrue(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
        self.assertTrue(all(torch.equal(v, b[0].state_dict()[k]) for k, v in a[0].state_dict().items()))

    def test_group_split_and_health_sample(self):
        records = [{"id": f"{c}-{i}", "label": c} for c in range(3) for i in range(20)]
        records += [dict(records[i]) for i in range(0, 60, 3)]
        train, dev = grouped_split(records)
        second = grouped_split(records)
        self.assertTrue(torch.equal(train, second[0]) and torch.equal(dev, second[1]))
        self.assertEqual(set(train.tolist()) | set(dev.tolist()), set(range(len(records))))
        self.assertFalse({records[i]["id"] for i in train} & {records[i]["id"] for i in dev})
        self.assertEqual({records[i]["label"] for i in dev}, {0, 1, 2})
        self.assertEqual({records[i]["label"] for i in train}, {0, 1, 2})
        ids = health_indices(records, per_class=8)
        self.assertEqual(len(ids), 24)
        self.assertEqual(len({records[i]["id"] for i in ids}), 24)
        with self.assertRaises(ValueError):
            grouped_split(records + [{"id": "0-0", "label": 2}])

    def test_ambiguous_groups_are_explicitly_excluded_from_both_partitions(self):
        records = [{"id": f"{c}-{i}", "label": c} for c in range(3) for i in range(20)]
        records.append({"id": "0-0", "label": 2})
        audit = record_audit(records)
        self.assertEqual(audit["conflicting_records"], 2)
        self.assertEqual(set(audit["conflicting_groups"]), {"0-0"})
        train, dev = grouped_split(records, excluded_ids=audit["conflicting_groups"])
        used = set(train.tolist()) | set(dev.tolist())
        self.assertEqual(used, set(range(1, 60)))
        ids = health_indices(records, per_class=100, excluded_ids=audit["conflicting_groups"])
        self.assertEqual(set(ids.tolist()), used)

    def test_selection_is_shared_dev_only_and_ties_are_fixed(self):
        rows = []
        for lr, epoch, scores in [(0.05, 50, (0.2, 0.4)), (0.005, 50, (0.3, 0.3)), (0.1, 200, (0.3, 0.3))]:
            for anchor, score in zip(("k0_student", "full_student"), scores):
                rows.append({"learning_rate": lr, "epoch": epoch, "anchor": anchor,
                             "dev_scores": {"top1": score}, "official_val_top1": 0.99 if lr == 0.1 else 0.0})
        result = select_recipe(rows)
        self.assertEqual((result["selected"]["learning_rate"], result["selected"]["epochs"]), (0.005, 50))
        self.assertFalse(result["official_validation_used_for_selection"])
        with self.assertRaises(ValueError):
            select_recipe(rows[:-1])

    def test_spectrum_detects_constant_and_rank_one(self):
        y = torch.arange(100) % 2
        constant = feature_health(torch.full((100, 8), 0.1), y)
        self.assertEqual(constant["effective_rank"], 0)
        rank_one = feature_health(torch.arange(100).float()[:, None].expand(-1, 8), y)
        self.assertAlmostEqual(rank_one["effective_rank"], 1, places=5)
        random = feature_health(torch.randn(100, 8, generator=torch.Generator().manual_seed(3)), y)
        self.assertGreater(random["effective_rank"], 6)
        with self.assertRaises(FloatingPointError):
            feature_health(torch.full((100, 8), float("nan")), y)

    def test_cache_identity_label_order_and_checksum(self):
        records = [{"id": "a", "label": 0}, {"id": "b", "label": 1}]
        manifest = {"records": records}
        meta = {"variant": "k0_teacher", "manifest": "test"}
        payload = {"features": torch.ones(2, 32), "labels": torch.tensor([0, 1]), "ids": ["a", "b"], "metadata": meta}
        validate_features(payload, meta, manifest, 32)
        for bad in ({**meta, "variant": "k0_student"}, {**meta, "manifest": "other"}):
            with self.assertRaises(ValueError):
                validate_features(payload, bad, manifest, 32)
        with self.assertRaises(ValueError):
            validate_features({**payload, "ids": ["b", "a"]}, meta, manifest, 32)
        with self.assertRaises(ValueError):
            validate_features({**payload, "labels": torch.tensor([1, 0])}, meta, manifest, 32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.pt"
            save_tensor(path, payload)
            from diagnostic_lib import file_sha
            from sepa_plan_b.data import write_json
            write_json(path.with_suffix(".json"), {"metadata": meta, "sha256": file_sha(path)})
            loaded = cached_extract(path, None, "teacher", manifest, self.config, "cpu", meta, lambda **kw: None)
            self.assertTrue(torch.equal(loaded["features"], payload["features"]))
            with path.open("ab") as f:
                f.write(b"tampered")
            with self.assertRaises(ValueError):
                cached_extract(path, None, "teacher", manifest, self.config, "cpu", meta, lambda **kw: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
