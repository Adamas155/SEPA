"""Small pixel-only orchestration fixtures, never pretrained-model evidence."""

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.spatial_readout import PROJECT_ROOT
from experiments.spatial_readout.common import digest, read_json
from experiments.spatial_readout.features import PixelFeatures, load_fitted
from experiments.spatial_readout import pipeline
from experiments.spatial_readout.tasks import make_design


class PipelineTests(unittest.TestCase):
    def test_selection_precedes_test_and_release_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests, designs, hashes = {}, {}, {}
            rng = np.random.default_rng(3)
            for role, n in (("train", 4), ("dev", 2), ("test", 2)):
                records = []
                for i in range(n):
                    path = root / f"{role}-{i}.png"
                    Image.fromarray(rng.integers(0, 256, (87, 104, 3), dtype=np.uint8)).save(path)
                    identity = hashlib.sha256(path.read_bytes()).hexdigest()
                    records.append({"path": path.name, "sha256": identity, "id": identity, "label": i % 2})
                manifest = {"root": str(root), "records": records, "role": role}
                manifest["fingerprint"] = digest(manifest)
                manifests[role] = manifest
                designs[role] = make_design([r["id"] for r in records], seed=8)
                hashes[role] = digest([role, "fixture-design"])
            config = {
                "heads": ["linear"], "probe_seeds": [0, 1, 2],
                "probe": {"epochs": 1, "patience": 1, "learning_rates": [0.01], "batch_images": 2},
                "extraction": {"device": "cpu", "batch_images": 2, "tile_batch_size": 9, "workers": 0},
                "smoke": {"epochs": 1, "patience": 1},
            }
            original_extract = pipeline.extract_cached
            calls = []
            def observed(model, dataset, path, identity, **kwargs):
                calls.append(Path(path).stem)
                if Path(path).name.startswith("test-"):
                    self.assertTrue((Path(path).parent / "selection_complete.json").is_file())
                    self.assertEqual(len(list(Path(path).parent.glob("head-linear-s*.pt"))), 3)
                return original_extract(model, dataset, path, identity, **kwargs)
            with patch.object(pipeline, "extract_cached", observed), redirect_stdout(io.StringIO()):
                rows, checks = pipeline.evaluate_one(
                    PixelFeatures("border_only"), "border_only", "reference", config, root,
                    {"manifests": manifests}, designs, hashes, smoke=True, release_features=True)
            self.assertEqual(calls, ["train-clean", "dev-clean", "test-clean", "test-border_masked"])
            self.assertEqual(len(rows), 114)
            self.assertTrue(all(r["engineering_only"] for r in rows))
            self.assertEqual({r["probe_seed"] for r in rows}, {0, 1, 2})
            self.assertTrue(checks["encoder_unchanged_during_probe"])
            self.assertTrue(checks["head_container_reordering"])
            result = root / "border_only"
            self.assertFalse((result / "train-clean.npy").exists())
            self.assertTrue((result / "train-clean.json").is_file())
            self.assertTrue((result / "scores-linear-s0-clean.npy").is_file())
            fitted = load_fitted(result / "head-linear-s0.pt")
            self.assertFalse(fitted["training_identity"]["test_used_for_selection"])
            self.assertTrue(read_json(result / "feature_release.json")["released"])

    def test_existing_result_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                pipeline.execute({}, directory, action="audit")

    def test_declared_groups_seeds_and_resource_units(self):
        config = pipeline.resolved_config(PROJECT_ROOT / "experiments/spatial_readout/config.json", device="cpu")
        self.assertEqual(config["probe_seeds"], [0, 1, 2])
        self.assertEqual({x["epochs"] for x in config["checkpoints"] if x["group"] == "A"}, {100})
        self.assertEqual({x["epochs"] for x in config["checkpoints"] if x["group"] == "B"}, {25})
        plan = pipeline.resource_plan({"train": 100, "dev": 20, "test": 5000}, config)
        self.assertEqual(plan["test_pairs_per_model_condition"], 360000)
        self.assertEqual(plan["test_retrieval_queries_per_model_condition"], 120000)
        self.assertEqual(plan["full_fp32_encoder_cache_bytes_each"], (5120 + 5000) * 9 * 384 * 4)


if __name__ == "__main__":
    unittest.main()
