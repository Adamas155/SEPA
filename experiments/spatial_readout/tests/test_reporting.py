"""Report-only fixtures check interpretation boundaries, not scientific results."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.spatial_readout import reporting


def fixtures():
    manifest = {
        "status": "completed", "engineering_only": False,
        "counts": {"train": 113000, "dev": 12000, "test": 5000},
        "checkpoints": [], "checks": {"fixture_check": True},
        "resource_plan": {"estimated_feature_bytes": 1234567},
        "config": {
            "probe_seeds": [0, 1, 2], "heads": ["linear"],
            "probe": {"epochs": 30, "learning_rates": [0.01]},
            "data": {"split_seed": 41, "design_seed": 42},
        },
        "split_manifest_hash": "fixture-split-identity",
        "design_hashes": {"test": "fixture-test-design"},
        "limitations": ["Test fixture only; no research execution."],
    }
    values = {
        "ijepa-e100": 0.2, "v1-k0-e100": 0.3, "v1-k3-e100": 0.4,
        "ijepa-e25": 0.85, "v2-k0-e25": 0.75, "v2-k3-e25": 0.65,
        "color_lowfreq": 0.1, "border_only": 0.15,
    }
    rows = []
    for group, (epochs, *models) in reporting.GROUPS.items():
        for model in models:
            manifest["checkpoints"].append({
                "name": model, "group": group, "status": "verified", "epochs": epochs,
                "pretraining_seed": 0, "path": f"/fixture/{model}.pt",
                "file_hash": f"fixture-{model}-identity", "model_config": {"dim": 384},
                "pe_protocol": {"name": "fixture-content-only"},
                "metadata": {"name": "raw-checkpoint-name", "epochs": 999},
            })
    for model, value in values.items():
        if model in reporting.REFERENCES:
            group, epochs, pretraining_seed, family = "reference", None, None, "pixel_control"
            dim = 54 if model == "color_lowfreq" else 192
        else:
            group = "A" if model.endswith("e100") else "B"
            epochs, pretraining_seed, dim = reporting.GROUPS[group][0], 0, 384
            family = model.split("-")[0]
        for seed in (0, 1, 2):
            for condition in ("clean", "border_masked"):
                offset = (seed - 1) * (0.02 if "k3" in model else 0.01)
                offset -= 0.05 if condition == "border_masked" else 0
                for task, direction, metric, _ in reporting.SUMMARY_METRICS:
                    rows.append({
                        "comparison_group": group, "model": model, "family": family,
                        "pretraining_epochs": epochs, "pretraining_seed": pretraining_seed,
                        "probe_seed": seed, "head": "linear", "condition": condition,
                        "task": task, "direction": direction, "metric": metric,
                        "value": value + offset, "n_images": 5000, "feature_dim": dim,
                        "parameter_count": 4 * dim + 2, "engineering_only": False,
                    })
    return manifest, rows


def render(manifest, rows, *, commands=None):
    with tempfile.TemporaryDirectory() as directory:
        path = reporting.write_report(directory, manifest, rows, commands=commands)
        return path.read_text(encoding="utf-8")


class ReportInterpretationTests(unittest.TestCase):
    def test_training_budgets_are_separated_and_differences_are_paired(self):
        manifest, rows = fixtures()
        report = render(manifest, rows)
        section_a = report.split("### A 组：100 轮预训练", 1)[1].split("### B 组：25 轮预训练", 1)[0]
        section_b = report.split("### B 组：25 轮预训练", 1)[1].split("### 共享的像素捷径参考", 1)[0]
        self.assertIn("ijepa-e100", section_a)
        self.assertNotIn("ijepa-e25", section_a)
        self.assertIn("ijepa-e25", section_b)
        self.assertNotIn("ijepa-e100", section_b)
        self.assertIn("v1-k3-e100 − ijepa-e100 | clean | macro AP | +0.200000 ± 0.010000", report)
        self.assertIn("v2-k3-e25 − ijepa-e25 | clean | macro AP | -0.200000 ± 0.010000", report)
        self.assertNotIn("v1-k3-e100 − ijepa-e25", report)
        self.assertNotIn("v2-k3-e25 − ijepa-e100", report)
        self.assertIn("| v1-k3-e100 / linear | macro AP | -0.050000 ± 0.000000 |", report)

    def test_engineering_smoke_never_generates_ranking_or_research_conclusions(self):
        manifest, rows = fixtures()
        manifest.update(status="engineering_smoke_only", engineering_only=True)
        manifest["counts"] = {"train": 32, "dev": 16, "test": 16}
        for row in rows:
            row.update(engineering_only=True, n_images=16)
        report = render(manifest, rows)
        self.assertNotIn("### A 组", report)
        self.assertNotIn("### 配对差值", report)
        self.assertNotIn("+0.200000", report)
        self.assertIn("尚未在完整协议上验证", report)
        self.assertIn("工程 smoke 只检查流程", report)
        self.assertIn("不能据此给模型排名或估计正式收益", report)

    def test_incomplete_or_mismatched_probe_seeds_do_not_get_a_comparison(self):
        manifest, rows = fixtures()
        rows = [row for row in rows if not (row["model"] == "v1-k0-e100" and row["probe_seed"] == 2)]
        report = render(manifest, rows)
        self.assertIn("v1-k3-e100 − v1-k0-e100 | clean | macro AP | 无法比较", report)
        self.assertIn("至少 3 个完全匹配的 probe seed", report)
        self.assertIn("现有 [0, 1, 2] / [0, 1]", report)

    def test_absent_pretraining_seed_is_not_inferred_from_probe_seed(self):
        manifest, rows = fixtures()
        for row in rows:
            if row["model"] == "v1-k0-e100":
                row["pretraining_seed"] = None
        report = render(manifest, rows)
        self.assertIn("v1-k0-e100：指标未记录匹配的预训练轮数和 seed=0", report)
        self.assertIn("v1-k3-e100 − v1-k0-e100 | clean | macro AP | 无法比较", report)

    def test_invalid_checkpoint_and_wrong_budget_rows_are_excluded(self):
        manifest, rows = fixtures()
        for checkpoint in manifest["checkpoints"]:
            if checkpoint["name"] == "v1-k0-e100":
                checkpoint.update(status="missing", error="Fixture checkpoint absent")
        for row in rows:
            if row["model"] == "ijepa-e100":
                row["pretraining_epochs"] = 25
        report = render(manifest, rows)
        self.assertIn("v1-k0-e100 | A | missing | 未记录 | 未记录", report)
        self.assertIn("Fixture checkpoint absent", report)
        self.assertIn("v1-k3-e100 − ijepa-e100 | clean | macro AP | 无法比较", report)
        self.assertIn("v1-k3-e100 − v1-k0-e100 | clean | macro AP | 无法比较", report)

    def test_cosine_is_direction_free_and_has_no_artificial_probe_seed_sd(self):
        manifest, rows = fixtures()
        cosine_rows = []
        for row in rows:
            if row["model"] == "ijepa-e100" and row["probe_seed"] == 0:
                cosine_rows.append({**row, "head": "cosine", "probe_seed": None, "parameter_count": 0})
        report = render(manifest, rows + cosine_rows)
        section = report.split("cosine（无训练、无方向区分能力）：", 1)[1].split("### B 组", 1)[0]
        self.assertIn("不训练 head", section)
        self.assertNotIn("±", section)
        self.assertNotIn("SD", section)
        self.assertIn("确定性 cosine", report)

    def test_optional_mlp_is_separate_and_caveats_are_preserved(self):
        manifest, rows = fixtures()
        mlp = [{**row, "head": "mlp", "parameter_count": row["feature_dim"] * 2 * 128 + 128 + 258}
               for row in rows]
        report = render(manifest, rows + mlp)
        for phrase in (
            "linear：", "mlp：", "无法学习任意 pair 交互", "输入分布变化",
            "不是完整 missing + shuffle", "共享 head", "1/12", "1/8",
            "不能单独证明没有使用边界", "也移除真实视觉内容", "test 不参与选择",
            "不衡量预训练 seed 方差", "未计算置信区间或 p 值",
        ):
            self.assertIn(phrase, report)
        self.assertIn("| v1-k3-e100 | mlp | 384 | 768 | 98690 |", report)
        self.assertIn("不代表跨预训练 seed 的统计稳定优势", report)

    def test_short_test_or_engineering_rows_cannot_masquerade_as_complete(self):
        manifest, rows = fixtures()
        short = copy.deepcopy(manifest)
        short["counts"]["test"] = 4999
        self.assertNotIn("### 配对差值", render(short, rows))
        rows[0]["engineering_only"] = True
        self.assertNotIn("### 配对差值", render(manifest, rows))

    def test_blocked_report_keeps_missing_evidence_and_pending_commands_literal(self):
        manifest = {
            "status": "blocked", "engineering_only": False, "counts": {},
            "checkpoints": [], "checks": {}, "error": "No checkpoint paths supplied",
            "executed_command": "python -m example audit --fixture-only",
        }
        command = "python -m example --literal '`$(unexecuted)`'\n```"
        with patch("subprocess.run") as run:
            report = render(manifest, [], commands={"formal pending": command})
        run.assert_not_called()
        self.assertIn(command, report)
        self.assertIn("No checkpoint paths supplied", report)
        self.assertIn("实际执行的命令（按运行 manifest 原文记录）：", report)
        self.assertIn("python -m example audit --fixture-only", report)
        self.assertIn("## 2. 尚未运行的命令", report)
        self.assertIn("未提供 checkpoint 清单", report)
        self.assertIn("4. 边界遮挡后优势是否仍存在？尚未在完整协议上验证。", report)
        self.assertNotIn("预计提升", report)


class ArtifactTests(unittest.TestCase):
    def test_csv_json_roundtrip_and_no_existing_artifact_is_overwritten(self):
        manifest, rows = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = reporting.write_metrics(directory, rows)
            self.assertTrue(csv_path.is_absolute())
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), rows)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), len(rows))
            self.assertEqual(csv_rows[0]["probe_seed"], "0")
            before = csv_path.read_bytes()
            with self.assertRaises(FileExistsError):
                reporting.write_metrics(directory, rows)
            self.assertEqual(csv_path.read_bytes(), before)
            report = reporting.write_report(directory, manifest, rows)
            report_before = report.read_bytes()
            with self.assertRaises(FileExistsError):
                reporting.write_report(directory, manifest, [])
            self.assertEqual(report.read_bytes(), report_before)

    def test_empty_metrics_still_have_a_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = reporting.write_metrics(directory, [])
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), [])
            self.assertEqual(csv_path.read_text(encoding="utf-8").strip(), ",".join(reporting.COLUMNS))

    def test_nonfinite_and_duplicate_rows_rejected_before_writing(self):
        _, rows = fixtures()
        invalid = [{**rows[0], "value": float("nan")}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                reporting.write_metrics(directory, invalid)
            with self.assertRaises(ValueError):
                reporting.write_metrics(directory, [rows[0], rows[0]])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
