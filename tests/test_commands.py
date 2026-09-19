"""CLI regression checks: planning must not train, and launch arguments stay intact."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sepa_plan_b.commands import main
from sepa_plan_b.commands.experiments import (
    EXPERIMENTS,
    build_command,
    child_environment,
)


def invoke(arguments):
    output = io.StringIO()
    with redirect_stdout(output):
        code = main(arguments)
    return code, output.getvalue()


class DiscoveryTests(unittest.TestCase):
    def test_help_and_experiment_discovery_need_only_standard_library(self):
        for arguments in (
            ["--help"],
            ["experiment", "list"],
            ["experiment", "show", "v2"],
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "-S", str(ROOT / "run.py"), *arguments],
                    cwd=tempfile.gettempdir(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_registered_scripts_exist(self):
        for entry in EXPERIMENTS.values():
            self.assertTrue((ROOT / "server" / entry.script).is_file(), entry.script)

    def test_removed_workflows_are_unavailable(self):
        _, current = invoke(["experiment", "list"])
        for name in (
            "pilot",
            "pilot-summary",
            "diagnostics",
            "diagnostics-serial",
            "diagnostics-recover",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, current)
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main(["experiment", "run", name, "--dry-run"])
                self.assertEqual(error.exception.code, 2)

    def test_forwarding_is_rejected_outside_experiment_run(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["experiment", "show", "v2", "--", "--resume"])
        self.assertEqual(error.exception.code, 2)


class LauncherTests(unittest.TestCase):
    def test_dry_run_does_not_spawn_or_request_power_management(self):
        with patch("sepa_plan_b.commands.experiments.subprocess.run") as spawn:
            code, output = invoke(
                ["experiment", "run", "v1-100", "--dry-run", "--", "--workers", "2"]
            )
        plan = json.loads(output)
        spawn.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(plan["executed"])
        self.assertEqual(
            plan["command"][-4:], ["--project", str(ROOT), "--workers", "2"]
        )
        self.assertNotIn("--shutdown-on-success", plan["command"])

    def test_command_preserves_spaces_and_forwards_each_argument_separately(self):
        with tempfile.TemporaryDirectory(prefix="sepa project ") as directory:
            root = Path(directory)
            (root / "server").mkdir()
            (root / "server" / EXPERIMENTS["v2"].script).touch()
            arguments = ["--worker", "--arm", "k3", "--resume"]
            command = build_command("v2", arguments, root)
        self.assertEqual(command[3:], arguments)
        self.assertEqual(command[2], str(root / "server/train_ijepa_v2.py"))

    def test_project_override_is_rejected(self):
        for arguments in (["--project", "/somewhere"], ["--project=/somewhere"]):
            with self.assertRaises(ValueError):
                build_command("v1-100", arguments)

    def test_child_environment_preserves_existing_import_paths(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            environment = child_environment(ROOT)
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep),
            [str(ROOT / "src"), str(ROOT / "server"), "/existing/path"],
        )

    def test_exit_code_and_working_directory_are_preserved(self):
        with (
            patch("sepa_plan_b.commands.experiments.sys.platform", "linux"),
            patch("sepa_plan_b.commands.experiments.subprocess.run") as spawn,
        ):
            spawn.return_value.returncode = 7
            code, _ = invoke(["experiment", "run", "v1-100", "--", "--workers", "1"])
        self.assertEqual(code, 7)
        self.assertEqual(spawn.call_args.kwargs["cwd"], ROOT)
        self.assertFalse(spawn.call_args.kwargs["check"])

    def test_missing_upstream_is_reported_before_process_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "server").mkdir()
            (root / "server" / EXPERIMENTS["v2"].script).touch()
            with (
                patch(
                    "sepa_plan_b.commands.experiments.project_root", return_value=root
                ),
                patch("sepa_plan_b.commands.experiments.sys.platform", "linux"),
                patch("sepa_plan_b.commands.experiments.subprocess.run") as spawn,
            ):
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main(["experiment", "run", "v2"])
                self.assertEqual(error.exception.code, 2)
                spawn.assert_not_called()


class MatrixTests(unittest.TestCase):
    def test_new_entry_runs_the_existing_resume_and_probe_integration_case(self):
        import test_contracts

        case = test_contracts.PipelineTests()
        case.setUp()
        try:
            with patch("test_contracts.main", main):
                case.test_matrix_resumes_existing_and_starts_missing_runs()
        finally:
            case.tearDown()

    def test_default_plan_has_one_seed_and_never_loads_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            with (
                patch("sepa_plan_b.data.manifests") as data,
                patch("sepa_plan_b.engine.train") as train,
            ):
                code, _ = invoke(
                    [
                        "matrix",
                        "--config",
                        str(ROOT / "configs/smoke.toml"),
                        "--include-full",
                        "--output",
                        str(output),
                    ]
                )
            plan = json.loads(output.read_text())
        data.assert_not_called()
        train.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(plan["n_runs"], 4)
        self.assertEqual({job["seed"] for job in plan["jobs"]}, {0})
        self.assertFalse(plan["executed"])

    def test_explicit_multiple_seeds_keep_the_paired_arm_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            invoke(
                [
                    "matrix",
                    "--config",
                    str(ROOT / "configs/smoke.toml"),
                    "--seeds",
                    "0,1",
                    "--ks",
                    "0,3",
                    "--output",
                    str(output),
                ]
            )
            jobs = json.loads(output.read_text())["jobs"]
        self.assertEqual(
            [(job["k"], job["seed"]) for job in jobs], [(0, 0), (0, 1), (3, 0), (3, 1)]
        )

    def test_invalid_matrix_does_not_write_a_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            for arguments in (
                ["--seeds", "0,0"],
                ["--seeds", "bad"],
                ["--ks", "1"],
                ["--seeds", "-1"],
            ):
                with (
                    self.subTest(arguments=arguments),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    invoke(
                        [
                            "matrix",
                            "--config",
                            str(ROOT / "configs/smoke.toml"),
                            "--output",
                            str(output),
                            *arguments,
                        ]
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
