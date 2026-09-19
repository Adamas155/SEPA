"""Small registry for the existing server scripts, without importing them.

Scripts keep their original paths and subprocess behavior. The launcher adds
only the project argument and import path that several scripts require.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Experiment:
    script: str
    description: str
    group: str
    project_argument: bool = False
    upstream_required: bool = False


EXPERIMENTS = {
    "prepare-data": Experiment(
        "prepare_public_imagenet100.py", "准备服务器 ImageNet-100 清单", "data", True
    ),
    "v1-100": Experiment("run_long100.py", "V1 四组 100 轮预训练", "train", True),
    "v1-evaluate": Experiment(
        "evaluate_long100.py", "V1 里程碑冻结评估", "evaluate", True
    ),
    "ijepa": Experiment(
        "ijepa_baseline.py", "同规模 I-JEPA 基线", "train", upstream_required=True
    ),
    "ijepa-fov": Experiment(
        "evaluate_ijepa_fov.py",
        "固定 I-JEPA 权重，比较推理视野",
        "evaluate",
        upstream_required=True,
    ),
    "ijepa-local": Experiment(
        "train_ijepa_local_student.py",
        "局部 student／全局 teacher 消融",
        "train",
        upstream_required=True,
    ),
    "v2": Experiment(
        "train_ijepa_v2.py",
        "V2 25 轮筛选；支持 --smoke、--resume、--evaluate",
        "train",
        upstream_required=True,
    ),
}


def project_root():
    return Path(__file__).resolve().parents[3]


def build_command(name, arguments, root=None):
    root = Path(root) if root is not None else project_root()
    entry = EXPERIMENTS[name]
    script = root / "server" / entry.script
    if not script.is_file():
        raise FileNotFoundError(f"Server entry point is missing: {script}")
    if any(
        value == "--project" or value.startswith("--project=") for value in arguments
    ):
        raise ValueError("The launcher supplies --project; do not pass it again")
    command = [sys.executable, "-u", str(script)]
    if entry.project_argument:
        command.extend(("--project", str(root)))
    return command + list(arguments)


def child_environment(root):
    environment = os.environ.copy()
    paths = [str(root / "src"), str(root / "server")]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def execute(args, forwarded, parser):
    if args.action == "list":
        for name, entry in EXPERIMENTS.items():
            print(f"{name:22} [{entry.group}] {entry.description}")
        return 0

    root = project_root()
    entry = EXPERIMENTS[args.name]
    if args.action == "show":
        print(
            json.dumps(
                {"name": args.name, **asdict(entry), "project": str(root)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        command = build_command(args.name, forwarded, root)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(
            json.dumps(
                {"cwd": str(root), "command": command, "executed": False},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if sys.platform == "win32":
        parser.error(
            "Server workflows require Linux or WSL; train/probe commands also support Windows"
        )
    if (
        entry.upstream_required
        and not (root / "vendor/ijepa-52c1ae9/UPSTREAM.json").is_file()
    ):
        parser.error(
            "This workflow requires the recorded I-JEPA snapshot under vendor/ijepa-52c1ae9"
        )
    completed = subprocess.run(
        command, cwd=root, env=child_environment(root), check=False
    )
    return completed.returncode
