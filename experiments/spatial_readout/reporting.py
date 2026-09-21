"""Write isolated metrics and a conservative report from recorded evidence only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import statistics


COLUMNS = (
    "comparison_group", "model", "family", "pretraining_epochs", "pretraining_seed",
    "probe_seed", "head", "condition", "task", "direction", "metric", "value",
    "n_images", "feature_dim", "parameter_count", "engineering_only",
)
GROUPS = {
    "A": (100, "ijepa-e100", "v1-k0-e100", "v1-k3-e100"),
    "B": (25, "ijepa-e25", "v2-k0-e25", "v2-k3-e25"),
}
REFERENCES = ("color_lowfreq", "border_only")
SUMMARY_METRICS = (
    ("adjacency", "V", "ap", "V AP"),
    ("adjacency", "H", "ap", "H AP"),
    ("adjacency", "macro", "ap", "macro AP"),
    ("adjacency", "macro", "auroc", "macro AUROC"),
    ("adjacency", "macro", "balanced_accuracy", "macro balanced accuracy"),
    ("retrieval", "macro", "recall_at_1", "macro Recall@1"),
    ("retrieval", "macro", "mrr", "macro MRR"),
)
PRIMARY_METRICS = (
    ("adjacency", "macro", "ap", "macro AP"),
    ("retrieval", "macro", "recall_at_1", "macro Recall@1"),
)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False)


def _rows(rows):
    result, seen = [], set()
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"Metric row {index} is not a mapping")
        row = _plain(source)
        missing = set(COLUMNS) - set(row)
        if missing:
            raise ValueError(f"Metric row {index} is missing fields: {sorted(missing)}")
        if row["comparison_group"] not in {"A", "B", "reference"}:
            raise ValueError("Metrics must belong to A, B, or reference")
        if row["head"] not in {"linear", "mlp", "cosine"}:
            raise ValueError("Unknown probe head")
        if row["condition"] not in {"clean", "border_masked"}:
            raise ValueError("Unknown test condition")
        if row["task"] not in {"adjacency", "retrieval"}:
            raise ValueError("Unknown spatial readout task")
        value = row["value"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Metric values must be numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Metric values must be finite and in [0, 1]")
        if not isinstance(row["engineering_only"], bool):
            raise ValueError("engineering_only must be a boolean")
        for key in ("n_images", "feature_dim", "parameter_count"):
            value = row[key]
            minimum = 0 if key == "parameter_count" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("pretraining_epochs", "pretraining_seed", "probe_seed"):
            value = row[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"{key} must be a nonnegative integer or null")
        identity = tuple(row[key] for key in (
            "comparison_group", "model", "head", "condition", "task", "direction",
            "metric", "probe_seed",
        ))
        if identity in seen:
            raise ValueError(f"Duplicate metric identity: {identity}")
        seen.add(identity)
        result.append(row)
    return result


def _write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return path.resolve()


def write_metrics(output_dir, rows):
    """Return ``(metrics.csv, metrics.json)``; never overwrite an existing file."""
    rows = _rows(rows)
    root = Path(output_dir)
    csv_path, json_path = root / "metrics.csv", root / "metrics.json"
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(path)
    extra = sorted(set().union(*(set(row) for row in rows)) - set(COLUMNS)) if rows else []
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[*COLUMNS, *extra], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    json_content = _json(rows) + "\n"
    _write_new(csv_path, buffer.getvalue())
    try:
        _write_new(json_path, json_content)
    except BaseException:
        csv_path.unlink()
        raise
    return csv_path.resolve(), json_path.resolve()


def _cell(value):
    if value is None:
        return "未记录"
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _fence(content, language=""):
    content = str(content)
    longest = max((len(match) for match in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{content}\n{fence}"


def _checkpoint(item):
    actual = item.get("actual", item.get("metadata", {}))
    actual = actual if isinstance(actual, Mapping) else {}
    # Adapter manifests carry verified fields at top level; their `metadata`
    # member is original checkpoint metadata and must not override those fields.
    value = {**actual, **item} if "actual" not in item else {**item, **actual}
    status = item.get("status", "verified" if item.get("checkpoint_verified") else "unknown")
    value["status"] = status
    value["name"] = item.get("name", actual.get("name", "未命名"))
    value["group"] = item.get("group", actual.get("group"))
    value["path"] = actual.get("path", item.get("path"))
    if status != "verified":
        for key in ("epochs", "pretraining_seed", "file_hash", "model_config", "pe_protocol"):
            value[key] = None
    return value


def _summary(values, *, signed=False):
    if not values:
        return "缺失"
    average = statistics.mean(values)
    formatted = f"{average:+.6f}" if signed else f"{average:.6f}"
    if len(values) < 2:
        return f"{formatted}（n=1，SD 不可估计）"
    return f"{formatted} ± {statistics.stdev(values):.6f}"


def _select(rows, model, head, condition, task, direction, metric):
    return [row for row in rows if (
        row["model"] == model and row["head"] == head and row["condition"] == condition
        and row["task"] == task and row["direction"] == direction and row["metric"] == metric
    )]


def _eligible(rows, checkpoints):
    verified = {item["name"]: item for item in checkpoints if item["status"] == "verified"}
    accepted, rejected = [], set()
    for row in rows:
        model, group = row["model"], row["comparison_group"]
        reason = None
        if row["engineering_only"]:
            reason = "工程指标不得参与正式比较"
        elif row["n_images"] != 5000:
            reason = "正式 test 必须包含 5,000 张原图"
        elif group == "reference":
            if model not in REFERENCES or row["pretraining_epochs"] is not None or row["pretraining_seed"] is not None:
                reason = "像素参考组的预训练元数据不一致"
        elif model not in GROUPS[group][1:]:
            reason = "模型名称不属于该训练预算比较组"
        else:
            epochs = GROUPS[group][0]
            checkpoint = verified.get(model, {})
            if row["pretraining_epochs"] != epochs or row["pretraining_seed"] != 0:
                reason = "指标未记录匹配的预训练轮数和 seed=0"
            elif checkpoint.get("group") != group or checkpoint.get("epochs") != epochs or checkpoint.get("pretraining_seed") != 0:
                reason = "缺少经验证且预算匹配的 checkpoint 元数据"
        if reason is not None:
            rejected.add(f"{model}：{reason}")
        else:
            accepted.append(row)
    return accepted, sorted(rejected)


def _paired_delta(rows, left, right, head, condition, metric):
    task, direction, name, _ = metric
    first = _select(rows, left, head, condition, task, direction, name)
    second = _select(rows, right, head, condition, task, direction, name)
    left_map = {row["probe_seed"]: row for row in first if row["probe_seed"] is not None}
    right_map = {row["probe_seed"]: row for row in second if row["probe_seed"] is not None}
    seeds = sorted(left_map.keys() & right_map.keys())
    if len(seeds) < 3 or left_map.keys() != right_map.keys():
        return None, f"不足：要求至少 3 个完全匹配的 probe seed；现有 {sorted(left_map)} / {sorted(right_map)}"
    for seed in seeds:
        if left_map[seed]["n_images"] != right_map[seed]["n_images"]:
            return None, "不足：原图数量不一致"
    differences = [left_map[seed]["value"] - right_map[seed]["value"] for seed in seeds]
    return differences, f"probe seeds={seeds}"


def _observed_comparison(rows, left, right, head, condition):
    estimates = [
        _paired_delta(rows, left, right, head, condition, metric)[0]
        for metric in PRIMARY_METRICS
    ]
    if any(value is None for value in estimates):
        return "数据不足，不能完成比较"
    positive = [statistics.mean(value) > 0 for value in estimates]
    if condition == "border_masked":
        clean = [
            _paired_delta(rows, left, right, head, "clean", metric)[0]
            for metric in PRIMARY_METRICS
        ]
        if any(value is None for value in clean):
            return "缺少完整 clean 对照，不能判断优势是否保留"
        if not all(statistics.mean(value) > 0 for value in clean):
            return "clean 的两个主要指标未同时显示正均值差，不能称为保留优势"
        if all(positive):
            return "clean 的正均值差在本次遮挡后仍为正，仅限当前 checkpoint"
        return "clean 的正均值差未在两个主要指标上同时保留"
    if all(positive):
        return "两个主要指标均值均更高，仅为本次观测"
    if not any(positive):
        return "两个主要指标均未显示正均值差"
    return "两个主要指标的差异方向不一致"


def _result_table(rows, models, head):
    selected = [row for row in rows if row["model"] in models and row["head"] == head]
    if not selected:
        return ["尚无符合该组协议的指标。", ""]
    labels = [item[3] for item in SUMMARY_METRICS]
    lines = [
        "| 模型 | 条件 | probe seeds | " + " | ".join(labels) + " |",
        "|---|---|---|" + "---|" * len(labels),
    ]
    for model in models:
        for condition in ("clean", "border_masked"):
            subset = [row for row in selected if row["model"] == model and row["condition"] == condition]
            if not subset:
                continue
            seeds = sorted({row["probe_seed"] for row in subset if row["probe_seed"] is not None})
            entries = [model, condition, str(seeds) if seeds else "不训练 head"]
            for task, direction, metric, _ in SUMMARY_METRICS:
                values = _select(subset, model, head, condition, task, direction, metric)
                if head == "cosine":
                    entries.append(f"{values[0]['value']:.6f}" if values else "缺失")
                else:
                    entries.append(_summary([row["value"] for row in values]))
            lines.append("| " + " | ".join(_cell(item) for item in entries) + " |")
    lines.append("")
    return lines


def _commands(commands):
    if commands is None:
        return ["调用方未提供尚未运行的命令。"]
    if isinstance(commands, Mapping):
        items = commands.items()
    elif isinstance(commands, (list, tuple)):
        items = [(f"命令 {index + 1}", value) for index, value in enumerate(commands)]
    else:
        raise TypeError("commands must be a mapping, list, or None")
    lines = ["以下内容由调用方列为尚未运行的命令；报告生成器仅记录原文，不执行命令。", ""]
    for name, content in items:
        lines.extend([f"{_cell(name)}：", "", _fence(content if isinstance(content, str) else _json(content)), ""])
    return lines


def write_report(output_dir, manifest, rows, *, commands=None):
    """Render an independent report without changing any previous project report."""
    rows = _rows(rows)
    manifest = _plain(manifest)
    checkpoints = [_checkpoint(item) for item in manifest.get("checkpoints", [])]
    counts = manifest.get("counts", {})
    status = manifest.get("status", "unknown")
    engineering = bool(manifest.get("engineering_only")) or any(row["engineering_only"] for row in rows)
    formal = status == "completed" and not engineering and counts.get("test") == 5000
    formal_rows, rejected = _eligible(rows, checkpoints) if formal else ([], [])
    lines = [
        "# SEPA 最小空间表征验证报告", "",
        "## 1. 已执行且验证的结果", "",
        f"记录状态：`{_cell(status)}`。原图数：probe-train={_cell(counts.get('train'))}，"
        f"probe-dev={_cell(counts.get('dev'))}，probe-test={_cell(counts.get('test'))}。", "",
    ]
    if formal:
        lines.extend([
            "以下正式指标仅描述已有 checkpoint 在本次局部读出协议下的结果。A 组是 100 轮预训练，B 组是 25 轮预训练；两组独立展示，不构成同训练预算排名。",
            "数值范围为 0–1。表中为 probe seed 均值 ± 样本标准差（ddof=1）；预训练 seed 固定为 0。probe seed 只衡量 head 拟合波动，不衡量预训练 seed 方差。少于 3 个匹配 probe seed 的比较不形成结论。", "",
        ])
    if manifest.get("error"):
        lines.extend([f"已记录的执行错误：{_cell(manifest['error'])}", ""])
    if manifest.get("executed_command"):
        executed = manifest["executed_command"]
        lines.extend([
            "实际执行的命令（按运行 manifest 原文记录）：", "",
            _fence(executed if isinstance(executed, str) else _json(executed)), "",
        ])
    if not formal:
        lines.extend([
            "**尚无可用于完整协议结论的正式结果。** 当前记录只说明实际执行的检查或小样本工程流程；不能据此给模型排名或估计正式收益。",
            "工程 smoke 的指标均应按 engineering_only 解读；若状态为 blocked，则缺失依赖仍未补齐。完整协议要求 5,000 张正式验证原图和至少 3 个 probe seed。", "",
        ])
    lines.extend([
        f"本次记录 {len(rows)} 条指标；逐方向及逐 probe seed 的原始值见 [metrics.csv](metrics.csv) 和 [metrics.json](metrics.json)。",
        "当前报告不把原图内的 72 个 pairs 当作独立统计样本。未计算置信区间或 p 值。", "",
        "### Checkpoint 与实际处理记录", "",
        "| 名称 | 组 | 状态 | 已验证轮数 | 已验证预训练 seed | 路径 | 文件哈希 |",
        "|---|---|---|---|---|---|---|",
    ])
    if not checkpoints:
        lines.append("| 未提供 checkpoint 清单 | — | 缺失 | — | — | — | — |")
    checkpoint_notes = []
    for item in checkpoints:
        lines.append("| " + " | ".join(_cell(item.get(key)) for key in (
            "name", "group", "status", "epochs", "pretraining_seed", "path", "file_hash",
        )) + " |")
        if item["status"] != "verified" and (item.get("reason") or item.get("error")):
            checkpoint_notes.extend([f"{_cell(item['name'])} 未验证原因：{_cell(item.get('reason', item.get('error')))}", ""])
    lines.append("")
    lines.extend(checkpoint_notes)
    for item in checkpoints:
        if item["status"] == "verified":
            actual = {key: item.get(key) for key in ("model_config", "pe_protocol")}
            lines.extend([f"{_cell(item['name'])} 的实际模型配置与 PE 记录：", "", _fence(_json(actual), "json"), ""])
    lines.extend(["### 已记录的实现检查", "", _fence(_json(manifest.get("checks", {})), "json"), ""])
    if manifest.get("unit_tests"):
        lines.extend(["单元与集成测试的实际执行记录：", "", _fence(_json(manifest["unit_tests"]), "json"), ""])
    configuration = manifest.get("config", {})
    selection = {key: configuration.get(key) for key in ("probe", "probe_seeds", "heads", "smoke")}
    split_record = {
        "split_manifest_hash": manifest.get("split_manifest_hash"),
        "design_hashes": manifest.get("design_hashes"),
        "split_seed": configuration.get("data", {}).get("split_seed"),
        "design_seed": configuration.get("data", {}).get("design_seed"),
    }
    lines.extend(["### 本次 probe 与划分记录", "", _fence(_json({"selection_config": selection, "split_record": split_record}), "json"), ""])
    lines.extend(["### 存储与运行量", "", "下列资源计划为调用方记录的估计或实测量；缺失项保持缺失，不代表预期方法收益。", "", _fence(_json(manifest.get("resource_plan", {})), "json"), ""])
    if rows:
        capacities = sorted({(row["model"], row["head"], row["feature_dim"], row["parameter_count"]) for row in rows})
        lines.extend(["### 实际输入维度和容量", "", "| 模型 | head | 单 tile 维度 | pair 输入维度 | 训练参数量 |", "|---|---|---|---|---|"])
        for model, head, dim, parameters in capacities:
            pair_dim = "不适用（直接 cosine）" if head == "cosine" else 2 * dim
            lines.append(f"| {_cell(model)} | {head} | {dim} | {pair_dim} | {parameters} |")
        lines.append("")
    if formal:
        for group, (epochs, *models) in GROUPS.items():
            lines.extend([f"### {group} 组：{epochs} 轮预训练", ""])
            for head in ("linear", "mlp", "cosine"):
                if head != "linear" and not any(row["head"] == head and row["model"] in models for row in formal_rows):
                    continue
                label = "cosine（无训练、无方向区分能力）" if head == "cosine" else head
                lines.extend([f"{label}：", "", *_result_table(formal_rows, models, head)])
        lines.extend(["### 共享的像素捷径参考", "", "颜色／低频和 border-only 属于 reference 组，在同一数据划分和任务上同时参照 A、B 两组；它们不具有 25 或 100 轮预训练预算。", ""])
        for head in ("linear", "mlp"):
            if any(row["model"] in REFERENCES and row["head"] == head for row in formal_rows):
                lines.extend([f"{head}：", "", *_result_table(formal_rows, REFERENCES, head)])
        lines.extend(["### 配对差值与边界遮挡诊断", "", "仅对同一组、同一 head、同一测试条件和完全匹配的 probe seeds 计算差值；下表未跨 A/B 配对。差值为左侧减右侧，正值仅表示这些固定 checkpoint 的观测均值更高。", "", "| 组 / head | 对比 | 条件 | 指标 | 平均差 ± probe seed 样本 SD | 配对记录 |", "|---|---|---|---|---|---|"])
        for group, (_, ijepa, k0, k3) in GROUPS.items():
            for head in ("linear", "mlp"):
                if not any(row["head"] == head and row["model"] == k3 for row in formal_rows):
                    continue
                for right in (k0, ijepa, *REFERENCES):
                    for condition in ("clean", "border_masked"):
                        for metric in PRIMARY_METRICS:
                            values, detail = _paired_delta(formal_rows, k3, right, head, condition, metric)
                            estimate = _summary(values, signed=True) if values is not None else "无法比较"
                            cells = (f"{group} / {head}", f"{k3} − {right}", condition, metric[3], estimate, detail)
                            lines.append("| " + " | ".join(_cell(item) for item in cells) + " |")
        lines.extend(["", "遮挡前后的绝对表现已列于各组表。下表为同一冻结 head 的 border_masked − clean；负值是本协议下的下降，不能直接证明只使用边界捷径。", "", "| 模型 / head | 指标 | 遮挡变化均值 ± 样本 SD |", "|---|---|---|"])
        for model, head in sorted({(row["model"], row["head"]) for row in formal_rows}):
            for metric in PRIMARY_METRICS:
                task, direction, name, label = metric
                clean = {row["probe_seed"]: row for row in _select(formal_rows, model, head, "clean", task, direction, name)}
                masked = {row["probe_seed"]: row for row in _select(formal_rows, model, head, "border_masked", task, direction, name)}
                if clean and clean.keys() == masked.keys() and (head == "cosine" or (None not in clean and len(clean) >= 3)):
                    delta = [masked[seed]["value"] - clean[seed]["value"] for seed in clean]
                    estimate = f"{delta[0]:+.6f}（确定性 cosine）" if head == "cosine" else _summary(delta, signed=True)
                    lines.append(f"| {_cell(model)} / {head} | {label} | {estimate} |")
                else:
                    lines.append(f"| {_cell(model)} / {head} | {label} | 缺失完全匹配的 clean / border_masked 记录 |")
        lines.append("")
        if rejected:
            lines.extend(["以下记录未参与正式比较：", "", *[f"- {_cell(reason)}" for reason in rejected], ""])
    lines.extend(["## 2. 尚未运行的命令", "", *_commands(commands), "", "## 3. 协议限制", ""])
    limits = [
        "固定几何为中心正方形裁剪、bicubic resize 至 240×240、3×3 个原生 80×80 tile；每 tile 为 25 个 16×16 patch，不旋转、不补边。主 probe 使用全部 9 个 tile；这不是完整 missing + shuffle 预训练任务的恢复评估。",
        "encoder 必须 eval、requires_grad=False，并在 no_grad/inference_mode 下逐 tile 独立编码。25 个输出 patch token 均值为 384 维；不读取其它 tile，也不使用 predictor、relation head 或 teacher。实际冻结和参数一致性以已记录的检查为准。",
        "不提供原图坐标、canonical tile ID、绝对位置、置换、移动标记或有位置含义的候选顺序。I-JEPA 使用对每个 tile 相同的顶部左侧 5×5 PE 窗口；V1/V2 保留各自训练时共享的 tile-internal PE，实际记录见上方清单。对全图训练 I-JEPA，这引入输入分布变化，结论仅限该内容型局部读出下的空间可读出性。",
        "图像先划分再生成 pairs。相同内容的重复原图不能跨 train/dev/test；具体去重方法、近重复限制、split seed 与哈希以本次 split manifest 和检查记录为准，类别标签不作为 probe 输入。",
        "标准化参数、BCE 正负权重只从 probe-train 估计；优化选择、early stopping 和 balanced accuracy 阈值只使用 probe-dev，test 不参与选择。dev balanced accuracy 阈值并列时选择最大的已观察分数阈值，score≥threshold 判为正。每张图保留全部 72 个有序非自身 pair，V/H 各 6 个正例；随机排序 AP 参考比例为 6/72 = 1/12 ≈ 0.083333。普通 accuracy 不是主指标。",
        "邻居检索沿用同一个 V/H head，每图 24 个有效方向 query、每个 8 个候选；Recall@1 随机参考值为 1/8 = 0.125。邻接与检索是共享 head 的互补读出，不是两份独立机制证据。候选顺序随机化；分数并列按并列组内均匀随机排序的解析期望计分，Recall@1 可为分数值，MRR 对并列位置的倒数名次取均值，均不依赖正确位置或候选顺序。",
        "Linear([h_i,h_j]) 是两个单 tile 线性项的相加，无法学习任意 pair 交互；它的可读出能力本身受限。若提供 MLP，结构固定为 Linear(2D,128)→GELU→Linear(128,2)，作为独立结果，对所有 encoder 使用相同结构和调参预算。",
        "Feature cosine 不训练关系 head，本身没有方向区分能力；其结果单列，不混入训练 probe 的 seed 方差或相同 head 的配对差值。颜色／低频及 border-only 使用相同深度与隐藏宽度的轻量 head，维度和实际参数量已单列。",
        "Border-only 只保留外围 8 px，内部固定均值填充，再作固定池化。它是捷径参考，SEPA 超过它不能单独证明没有使用边界。",
        "边界遮挡将 test tile 外围 8 px 填充为预处理固定均值，保持 encoder、probe 和 canonical 标签不变；所有模型接受相同遮挡。它既带来分布偏移，也移除真实视觉内容，掉分不能直接解释为模型只学到了捷径。",
        "至少 3 个 probe seed 只描述 head 拟合波动；预训练只有 seed=0，不代表跨预训练 seed 的统计稳定优势。本报告不使用 pair 级独立样本假设，也不报告方法显著性或看完 test 才选择的通过阈值。",
    ]
    limits.extend(str(item) for item in manifest.get("limitations", []))
    lines.extend([f"- {item}" for item in limits])
    lines.extend(["", "## 4. 可支持的结论", ""])
    if not formal:
        lines.extend([
            "1. k3 是否超过匹配的 k0？尚未在完整协议上验证，不能下结论。",
            "2. k3 是否超过同训练轮数的 I-JEPA？尚未在完整协议上验证，A/B 缺失项不能由另一组替代。",
            "3. 是否超过简单捷径基线？工程 smoke 只检查流程，不能据其小样本指标作优势判断。",
            "4. 边界遮挡后优势是否仍存在？尚未在完整协议上验证。", "",
            "目前可支持的结论仅限上述实际记录的实现检查；没有填写预计收益，也没有启动补充预训练。",
        ])
    else:
        lines.append("以下逐项陈述已有数据的数值差；正值不等于跨预训练 seed 的稳定优势，缺失项明确保留。")
        lines.append("")
        for group, (epochs, ijepa, k0, k3) in GROUPS.items():
            heads = [head for head in ("linear", "mlp") if any(row["model"] == k3 and row["head"] == head for row in formal_rows)]
            if not heads:
                lines.extend([f"{group} 组（{epochs} 轮）：k3 数据或已验证 checkpoint 不齐，四个研究问题均无法完成本组比较。", ""])
                continue
            for head in heads:
                lines.extend([f"{group} 组（{epochs} 轮），{head}：", ""])
                for number, question, opponents, condition in (
                    (1, "k3 是否超过 k0", (k0,), "clean"),
                    (2, "k3 是否超过同轮数 I-JEPA", (ijepa,), "clean"),
                    (3, "是否超过简单捷径基线", REFERENCES, "clean"),
                    (4, "边界遮挡后优势是否仍存在", (k0, ijepa, *REFERENCES), "border_masked"),
                ):
                    statements = []
                    for opponent in opponents:
                        values = []
                        for metric in PRIMARY_METRICS:
                            delta, reason = _paired_delta(formal_rows, k3, opponent, head, condition, metric)
                            values.append(f"{metric[3]} Δ={_summary(delta, signed=True)}" if delta is not None else f"{metric[3]} {reason}")
                        observation = _observed_comparison(formal_rows, k3, opponent, head, condition)
                        statements.append(f"相对 {opponent}：" + "，".join(values) + f"（{observation}）")
                    lines.append(f"{number}. {question}？" + "；".join(statements) + "。")
                lines.append("")
        lines.append("可支持的范围是这些已验证 checkpoint 在相应预算和局部读出协议下的观测差异；不外推为全部空间能力排名，也不据此确认下一轮训练或 future work。")
    lines.append("")
    return _write_new(Path(output_dir) / "report.md", "\n".join(lines))
