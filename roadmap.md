# SEPA Roadmap — 新轮执行副本

> 更新：2026-09-11。以学长提供的 SEPA（Plan B）为新方案依据；不以本机旧工程的结构约束新实现。
> 用户已确认：修正位置输入约定和辅助损失梯度约定；重新进行实验，旧实验分数、checkpoint、完成状态和 NO-GO 结论不作为本轮证据。
> 本次不沿用旧训练入口。候选代码须先审查，适配代价大或逻辑耦合的模块可重写。
> 工程采用 native80（240→9×80，无补边）与 ViT-S/16，并保留 pad74 对照；缺失的 10.7 协议仍未补齐。
> 本文件保留方法设计约定；实际采用的配置、已完成实验与结论见 [实验报告](reports/SEPA实验报告_2026-09-16.md)。原微信附件保持原样。


## 0. 锁定的项目目标

本项目的目标是开发 **Spatial-Embedding Predictive Architecture（SEPA）**，
用于通用视觉表征学习。

SEPA 应通过从以下两种条件同时成立的 patch 中预测 canonical 语义 embedding，
来学习可迁移的表征：

- 部分缺失；
- 空间上被置换。

最终目标不是为解拼图而解拼图。最终目标是学到比标准 masked JEPA 式预训练
更好地编码局部视觉结构与 patch 级空间组织的表征。

架构有效性应首先在通用自然图像数据集（ImageNet-100、COCO 等）上验证。
医学影像（组织病理等）不作为主评估领域，不追求在其上的性能指标；仅在通用
数据上的证据成立之后，才可作为可选的 domain-transfer 扩展。

该目标被锁定为项目级目标。对最终目标的任何改动，都必须在围绕新目标更新
文档、实验或实现计划之前，获得用户的显式手动确认。

> 变更记录：v1 目标为"组织病理学表征学习"。经用户确认，改为通用视觉表征
> 学习，评估领域由 CRC/NCT 改为 ImageNet-100 / COCO 等通用数据集。

## 1. 核心假设

标准 masked JEPA 在预测缺失目标 embedding 时，可以利用一个基本完整的
canonical 上下文。这会使模型对跨 tile 空间对应关系的学习压力有用但不充分。

SEPA 改变了预测问题：

```text
input:  可见 tile（observed 位置受控腐蚀）+ 缺失 slot
target: canonical teacher embedding
loss:   在 canonical 目标位置上的 latent 预测
```

假设如下：

> 通过破坏直接的空间对齐而保持目标 canonical，SEPA 使模型更依赖局部视觉
> 内容、tile 身份、相对排布与空间对应关系。若这种压力是有意义的而非仅仅
> 更难，它应当在迁移任务和空间探针上优于标准 masked JEPA 以及匹配的
> 非空间对照。

## 2. 方法边界

SEPA 的设计应遵守以下边界：

- student 上下文可以包含 observed 位置被置换的可见 tile 与缺失 slot。
- teacher 目标必须保持干净且 canonical。
- 模型在 canonical 位置上预测目标 embedding。
- 预训练 loss 必须是 JEPA 式 latent 预测，而非有监督的置换分类。
- 下游使用的 encoder 在推理时不得依赖 oracle layout 标签、trajectory
  或任何解拼图元数据。
- 任何辅助空间信号都必须以"提升表征质量"为正当性依据，而非仅仅降低
  预训练 loss。

结论必须与证据匹配：

- 若只有分类提升，宣称语义表征更好。
- 若空间探针提升，宣称空间表征更好。
- 若稠密分割或检测提升，才可作更强的像素级或稠密语义结论。

## 3. 已冻结的第一版形式化（v1）

以下沿用学长提供的 v1 设计；其中位置输入与关系梯度条款已按本次用户确认修改。
学长引用的完整协议文档（3.1–3.5、4、6、10.7）尚未提供，未决配置不得以旧工程默认值代替。

| 项目 | 冻结值 |
|---|---|
| grid | 3×3，canonical 位置 $i\in\{0,\dots,8\}$ |
| 缺失 slot 数 | $\lvert S\rvert=2$，固定，仅从 8 个非中心 slot 均匀采样 |
| masking 依据 | 按 canonical 位置 mask；masked slot 保持在 canonical 位置，query 为 $q_s=m+\mathrm{PE}(s)$ |
| 置换范围 | 从 6 个可见非中心 tile 中采样大小为 $k$ 的子集 $D$，在 $D$ 上做 uniform random permutation $\sigma$（允许 fixed point）；$D$ 外不动 |
| 腐蚀强度 | $k$ 为 run 级超参，$k\in\{0,2,3,4,5,6\}$；$k=0$ 退化为 JEPA-tile，$k=6$ 退化为 Plan A-anchored；$k=3$ 为中间臂 |
| anchor | 中心 slot（$i=4$）永远可见、永远 canonical、不参与置换 |
| 跨 tile 位置注入点 | 仅在 predictor 输入：$t_i=h_i+\mathrm{PE}_{\mathrm{slot}}(o_i)$，$o_i$ 是可见内容当前所在的 observed slot；encoder 不接收整图 canonical/observed slot 坐标 |
| tile 内位置 | encoder 允许使用同一套 tile-local patch PE，表示 patch 在自身 tile 内的位置；该 PE 不含该 tile 在整图中的身份 |
| student 信息边界 | predictor 允许接收 observed slot 与待预测的 canonical query slot；禁止输入可见 tile 的真实 canonical 身份、完整 canonical→observed 对照表及其逆映射、移位标记或 $k$ 标签；encoder 只接收可见 tile 像素 |
| predictor | image-based JEPA predictor 标准形式，可见 token 与 query token 联合 self-attention |
| loss 位置 | 仅在 masked canonical slot $s\in S$ 上计算 $\mathcal{L}_{\mathrm{sem}}$；被移位 tile 的 canonical slot 不计 loss（v1） |
| relation head | content-only 辅助损失，输入 encoder 的 $(h_i,h_j)$；不得接收整图 slot PE、slot 身份或 predictor 中间态；关系输出不送入语义 predictor/aggregation，$\mathcal{L}_{\mathrm{rel}}$ 的梯度回传 student encoder 与关系头 |
| tile gap / jitter | Phase 1 不使用；以 border-only 基线监控边界捷径 |
| 输入分辨率 | 科学协议待冻结；工程默认240→9块80×80、无补边。保留224→中心裁剪222→9块74×74→逐块补至80的对照；实际采用的设置见本轮实验报告。不得直接让 /16 卷积截断74px边缘 |
| 下游评估 | 严格 encoder-only |

接口说明：数据变换内部可用 $o_i=\pi(i)$ 描述置换，但输入 predictor 的是内容及其当前位置，不能同时提供该内容的真实 canonical 身份。masked canonical slot 用于构造 $q_s=m+\mathrm{PE}_{\mathrm{slot}}(s)$ 与选取 teacher target，允许进入 query 侧。

关系标签可以由 canonical 元数据在 loss 侧生成；元数据不作为 relation head 的输入。禁止在辅助训练中使用 `head(h.detach())`；teacher target 始终 stop-gradient。允许关系损失更新 encoder 的 tile 内 patch PE，禁止依赖 predictor 的整图 slot PE。

observed 位置与 canonical 位置的互信息由 $k$ 解析给出（$n'=6$）：

| $k$ | 移位概率 | MI (bits) | 备注 |
|---|---|---|---|
| 0 | 0 | 2.58 | JEPA-tile |
| 2 | 0.17 | 1.55 | |
| 3 | 0.33 | 0.89 | 中间臂 |
| 4 | 0.50 | 0.42 | |
| 5 | 0.67 | 0.12 | |
| 6 | 0.83 | 0.00 | Plan A-anchored |

**v1 明确不做：** 置换距离限制、curriculum、每样本随机 $k$、tile gap。
这些留作第 4 节 Phase F 之后或第 5 节 Stage 3 的后续轴。

对以上任一项的修改需重新走手动确认流程。

## 4. 最小实验阶梯

### 4.0 数据集

| 用途 | 数据集 | 说明 |
|---|---|---|
| 预训练（主） | ImageNet-100 | 约 126k 训练图，100 类；无标签使用 |
| 预训练（扩展） | ImageNet-1k | 仅在 ImageNet-100 证据成立后 |
| 表征评估（主） | ImageNet-100 val | linear probe、kNN、low-shot |
| 跨数据集迁移 | CIFAR-100、Flowers-102、Pets、DTD 等 | frozen encoder linear probe |
| 稠密任务 | COCO（检测 / 实例分割）、ADE20K（语义分割） | Phase F |
| 鲁棒性 | ImageNet-C、ImageNet-R、ImageNet-Sketch | Phase F |
| 空间探针 | ImageNet-100 val 上按 3×3 切 tile 构造 | Phase E |
| 可选 domain transfer | CRC / NCT 等病理数据 | 非主线，仅在 Phase F 之后 |

所有基线与对照使用完全相同的预训练数据集与数据顺序。

### Phase A：定义 SEPA 数据变换

交付一个确定性、可测试的变换，把 canonical patch 变为 SEPA 训练样本：

- canonical teacher layout；
- 含缺失 slot 的 student layout（$\lvert S\rvert=2$，非中心）；
- 含受控置换的 student layout（子集 $D$、置换 $\sigma$、observed 位置 $\pi$，中心固定）；
- 从 canonical 位置到 teacher embedding 的目标索引映射；
- 泄露检查：隐藏 tile 的像素和 teacher latent 仅位于 target 分支；$k$ 标签、移位指示、可见 tile 真实 canonical 身份及完整真实置换对照表不进入 student 的预测路径。predictor 可以接收 observed slot 和 masked query slot；
- **位置先验捷径检查（自然图像下尤其重要）**：自然图像存在强位置–内容
  先验（天空在上、地面在下、主体居中）。记录 per-tile 颜色均值与低频统计
  对 canonical 位置的线性可预测性；若显著高于 chance，需引入保内容的轻度
  color jitter 和/或随机水平翻转与 random resized crop，并把该可预测性
  作为 Phase E 空间探针的第二个捷径下界。

Phase A 成功的条件是：变换无歧义、可复现、无目标泄露。

### Phase B：实现 SEPA 目标函数

在与标准 masked JEPA 匹配的架构与预算下实现 SEPA 预训练目标。

第一版实现保持最小：

- 与基线相同的 encoder 族（ViT-S/16 或 ViT-B/16 级别，按 10.7 冻结）；
- 相同的优化器、EMA、增广、数据顺序与训练长度；
- 相同的 predictor 容量，除非有显式理由；
- canonical teacher embedding 作为 stop-gradient 目标；
- 在 masked canonical slot 上的归一化 latent 预测 loss；
- 总 loss $\mathcal{L}=\mathcal{L}_{\mathrm{sem}}+\lambda\,\mathcal{L}_{\mathrm{rel}}$，
  $\lambda$ 按 10.7 冻结值；Stage 1 中 $\lambda=0$。

### Phase C：健全性检查

在解读下游性能之前，验证：

- 无 NaN、无 collapse；
- teacher 无梯度；
- 目标与预测方差健康；
- mask 数量与 loss 位置与设计一致（$\lvert S\rvert$ 与 loss 项数匹配）；
- 置换元数据不泄露目标像素或类别标签；
- 运行时间与参数量与基线可比；
- **敏感性断言：** 固定输入，比较 $\pi=\mathrm{id}$ 与 $\pi\neq\mathrm{id}$
  下 predictor 输出差 $\Delta$，$k>0$ 训练的模型 $\Delta$ 须显著大于数值噪声；
- **position-trust 曲线：** 按训练步与 $k$ 记录 $\Delta$，预期随 $k$ 单调递减，
  $k=6$ 时随训练下降；
- **relation head 位置隔离：** 单独对 $\mathcal{L}_{\mathrm{rel}}$ 检查梯度：依赖 student encoder 和关系头，不依赖 predictor 的整图 slot PE、predictor 参数或 teacher；不将 encoder 内部 patch PE 误判为禁止的位置输入。

### Phase D：表征评估

评估 SEPA 是否提升表征质量：

- ImageNet-100 frozen linear probe 作为初始基准；
- top-1 / top-5 accuracy、kNN（k=20）accuracy；
- low-shot 探针：1%、10% 标签比例；
- 跨数据集 frozen linear probe（CIFAR-100、Flowers、Pets、DTD 中至少两个）；
- encoder-only 推理，无任何特权拼图元数据。

### Phase E：空间证据

加入显式空间探针，证明提升来自空间而非仅是泛化正则化。探针在
ImageNet-100 val 上按 3×3 切 tile 构造，frozen encoder + 线性/浅层头：

- 相对位置预测（给定两 tile embedding，预测 8 方向关系）；
- 邻接预测（两 tile 是否相邻）；
- tile 匹配（跨增广视图的同一 tile 识别）；
- layout 恢复探针（9 tile 全排列恢复）；
- 局部连续性或邻居一致性诊断。

所有空间探针须与两个捷径下界对比：border-only 基线（边界像素捷径）与
位置先验基线（Phase A 记录的颜色/低频可预测性）。若 SEPA 不显著高于
两者，Phase 2 引入 tile gap 与更强增广后重跑。

除非 SEPA 在至少部分空间证据上同时超过标准 masked JEPA 与匹配的非空间
对照，否则不得将其表述为空间表征贡献。

### Phase F：更广泛的下游任务

在获得首批分类与空间探针证据后，扩展到更一般的视觉任务：

- ImageNet-1k 预训练与 linear probe / fine-tune；
- COCO 目标检测与实例分割（ViTDet 或 Mask R-CNN 头，frozen 与 fine-tune 两种设置）；
- ADE20K 语义分割（UperNet 或线性头）；
- ImageNet-C / -R / -Sketch 鲁棒性；
- 可选：病理数据（CRC / NCT）domain transfer，仅作为"通用表征是否迁移到
  远域"的附加证据，不作为主指标。

作强像素级语义结论前必须有稠密任务（COCO / ADE20K）结果。

## 5. 必需的对照与实验分期

SEPA 必须与能区分"真实空间学习"与"泛化难度"的对照比较。

### 5.1 基线

| 名称 | 定义 | 隔离的效应 |
|---|---|---|
| JEPA-full | 整图 encoder 带 PE 的 I-JEPA 式基线；当前实现为匹配tile mask的整图对照，不是官方多块mask recipe复现 | 检验整图上下文编码；正式领域标准比较仍需明确基线recipe |
| JEPA-tile | 本协议 $k=0$ | tile-local encoder 本身 |
| Plan A-anchored | 本协议 $k=6$ | 位置先验为零 |
| border-only detector | 仅取每 tile 外围 8 px 薄带的浅层模型 | 空间探针的捷径下界 |
| position-prior detector | 仅取每 tile 颜色均值与低频统计的线性模型 | 自然图像位置先验捷径下界 |

### 5.2 Stage 1 — $k$ 效应存在性检验

- 臂：$k\in\{0,3,6\}$，$\lambda=0$；外加 JEPA-full。
- 预训练数据：ImageNet-100。
- 10 seeds，clean frozen protocol（10.7）。
- 预注册预测：ImageNet-100 linear probe top-1 呈倒 U，$k=3$ 高于 $k=0$
  与 $k=6$ 各至少 1.0 个百分点（阈值需在 10.7 中按 seed 方差校准）。

### 5.3 Stage 2 — relation loss 在最优 $k^*$ 上的作用

- 固定 Stage 1 最优 $k^*$。
- 臂：no relation / undirected aux / directed aux /
  relation-guided aggregation（负对照）。
- 附加：在 $k=6$ 上从头运行 undirected aux，检验其在本轮数据集与协议下的作用，不将旧增益视为既定前提。

### 5.4 Stage 3 — 可选，当前不执行

- 对被移位 tile 的 canonical slot 加 query 与 loss（隐式 assignment）；
  若启用，Phase E 主探针须更换为与之独立的任务；
- 置换距离轴（仅相邻交换）；
- 每样本随机 $k$；
- 分辨率轴（tile 尺寸 64 / 96 / 128）。

关键检验：

> SEPA 改善 encoder 表征，是因为它学到了空间上有意义的视觉结构，还是仅
> 因为 pretext 任务更难？

若增益在匹配的非空间对照面前消失，贡献应重新表述为正则化而非空间理解。

## 6. 判定准则

### 6.1 Stage 1 终止条件

若

$$
\mathrm{top1}(k{=}3)\le\max\bigl(\mathrm{top1}(k{=}0),\,\mathrm{top1}(k{=}6)\bigr)+\delta,
$$

其中 $\delta$ 为 10.7 中按 seed 方差校准的阈值（初值 1.0 个百分点），
则位置腐蚀不具独立价值，Plan B 终止，回归 Plan A 路线。

### 6.2 成为主方法的条件

SEPA 仅在同时满足以下全部条件时才成为强主方法：

- 在匹配预算下优于标准 masked JEPA；
- 优于匹配的非空间难度对照；
- 多 seed 下增益稳定；
- 至少一个空间探针提升，且显著高于 border-only 与 position-prior 两个下界；
- 不依赖推理时的特权元数据；
- 支持超出 ImageNet-100 linear probe 的更广泛下游任务（至少一个跨数据集
  迁移 + 一个稠密任务），或有清晰的计划。

若 SEPA 降低预训练 loss 但不提升下游表征质量，不视为成功。

若 SEPA 提升分类但在所有空间探针上失败，方法可能仍有用，但空间理解的
结论必须弱化。

## 7. 新轮实验与路线边界

本轮以学长提供的 SEPA（Plan B）为方法定义，重新训练和评估全部正式对照。不使用旧实验分数、旧 checkpoint、旧 feature cache 或旧完成标记支撑本轮结论；所有待跑实验均从未执行状态开始。

- **Full-shuffled Jigsaw-JEPA**：仅作路线背景；当前 $k=6$ 臂依照本文件的中心锚点、两格 mask 和 latent 监督定义重新实现，不能用旧分数代替。
- **学长所述 Hybrid Masked-Jigsaw JEPA 路线**：按其原文理解为整图 encoder 与独立 puzzle-solver 分支；这是学长定义的参照路线，不是本机旧实现的状态声明。新 SEPA 使用 tile-local encoder、predictor observed PE 与隐式 latent 预测，不因本机旧实现而改回其他结构。
- **Graph-Calibrated Hybrid JEPA**：不在当前主线中恢复。
- **Oracle Trajectory-Conditioned JEPA**：不作为当前训练目标；不引入 oracle trajectory 输入。
- **relation-guided aggregation**：按学长分期保留为 Stage 2 对照，退出主线；是否有效由本轮结果判断，不沿用历史 NO-GO。
- **代码复用**：仅迁移审查通过且符合新接口的独立算法或模块；新工程不直接 import 旧研究工程及其默认配置。新数据变换、训练入口和评估协议重新组织。

## 7.1 本轮统一的信息流

```text
同一幅图的共同几何视图 → 9 块 canonical tiles
  student: 选7块可见tile → 共享tile-local encoder → h_i
           h_i + observed-slot PE，与2个masked query共同进入predictor → L_sem
           (h_i,h_j) → content-only relation head → L_rel（Stage 2）
  teacher: 干净canonical tiles → EMA teacher → 按masked canonical slot选target

L_sem 与 L_rel 更新student encoder；teacher只做EMA更新。
关系预测结果不进入语义predictor。真实置换对应关系仅留在数据与loss侧。
```

## 8. 实施启动与待补事项

第 3 节位置与梯度接口已按用户确认修正，并已实现于新的 `src/sepa_plan_b/` 工程。Phase A 数据、Phase B 预训练、Phase C 工程断言与日志、Phase D 冻结分类/迁移入口、Phase E 邻接/方向/位置探针已落地；Stage 2 关系模块可独立配置。实现和短程跑通不等于完成科学实验。

进入正式实验前仍需补齐：

- 10.7 冻结值：输入分辨率与 tile 尺寸、encoder 规模、$\lambda$、
  clean frozen protocol 细节、Stage 1 终止阈值 $\delta$；
- ImageNet-100 类别列表（沿用 CMC 的 100 类划分或自定义）与数据顺序 seed；
- Phase A 泄露检查与位置先验捷径检查的通过阈值；
- 本轮新的代码版本、数据清单、配置哈希与结果目录；正式产物必须能证明属于新轮实验。

代码已自动记录源文件哈希、配置、数据清单及图像内容身份。跨增广tile匹配、全排列layout恢复、非空间难度对照以及Phase F/Stage 3仍为后续工作，不能用现有探针替代它们的证据。

进入 Stage 2 前需完成：

- Stage 1 全部 10 seeds 结果与 6.1 判定；
- Phase C 的敏感性断言与 position-trust 曲线读数。

后续对项目目标或已明确冻结的科学设置的修改按用户决定执行。本次已经确认的位置接口、辅助梯度接口与重新实验安排无需再次确认。
