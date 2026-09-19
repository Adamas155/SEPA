# I-JEPA 局部 student / 全局 teacher 预训练对照（seed 0）

2026-09-14 13:25:34（UTC+8）在 `connect.westb.seetacloud.com:47026` 启动，RTX 4090 D 24 GB，PID 2129。预训练于 22:12 完成，四组评估于 22:56 完成，任务状态为 `complete`。此次未设置自动关机。

## 要验证的问题

上一轮对已完成的 I-JEPA 权重改变推理视野，student Top-1 从整图的 52.92% 变为局部 80×80 的 51.54%，teacher 从 52.86% 变为 51.50%。这说明仅改变这些权重的推理视野，下降约 1.4 个百分点；不能据此排除局部视野在预训练阶段的影响。

本轮从头初始化，用与原 I-JEPA 对照完全相同的 seed、图像、增强、上下文/目标掩码、参数初始化、优化器和训练步数，只改变 student encoder 每一层的注意力连接。

| 部分 | 原 I-JEPA 对照 | 本轮 |
|---|---|---|
| 输入图像 | 240×240；16×16 patch | 相同 |
| Student 可见 patch | 原 multiblock 掩码保留的上下文 patch | 完全相同；不补回隐藏 patch |
| Student 注意力 | 所有可见 patch 间可交流 | 仅同一原生 80×80 图块内可交流 |
| 位置编码 | 原 15×15 网格固定位置编码 | 相同；保留全图位置 |
| Teacher | 全图 225 patch，EMA 更新 | 相同结构和更新规则；EMA 跟随本轮 student |
| Predictor | 全局 patch 级预测，6 层、384 维 | 相同 |
| 目标与损失 | Teacher 输出 LayerNorm、Smooth L1 | 相同 |
| 预训练 | ViT-S/16；seed 0；batch 64；100 epoch；BF16 | 相同 |

代码一次接收 240×240 图像，以块对角注意力掩码限制连接。其局部输出与逐个输入实际 80×80 图块、保留相应位置编码并只提取可见 patch 的参考实现一致。每块最多有 25 个 patch；训练时隐藏目标后，每块可见数量可能更少或为零。没有补边或把每块放大到 240×240。

Teacher 仍可获得全图上下文，predictor 仍能融合不同图块的可见信息。本轮不能代表 teacher 和 student 都局部的完整 SEPA，也不能单独检验 SEPA 的损失或 predictor 设计。

## 最终结果

| 预训练 | 分支与评估视野 | Top-1 | Top-5 | kNN | 有效秩 |
|---|---|---:|---:|---:|---:|
| 原全局 I-JEPA | student，global240 | 52.92% | 80.84% | 42.42% | 90.48 |
| 原全局 I-JEPA | student，local80 | 51.54% | 79.84% | 38.88% | 72.22 |
| 本轮局部 student | student，global240 | 46.90% | 76.76% | 31.52% | 26.01 |
| 本轮局部 student | student，local80 | 46.70% | 76.22% | 31.18% | 25.00 |
| 本轮局部 student | teacher，global240 | 46.92% | 76.74% | 31.32% | 26.02 |
| 本轮局部 student | teacher，local80 | 46.44% | 76.36% | 30.96% | 25.00 |
| 原 SEPA k3 | student，native80 | 30.34% | 59.48% | 20.16% | — |

在相同 global240 评估下，把 student 的预训练注意力限制在 80×80 图块内，使 Top-1 从 52.92% 降至 46.90%，下降 6.02 个百分点。在相同 local80 评估下，从 51.54% 降至 46.70%，下降 4.84 个百分点。因此，局部视野在预训练阶段确实有明显代价，不是只有推理时改变输入分布造成的约 1.4 个百分点。

本轮内部把 local80 改回 global240 推理，只提高 student 0.20、teacher 0.48 个百分点。训练完成后再恢复全局注意力不能找回全局预训练的能力。Teacher 虽然训练时看全图，但它是局部 student 参数的 EMA，并不独立反向学习，所以也停留在约 46.9%。

局部 student 的有效秩约 25，明显低于原全局 I-JEPA 的 90.48，也低于原权重仅在推理时局部化的 72.22；同时 Top-1 与 kNN 都下降。这支持“局部预训练让 encoder 表征明显压缩”的判断，但有效秩本身是诊断量，不能单独证明具体因果机制。

这个因素仍不能解释 I-JEPA 与 SEPA 的全部差距。原全局 I-JEPA 与 SEPA k3 相差 22.58 个百分点；使用 I-JEPA 目标但限制 student 视野后，local80 Top-1 仍有 46.70%，比 SEPA k3 高 16.36 个百分点。按这次单变量对照，student 局部视野解释了约 4.8–6.2 个百分点，剩余差距需要继续检查 teacher 视野、patch 级目标、predictor、归一化/损失和优化配置。单个 seed 适合判断这次幅度很大的趋势，但还不能给置信区间。

## 固定预算与最终评估

- 126,684 张训练图、5,000 张验证图、100 类；与上一组的数据指纹一致。
- 每 epoch 1,980 步，最后一批 28 张；共 198,000 步。
- LR 0→1e-4→1e-6，warmup 2.5 epoch；原优化器的 weight decay 分组和 0.04→0.4 调度；EMA 0.996→1。
- 每 epoch 保存可续训 checkpoint，额外保存第 10、25、50、100 epoch。
- 100 epoch 后自动完成 student/teacher × local80/global240 四种冻结特征评估，各自重新拟合相同的线性分类器（seed 0、SGD LR 0.005、200 epoch），并计算 Top-1、Top-5、kNN 和特征健康指标。
- 评估统一使用中心正方形裁切、240px bicubic、全 225 个输出按原图顺序平均；特征和分类器均为 FP32。验证集不用于挑选 checkpoint 或调参。

新显卡与上一组的 RTX 5090 系列不同；数据与初始化保持一致，不承诺跨显卡浮点轨迹逐位相同。预训练实际用时 31,584 秒，即 8 小时 46 分；包含四组最终评估共约 9 小时 31 分。第 198,000 步 loss 为 0.1323，末段均值为 0.1396，LR 与 weight decay 正确到达 1e-6 和 0.4。

## 已完成的验证

15 项检查通过，包括初始模型与优化器一致、关闭限制时与原全局 forward 逐位一致、局部 forward 及所有可训练参数梯度与独立原生 80px 参考实现一致、空图块/单 token/多 context 掩码顺序、隐藏像素隔离、跨块隔离、teacher 全局负对照、快速分块评估与训练 forward 一致、worker 无关的数据恢复、BF16 有限梯度、真实最后 28 张小批量和调度终点。

实际保存并恢复 checkpoint 后，下一步的指标及 student、predictor、teacher 全部参数逐位一致。

- 原生逐块 forward 最大误差：1.43e-6。
- 可训练参数梯度最大误差：1.16e-9。
- 快速分块评估适配器最大误差：1.91e-6。

## 文件与复现

- 训练入口：`server/train_ijepa_local_student.py`。
- 检查：`server/test_ijepa_local_student.py`。
- 本地回执：`server/evidence/ijepa-local-student-20260914/`。
- Run ID：`3e44ae15a866f0dc00fda530bd8782c21925577a619e496dfaf784d7cd36793c`。
- 最终 checkpoint SHA-256：`d7b746eba0ffae724358abbc0444e6d7339c4e14d8fc17b624bddeb3bf9f1c18`。
- 服务器项目：`/root/autodl-tmp/sepa-plan-b-a659af079672/sepa-plan-b`。
- 运行目录：`runs-ijepa/local-student-global-teacher-s0-3e44ae15a866`。
- 状态：`server-logs/ijepa_local_student_status.json`；日志：`server-logs/ijepa_local_student.log`。

运行命令为 `.venv/bin/python -u server/train_ijepa_local_student.py`。若任务意外中断，在确认旧进程已结束后加 `--resume`，恢复上个完整 epoch 的模型、优化器、调度器、CUDA RNG 与数据进度。运行入口使用互斥锁阻止重复执行，且没有关机选项或关机调用。已完成的评估结果会先校验协议和文件哈希再复用。
