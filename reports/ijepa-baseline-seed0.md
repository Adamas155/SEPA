# I-JEPA 同规模对照：2026-09-13

用户指定服务器 `ssh -p 39999 root@connect.westb.seetacloud.com`。2026-09-13 18:48:16（台北时间）提交正式训练，PID **2352**。随后确认进程存活，训练日志连续记录 step 100、125、150；不是仅提交启动命令。

18:51:34 再次确认已到 **1,800/198,000 步（0.909 epoch）**，进程仍存活；最近 500 步平均约 0.107 秒/步，预计剩余预训练约 5.8 小时。之后还有四组冻结分类评估，初步为全流程预留约 7–9 小时。硬件为 RTX 5090 32 GB。期间一次 SSH 状态查询超时，重连成功后确认训练持续推进，没有重复启动。

项目根目录：`/root/autodl-tmp/sepa-plan-b-a659af079672/sepa-plan-b`。

运行目录：`runs-ijepa/official-small240-s0-3e242d05b38e`。

完整运行 ID：`3e242d05b38e789016cceb96b7c1d55983842c5d640b508965442a6576b8558f`。

## 固定协议

- 固定官方提交：`facebookresearch/ijepa@52c1ae95d05f743e000e8f10a1f3a79b10cff048`。从本地 Git 的提交对象导出干净快照，不使用工作区未追踪的旧病理训练脚本；逐文件校验官方 Python 源码。
- 官方 ViT-S/16：12 层、384 维、6 个 attention heads，约 2,159 万可训练参数。
- 全图输入 240×240，共 15×15 个 patch；官方 predictor 为 6 层、384 维，约 1,094 万可训练参数。
- 同一 ImageNet-100：训练 126,684 张、验证 5,000 张，清单指纹与本轮 SEPA 完全一致。
- seed=0、batch=64、100 epoch；每轮 1,980 步，最后一批 28 张，合计 198,000 步。
- 官方多区域 mask、patch 级目标、teacher 特征 LayerNorm、Smooth L1。
- 峰值 LR=1e-4、warmup=4,950 步、最终 LR=1e-6；AdamW 排除 bias/一维参数的 weight decay，WD 从 0.04 余弦增加到 0.4；EMA 从 0.996 线性增加。
- 官方 RandomResizedCrop 0.3–1.0，不用 flip/color jitter/blur。BF16 autocast、FP32 参数；8 个数据加载进程、8 个 CPU 计算线程。
- 每轮保存完整 checkpoint；第 10、25、50、100 轮另存权重和 SHA256。支持相同协议下恢复，恢复检查包含模型、优化器、调度进度和 RNG。

这是一项**官方 I-JEPA 算法的同数据、同 encoder 规模、同训练轮数适配实验**。不是论文 ImageNet-1K 数值复现，也不是等 FLOPs 比较。predictor、增强、优化器细节及预训练精度与 SEPA 有明确差异，不能将结果差异全部归因于单个因素。峰值 LR 和 warmup 沿用 SEPA 的值；单次运行不代表 I-JEPA 的最优设置。

## 训练前检查

11 项真实图片检查通过：隐藏像素隔离、目标 block/样本索引顺序、图块还原整图、不同 worker 数的数据和 mask 一致性、恢复后的样本一致性、encoder 参数更新、teacher 无梯度、恢复后权重逐位一致、最后短 batch、调度终值、BF16 梯度有限。

官方 mask 采样器在反复失败时可能放宽 overlap 限制；适配器会检查并重采样，确保 context 与 target 不重叠。数据增强和 mask 都按样本/批次设定独立种子，避免 DataLoader 调度改变实验。

小规模检查峰值显存约 4.9 GB，预热后单次更新约 0.071 秒。正式运行早期包含数据加载约 0.11 秒/步；估时应以持续训练日志为准。

## 自动评估及关机

固定使用第 100 轮 checkpoint，官方验证集不参与挑选 checkpoint 或超参数：

1. I-JEPA student 和 EMA teacher 分别提取整图特征。
2. 用同一中心裁剪/缩放及 FP32 特征提取协议，复用 SEPA 的冻结线性分类评估：SGD、LR=0.005、200 轮、batch=128，标准化统计只来自训练集。
3. 报告 Top-1、Top-5、kNN、训练集特征有效秩/方差/余弦统计，保留分类器和预测结果。
4. 补测已完成的 SEPA k=0、k=3 的 100 epoch EMA teacher，使用相同评估流程。
5. 所有评估成功、结果落盘并同步磁盘后，执行平台关机脚本。失败时记录 `needs_review`，不会把失败当成完成。

## 状态与恢复

- `server-logs/ijepa_status.json`：当前阶段、步数、速度、loss、进程和结果位置。
- `server-logs/ijepa_launch.json`：服务器端口、PID、启动日志。
- `server-logs/ijepa_smoke_receipt.json`：全部检查、代码指纹、参数量和速度回执。
- `server-logs/ijepa-20260913T104816Z.log`：本次后台进程日志。
- 运行目录内 `protocol.json`、`history.jsonl`、`latest.pt`、`milestones/`、`evaluation/`、`summary.json`。
- 本地启动证据在 `server/evidence/ijepa-20260913/`，为采样快照，不会自动更新。
- 恢复前确认旧进程已退出，再使用同一脚本的 `--resume --shutdown-on-success`。不要更改源码或配置后加载旧 checkpoint。

## 最终结果

2026-09-14 01:12:54（台北时间）全部完成，训练达到 198,000/198,000 步。预训练用时 21,123 秒，约 **5 小时 52 分钟**；随后依次完成四组冻结特征与线性分类评估。四个里程碑 checkpoint、最终 checkpoint、分类器、预测结果和特征健康诊断均已落盘。

| Encoder | Top-1 | Top-5 | kNN | 有效秩（384 维） |
|---|---:|---:|---:|---:|
| I-JEPA student | **52.92%** | **80.84%** | **42.42%** | 90.48 |
| I-JEPA EMA teacher | 52.86% | 80.74% | 42.36% | **91.37** |
| SEPA k=0 student | 26.78% | 55.54% | 18.20% | 未在本次重算 |
| SEPA k=0 EMA teacher | 27.00% | 55.30% | 18.40% | 63.63 |
| SEPA k=3 student | 30.34% | 59.48% | 20.16% | 未在本次重算 |
| SEPA k=3 EMA teacher | 30.60% | 59.68% | 20.30% | 49.33 |

以相同分支比较，I-JEPA EMA teacher 比 SEPA k=3 EMA teacher 高 **22.26 个 Top-1 百分点**；student 比 student 高 **22.58 个百分点**。EMA 不能解释原来的低分：SEPA k=0、k=3 换用 teacher 仅分别提高 0.22、0.26 个百分点，I-JEPA teacher 反而比 student 低 0.06 个百分点。

同一个线性分类器协议在第 200 轮的在线训练 Top-1：I-JEPA student 62.34%、teacher 62.26%、SEPA k=0 teacher 34.49%、SEPA k=3 teacher 37.68%。差距在训练集上也存在，但这些数值本身不能证明优化已经收敛，也不能单独排除分类器设置与不同特征分布的相互作用。

特征也没有完全变成常数，但 SEPA 的结构明显较弱。I-JEPA teacher 有效秩为 91.37，第一主成分占 5.74%；SEPA k=3 teacher 有效秩 49.33，第一主成分占 11.03%。中心化后同类余弦均值分别为 0.1564 和 0.0993，异类均接近零；I-JEPA 的类别相关结构更清楚。

## 结论边界

这项结果能够确认：**当前 SEPA 方案在固定语义分类评估下明显弱于 I-JEPA 对照，30% 不是这套数据与 ViT-S/16 的表现上限。** 同一数据与冻结分类评估流程能够得到明显更高的分数；EMA 分支选择的影响较小。实验尚未排除 SEPA 专属实现问题、训练配置不适配或特征与分类器设置之间的相互作用，也不能断言延长训练完全没有作用。

它尚不能单独证明“任何 SEPA 原理都不成立”，因为 I-JEPA 与 SEPA 同时存在几项设计差异：整图联合注意力与 tile-local 编码、patch 级多块目标与 tile 均值目标、Smooth L1 与归一化平方距离、predictor 容量和优化细节。当前实验定位的是这组差异的合计影响，不能把 22 个百分点全部归因于置换。

SEPA k=3 仍比自身 k=0 高约 3.6 个百分点，说明置换在这套弱基座上有相对收益；但这种相对收益无法弥补相对 I-JEPA 的大幅下降，也尚未证明空间能力。下一步应按单因素消融定位下降来源，优先检查 tile-local teacher/encoder 和 tile 均值目标，而不是继续延长现有 SEPA 训练。

运行状态为 `complete`，最终 checkpoint SHA256 为 `7a651508975a97ab68b7a67129bea0cb737f1ef8ff2f3bb9ffb154fa7c0cd412`。自动关机请求已在结果写盘并同步后触发，原 PID 已退出；SSH 查询时实例仍可短暂访问，因此记录为“关机已请求”，不将 SSH 可访问性作为关机成功与否的判据。
