# SEPA V2 25-epoch 筛选（seed 0）

2026-09-15 00:42:41（UTC+8）在 `connect.westb.seetacloud.com:47026` 启动，RTX 4090 D 24 GB。两组均完成25 epoch（49,500步），统一内部评估于03:53:04完成；00:42启动至评估结束共约3小时10分钟。03:53:28记录了校验成功后的关机请求。以下为22:28再次读取并校验后的最终结果。

## 最终筛选结果

以下三组均为25 epoch、seed 0、全局student encoder；统一使用113,930张内部训练标签拟合线性分类器，在12,668张内部开发图像上评估。原126,684张训练图中的86条冲突重复记录被排除，相同图像内容不会跨内部划分。这是从预训练集划出的开发集，图像曾参与无标签预训练；不是5,000张ImageNet-100官方验证集，也不能与此前100 epoch的官方验证准确率直接相减。

| 方法 | Top-1 | Top-5 | kNN | 特征有效秩 |
|---|---:|---:|---:|---:|
| I-JEPA基线 | 49.86% | 77.04% | 41.40% | 62.30 |
| V2-k0：75%全局+25%局部，不置换 | 45.26% | 73.79% | 33.75% | 30.90 |
| V2-k3：相同混合比例，局部分支置换 | 46.92% | 75.43% | 36.81% | 34.74 |

- k3相对匹配的k0：Top-1增加1.66个百分点、kNN增加3.06个百分点；这是本次单seed下受控置换的正向结果。
- k0相对纯I-JEPA：Top-1下降4.59个百分点。k3相对纯I-JEPA：Top-1仍下降2.94个百分点、kNN下降4.59个百分点。
- 因此这套V2在25 epoch时尚未带来超过纯I-JEPA的整体分类收益，未达到先前提出的“与同epoch基线差距不超过2个百分点”筛选条件；不自动继续到100 epoch。
- 有效秩在相同10,000张训练图上计算，V2-k3为34.74，基线为62.30，说明方差更集中。该诊断不能单独证明表征坍塌或其因果机制。
- 本轮实际完成分类、kNN与特征健康评估，尚未运行空间探针。启动前predictor对置换的响应检查不等于学到了更好的空间表征。若继续研究，应先补齐冻结checkpoint上的空间探针与捷径对照，再决定是否追加预训练。
- 以上是一个seed的初步趋势；没有跨seed方差或置信区间。三组probe均按相同配置跑200轮，这本身不证明它们都已充分收敛。

## 完整性与完成时间

22:28重新校验通过：三组结果/分类器回执与特征缓存SHA-256、数据身份与checkpoint步数、k0/k3模型和调度器状态均为49,500步、baseline确为原I-JEPA第25 epoch。训练日志中的loss与梯度范数均有限。

- k0预训练于03:17:54结束，运行时间约2小时35分。
- k3预训练于03:25:12结束，运行时间约2小时35分。
- 三组内部评估于03:53:04全部结束，协调器于03:53:05标记complete。
- 汇总SHA-256：`ac2df5cdff25ad7bdaccb4d4118572f7495e98b602dbf371f093e35db55d4dfd`。
- k0最终checkpoint：`799dcf0f8b36a7b486712b229d31c2938db669703fbe90002763ff6da21911eb`。
- k3最终checkpoint：`2f59bb6a0c543e4c2e885506c94cf28b4c39217121c1c80efb2bc189172b4427`。
- 关机监控于03:53:28写入校验通过后的requested回执；没有平台停止计费回执。此次读取时容器PID 1的启动时间为当日20:49:52，说明当前实例已在实验完成后重新启动。

最终汇总、三份结果与回执、特征缓存回执、关机记录已下载到 `server/evidence/ijepa-v2-20260915/`。大体积checkpoint和特征张量仍保存在服务器。

## 目的

当前 SEPA k3 的 ImageNet-100 Top-1 为 30.34%，但同数据的全局 I-JEPA 为 52.92%。已完成的单变量实验表明，只把 I-JEPA student 的预训练注意力限制在80×80图块内，会使 Top-1 降至约46.7%，不能解释全部差距。

V2 保留强 I-JEPA 语义训练，把 SEPA 作为25%的空间训练分支。k0与k3唯一的差别是局部分支是否对三个可见非中心图块进行受控置换。

## 固定设计

| 项目 | V2设计 |
|---|---|
| 数据与顺序 | 原126,684张ImageNet-100训练图、seed 0、batch 64 |
| 输入 | 240×240；严格切成3×3个原生80×80图块；无补边、无放大 |
| 批次比例 | 固定循环：3个标准全局I-JEPA批次 + 1个局部SEPA批次 |
| 全局分支 | 官方I-JEPA encoder、multiblock mask、teacher目标、predictor、loss完全不变 |
| 局部student | 7个可见图块独立编码；每块保留25个patch token；共享5×5块内PE，不含canonical或observed图块身份 |
| 局部teacher | 干净的240×240 canonical整图，EMA，无梯度 |
| Predictor | 官方6层、384维；为context加入observed位置，为query加入canonical位置 |
| 局部目标 | 两个隐藏非中心canonical图块的全部50个patch embedding |
| k0 | 图块不置换，作为匹配的局部难度对照 |
| k3 | 从6个可见非中心图块选3个做uniform permutation，允许fixed point；中心固定 |
| 训练日程 | 沿用100-epoch I-JEPA LR、WD和EMA日程，先运行至第25 epoch；若晋级可从checkpoint无缝继续 |

两组各49,500步，其中37,125个全局批次、12,375个局部批次。最初采用顺序执行；实测单任务只分配约5.5 GB CUDA显存后，按用户要求改为双进程并行。

## 筛选评估

25 epoch后，对原I-JEPA第25 epoch、V2-k0和V2-k3的全局student提取同一批训练特征。按图像内容分组做固定90/10内部划分；冲突重复图像被排除，重复内容不会跨划分泄露。使用统一的FP32、200-epoch线性probe、kNN和固定10,000张特征健康子集。ImageNet-100 val在筛选阶段保持未使用。

只有当k3相对k0产生空间信号且分类/有效秩没有明显恶化时，才继续到100 epoch和正式验证集评估。

## 启动前检查

15项检查通过：全局更新与原I-JEPA的损失、模型及优化器状态逐位一致；k0/k3隐藏图块完全配对；两个隐藏图块均非中心且中心固定；置换映射方向正确；局部编码与逐个原生80px图块参考一致；encoder不接收图块slot身份；隐藏像素隔离；predictor会响应observed位置变化；50个patch目标；真实BF16局部更新；teacher仅EMA无梯度；局部分支checkpoint精确恢复；75/25比例和第25 epoch最后28张小批次正确。

- 原生80px参考最大误差：2.15e-6。
- k0/k3 predictor平均输出差：9.63e-4，高于数值噪声。
- 冒烟局部步耗时：0.186秒；峰值CUDA分配约5.06 GB。

## 运行记录

- Screen ID：`0242b3dd60e1cecded377ad5f174900432ccfb70e097deea6a968cd1a374363c`。
- k0 Run ID：`656eafd79cbc84ded469a18149f6d69514681cfec80f90d2c7c69dd4c6956ab1`。
- k3 Run ID：`bf70e2311020f1e8dfb6ad26066aecbc450c22816fe059766411879288733c44`。
- 服务器根目录：`/root/autodl-tmp/sepa-plan-b-a659af079672/sepa-plan-b`。
- 状态：`server-logs/ijepa_v2_status.json`、`server-logs/ijepa_v2_k0_status.json`、`server-logs/ijepa_v2_k3_status.json`。
- 日志：`server-logs/ijepa_v2.log`。
- 代码：`server/train_ijepa_v2.py`；检查：`server/test_ijepa_v2.py`；配置：`server/ijepa_v2_config.json`。
- 并行协调：`server/coordinate_ijepa_v2_parallel.py`；成功后关机：`server/shutdown_after_ijepa_v2.py`。
- 本地证据：`server/evidence/ijepa-v2-20260915/`。

正式运行已确认全局和局部分支均执行，初始loss、梯度和显存正常。00:49时k0已越过第1 epoch并保存checkpoint；停止仅负责顺序等待的旧协调进程，k0训练进程未中断，随后由 `server/coordinate_ijepa_v2_parallel.py` 启动k3并等待两组完成后统一评估。

00:50:30时，k0在第2,925步、k3在第100步，两者均持续推进。并行后GPU利用率连续读数为98%，显存约11.8/24.6 GB，功耗约307 W；k0/k3近期步速约0.17/0.22秒。按较慢的k3估算，并行训练约3小时，加入checkpoint和内部评估后约3.5小时。并行协调进程PID 19947，k0 PID 19351，k3 PID 19951；并行启动与状态回执保存在本地证据目录。运行中断后可从上一个完整epoch恢复；互斥锁阻止重复arm进程。

01:43:20启动关机监控PID 21746。它等待并行协调状态变为 `complete`，然后重新验证三组内部评估结果及分类器回执、k0/k3第25 epoch checkpoint与里程碑SHA-256；全部通过才写入 `shutdown.json` 并请求关机。如果训练、评估或校验失败，状态写为 `needs_review` 并保留服务器。01:43:42时k0/k3分别在第19,700/16,850步，GPU利用率100%，训练未受监控影响。
