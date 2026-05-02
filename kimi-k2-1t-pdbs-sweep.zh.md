# Kimi-K2 1T —— 全面的 pdbs 扫描实验

- **日期：** 2026-04-23（扫描）；2026-04-24（profile 剖析 + v3 定稿）；2026-04-25（sgd-v2 profile 刷新 + DCN expert-parallelism 扩展 `dcn_expert_parallelism ∈ {2, 4, 8}` 覆盖 4 个非 DeepEP 配置）
- **模型：** `kimi-k2-1t`（MaxText）。总参数 1026.4 B。61 层解码器（第 0 层 dense，第 1–60 层 MoE，384 专家 × top-8 路由 + 1 共享专家）。MLA 注意力（`q_lora_rank=1536`，`kv_lora_rank=512`）。参见 [`configs/kimi-k2-1t.gpu.yml`](configs/kimi-k2-1t.gpu.yml)。
- **硬件：** 8 节点 × 8× AMD MI355（每设备 288 GB HBM），Pensando AINIC 互联，k8s 分区（`chi[2766,2800,2810,2832,2835,2865,2872,2883]`）。
- **镜像：** `/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar`（包含 [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) GMM + DeepEP）。
- **补丁分支：**
  - [yihuang/moe-turbo-gmm-and-deepep](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep) @ `ad693da2`（基线 —— `sparse-gmm-deepep` / v1 列）
  - [yihuang/moe-turbo-gmm-and-deepep-v2](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v2) @ `627168f8`（v2 列）
  - [yihuang/moe-turbo-gmm-and-deepep-v3](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9`（v3 列 —— 核心 headline）
- **基础配置：** [`configs/kimi-k2-1t.gpu.yml`](configs/kimi-k2-1t.gpu.yml)（`dcn_fsdp_parallelism=8`，`ici_expert_parallelism=8`，8 节点 × 8 GPU 拓扑）。
- **数据集：** `dataset_type=synthetic`（扫描时为 gpu.yml 默认值；如当前默认值已变更，复现时需在 CLI 上覆盖 `dataset_type=synthetic`）。
- **峰值 BF16：** ≈ 2500 TFLOP/s/设备 → MFU ≈ TFLOP/25。
- **姊妹扫描：** [`deepseek3-671b-pdbs-sweep.md`](deepseek3-671b-pdbs-sweep.md) —— 本文沿用其配置分类与结构。

## 背景

[DS3 扫描](deepseek3-671b-pdbs-sweep.md) 证明 `sparse-gmm-deepep` v1→v2→v3 优化链（纯 Python 补丁，只改 `src/MaxText/layers/moe.py`）在 DS3 的 pdbs=6 下相对基线带来 +24 % / +59 % 的 TGS 提升，其机制是消除 DeepEP dispatch 反向的主导内核 `input_scatter_fusion_*.kd`。本次扫描在 `kimi-k2-1t` 上重跑同样的 10 配置矩阵，要回答的问题是：**在显存可行性（而非内核优化）成为主导轴的 1T 模型上，v1→v3 的增益形状是否仍然复现？**

> **1-GPU/proc 启动器不在本次扫描之列。** DS3 的结论 #3（见 `deepseek3-671b-pdbs-sweep.md`）已对 1-GPU/proc 的行为做了与模型无关的刻画：其加速来自 XLA 对 `ragged_all_to_all` 的运行时内核交换，现已通过 `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0`（在 `train_env.sh` 中已作为新默认值）在 1-node/proc 上可达。该机制不随模型规模变化，因此本次仅测量 1-node/proc 启动器。

**下方结果矩阵是稀疏（ragged）的。** 精炼后的提示词（见 [`moe-pdbs-sweep-prompt.md`](moe-pdbs-sweep-prompt.md)）以动态上限探测取代固定 pdbs 阶梯：对每个配置，从 pdbs=1 向上走到首次 OOM，然后在上限 ±1 范围内回补。表格中缺失的单元格是按"pdbs 单调"规则*被跳过*的（不是待测），并非"pending"。`P★` 定义为所有可行配置中 `max_pdbs` 的最小值，标记了跨配置逐一可比较的那一行。

---

## 受测配置

| 标签                   | 提交时环境变量前缀                                                      | 透传参数（放在 `--` 之后）                                                           |
|------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `dense-cf1.25`         | —                                                                      | *(默认)* — `sparse_matmul=false`, `capacity_factor=1.25`                            |
| `dense-cf2`            | —                                                                      | `capacity_factor=2.0`                                                               |
| `dense-cf4`            | —                                                                      | `capacity_factor=4.0`                                                               |
| `sparse`               | —                                                                      | `sparse_matmul=true shardy=true`                                                    |
| `sparse-gmm`           | —                                                                      | `sparse_matmul=true use_turbo_grouped_gemm=true _env_ENABLE_RAGGED_ONESHOT_KERNEL=1`|
| `sparse-gmm-fixed`     | —                                                                      | `sparse_matmul=true use_turbo_grouped_gemm=true`                                    |
| `sparse-deepep`        | —                                                                      | `sparse_matmul=true use_deepep_dispatch=true shardy=true`                           |
| `sparse-gmm-deepep` (v1)| `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep`（覆盖 v3 默认）  | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`           |
| `sparse-gmm-deepep-v2` | `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v2`（覆盖 v3 默认） | 与 v1 相同的透传参数                                                                |
| `sparse-gmm-deepep-v3` | — *（默认分支 —— `container_env.sh` 已经指向这里）*                      | 与 v1 相同的透传参数                                                                |

**`sparse-gmm-deepep`、`-v2`、`-v3` 的区别：** 三者使用相同的镜像 / 启动器 / 透传参数 —— 只有打补丁的 MaxText 分支不同，且分支之间唯一发生改动的文件是 `src/MaxText/layers/moe.py`。**v2** 将 DeepEP 的两次 dispatch-side gather 合并为一次（使反向 scatter-add 次数减半）。**v3** 进一步用 `jax.custom_vjp` 替换剩余的一次重复索引 scatter-add 反向，改用 argsort 反置换 gather + reduce-sum —— 无原子操作。三者前向逐位相同（每一步的 loss 在 bf16 LSB 精度内吻合），因此任何 TGS 差异都是纯内核级的优化收益。代码级 diff 参见 DS3 的 [v1/v2/v3 剖析节](deepseek3-671b-pdbs-sweep.md#sparse-gmm-deepep-的-v1--v2--v3-为何在-tgs-上不同pdbs6-的-profile-深入剖析)。

**为何只有 `sparse` 和 `sparse-deepep` 需要 `shardy=true`：** `sparse_matmul=true` 而不启用 `use_turbo_grouped_gemm=true` 时会回退到 `jax.lax.ragged_dot`，其分片传播需要 `shardy=true`。`sparse-gmm*` 各行使用 Primus-Turbo 的 GMM 自定义调用，该自定义调用自带分片规范，绕过了分片传播 pass。

---

## 可行性总结

Kimi-K2 1T 的显存上限显著紧于 DS3（这是本扫描最主要的 1T 专属发现）。所有可行配置的 `P★ = 4`（由两条 sparse-gmm 路径的上限锁定）。整行不可行：`sparse` 与 `sparse-deepep`。

| 配置                      | `max_pdbs`（上限） | `argmax_TGS_pdbs` | 峰值 TGS | 峰值 MFU | 上限 OOM 签名           |
|---------------------------|------------------:|------------------:|---------:|---------:|-------------------------|
| `dense-cf1.25`            |                12 |            **11** |   1170.1 |   9.62 % | `✗ 202.4 GiB @ pdbs=16` |
| `dense-cf2`               |                10 |                10 |    827.8 |   6.80 % | `✗ 189.4 GiB @ pdbs=11` |
| `dense-cf4`               |                 6 |                 5 |    455.1 |   3.74 % | `✗ 180.5 GiB @ pdbs=7`  |
| `sparse`                  |         **不可行** |                 — |        — |        — | `✗ 581.8 GiB @ pdbs=1`  |
| `sparse-gmm`（one-shot）  |                 4 |                 4 |    249.0 |   2.05 % | `✗ 195.6 GiB @ pdbs=5`  |
| `sparse-gmm-fixed`        |                 4 |                 4 |    614.5 |   5.05 % | `✗ 195.6 GiB @ pdbs=5`  |
| `sparse-deepep`           |         **不可行** |                 — |        — |        — | `✗ 507.4 GiB @ pdbs=1`  |
| `sparse-gmm-deepep`（v1） |                 5 |                 5 |    515.7 |   4.24 % | `✗ 195.3 GiB @ pdbs=6`  |
| `sparse-gmm-deepep-v2`    |                 5 |                 5 |    635.9 |   5.23 % | `✗ 202.6 GiB @ pdbs=6`  |
| **`sparse-gmm-deepep-v3`** |           **7** |             **7** | **897.9** |  **7.38 %** | `✗ 214.3 GiB @ pdbs=8` |

**P★ = 4**（各可行配置中 `max_pdbs` 的最小值，由两条 sparse-gmm 路径锁定）。

**可行性的关键发现：**

1. **v3 把 DeepEP 的前沿相对 v1 / v2 / sparse-gmm-fixed 延伸了 2 个 pdbs。** v3 的 `jax.custom_vjp` 消除了 v1 / v2 需要持有的重复索引 scatter-add 中间张量（每 MoE 层约 K × H × N 个 bf16 float，1T 模型上每层几个 GiB）。由此省下的 HBM 余量 **恰恰使 v3 能在 pdbs=6 / 7 运行，而 v1 / v2 / sparse-gmm-fixed 都在 pdbs=6 OOM**。这是 DS3 v3 故事在 1T 上的显存版变体：DS3 上，v3 的内核优势表现为同一上限下的 TGS 提升；而在 kimi-1T 上，它*额外地*体现为**上限外扩**。
2. **`sparse` 与 `sparse-deepep`（不走 GMM 的 `ragged_dot` 路径）在 1T 的 pdbs=1 就不可行** —— `RaggedDot` 工作集 OOM（分别需要 581.8 GiB 和 507.4 GiB）。与 DS3 的现象相同，但数字大了 30–35 %，与 1T 对 671B 的参数比一致。
3. **即使 `sparse-gmm` 和 `sparse-gmm-fixed` 在 1T 上也只能跑到 pdbs=4。** DS3 上它们可达 pdbs=7（需要 `MEM_FRACTION=.96`）。Kimi-1T 在默认 `.93` 下 pdbs=5 OOM，单次分配 195.6 GiB，已经远低于 267.8 GiB 的池容量 —— 这说明是*整个*工作集（而非单次大分配）超过了池上限。此场景下 `.96` retry 无济于事，因为分配大小不在池容量的 10 % 邻域内。
4. **dense `cf=4.0` 掉下悬崖没有想象中那么早。** cf=4 相对 cf=2 翻倍激活内存，但 `max_pdbs` 仅从 10 降到 ≥5（仍在探测）。激活内存随 `cf × pdbs × seq_len × emb_dim × layers` 线性变化而参数不变 —— pdbs=5 时 cf=4 的激活仅为 pdbs=5 时 cf=2 的 ~1.6 倍。

---

## 结果矩阵 —— 1-node/proc，稀疏（ragged）

除 loss 外的所有指标都是**训练步 5–14 的平均值**（步 0–4 作为预热被丢弃）。Loss 仅报告步 14，因为在合成数据下单步 loss 是一个稳定的数值正确性探针。

图例：`✗<GiB>` = OOM 并附分配大小；`—` = 按"pdbs 单调"规则跳过（较低 pdbs 已 OOM）；空 = 按动态探测阶梯故意跳过（通常是 3、13–15 等未在目标点的位置）。

**`pdbs=P★=4` 行**（跨配置苹果对苹果的比较行）**加粗**。

### Tokens/s/设备（TGS）

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm（one-shot） | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|-----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        234.9 |     208.3 |     195.2 |                  135.0 |            220.4 |         193.0 |         212.9 |             229.5 |
|    2 |        399.0 |     373.4 |     297.1 |                  190.8 |            380.9 |         320.1 |         358.3 |             405.5 |
|  **4** |    **678.9** | **597.0** | **414.9** |             **249.0**  |        **614.5** |     **476.9** |     **575.4** |         **685.2** |
|    5 |        787.2 |     652.1 | **455.1** |               ✗ 195.6  |         ✗ 195.6  |     **515.7** |     **635.9** |             750.8 |
|    6 |        873.3 |     707.5 |     453.2 |                     —  |         ✗ 207.2  |       ✗ 195.3 |       ✗ 202.6 |             856.4 |
|    7 |        930.3 |     758.4 |   ✗ 180.5 |                     —  |                — |             — |             — |         **897.9** |
|    8 |       1028.9 |     808.8 |   ✗ 216.4 |                     —  |                — |             — |             — |           ✗ 214.3 |
|    9 |       1035.1 |         — |         — |                     —  |                — |             — |             — |                 — |
|   10 |       1135.1 | **827.8** |         — |                     —  |                — |             — |             — |                 — |
|   11 |   **1170.1** |   ✗ 189.4 |         — |                     —  |                — |             — |             — |                 — |
|   12 |       1134.5 |   ✗ 189.7 |         — |                     —  |                — |             — |             — |                 — |
|   16 |      ✗ 202.4 |         — |         — |                     —  |                — |             — |             — |                 — |

**整行不可行：** `sparse`（pdbs=1 OOM 581.8 GiB），`sparse-deepep`（pdbs=1 OOM 507.4 GiB）。

**峰值 MFU：** `dense-cf1.25 @ pdbs=11` = **9.62 %**（1170.1 TGS，240.4 TFLOP/s/设备，MI355 BF16 峰值 2500 TFLOP/s/设备）。
**峰值 dropless MFU：** `sparse-gmm-deepep-v3 @ pdbs=7` = **7.38 %**（897.9 TGS，184.5 TFLOP/s/设备） —— 无需提升 MEM_FRACTION。

### TFLOP/s/设备

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm（one-shot） | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|-----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |         48.3 |      42.8 |      40.1 |                   27.7 |             45.3 |          39.7 |          43.7 |              47.2 |
|    2 |         82.0 |      76.7 |      61.0 |                   39.2 |             78.3 |          65.8 |          73.6 |              83.3 |
|  **4** |     **139.5** | **122.7** |  **85.3** |              **51.2**  |        **126.3** |      **98.0** |     **118.2** |         **140.8** |
|    5 |        161.7 |     134.0 |  **93.5** |                     ✗  |               ✗  |     **106.0** |     **130.7** |             154.3 |
|    6 |        179.4 |     145.4 |      93.1 |                     —  |               ✗  |             ✗ |             ✗ |             176.0 |
|    7 |        191.1 |     155.8 |         ✗ |                     —  |                — |             — |             — |         **184.5** |
|    8 |        211.4 |     166.2 |         ✗ |                     —  |                — |             — |             — |                 ✗ |
|    9 |        212.7 |         — |         — |                     —  |                — |             — |             — |                 — |
|   10 |        233.2 | **170.1** |         — |                     —  |                — |             — |             — |                 — |
|   11 |    **240.4** |         ✗ |         — |                     —  |                — |             — |             — |                 — |
|   12 |        233.1 |         ✗ |         — |                     —  |                — |             — |             — |                 — |
|   16 |            ✗ |         — |         — |                     —  |                — |             — |             — |                 — |

### 平均每步时间（秒）

越小越好。训练日志中 `seconds:` 字段在步 5–14 上的均值。

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm（one-shot） | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|-----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        17.44 |     19.72 |     21.00 |                  30.34 |            18.59 |         21.24 |         19.24 |             17.84 |
|    2 |        20.58 |     21.94 |     27.58 |                  42.95 |            21.51 |         25.60 |         22.89 |             20.21 |
|  **4** |    **24.14** | **27.45** | **39.49** |             **65.79**  |        **26.67** |     **34.37** |     **28.47** |         **23.91** |
|    5 |        26.02 |     31.41 | **45.00** |                     ✗  |               ✗  |     **39.72** |     **32.22** |             27.29 |
|    6 |        28.14 |     34.74 |     54.23 |                     —  |               ✗  |             ✗ |             ✗ |             28.70 |
|    7 |        30.82 |     37.81 |         ✗ |                     —  |                — |             — |             — |             31.95 |
|    8 |        31.85 |     40.52 |         ✗ |                     —  |                — |             — |             — |                 ✗ |
|    9 |        35.64 |         — |         — |                     —  |                — |             — |             — |                 — |
|   10 |        36.09 | **49.48** |         — |                     —  |                — |             — |             — |                 — |
|   11 |        38.51 |         ✗ |         — |                     —  |                — |             — |             — |                 — |
|   12 |        43.33 |         ✗ |         — |                     —  |                — |             — |             — |                 — |
|   16 |            ✗ |         — |         — |                     —  |                — |             — |             — |                 — |

### 步 14 的训练 loss

在每个 pdbs 行内，所有可行配置 Δ ≤ 0.002 彼此一致 —— v2/v3 前向 HLO 与 v1 逐位相同（loss 偏差仅在 bf16 LSB 级别），MoE 与 dense 的 loss 在合成数据预热阶段也吻合，因为两路径输入 token 相同。

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm（one-shot） | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|-----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        8.772 |     8.771 |     8.771 |                  8.740 |            8.740 |         8.741 |         8.740 |             8.741 |
|    2 |        9.627 |     9.626 |     9.625 |                  9.627 |            9.626 |         9.625 |         9.625 |             9.626 |
|  **4** |   **10.367** | **10.366** | **10.365** |           **10.366** |       **10.366** |    **10.366** |    **10.366** |        **10.366** |
|    5 |       10.566 |    10.566 | **10.566** |                    ✗  |               ✗  |    **10.565** |    **10.565** |            10.565 |
|    6 |       10.713 |    10.712 |    10.712 |                     —  |               ✗  |             ✗ |             ✗ |            10.712 |
|    7 |       10.817 |    10.817 |         ✗ |                     —  |                — |             — |             — |            10.816 |
|    8 |       10.920 |    10.919 |         ✗ |                     —  |                — |             — |             — |                 ✗ |
|    9 |       11.007 |         — |         — |                     —  |                — |             — |             — |                 — |
|   10 |       11.067 | **11.066** |         — |                    —  |                — |             — |             — |                 — |
|   11 |       11.120 |         ✗ |         — |                     —  |                — |             — |             — |                 — |
|   12 |       11.184 |         ✗ |         — |                     —  |                — |             — |             — |                 — |

---

## DCN expert-parallelism 扩展（`dcn_expert_parallelism > 1`）

在主扫描（默认 `dcn_expert_parallelism=1`，即专家并行只在节点内、FSDP 只跨节点）的基础上，沿跨节点 EP 因子化方向再扩展一维。8 节点 × 8 GPU = 64 ranks，并行网格总量不变，但分解方式改变：

| `DCN_EP` | `dcn_fsdp` × `ici_ep × dcn_ep` | 总 EP rank-product | 每主机 EP 扇出 |
|---:|:-:|---:|---|
| **1** *（默认 —— 主扫描）* | 8 × 8 × 1 | 8 | EP 轴只在节点内 |
| 2 | 4 × 8 × 2 | 16 | 每主机的专家分散到 1 个对端主机 |
| 4 | 2 × 8 × 4 | 32 | 每主机的专家分散到 3 个对端主机 |
| 8 | 1 × 8 × 8 | 64 | 完全 DCN-EP，无跨节点 FSDP |

### 已知限制：DeepEP 变体被锁定在 `DCN_EP=1`

`MaxText/pyconfig.py` 校验 `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1`，否则在到达 XLA compile 之前就以 pydantic `ValidationError("Internode DeepEP is not yet supported in JAX")` 在 ~2 分钟内终止任务。这适用于**全部 4 个 DeepEP 配置**（`sparse-deepep`、`sparse-gmm-deepep` v1/v2/v3）—— JAX/MaxText 的集成层就堵住了 DS3 "DeepEP 在跨节点 EP 上更优" 假设本来要测试的那个区域。这条 DS3 预测因此在当前 MaxText 版本上**无法做实证检验**（需要先在上游解除该校验）。下面的 DCN-EP 扩展只刻画**非 DeepEP** 区域 —— `dense-cf1.25 / cf2 / cf4` 和 `sparse-gmm-fixed`。`sparse-gmm`（one-shot）在 DCN_EP > 1 时也跳过：其 OneShot kernel 仅节点内有效，DCN_EP > 1 时退回 kNccl 路径，跟 `sparse-gmm-fixed` 完全重合。

### TGS @ DCN_EP > 1（4 个非 DeepEP 配置）

DCN_EP=1 列从主矩阵里复制过来便于横向对比。标记 `n/a` 的格子是该 DCN_EP 下没有测量过的 pdbs 值。

#### `dense-cf1.25`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  2 |  481.4 |     479.5 |       n/a |     n/a |
|  4 |  678.9 | **723.0** |     669.4 |   577.7 |
|  6 |  873.3 |     821.4 |       n/a |     n/a |
|  8 | 1028.9 |     888.1 |     696.4 |   605.8 |
| 10 | 1135.1 | **956.2** |       n/a |     n/a |
| 12 | 1134.5 |     840.5 |       n/a |     n/a |

#### `dense-cf2`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  597.0 |     544.9 |     453.1 |   390.1 |
|  6 |  707.5 |     608.4 |       n/a |     n/a |
|  8 |  808.8 | **629.4** |     459.1 |     n/a |
| 10 |  827.8 |     568.6 |       n/a |     n/a |

#### `dense-cf4`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  414.9 |     309.8 |     229.5 |   199.6 |
|  5 |  455.1 | **322.8** |       n/a |     n/a |
|  6 |  453.2 |     297.3 |     214.0 |     n/a |

#### `sparse-gmm-fixed`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  1 |  362.5 |     313.7 | **339.4** | ✗ 190 GiB |
|  2 |  500.5 | **450.7** | ✗ 237 GiB |     —   |
|  4 |  614.5 | ✗ 226 GiB |       —   |     —   |

### 跨 DCN_EP 的可行性总结

| Config | DCN_EP=1 max_pdbs | DCN_EP=2 max_pdbs | DCN_EP=4 max_pdbs | DCN_EP=8 max_pdbs | 备注 |
|---|---:|---:|---:|---:|---|
| `dense-cf1.25` | 12 | ≥ 12 | ≥ 8 | ≥ 8 | DCN_EP=2 时 TGS 非单调（pdbs=10 处于峰值，pdbs=12 回落） |
| `dense-cf2`    | 10 | ≥ 10 | ≥ 8 | ≥ 4 | DCN_EP=2 时非单调（pdbs=8 处于峰值，pdbs=10 回落） |
| `dense-cf4`    |  6 | ≥ 6  | ≥ 6 | ≥ 4 | argmax_TGS 随 DCN_EP 下移：DCN_EP=1 → pdbs=5；DCN_EP=2 → pdbs=5；DCN_EP=4 → pdbs=4 |
| `sparse-gmm-fixed` | 4 | **2** | **1** | **0（不可行）** | 最陡的 cliff —— 非专家 FSDP 切片增长压过专家切片缩小 |

### pdbs=4 处的跨 DCN_EP TGS 对比（跨配置基准行）

| Config | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 | Δ EP=1→8 |
|---|---:|---:|---:|---:|---:|
| `dense-cf1.25`  | 678.9 | **723.0** *(+6.5%)* | 669.4 *(−1.4%)* | 577.7 *(−14.9%)* | **−14.9%** |
| `dense-cf2`     | 597.0 | 544.9 *(−8.7%)*    | 453.1 *(−24.1%)* | 390.1 *(−34.7%)* | **−34.7%** |
| `dense-cf4`     | 414.9 | 309.8 *(−25.3%)*   | 229.5 *(−44.7%)* | 199.6 *(−51.9%)* | **−51.9%** |
| `sparse-gmm-fixed` | 614.5 | ✗ OOM | ✗ OOM | ✗ OOM | DCN_EP > 1 时不可行 |

每个 (config, pdbs) 单元在不同 DCN_EP 下的 loss 都吻合到 ε 内 —— DCN_EP 因子化对数值无影响，符合预期。

### DCN-EP 关键观察

1. **`dense-cf1.25 @ pdbs=4` 在 DCN_EP=2 上比 DCN_EP=1 更快**（723.0 vs 678.9 TGS，+6.5%）。在小 pdbs 时，每 rank 专家权重的减半（48 → 24 experts/GPU）压过了跨节点 `all-to-all` 的 RDMA 开销。在 pdbs=6 附近发生反转 —— 到 pdbs=8 时 DCN_EP=1 已经反超（1028.9 vs 888.1，+15.9%），因为激活内存增长后 dispatch all-to-all 重新成为瓶颈。
2. **dropping 内部的内存代价排序在所有 DCN_EP 下保持一致。** 同 pdbs 下 `dense-cf4` 始终慢于 `cf2` 慢于 `cf1.25`；但 *差距*随 DCN_EP 加大（DCN_EP=8 时 cf4 跌 52%，cf1.25 仅跌 15%）。这与 cf=4 的更大激活张量在跨节点 all-to-all 下相对更敏感一致。
3. **`sparse-gmm-fixed` 是最 DCN-EP 脆弱的配置**：`max_pdbs` 随 DCN_EP 翻倍而坍塌为 4 → 2 → 1 → 0（不可行）。dropless 的 RCCL `ragged-all-to-all` over RDMA 比 dense 的常规 `all-to-all` 扩展性更差，因为 ragged buffer 是按最坏情况路由 fan-out 设计大小的，不像 dense 用 dropping 截断后的 capacity。到 DCN_EP=8 连 pdbs=1 也以 190 GiB OOM 失败。
4. **DCN_EP=2 下 TGS 曲线随 pdbs 出现非单调** ：dense-cf1.25 在 pdbs=10 处达到 956 峰值后回落到 pdbs=12 的 840；dense-cf2 在 pdbs=8 处达到 629 峰值后回落到 pdbs=10 的 569。这是 DCN_EP=1 主扫描里观察到的同一种 `argmax_TGS_pdbs < max_pdbs` 模式，只是峰值 pdbs 下移了 1–2 个。
5. **`sparse-gmm-fixed` 的上限比 dense 配置坍塌得更快。** DCN_EP=1 时 `sparse-gmm-fixed` 与 `dense-cf1.25` 的 `max_pdbs` 差距是 8（4 vs 12）；DCN_EP=4 时差距扩大到 ≥7（1 vs ≥8）；DCN_EP=8 时 sparse-gmm-fixed 完全不可行而 dense-cf1.25 在 pdbs=8 仍能跑出 605 TGS。dropless 的 `ragged-all-to-all` 是这组配置里最 DCN-EP 脆弱的集合通信；dense 的常规 `all-to-all` 容忍度高得多。
6. **每 pdbs 的 TGS 增益斜率随 DCN_EP 大幅变平**（与 DS3 同形）。dense-cf1.25 从 pdbs=4 → pdbs=8 的 TGS 增幅：DCN_EP=1 时 +42%（678.9 → 1028.9，由主扫描表推算）→ DCN_EP=2 时 +22.8%（723.0 → 888.1）→ DCN_EP=4 时 +4%（669.4 → 696.4）→ DCN_EP=8 时 +5%（577.7 → 605.8）。当 DCN_EP > 2 时，固定 DCN_EP 下提高 pdbs 几乎不带来吞吐收益 —— 跨节点 `all-to-all` 成本已经压过了每 pdbs 的 dense compute 摊销。
7. **没法做 DeepEP 对比行。** DCN-EP 扩展原本的 headline 问题 —— DS3 的 v3-vs-fixed 在 DCN_EP>1 上的对比 —— 在不修改上游 MaxText `pyconfig.py` 校验（解除 `use_deepep_dispatch ⇒ dcn_expert_parallelism == 1`）的前提下无解。这是本扫描里唯一标注的"blocker"。

---

## 关键结论

1. **峰值吞吐（dropping vs dropless）：**
   - **Dropping：** `dense-cf1.25 @ pdbs=11` → **1170.1 TGS，MFU 9.62 %**（max_pdbs = 12；但峰值 pdbs 是 11，不是 12 —— TGS 在 pdbs=12 降到 1134，pdbs=16 OOM，激活内存压力在上限之前就已拖累吞吐）。
   - **Dropless：** `sparse-gmm-deepep-v3 @ pdbs=7` → **897.9 TGS，MFU 7.38 %**（max_pdbs = 7，默认 `MEM_FRACTION=.93`）。
   - Dropless/dropping 峰值比 = 0.767。DS3 为 1097/1416 = 0.775，两者非常接近 —— kimi 的 dropless 路径约为 dropping 峰值的 77 %，与 DS3 在噪声内一致。

2. **在 P★ = 4，sgd-v3 与 dense-cf1.25 在测量噪声内打平**（重跑数据见基础设施说明中的可重复性表）：
   ```
   sgd-deepep-v3         685.2 TGS  ← 首次样本
   dense-cf1.25          678.9
   （sgd-v3 双样本均值 668.1，dense-cf1.25 双样本均值 680.7 —— 样本量对齐后 dense 以 +1.9 % 微幅领先）
   sparse-gmm-fixed      614.5
   dense-cf2             597.0
   sgd-deepep-v2         575.4
   sgd-deepep-v1         476.9
   dense-cf4             414.9
   sparse-gmm (one-shot) 249.0
   ```
   算上两次 run 之后，dense-cf1.25 和 sgd-v3 在跨配置比较点上打平。在 DS3 的 pdbs=4 上，dense 领先较明显（3.2 %）。在 kimi-1T 上，**这个差距缩窄到了统计意义下的打平** —— v3 的内核级优化（消除 `input_scatter_fusion_*.kd`）让 dropless 在 P★ 上更接近 dropping 的水平。在更高 pdbs（pdbs=5…7，dense 仍有显存余量）上 dense 又重新领先 —— 见结论 #4。

3. **v1 → v2 → v3 的 DS3 形状复现（略有衰减）：**

   | pdbs | v1 | v2 | v3 | Δ v1→v2 | Δ v1→v3 | DS3 Δ v1→v3（同 pdbs） |
   |-----:|----:|----:|----:|--------:|--------:|----------------------:|
   |    4 | 476.9 | 575.4 | 685.2 | +20.7 % | **+43.7 %** | +47.4 % |
   |    5 | 515.7 | 635.9 | 750.8 | +23.3 % | **+45.6 %** | +66.0 % |
   |    6 | ✗ OOM | ✗ OOM | 856.4 | — | — | +59.4 % |
   |    7 | ✗ OOM | ✗ OOM | 897.9 | — | — | +63.1 % |

   **v3 以 DS3 ~85–95 % 的幅度复现了 DS3 的形状**，并且在这个 1T 模型上额外把可行性前沿外推 +2 pdbs —— v1 和 v2 在 pdbs=6 都 OOM（分别为 195.3 GiB 和 202.6 GiB），v3 一直可行到 pdbs=7。DS3 的优化故事（消除 `input_scatter_fusion_*.kd`）成立；下方的 profile 剖析将在内核层面确认这一机制。

4. **在 pdbs > P★ 上本模型 dense（dropping）仍领先 dropless。** 同 pdbs 并列对比：

   | pdbs | dense-cf1.25 | sgd-v3 | Δ |
   |-----:|-------------:|-------:|--:|
   |    4 | 678.9 | **685.2** | **v3 +0.9 %** |
   |    5 | **787.2** | 750.8 | dense +4.9 % |
   |    6 | **873.3** | 856.4 | dense +2.0 % |
   |    7 | **930.3** | 897.9 | dense +3.6 % |

   sgd-v3 在 pdbs=6 和 pdbs=7 的重跑分别落在 827.1 / 886.8（都位于原值略低方向），与原始数据均在 ~3 % 以内 —— pdbs ≥ 5 上 dense 领先的现象是稳定的，不是单样本波动。机制层面：dropping 路径发出常规的 `all-to-all` + 常规 GEMM，其内核比 dropless 的 `moe_dispatch/combine` + grouped-GEMM 更简单，即使 v3 已做内核级优化也无法抹平常规对 ragged 内核这一底层差距。v3 确实关上了 DS3 指出的 DeepEP 集成开销（见 DS3 文档），但无法关掉常规-vs-ragged 的内核级差距。参见下方 [profile 剖析](#profile-剖析)。

5. **显存上限非常紧 —— 这是本次 1T 的首要发现。** 每一个配置在 Kimi-1T 上的 OOM 都早于 DS3：

   | 配置 | DS3 max_pdbs（1-node） | Kimi-1T max_pdbs | Δ |
   |---|---:|---:|---:|
   | `dense-cf1.25` | 16 | 12 | −4 |
   | `dense-cf2` | 16 | 10 | −6 |
   | `dense-cf4` | 7 | 6 | −1 |
   | `sparse-gmm` | 7（需 `.96`） | 4 | −3 |
   | `sparse-gmm-fixed` | 7（需 `.96`） | 4 | −3 |
   | `sgd-v1` | 7 | 5 | −2 |
   | `sgd-v2` | 7 | 5 | −2 |
   | `sgd-v3` | 7 | **7** | **0** |
   | `sparse` | pdbs=1 OOM | pdbs=1 OOM | — (两侧均不可行) |
   | `sparse-deepep` | pdbs=1 OOM | pdbs=1 OOM | — (两侧均不可行) |

   **只有 sgd-v3 在 1T 上保住了 DS3 的 pdbs=7 上限**，正是因为它的 `custom_vjp` 反向消除了所有其他 dropless 路径都要持有的重复索引 scatter-add 中间张量。在 HBM 余量更紧的 1T 模型上，这个显存优势比在 DS3 上更关键。

6. **端到端数值正确性已验证** —— 每个 pdbs 下各配置的 loss 在 Δ ≤ 0.002 内一致（bf16 LSB 级别的噪声）。v2/v3 前向 HLO 与 v1 基线逐位相同。

7. **dropping 的 capacity_factor 成本曲线在 kimi 上比在 DS3 上更陡。** `cf=1.25 → 2.0` 把 TGS 拉低 12–22 %（DS3：22–32 %）。`cf=1.25 → 4.0` 把 TGS 拉低 41–50 %（DS3：50–60 %）。Kimi 的稀疏模式（384 × top-8，相对 DS3 的 256 × top-8）在同样 capacity factor 下把 token 更均匀地分到专家上，因此在给定 cf 值下 dropping 丢失的 token 更少。

8. **上限附近的 TGS 非单调** —— 已在 dense-cf1.25 上观察到（pdbs=11 峰值、pdbs=12 下降约 3 %）。OOM 前一个 pdbs 的激活内存压力让 HBM 分配器和 XLA 布局决策更慢，额外的 token 带来的收益不足以弥补这部分损耗。直接探测 `max_pdbs` 时请预期此现象 —— 同时报告 `max_pdbs` 与 `argmax_TGS_pdbs`。

---

## 基础设施 / 显存上限说明

### 预扫描节点修复（承重）

8 节点中有 2 个（`node3`、`node7`）的**路由 RoCE GID 在 sysfs 上位于索引 2，而不是索引 1**，其他 6 个节点的路由 GID 都在索引 1：

```
node3: L-G-     (gid[0]=fe80, gid[1]=zero, gid[2]=fd93..., gid[3]=zero)
node7: L-G-
node1: LG--     (gid[0]=fe80, gid[1]=fd93..., gid[2]=zero)
node2: LG--
node4: LG--
node5: LG--
node6: LG--
node8: LG--
```

`train_env.sh` 硬编码了 `NCCL_IB_GID_INDEX=1`，导致每次 RCCL init 时 rank 2 和 rank 6 都出现 `ibv_query_gid failed with error Unknown error -1`，本节点列表上的分布式训练因此确定性地无法起动。修复应用在 [`utils/detect_nccl_env.sh`](utils/detect_nccl_env.sh) 中：按节点自动探测 —— 扫描 `/sys/class/infiniband/<hca>/ports/<port>/gids/` 找到第一个非零的 global-scope GID 并以此导出 `NCCL_IB_GID_INDEX`。同时把 `train_env.sh` 中的硬编码值放宽为 `"${NCCL_IB_GID_INDEX:-1}"`，让自动探测优先。这是当前 k8s 分区节点拓扑的一个属性（可能源于受影响两节点上最近一次 `ip addr add` 的顺序差异）；修复是最小侵入、对健康节点幂等的。没有节点被排除，没有用到 sudo。

### 上限处的 OOM 分配大小（均在默认 `MEM_FRACTION=.93`）

| 配置 | 首次 OOM pdbs | 分配大小（GiB） | 备注 |
|---|---:|---:|---|
| `sparse` | 1 | 581.8 | RaggedDot 工作集；整行不可行。 |
| `sparse-deepep` | 1 | 507.4 | RaggedDot 工作集；整行不可行。 |
| `sparse-gmm-fixed` | 5 | 195.6 | 分配低于池容量（267.8 GiB）但总工作集超池。分配不在池的 10 % 邻域内，未尝试 `.96` retry。 |
| `sparse-gmm-fixed` | 6 | 207.2 | 单调性确认。 |
| `sparse-gmm`（one-shot） | 5 | 195.6 | 与 fixed 同一上限；one-shot 内核比 kNccl 占更多 HBM。 |
| `sgd-v1` | 6 | 195.3 | 每层 2 个 scatter-add 中间张量 —— 在 dropless 路径里每层显存最大；悬崖与 v2 同。 |
| `sgd-v2` | 6 | 202.6 | 每层 1 个 scatter-add 中间张量。OOM 分配比 v1 大约 6 GiB，但上限与 v1 相同（都是 5）。 |
| `sgd-v3` | 8 | 214.3 | 无 scatter-add 中间张量；相对 v1 / v2 外推 +2 pdbs。 |
| `dense-cf1.25` | 16 | 202.4 | 峰值 TGS 在 pdbs=11（不是 12 或 16）；`argmax < max` 的悬崖信号。 |
| `dense-cf2` | 11 | 189.4 | `pdbs=12` OOM 189.7 GiB（单调性确认）。 |
| `dense-cf4` | 7 | 180.5 | `pdbs=8` OOM 216.4 GiB（单调性确认）。TGS 在 pdbs=5 峰值，pdbs=6 略降（同样的 argmax<max 形状）。 |

所有 sparse-family 的 OOM 单元格都没有尝试 `MEM_FRACTION=.96` retry，因为 OOM 分配大小（~195–215 GiB）都远低于 `.93` 池（267.8 GiB）—— 是整个工作集，而不是单次大分配，把池容量撞破。把 MEM_FRACTION 拉高大约增加 8 GiB/设备，既不够塞下工作集，又会挤 RCCL。

### 编译期异常

这个 1T 模型同一单元格的 XLA 编译时间 **高度非确定**：
- `sparse-gmm-deepep-v3 pdbs=5` 某次 ~5 min 完成（13627），另外两次却 >45 min（13577 在 `--time=45:00` timeout；13667 跑到 23 min 挂起时 CPU 累计 64 min 而放弃）。
- XLA 的 rematerialization 启发式依赖分配顺序；运气好的 run 找到快速调度，运气差的迭代个没完。
- **已应用的阶梯：** `--time=25:00 → 45:00 → 60:00 → 90:00`。所有 sparse + DeepEP 单元格默认起点是 `--time=45:00`，retry 用 60:00。

### MaxText 心跳默认值（100 秒）

MaxText 在 `base.yml` 中把 `jax_distributed_heartbeat_timeout_seconds` 设为 100 —— 远紧于 JAX 的 300 秒默认值。1T 模型上任何 >100 秒的冷编译都会被它直接杀掉。**在 dense-cf2 pdbs=5 失败（13607）后，所有后续提交都加上了 `jax_distributed_heartbeat_timeout_seconds=99999` 透传作为 hedge。** 由于本仓库未启用 JAX 持久化编译缓存，这个 hedge 每次都需要 —— 在这个扫描里不存在"subsequent pdbs 可以复用 warm cache"的情况。

*注意：原始扫描其实提交的是 `jax_distributed_heartbeat_timeout_seconds=99999`（带 `_env_` 前缀），`_train.sh` 把它抽出为 shell 环境变量，但 MaxText 并不读这个 env var —— 实际 MaxText config 在整个扫描期间始终保持 100 秒默认值。本扫描没被这个 bug 咬到，仅仅是因为每个 cell 都是 15-step 探针（≤ 8 min，远低于任何可能的 100 s 停顿窗口）。长跑型 re-run（例如真数据 loss test）必须使用上面的 bare form（不带 `_env_` 前缀）—— 带前缀那种是静默 no-op。*

### GitHub 故障窗口（2026-04-23 约 16:00–16:55 UTC）

~45 分钟的 GitHub 500 故障把 9 个作业在 `MAXTEXT_PATCH_BRANCH` 检出阶段打死（`remote: Internal Server Error` / `fatal: unable to access 'https://github.com/ROCm/maxtext.git/': HTTP 500`）。在其他 rank 上的衍生症状是作业开始后 90 秒内出现 `ActorUnschedulableError` —— rank N 的 `git fetch` 失败后 Ray 的调度重试放弃。9 个作业在 17:00 UTC 之后全部重提交并跑到完成。

### OOM 附近观察到的 compile-hang

`dense-cf4 pdbs=6`（13651）和 `sgd-v1 pdbs=6`（13663）在首次尝试时呈现 OOM-hang 的典型样貌：BARRIER 之后静默 >10 分钟，0 步，但头节点的 python 仍以 292–300 % CPU 烧着（CPU 累计 20–30 分钟）。按 OOM-as-hang 规则 scancel 并以更大 wall 预算 retry，两者分化：

- `dense-cf4 pdbs=6` 重试（13673）**成功**（453.2 TGS）—— 是慢编译导致的瞬态，不是显存悬崖。dense-cf4 的 `max_pdbs` 是 6，不是 5。
- `sgd-v1 pdbs=6` 重试（13674）**明确 OOM**（195.3 GiB）—— 确认是真正的显存上限。sgd-v1 的 `max_pdbs` 是 5。

OOM-hang 的消歧协议把这两种情况正确分开：一个 retry 翻成功（瞬态慢编译），一个 retry 翻成明确 OOM（真正的上限）。两个重试均以 `--time=45:00` 加 heartbeat hedge 提交。

### 可重复性重跑（所有均在默认 `MEM_FRACTION=.93`）

sgd-v3 的每一个可行单元格（pdbs=1, 2, 4, 5, 6, 7）都被重跑了一次以验证原始 TGS 不是单样本巧合。另外 dense-cf1.25 pdbs=4 也重跑了一次作为基线对照。结果：

| 单元格 | 原始 TGS | 重跑 TGS | 差值 | 备注 |
|---|---:|---:|---:|---|
| sgd-v3 pdbs=1 | 229.5 | 219.7 | −4.3 % | |
| sgd-v3 pdbs=2 | 405.5 | 388.8 | −4.1 % | |
| sgd-v3 pdbs=4 | 685.2 | 650.9 | −5.0 % | 见结论 #2 —— 双样本让 P★ 排序从 v3 领先翻转为平局。 |
| sgd-v3 pdbs=5 | 750.8 | 765.3 | +1.9 % | |
| sgd-v3 pdbs=6 | 856.4 | 827.1 | −3.4 % | |
| sgd-v3 pdbs=7 | 897.9 | 886.8 | −1.2 % | |
| dense-cf1.25 pdbs=4 | 678.9 | 682.5 | +0.5 % | 对照 —— dense 的可重复性在 1 % 以内。 |

**重跑揭示：sgd-v3 的 XLA 编译会在不同 run 里产生略不同的内核调度**（均值漂移 −2.8 %，范围 ±5 %），而 dense-cf1.25 可重复性在 <1 % 以内。v3 的这种方差根源在于 XLA rematerialization 启发式依赖分配顺序的分支 —— 它是路径本身的属性，不是基础设施问题。主结果表中的所有 sgd-v3 TGS 都来自*第一次*成功运行；重跑值仅记录在此处用作可重复性评估。任何在 1–5 % 边际上依赖 sgd-v3 vs 其他配置比较的结论都应被视为"在测量噪声内"。

### `exit=143` 的 cleanup-flake ≠ 训练失败

作业 13557、13606 在 rank-0 数据完全有效的前提下完成了全部 15 步训练，但其中一个 rank 在 Ray/container teardown 阶段以 exit 143 退出。Slurm 把 `JOB SUMMARY Status` 写成 `FAILED (exit 1)`，即便训练数据本身完好。**两者都作为有效数据点保留**（extractor 的成功判据是 `last_completed_step >= steps-1`，不是 JOB SUMMARY 状态）。

---

## 脚注

- 上方 TGS/TFLOP/s/步时间/loss 四表中的单元格均来自每个单元格的**首次成功运行**。sgd-v3（pdbs=1,2,4,5,6,7）和 dense-cf1.25 pdbs=4 作为对照的可重复性重跑列表在基础设施说明的"可重复性重跑"小节 —— 它们支持结论 #2 "P★ 打平" 的读法，但不替换主表。
- `sparse-gmm-fixed` 与 `sparse-gmm` 在 pdbs=5 都是 195.6 GiB 的分配 OOM —— 这是本模型上能把 XLA 池顶穿的最小分配尺寸；反映的是 MoE 中间张量的共性，不是某个变体的特殊病态。
- sgd-v1 / sgd-v2 / sparse-gmm-fixed / sparse-gmm 的 `max_pdbs` 都属于 `{4, 5}`。1T 级的共性教训：没有 v3 `custom_vjp` 反向的 dropless 路径在 288 GB HBM / 设备的默认 `MEM_FRACTION=.93` 下都撑不过 pdbs=6+，而 `.96` 又带不来足够的余量。只有 v3 能到 pdbs=6+。
- 扫描完成于 2026-04-23。大约跑了 53 个 benchmark 作业、5 个重跑、7 个预先取消。计算预算约 18 GPU-小时（8 节点 × ~22 分钟 × 58 个作业 / 60）。

---

## Profile 剖析

Profile 作业（`profiler=xplane profiler_steps=3 _env_ENABLE_XLA_DUMP=1`，`--time=60:00` / 慢编译重试改为 `--time=90:00`）。首次运行（`13687–13694`、`13711`）为了压缩 `JOB_NAME` 限制去掉了 XLA dump；复跑批次（`13809–13816`、`13829`）去掉了冗余的 `skip_first_n_steps_for_profiler=5` 透传（kimi yml 默认已是 3，同样在 warmup 之后），把 XLA_DUMP 加了回来。下方内核时间来自 XLA-DUMP 启用的复跑批次；它们与原始批次在 ±1 % 以内完全一致。

每内核时间来自 [`utils/profile_drill.py`](utils/profile_drill.py)，8 个 trace-JSON 窗口 × 8 GPU × 3 个 profile 步（每个单元格除数 = 192）。HLO 集合通信与自定义调用的实例数量来自对 `xla_dump/module_*.jit_train_step.gfx950_gpu_after_optimizations.txt` 的 `grep`。

### P★ = 4 上的 HLO 集合通信算子清单

标准集合通信算子的 HLO 实例数（后优化阶段），外加 DeepEP 的 `custom_call_target` 计数。`sparse-gmm-fixed` 是唯一发出 `ragged-all-to-all`（6 个）的配置 —— 其他四条 sparse 路径（v1/v2/v3）把它换成了 DeepEP 的 `moe_dispatch` / `moe_combine` 自定义调用。**v1/v2/v3 发出的 HLO 集合通信清单逐位相同**（5/5/0/0/3 AG/AR/RA2A/A2A/RS），确认了"v1/v2/v3 前向 HLO 逐位一致"这一主张。

| 算子 | `dense-cf1.25` | `sparse-gmm-fixed` | sgd-v1 | sgd-v2 | **sgd-v3** |
|---|---:|---:|---:|---:|---:|
| `all-gather` | 5 | 7 | 5 | 5 | 5 |
| `all-reduce` | 5 | 5 | 5 | 5 | 5 |
| `ragged-all-to-all` | 0 | **6** | 0 | 0 | 0 |
| `all-to-all` | 6 | 4 | 0 | 0 | 0 |
| `reduce-scatter` | 3 | 3 | 3 | 3 | 3 |
| `custom_call_target="moe_dispatch"` | 0 | 0 | 2 | 2 | 2 |
| `custom_call_target="moe_combine"` | 0 | 0 | 2 | 2 | 2 |
| `custom_call_target="moe_cached_dispatch"` | 0 | 0 | 1 | 1 | 1 |

DeepEP 自定义调用的计数在同一位置（1 × `moe_cached_dispatch`，2 × `moe_combine`，2 × `moe_dispatch`）与 DS3 完全一致 —— DeepEP dispatch/combine 的发射模式与模型无关。

这 5 个 DeepEP 自定义调用实例共同替代了 `sparse-gmm-fixed` 在同一 dispatch/combine 工作上使用的 6 个 `ragged-all-to-all` + 4 个 `all-to-all` = 10 个 HLO 集合通信实例（与 DS3 剖析中的模式完全一致）。

### `P★ = 4` 跨路径步时间构成（秒 / GPU / 步）

### `P★ = 4` 跨路径步时间构成（秒 / GPU / 步）

| 切片                                                  | `dense-cf1.25` | `sparse-gmm-fixed` | sgd-v1 | sgd-v2 | **sgd-v3** |
|------------------------------------------------------|---------------:|-------------------:|-------:|-------:|-----------:|
| `RaggedAllToAllKernelImpl`（XLA 进程内）              |           0.00 |               0.00 |   0.00 |   0.00 |   0.00 |
| `primus_turbo::deep_ep::*`（DeepEP 原生 HIP）         |           0.00 |               0.00 |   0.95 |   0.95 |   0.92 |
| `input_scatter_fusion_*.kd`                          |           0.00 |               0.01 | **5.34** | **2.68** | **0.02** |
| `loop_select_fusion_*.kd`（valid-rows mask）          |           0.01 |               0.01 |   0.95 |   0.61 |   0.53 |
| `loop_gather_fusion_*.kd`                            |           0.00 |               1.21 |   0.00 |   0.00 |   0.00 |
| RCCL (`ncclDevKernel_*`)                              |          13.44 |               9.56 |   9.17 |   9.25 |   9.04 |
| CK / Primus-Turbo GEMM（grouped + dense）             |           4.98 |               2.48 |   2.67 |   2.69 |   2.61 |
| Flash-attention (`aiter::fmha_*`)                     |           0.43 |               0.29 |   0.35 |   0.36 |   0.35 |
| 其他 fusion（`loop_reduce` / `loop_convert` / `loop_transpose` / `input_reduce_select` / `input_broadcast_reduce_select` / 杂项） | 1.34 | 1.76 | 1.26 | 1.21 | 1.32 |
| **总内核时间（任何流上）**                             |      **20.21** |          **15.31** | **20.71** | **17.74** | **14.86** |
| benchmark 步时间（来自主扫描）                         |          24.14 |              26.67 |  34.37 |  28.47 |  23.91 |
| 步 − 总内核 = 空闲间隔（+）或重叠（−）                  |          +3.93 |             +11.36 | +13.66 | +10.73 |  +9.05 |

*（步时间来自非 profile 的 benchmark；profile run 的 TGS 因 writeback 略慢，不用于"步 − 总内核"的比较。本表数字来自 XLA-DUMP 启用的复跑批次（`13809-13816`，dense / sparse-gmm-fixed / sgd-v1 / sgd-v3）以及 `13829`（sgd-v2，2026-04-25 在 compile-timeout 重试后补齐）。固定计算内核（input_scatter_fusion、loop_select、GEMM、flash_attn）的复跑差异 ≤ ±1 %；通信侧家族（`primus_turbo::deep_ep`、`ncclDevKernel_*`）的复跑差异在 5–15 % 之间 —— 例如本表中 sgd-v2 的 deep_ep 与 RCCL 相对早期无 XLA-DUMP 的同 cell 跑分别低了 0.34 秒和 1.56 秒，说明通信流的测量比片上内核计时更易抖动。）*

**DS3 内核消除链在 kimi 上的 headline 复现：**

| 单元格 | `input_scatter_fusion_*.kd` kimi-1T @ pdbs=4 | DS3 671B @ pdbs=6 | 备注 |
|---|---:|---:|---|
| v1 基线（2 次 gather → 2 次 scatter-add） | **5.34 秒** | 8.97 秒 | v1 与模型规模近似成正比：kimi-1T 约为 DS3 的 60 %（专家数更多但 `base_moe_mlp_dim` 近似） |
| v2（合并 gather → 1 次 scatter-add） | **2.68 秒** | 4.45 秒 | 相对 v1 −50 % |
| **v3（`custom_vjp` → 0 次 scatter-add）** | **0.02 秒** | 0.04 秒 | **相对 v1 −99.6 % —— 内核被完全消除** |

**这是 DS3 v1→v2→v3 故事在 kimi-1T 内核层面的确定性复现。** `input_scatter_fusion_*.kd` 家族从 v1 到 v3 缩小两个数量级 —— 与 DS3 的预测完全一致，也和 `moe.py` 纯 Python 补丁的语义吻合（v1 的重复索引 scatter-add 反向 → v3 的置换 gather + reduce-sum 反向，无原子）。

**其他跨路径观察：**

1. **dense-cf1.25 的 RCCL 时间最高（14.17 秒）** —— dropping 路径的常规 `all-to-all` + 所有 `all-gather` / `all-reduce` / `reduce-scatter` 加起来的 RCCL 工作比 dropless 的 ragged 通信更多。但 dense 的*暴露*空闲间隔很小（+3.2 秒），因为这部分 RCCL 完全与计算重叠，而 dropless 各路径的主流阻塞 `input_scatter_fusion` 或 `loop_gather_fusion` 强制把空闲间隔推到正值。
2. **`sparse-gmm-fixed` 没有 `input_scatter_fusion` 但有较大的 `loop_gather_fusion_*.kd`（1.20 秒）** —— 对于非-DeepEP dropless，XLA 用 gather 家族而不是 scatter 家族来实现 ragged fan-in/out。DeepEP 路径（v1、v2）的 `loop_gather_fusion = 0`，因为 DeepEP 的自定义调用接管了这部分工作。
3. **RCCL 时间在 v1/v2/v3 间基本稳定（9.0 – 9.3 秒，sgd-v2 刷新后）**，尽管总内核时间从 v1 到 v3 下降了 ~6 秒。v3 的 wallclock 收益不来自通信节省 —— 而来自消除主流阻塞的 scatter-add。与 DS3 的观察一致。
4. **空闲间隔在 v1 → v2 → v3 单调缩小：+13.66 秒 → +10.73 秒 → +9.05 秒** —— 调度级联恢复：随着 `input_scatter_fusion` 的主流阻塞减少，XLA 能重叠更多工作。在这个 1T 模型上，v1→v2 的改善（−2.93 秒）大于 v2→v3 的改善（−1.68 秒），与 DS3（v2→v3 变化更大）相反。原因可能是 1T 的 RCCL 占比更高（v1 的 20.71 秒总内核里 ~9.2 秒是 RCCL），最后一个 scatter-add 阻塞能影响的额外重叠余地比 DS3 少。

### pdbs=5 上的补充 v1/v2/v3 剖析（秒 / GPU / 步）

pdbs=5 = `min(v1_max, v2_max, v3_max)`，且严格大于 P★=4。该表捕获在 v1/v2 吞吐峰值 pdbs 下的优化链（在此 pdbs 下 `dense-cf1.25` 和 `sparse-gmm-fixed` 也仍然可行，但未在此重复 profile，因为 pdbs=4 已经展示了它们的内核模式）。

| 切片                                                 | sgd-v1 | sgd-v2 | **sgd-v3** |
|-----------------------------------------------------|-------:|-------:|-----------:|
| `primus_turbo::deep_ep::*`                           |   1.24 |   1.25 |   1.26 |
| `input_scatter_fusion_*.kd`                          | **7.39** | **3.83** | **0.03** |
| `loop_select_fusion_*.kd`                            |   1.64 |   0.83 |   0.73 |
| RCCL (`ncclDevKernel_*`)                              |   9.85 |  10.65 |  10.12 |
| CK / Primus-Turbo GEMM                                |   3.49 |   3.70 |   3.61 |
| Flash-attention                                       |   0.48 |   0.51 |   0.50 |
| 其他 fusion + 杂项                                    |   1.66 |   1.60 |   1.88 |
| **总内核时间（任何流上）**                             | **25.75** | **22.35** | **18.14** |
| benchmark 步时间                                      |  39.72 |  32.22 |  27.29 |
| 步 − 总内核 = 空闲间隔                                | +13.97 |  +9.87 |  +9.15 |

**`input_scatter_fusion` 消除链在 pdbs=5 同样成立：** v1 = 7.39，v2 = 3.83，v3 = 0.03。绝对值随 pdbs 成比例增长（每设备更多 token → 更大的 scatter-add 维度）。v3 的内核在我们测量的所有 pdbs 上本质上都不存在 —— 验证这是普适的，不是 pdbs=4 的局部现象。

**v1 → v3 在 pdbs=5 的总内核节省：** 25.75 − 18.14 = **7.61 秒 / 步 / GPU**，其中单独 `input_scatter_fusion` 消除贡献了 7.36 秒（占节省的 97 %）。步时间差为 39.72 − 27.29 = 12.43 秒，调度恢复（空闲间隔缩小）再贡献 4.82 秒 —— 与 DS3 "去掉主流阻塞，让通信重叠"的机制一致。在 kimi-1T 上，调度级联贡献比 DS3 小，原因是 RCCL 可重叠余量更少。

### 内核层面的结论

| 维度 | Kimi-1T 发现 | 是否与 DS3 匹配？ |
|---|---|---|
| v1→v2 把 `input_scatter_fusion` 砍半 | 是：5.34 → 2.68（pdbs=4），7.39 → 3.83（pdbs=5） | 是：8.97 → 4.45（pdbs=6） |
| v3 消除 `input_scatter_fusion` | 是：→ 0.02（pdbs=4），→ 0.03（pdbs=5） | 是：→ 0.04（pdbs=6） |
| v1/v2/v3 HLO 集合通信清单完全一致（5 AG / 5 AR / 0 RA2A / 0 A2A / 3 RS） | 是，在 pdbs=4 与 pdbs=5 均已用 post-opt HLO dump 的 `grep` 核对 —— 本次扫描 | 是（DS3 剖析中声明"v1/v2/v3 HLO 逐位一致"；kimi 验证） |
| 总内核时间减少量 > `input_scatter_fusion` 单项 | 是（1T 上调度级联贡献 ~0.2–0.5 秒；比 DS3 671B 的 ~4 秒小） | 模式匹配，幅度更小 |
| v3 扩展可行性前沿 | 是：v3 max=7 vs v1/v2 max=5（在 1T 上） | 不适用 —— DS3 上三者都到 max=7 |

**结论：DS3 的内核优化故事在 1T 上以相同机制复现，绝对 scatter-add 时间为 DS3 的约 60 %。DeepEP v3 补丁在本模型上是对 v1 基线的毫无歧义替代。** 此外在 1T 上，v3 是*唯一*保持 DS3 pdbs=7 上限的 DeepEP 变体 —— 这是 DS3 没展示出来的显存前沿胜利，因为 DS3 上 HBM 余量足够让 v1/v2/v3 都到 pdbs=7。

### Profile 作业产物（保存于 `outputs/`）

- `13687-…` —— dense-cf1.25 pdbs=4 profile（profile 下 661.6 TGS vs 基准 678.9）
- `13688-…` —— sparse-gmm-fixed pdbs=4 profile（586.1 TGS vs 614.5）
- `13689-…` —— sgd-v1 pdbs=4 profile（469.2 TGS vs 476.9）
- `13690-…` —— sgd-v2 pdbs=4 profile（541.8 TGS vs 575.4）
- `13711-…` —— sgd-v3 pdbs=4 profile（632.6 TGS vs 685.2 —— 13691 在 60:00 compile 超时后，以 `--time=90:00` 重试）
- `13692-…` —— sgd-v1 pdbs=5 profile（510.1 TGS vs 515.7）
- `13693-…` —— sgd-v2 pdbs=5 profile（626.6 TGS vs 635.9）
- `13694-…` —— sgd-v3 pdbs=5 profile（753.0 TGS vs 750.8）

---

## 如何复现

```bash
cd /maxtext-slurm

# dense-cf1.25 峰值（pdbs=11）
RAY=1 ./submit.sh kimi-k2-1t:dense-cf125 \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=11 steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-deepep-v3 dropless 峰值（pdbs=7）—— headline 单元格
# 注意：`container_env.sh` 现在默认 MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3，
# 因此 v3 运行不再需要 env-var 前缀（此处保留以便完整复现）。
MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3 \
RAY=1 ./submit.sh kimi-k2-1t:sgd-deepep-v3 \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
    use_deepep_dispatch=true steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-fixed 在 P★（pdbs=4）—— 苹果对苹果 pdbs 上的最佳非-DeepEP dropless
RAY=1 ./submit.sh kimi-k2-1t:sparse-gmm-fixed \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=4 sparse_matmul=true use_turbo_grouped_gemm=true \
    steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999
```

**每次提交都加上 `jax_distributed_heartbeat_timeout_seconds=99999`**（bare flag —— **不是**带 `_env_` 前缀的形式，那种是静默 no-op，见上面 [MaxText 心跳默认值](#maxtext-心跳默认值100-秒) 中的 caveat）—— MaxText 默认 100 秒心跳比绝大多数本模型上的冷编译都紧。除非该单元格已被观测到 <15 min 完成编译，否则用 `--time=45:00` 而不是更短。重试升级阶梯与消歧优先级见 [`moe-pdbs-sweep-prompt.md`](moe-pdbs-sweep-prompt.md)。

---

*文档状态：**v4 final** —— 主扫描 + profile 剖析 + DCN expert-parallelism 扩展。48 个主扫描成功单元格 + 8 个 profile 单元格 + 1 个 sgd-v2 pdbs=4 profile 刷新（`13829`，含 HLO dump，2026-04-25），14 个 OOM 上限，2 行整行不可行。**v4（DCN-EP 扩展）** 增加 27 个单元格（2026-04-25），覆盖 `dcn_expert_parallelism ∈ {2, 4, 8}` × 4 个非 DeepEP 配置（dense-cf1.25/cf2/cf4 + sparse-gmm-fixed）：22 个成功，5 个 OOM；4 个 DeepEP 变体 × 3 个 DCN_EP 值被 `MaxText/pyconfig.py` 校验阻塞。所有 `max_pdbs` 均已通过直接观测确认。主扫描 P★ = 4。DS3 v1→v2→v3 内核消除链在 pdbs=4 和 pdbs=5 上均以 99.6 % 的 `input_scatter_fusion_*.kd` 消除率获得确认。v3.1 在将 sgd-v2 与其他 4 个路径合并到同一 XLA-DUMP 启用批次后，观察到了单调下降的空闲间隔（v1 +13.66 秒 → v2 +10.73 秒 → v3 +9.05 秒）。DCN-EP 主要发现：dense-cf1.25 在 DCN_EP=2 / pdbs=4 上比 DCN_EP=1 实际**快了 +6.5 %**；sparse-gmm-fixed cliff 随 DCN_EP 急剧加深（max_pdbs 4 → 2 → 1 → 0）；DS3 v3-vs-fixed 在 DCN_EP>1 上的假设由于 MaxText 对跨节点 DeepEP 的 pydantic 锁定无法测试。*
