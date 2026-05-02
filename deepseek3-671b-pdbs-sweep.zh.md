# DeepSeek-V3 671B —— 全面的 pdbs 扫描实验

- **日期：** 2026-04-17（首次扫描）；2026-04-18（扫描后扩展 —— `sparse-gmm-fixed` 列及关键结论 #8）；2026-04-22（扫描后扩展 —— `sparse-gmm-deepep-v2` / `sparse-gmm-deepep-v3` 列及关键结论 #9）；2026-04-25（DCN expert-parallelism 扩展 `dcn_expert_parallelism ∈ {2, 4, 8}` 覆盖 4 个非 DeepEP 配置，详见主矩阵之后的 ["DCN expert-parallelism 扩展"](#dcn-expert-parallelism-扩展dcn_expert_parallelism--1) 一节）
- **模型：** `deepseek3-671b` (MaxText)
- **硬件：** 8 节点 × 8× AMD MI355 (288 GB HBM / 设备), Pensando AINIC 互联
- **镜像：** `/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar` (包含 [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) GMM + DeepEP)
- **补丁分支：**
  - [yihuang/moe-turbo-gmm-and-deepep](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep) @ `ad693da2`（基线 —— `sparse-gmm-deepep` 列）
  - [yihuang/moe-turbo-gmm-and-deepep-v2](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v2) @ `627168f8`（在基线上加 1 个提交 —— `sparse-gmm-deepep-v2` 列）
  - [yihuang/moe-turbo-gmm-and-deepep-v3](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9`（在基线上加 2 个提交 —— `sparse-gmm-deepep-v3` 列）
- **基础配置：** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)
- **数据集：** `dataset_type=synthetic`（扫描时为 gpu.yml 默认值；该 yml 后续已切换为 `grain`/c4 —— 复现时需在 CLI 上覆盖 `dataset_type=synthetic`）。
- **峰值 BF16：** ≈ 2500 TFLOP/s/设备 → MFU ≈ TFLOP/25

## 背景

1-GPU/proc 启动器（[maxtext-slurm#111](https://github.com/AMD-AGI/maxtext-slurm/pull/111)）最初是为了启用 [**mori-EP**](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md)（AMD 的高性能 MoE dispatch/combine 内核库，思路与 DeepEP 类似，要求每个 GPU 一个 JAX 进程）而添加的，但在本模型上最终未能跑通。在压测该启动器的过程中，我们意外发现 `sparse-gmm 1-GPU/proc` 比 `sparse-gmm 1-node/proc` **快约 3×** —— 一个与 mori-EP 毫无关系的差距。Profiling（pdbs=6，作业 **12895 / 12916 / 12897**）将成因追溯到 XLA 的 `ragged-all-to-all` thunk：当所有 EP rank 共享同一进程（1-node/proc）时，它会选用朴素的进程内内核 `RaggedAllToAllKernelImpl<8l>`；而当它们跨进程（1-GPU/proc）时，会选用快得多的 `kNccl` 路径。一旦机制明确，我们就发现更快的这条路径也可以在 1-node/proc 上通过一个 XLA 标志强制启用 —— 切换启动器的做法也就不再必要。该修复已随 [maxtext-slurm#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112) 合入；下方结果矩阵中的 `sparse-gmm-fixed (1-node)` 列正是其效果的度量。详细的分析在 [三种 sparse 变体的 TGS 为何不同？](#三种-sparse-变体的-tgs-为何不同pdbs6-的-profile-深入剖析) 一节。

> **默认设置变更（2026-04-18，[maxtext-slurm#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112)）：** `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` 现已成为 `train_env.sh` 的默认值（初次扫描期间为可选开启）。在 1-node/proc 上运行 `sparse-gmm` 而不带任何额外标志，现在得到的就是下方 `sparse-gmm-fixed (1-node)` 列的结果 —— `sparse-gmm (1-node / 1-GPU)` 列展示的是 XLA 进程内 one-shot 内核仍然生效时的历史数据，供对比参考。该变更在 dense / deepep / 1-GPU 路径上已验证为空操作（见结论 #8）；如需恢复 XLA 的 one-shot 内核（仅用于调试），设置 `_env_ENABLE_RAGGED_ONESHOT_KERNEL=1`。

---

## 受测配置

| 标签                   | 透传参数                                                                                     |
|------------------------|---------------------------------------------------------------------------------------------|
| `dense-cf1.25`         | *(默认)* — `sparse_matmul=false`, `capacity_factor=1.25`                                   |
| `dense-cf2`            | `capacity_factor=2.0`                                                                       |
| `dense-cf4`            | `capacity_factor=4.0`                                                                       |
| `sparse`               | `sparse_matmul=true shardy=true`                                                            |
| `sparse-gmm`           | `sparse_matmul=true use_turbo_grouped_gemm=true _env_ENABLE_RAGGED_ONESHOT_KERNEL=1`        |
| `sparse-deepep`        | `sparse_matmul=true use_deepep_dispatch=true shardy=true`                                   |
| `sparse-gmm-deepep`    | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`（补丁分支 `yihuang/moe-turbo-gmm-and-deepep`）|
| `sparse-gmm-deepep-v2` | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`（补丁分支 `yihuang/moe-turbo-gmm-and-deepep-v2`）|
| `sparse-gmm-deepep-v3` | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`（补丁分支 `yihuang/moe-turbo-gmm-and-deepep-v3`）|
| `sparse-gmm-fixed`     | `sparse_matmul=true use_turbo_grouped_gemm=true`                                            |

**`sparse-gmm-deepep`、`-v2`、`-v3` 的区别：** 这三行使用相同的镜像 / 启动器 / 透传参数 —— 唯一的差异是打补丁的 MaxText 分支，且分支之间唯一不同的文件是 `src/MaxText/layers/moe.py`。**v2**（提交 `627168f8`）将 DeepEP fan-out 的 gather 与排序置换合成为单次 gather（使反向中的 dispatch-side scatter-add 次数减半）。**v3**（提交 `f59be3c9`）进一步用一个 `jax.custom_vjp` 替换由此产生的重复索引 scatter-add 反向：它用 `argsort` 反转置换 + reduce-sum 折叠 top-K 重复 —— 无原子操作。三者的前向输出逐位一致（每一步的 loss 均在 bf16 LSB 噪声内吻合）；两处改动一起把 dispatch 反向中占主导的 `input_scatter_fusion_*.kd` 内核彻底消掉。

**为什么只有 `sparse` 和 `sparse-deepep` 需要携带 `shardy=true`：** `sparse_matmul=true` 在未启用 `use_turbo_grouped_gemm=true` 时会回退到 `jax.lax.ragged_dot` 进行专家矩阵乘法，其分片传播需要 `shardy=true`（Shardy 框架，XLA 对 GSPMD 的继任者）—— 这正是这两行需要额外携带该标志的原因。`sparse-gmm` 和 `sparse-gmm-deepep` 无需此标志，因为 Primus-Turbo GMM 自定义调用自带分片规范，绕过了分片传播过程。`use_deepep_dispatch=true` 将 `ragged_all_to_all` 替换为 Primus-Turbo 的 DeepEP 节点内自定义调用，本身**并不**引入 shardy 要求 —— 该要求是 `ragged_dot` 矩阵乘法路径的属性，与 dispatch 路径正交。

**`_env_ENABLE_RAGGED_ONESHOT_KERNEL` 的含义：** 直接映射到 XLA 标志 `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel`。取 **`=1`**（上表中的 `sparse-gmm` 行；[#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112) 之前的 XLA 原始默认值）时，1-node/proc 使用进程内 one-shot 内核 `stream_executor::gpu::RaggedAllToAllKernelImpl<8l>` 处理 `ragged-all-to-all`，产出 `sparse-gmm (1-node / 1-GPU)` 列的数字。取 **`=0`**（上表中的 `sparse-gmm-fixed` 行；现为 `train_env.sh` 的默认值）时，XLA 的 ragged thunk 回退到 `kNccl` 路径 —— 即 1-GPU/proc 运行时自动使用的同一种 RCCL 降级 —— 产出 `sparse-gmm-fixed (1-node)` 列。`train_env.sh` 通过 append 的方式设置此环境变量，保留了镜像默认 `XLA_FLAGS` 中的其他调优项（`xla_gpu_enable_cublaslt`、`xla_gpu_enable_latency_hiding_scheduler` 等）。由于 `0` 现在已是 repo 默认，普通的 `sparse-gmm` 提交即可复现 `sparse-gmm-fixed` 列；只有复现历史 pre-flag 列时才需要显式 `=1` 覆盖。

## 启动器模式

| 标签            | 透传参数                                | 说明                                                                 |
|-----------------|-----------------------------------------|----------------------------------------------------------------------|
| `1-node/proc`   | *(默认)*                                | 每个节点一个 Python 进程；JAX 可见 8 个本地设备                      |
| `1-GPU/proc`    | `_env_ONE_GPU_PER_PROCESS=true`         | 每个节点 8 个进程；每个 JAX 进程恰好拥有一个 GPU                     |

> **这是进程粒度模式，不是作业规模。** 下方结果矩阵中的每个单元格都在**完整的 8 节点 × 64 GPU** 拓扑上测量，与启动器无关；唯一的区别是每个节点运行 **一个** JAX 进程（`1-node/proc`）还是 **八个**（`1-GPU/proc`）。本文后续缩写形式的列标题（如 `sparse-gmm (1-node / 1-GPU)` 和 `sparse-gmm-fixed (1-node)`）指的是这些启动器模式，**而不是** 1 节点或 1 GPU 的作业规模。

---

## 可行性总结

**可行（1-node / 1-GPU）：**
- `dense-cf1.25` ✓ / ✓
- `dense-cf2` ✓ / ✓
- `dense-cf4` ✓ / ✓
- `sparse-gmm` ✓ / ✓
- `sparse-gmm-deepep` ✓ / ✗ （AssertionError —— 见下文）
- `sparse-gmm-deepep-v2` ✓ / ✗ （与基线相同的 AssertionError）
- `sparse-gmm-deepep-v3` ✓ / ✗ （与基线相同的 AssertionError）
- `sparse-gmm-fixed` ✓ / — （仅限 1-node；在 1-GPU/proc 上该标志多余，因为该启动器本已自动走 `kNccl` 路径）

**在 pdbs=1 时不可行（该类别被跳过）：**

| 配置                | 1-node                               | 1-GPU                                 |
|---------------------|--------------------------------------|---------------------------------------|
| `sparse`            | OOM 444 GiB (RaggedDot)              | OOM 444 GiB (RaggedDot)               |
| `sparse-deepep`     | OOM 375 GiB (RaggedDot)              | `AssertionError: EP ranks=1`          |
| `sparse-gmm-deepep` | *(可行)*                             | 在每个 pdbs 下都报 `AssertionError: EP ranks=1` |

`AssertionError: Unsupported number of EP ranks: 1` 源于 Primus-Turbo 的 DeepEP Python 绑定，它将 `num_ranks = jax.local_device_count()` 写死。在 1-GPU/proc 模式下这始终为 1。这是一个根本性的 API 不兼容问题。

---

## 结果矩阵 —— 每个单元格 `1-node / 1-GPU`

除 loss 外的所有指标均为**训练步 5–14 的均值**（步 0–4 作为预热被丢弃）。Loss 仅报告步 14 的值，因为在合成数据下单步 loss 是一个稳定的数值正确性探针。

图例：`✗` = OOM；`—` = 跳过（该组合在较小 pdbs 就已 OOM）。

### Tokens/s/设备 (TGS)

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 333 / 326                     | 311 / 307                  | 265 / 259                  | 163 / 281                    | 305                       | 271                        | 294                           | **317**                       |
| 2    | 563 / 535                     | 497 / 471                  | 374 / 369                  | 224 / 496                    | 514                       | 415                        | 454                           | **545**                       |
| 4    | 867 / 822                     | 721 / 723                  | 500 / 493                  | 275 / 768                    | 782                       | 569                        | 676                           | **839**                       |
| 5    | 962 / 959                     | 796 / 775                  | 535 / 531                  | 288 / 872                    | 880                       | 614                        | 751                           | **948**                       |
| 6    | 1040 / 1043                   | 835 / 808                  | 543 / 548                  | 298 / 942                    | 949                       | 647                        | 806                           | **1030**                      |
| 7    | 1086 / 1080                   | 884 / 867                  | 560 / 540                  | 302ᵃ / ✗ᵇ                   | 989ᵃ                      | 673                        | 836                           | **1097**                      |
| 8    | 1191 / 1171                   | 918 / 928                  | 566 / 571                  | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 1416 / 1387                   | 968 / 966                  | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

### TFLOP/s/设备

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 83.3 / 81.6                   | 77.9 / 76.8                | 66.2 / 64.9                | 40.9 / 70.5                  | 76.3                      | 67.8                       | 73.6                          | 79.3                          |
| 2    | 140.9 / 134.1                 | 124.4 / 117.9              | 93.6 / 92.4                | 56.1 / 124.3                 | 128.8                     | 103.8                      | 113.8                         | 136.5                         |
| 4    | 217.2 / 205.8                 | 180.7 / 181.2              | 125.2 / 123.5              | 68.9 / 192.5                 | 195.8                     | 142.5                      | 169.2                         | 210.1                         |
| 5    | 241.0 / 240.2                 | 199.4 / 194.1              | 133.9 / 133.1              | 72.1 / 218.5                 | 220.4                     | 153.8                      | 188.1                         | 237.5                         |
| 6    | 260.6 / 261.2                 | 209.2 / 202.5              | 135.9 / 137.2              | 74.6 / 236.0                 | 237.6                     | 162.2                      | 201.9                         | 257.8                         |
| 7    | 272.1 / 270.5                 | 221.3 / 217.0              | 140.1 / 135.3              | 75.7ᵃ / ✗ᵇ                  | 247.7ᵃ                    | 168.5                      | 209.3                         | **274.8**                     |
| 8    | 298.2 / 293.3                 | 230.0 / 232.3              | 141.7 / 143.0              | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 354.7 / 347.3                 | 242.4 / 242.0              | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

**峰值 MFU：** `dense-cf1.25 @ pdbs=16, 1-node` = **14.19 %**（354.7 TFLOP/s/设备，MI355 上 BF16 峰值 ≈ 2500 TFLOP/s/设备）。**峰值 dropless MFU：** `sparse-gmm-deepep-v3 @ pdbs=7, 1-node` = **10.99 %**（274.8 TFLOP/s/设备） —— 无需提升 `MEM_FRACTION`。

### 平均单步时间（秒）

越低越好。取步 5–14 的单步 wall time（训练日志中 `seconds:` 字段）均值。

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 12.3 / 12.6                   | 13.2 / 13.4                | 15.5 / 15.8                | 25.1 / 14.6                  | 13.4                      | 15.1                       | 13.9                          | **12.9**                      |
| 2    | 14.6 / 15.3                   | 16.5 / 17.4                | 21.9 / 22.2                | 36.6 / 16.5                  | 15.9                      | 19.8                       | 18.1                          | **15.0**                      |
| 4    | 18.9 / 20.0                   | 22.7 / 22.7                | 32.8 / 33.2                | 59.6 / 21.3                  | 21.0                      | 28.8                       | 24.3                          | **19.5**                      |
| 5    | 21.3 / 21.4                   | 25.7 / 26.4                | 38.3 / 38.5                | 71.2 / 23.5                  | 23.3                      | 33.4                       | 27.3                          | **21.6**                      |
| 6    | 23.6 / 23.6                   | 29.4 / 30.4                | 45.3 / 44.9                | 82.5 / 26.1                  | 25.9                      | 38.0                       | 30.5                          | **23.9**                      |
| 7    | 26.4 / 26.5                   | 32.4 / 33.1                | 51.2 / 53.1                | 94.8ᵃ / ✗ᵇ                  | 29.0ᵃ                     | 42.6                       | 34.3                          | **26.1**                      |
| 8    | 27.5 / 28.0                   | 35.7 / 35.3                | 57.9 / 57.4                | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 46.3 / 47.3                   | 67.7 / 67.8                | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

### 步 14 的训练 loss

在每个 pdbs 行内，所有配置/启动器彼此 Δ ≤ 0.003 一致 —— 启动器的选择不会扰动数值结果，且 v2/v3 的 `moe.py` 改动在前向上与基线逐位相同（偏差仅在 bf16 LSB 级别）。sparse 仅在 pdbs=1 时比 dense 低约 0.02，因为 sparse 是 dropless 的，而 dense 会丢弃 `capacity_factor` 容纳不下的 token。

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 7.714 / 7.715                 | 7.712 / 7.714              | 7.713 / 7.714              | 7.692 / 7.694                | 7.694                     | 7.694                      | 7.692                         | 7.693                         |
| 2    | 8.594 / 8.594                 | 8.594 / 8.594              | 8.592 / 8.592              | 8.592 / 8.591                | 8.592                     | 8.592                      | 8.592                         | 8.592                         |
| 4    | 9.439 / 9.439                 | 9.439 / 9.439              | 9.437 / 9.437              | 9.438 / 9.438                | 9.437                     | 9.437                      | 9.438                         | 9.438                         |
| 5    | 9.684 / 9.684                 | 9.682 / 9.682              | 9.682 / 9.682              | 9.682 / 9.681                | 9.681                     | 9.680                      | 9.680                         | 9.680                         |
| 6    | 9.884 / 9.884                 | 9.884 / 9.884              | 9.883 / 9.883              | 9.883 / 9.883                | 9.883                     | 9.883                      | 9.883                         | 9.883                         |
| 7    | 10.031 / 10.031               | 10.030 / 10.030            | 10.030 / 10.030            | 10.030ᵃ / ✗ᵇ                | 10.030ᵃ                   | 10.029                     | 10.029                        | 10.029                        |
| 8    | 10.157 / 10.157               | 10.157 / 10.157            | 10.156 / 10.156            | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 10.821 / 10.821               | 10.820 / 10.820            | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

---

## DCN expert-parallelism 扩展（`dcn_expert_parallelism > 1`）

*（2026-04-25 添加。）* 在主矩阵（默认 `dcn_expert_parallelism=1`，即专家并行只在节点内、FSDP 只跨节点）的基础上，沿跨节点 EP 因子化方向再扩展一维。8 节点 × 8 GPU = 64 ranks，并行网格总量不变，但分解方式改变：

| `DCN_EP` | `dcn_fsdp` × `ici_ep × dcn_ep` | 总 EP rank-product | 每主机 EP 扇出 |
|---:|:-:|---:|---|
| **1** *（默认 —— 主矩阵）* | 8 × 8 × 1 | 8 | EP 轴只在节点内 |
| 2 | 4 × 8 × 2 | 16 | 每主机的专家分散到 1 个对端主机 |
| 4 | 2 × 8 × 4 | 32 | 每主机的专家分散到 3 个对端主机 |
| 8 | 1 × 8 × 8 | 64 | 完全 DCN-EP，无跨节点 FSDP |

### 已知限制：DeepEP 变体被锁定在 `DCN_EP=1`

`MaxText/pyconfig.py` 校验 `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1`，否则在到达 XLA compile 之前就以 pydantic `ValidationError("Internode DeepEP is not yet supported in JAX")` 在 ~2 分钟内终止任务。这适用于主矩阵中的**全部 4 个 DeepEP 配置**（`sparse-deepep`、`sparse-gmm-deepep` v1/v2/v3）—— JAX/MaxText 的集成层堵住了 [DeepEP 现状一节"DeepEP 真正能发光的场景"标注](#deepep-现状内核不差xla-集成代价大--以及它真正能发光的场景)中所说的"Inter-node EP（RDMA 支持的 AllToAll）"区域。下面的 DCN-EP 扩展只刻画**非 DeepEP** 区域 —— 3 个 dense-cf 配置和 `sparse-gmm-fixed`。`sparse-gmm`（one-shot）在 DCN_EP > 1 时也跳过：其 OneShot kernel 仅节点内有效，DCN_EP > 1 时退回 kNccl 路径，跟 `sparse-gmm-fixed` 完全重合。

### TGS @ DCN_EP > 1（4 个非 DeepEP 配置，1-node/proc）

DCN_EP=1 列从主矩阵（1-node 列）复制过来便于横向对比。标记 `n/a` 的格子是该 DCN_EP 下没有测量过的 pdbs 值。

#### `dense-cf1.25`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  867 |     799.7 |     649.3 |   537.8 |
|  8 | 1191 |     898.8 |     667.6 |   585.5 |

#### `dense-cf2`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  721 |     577.5 |     443.0 |   377.9 |
|  8 |  918 |     620.3 |     449.0 |   393.4 |

#### `dense-cf4`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  500 |     339.6 |     237.7 |   204.5 |
|  6 |  543 |     337.0 | ✗ OOM-hang | (跳过：pdbs=4 已在上限) |

#### `sparse-gmm-fixed`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  1 |  305 |     364.7 |     338.4 |   333.6 |
|  2 |  514 |     497.3 | ✗ 214.6 GiB |     n/a |
|  4 |  782 | ✗ 224 GiB | ✗ 332 GiB |     n/a |

### pdbs=4 处的跨 DCN_EP TGS 对比（跨配置基准行）

| Config | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 | Δ EP=1→8 |
|---|---:|---:|---:|---:|---:|
| `dense-cf1.25`  | 867 | 799.7 *(−7.7%)*    | 649.3 *(−25.1%)*  | 537.8 *(−38.0%)* | **−38.0%** |
| `dense-cf2`     | 721 | 577.5 *(−19.9%)*   | 443.0 *(−38.6%)*  | 377.9 *(−47.6%)* | **−47.6%** |
| `dense-cf4`     | 500 | 339.6 *(−32.1%)*   | 237.7 *(−52.5%)*  | 204.5 *(−59.1%)* | **−59.1%** |
| `sparse-gmm-fixed` | 782 | ✗ OOM   | ✗ OOM    | ✗ OOM (隐含) | DCN_EP > 1 / pdbs=4 时不可行 |

每个 (config, pdbs) 单元在不同 DCN_EP 下的 loss 都吻合到 ε 内（pdbs=4 下 dense-cf1.25 的 cross-DCN_EP loss 都是 9.439 / 9.439 / 9.439；dense-cf2 / cf4 / sparse-gmm-fixed 同样吻合）—— DCN_EP 因子化对数值无影响，符合预期。

### 跨 DCN_EP 的可行性总结（sparse-gmm-fixed cliff）

| Config | DCN_EP=1 max_pdbs | DCN_EP=2 max_pdbs | DCN_EP=4 max_pdbs | DCN_EP=8 max_pdbs |
|---|---:|---:|---:|---:|
| `dense-cf1.25` | ≥ 16 | ≥ 8 | ≥ 8 | ≥ 8 |
| `dense-cf2`    | ≥ 16 | ≥ 8 | ≥ 8 | ≥ 8 |
| `dense-cf4`    | 7   | ≥ 6 | **4 或 5** *(pdbs=4 ✓ 237.7；pdbs=6 OOM-hang；pdbs=5 未探测)* | **4 或 5** *(pdbs=4 ✓ 204.5；pdbs=6 未探测)* |
| `sparse-gmm-fixed` | 7 | **2** *(pdbs=3 未探测；pdbs=4 OOM)* | **1** *(pdbs=2 OOM 214.6 GiB)* | **1** *(pdbs=1 → 333.6 TGS，pdbs=2/4 未探测)* |

### DCN-EP 关键观察

1. **DS3 dense-cf1.25 在 pdbs=4 上随 DCN_EP 单调下降**（867 → 800 → 649 → 538，每翻倍 −7.7% / −18.7% / −17.2%）。这与 kimi-1T DCN-EP 扩展的[头条发现](kimi-k2-1t-pdbs-sweep.zh.md#dcn-ep-关键观察)在性质上不同：在 kimi-1T 上 dense-cf1.25 在 DCN_EP=2 上反而**比 DCN_EP=1 快了 +6.5%**。kimi-1T 有 384 个专家（DCN_EP=2 时 48 → 24 个专家/GPU，节省了大量每 rank 的专家权重显存，足以补偿跨节点 `all-to-all` RDMA 成本），而 DS3 只有 256 个专家（32 → 16 个专家/GPU 节省得更少，RDMA 成本从一开始就占主导）。**在专家数较少的 MoE 模型上，DCN_EP > 1 是纯成本，没有补偿赢面**；在专家数较多的 MoE 模型上，存在一个小 pdbs 窗口让 DCN_EP=2 能赢。
2. **DCN_EP > 1 时 pdbs=4 → pdbs=8 的 TGS 提升幅度急剧坍塌。** 在 DCN_EP=1 上 dense-cf1.25 从 pdbs=4 → pdbs=8 提升 +37%（867 → 1191）；DCN_EP=2 时缩小到 +12%（799.7 → 898.8）；DCN_EP=4 时只剩 +2.8%（649.3 → 667.6）；DCN_EP=8 时为 +8.9%（537.8 → 585.5）。dense-cf2 表现出同一模式（+27% / +7.4% / +1.4% / +4.1%）。这意味着 **每 pdbs 的吞吐增益斜率随 DCN_EP 大幅变平** —— 跨节点 `all-to-all` 成本变成每步的主导项，淹没了每 pdbs 的 dense compute 摊销收益。在 DCN_EP 高的情况下，提高 pdbs 越过一个小的阈值之后，几乎不再带来吞吐收益。
3. **capacity_factor 对 DCN-EP 的敏感度急剧加大**（与 kimi-1T 同形）。dense-cf1.25 从 EP=1→8 只跌 38%，而 dense-cf4 跌 59%。cf=4 的更大激活张量在跨节点 `all-to-all` 下相对更敏感得多。
4. **`sparse-gmm-fixed` cliff 随 DCN_EP 急剧加深**，与 kimi-1T 同形但起点上限更低：max_pdbs 随 DCN_EP 翻倍而坍塌为 7 → 2 → 1 → 1。dropless 的 RCCL `ragged-all-to-all` over RDMA 是这组配置里最 DCN-EP 脆弱的集合通信；dense 的常规 `all-to-all` 容忍度高得多。在 DCN_EP=8 时 sparse-gmm-fixed pdbs=1 仍能跑出 333.6 TGS（vs kimi-1T 在 DCN_EP=8 时连 pdbs=1 也不可行），反映 DS3 整体上每配置的内存压力更轻。
5. **`sparse-gmm-fixed` 与 dense 配置在 DCN_EP=4 上的编译时间显著拉长**：sparse-gmm-fixed pdbs=1 第一次尝试编译超过 45 分钟被作为可疑 hang scancel；`--time=90:00` 重试干净跑出 338.4 TGS。`dense-cf1.25` 在 DCN_EP=4 / pdbs=4 是同样的模式（首次撞上 `runaway-log` 损坏，二次 17 min hang，三次 `--time=90:00` 跑出 649.3 TGS），`dense-cf4` 在 DCN_EP=4 / pdbs=6 也类似（编译 50 min hang，scancel）。**对 DS3 在 DCN_EP=4 上而言，默认 `--time=60:00` 有时不够 —— 该 DCN_EP 下任何新探测都建议直接使用 `--time=90:00`**。
6. **没法做 DeepEP 对比行** —— [DeepEP "真正能发光"的"Inter-node EP（RDMA 支持的 AllToAll）"假设](#deepep-现状内核不差xla-集成代价大--以及它真正能发光的场景)（跨节点 EP 触发 DeepEP 的 RDMA dispatch 相对 RCCL `all-to-all over RDMA` 的优势）**在当前 MaxText 上仍无法实证检验**。前提是先解除 `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` 的校验。在该改动落地之前，`sparse-gmm-fixed` 是本节里唯一的 dropless DCN-EP 曲线。

---

## 关键结论

1. **峰值吞吐：**
   - **Dropping：** `dense-cf1.25 @ pdbs=16` → 1416 TGS，MFU 14.19 %（1-node/proc）。
   - **Dropless：** `sparse-gmm-deepep-v3 @ pdbs=7` → **1097 TGS**，MFU **10.99 %**（1-node/proc，默认 `MEM_FRACTION=.93`）。相对此前的 dropless 峰值 `sparse-gmm-fixed @ pdbs=7`（989 TGS，且需要 `MEM_FRACTION=.96`）提升 +10.9 %，同时不需要调高显存比例。
2. **启动器对 dense 的影响很小**（每个 pdbs 下 ≤ ±8 %，无一致的赢家）—— 与 sparse-gmm（见结论 #3）形成鲜明对比，在后者中启动器选择驱动了 1.7–3.2× 的巨大差异。
3. **启动器对 `sparse-gmm` 的影响是戏剧性的**：1-GPU/proc 比 1-node/proc 快 1.7×（pdbs=1）→ 3.2×（pdbs=6）。当 EP 轴是进程内的（1-node）时，XLA 将 `ragged_all_to_all` 降级为进程内内核；而当 EP 轴跨越进程（1-GPU）时，XLA 降级为 RCCL；对于该集合通信操作，RCCL 快得多。
4. **最佳 sparse 路径：1-node/proc 上的 `sparse-gmm-deepep-v3`**（2026-04-22 后；详见结论 #9）。
   - **在每个 pdbs 上都击败其他所有 sparse 路径**：pdbs=6 下 1030 vs 949（比此前最佳的 `sparse-gmm-fixed` 快 +8.5 %），pdbs=7 下 1097 vs 989（+10.9 %）。
   - **在默认 `MEM_FRACTION=.93` 下运行** —— pdbs=7 开箱即用，而 `sparse-gmm-fixed @ pdbs=7` 需要 `MEM_FRACTION=.96`（见脚注 ᵃ），1-GPU/proc 在 pdbs=7 则永久 OOM。
   - **与 `sparse-gmm-fixed` 的部署形态完全一致**：1-node/proc 启动器、相同的 Docker 镜像、相同的透传参数 —— 唯一差异是打补丁的 MaxText 分支（`yihuang/moe-turbo-gmm-and-deepep-v3`，自 2026-04-30 起已成为 `container_env.sh` 默认值）。无需新的 XLA 标志，无需更换启动器。（本扫描期间还需要显式设置 `MAXTEXT_PATCH_BRANCH=…`；新提交直接继承默认值。）
   - **严格优于历史 sparse 路径排名**：pdbs=6 下比 stock `sparse-gmm 1-node` 快 3.5×（1030 vs 298）、比基线 `sparse-gmm-deepep` 快 1.59×（1030 vs 647）、比 `sparse-gmm-fixed` 快 1.08×（1030 vs 949）。此前"`sparse-gmm-fixed` 是最佳 dropless 路径"和"`sparse-gmm-deepep` 被集成开销拖垮"两条结论均已过时 —— 一个纯 Python 补丁（基线之上两个提交）就把大部分集成开销消掉了。
5. **更高的 capacity_factor 代价昂贵**：`cf=1.25 → 2.0` 将 TGS 降低 22–32 %；`cf=1.25 → 4.0` 将其降低 50–60 %。`cf=4.0` 在两种启动器下 pdbs=16 均 OOM，且在 1-node 上 pdbs=6 编译不稳定（现已重跑解决）。
6. **在本硬件上同 pdbs 下 dense（dropping）的 TGS 始终击败 sparse-gmm（dropless）。** 所测试的三个 dense 配置（cf=1.25/2.0/4.0）全部采用 dropping，固定了每个专家的容量，使 MaxText 能发出常规 `all-to-all` + 常规 matmul，而不是 `ragged_all_to_all` + ragged matmul。这一内核简化是每个 pdbs 下 TGS 差距的主要来源 —— 这是 dropping 带来的结果，而不是 sparse 需要去克服的东西。本次 15 步吞吐扫描并不检验长期 loss 收敛，而那里才是 dropping 所丢弃的 token 通常以模型质量代价显现的地方；每个 pdbs 内 ≤ 0.003 的 loss 一致性仅是数值正确性探针，并非质量对比。
7. **端到端数值正确性已验证** —— 所有 loss 值在每个 pdbs 内跨启动器 × 配置的 Δ ≤ 0.003。
8. **`sparse-gmm 1-node` 的病态可通过 XLA 标志修复，而不仅限于切换启动器。** `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0`（→ `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`）强制 XLA 的 ragged-all-to-all thunk 使用 `kNccl` 降级，而不是朴素的进程内 one-shot 内核（`RaggedAllToAllKernelImpl<8l>`）。以往只能通过 1-GPU/proc 在运行时自动获得的 3.2× 加速，现在通过编译期标志在 1-node/proc 上也能获得，且 `0` 是默认值，因此普通的 `sparse-gmm` 提交会自动享受此加速。该改动是**严格加性**的 —— 已在实验中验证：对于不发出 `ragged-all-to-all` HLO 算子的配置为空操作：`sparse-gmm-deepep`（HLO 中有 0 个 ragged 算子；已被 DeepEP 的 `moe_dispatch`/`moe_combine` 自定义调用吸收）显示 647 → 642 TGS 的变化，`dense-cf1.25`（从不发出 ragged_all_to_all）显示 1040 → 1026 TGS，两者均在 ≤ 1.3 % 的启动器噪声范围内。
9. **`sparse-gmm-deepep` 在纯 Python 层仍有可观优化空间 —— v2 和 v3 在 pdbs=6 下相对基线分别实现 +24 % 与 +59 % 的 TGS 提升，无需任何库级改动。** v3 的 dispatch 反向不再走重复索引 scatter-add（在 MI355 上该模式会被编译成占据主流、拖垮基线 DeepEP 单步时间的 `input_scatter_fusion_*.kd` 内核）；一个 `jax.custom_vjp` 通过反转排序置换 + reduce-sum 折叠 top-K 重复，完全消掉了该内核。v2 是中间过渡形态（把两次 dispatch-side gather 合并为一次；将 scatter-add 次数减半但仍有原子操作）。前向输出与基线逐位一致 —— 每个 pdbs 下每一步的 loss 都在 bf16 LSB 噪声内吻合。净效果：在 pdbs ∈ 1…7 下 `sparse-gmm-deepep-v3` 全部胜过 `sparse-gmm-fixed`，且在默认 `MEM_FRACTION=.93` 下就能把 pdbs=7 的前沿做出来 —— 而 `sparse-gmm-fixed` 在 pdbs=7 需要 `.96` 才装得下。关于这两处改动对应的内核成本分析，参见 [DeepEP 现状](#deepep-现状内核不差xla-集成代价大--以及它真正能发光的场景) 一节；v3 从 Python 侧大幅覆盖了该节中"方向 1"所指的 `input_scatter_fusion_*.kd` 8.97 秒主流开销。

---

## 基础设施 / 显存上限说明

- **`sparse-gmm 1-node pdbs=7` 需要 `XLA_PYTHON_CLIENT_MEM_FRACTION=.96`**（默认 `.93` → 静默的 RCCL 初始化挂起）。XLA 的工作集为 274.6 GiB；默认池只有 267.8 GiB。提交时请附带 `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96`。
- **`sparse-gmm-fixed 1-node pdbs=7` 同样需要 `MEM_FRACTION=.96`**（默认 `.93` → RESOURCE_EXHAUSTED，分配 217.3 GiB）。`kNccl` 路径比 one-shot 内核用更少内存（主导分配从 275 → 217 GiB），但仍超过默认池。提升到 `.96`（285 GiB 池）后作业干净地以 989 TGS 运行。据此扫描，这是观察到的最佳 dropless 配置 —— **989 TGS vs `sparse-gmm 1-GPU pdbs=6` 的 942 TGS**，另加更大的 pdbs。
- **`sparse-gmm-fixed 1-node pdbs=8, 16` 仍不可行**（pdbs=8 下 OOM 242 GiB，pdbs=16 下 OOM 367 GiB，均在 `MEM_FRACTION=.96`）。该标志改变的是集合通信降级，但无法将 MoE 工作集缩减到足以突破 288 GB HBM 上 pdbs=7 的可行性上限。要在该模型/硬件上超越 pdbs=7，需要换轴（FP8、DCN-EP、更大的硬件）。
- **`sparse-gmm 1-GPU pdbs=7` 在 288 GB MI355 上硬件不可行。** XLA 需要 274.6 GiB；剩余的 HBM（~13 GiB）不足以容纳 1-GPU/proc 每进程的 RCCL peer-access 缓冲区。已在 `MEM_FRACTION ∈ {.96, .97, .98, .99}` 下验证 —— 均以 `Cuda failure 'out of memory'`（出自 RCCL 的 `alloc.h:376`）失败。
- **1-GPU/proc 相比 1-node/proc 每 GPU 额外消耗 ~5–15 GiB HBM**（每进程运行时重复、每进程的 RCCL 通道缓冲区、通过 `hipIpcOpenMemHandle` 的 IPC peer-access 缓冲区，以及跨 8 个独立分配器更高的分配器碎片）。在 pdbs=7 时，1-GPU/proc 位于刀锋的错误一侧。
- **`dense-cf4 1-node pdbs=6`**：原始作业 (12690) 触发了编译期 OOM（XLA 重计算无法降到 278 GiB 以下）。在默认 `MEM_FRACTION=.93` 下的朴素重跑（作业 12886）以 543 TGS 成功 —— 证实首次运行是瞬态的 XLA 调度波动，而非真正的上限（XLA 的 `hlo_rematerialization` 是一种启发式算法，相对 pdbs 并非严格单调；pdbs=6 生成的 DAG 碰巧第一次把它难住了，而 pdbs=7 和 8 编译出的调度恰好能装下）。重试结果已录入上述三张表。
- **将 `MEM_FRACTION` 推得过高会饿死 RCCL**：同一单元格在 `MEM_FRACTION=.96` 下的另一次重试（作业 12885）实际上失败了 —— 在 96% 下，XLA 池外每 GPU 仅剩 ~11 GiB，不足以容纳 RCCL scratch。两个 rank 在集合通信初始化时被 OOM 杀死。默认 `.93` 留出每 GPU ~20 GiB 空闲，这是本模型的正确量级。教训：只在 OOM 前的日志显示 XLA 达到分配上限时才提高 `MEM_FRACTION`（而不是当训练静默死亡时）。
- **1-node/proc 上的 RCCL 初始化挂起具有不稳定性** —— `dense-cf1.25 1-node pdbs=2` 和 `pdbs=6` 在首次提交时都在 RCCL 初始化处挂起，需要 1–2 次重试。没有确定性的根因；重试可解决。
- **扫描期间发生过两次外部取消**（集群管理员活动）。所有 ~10 个受影响的作业都被重新提交，运行至完成或到达各自的自然 OOM。

---

## 脚注

- **ᵃ** `sparse-gmm 1-node pdbs=7` 仅在 `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96` 下测试；默认 `.93` 在 RCCL 初始化处挂起。`sparse-gmm-fixed 1-node pdbs=7` 同样需要 `MEM_FRACTION=.96`（默认 `.93` 在分配 217 GiB 时 OOM；kNccl 路径比 one-shot 内核用更少内存，但仍超过默认池容量）。
- **ᵇ** `sparse-gmm 1-GPU pdbs=7` 在所测试的每个 `MEM_FRACTION`（`.93 → .99`）下均不可行；RCCL OOM。
- **ᶜ** `sparse-gmm-fixed 1-node pdbs=8` 在 `MEM_FRACTION=.96` 下 OOM 于 242 GiB 分配；pdbs=16 OOM 于 367 GiB 分配。kNccl 路径相对 one-shot 内核节省的内存不足以越过 288 GB HBM 上 pdbs=7 的可行性上限。

---

## 三种 sparse 变体的 TGS 为何不同？（pdbs=6 的 profile 深入剖析）

Profiling 作业：**12895**（`sparse-gmm 1-node`）、**12916**（`sparse-gmm 1-GPU`）、**12897**（`sparse-gmm-deepep 1-node`）。每个作业都以 `profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 _env_ENABLE_XLA_DUMP=1` 重跑。下方的每内核耗时取自 xplane trace JSON，按全部 64 GPU × 3 profiled 步（除数 = 192）求平均，使用与 v1 / v2 / v3 剖析同一脚本 [`utils/profile_drill.py`](utils/profile_drill.py)（方法论见 [`skills/profile-drill/SKILL.md`](skills/profile-drill/SKILL.md)）。每个内核的 per-GPU 方差 ≤ 3 %（在 `RaggedAllToAllKernelImpl` 上验证：min 27.5 s，p50 28.4 s，max 29.3 s / 步）。

### 步时间构成（每 GPU 每步，秒，`pdbs=6`）

| 切片                                          | 1-node sparse-gmm (步 82.5 秒) | 1-GPU sparse-gmm (步 26.1 秒) | 1-node sparse-gmm-deepep (步 38.0 秒) |
|----------------------------------------------|------------------------------:|------------------------------:|--------------------------------------:|
| `RaggedAllToAllKernelImpl`（XLA 进程内内核） |                     **28.36** |                          0.00 |                                  0.00 |
| `primus_turbo::deep_ep::*`（DeepEP 原生 HIP）|                          0.00 |                          0.00 |                                  1.62 |
| `input_scatter_fusion_*.kd`                  |                          0.01 |                          0.02 |                              **8.97** |
| `loop_select_fusion_*.kd`（valid-rows mask） |                          0.00 |                          0.01 |                                  1.52 |
| `loop_gather_fusion_*.kd`                    |                          1.79 |                          3.47 |                                  0.00 |
| RCCL (`ncclDevKernel_*`)                     |                          7.50 |                     **15.33** |                                  6.90 |
| CK / Primus-Turbo grouped GEMM + dense GEMM  |                          4.14 |                          8.21 |                                  4.80 |
| Flash-attention (`aiter::fmha_*`)            |                          0.87 |                          1.83 |                                  1.22 |
| 其他 fusion（convert / reduce / transpose / …） |                        2.45 |                          4.64 |                                  1.90 |
| **总内核时间（任何流上）**                    |                     **45.12** |                     **33.51** |                             **26.93** |
| 步 − 总内核 = 空闲间隔（+）或重叠（−）        |                         +37.4 |                          −7.4 |                                 +11.1 |

数字由原始 trace 事件中所有 `.kd` 内核、RCCL `ncclDevKernel_*`，以及 `stream_executor::gpu::*` / `primus_turbo::*` / `ck_tile::*` / `aiter::*` / `Cijk_*` 族的 `dur` 求和，再除以自动检测到的 `64 GPU × 3 profiled 步` = 192 得到。最后一行是 **步 − 总内核**（并非某个内核桶）：

- **1-node sparse-gmm 的 +37.4 秒** —— `RaggedAllToAllKernelImpl` 位于主计算流上，其 `dur` 报告约 155 ms/次，但这期间 SM 主要在等待串行的 xGMI peer 传输；下游内核无法启动，因此该流累计了大量没有任何内核事件归属的实打实 wall-clock。这是"阻塞性内核拖慢一切"的典型特征。
- **1-GPU sparse-gmm 的 −7.4 秒** —— 总内核时间*超过*步时间，说明计算流与 RCCL 通信流的内核确实在重叠。15.33 秒的 RCCL 桶大多藏在约 14 秒的计算工作之后。健康。
- **1-node sparse-gmm-deepep 的 +11.1 秒** —— 1-node 病症的缩小版：`input_scatter_fusion_2.kd`（≈ 4.4 秒）同样阻塞主流，但它造成的停滞预算远小于 1-node one-shot 内核。

注意 GEMM 与 flash-attention 列出现的反直觉差异（1-GPU 列两者均约为 1-node 列的 2×）。单次 GEMM 调用的原始延迟在两种启动器下相同 —— 这 2× 差别是观察伪影：在 1-node/proc 下，进程内 8 个 GPU 的 GEMM 通过 xGMI 感知调度相互交错，XLA profiler 把一部分时间归为非内核开销（折叠进上面 +37.4 秒的空闲间隔）；而在 1-GPU/proc 下每个进程只观察到自己 GEMM 的完整内核事件。1-GPU 列是更忠实的 per-GPU 读数。

### HLO 集合通信算子清单（两种 sparse-gmm 变体完全一致）

| 算子                | sparse-gmm (两者) | sparse-gmm-deepep |
|---------------------|-------------------|-------------------|
| `all-gather`        | 18                | 14                |
| `all-reduce`        | 12                | 12                |
| `ragged-all-to-all` | **6**             | 0                 |
| `all-to-all`        | 4                 | 0                 |
| `reduce-scatter`    | 4                 | 4                 |

`sparse-gmm 1-node` 和 `sparse-gmm 1-GPU` 发出的 HLO 是逐位相同的（相同的 XLA 降级，相同的 6 个 `ragged-all-to-all` 算子）。只有这些集合通信的**运行时降级**不同，因为 `ici_expert_parallelism=8` 轴在 1-node 启动器下是进程内的，而在 1-GPU 启动器下是跨进程的。

`sparse-gmm-deepep` 列中 `ragged-all-to-all` 与 `all-to-all` 的零计数**并不**意味着 DeepEP 没有 dispatch 通信。`use_deepep_dispatch=true` 将这些 HLO 集合通信替换为三种 XLA `custom_call` 目标 —— 直接在作业 12897 的 HLO dump 中计数：

| `custom_call_target`    | HLO 实例数 |
|-------------------------|-----------|
| `moe_dispatch`          | 2         |
| `moe_combine`           | 2         |
| `moe_cached_dispatch`   | 1         |

这 5 个 custom-call 实例共同替代了原版 `sparse-gmm`（即未启用 `use_deepep_dispatch=true`）中用于相同 dispatch/combine 工作的 6 个 `ragged-all-to-all` + 4 个 `all-to-all` = 10 个 HLO 集合通信实例。它们不在上表的集合通信算子清单内，因为该表仅统计标准 HLO 集合通信，不包含 custom-call 算子。在运行时，这些 custom-call 的工作出现在每内核 profile 的两个地方：**8.97 秒/步/GPU 的 `input_scatter_fusion_*.kd`**（实现 custom call 周边 token 重排逻辑的 XLA fusion —— 新的 XLA 桶主导内核，见下文）和 **1.62 秒/步/GPU 的原生 `primus_turbo::deep_ep::intranode::*` HIP 内核**（`dispatch`、`combine`、`cached_notify_combine`、`cached_notify_dispatch`、`get_dispatch_layout`、`notify_dispatch`）。`all-gather` 计数的下降（18 → 14）同理 —— DeepEP 的 combine 自定义调用吸收了原版路径需要单独表达的一些 token 重排 all-gather。

### 28.4 秒/步的确凿证据：`RaggedAllToAllKernelImpl`

当 EP 轴是**进程内的**（1-node/proc 在一个 JAX 进程中有 8 个本地设备）时，XLA 将 `ragged-all-to-all` 降级为进程内内核 `stream_executor::gpu::RaggedAllToAllKernelImpl<8l>`。这是一个朴素的"每个设备循环遍历每个 peer，通过 peer memory access 复制 ragged 段"的实现 —— 跨 peer 串行、未流水化、未分块。在 `pdbs=6` 下它以**约 155 ms/次 × 每步 183 次调用 = 28.4 秒/步/GPU** 运行 —— 单步中最大的单个内核（占 45.1 秒总内核时间的 63 %，占 82.5 秒 wallclock 的 34 %），而且由于它位于主计算流上，无法与计算重叠。除了直接的 28.4 秒外，该内核串行化的 peer 传输还让下游内核无法启动，在上表中打开了 +37.4 秒的"空闲间隔"行 —— 即每在 `RaggedAllToAllKernelImpl` 内部花费 1 秒，另有约 1.3 秒 wall-clock 流逝、主流上什么都运行不了。

当 EP 轴是**跨进程的**（1-GPU/proc 每个 JAX 进程只有 1 个本地设备）时，XLA 无法使用进程内内核，转而回退到 RCCL `AllToAll`。该内核消失（28.4 → 0 秒），其工作多出约 7.8 秒到 RCCL 桶（7.50 → 15.33 秒），并且 RCCL 现在运行在能与计算重叠的通信流上 —— 使"步 − 总内核"从 **+37.4 秒空闲**翻转为 **−7.4 秒重叠**。

净得的 3.2× TGS 胜势（pdbs=6 下 298 → 942 TGS，步 82.5 → 26.1 秒，节省 **56.4 秒**）可分解为：

| 组件                                                    | 步时间 Δ |
|---------------------------------------------------------|---------:|
| `RaggedAllToAllKernelImpl` 从主流移除                    | −28.4 秒 |
| 主流阻塞者消失后，空闲间隔塌陷                           | −44.8 秒（从 +37.4 到 −7.4） |
| RCCL 内核增长（它现在承担 ragged 传输，但在通信流上）    |  +7.8 秒 |
| 其他内核 Δ（GEMM +4.1、FA +1.0、loop_gather +1.7、其他 +2.2） | +9.0 秒 |
| —（最后两行部分抵消了调度收益）                          |         |
| **净值**                                                | **−56.4 秒** |

**≈ 50 % 来自原始内核移除（28.4 秒），≈ 50 % 来自调度级联恢复** —— 与 `sparse-gmm-deepep` 从 v2 → v3 稍后展示的"去掉主流阻塞者，让其余部分重叠"机制一致，只是换了另一个内核。仅就内核质量一项，RCCL AllToAll-over-IPC 每次调用约 **~2.6 ms**，而 one-shot 内核每次调用约 **~155 ms** —— 快 60 倍，纯粹因为前者是手工调优、分块化、跨 peer 并行流水化的，后者是顺序执行的。

**这正是 [#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112) 所修复的内容。** 既然这项加速完全是运行时内核交换带来的、而非启动器本身的性质，那么进程内 one-shot 内核也可以直接通过 XLA 标志 `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` 禁用 —— 即便所有 EP rank 都在同进程内，ragged thunk 也会在 thunk 选择期选择 `kNccl`。结果矩阵中的 `sparse-gmm-fixed (1-node)` 列所测量的正是这条路径：与 1-GPU/proc 的回退路径在运行时完全等价，但运行在 1-node/proc 启动器上，通过环境变量 `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0`（现已成为 `train_env.sh` 的默认值；详见结论 #8）启用。HLO 与 stock `sparse-gmm` 保持逐位一致 —— 改变的只是 thunk 的实现类型选择。

### 为什么 RCCL 每次调用快 ~60×

单次调用的加速（155 ms → ~2.6 ms，60×）来自四项可叠加的硬件利用率差异：

| 维度          | `RaggedAllToAllKernelImpl`          | RCCL `AllToAll`                                                    |
|---------------|-------------------------------------|--------------------------------------------------------------------|
| Peer 并发     | 串行 —— 一次只完成一个 (src, dst) 拷贝 | 所有 7 个 peer 的发送 + 接收同时在飞                               |
| 链路利用     | 每轮只激活单条 xGMI 链路             | 通过多通道并行驱动多条 xGMI 链路                                   |
| 分块 / 流水线 | 无 —— 每个 peer 整块复制             | 每 peer 缓冲区切块，沿链路流水化                                   |
| SM 利用率     | 协调瓶颈（仅少量 thread-block）      | 大量 SM 并发驱动 DMA + 拷贝 + 同步                                |

朴素内核**每轮受单条 xGMI 链路的带宽封顶**，并在 peer 之间承担 launch / barrier 开销 —— 实测 155 ms/次与本模型 ragged-all-to-all 形状在该串行路径上的预期一致。RCCL 的 grouped P2P 路径把同样的流量并行分散到多条链路、多个通道上，每 peer 的传输重叠而非串行，逼近 MI355 节点内 xGMI 实际可提供的聚合带宽 —— 实测 ~2.6 ms/次。

ragged 语义来自算子本身，而非 padding：RCCL 的 `RaggedAllToAllThunk` 按每个 peer 一对的方式发出 grouped `ncclSend`/`ncclRecv`，使用从 HLO 算子的 ragged-offset 操作数中运行时计算得到的**实际**发送/接收长度。没有 padding 导致的字节浪费，也没有丢弃任何 token。

### DeepEP 现状：内核不差、XLA 集成代价大 —— 以及它真正能发光的场景

**内核本身没问题。** 每步每 GPU，DeepEP 的原生 HIP 内核（`primus_turbo::deep_ep::intranode::*` —— dispatch / combine / layout / notify）约 1.62 秒，与 `sparse-gmm 1-GPU` 中 RCCL grouped `ncclSend`/`ncclRecv` 做等价传输所需的 ~0.9 秒同一数量级。DeepEP 的设计初衷（MoE 专用、融合、xGMI IPC、最小化 launch）在内核层面确实兑现了；两者差距主要来自流放置，而非内核本身的效率。

**决定胜负的是 XLA 集成，不是内核。** `moe_dispatch` / `moe_combine` 自定义调用输出的 token 布局与 `grouped_gemm` 期望的 "按专家分组" 布局不一致，XLA 因此插入 `input_scatter_fusion_*.kd` 在两者之间做桥接 —— 在主流上花费 8.97 秒/步的 token 重排计算。sparse-gmm 的 RCCL 路径不需要这一步，因为其输出形状可以直接喂给 GMM。总内核时间（任何流上）直接说明问题：

| 族                                    | `sparse-gmm 1-GPU` | `sparse-gmm-deepep 1-node` |
|---------------------------------------|-------------------:|---------------------------:|
| DeepEP HIP 内核（dispatch/combine/…） |                  0 |                       1.62 |
| `input_scatter_fusion_*.kd`           |               0.02 |                   **8.97** |
| `loop_select_fusion_*.kd`             |               0.01 |                       1.52 |
| `loop_gather_fusion_*.kd`             |               3.47 |                       0.00 |
| RCCL (`ncclDevKernel_*`)              |              15.33 |                       6.90 |
| CK / Primus-Turbo GEMM                |               8.21 |                       4.80 |
| Flash-attention (`aiter::fmha`)       |               1.83 |                       1.22 |
| 其他 fusion + 杂项                    |               4.64 |                       1.90 |
| **总内核时间（任何流上）**            |          **33.51** |                  **26.93** |
| **步时间**                            |          **26.10** |                  **38.00** |
| 步 − 总内核                            |            −7.4 秒 |                    +11.1 秒 |

DeepEP 1-node 在 38.0 秒的步时间里有 **11.1 秒的空闲间隔** —— 约 29 % 的 wallclock 是主流空闲时间，即便 DeepEP 被设计为更快的路径。这段空闲由主流阻塞的 `input_scatter_fusion_2.kd`（≈ 4.4 秒，v3 之前无法去掉原子操作）强制产生。`sparse-gmm 1-GPU` 没有这样的阻塞者，其计算流与 RCCL 流可以重叠（步时间 26.1 秒，负空闲间隔 −7.4 秒）。让 647 TGS（sparse-gmm-deepep 1-node）落在 942 TGS（sparse-gmm 1-GPU）之下的，正是这一 XLA 生成的 fusion 家族，而非 DeepEP 自己的 HIP 内核。DeepEP 仍能击败原版 sparse-gmm 1-node（298 → 647），是因为该路径的 `RaggedAllToAllKernelImpl`（28.4 秒内核 + 37.4 秒空闲间隔 = 65.8 秒主流浪费）远比 DeepEP 的 8.97 秒 scatter + 1.62 秒原生内核 + 11.1 秒间隔 ≈ 21.7 秒更糟糕。（跨启动器比较只用步时间和空闲间隔 —— 1-node/proc 的每内核总和被 attribution 压低，参见上文关于 1-node 与 1-GPU 下 GEMM / flash-attn 的注记。）

**要抹平这个差距** —— 两条方向都属于 Primus-Turbo 的内核工程改动，不是 JAX/XLA 配置能解决的（**已从 Python 侧部分解决**，参见下方 v1/v2/v3 剖析：`sparse-gmm-deepep-v3` 通过一个 `custom_vjp` 把 `input_scatter_fusion` 内核彻底消掉了）：

1. **让 dispatch 直接输出 GMM 兼容的布局。** 把按专家的重排融合进 DeepEP 的 dispatch 内核，让 `input_scatter_fusion_*.kd` 不再被生成。v3 已从 Python 侧近似做到这一点（相对 v1 +59 % TGS）；内核级修复会让 v1 的 Python 代码发出与 v3 等价的 HLO，并可推广到其他 MoE 前端。**这是真正的抓手。**
2. **把 DeepEP 自定义调用包装为 async-start / async-done。** 让 XLA 的延迟隐藏调度器有机会把 DeepEP 的 1.62 秒藏到计算之后。目前被上游约束堵死：JAX 前端发出的是 StableHLO，其合法化 pass 拒绝 `mhlo.async_start`（已亲测 —— XLA 自己的 async collective creator 会在编译后期把 collectives 改写成 async 形式，但用户代码没有公开入口在前端发出 async 自定义调用）。解除阻塞需要 XLA / StableHLO 层面的变更，而不是 Primus-Turbo 的变更。即便实现了，上限也只有约 5 % TGS；替代不了方向 1。

**DeepEP 的设计真正发光的场景**（本次扫描未涉及的领域）：

- **跨节点 EP（基于 RDMA 的 AllToAll）。** RCCL `AllToAll` 走 RDMA 时要承担 round-trip 建立和 ring/tree 开销，而 DeepEP 的直接 RDMA dispatch 可以绕开。本次扫描全部是 `ici_expert_parallelism=8`（纯节点内）；`dcn_expert_parallelism > 1` 的假想配置才会触发这种情况。
- **FP8 dispatch。** DeepEP 内核原生支持 FP8 输入；原版 `ragged_all_to_all` 路径做不到同等高效。线上字节数减半，DeepEP 所针对的那个时间桶就直接减半。
- **H800 / NVLink 栈，也就是 DeepEP 的发源硬件。** 该硬件上 NCCL 的开销特征与 MI355 上 RCCL 不同，DeepEP 的内核融合收益在那里能更好地叠加。

这些场景都无法仅通过 `maxtext-slurm` 配置切换到达 —— 需要不同的拓扑、精度或硬件。

**"让 DeepEP 1-GPU/proc 可用" 这条捷径，单独走不值。** Primus-Turbo 硬编码了 `num_ranks = jax.local_device_count()`，1-GPU/proc 下触发 `AssertionError: Unsupported number of EP ranks: 1`。即便单独修好这个绑定也没用：(a) `sparse-gmm 1-GPU` 上 dispatch 传输已经只占 ~0.9 秒/步/GPU，还几乎完全被计算重叠 —— 把它压到零最多只省 <1 % 的步时间；(b) ~7.8 秒/步/GPU 的真正暴露通信主导来自 34 个*非* dispatch 集合通信（18 `all-gather` + 12 `all-reduce` + 4 `reduce-scatter`），DeepEP 一个都不替换；(c) 方向 1 没先落地的话，1-GPU DeepEP 移植会继承同样的 8.97 秒 `input_scatter_fusion_*.kd` 主流开销，**比原版 `sparse-gmm 1-GPU` 更慢**，而不是更快。应优先推进方向 1；等 DeepEP 在 1-node 上已经有竞争力，启动器问题就自然无关紧要了。

### 结论

sparse MoE 在 1-node → 1-GPU 的全部加速都归结为**摆脱 `RaggedAllToAllKernelImpl`**。三种已知方式：

1. **使用 1-GPU-per-process** 让 EP 轴变为跨进程，回退到 RCCL `AllToAll`（重叠、快 3.2×）。
2. **在 1-node 上使用 `use_deepep_dispatch=true`**，让集合通信被 Primus-Turbo 专用的节点内内核替换（比原版路径快 2.2×，但在 MI355 上对本模型仍逊于方案 1）。
3. **在 1-node 上设置 `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0`**（新增，首次扫描完成后补充；现已成为 `train_env.sh` 的默认值）。这会向 XLA 传递 `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`，禁用进程内 one-shot 内核并强制 ragged thunk 走 `kNccl` 代码路径 —— 即 1-GPU/proc 运行时自动触发的同一种降级。每个 pdbs 上的 TGS 都击败方案 1，同时保留 1-node/proc 的 HBM 余量。**这是新的最佳路径。**

按 pdbs=6 的峰值 TGS 排名（MI355，EP=8，BF16）：

| 方案 | 1-node TGS | 1-GPU TGS | 最适用场景 |
|---|---|---|---|
| Stock `sparse-gmm`（one-shot 内核） | 298 | 942 | —（已被取代） |
| 方案 2：`sparse-gmm-deepep`（基线） | 647 | ✗（EP ranks=1） | —（已被 v3 取代） |
| 方案 1：`sparse-gmm 1-GPU` | — | 942 | 仅 pdbs ≤ 6；每 GPU 付出 5-15 GiB HBM |
| 方案 3：`sparse-gmm-fixed` | 949 | — | pdbs=1…7（pdbs=7 需 `MEM_FRACTION=.96`） |
| **方案 4：`sparse-gmm-deepep-v3`** | **1030** | — | **pdbs=1…7**（pdbs=7 在默认 `MEM_FRACTION=.93` 下即可行） |

方案 4 在本工作负载下取代了方案 1、2 和 3：pdbs=6 下比方案 3 快 +8.5 %，pdbs=7 下快 +10.9 %（1097 vs 989），保持默认显存预算，并且无需调高 `MEM_FRACTION` 就能扩展到 pdbs=7 的 dropless 峰值。它的部署形态与方案 3 完全相同（1-node/proc 启动器，无需额外 Docker 镜像，无需额外环境变量 —— `yihuang/moe-turbo-gmm-and-deepep-v3` 自 2026-04-30 起已是 `container_env.sh` 默认值）。

关于 DeepEP 在本工作负载下的代价去向，以及让它具备竞争力所需的 Primus-Turbo 侧改动，详见上面的 `DeepEP 现状` 一节。

### Profile 作业产物（保存于 `outputs/`）

- `12895-…-TGS_292.811/` —— 1-node sparse-gmm pdbs=6 profile
- `12916-…-TGS_907.556/` —— 1-GPU sparse-gmm pdbs=6 profile
- `12897-…-TGS_631.150/` —— 1-node sparse-gmm-deepep pdbs=6 profile

---

## `sparse-gmm-deepep` 的 v1 / v2 / v3 为何在 TGS 上不同？（pdbs=6 的 profile 深入剖析）

Profiling 作业 —— 三者均以 `profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 _env_ENABLE_XLA_DUMP=1` 在相同的 8 个节点上运行：

- **12897** —— v1 基线 `sparse-gmm-deepep`，分支 `yihuang/moe-turbo-gmm-and-deepep` @ `ad693da2`
- **13412** —— v2，分支 `yihuang/moe-turbo-gmm-and-deepep-v2` @ `627168f8`
- **13382** —— v3，分支 `yihuang/moe-turbo-gmm-and-deepep-v3` @ `f59be3c9`

三者共享镜像、启动器、透传参数，以及 HLO 集合通信清单 —— 分支之间唯一不同的文件是 `src/MaxText/layers/moe.py`。下文的每内核耗时均从各 host 的 trace JSON 中按全部 64 GPU × 3 个 profiled 步（除数 = 192）取平均。列头给出的 TGS / 步时间来自非 profile 运行 **12897**（v1，TGS=647）、**13292**（v2，TGS=806）、**13370**（v3，TGS=1030），这些结果也填入了主结果矩阵；与 profile 运行的步时间差值在各处都 <1 秒，唯独 v3 的 profile 步因 writeback 轻微偏慢。

### 代码层面的差异

三者的不同之处在于 DeepEP 接收的 token 如何被 fan-out 到按专家排序后的 dispatch 布局上。喂给 grouped-GEMM 的 `x` 张量在三个版本中完全相同 —— 区别在于 XLA 把这段前向—反向链编译成了多少个 gather / scatter 内核。

**v1**（`ad693da2`）—— 两次链式 gather，反向两次重复索引 scatter-add：

```python
expanded_x = recv_x[token_indices]                 # fan-out gather（K 路重复）
x          = expanded_x[sort_idx]                  # 排序置换 gather
x          = jnp.where(_deepep_valid_rows, x, 0)
```

**v2**（`627168f8`）—— 将两次 gather 合并为一次；反向保留一次重复索引 scatter-add：

```python
composed_idx = token_indices[sort_idx]             # 合成置换
x            = recv_x[composed_idx]                # 单次 K 路重复 gather
x            = jnp.where(_deepep_valid_rows, x, 0)
```

**v3**（`f59be3c9`）—— 前向与 v2 相同，但用 `jax.custom_vjp` 把反向的重复索引 scatter-add 替换为 `argsort(sort_idx) + reshape + reduce-sum(axis=K)` —— 无原子操作：

```python
x = _deepep_dispatch_fan_out(recv_x, sort_idx, num_topk)  # 与 v2 同样的前向输出
x = jnp.where(_deepep_valid_rows, x, 0)
# _deepep_dispatch_fan_out_bwd 内部：
#   grad_fanned   = grad_x[argsort(sort_idx)]      # 置换 gather，无原子
#   grad_recv_x   = grad_fanned.reshape(N, K, H).sum(axis=1)   # 求和归约，无原子
```

三者前向值逐位一致（每一步的 loss 在 bf16 LSB 精度内相同）；只有*反向* HLO 的结构不同。

### 主流 `input_scatter_fusion_*.kd` —— 主导内核的消失

逐内核分解显示，整个 `pdbs=6` 的 TGS 提升完全由 `input_scatter_fusion_*.kd` 家族随反向 HLO 中重复索引结构的消除而缩小所驱动：

| 版本 | 重头的 scatter-add 内核 | `input_scatter_fusion_*.kd` 秒 / GPU / 步 |
|---|---|---|
| v1 基线（2 次 gather、2 次 scatter-add） | `_2.kd` **4.39 s** + `_3.kd` **4.54 s** | 4.39 + 4.54 + 0.04 = **8.97** |
| v2（合并：1 次 gather、1 次 scatter-add） | 仅 `_2.kd` **4.41 s** | 4.41 + 0.04 = **4.45** |
| v3（custom_vjp：reduce-sum 反向） | *（无 —— 所有变体均 < 25 ms）* | **0.04** |

v1/v2 中每一个重头 `input_scatter_fusion_*.kd` 都是在 bf16[N*K, H] ≈ [1.57 M, 7168] 上的重复索引原子 scatter-add。在 MI355 上，这些原子操作一次只通过 HBM 流动一个 peer-word，无法与其后的 grouped-GEMM 重叠 —— 正是前一 profile drill-down 中归因于 1-node DeepEP 劣势的"主流繁忙"桶。v3 的反向通过一次便宜的置换 gather（无原子）加上在 top-K 轴上的连续 reduce-sum（无原子）得到相同的 `grad_recv_x`，因此 XLA 只发出若干每个 < 25 ms / 步 / GPU 的微小 fusion 变体 —— 比 v1 或 v2 小两个数量级。

### 步时间构成（每 GPU 每步，秒，`pdbs=6`）

| 切片                                         | v1（步 38.0 秒） | v2（步 30.5 秒） | v3（步 23.9 秒） |
|---------------------------------------------|-----------------:|-----------------:|-----------------:|
| `input_scatter_fusion_*.kd`                 |         **8.97** |         **4.45** |         **0.04** |
| `loop_select_fusion_*.kd`（valid-rows mask）|             1.52 |             0.93 |             0.83 |
| RCCL (`ncclDevKernel_*`)                    |             6.90 |             6.89 |             8.18 |
| CK / Primus-Turbo grouped GEMM + dense GEMM |             6.41 |             6.41 |             6.71 |
| Flash-attention (`aiter::fmha_*`)           |             1.22 |             1.22 |             1.27 |
| 其他 fusion（convert / reduce / transpose / select） |     1.95 |             1.80 |             2.10 |
| **总内核时间（任何流上）**                   |        **26.93** |        **21.69** |        **19.13** |
| **步时间（TGS 推算的稳态值）**               |        **38.00** |        **30.50** |        **23.90** |
| 步 − 总内核 = 调度空隙 + 重叠间隔             |            11.07 |             8.81 |             4.77 |

上述各行均由同一脚本（[`utils/profile_drill.py`](utils/profile_drill.py)，方法论见 [`skills/profile-drill/SKILL.md`](skills/profile-drill/SKILL.md)）从全部 8 个 host trace JSON × 8 GPU × 3 profiled 步 = 192 个 gpu-step 采样求平均。"总内核时间"一行把主计算流、RCCL 通信流以及任何辅助流都加在一起 —— 因此 `步 − 总内核` 只是纯空闲时间的下界；真正的"调度无法重叠"间隔更小，因为 RCCL 与计算共享一条执行时间线。

### 为什么 v1 → v2 与 v2 → v3 都节省了超过消除的内核时间

两次转变的节省都是"超线性"的，但 v2 → v3 比 v1 → v2 显著更甚：

| 转变 | Δ `input_scatter_fusion` | Δ 总内核时间 | Δ 步时间 | 步 / 内核 比率 |
|---|---:|---:|---:|---:|
| v1 → v2 | −4.52 秒 | −5.24 秒 | −7.50 秒 | **143 %** |
| v2 → v3 | −4.41 秒 | −2.56 秒 | −6.60 秒 | **258 %** |

**v1 → v2（143 %）。** 消除两个重头 `input_scatter_fusion` 之一，直接从主流削去 4.5 秒；`loop_select_fusion` mask 也从 1.52 缩到 0.93 秒 —— 因为两条 gather/mask 链之一被折叠。调度器又额外回收了约 2 秒的可重叠时间，但并不彻底：v1 的*另一个*主流阻塞 `input_scatter_fusion` 还留着，调度器的手仍未完全自由。

**v2 → v3（258 %）。** 最后一个重头 `input_scatter_fusion` 消失。步时间下降 6.6 秒，而净内核时间只下降 2.6 秒 —— wallclock 节省的*不到一半*来自原始内核移除。剩下的来自调度级联：既然主流上已没有原子 scatter-add 阻塞，XLA 的延迟隐藏调度器就可以重排 grouped-GEMM 反向并将 RCCL dispatch / combine 推入原本被堵住的重叠槽位。证据：RCCL *内核*时间实际上从 v2 的 6.89 秒*上升*到 v3 的 8.18 秒（+1.29 秒），即 XLA 分派了更多通信工作 —— 但其*暴露*（非重叠）wallclock 份额缩减得足以抵消这 1.29 秒再加上 ~5 秒。

从机制上讲，这与前面 `sparse-gmm 1-node → 1-GPU` drill-down 把 3.2× 加速归因于的现象是同一套逻辑（去掉主流阻塞内核，让通信能够重叠）。同一种现象、不同的内核：那里阻塞者是 `RaggedAllToAllKernelImpl`（53 秒/步/GPU，纯 XLA-runtime 内核）；这里是 `input_scatter_fusion_2.kd`（4.4 秒/步/GPU，XLA 发出的重复索引 scatter-add）。消除 `input_scatter_fusion` 正是"DeepEP 现状"一节曾预测能拉近 DeepEP 与 `sparse-gmm 1-GPU` 差距的"方向 1"—— v3 从 Python 侧通过一个 22 行的 `custom_vjp` 关掉了它，而不是通过该节当初指向的 Primus-Turbo 内核改造。

### 结论

三者发出的 HLO 集合通信清单相同，喂给 grouped-GEMM 的前向张量也相同；变化的只是 XLA 的 autodiff 在它们之间插入了多少个原子繁重、主流阻塞的 scatter-add 内核 —— v1：2 个；v2：1 个；v3：0 个。鉴于前向逐位一致、部署形态相同（同镜像、同标志，只是换了补丁分支），在本硬件上 v3 是对基线 `sparse-gmm-deepep` 的毫无歧义的替代。

| 版本 | 补丁分支 | pdbs=6 TGS | pdbs=6 步时间 | Dropless 峰值 pdbs | Dropless 峰值 TGS |
|---|---|---|---|---|---|
| v1（基线） | `yihuang/moe-turbo-gmm-and-deepep` @ `ad693da2` | 647 | 38.0 秒 | 7 | 673 |
| v2 | `yihuang/moe-turbo-gmm-and-deepep-v2` @ `627168f8` | 806（+25 %） | 30.5 秒 | 7 | 836（+24 %） |
| **v3** | `yihuang/moe-turbo-gmm-and-deepep-v3` @ `f59be3c9` | **1030（+59 %）** | **23.9 秒** | **7** | **1097（+63 %）** |

### Profile 作业产物（保存于 `outputs/`）

- `12897-…-TGS_631.150/` —— v1（基线 sparse-gmm-deepep）pdbs=6 profile
- `13412-…-dataset_type_synthetic-profiler_xplane…/` —— v2（`sparse-gmm-deepep-v2`）pdbs=6 profile
- `13382-…-dataset_type_synthetic-profiler_xplane…/` —— v3（`sparse-gmm-deepep-v3`）pdbs=6 profile

---

## 如何复现

```bash
cd /maxtext-slurm
export DOCKER_IMAGE=/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar
# 本扫描在 container_env.sh 默认 v3（2026-04-30 改动）之前完成。
# 要复现 v1 基线数字，下面的 patch-branch 覆盖现已成为必需：
export MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep

# 示例：dense-cf1.25, 1-GPU/proc, pdbs=7
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=7 _env_ONE_GPU_PER_PROCESS=true

# 示例：sparse-gmm-deepep（v1）, 1-node/proc, pdbs=6
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true

# 示例：sparse-gmm, 1-node/proc, pdbs=6（默认变更后；由于 train_env.sh 默认禁用
# one-shot 内核，这条命令复现的就是 `sparse-gmm-fixed (1-node)` 列的结果。
# 等价的显式形式：再加上 _env_ENABLE_RAGGED_ONESHOT_KERNEL=0）。
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true

# 示例：sparse-gmm, 1-node/proc, pdbs=7 —— 峰值 dropless（989 TGS）。由于 kNccl
# 路径在 pdbs=7 的工作集超出默认 `.93` 内存池（单次分配 ~217 GiB），需要
# 更高的 MEM_FRACTION。
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96

# 示例：复现历史 `sparse-gmm (1-node)` 列（开启该标志之前的行为，此时 XLA 的
# 进程内 one-shot 内核仍然生效 —— 例如 pdbs=6 下 298 TGS、pdbs=7 下 302 TGS）。
# 由于当前默认已改为 kernel-disabled，需要显式写出覆盖。
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true \
    _env_ENABLE_RAGGED_ONESHOT_KERNEL=1
```
