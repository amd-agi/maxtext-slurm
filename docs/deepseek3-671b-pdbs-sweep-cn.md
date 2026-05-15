# DeepSeek-V3 671B —— B200 上的 pdbs 扫描实验

- **日期：** 2026-04-10（首轮基线扫描）；2026-05-08（XLA Flag tuning 与 mem_fraction 扩展，详见结果矩阵）
- **模型：** `deepseek3-671b` (MaxText)
- **硬件：** 8 节点 × 8× NVIDIA B200 (179.1 GiB HBM / dev), InfiniBand 互联
- **镜像：** `nvcr.io/nvidia/jax:26.03-maxtext-py3`
- **补丁分支：** [`llying/benchmark-on-nv-b200`](https://github.com/AMD-AGI/maxtext-slurm/tree/llying/benchmark-on-nv-b200) @ `5f68243`
- **基础配置：** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)
- **数据来源：** [`docs/b200-benchmark-report.md`](b200-benchmark-report.md)（按 precision × capacity_factor 重新组织）
- **峰值：** BF16 ≈ 2,250 TFLOP/s/dev；FP8 ≈ 4,500 TFLOP/s/dev
- **XLA_PYTHON_CLIENT_MEM_FRACTION 默认：** `0.93`（预分配约 165.87 GiB / dev；后续部分 run 上调至 `.95 / .96 / .97`）

## 背景

本文档以 [`AMD-AGI/maxtext-slurm@yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md)（MI355 集群上的全面 pdbs 扫描）为参考模板，整理出 B200 集群上 DeepSeek3-671B 的同形扫描视图。原始数据全部来自 [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) —— 这里只是按照"precision (BF16 / FP8) × capacity_factor 变体 (dense-cf1.25 / dense-cf1 / dense-cf2 / dense-cf4 / sparse_matmul)"做了二维切分。pdbs 指 `per_device_batch_size`。

> **B200 与 MI355 的关键差异：** B200 单卡 179.1 GiB HBM (MI355 为 288 GiB)，bf16 峰值 2250 TFLOP/s (MI355 为 ≈2500)，互联用 InfiniBand 而非 Pensando AINIC。因此 B200 上同一模型/配置的可行 pdbs 上限远低于 MI355；MI355 报告里所讨论的 `sparse-gmm-*` / DeepEP / `_env_ENABLE_RAGGED_ONESHOT_KERNEL` 等 XLA / Primus-Turbo 路径在本镜像下并未开放（详见各 `sparse_matmul` 表格说明）。

## 受测配置

每个表格固定一个 (precision, capacity_factor) 组合，行内变化的是 `pdbs` 与具体 XLA / 内存 flag 集合。

| 标签            | 透传参数                                                   |
|-----------------|----------------------------------------------------------|
| `dense-cf1.25`  | *(默认)* — `sparse_matmul=false`, `capacity_factor=1.25`  |
| `dense-cf1`     | `capacity_factor=1.0`                                     |
| `dense-cf2`     | `capacity_factor=2.0`                                     |
| `dense-cf4`     | `capacity_factor=4.0`                                     |
| `sparse_matmul` | `sparse_matmul=true shardy=true`（B200 上需 `shardy=true`，否则 `RaggedDot` 拒绝编译） |

**XLA Flag set 缩写**（与 b200-benchmark-report 一致；所有 run 末尾都会 append `--xla_gpu_enable_command_buffer=''`，来自 `train_env.sh` 的 JAX-0.8.2 fix）：

- **AMD-parity**（镜像默认 `XLA_FLAGS`，便于 AMD / NV 交叉验证）—— 完整 flag 串：

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_memory_limit_slop_factor=95
  --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592   # 8 GiB
  --xla_gpu_all_gather_combine_threshold_bytes=8589934592       # 8 GiB
  --xla_gpu_enable_triton_gemm=false
  --xla_gpu_enable_cublaslt=true
  --xla_gpu_autotune_level=0
  --xla_gpu_enable_all_gather_combine_by_dim=false
  --xla_gpu_enable_command_buffer=''
  ```

  缩写形式：`slop_factor=95, reduce_scatter/all_gather_combine=8 GiB, triton_gemm=false, cublaslt=true, autotune_level=0, all_gather_combine_by_dim=false`。

- **NV defaults**（`_env_XLA_FLAGS_REPLACE` 整体替换为下列两项，丢弃所有 AMD-parity flag）—— 完整 flag 串：

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_enable_command_buffer=''
  ```

  即 `XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=''`。

- **NV + overlap4** — NV defaults 之上通过 `_env_EXTRA_XLA_FLAGS` 追加 `--xla_gpu_experimental_parallel_collective_overlap_limit=4`。
- **AMD + xxx** / **NV + xxx**（如表中 "AMD + autotune=4"、"NV + overlap4"、"NV + combine_by_dim=true"）—— 在对应 base 集合（AMD-parity 或 NV defaults）之上通过 `_env_EXTRA_XLA_FLAGS` 追加列出的覆盖项，或扩展 `XLA_FLAGS_REPLACE`。
- **mem95 / mem97** — `_env_XLA_PYTHON_CLIENT_MEM_FRACTION` 从默认 `.93` 上调至 `.95 / .97`（这是 JAX 预分配池占比，与 XLA flag 正交）。
- **slop95** — 追加 `--xla_gpu_memory_limit_slop_factor=95`（NV defaults 默认无此项；与 AMD-parity 中的同名 flag 等价）。

图例：`✗` = OOM；`—` = 未测试；`SEGFAULT` / `IB HANG` / `IBV_WC_RETRY_EXC_ERR` 等保留作业原始失败状态。

**主要性能指标 = `Tok/s/dev`；辅助指标 = `TFLOP/s/dev` 与 MFU。** 本文档所有"提升 / 降低 / 加速 / 退化 / +X% / −X%"等表述**均以 Tok/s/dev 为准**（NV vs AMD、跨 cf、跨 quantization 对比的标准量）—— Tok/s/dev 不受 FLOP 计数约定影响（FP8 与 BF16 的 peak 不同，TFLOP/s/dev 跨精度直接对比会失真，而 Tok/s/dev 直接对应实际训练吞吐）。`TFLOP/s/dev` 与 `MFU` 列仍保留作为辅助参考（用于看 compute intensity / 利用率分布）。

**Tok/s/dev 计算约定：** 所有表格里的 `Tok/s/dev` 列 = `per_device_batch_size × max_target_length / step_time`（DS3-671B 默认 `max_target_length = 4096`，定义见 [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)）。原始 `b200-benchmark-report.md` 的 Section H (Jobs 4231–4247) / Section I (Jobs 4396–4419) 子表早期只列了 `TFLOP/s/dev`，已据上式补齐 `Tok/s/dev` 列。
---

## BF16

### `dense-cf1.25`（BF16, `capacity_factor=1.25`, `sparse_matmul=false`）

行按 pdbs 升序，组内按 Job ID 升序；同一 pdbs 下加粗当前最佳 Tok/s/dev 的 run（TFLOP/s/dev 同时加粗作为辅助参考，二者排序一致）。

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  4 | 1196 | AMD-parity | SUCCESS | 18.47 |  9.85 |   221.8 |   886 | **首个可行 pdbs**；约 22 GiB headroom |
|  6 | 1198 | AMD-parity | SUCCESS | 23.34 | 11.71 |   263.5 | 1,048 | OOM 边界探针；2.5 GiB headroom |
|  7 | 1200 | AMD-parity | SUCCESS | 26.51 | 12.04 |   270.8 | 1,081 | **AMD-parity 下 max pdbs（无其他优化）** |
|  7 | 1210 | AMD-parity + `profiler=xplane` | SUCCESS | 26.43 | 12.08 |   271.8 | 1,085 | xplane traces captured |
|  7 | 1213 | AMD-parity + `ici_fsdp=2, ici_ep=4` | SUCCESS | 26.29 | 12.14 |   273.2 | 1,090 | 轻微 +0.9% |
|  7 | 1214 | AMD-parity + `megablox=True` | SUCCESS | 26.55 | 12.02 |   270.5 | 1,080 | 中性 / 略差 |
|  7 | 1524 | NV defaults | SUCCESS | 24.23 | 13.17 |   296.4 | 1,183 | **NV +9.4% vs AMD-parity** |
|  7 | 1525 | AMD + `autotune_level=4` | SUCCESS | 26.50 | 12.04 |   270.9 | 1,082 | isolation 无效 |
|  7 | 1526 | AMD + `triton_gemm=true` | SUCCESS | 100.58 |  3.17 |    71.3 |   285 | **灾难性 −74%** — 永不使用 |
|  7 | 1527 | AMD + `slop_factor=300` | ✗ OOM | -- | -- | -- | -- | 108.71 GiB |
|  7 | 1528 | AMD + `combine=256 B` | IB HANG | -- | -- | -- | -- | NCCL 卡死；resubmitted as 1588 |
|  7 | 1529 | AMD + pipelined collectives | SUCCESS | 26.71 | 11.95 |   268.9 | 1,073 | −0.7% — 与 LHS 冲突 |
|  7 | 1530 | AMD + `combine_by_dim=true` | SUCCESS | 25.10 | 12.71 |   286.0 | 1,142 | **第二大单 flag 增益 +5.6%** |
|  7 | 1588 | AMD + `combine=256 B` (重试) | SUCCESS | 24.27 | 13.16 |   295.7 | 1,182 | +9.3% — 约等同 NV defaults |
|  7 | 1589 | NV + `combine_by_dim=true` | SUCCESS | 24.27 | 13.15 |   295.9 | 1,181 | NV defaults 之上无额外增益 |
|  7 | 1590 | NV + `megablox=True` | SUCCESS | 24.25 | 13.16 |   296.1 | 1,182 | 中性 |
|  7 | 1591 | NV + `shardy=true` | SUCCESS | 24.27 | 13.16 |   296.1 | 1,181 | dense 路径 Shardy 中性 |
|  7 | 4231 | NV + `while_loop_double_buffering=true` | FAILED | -- | -- | -- | -- | LHS 超预算 + XLA IndexError |
|  7 | 4233 | NV + `pipelined_all_gather=true` | FAILED | -- | -- | -- | -- | LHS 124.6 > 109.4 GiB |
|  7 | 4234 | NV + `pipelined_reduce_scatter=true` | FAILED | -- | -- | -- | -- | LHS 122.3 > 109.4 GiB |
|  7 | 4235 | NV + `pipelined_all_reduce=true` | FAILED | -- | -- | -- | -- | LHS 133.4 > 109.4 GiB |
|  7 | 4236 | NV + `highest_priority_async_stream=true` | SUCCESS | 24.30 | 13.13 |   295.5 | 1,180 | 中性 (−0.3%) |
|  7 | 4237 | NV + `parallel_collective_overlap_limit=2` | SUCCESS | 23.98 | 13.31 |   299.5 | 1,196 | +1.0% |
|  7 | 4238 | NV + `parallel_collective_overlap_limit=4` | SUCCESS | 22.24 | 14.35 | **322.8** | **1,289** | **+9.0% Tok/s/dev — 单 flag 最佳** |
|  7 | 4239 | NV + `parallel_collective_overlap_limit=8` | SUCCESS | 23.34 | 13.68 |   307.7 | 1,228 | +3.8% |
|  7 | 4240 | NV + `{ag,rs}_combine=256 MiB` | SUCCESS | 24.50 | 13.03 |   293.2 | 1,170 | −1.1% |
|  7 | 4241 | NV + `{ag,rs}_combine=512 MiB` | SUCCESS | 25.13 | 12.70 |   285.8 | 1,141 | −3.6% |
|  7 | 4242 | NV + `{ag,rs}_combine=1 GiB` | SUCCESS | 25.18 | 12.68 |   285.2 | 1,139 | −3.8% |
|  7 | 4243 | NV + `{ag,rs}_combine=2 GiB` | SUCCESS | 25.42 | 12.56 |   282.5 | 1,128 | −4.7% |
|  7 | 4244 | NV + `{ag,rs}_combine=4 GiB` | SUCCESS | 26.40 | 12.09 |   272.0 | 1,086 | −8.2% |
|  7 | 4245 | NV + `ag_combine=256 MiB` | SUCCESS | 24.38 | 13.09 |   294.5 | 1,176 | −0.6% |
|  7 | 4246 | NV + `ag_combine=1 GiB` | TIMEOUT | -- | -- | -- | -- | LHS 120.6 > 109.4 GiB |
|  7 | 4247 | NV + `ag_combine=4 GiB` | CANCELLED | -- | -- | -- | -- | Reservation 到期 |
|  7 | 4396 | NV + `ag_combine=1 GiB + slop95` | SUCCESS | 25.15 | 12.69 |   285.5 | 1,140 | Rescued 4246 |
|  7 | 4397 | NV + `ag_combine=4 GiB + slop95` | SUCCESS | 25.81 | 12.36 |   278.2 | 1,111 | Rescued 4247 |
|  7 | 4399 | NV + `pip-ag + slop95` | SUCCESS | 26.19 | 12.19 |   274.2 | 1,095 | Rescued 4233 |
|  7 | 4400 | NV + `pip-rs + slop95` | SUCCESS | 24.31 | 13.13 |   295.4 | 1,179 | Rescued 4234，中性 |
|  8 | 1195 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 108.63 GiB |
|  8 | 1203 | AMD + `optimizer_memory_host_offload=True` | ✗ OOM | -- | -- | -- | -- | offload 反而更糟 (146.95 GiB) |
|  8 | 1204 | AMD + `shard_exp_on_fsdp=True` | FAILED | -- | -- | -- | -- | args 492 GiB > limit 178 GiB |
|  8 | 1205 | AMD + offload + shard_exp | ✗ OOM | -- | -- | -- | -- | CUDA OOM |
|  8 | 4401 | NV + overlap4 + slop95 | ✗ OOM | -- | -- | -- | -- | 113.79 GiB |
|  8 | 4403 | NV defaults + mem95 | SUCCESS | 29.76 | 12.26 |   275.8 | 1,101 | **mem95 解锁 pdbs=8** |
|  8 | 4404 | NV defaults + mem97 | SUCCESS | 29.17 | 12.51 |   281.4 | 1,123 | mem97 解锁 pdbs=8 |
|  8 | 4407 | NV + overlap4 + mem97 | SUCCESS | 28.09 | 12.98 | **292.1** | **1,167** | **mem97 解锁 pdbs=8 + overlap4** |
|  9 | 4408 | NV + mem97 | ✗ OOM | -- | -- | -- | -- | 115.10 GiB — `cf=1.25` 在 B200 上 pdbs=8 是硬上限 |
| 12 | 1194 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 140.37 GiB |
| 16 | 1193 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 171.96 GiB — baseline 直接 OOM |

**核心观察（以 Tok/s/dev 为准）：**

- **pdbs=8 是 BF16 cf=1.25 在 B200 上的硬上限**（pdbs=9 即便 mem97 也 OOM 115.10 GiB）。
- **NV defaults vs AMD-parity 稳定 +9.4% Tok/s/dev**：Job 1524 (NV) 1,183 vs Job 1200 (AMD) 1,081 = +9.4%；多数 AMD-parity 单 flag isolation 都中性或退化。
- **pdbs=7 + `overlap_limit=4` (Job 4238) 是单 flag 全场最佳**：**1,289 Tok/s/dev** (+9.0% vs NV defaults 1,183；+19.2% vs AMD-parity 1,081)，辅助指标 322.8 TFLOP/s/dev / 14.35% MFU；提升到 `overlap=8` 退化到 1,228 (+3.8%)。
- **mem97 + overlap4 + pdbs=8 (Job 4407) 是 BF16 cf=1.25 全场最佳吞吐**：**1,167 Tok/s/dev**（单 pdbs 来看比 4238 低 −9.5%，但全局 batch +14% 抵消有余）；辅助 292.1 TFLOP/s/dev。
- **Combine threshold 在 B200 上单调退化**：从 256 MiB → 4 GiB 的 Tok/s/dev 落差 −1.1% → −8.2%（4240=1,170 → 4241=1,141 → 4242=1,139 → 4243=1,128 → 4244=1,086 vs 1,183 baseline），256 B（与 NV defaults 等价）反而最优 —— 与 MI355 部分文献趋势相反。

---

### `dense-cf1`（BF16, `capacity_factor=1.0`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 1207 | AMD-parity | SUCCESS | 23.71 | 13.46 | **302.9** | 1,209 | **cf=1.0 = 单个最大优化 (+11.8% Tok/s/dev)** vs cf=1.25 bs=7 (1,081) |
|  7 | 1212 | AMD-parity + `gradient_accumulation_steps=2` | ✗ OOM | -- | -- | -- | -- | 118.91 GiB — ga 反而增加显存 |
|  7 | 1216 | AMD-parity + `megablox=True` | SUCCESS | 23.62 | 13.51 |   304.0 | 1,214 | 相对单独 cf=1.0 微小额外增益 |
|  7 | 4398 | NV + overlap4 + slop95 | FAILED | -- | -- | -- | -- | SEGFAULT（LHS 超预算） |
|  7 | 4406 | NV + overlap4 + mem97 | SUCCESS | 23.42 | 13.63 |   306.7 | 1,224 | +1.3% over plain cf=1.0；< cf=1.25 + overlap4 peak |
|  8 | 1208 | AMD-parity | SUCCESS | 26.38 | 13.83 | **312.2** | 1,242 | **cf=1.0 解锁 pdbs=8**；soft-over 4.6 GiB |
|  8 | 1235 | AMD + `megablox=True` | SUCCESS | 26.35 | 13.84 |   311.4 | 1,243 | **AMD-parity 下 BF16 最佳**（仅边际） |
|  8 | 4402 | NV + overlap4 + slop95 | FAILED | -- | -- | -- | -- | SEGFAULT（LHS 超预算） |
|  9 | 1218 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 109.83 GiB |
|  9 | 1236 | AMD + `num_vocab_tiling=4` | FAILED | -- | -- | -- | -- | dtype assertion (f32 vs bf16) — 与 DS3-671B 不兼容 |
|  9 | 1237 | AMD + `grad_dtype=bfloat16` | ✗ OOM | -- | -- | -- | -- | 109.83 GiB — grad_dtype 零节省 |
|  9 | 1238 | AMD + `grad_dtype=bf16 + vocab_tiling=4` | FAILED | -- | -- | -- | -- | 与 1236 同 dtype assertion |
|  9 | 4405 | NV + mem97 | SUCCESS | 28.99 | 14.15 | **318.4** | **1,272** | **mem97 解锁 pdbs=9；BF16 cf=1.0 峰值 MFU** |
| 10 | 1219 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 116.11 GiB |

**核心观察（以 Tok/s/dev 为准）：**

- **cf=1.0 是 B200 上单个最大的 BF16 优化：+11.8% Tok/s/dev**（Job 1207 cf=1.0 bs=7 = 1,209 vs Job 1200 cf=1.25 bs=7 = 1,081；同 AMD-parity flag set 同 pdbs）—— 减小 dispatch padding 同时节省显存 + 计算。
- **pdbs=9 解锁需要 mem97**（Job 4405：**1,272 Tok/s/dev**，辅助 318.4 TFLOP/s/dev / 14.15% MFU = BF16 全场最高 MFU）—— 比 cf=1.0 pdbs=8 baseline (1,242) 再 +2.4%。
- **B200 上 BF16 cf=1.0 hard ceiling = pdbs=9**（pdbs=10 即便 AMD-parity 也 OOM 116.11 GiB）。
- 多个 "减显存" 备选（`num_vocab_tiling`、`grad_dtype=bf16`、`optimizer_memory_host_offload`、`shard_exp_on_fsdp`）在 DS3-671B 上要么 dtype 不兼容，要么零节省 / 反而更大 —— 都没有为 Tok/s/dev 带来净增益。

---

### `dense-cf2`（BF16, `capacity_factor=2.0`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  4 | 1595 | AMD-parity | SUCCESS | 22.04 |  8.28 |   186.2 |   743 | viable 但 −37.2% Tok/s/dev vs cf=1.25 NV defaults (1,183) |
|  5 | 4163 | AMD-parity | SUCCESS | 25.53 |  8.93 |   200.9 |   802 | cf=2.0 边界 (AMD) |
|  5 | 4164 | NV defaults | SUCCESS | 23.69 |  9.62 |   216.5 |   864 | **cf=2.0 最佳 (NV defaults)** — NV +7.7% Tok/s/dev vs AMD 同 pdbs |
|  5 | 4416 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR（网络不稳定） |
|  6 | 4171 | NV defaults | ✗ OOM | -- | -- | -- | -- | 109.38 GiB |
|  6 | 4177 | NV + `slop_factor=95` | ✗ OOM | -- | -- | -- | -- | 109.38 GiB — slop95 无效 |
|  6 | 4417 | NV + overlap4 + mem97 | SUCCESS | 28.43 |  9.62 |   216.5 |   864 | **mem97 解锁 pdbs=6**；全局 batch +20% |
|  7 | 1522 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 116.84 GiB |
|  7 | 4423 | NV + overlap4 + mem97 | SUCCESS | 31.40 | 10.16 | **228.7** |   913 | **mem97 解锁 pdbs=7** — cf=2.0 全场最佳 |

**核心观察（以 Tok/s/dev 为准）：**

- **cf=2.0 把 max pdbs 从 cf=1.0 的 9 砍到 5（默认）/ 7（mem97）**；每提升一个 capacity_factor 等级，dispatch padding 翻倍，可行 pdbs 约减半。
- **NV defaults 在 cf=2.0 上同样 +7.7% Tok/s/dev**：Job 4164 (NV) 864 vs Job 4163 (AMD) 802 = +7.7%。NV 优势对 cf 不敏感（cf=1.25 也是 +9.4%）。
- **mem97 + overlap4 是 cf=2.0 唯一突破 pdbs=5 上限的路径**（4417 pdbs=6 = 864 Tok/s/dev / 4423 pdbs=7 = 913 Tok/s/dev 均靠 mem97），但已逼近 IB 网络稳定上限（4416 pdbs=5 即使有 mem97 也 IB 错误退出）。


---

### `dense-cf4`（BF16, `capacity_factor=4.0`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  2 | 1596 | AMD-parity | SUCCESS | 20.42 |  4.47 |   100.5 |   401 | viable 但 −66.1% Tok/s/dev vs cf=1.25 NV defaults (1,183) |
|  2 | 4176 | NV defaults | SUCCESS | 18.98 |  4.80 | **108.1** |   432 | **cf=4.0 NV defaults 最佳** — NV +7.7% Tok/s/dev vs AMD |
|  2 | 4418 | NV + overlap4 + mem97 | SUCCESS | 20.44 |  4.46 |   100.4 |   401 | overlap4 在 cf=4.0 + pdbs=2 上微弱退化 |
|  3 | 4175 | NV defaults | SEGFAULT | -- | -- | -- | -- | 静默崩溃于 XLA compile；~14 GB coredump |
|  3 | 4419 | NV + overlap4 + mem97 | SUCCESS | 26.39 |  5.18 | **116.6** | **466** | **mem97 解锁 pdbs=3** — cf=4.0 全场最佳 |
|  4 | 4172 | NV defaults | ✗ OOM | -- | -- | -- | -- | 108.26 GiB |
|  4 | 4424 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR（取消以让位 profiling） |
|  5 | 4165 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 126.30 GiB |
|  5 | 4166 | NV defaults | ✗ OOM | -- | -- | -- | -- | 126.42 GiB — flag set 在此 margin 下基本无关 |
|  7 | 1523 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 145.29 GiB |

**核心观察（以 Tok/s/dev 为准）：**

- **cf=4.0 在 B200 上 max pdbs = 3**（mem97），最佳 **466 Tok/s/dev** (Job 4419)；辅助 116.6 TFLOP/s/dev。与 cf=1.25 best (1,289) 相比 **−63.8%**；与 cf=1.0 best (1,272) 相比 **−63.4%**。
- **NV defaults vs AMD-parity 在 cf=4.0 上同样 +7.7% Tok/s/dev**：Job 4176 (NV bs=2) 432 vs Job 1596 (AMD bs=2) 401 = +7.7%。NV 优势在所有 cf 上一致。
- pdbs=3 在 NV defaults 单独下 SEGFAULT (4175)，但加 overlap4 + mem97 干净跑出来 — XLA scheduling 对该 OOM 边界的稳定性敏感。
- pdbs=4 在 mem97 下触发 IB 重试错误（4424），属于 B200 cluster IB 在 cf=4.0 大 buffer 下的边界状况。
- **cf=4.0 与 cf=2.0 在 B200 上几乎都属于"为研究 capacity_factor 影响"的 ablation，而非生产配置**（Tok/s/dev 比 cf=1.0/1.25 低 28~64%）。

---

### `sparse_matmul`（BF16, `sparse_matmul=True + shardy=True`）

| pdbs | Job ID | XLA Flags / Run | Status | Failed alloc | 备注 |
|---:|---:|---|---|---:|---|
|  1 | 4198 | NV + `one_shot=true` | ✗ OOM | **112 GiB** | 关键数据点 — 单 pdbs 单位需 ~112 GiB |
|  1 | 4201 | NV + `one_shot=true + slop95` | ✗ OOM | 112 GiB | byte-identical to 4198，slop95 无效 |
|  1 | 4229 | NV + `one_shot=true + mem_fraction=.95` | ✗ OOM | 112 GiB | byte-identical，mem95 也无效 |
|  2 | 4189 | NV + `one_shot=true` | ✗ OOM | 224 GiB | = 2 × 112 |
|  2 | 4190 | NV + `one_shot=false` | ✗ OOM | 224 GiB | byte-identical to 4189，one_shot toggle 无显存效果 |
|  7 | 1215 | AMD-parity (no shardy) | FAILED | -- | RaggedDot requires Shardy（合法化失败） |
|  7 | 1217 | AMD-parity + `cf=1.0` (no shardy) | FAILED | -- | 与 1215 同因 |
|  7 | 1239 | AMD-parity + `shardy=true + cf=1.0` | ✗ OOM | **2.28 TiB** | Shardy plan 病理性放大 |
|  7 | 4182 | NV + `one_shot=true + cf=1.0` | ✗ OOM | 784 GiB / 224 GiB | 异质失败（5 ranks / 3 ranks） |
|  7 | 4183 | NV + `one_shot=false + cf=1.0` | ✗ OOM | 784 GiB / 224 GiB | byte-identical to 4182 |
|  8 | 1240 | AMD-parity + `shardy=true + cf=1.0` | ✗ OOM | **2.60 TiB** | Shardy plan 同形病理 |

**核心观察（来自 b200-benchmark-report Section G）：**

1. **`sparse_matmul=True + shardy=True` 在 8N B200 上 planning 出单 pdbs 单位 ~112 GiB 的分配。** B200 单卡可用 XLA budget ≈ 102 GiB，即便 pdbs=1 也差 ~10 GiB，**所有 pdbs 不可行**。
2. **`xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel` toggle 对显存零效果**（4182 vs 4183 / 4189 vs 4190 均 byte-identical）。该 flag 控制 ragged-a2a 集合通信的执行策略而非 XLA buffer planning。
3. **`slop_factor=95` 与 `mem_fraction=.95` 也均无效**（4201 / 4229 byte-identical to 4198）—— 失败发生在 XLA planner 的 shape-based 可行性检查中，发生在 BFC arena 与 mem pool 之前。
4. **NV defaults 相对 AMD-parity 把最坏分配缩了约 3.3×（2.60 TiB → 784 GiB）**，但仍远超 179 GiB B200 HBM。Shardy + sparse_matmul 的本质问题在 MaxText `e26c2ac7` + JAX 26.03 镜像下未解。
5. **B200 上 sparse 路径不可用**，MI355 报告里讨论的 `sparse-gmm` / `sparse-gmm-fixed` / `sparse-gmm-deepep-v*`（依赖 Primus-Turbo + DeepEP）在本镜像下没有携带，无对应可测列。生产路径只能走 `sparse_matmul=False`（dense MoE）。

---

## FP8

### `dense-cf1.25`（FP8, `capacity_factor=1.25`, `quantization=fp8`）

> FP8 MFU% 相对 FP8 peak (4,500 TFLOP/s for B200)；BF16-equivalent MFU = MFU × 2.

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 1592 | NV defaults | SUCCESS | 21.40 |  7.46* |   335.5 | 1,340 | **prior best (cf=1.25)** |
|  8 | 1593 | NV defaults | ✗ OOM | -- | -- | -- | -- | 107.70 GiB |
|  8 | 4412 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR |
|  9 | 4413 | NV + mem97 | SUCCESS | 26.35 |  7.79* | **350.4** | **1,399** | **mem97 解锁 pdbs=9 — FP8 cf=1.25 最佳 (+4.4% Tok/s/dev vs 1592 1,340)** |
| 10 | 1594 | NV defaults | ✗ OOM | -- | -- | -- | -- | 122.93 GiB |

**核心观察（以 Tok/s/dev 为准）：**

- **FP8 cf=1.25 + mem97 + pdbs=9 (Job 4413) 把 NV defaults baseline 推到 1,399 Tok/s/dev**（+4.4% vs 1592 的 1,340；辅助 350.4 TFLOP/s/dev）。
- FP8 cf=1.25 ceiling = pdbs=9（pdbs=10 在 NV defaults 下 OOM 122.93 GiB；pdbs=8 + overlap4 + mem97 触发 IB 错误）。
- 相对 BF16 cf=1.25 同 pdbs=7 baseline (Job 1524 = 1,183 Tok/s/dev)，FP8 cf=1.25 pdbs=7 (Job 1592 = 1,340) = **+13.3% Tok/s/dev**。

### `dense-cf1`（FP8, `capacity_factor=1.0`, `quantization=fp8`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 4161 | NV defaults | SUCCESS | 19.46 |  8.20* |   369.0 | 1,473 | cf=1.0 +10.0% Tok/s/dev vs cf=1.25 (1592 1,340) |
|  8 | 4170 | NV + `slop_factor=95` | SUCCESS | 21.55 |  8.46* | **380.9** | **1,521** | **OVERALL BEST FP8** — slop95 解锁 pdbs=8 (+3.3% Tok/s/dev over pdbs=7) |
|  9 | 4178 | NV + `slop_factor=95` | ✗ OOM | -- | -- | -- | -- | 112.39 GiB |
|  9 | 4409 | NV + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR |
| 12 | 1241 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 118.74 GiB（~比 BF16 bs=12 少 15%） |
| 16 | 1242 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 147.38 GiB |

**核心观察（FP8 全场最佳；以 Tok/s/dev 为准）：**

- **`fp8 cf=1.0 + slop95 + pdbs=8` (Job 4170) 是 DS3-671B B200 全场最佳吞吐：1,521 Tok/s/dev**（辅助 380.9 TFLOP/s/dev / FP8 MFU 8.46% = BF16-eq MFU 16.92%）。相对 BF16 best (Job 1208 cf=1.0 pdbs=8 = 1,242 Tok/s/dev) = **+22.5% Tok/s/dev**；相对 BF16 + mem97 best (Job 4405 pdbs=9 = 1,272 Tok/s/dev) = **+19.6%**。
- **FP8 cf=1.0 vs FP8 cf=1.25 同 pdbs=7：+10.0% Tok/s/dev**（Job 4161 = 1,473 vs Job 1592 = 1,340）—— 与 BF16 上的 cf=1.0 增益 (+11.8%) 量级一致。
- **`slop_factor=95` 是 FP8 cf=1.0 解锁 pdbs=8 的关键**（默认 `.93` 下 1593 / 4178 都 OOM）—— 与 BF16 cf=1.25 同 pattern。
- pdbs=9 在 slop95 / mem97 下都过不去（4178 OOM / 4409 IBV 错误），**pdbs=8 是 FP8 cf=1.0 在本硬件下的最终 ceiling**。

### FP8 未覆盖配置说明（`dense-cf2` / `dense-cf4` / `sparse_matmul`）

本次 B200 扫描的目的是**与 MI355 参考报告做同形对照**，因此 B200 上只跑了 MI355 上已经跑通、能产出可比数字的那些 (precision, config) 组合。这三项在 MI355 参考报告里**就没跑通 / 没列出可比的 FP8 数据**，所以 B200 上也对应跳过：

| FP8 config | 跳过原因 | MI355 参考报告对应状态 |
|---|---|---|
| `dense-cf2` (FP8) | 无 MI355 FP8 cf=2.0 基准可对照 | MI355 参考报告所有 FP8 列均未跑（该报告聚焦 BF16 + sparse-gmm-deepep 内核分析），cf=2.0 仅有 BF16 列 |
| `dense-cf4` (FP8) | 同上 | 同上 |
| `sparse_matmul` (FP8) | BF16 sparse_matmul 在 B200 上 pdbs=1 即 OOM 112 GiB —— 该失败发生在 XLA planner 的 shape-based 可行性检查里，与 dtype 无关（dispatch routing 是 index dtype，FP8 无法压缩）。MI355 上能跑通是因为换了 `sparse-gmm`/`sparse-gmm-deepep-v3` 这条 Primus-Turbo 内核路径，在本 B200 镜像 `nvcr.io/nvidia/jax:26.03-maxtext-py3` 下并未携带 | MI355 参考报告该列也只在 BF16 下跑出数据；FP8 dispatch 在 DeepEP 内核原生支持（参考报告 "DeepEP 真正能发光的场景" 一节），但在 B200 + JAX 26.03 镜像下没有相同的内核入口 |

**总结：** FP8 在 B200 上只对 `dense-cf1.25` / `dense-cf1` 两项可行（也是 MI355 参考报告里相对应可比的两项；MI355 的 dense-cf1.25 vs dense-cf1 对应本报告的 1592 / 4161 / 4170 系列）。`cf=2.0` / `cf=4.0` / `sparse_matmul` 三项在 MI355 上 FP8 数据也缺，对照无意义，因此 B200 上也未投入测试资源；如果未来 MI355 参考报告补齐这些 FP8 列，再回到 B200 上补对应的 run。

---

## 全场最佳汇总

按 **Tok/s/dev**（主指标）降序排列；TFLOP/s/dev / MFU / Step 为辅助指标。

| Rank | Precision | Config | pdbs | **Tok/s/dev** | Δ vs BF16 best | Job | XLA Flags | Step (s) | TFLOP/s/dev | MFU (%) |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| **1** | **FP8** | **dense-cf1** | **8** | **1,521** | **+19.6%** | **4170** | **NV + slop95** | **21.55** | **380.9** | **8.46\*** |
| 2 | FP8 | dense-cf1.25 | 9 | 1,399 | +10.0% | 4413 | NV + mem97 | 26.35 | 350.4 | 7.79* |
| 3 | BF16 | dense-cf1.25 | 7 | 1,289 | +1.3% | 4238 | NV + overlap4 | 22.24 | 322.8 | 14.35 |
| **4** | **BF16** | **dense-cf1** | **9** | **1,272** | **0** (基准) | **4405** | **NV + mem97** | **28.99** | **318.4** | **14.15** |
| 5 | BF16 | dense-cf1.25 | 8 | 1,167 | −8.3% | 4407 | NV + overlap4 + mem97 | 28.09 | 292.1 | 12.98 |
| 6 | BF16 | dense-cf2 | 7 | 913 | −28.2% | 4423 | NV + overlap4 + mem97 | 31.40 | 228.7 | 10.16 |
| 7 | BF16 | dense-cf4 | 3 | 466 | −63.4% | 4419 | NV + overlap4 + mem97 | 26.39 | 116.6 | 5.18 |
| — | BF16 | sparse_matmul | — | **不可行** | — | — | — | — | — | — |

\* FP8 MFU% 相对 FP8 peak (4,500 TFLOP/s)；BF16-equivalent MFU = MFU × 2 = 16.92% (Job 4170) / 15.58% (Job 4413)。"Δ vs BF16 best" 列以 BF16 全场最佳 Job 4405 (1,272 Tok/s/dev) 为基准。

---

## 关键结论

> 所有"+X% / −X%"均以 **Tok/s/dev**（主指标）计算；TFLOP/s/dev 与 MFU 同时给出作为辅助参考。

1. **B200 全场 best = FP8 cf=1.0 pdbs=8 + slop95 (Job 4170)：1,521 Tok/s/dev**（辅助 380.9 TFLOP/s/dev、FP8 MFU 8.46% = BF16-eq MFU 16.92%）。相对 B200 BF16 全场 best (Job 4405 cf=1.0 pdbs=9 + mem97 = **1,272 Tok/s/dev**) 提升 **+19.6%**；相对 B200 原始 BF16 baseline (Job 1208 cf=1.0 pdbs=8 = 1,242) 提升 **+22.5%**。**但仍低于 MI355 BF16 cf=1.0 pdbs=16 peak 1,598 共 −4.8%** —— FP8 在 B200 上仅能反超 MI355 BF16 cf=1.25 peak (1,416)，未能反超 MI355 BF16 cf=1.0 peak。FP8 在 DS3-671B 上为净增益；在 Kimi-K2-1T 上为净损失（4168 BF16 = 248 vs 4169 FP8 = 226，**−8.9%**）—— 两个模型都走 dense_matmul 路径，符号反转的根因尚未做 profile 定位（候选包括 FP8 quantize/dequantize 在不同 MoE 拓扑下与 GEMM 加速的相对权重、模型层数 × 专家数对每步 quant 开销的放大）。
2. **`cf=1.0` 是 BF16 单个最大优化：+11.8% Tok/s/dev**（cf=1.0 bs=7 = 1,209 vs cf=1.25 bs=7 = 1,081）—— 减小 dispatch padding 同时节省显存与计算；推荐作为 dense BF16 默认值。
3. **`mem97` 解锁的 pdbs 跃迁是 BF16 的第二大调优旋钮**：把 cf=1.25 ceiling 从 pdbs=7 (1,183 Tok/s/dev) → pdbs=8 (1,167 + overlap4 = 1,167 单卡 / +14% 全局 batch)、cf=1.0 ceiling 从 pdbs=8 (1,242) → pdbs=9 (**1,272**, +2.4% 单卡 / +12% 全局)、cf=2.0 ceiling 从 pdbs=5 (864) → pdbs=7 (**913**, +5.7%)、cf=4.0 ceiling 从 pdbs=2 (432) → pdbs=3 (**466**, +7.9%)。但 `mem97 + overlap4 + 大 cf` 会触发 IB 错误（4412 / 4416 / 4424 / 4409），属于 B200 cluster 在 buffer 压力下的网络边界。
4. **`xla_gpu_experimental_parallel_collective_overlap_limit=4` 是 BF16 单个最大 XLA flag 增益**：Job 4238 = **1,289 Tok/s/dev = +9.0% vs NV defaults 1,183**（也是 BF16 cf=1.25 单 pdbs 全场最佳）；提升到 `overlap=8` (Job 4239) 退化到 1,228 (+3.8%)。
5. **`capacity_factor` 强支配 max pdbs 与 Tok/s/dev**：cf=1.0 best = 1,272 → cf=1.25 best = 1,289 (因 pdbs=7 + overlap4 略胜) → cf=2.0 best = 913 (−29% vs cf=1.0) → cf=4.0 best = 466 (−63% vs cf=1.0)。每翻倍 cf，pdbs ceiling 约减半，Tok/s/dev 约腰斩。
6. **NV defaults vs AMD-parity 稳定 +7~10% Tok/s/dev**：cf=1.25 = +9.4% (1524 1,183 vs 1200 1,081)、cf=2.0 = +7.7% (4164 864 vs 4163 802)、cf=4.0 = +7.7% (4176 432 vs 1596 401) —— flag 集合差异跨 cf 通用。
7. **`sparse_matmul + shardy` 在 8N B200 上不可行**（即使 pdbs=1 也差 ~10 GiB），失败发生在 XLA planner 的 shape-based 可行性检查阶段，runtime allocator level 的所有 workaround（slop95 / mem95）均 byte-identical 失败。Tok/s/dev = 不可测 / `不可行`。**与 MI355 报告里讨论的 `sparse-gmm-*` / DeepEP 系列变体在本镜像下未携带**，不构成 B200 的可比维度。生产 sparse MoE 在 B200 + JAX 26.03 上需要先解决 Shardy 的 buffer-planning 病态。
8. **若干"省显存" flags 在 DS3-671B 上无效或反向**：`optimizer_memory_host_offload=True` 让显存更大 (1203)，`shard_exp_on_fsdp=True` 把 args 推到 492 GiB > limit (1204)，`grad_dtype=bfloat16` 零节省 (1237)，`num_vocab_tiling=4` 与模型 dtype 不兼容 (1236, 1238)，`remat_policy=minimal_flash` 飙到 744 GiB (1211)。这些都没能解锁更高 pdbs，Tok/s/dev 上对应零增益或负增益；`mem97` 是唯一稳定有效的扩 pdbs 调优旋钮。
9. **Combine threshold 在 B200 上趋势与 AMD 部分文献相反**：Tok/s/dev 从 256 MiB → 4 GiB 单调退化 −1.1% → −8.2%（4240 1,170 → 4241 1,141 → 4242 1,139 → 4243 1,128 → 4244 1,086，全对照 NV defaults 1,183）；与 256 B 等效的 NV defaults 是最优。AMD-parity 默认的 8 GiB 是该路径上的全局最差点（−8.2% Tok/s/dev）。
10. **Triton GEMM 不可用**：`xla_gpu_enable_triton_gemm=true` 让 Tok/s/dev 从 1,081 (Job 1200) 暴跌到 **285 (Job 1526) = −73.6%**，step time 飙到 100s+；cuBLAS 在 DS3-671B 的 GEMM shape 上完胜 Triton 生成内核。**永不使用**。

---

## 与 MI355 参考报告的对照

参考报告：
- [`deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md)（MI355 主 pdbs 扫描，镜像默认 XLA）—— 提供 cf=1.25 / cf=2.0 / cf=4.0 / cf=1.0 全 pdbs 谱线。
- [`pp-vs-fsdp-deepseek3-671b.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/pp-vs-fsdp-deepseek3-671b.zh.md)（MI355 PP vs FSDP + XLA flag 调优）—— 提供 dense-cf1.25 与 sgd-v3 在 **EP+FSDP（FSDP=8）** 与 **PP+EP（PP=8）** 两种 DCN 拓扑下的最佳调优配方。

每个 cf 块给出：(1) B200 当前 cf 的 BF16 best、(2) MI355 在**同 pdbs** 下的多档成绩（untuned + EP+FSDP 调优 + PP+EP 调优，如有）、(3) MI355 该 cf 的 peak。MI355 列均取 BF16、1-node/proc 启动器。

> MI355 的 pdbs 网格是 `{1, 2, 4, 5, 6, 7, 8, 16}`，**跳过了 3 和 9** —— B200 上分别在 `dense-cf4 pdbs=3` 与 `dense-cf1 pdbs=9` 命中这两个空档，对应行标注 *n/a*。
>
> **PP+EP / EP+FSDP 调优只覆盖 `dense-cf1.25` 与 `sgd-v3`（sparse-gmm-deepep-v3）**。MI355 的 cf=2.0 / cf=4.0 没有调优数据，仍只能用 untuned 行对照。

### `dense-cf1.25`（BF16, `capacity_factor=1.25`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**                              | B200  | **1,289** |  7 | NV + overlap4 (Job 4238)                                       | 322.8 | 14.35 | baseline |
| MI355 同 pdbs=7（image default XLA）       | MI355 |   1,086   |  7 | FSDP=8 + 镜像默认 XLA, 1-node/proc                              | 272.1 | 10.88 | **−15.7%** |
| **MI355 同 pdbs=7（EP+FSDP XLA 调优）** ⭐  | MI355 | **1,208** |  7 | FSDP=8 + `ag_combine=1 GiB` XLA flag (Job 14629)              | ≈302.6 | ≈12.10 | **−6.3%** |
| **MI355 同 pdbs=7（PP+EP 调优）** ⭐⭐       | MI355 | **1,224** |  7 | PP=8 + `overlap_limit=2` (Jobs 14672/14673 平均, n=2)         | ≈306.6 | ≈12.27 | **−5.0%** |
| **MI355 peak**（untuned）                  | MI355 | **1,416** | 16 | FSDP=8 + 镜像默认 XLA, 1-node/proc                              | 354.7 | 14.19 | **+9.9%** |

- **B200 single-pdbs 对 MI355 untuned 优势 +18.7%（1,289 vs 1,086），对 MI355 EP+FSDP XLA 调优后缩到 +6.7%（1,289 vs 1,208），对 MI355 PP+EP 调优后再缩到 +5.3%（1,289 vs 1,224）。** MI355 调优把"+19% 的吞吐差"吃掉一大半 —— B200 在同 pdbs 上仍领先，但幅度从 ~一档（19%）压到 ~jitter 量级（5%）。
- 关键 enabler 对比：
  - **MI355 EP+FSDP 调优旋钮 = `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`**（把镜像默认 8 GiB 单次 all-gather 拆成 ~4–5 个 1 GiB chunk，回收 ~3 秒/步的暴露通信，单 flag 即给 **+20.5% Tok/s/dev**：1,002.4 → 1,207.9 在 FSDP=8 image-default baseline 之上）。
  - **MI355 PP+EP 调优旋钮 = `--xla_gpu_experimental_parallel_collective_overlap_limit=2`** + **取消 EP+FSDP 的 `ag=1 GiB` 覆盖**（恢复 ag=8 GiB 镜像默认；PP=8 拓扑下 ICI all-gather + DCN collective-permute 双 fabric 并发；单 flag 在 PP=8 image-default 之上 **+5.8% Tok/s/dev**：1,156.9 → 1,224.1。与 PP=8 baseline-ag1G 1,080.2 相比为 +13.3%）。
  - **B200 对应的调优旋钮 = `overlap_limit=4`**（NV defaults 之上 +9.0% Tok/s/dev：1,183 → 1,289，Job 4238）—— XLA flag 符号一致（让并发 in-flight collectives 增加），但**甜点不同**（MI355 PP=8 = 2，B200 = 4），反映两平台 ICI + DCN fabric 利用率特征差异；MI355 FSDP=8 上 `overlap_limit=2/4/8` 全部退化 −4~−5%（ICI 已经被 ag 吃满），与 PP=8 上 +5.8% 完全符号反转。
- **PP+EP vs EP+FSDP 在 MI355 dense-cf1.25 上 PP=8 微胜**：1,224 (PP+EP) > 1,208 (EP+FSDP) = +1.34%。**但在 MI355 sgd-v3 上结构反转**：FSDP=8 tuned 1,135.7 > PP=8 tuned 999.8 = +13.6%，差距来自 DeepEP per-microbatch 开销 (~1.3s/step) + nn.scan carry-dep 让 collective-permute `is_pipelined=false` + pipeline bubble = 7/63 = 11.1%。即"PP+EP 赢 FSDP" 只在 dense_matmul 分支成立，sparse_matmul-DeepEP 分支永远是 FSDP 赢。
- B200 vs MI355 peak（pdbs=16 untuned）仍 −9.0% —— peak 落差全部来自 pdbs ceiling（B200 max=8 vs MI355 max=16），与单卡吞吐无关。**MI355 在 pdbs=16 上 + 调优**未实测（PP-vs-FSDP 报告 sweep 全部固定 pdbs=7），无法定量该上限。
- 单卡 MFU 14.35% (B200) > 14.19% (MI355 peak) —— B200 的 BF16 peak (2,250) 比 MI355 (2,500) 低 10%，但实际 TFLOP/s 利用率反而略胜，与 MFU 同档（MI355 pdbs=7 tuned MFU 仅 ~12.27%，B200 单卡领先约 +17% MFU 同 pdbs）。

### `dense-cf1`（BF16, `capacity_factor=1.0`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | 备注 |
|---|---|---:|---:|---|---:|---:|---|
| **B200 best**                    | B200  | **1,272** |  9 | NV + mem97 (Job 4405)                       | 318.4 | 14.15 | mem97 解锁 pdbs=9，BF16 全场最高 MFU |
| **MI355 peak (cf=1.0)** ⭐      | MI355 | **1,598** | 16 | FSDP=8 + 镜像默认 XLA, 1-node/proc          | **400** | **16.01** | **B200 cf=1 best (1,272) −20.4%** vs MI355 cf=1.0 peak |
| MI355 同 pdbs=16, cf=1.25 baseline 参考 | MI355 |   1,418   | 16 | FSDP=8 + 镜像默认 XLA                       | 354   | 14.20 | cf=1.0 vs cf=1.25 同 pdbs=16 提升 +12.7% Tok/s/dev / +12.8% MFU |

- **MI355 cf=1.0 在 pdbs=16 上实测达到 1,598 Tok/s/dev**，对应 MFU 16.01% —— 是当前已知 DS3-671B BF16 MoE 的最高 MFU。
- **B200 cf=1 best (1,272 @ pdbs=9) vs MI355 cf=1.0 peak (1,598 @ pdbs=16) = −20.4%**。peak 差距同时来自 pdbs ceiling（B200 max=9 vs MI355 max=16）与单卡 MFU 落后（14.15% vs 16.01%）。
- **B200 cf=1 best MFU (14.15%) 低于 MI355 cf=1.0 peak MFU (16.01%)**，差距 −1.86 pp 是 B200 在 cf=1.0 路径上**唯一**落后于 MI355 的 single-kernel 利用率维度（cf=1.25 / cf=2.0 / cf=4.0 上 B200 同 pdbs MFU 均领先 MI355）。MI355 cf=1.0 实测点仅有 pdbs=16，未做小 pdbs 扫描，无法做严格同 pdbs 单卡比较。
- 反观 FP8：B200 FP8 cf=1.0 pdbs=8 (Job 4170) = 1,521 Tok/s/dev，比 MI355 cf=1.0 peak 1,598 低 **−4.8%**（FP8 BF16-eq MFU 16.92% 略高于 MI355 cf=1.0 16.01%，但 pdbs 仍差 2 档：B200=8 vs MI355=16）—— **FP8 未能让 B200 反超 MI355 BF16 cf=1.0 peak**，原因仍是 HBM 不足导致 pdbs ceiling 受限。

### `dense-cf2`（BF16, `capacity_factor=2.0`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**          | B200  | **913**   |  7 | NV + overlap4 + mem97 (Job 4423) | 228.7 | 10.16 | baseline |
| MI355 同 pdbs=7        | MI355 |   884     |  7 | 1-node/proc                     | 221.3 |  8.85 | **−3.2%** |
| **MI355 peak**         | MI355 | **968**   | 16 | 1-node/proc                     | 242.4 |  9.70 | **+6.0%** |

- 同 pdbs=7 上 B200 比 MI355 快 **+3.3%**（913 vs 884，MFU 10.16% vs 8.85%）—— B200 单卡仍领先，但优势从 cf=1.25 的 +18.7% 大幅收窄。两平台 MFU 同时下滑（B200 14.35% → 10.16%，−4.19 pp；MI355 10.88% → 8.85%，−2.03 pp），且 B200 下滑更大；dense_matmul 路径下 cf 只影响 dispatch / combine einsum 与 dense GEMM 的 capacity 维度，不引入 all-to-all。MI355 收窄差距的可能原因（未经 profile 证实）是其 1.6× HBM 容量在 cf=2.0 dispatch buffer 放大时给编译器留出更多 overlap window；要定性需要单独的 cf=1.25 vs cf=2.0 profile 对照。
- MI355 peak 利用更大的 pdbs=16 反超 +6.0%；B200 在 pdbs=7 已经触顶（pdbs=8 即使 mem97+overlap4 也会触发 IB 错误）。

### `dense-cf4`（BF16, `capacity_factor=4.0`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**          | B200  | **466**   | 3 | NV + overlap4 + mem97 (Job 4419) | 116.6 | 5.18 | baseline |
| MI355 同 pdbs=3        | MI355 | *n/a*     | 3 | *n/a*                           | —     | —    | MI355 网格跳从 2 到 4 |
| MI355 邻近 pdbs=2      | MI355 |   374     | 2 | 1-node/proc                     | 93.6  | 3.74 | B200 同 pdbs=2 (Job 4176) = 432 → **+15.5%** vs MI355 同 pdbs |
| MI355 邻近 pdbs=4      | MI355 |   500     | 4 | 1-node/proc                     | 125.2 | 5.01 | B200 pdbs=4 OOM；MI355 +7.3% vs B200 best |
| **MI355 peak**         | MI355 | **566**   | 8 | 1-node/proc                     | 141.7 | 5.67 | **+21.5%** vs B200 best |

- 同 pdbs=2 上 B200 比 MI355 快 **+15.5%**（432 vs 374）—— B200 NV defaults 的 flag 优势在 cf=4 上一致存在。
- 但 cf=4 上 MI355 max pdbs = 8 而 B200 max pdbs = 3（mem97 解锁），ceiling 落差 2.7× —— peak 上 MI355 领先 +21.5%。
- cf=4 是 B200 受 HBM 限制最严重的 dense 配置（dispatch 张量 ~ pdbs × cf 线性放大）。

### 跨 cf BF16 综合对比

| Config | B200 best (pdbs) | MI355 同 pdbs untuned (TGS) | MI355 同 pdbs **tuned** (TGS, 配方) | MI355 peak (pdbs, TGS) | B200 vs MI355 untuned 同 pdbs | B200 vs MI355 **tuned 同 pdbs** | B200 vs MI355 peak |
|---|---|---|---|---|---:|---:|---:|
| `dense-cf1.25` | **1,289** (7) | 1,086 (pdbs=7) | **1,224** (pdbs=7, PP=8+overlap2) ⭐ / 1,208 (pdbs=7, FSDP=8+ag1G) | 1,416 (pdbs=16) | **+18.7%** | **+5.3%（PP+EP）/ +6.7%（EP+FSDP）** | **−9.0%** |
| `dense-cf1`    | **1,272** (9) | n/a | n/a | **1,598** (pdbs=16, FSDP=8 image default) ⭐ | — | — | **−20.4%** |
| `dense-cf2`    | **913** (7)   | 884 (pdbs=7) | n/a | 968 (pdbs=16) | **+3.3%** | — | **−5.7%** |
| `dense-cf4`    | **466** (3)   | 邻近 374 (pdbs=2)、500 (pdbs=4) | n/a | 566 (pdbs=8) | **+15.5%**（对照 pdbs=2） | — | **−17.7%** |

dropless（sparse_matmul）路径补充对照：

| Path | B200 | MI355 best Tok/s/dev (pdbs, 配置) | 备注 |
|---|---|---|---|
| `sparse-gmm-deepep-v3` (sgd-v3) | **不可行**（sparse_matmul pdbs=1 即 OOM；镜像不带 DeepEP/Primus-Turbo） | **1,135.7** (pdbs=7, FSDP=8 + `ag_combine=1 GiB` XLA 调优, Job 14602)；999.8 (pdbs=7, PP=8 + overlap2+async) | MI355 EP+FSDP 调优后达到当前 dropless 路径最佳；PP=8 在 sgd-v3 上结构性 −12%（DeepEP per-microbatch + nn.scan carry + bubble 11.1%） |
| `sparse-gmm-fixed` (sgf) | **不可行**（同上） | **OOM** (`ragged_all_to_all` 物化 num_ranks × tokens × hidden 接收缓冲区，pdbs=7 即 217 GiB temp) | 在 MI355 上 sgf 已被 sgd-v3 取代为不可用路径 —— B200 与 MI355 在该列对齐为不可用 |

**关键观察：**

1. **同 pdbs B200 vs MI355 在 cf=1.25 上 untuned 领先 +18.7%、调优后压缩到 +5%；cf=2.0 上同 pdbs=7 领先压缩到 +3.3%；cf=4.0 上同 pdbs=2 仍领先 +15.5%。** cf=1.0 上 MI355 仅有 pdbs=16 实测（1,598，MFU 16.01% 是当前已知 DS3-671B BF16 MoE 最高 MFU），缺少小 pdbs 扫描，无法做严格同 pdbs 比较；但 MI355 cf=1.0 peak MFU (16.01%) 高于 B200 cf=1 best MFU (14.15%)，是 B200 在 BF16 路径 single-kernel 利用率上**唯一明确落后**MI355 的 cf 配置。
2. **MI355 peak 反超 B200 peak 在所有 cf 上一致存在**：cf=1.25 上 B200 −9.0%、cf=2.0 上 B200 −5.7%、cf=4.0 上 B200 −17.7%，**cf=1.0 上 B200 −20.4%**（B200 cf=1 best 1,272 vs MI355 cf=1.0 peak 1,598）—— cf=1.0 是 B200 在 BF16 路径上落后最严重的 cf 配置。peak 差距的根因主要是 pdbs ceiling（B200 max=9 vs MI355 max=16；HBM 179 vs 288 GiB），但 cf=1.0 上"MI355 单卡 MFU 也胜"叠加进来，让本来主要由 pdbs ceiling 决定的 peak 差距进一步放大。
3. **B200 反超 MI355 peak 的唯一手段是 FP8 —— 但此前 cf=1.0 实测数据公开后，FP8 也未能完全反超 MI355 BF16 cf=1.0 peak**：B200 FP8 cf=1.0 pdbs=8 (Job 4170) = **1,521 Tok/s/dev vs MI355 BF16 cf=1.0 peak 1,598 = −4.8%**；但仍高于 MI355 BF16 cf=1.25 调优 peak (1,224, +24.2%) 与 MI355 BF16 cf=1.25 untuned peak (1,416, +7.4%)。FP8 BF16-eq MFU 16.92% 仍是 B200 BF16-eq 单项最高，**但绝对 Tok/s/dev 仍受 HBM 上限拖累**（B200 FP8 cf=1.0 max pdbs=8 vs MI355 BF16 cf=1.0 max pdbs=16，全局 batch 差 2×）。**结论修订：FP8 让 B200 反超 MI355 cf=1.25 peak，但还不足以反超 MI355 cf=1.0 peak**；要彻底反超 MI355 BF16 路径，需要 B200 也能扩到 pdbs ≥ 12（当前 FP8 cf=1.0 max pdbs=8 是硬上限，pdbs=9 在 slop95 / mem97 下都过不去）。
4. **B200 dropless 路径缺失 vs MI355**：MI355 上 sgd-v3 经 EP+FSDP 调优后跑出 **1,135.7 Tok/s/dev**（Job 14602），是当前 MI355 + JAX MoE 路径下生产推荐配方。B200 上 `sparse_matmul` 在 pdbs=1 即 OOM（XLA planner shape 检查失败），且本镜像不携带 Primus-Turbo / DeepEP 路径，**B200 没有可比 dropless 列**。生产场景需要 dropless（数值收敛敏感）时，B200 + JAX 26.03 镜像还不是可行平台。
5. **MI355 上 PP=8 vs FSDP=8 拓扑选择是 path × branch 决定的**：dense-cf1.25 PP=8 调优 (1,224) **超过** FSDP=8 调优 (1,208) 约 +1.3%；但 sgd-v3 PP=8 (999.8) 比 FSDP=8 (1,135.7) 低 −12.0%（DeepEP per-microbatch + nn.scan carry + bubble 11.1% 三项叠加结构性差距）。B200 上未扫描 PP=8 拓扑，但 B200 IB 在 mem97 + 大 buffer 时已经触发 RDMA 错误（Jobs 4412 / 4416 / 4424 / 4409），暗示 B200 InfiniBand 在 EP > ici 区域可能比 MI355 Pensando 更脆弱 —— PP=8 在 B200 上是否成立、是否同样符号反转，是未来扫描的待补维度。
6. **关键 enabler 的等价对应（更新版）**：

   | 平台 / 路径 | 关键 enabler 1 | 关键 enabler 2 | 单 enabler Tok/s/dev 提升 |
   |---|---|---|---|
   | MI355 dense FSDP=8 | `ag_combine_threshold=1 GiB`（拆解 8 GiB 默认） | *(单 flag 即接近天花板)* | **+20.5%** (1,002.4 → 1,207.9) |
   | MI355 dense PP=8   | `overlap_limit=2`（2-fabric 并发, ICI+DCN p2p）+ 取消 EP+FSDP 的 ag=1 GiB | *(其他 flag jitter 内)* | **+5.8%** vs PP image default (1,156.9 → 1,224.1)；+13.3% vs PP baseline-ag1G (1,080.2) |
   | MI355 sgd-v3 FSDP=8 | `ag_combine_threshold=1 GiB`         | `MAXTEXT_PATCH_BRANCH=…-v3`（消掉 `input_scatter_fusion_*.kd` 主流阻塞） | ag-flag **+11.6%** (1,017.7 → 1,135.7)；v1→v3 patch **+63%** dropless |
   | MI355 sgd-v3 PP=8   | `overlap_limit=2 + async_priority`   | *(结构性 bubble 11.1% + DeepEP per-microbatch 不可消)* | **+4.7%** PP=8 baseline-ag1G → 999.8 |
   | B200 BF16 dense   | `overlap_limit=4`（NV defaults 之上） | `mem97`（解锁 +1 pdbs，单卡略降但全局 batch +14%） | **+9.0%** pdbs=7 (1,183 → 1,289) / +14% global batch |
   | B200 FP8 dense    | `quantization=fp8` | `slop_factor=95` 解锁 pdbs=8 | FP8 **+22.5%** vs BF16 baseline / slop95 +3.3% pdbs=7→8 |



---

## 如何复现

```bash
# 全场最佳：FP8 cf=1.0 + pdbs=8 + slop95
./submit.sh deepseek3-671b::fp8-bs8-cf1-nv-slop95 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=8 capacity_factor=1.0 quantization=fp8 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=95'

# BF16 单 flag 最佳：cf=1.25 + pdbs=7 + overlap_limit=4
./submit.sh deepseek3-671b::bf16-tune-overlap4 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=7 capacity_factor=1.25 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_experimental_parallel_collective_overlap_limit=4'

# BF16 cf=1.0 + 大 pdbs 解锁：pdbs=9 + mem97
./submit.sh deepseek3-671b::bf16-bs9-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=9 capacity_factor=1.0 \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# BF16 cf=2.0 max pdbs：mem97 + overlap4
./submit.sh deepseek3-671b::bf16-cf2-bs7-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=7 capacity_factor=2.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_experimental_parallel_collective_overlap_limit=4' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'
```

---

## 参考

- 原始数据：[`docs/b200-benchmark-report.md`](b200-benchmark-report.md)（按 Run / Job 的时间顺序记录；本文档按 (precision, config) 切片重排）
- MI355 同形扫描：[`deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md)（结构与术语来源；本文档跟随其 dense-cf{1.25,1,2,4} + sparse 五分类切法）
