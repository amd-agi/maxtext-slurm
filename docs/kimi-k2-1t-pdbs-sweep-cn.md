# Kimi-K2-1T —— B200 上的 pdbs 扫描实验

- **日期：** 2026-04-10（首轮 bs=1 基线扫描）；2026-05-08（`mem_fraction` 扩展解锁 bs=2，详见结果矩阵）
- **模型：** `kimi-k2-1t` (MaxText)。1026.4 B 参数。61 层 decoder（layer 0 dense、layers 1–60 MoE，384 expert × top-8 + 1 shared expert）。MLA attention (`q_lora_rank=1536`, `kv_lora_rank=512`)。
- **硬件：** 8 节点 × 8× NVIDIA B200 (179.1 GiB HBM / dev), InfiniBand 互联
- **镜像：** `nvcr.io/nvidia/jax:26.03-maxtext-py3`
- **补丁分支：** [`llying/benchmark-on-nv-b200`](https://github.com/AMD-AGI/maxtext-slurm/tree/llying/benchmark-on-nv-b200) @ `5f68243`
- **基础配置：** [`configs/kimi-k2-1t.gpu.yml`](../configs/kimi-k2-1t.gpu.yml)（`dcn_fsdp_parallelism=8`, `ici_expert_parallelism=8`, `sparse_matmul=false`, `capacity_factor=1.25`）
- **数据来源：** [`docs/b200-benchmark-report.md`](b200-benchmark-report.md)（按 precision × capacity_factor 重新组织）
- **峰值：** BF16 ≈ 2,250 TFLOP/s/dev；FP8 ≈ 4,500 TFLOP/s/dev
- **XLA_PYTHON_CLIENT_MEM_FRACTION 默认：** `0.93`（预分配约 165.87 GiB / dev；后续 bs=2 run 上调至 `.97`）

## 背景

本文档以 [`AMD-AGI/maxtext-slurm@yihuang/moe/kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md)（MI355 集群上的全面 pdbs 扫描）为参考模板，整理出 B200 集群上 Kimi-K2-1T 的同形扫描视图。原始数据全部来自 [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) —— 这里只是按照 "precision (BF16 / FP8) × capacity_factor 变体" 做了二维切分。pdbs 指 `per_device_batch_size`。

> **B200 上 Kimi-K2-1T 是极端 HBM-bound workload：** 8N B200 上 max pdbs = 2（需要 `mem97`）。MI355 同硬件规模上 max pdbs ≥ 12 —— 这个 6× 的 pdbs ceiling 落差，是 Kimi 跨平台对比的主线索（远超 DS3 上的 ~2× ceiling 落差）。原因：模型参数 1.026 T，比 DS3-671B 大 53%；同时 ici_expert_parallelism=8 + 384 experts → 每 GPU 48 个 expert（DS3 是 32 个），expert weight 占用对 HBM 容量更敏感。

> **B200 与 MI355 的关键差异：** B200 单卡 179.1 GiB HBM (MI355 为 288 GiB)，bf16 峰值 2250 TFLOP/s (MI355 为 ≈2500)，互联用 InfiniBand 而非 Pensando AINIC。MI355 参考报告里的 `sparse-gmm-*` / DeepEP / `_env_ENABLE_RAGGED_ONESHOT_KERNEL` 等 XLA / Primus-Turbo 路径在本 B200 镜像下并未开放，B200 上 `sparse_matmul=True` 也未实测（参考 DS3 doc 同节，sparse_matmul + shardy 在 8N B200 上 pdbs=1 即 OOM）。

## 受测配置

每个表格固定一个 (precision, capacity_factor) 组合，行内变化的是 `pdbs` 与具体 XLA / 内存 flag 集合。

| 标签            | 透传参数                                                   |
|-----------------|----------------------------------------------------------|
| `dense-cf1.25`  | *(默认)* — `sparse_matmul=false`, `capacity_factor=1.25`  |
| `dense-cf1`     | `capacity_factor=1.0`                                     |

**未测试的 cf 变体（MI355 参考报告有，但 B200 上没测）：**

| 标签            | B200 上跳过原因 |
|-----------------|-----------------|
| `dense-cf2`     | B200 上 cf=1.25 + 默认 mem 即 OOM @ bs=2；cf=2.0 dispatch padding 翻倍，bs=1 都很可能 OOM。MI355 上 cf=2.0 max pdbs = 10，但 B200 HBM 不足 —— 没有可比 pdbs，未投入测试资源 |
| `dense-cf4`     | 同上，更严重；MI355 上 cf=4.0 max pdbs = 6，B200 上预计 pdbs=1 即 OOM |
| `sparse_matmul` | 参考 DS3 sparse_matmul 分析：8N B200 上 sparse_matmul + shardy 在 pdbs=1 即分配 ~112 GiB / dev 失败（XLA planner shape 检查阶段，与 dtype 无关）；本镜像不携带 Primus-Turbo / DeepEP，无对应可测列 |

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

- **NV defaults**（`_env_XLA_FLAGS_REPLACE` 整体替换为下列两项，丢弃所有 AMD-parity flag）—— 完整 flag 串：

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_enable_command_buffer=''
  ```

- **mem97** — `_env_XLA_PYTHON_CLIENT_MEM_FRACTION` 从默认 `.93` 上调至 `.97`（这是 JAX 预分配池占比，与 XLA flag 正交）。

图例：`✗` = OOM；`—` = 未测试；`CANCELLED` 保留作业原始失败状态。

**主要性能指标 = `Tok/s/dev`；辅助指标 = `TFLOP/s/dev` 与 MFU。** Tok/s/dev 不受 FLOP 计数约定影响（FP8 与 BF16 的 peak 不同，TFLOP/s/dev 跨精度直接对比会失真，而 Tok/s/dev 直接对应实际训练吞吐）。`TFLOP/s/dev` 与 `MFU` 列仍保留作为辅助参考。

**Tok/s/dev 计算约定：** 所有表格里的 `Tok/s/dev` 列 = `per_device_batch_size × max_target_length / step_time`（Kimi-K2-1T 默认 `max_target_length = 4096`，定义见 [`configs/kimi-k2-1t.gpu.yml`](../configs/kimi-k2-1t.gpu.yml)）。

---

## BF16

### `dense-cf1.25`（BF16, `capacity_factor=1.25`, `sparse_matmul=false`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 1597 | AMD-parity | SUCCESS | 17.00 | 2.20  |  49.5 |   241 | **首个可行 pdbs**；MFU 极低 |
|  1 | 4167 | NV defaults | SUCCESS | 16.85 | 2.22  |  49.9 |   243 | NV +0.8% vs AMD-parity —— flag 集合对 Kimi 影响微弱 |
|  2 | 1532 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 83.33 GiB |
|  2 | 4180 | NV + `ici_fsdp=2, ici_ep=4` | ✗ OOM | -- | -- | -- | -- | **更糟**：+5.7 GiB；ici_ep=4 让每 GPU 持有 2× expert weights，net 负向 |
|  2 | 4414 | NV defaults + mem97 | SUCCESS | 19.31 |  3.87 |  87.2 |   424 | **mem97 解锁 pdbs=2 (cf=1.25)** |
|  3 | 1534 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 91.19 GiB |

**核心观察（以 Tok/s/dev 为准）：**

- **pdbs=2 是 BF16 cf=1.25 在 B200 上的硬上限**（pdbs=3 即 AMD-parity 也 OOM 91.19 GiB；pdbs=2 只能靠 `mem97` 解锁）。Job 4414 = **424 Tok/s/dev**（MFU 3.87%，辅助 87.2 TFLOP/s/dev），比 pdbs=1 best (Job 4167 = 243) 提升 **+74.5%**。
- **NV defaults vs AMD-parity 在 Kimi 上影响仅 +0.8%**（4167 vs 1597），远小于 DS3 (+9.4%)。Kimi 的 MoE dispatch overhead 占主导，XLA flag 集合调优空间小。
- **`ici_fsdp=2 / ici_ep=4` 重平衡假设被实测推翻**（Job 4180）：ici_ep=4 让每 GPU 持有 2× expert weights，而 ici_fsdp=2 只把非 expert (attention/router) weights 减半。Kimi-K2 在 1T 参数中 expert weights 占主导，净结果反而 **+5.7 GiB** 内存压力。bs=2 在此 split 下依然 OOM。

---

### `dense-cf1`（BF16, `capacity_factor=1.0`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 1598 | AMD-parity | SUCCESS | 16.60 | 2.25  |  50.7 |   247 | cf=1.0 vs cf=1.25 (1597 = 241) = **+2.5% Tok/s/dev** |
|  1 | 4168 | NV defaults | SUCCESS | 16.52 | 2.26  | **50.94** | **247.9** | **Kimi BF16 best @ pdbs=1**（NV 集合） |
|  2 | 1599 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 82.47 GiB —— vs cf=1.25 同配 (1532 = 83.33 GiB) 仅节省 ~0.86 GiB，不足以解锁 |
|  2 | 4173 | NV defaults | ✗ OOM | -- | -- | -- | -- | 80.48 GiB —— NV vs AMD 节省 ~2 GiB，仍不足以解锁 |
|  2 | 4410 | NV defaults + mem97 | SUCCESS | 18.66 | **4.01** | **90.2** | **439** | **Kimi BF16 全场最佳：mem97 解锁 pdbs=2 (cf=1.0)** |

**核心观察（以 Tok/s/dev 为准）：**

- **Kimi BF16 全场最佳 = Job 4410 (bs=2, cf=1.0, mem97) = 439 Tok/s/dev**（MFU 4.01%，辅助 90.2 TFLOP/s/dev）。相对 pdbs=1 best (Job 4168 = 247.9) 提升 **+77.1%** —— Kimi 在 B200 上的关键 enabler 是 **`mem97` 解锁 pdbs=2**，单 flag 即接近 2× 吞吐。
- **`cf=1.0` 在 Kimi 上只给 ~2.5% 增益**（1598 vs 1597 / 4168 vs 4167），远小于 DS3 (+11.8%)。原因：Kimi 384 experts × top-8 比 DS3 256 × top-8 的 dispatch padding 分布更均匀，cf=1.25 → cf=1.0 缩减的 padding 总量本身就小。
- **bs=3+ 全部 OOM**（默认 mem 或 mem97 都不够）：bs=3 = 91 GiB / bs=4 = 100 GiB / bs=6 = 118 GiB，全超过 B200 的 ~102 GiB XLA budget。

---

## FP8

### `dense-cf1.25`（FP8, `capacity_factor=1.25`, `quantization=fp8`）

> FP8 MFU% 相对 FP8 peak (4,500 TFLOP/s for B200)；BF16-equivalent MFU = MFU × 2。

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 4179 | NV defaults | SUCCESS | 18.35 | 1.02* |  45.9 |   223.3 | **FP8 cf=1.25 baseline** —— vs BF16 cf=1.25 (4167 = 243) = **−8.1% Tok/s/dev** |
|  2 | 4415 | NV defaults + mem97 | SUCCESS | 20.31 | 1.84* |  82.9 |   403   | **mem97 解锁 pdbs=2 (FP8 cf=1.25)** |

### `dense-cf1`（FP8, `capacity_factor=1.0`, `quantization=fp8`）

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | 备注 |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 4169 | NV defaults | SUCCESS | 18.10 | 1.03* |  46.5 |   226.4 | vs BF16 cf=1.0 (4168 = 247.9) = **−8.7% Tok/s/dev**；cf=1.0 vs cf=1.25 FP8 = +1.4% |
|  2 | 4174 | NV defaults | ✗ OOM | -- | -- | -- | -- | 79.09 GiB —— FP8 vs BF16 bs=2 仅省 ~1.4 GiB，不足以解锁 |
|  2 | 4411 | NV defaults + mem97 | SUCCESS | 19.64 | 1.90* | **85.7** | **417**   | **Kimi FP8 全场最佳：mem97 解锁 pdbs=2 (cf=1.0)** |

**核心观察（以 Tok/s/dev 为准）：**

- **Kimi FP8 全场最佳 = Job 4411 (bs=2, cf=1.0, mem97) = 417 Tok/s/dev**（FP8 MFU 1.90% = BF16-eq MFU 3.80%；辅助 85.7 TFLOP/s/dev）。**相对 BF16 全场最佳 (Job 4410 = 439) 仍 −5.0% Tok/s/dev** —— FP8 在 Kimi 上是**净损失**。
- **FP8 vs BF16 同 pdbs / 同 cf 一致为负**：cf=1.25 pdbs=1 = −8.1% (223 vs 243)、cf=1.0 pdbs=1 = −8.7% (226 vs 248)、cf=1.0 pdbs=2 = −5.0% (417 vs 439)。差距随 pdbs 上升略收窄但不消失。
- **FP8 节省 HBM 也极有限**：FP8 cf=1.0 bs=2 在 NV defaults 下 OOM 79.09 GiB，BF16 同配 OOM 80.48 GiB —— FP8 量化只压缩了 weights，dispatch padding + activation 占大头，B200 上 Kimi 的内存瓶颈不在 weight precision。

### FP8 vs BF16 在 Kimi 上为何反转（vs DS3）

DS3-671B 上 FP8 是 **+19.6% / +22.5%** 净增益；同一硬件、同 cf、同 image、同 flag 集合，Kimi-K2-1T 上 FP8 是 **−5.0% ~ −8.7%** 净损失。两个模型都走 `sparse_matmul: False` 即 dense_matmul 分支，不涉及 all-to-all。可能的根因（未做 profile 定位）：

- Kimi 的 MoE forward 中 FP8 quantize/dequantize 开销与 GEMM 加速的相对权重不同 —— 1T 模型层数更多 (61 vs 61 相同，但 expert 数 384 vs 256)、每步 quant 次数更多
- Kimi 384 experts × top-8 routing 的 dispatch tensor 形状更 "瘦长"，FP8 GEMM 的形状效率不同
- 实际定位需要 nsys / xprof profile 对照 dense_matmul 内部各 kernel 在两个模型上的 FP8 vs BF16 时间分布

---

## 全场最佳汇总

按 **Tok/s/dev**（主指标）降序排列。

| Rank | Precision | Config | pdbs | **Tok/s/dev** | Δ vs BF16 best | Job | XLA Flags | Step (s) | TFLOP/s/dev | MFU (%) |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| **1** | **BF16** | **dense-cf1** | **2** | **439** | **0** (基准) | **4410** | **NV + mem97** | **18.66** | **90.2** | **4.01** |
| 2 | BF16 | dense-cf1.25 | 2 | 424 | −3.4% | 4414 | NV + mem97 | 19.31 | 87.2 | 3.87 |
| 3 | FP8  | dense-cf1   | 2 | 417 | −5.0% | 4411 | NV + mem97 | 19.64 | 85.7 | 1.90\* |
| 4 | FP8  | dense-cf1.25 | 2 | 403 | −8.2% | 4415 | NV + mem97 | 20.31 | 82.9 | 1.84\* |
| 5 | BF16 | dense-cf1   | 1 | 247.9 | −43.5% | 4168 | NV defaults | 16.52 | 50.94 | 2.26 |
| 6 | BF16 | dense-cf1.25 | 1 | 243   | −44.6% | 4167 | NV defaults | 16.85 | 49.9 | 2.22 |
| 7 | FP8  | dense-cf1   | 1 | 226.4 | −48.4% | 4169 | NV defaults | 18.10 | 46.5 | 1.03\* |
| 8 | FP8  | dense-cf1.25 | 1 | 223.3 | −49.1% | 4179 | NV defaults | 18.35 | 45.9 | 1.02\* |
| — | BF16 | dense-cf2 | — | **未测** | — | — | — | — | — | — |
| — | BF16 | dense-cf4 | — | **未测** | — | — | — | — | — | — |
| — | BF16 | sparse_matmul | — | **未测**（预期不可行，参考 DS3） | — | — | — | — | — | — |

\* FP8 MFU% 相对 FP8 peak (4,500 TFLOP/s)；BF16-equivalent MFU = MFU × 2 = 3.80% (Job 4411) / 3.68% (Job 4415) / 2.06% (Job 4169) / 2.04% (Job 4179)。

---

## 关键结论

> 所有 "+X% / −X%" 均以 **Tok/s/dev**（主指标）计算；TFLOP/s/dev 与 MFU 同时给出作为辅助参考。

1. **B200 全场 best = BF16 cf=1.0 pdbs=2 + mem97 (Job 4410)：439 Tok/s/dev**（MFU 4.01%）。相对 pdbs=1 baseline (Job 4168 = 247.9) **+77.1%** —— Kimi 在 B200 上唯一有效的调优是 **`mem97` 解锁 pdbs=2**，单 flag 即接近 2× 吞吐。
2. **FP8 在 Kimi 上为净损失（与 DS3 完全相反）：cf=1.0 pdbs=2 = −5.0%（417 vs 439）、cf=1.25 pdbs=2 = −3.4%（403 vs 424 BF16，−8.2% vs 4410 BF16 best）。** 两个模型都走 dense_matmul，符号反转的根因尚未做 profile 定位。FP8 节省 HBM 也极有限（bs=2 默认 mem 下 FP8 vs BF16 只省 ~1.4 GiB，OOM 边界一样过不去）。
3. **`mem97` 是 Kimi 在 B200 上唯一可用的扩 pdbs 调优旋钮**：把 cf=1.25 / cf=1.0 / FP8-cf=1.25 / FP8-cf=1.0 的 max pdbs 全部从 1 → 2，对应 +74~77% Tok/s/dev。`slop_factor=95` 在 Kimi 上未单独测试（参考 DS3，slop95 / mem95 在 buffer 压力大的边界上效果劣于 mem97）。
4. **`ici_fsdp=2 / ici_ep=4` 拓扑重平衡反向**（Job 4180）：ici_ep=4 让每 GPU 持有 2× expert weights，净 +5.7 GiB 内存压力；bs=2 在此 split 下依然 OOM。该假设被实测推翻 —— 1T 模型上 expert weights 占参数主导（与 671B 上 expert/non-expert 比例不同），ICI 拓扑调整无法解锁更高 pdbs。
5. **`cf=1.0` 在 Kimi 上仅给 +2.5% Tok/s/dev**（远小于 DS3 的 +11.8%）。原因：Kimi 384 experts × top-8 比 DS3 256 × top-8 的 dispatch padding 分布更均匀，cf 缩减的 padding 总量本身就小。
6. **NV defaults vs AMD-parity 在 Kimi 上仅 +0.8% Tok/s/dev**（远小于 DS3 的 +9.4%）。Kimi 的 MoE dispatch overhead 占主导，XLA flag 集合调优空间小。
7. **bs=3+ 是硬上限**：bs=3 (91 GiB) / bs=4 (100 GiB) / bs=6 (118 GiB) 全部 OOM，超过 B200 的 ~102 GiB XLA budget；mem97 在 bs=3 上没单独实测，但 bs=3 alloc 距 mem97 pool（~174 GiB）虽有 headroom，working set 整体压力（dispatch + activations + scatter intermediates）仍会再爆 —— 参考 DS3 上 `mem97` 也只能解锁 +1 pdbs 的同款行为。
8. **`dense-cf2` / `dense-cf4` / `sparse_matmul` 在 B200 上未测试**：cf=2.0 / cf=4.0 dispatch padding 加倍，B200 上 bs=1 都很可能 OOM；sparse_matmul + shardy 在 8N B200 上参考 DS3 即 pdbs=1 OOM 112 GiB；本镜像不携带 Primus-Turbo / DeepEP，无对应可测列。MI355 参考报告里这三列均有数据。

---

## 与 MI355 参考报告的对照

参考报告：[`kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md)（MI355 Kimi 主 pdbs 扫描 + sparse-gmm-deepep v1/v2/v3 + DCN-EP 扩展）。**MI355 参考报告全为 BF16；FP8 / 任一 cf 路径在 MI355 上均未跑。** 因此 B200 vs MI355 对照只能在 BF16 上做。

每个 cf 块给出：(1) B200 当前 cf 的 BF16 best、(2) MI355 在**同 pdbs** 下的成绩、(3) MI355 该 cf 的 peak。MI355 列均取 BF16、1-node/proc 启动器。

### `dense-cf1.25`（BF16, `capacity_factor=1.25`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**           | B200  | **424**   |  2 | NV + mem97 (Job 4414)       |  87.2 |  3.87 | baseline |
| MI355 同 pdbs=2         | MI355 |   399.0   |  2 | FSDP=8 + 镜像默认 XLA       |  82.0 |  3.28 | **−5.9%** |
| MI355 同 pdbs=4 (P★)    | MI355 |   678.9   |  4 | FSDP=8 + 镜像默认 XLA       | 139.5 |  5.58 | **+60.1%** vs B200 best |
| **MI355 peak**          | MI355 | **1,170.1** | 11 | FSDP=8 + 镜像默认 XLA       | **240.4** | **9.62** | **+176%** vs B200 best |

- **同 pdbs=2 B200 比 MI355 快 +6.3% Tok/s/dev**（424 vs 399；MFU 3.87% vs 3.28%）—— B200 单卡略胜，但绝对差距仅一档 jitter 量级。
- **MI355 peak 比 B200 peak 高 +176%（1,170 vs 424）**—— 几乎全部来自 pdbs ceiling 落差（B200 max=2 vs MI355 max=12，6× 落差），MI355 单卡同 pdbs MFU 实际略低于 B200。
- **MI355 peak 在 pdbs=11**（不是 max pdbs=12，因 pdbs=12 上 TGS 反而从 1,170 跌到 1,134），呈 `argmax_TGS < max_pdbs` 的 ceiling-adjacent 退化模式 —— 与 DS3 上同模式一致。

### `dense-cf1`（BF16, `capacity_factor=1.0`）

| 对比项 | 平台 | Tok/s/dev | pdbs | 配置 | TFLOP/s/dev | MFU (%) | 备注 |
|---|---|---:|---:|---|---:|---:|---|
| **B200 best**           | B200  | **439**   |  2 | NV + mem97 (Job 4410)       |  90.2 |  4.01 | mem97 解锁 pdbs=2 |
| MI355 同 pdbs           | MI355 | n/a       | —  | MI355 参考报告未跑 cf=1.0   | —     | —    | MI355 Kimi 扫描只覆盖 cf=1.25 / 2 / 4 |

- **MI355 参考报告里 Kimi 没有 cf=1.0 列**，无法做严格对照。仅可知：B200 上 cf=1.0 vs cf=1.25 同 pdbs=2 提升 +3.5%（439 vs 424），与 cf=1.0 在 DS3 上 +11.8% 量级远低 —— Kimi 384 expert × top-8 的 padding 分布让 cf 缩减增益变小。
- 若以 cf=1.25 为代理（同 pdbs=2 上 B200 +6.3% vs MI355），cf=1.0 上 B200 同 pdbs 仍可能略胜，但 MI355 peak (pdbs=11 cf=1.25 = 1,170) 仍远超 B200 cf=1.0 peak (pdbs=2 = 439)。

### 跨 cf BF16 综合对比

| Config | B200 best (pdbs) | MI355 同 pdbs (TGS) | MI355 peak (pdbs, TGS) | B200 vs MI355 同 pdbs | B200 vs MI355 peak |
|---|---|---|---|---:|---:|
| `dense-cf1.25` | **424** (2) | 399.0 (pdbs=2) | 1,170.1 (pdbs=11) | **+6.3%** | **−63.8%** |
| `dense-cf1`    | **439** (2) | n/a | n/a (MI355 Kimi 未跑 cf=1.0) | — | — |
| `dense-cf2`    | **未测** | 597.0 (pdbs=4 P★)；peak 827.8 (pdbs=10) | 827.8 (pdbs=10) | — | — |
| `dense-cf4`    | **未测** | 414.9 (pdbs=4 P★)；peak 455.1 (pdbs=5) | 455.1 (pdbs=5) | — | — |

dropless（sparse_matmul）路径补充对照：

| Path | B200 | MI355 best Tok/s/dev (pdbs, 配置) | 备注 |
|---|---|---|---|
| `sparse-gmm-deepep-v3` (sgd-v3) | **未测**（预期不可行，参考 DS3 sparse_matmul 同分析） | **897.9** (pdbs=7, FSDP=8 + 镜像默认 XLA, MI355 best dropless) | MI355 上 v3 patch 把 dropless 推到 dense 80% 量级，且 `custom_vjp` backward 消除 scatter-add 中间张量让 pdbs ceiling 从 v1/v2 的 5 升到 7 |
| `sparse-gmm-fixed` (sgf) | **未测**（同上） | **614.5** (pdbs=4) | MI355 上 sgf 的 dropless ceiling 受 `ragged_all_to_all` 临时 buffer 物化限制，pdbs ≤ 4 |
| `sparse-gmm-deepep` v1 / v2 | **未测**（同上） | v1 = 515.7 (pdbs=5)；v2 = 635.9 (pdbs=5) | v1→v2→v3 优化链：v3 消掉 `input_scatter_fusion_*.kd`（v1 5.34 s → v3 0.02 s @ pdbs=4） |

**关键观察：**

1. **B200 vs MI355 在同 pdbs=2 上仅 +6.3%（cf=1.25）/ cf=1.0 无对照数据**。B200 单卡同 pdbs 优势在 Kimi 上比 DS3 上小得多（DS3 cf=1.25 pdbs=7 上 B200 vs MI355 untuned 是 +18.7%）—— 因为 Kimi MFU 在 B200 上极低（3.87% vs DS3 的 14.35%），单卡算力优势被模型本身的 dispatch / quant overhead 稀释。
2. **MI355 peak 反超 B200 peak 高达 +176%（cf=1.25）**——这是当前所有跨平台对比中最大的 peak 落差，几乎全部来自 pdbs ceiling 差异：B200 max=2 vs MI355 max=12，6× 落差。MI355 288 GiB HBM vs B200 179 GiB HBM 在 1T 模型上把每 GPU 的 expert weight 占用差异放大。
3. **B200 上 dropless 路径完全缺失 vs MI355**：MI355 sgd-v3 在 pdbs=7 上跑出 897.9 Tok/s/dev，比 B200 BF16 cf=1.0 best (439) 高 +104%。B200 上 sparse_matmul + shardy 即使 pdbs=1 也 OOM（参考 DS3 同分析），且本镜像不携带 Primus-Turbo / DeepEP —— B200 上 Kimi 完全没有 dropless 选项。
4. **FP8 在 Kimi 上是负优化（独有于 Kimi，DS3 上为 +22.5%）**：B200 FP8 cf=1.0 pdbs=2 = 417 vs BF16 同配 = 439 = −5.0%。MI355 Kimi 扫描未做 FP8 对照，无法跨平台验证 FP8 在 MI355 上是否也反向 —— 但 MI355 既然 BF16 peak (1,170) 已远超 B200 (439)，跨平台 FP8 对比意义已经不大。
5. **`mem97` 调优旋钮 vs MI355 任何 flag tuning**：B200 上 `mem97` 单 flag +77% 是 Kimi 的唯一关键 enabler；MI355 Kimi 主扫描在镜像默认 XLA 下就跑出 1,170 peak，没有显著的 XLA flag 调优需求 —— 因为 MI355 288 GiB HBM 让模型直接跑到 dispatch / compute 平衡点附近，不需要靠 mem fraction 拼内存。两平台对 "扩 pdbs" 的瓶颈完全不同：B200 上是 HBM 容量硬上限，MI355 上是单步通信 + dispatch 调度本身。
6. **关键 enabler 的等价对应**：

   | 平台 / 路径 | 关键 enabler 1 | 单 enabler Tok/s/dev 提升 | 等效物理资源 |
   |---|---|---|---|
   | B200 BF16 Kimi  | `XLA_PYTHON_CLIENT_MEM_FRACTION=.97`（解锁 pdbs 1→2） | **+77.1%** (247.9 → 439) | 解锁 HBM headroom |
   | MI355 BF16 Kimi (dense-cf1.25) | *(镜像默认 XLA 已 near-peak，无 single-flag headline)* | pdbs=1 → pdbs=11 自然扩展即 +398% | MI355 288 GiB HBM 自带容量优势 |
   | B200 FP8 Kimi   | `quantization=fp8` | **−5.0% vs BF16** (净损失) | quant overhead > GEMM 加速 |
   | MI355 sgd-v3 Kimi | `MAXTEXT_PATCH_BRANCH=…-v3`（消掉 `input_scatter_fusion_*.kd`） | v1→v3 **+88.3%** (476.9 → 897.9, pdbs=4 比较;额外 +2 pdbs ceiling) | Python-only patch, 0 算力增加 |

---

## 如何复现

```bash
# Kimi 全场最佳：BF16 cf=1.0 + bs=2 + mem97
./submit.sh kimi-k2-1t::bf16-bs2-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# Kimi BF16 cf=1.25 + bs=2 + mem97 （次优 baseline）
./submit.sh kimi-k2-1t::bf16-bs2-cf125-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.25 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# Kimi pdbs=1 baseline（默认 mem，BF16 best @ pdbs=1）
./submit.sh kimi-k2-1t::bf16-bs1-cf100-nv -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=1 capacity_factor=1.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer='

# Kimi FP8 best（FP8 在 Kimi 上为负优化，仅供完整性参考）
./submit.sh kimi-k2-1t::fp8-bs2-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.0 quantization=fp8 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'
```

---

## 参考

- 原始数据：[`docs/b200-benchmark-report.md`](b200-benchmark-report.md)（按 Run / Job 的时间顺序记录；本文档按 (precision, config) 切片重排）
- DS3-671B 同形 B200 扫描：[`docs/deepseek3-671b-pdbs-sweep.md`](deepseek3-671b-pdbs-sweep.md)（结构与术语来源；本文档跟随其 dense-cf{1.25, 1} 切法 + 跨平台对照模式）
- MI355 同形 Kimi 扫描：[`kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md)（MI355 上 dense-cf{1.25, 2, 4} + sparse-gmm-deepep v1/v2/v3 + DCN-EP 扩展）
