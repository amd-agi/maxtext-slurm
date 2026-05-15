# DeepSeek-V3 671B — pdbs Sweep on B200

- **Date:** 2026-04-10 (initial baseline sweep); 2026-05-08 (XLA flag tuning + `mem_fraction` extension; see the results matrix)
- **Model:** `deepseek3-671b` (MaxText)
- **Hardware:** 8 nodes × 8× NVIDIA B200 (179.1 GiB HBM / dev), InfiniBand fabric
- **Image:** `nvcr.io/nvidia/jax:26.03-maxtext-py3`
- **Patch branch:** [`llying/benchmark-on-nv-b200`](https://github.com/AMD-AGI/maxtext-slurm/tree/llying/benchmark-on-nv-b200) @ `5f68243`
- **Base config:** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)
- **Source data:** [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) (reorganized by precision × capacity_factor)
- **Peak:** BF16 ≈ 2,250 TFLOP/s/dev; FP8 ≈ 4,500 TFLOP/s/dev
- **XLA_PYTHON_CLIENT_MEM_FRACTION default:** `0.93` (preallocates ~165.87 GiB / dev; later runs bumped to `.95 / .96 / .97`)

## Background

This document follows [`AMD-AGI/maxtext-slurm@yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md) (the full pdbs sweep on the MI355 cluster) as a structural template, organizing the B200 DeepSeek3-671B results in the same shape. All raw data comes from [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) — this doc just slices it along (precision (BF16 / FP8) × capacity_factor variant (dense-cf1.25 / dense-cf1 / dense-cf2 / dense-cf4 / sparse_matmul)). `pdbs` is shorthand for `per_device_batch_size`.

> **Key B200 vs MI355 differences:** B200 has 179.1 GiB HBM / device (MI355: 288 GiB), BF16 peak 2,250 TFLOP/s (MI355 ≈ 2,500), and uses InfiniBand instead of Pensando AINIC. As a result, the viable pdbs ceiling on B200 for the same model / config is far below MI355. The `sparse-gmm-*` / DeepEP / `_env_ENABLE_RAGGED_ONESHOT_KERNEL` XLA / Primus-Turbo paths discussed in the MI355 reference are not exposed under this image (see the `sparse_matmul` tables below).

## Configs under test

Each table fixes a (precision, capacity_factor) combination; rows vary `pdbs` and the specific XLA / memory flag set.

| Tag             | Passthrough flags                                          |
|-----------------|----------------------------------------------------------|
| `dense-cf1.25`  | *(default)* — `sparse_matmul=false`, `capacity_factor=1.25`  |
| `dense-cf1`     | `capacity_factor=1.0`                                     |
| `dense-cf2`     | `capacity_factor=2.0`                                     |
| `dense-cf4`     | `capacity_factor=4.0`                                     |
| `sparse_matmul` | `sparse_matmul=true shardy=true` (`shardy=true` is required on B200, otherwise `RaggedDot` refuses to compile) |

**XLA flag-set shorthands** (same as the b200-benchmark-report; every run appends `--xla_gpu_enable_command_buffer=''` from `train_env.sh`'s JAX-0.8.2 fix):

- **AMD-parity** (the image-default `XLA_FLAGS`, for AMD / NV cross-validation) — full flag string:

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

  Shorthand form: `slop_factor=95, reduce_scatter/all_gather_combine=8 GiB, triton_gemm=false, cublaslt=true, autotune_level=0, all_gather_combine_by_dim=false`.

- **NV defaults** (`_env_XLA_FLAGS_REPLACE` wholesale replaces the flag set with the two flags below, dropping all AMD-parity flags) — full flag string:

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_enable_command_buffer=''
  ```

  I.e. `XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=''`.

- **NV + overlap4** — NV defaults plus `--xla_gpu_experimental_parallel_collective_overlap_limit=4` appended via `_env_EXTRA_XLA_FLAGS`.
- **AMD + xxx** / **NV + xxx** (e.g. "AMD + autotune=4", "NV + overlap4", "NV + combine_by_dim=true") — overrides appended on top of the corresponding base set (AMD-parity or NV defaults) via `_env_EXTRA_XLA_FLAGS`, or extending `XLA_FLAGS_REPLACE`.
- **mem95 / mem97** — `_env_XLA_PYTHON_CLIENT_MEM_FRACTION` raised from the default `.93` to `.95 / .97` (JAX preallocation pool fraction, orthogonal to XLA flags).
- **slop95** — appends `--xla_gpu_memory_limit_slop_factor=95` (NV defaults don't have this by default; equivalent to the same-named flag in AMD-parity).

Legend: `✗` = OOM; `—` = untested; `SEGFAULT` / `IB HANG` / `IBV_WC_RETRY_EXC_ERR` etc. preserve the original Slurm job failure status.

**Primary metric = `Tok/s/dev`; auxiliary metrics = `TFLOP/s/dev` and MFU.** Every "gain / loss / speedup / regression / +X% / −X%" statement in this doc **uses Tok/s/dev** (the standard quantity for NV vs AMD, cross-cf, cross-quantization comparisons) — Tok/s/dev is unaffected by FLOP-counting conventions (FP8 and BF16 have different peaks, so TFLOP/s/dev cross-precision comparisons are distorted, while Tok/s/dev maps directly to actual training throughput). `TFLOP/s/dev` and `MFU` columns are kept as auxiliary references (for compute intensity / utilization distribution).

**Tok/s/dev convention:** every `Tok/s/dev` column in this doc = `per_device_batch_size × max_target_length / step_time` (DS3-671B defaults to `max_target_length = 4096`, see [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)). The Section H (Jobs 4231–4247) / Section I (Jobs 4396–4419) sub-tables in the original `b200-benchmark-report.md` initially listed only `TFLOP/s/dev`; the `Tok/s/dev` column has been backfilled using the formula above.
---

## BF16

### `dense-cf1.25` (BF16, `capacity_factor=1.25`, `sparse_matmul=false`)

Rows are ordered by pdbs ascending, and within a group by Job ID ascending; at the same pdbs the best Tok/s/dev run is bolded (TFLOP/s/dev is bolded together as an auxiliary reference; the two sorts agree).

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  4 | 1196 | AMD-parity | SUCCESS | 18.47 |  9.85 |   221.8 |   886 | **first viable pdbs**; ~22 GiB headroom |
|  6 | 1198 | AMD-parity | SUCCESS | 23.34 | 11.71 |   263.5 | 1,048 | OOM-edge probe; 2.5 GiB headroom |
|  7 | 1200 | AMD-parity | SUCCESS | 26.51 | 12.04 |   270.8 | 1,081 | **max pdbs under AMD-parity (no other optimizations)** |
|  7 | 1210 | AMD-parity + `profiler=xplane` | SUCCESS | 26.43 | 12.08 |   271.8 | 1,085 | xplane traces captured |
|  7 | 1213 | AMD-parity + `ici_fsdp=2, ici_ep=4` | SUCCESS | 26.29 | 12.14 |   273.2 | 1,090 | small +0.9% |
|  7 | 1214 | AMD-parity + `megablox=True` | SUCCESS | 26.55 | 12.02 |   270.5 | 1,080 | neutral / slightly worse |
|  7 | 1524 | NV defaults | SUCCESS | 24.23 | 13.17 |   296.4 | 1,183 | **NV +9.4% vs AMD-parity** |
|  7 | 1525 | AMD + `autotune_level=4` | SUCCESS | 26.50 | 12.04 |   270.9 | 1,082 | isolation has no effect |
|  7 | 1526 | AMD + `triton_gemm=true` | SUCCESS | 100.58 |  3.17 |    71.3 |   285 | **catastrophic −74%** — never use |
|  7 | 1527 | AMD + `slop_factor=300` | ✗ OOM | -- | -- | -- | -- | 108.71 GiB |
|  7 | 1528 | AMD + `combine=256 B` | IB HANG | -- | -- | -- | -- | NCCL stall; resubmitted as 1588 |
|  7 | 1529 | AMD + pipelined collectives | SUCCESS | 26.71 | 11.95 |   268.9 | 1,073 | −0.7% — conflicts with LHS |
|  7 | 1530 | AMD + `combine_by_dim=true` | SUCCESS | 25.10 | 12.71 |   286.0 | 1,142 | **second-largest single-flag gain +5.6%** |
|  7 | 1588 | AMD + `combine=256 B` (retry) | SUCCESS | 24.27 | 13.16 |   295.7 | 1,182 | +9.3% — roughly matches NV defaults |
|  7 | 1589 | NV + `combine_by_dim=true` | SUCCESS | 24.27 | 13.15 |   295.9 | 1,181 | no extra gain over NV defaults |
|  7 | 1590 | NV + `megablox=True` | SUCCESS | 24.25 | 13.16 |   296.1 | 1,182 | neutral |
|  7 | 1591 | NV + `shardy=true` | SUCCESS | 24.27 | 13.16 |   296.1 | 1,181 | Shardy neutral on the dense path |
|  7 | 4231 | NV + `while_loop_double_buffering=true` | FAILED | -- | -- | -- | -- | LHS over-budget + XLA IndexError |
|  7 | 4233 | NV + `pipelined_all_gather=true` | FAILED | -- | -- | -- | -- | LHS 124.6 > 109.4 GiB |
|  7 | 4234 | NV + `pipelined_reduce_scatter=true` | FAILED | -- | -- | -- | -- | LHS 122.3 > 109.4 GiB |
|  7 | 4235 | NV + `pipelined_all_reduce=true` | FAILED | -- | -- | -- | -- | LHS 133.4 > 109.4 GiB |
|  7 | 4236 | NV + `highest_priority_async_stream=true` | SUCCESS | 24.30 | 13.13 |   295.5 | 1,180 | neutral (−0.3%) |
|  7 | 4237 | NV + `parallel_collective_overlap_limit=2` | SUCCESS | 23.98 | 13.31 |   299.5 | 1,196 | +1.0% |
|  7 | 4238 | NV + `parallel_collective_overlap_limit=4` | SUCCESS | 22.24 | 14.35 | **322.8** | **1,289** | **+9.0% Tok/s/dev — best single flag overall** |
|  7 | 4239 | NV + `parallel_collective_overlap_limit=8` | SUCCESS | 23.34 | 13.68 |   307.7 | 1,228 | +3.8% |
|  7 | 4240 | NV + `{ag,rs}_combine=256 MiB` | SUCCESS | 24.50 | 13.03 |   293.2 | 1,170 | −1.1% |
|  7 | 4241 | NV + `{ag,rs}_combine=512 MiB` | SUCCESS | 25.13 | 12.70 |   285.8 | 1,141 | −3.6% |
|  7 | 4242 | NV + `{ag,rs}_combine=1 GiB` | SUCCESS | 25.18 | 12.68 |   285.2 | 1,139 | −3.8% |
|  7 | 4243 | NV + `{ag,rs}_combine=2 GiB` | SUCCESS | 25.42 | 12.56 |   282.5 | 1,128 | −4.7% |
|  7 | 4244 | NV + `{ag,rs}_combine=4 GiB` | SUCCESS | 26.40 | 12.09 |   272.0 | 1,086 | −8.2% |
|  7 | 4245 | NV + `ag_combine=256 MiB` | SUCCESS | 24.38 | 13.09 |   294.5 | 1,176 | −0.6% |
|  7 | 4246 | NV + `ag_combine=1 GiB` | TIMEOUT | -- | -- | -- | -- | LHS 120.6 > 109.4 GiB |
|  7 | 4247 | NV + `ag_combine=4 GiB` | CANCELLED | -- | -- | -- | -- | reservation expired |
|  7 | 4396 | NV + `ag_combine=1 GiB + slop95` | SUCCESS | 25.15 | 12.69 |   285.5 | 1,140 | rescued 4246 |
|  7 | 4397 | NV + `ag_combine=4 GiB + slop95` | SUCCESS | 25.81 | 12.36 |   278.2 | 1,111 | rescued 4247 |
|  7 | 4399 | NV + `pip-ag + slop95` | SUCCESS | 26.19 | 12.19 |   274.2 | 1,095 | rescued 4233 |
|  7 | 4400 | NV + `pip-rs + slop95` | SUCCESS | 24.31 | 13.13 |   295.4 | 1,179 | rescued 4234, neutral |
|  8 | 1195 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 108.63 GiB |
|  8 | 1203 | AMD + `optimizer_memory_host_offload=True` | ✗ OOM | -- | -- | -- | -- | offload makes it worse (146.95 GiB) |
|  8 | 1204 | AMD + `shard_exp_on_fsdp=True` | FAILED | -- | -- | -- | -- | args 492 GiB > limit 178 GiB |
|  8 | 1205 | AMD + offload + shard_exp | ✗ OOM | -- | -- | -- | -- | CUDA OOM |
|  8 | 4401 | NV + overlap4 + slop95 | ✗ OOM | -- | -- | -- | -- | 113.79 GiB |
|  8 | 4403 | NV defaults + mem95 | SUCCESS | 29.76 | 12.26 |   275.8 | 1,101 | **mem95 unlocks pdbs=8** |
|  8 | 4404 | NV defaults + mem97 | SUCCESS | 29.17 | 12.51 |   281.4 | 1,123 | mem97 unlocks pdbs=8 |
|  8 | 4407 | NV + overlap4 + mem97 | SUCCESS | 28.09 | 12.98 | **292.1** | **1,167** | **mem97 unlocks pdbs=8 + overlap4** |
|  9 | 4408 | NV + mem97 | ✗ OOM | -- | -- | -- | -- | 115.10 GiB — `cf=1.25` pdbs=8 is the hard ceiling on B200 |
| 12 | 1194 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 140.37 GiB |
| 16 | 1193 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 171.96 GiB — baseline OOMs directly |

**Core observations (Tok/s/dev–primary):**

- **pdbs=8 is the BF16 cf=1.25 hard ceiling on B200** (pdbs=9 OOMs at 115.10 GiB even with mem97).
- **NV defaults vs AMD-parity gives a stable +9.4% Tok/s/dev**: Job 1524 (NV) 1,183 vs Job 1200 (AMD) 1,081 = +9.4%; most AMD-parity single-flag isolations are neutral or regress.
- **pdbs=7 + `overlap_limit=4` (Job 4238) is the best single-flag run overall**: **1,289 Tok/s/dev** (+9.0% vs NV defaults 1,183; +19.2% vs AMD-parity 1,081), auxiliary 322.8 TFLOP/s/dev / 14.35% MFU; raising to `overlap=8` regresses to 1,228 (+3.8%).
- **mem97 + overlap4 + pdbs=8 (Job 4407) is the BF16 cf=1.25 best throughput overall**: **1,167 Tok/s/dev** (per-pdbs it is −9.5% below 4238, but the +14% global batch more than offsets); auxiliary 292.1 TFLOP/s/dev.
- **Combine threshold monotonically regresses on B200**: from 256 MiB → 4 GiB the Tok/s/dev drop goes −1.1% → −8.2% (4240=1,170 → 4241=1,141 → 4242=1,139 → 4243=1,128 → 4244=1,086 vs 1,183 baseline), and 256 B (equivalent to NV defaults) is actually best — opposite to part of the MI355 literature trend.

---

### `dense-cf1` (BF16, `capacity_factor=1.0`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 1207 | AMD-parity | SUCCESS | 23.71 | 13.46 | **302.9** | 1,209 | **cf=1.0 = the single largest optimization (+11.8% Tok/s/dev)** vs cf=1.25 bs=7 (1,081) |
|  7 | 1212 | AMD-parity + `gradient_accumulation_steps=2` | ✗ OOM | -- | -- | -- | -- | 118.91 GiB — ga actually increases memory |
|  7 | 1216 | AMD-parity + `megablox=True` | SUCCESS | 23.62 | 13.51 |   304.0 | 1,214 | tiny extra gain over plain cf=1.0 |
|  7 | 4398 | NV + overlap4 + slop95 | FAILED | -- | -- | -- | -- | SEGFAULT (LHS over budget) |
|  7 | 4406 | NV + overlap4 + mem97 | SUCCESS | 23.42 | 13.63 |   306.7 | 1,224 | +1.3% over plain cf=1.0; < cf=1.25 + overlap4 peak |
|  8 | 1208 | AMD-parity | SUCCESS | 26.38 | 13.83 | **312.2** | 1,242 | **cf=1.0 unlocks pdbs=8**; soft-over 4.6 GiB |
|  8 | 1235 | AMD + `megablox=True` | SUCCESS | 26.35 | 13.84 |   311.4 | 1,243 | **best BF16 under AMD-parity** (marginal) |
|  8 | 4402 | NV + overlap4 + slop95 | FAILED | -- | -- | -- | -- | SEGFAULT (LHS over budget) |
|  9 | 1218 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 109.83 GiB |
|  9 | 1236 | AMD + `num_vocab_tiling=4` | FAILED | -- | -- | -- | -- | dtype assertion (f32 vs bf16) — incompatible with DS3-671B |
|  9 | 1237 | AMD + `grad_dtype=bfloat16` | ✗ OOM | -- | -- | -- | -- | 109.83 GiB — grad_dtype gives zero savings |
|  9 | 1238 | AMD + `grad_dtype=bf16 + vocab_tiling=4` | FAILED | -- | -- | -- | -- | same dtype assertion as 1236 |
|  9 | 4405 | NV + mem97 | SUCCESS | 28.99 | 14.15 | **318.4** | **1,272** | **mem97 unlocks pdbs=9; BF16 cf=1.0 peak MFU** |
| 10 | 1219 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 116.11 GiB |

**Core observations (Tok/s/dev–primary):**

- **`cf=1.0` is the single largest BF16 optimization on B200: +11.8% Tok/s/dev** (Job 1207 cf=1.0 bs=7 = 1,209 vs Job 1200 cf=1.25 bs=7 = 1,081; same AMD-parity flag set, same pdbs) — reduces dispatch padding while saving both memory and compute.
- **Unlocking pdbs=9 requires mem97** (Job 4405: **1,272 Tok/s/dev**, auxiliary 318.4 TFLOP/s/dev / 14.15% MFU = highest MFU across all BF16 runs) — another +2.4% over the cf=1.0 pdbs=8 baseline (1,242).
- **BF16 cf=1.0 hard ceiling on B200 = pdbs=9** (pdbs=10 OOMs at 116.11 GiB even under AMD-parity).
- Several "memory savers" (`num_vocab_tiling`, `grad_dtype=bf16`, `optimizer_memory_host_offload`, `shard_exp_on_fsdp`) are either dtype-incompatible with DS3-671B or save zero / are actually worse — none gives a net Tok/s/dev gain.

---

### `dense-cf2` (BF16, `capacity_factor=2.0`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  4 | 1595 | AMD-parity | SUCCESS | 22.04 |  8.28 |   186.2 |   743 | viable but −37.2% Tok/s/dev vs cf=1.25 NV defaults (1,183) |
|  5 | 4163 | AMD-parity | SUCCESS | 25.53 |  8.93 |   200.9 |   802 | cf=2.0 edge (AMD) |
|  5 | 4164 | NV defaults | SUCCESS | 23.69 |  9.62 |   216.5 |   864 | **cf=2.0 best (NV defaults)** — NV +7.7% Tok/s/dev vs AMD at same pdbs |
|  5 | 4416 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR (network flakiness) |
|  6 | 4171 | NV defaults | ✗ OOM | -- | -- | -- | -- | 109.38 GiB |
|  6 | 4177 | NV + `slop_factor=95` | ✗ OOM | -- | -- | -- | -- | 109.38 GiB — slop95 has no effect |
|  6 | 4417 | NV + overlap4 + mem97 | SUCCESS | 28.43 |  9.62 |   216.5 |   864 | **mem97 unlocks pdbs=6**; +20% global batch |
|  7 | 1522 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 116.84 GiB |
|  7 | 4423 | NV + overlap4 + mem97 | SUCCESS | 31.40 | 10.16 | **228.7** |   913 | **mem97 unlocks pdbs=7** — cf=2.0 best overall |

**Core observations (Tok/s/dev–primary):**

- **cf=2.0 cuts max pdbs from cf=1.0's 9 down to 5 (default) / 7 (mem97)**; every step up in capacity_factor doubles dispatch padding and roughly halves viable pdbs.
- **NV defaults still wins +7.7% Tok/s/dev on cf=2.0**: Job 4164 (NV) 864 vs Job 4163 (AMD) 802 = +7.7%. The NV advantage is cf-insensitive (cf=1.25 also gives +9.4%).
- **mem97 + overlap4 is the only path that breaks the pdbs=5 ceiling on cf=2.0** (4417 pdbs=6 = 864 Tok/s/dev / 4423 pdbs=7 = 913 Tok/s/dev both rely on mem97), but it is already against the IB network stability ceiling (4416 pdbs=5 exits with an IB error even with mem97).


---

### `dense-cf4` (BF16, `capacity_factor=4.0`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  2 | 1596 | AMD-parity | SUCCESS | 20.42 |  4.47 |   100.5 |   401 | viable but −66.1% Tok/s/dev vs cf=1.25 NV defaults (1,183) |
|  2 | 4176 | NV defaults | SUCCESS | 18.98 |  4.80 | **108.1** |   432 | **cf=4.0 NV defaults best** — NV +7.7% Tok/s/dev vs AMD |
|  2 | 4418 | NV + overlap4 + mem97 | SUCCESS | 20.44 |  4.46 |   100.4 |   401 | overlap4 slightly regresses on cf=4.0 + pdbs=2 |
|  3 | 4175 | NV defaults | SEGFAULT | -- | -- | -- | -- | silent crash in XLA compile; ~14 GB coredump |
|  3 | 4419 | NV + overlap4 + mem97 | SUCCESS | 26.39 |  5.18 | **116.6** | **466** | **mem97 unlocks pdbs=3** — cf=4.0 best overall |
|  4 | 4172 | NV defaults | ✗ OOM | -- | -- | -- | -- | 108.26 GiB |
|  4 | 4424 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR (cancelled to make room for profiling) |
|  5 | 4165 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 126.30 GiB |
|  5 | 4166 | NV defaults | ✗ OOM | -- | -- | -- | -- | 126.42 GiB — flag set is essentially irrelevant at this margin |
|  7 | 1523 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 145.29 GiB |

**Core observations (Tok/s/dev–primary):**

- **cf=4.0 max pdbs on B200 = 3** (mem97), best **466 Tok/s/dev** (Job 4419); auxiliary 116.6 TFLOP/s/dev. Vs cf=1.25 best (1,289) = **−63.8%**; vs cf=1.0 best (1,272) = **−63.4%**.
- **NV defaults vs AMD-parity on cf=4.0 again +7.7% Tok/s/dev**: Job 4176 (NV bs=2) 432 vs Job 1596 (AMD bs=2) 401 = +7.7%. The NV advantage is consistent across all cf.
- pdbs=3 SEGFAULTs under bare NV defaults (4175) but runs cleanly with overlap4 + mem97 — XLA scheduling is sensitive to stability at that OOM edge.
- pdbs=4 with mem97 trips IB retry errors (4424), a B200 cluster IB edge case under large buffers at cf=4.0.
- **cf=4.0 and cf=2.0 on B200 are essentially "ablation runs for studying the capacity_factor effect" rather than production configs** (Tok/s/dev is 28–64% below cf=1.0 / 1.25).

---

### `sparse_matmul` (BF16, `sparse_matmul=True + shardy=True`)

| pdbs | Job ID | XLA Flags / Run | Status | Failed alloc | Notes |
|---:|---:|---|---|---:|---|
|  1 | 4198 | NV + `one_shot=true` | ✗ OOM | **112 GiB** | key data point — single pdbs unit needs ~112 GiB |
|  1 | 4201 | NV + `one_shot=true + slop95` | ✗ OOM | 112 GiB | byte-identical to 4198, slop95 has no effect |
|  1 | 4229 | NV + `one_shot=true + mem_fraction=.95` | ✗ OOM | 112 GiB | byte-identical, mem95 also has no effect |
|  2 | 4189 | NV + `one_shot=true` | ✗ OOM | 224 GiB | = 2 × 112 |
|  2 | 4190 | NV + `one_shot=false` | ✗ OOM | 224 GiB | byte-identical to 4189, one_shot toggle has no memory effect |
|  7 | 1215 | AMD-parity (no shardy) | FAILED | -- | RaggedDot requires Shardy (legalization failed) |
|  7 | 1217 | AMD-parity + `cf=1.0` (no shardy) | FAILED | -- | same root cause as 1215 |
|  7 | 1239 | AMD-parity + `shardy=true + cf=1.0` | ✗ OOM | **2.28 TiB** | pathological Shardy plan blow-up |
|  7 | 4182 | NV + `one_shot=true + cf=1.0` | ✗ OOM | 784 GiB / 224 GiB | heterogeneous failure (5 ranks / 3 ranks) |
|  7 | 4183 | NV + `one_shot=false + cf=1.0` | ✗ OOM | 784 GiB / 224 GiB | byte-identical to 4182 |
|  8 | 1240 | AMD-parity + `shardy=true + cf=1.0` | ✗ OOM | **2.60 TiB** | same-shape Shardy pathology |

**Core observations (from b200-benchmark-report Section G):**

1. **`sparse_matmul=True + shardy=True` on 8N B200 plans ~112 GiB of allocation per pdbs unit.** B200's per-GPU XLA budget ≈ 102 GiB, so even pdbs=1 falls ~10 GiB short, **no pdbs is feasible**.
2. **The `xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel` toggle has zero memory effect** (4182 vs 4183 / 4189 vs 4190 are byte-identical). The flag controls the execution strategy for ragged-a2a collectives, not the XLA buffer planning.
3. **`slop_factor=95` and `mem_fraction=.95` also have no effect** (4201 / 4229 are byte-identical to 4198) — the failure happens in XLA's shape-based feasibility check inside the planner, before BFC arena / mem pool come into play.
4. **NV defaults shrinks the worst-case allocation by ~3.3× relative to AMD-parity (2.60 TiB → 784 GiB)**, but it is still far above B200's 179 GiB HBM. The root issue of Shardy + sparse_matmul is unsolved under MaxText `e26c2ac7` + the JAX 26.03 image.
5. **The sparse path is unavailable on B200**, and the `sparse-gmm` / `sparse-gmm-fixed` / `sparse-gmm-deepep-v*` variants discussed in the MI355 reference (which rely on Primus-Turbo + DeepEP) are not carried under this image — no comparable measurable column. Production paths can only use `sparse_matmul=False` (dense MoE).

---

## FP8

### `dense-cf1.25` (FP8, `capacity_factor=1.25`, `quantization=fp8`)

> FP8 MFU% is relative to FP8 peak (4,500 TFLOP/s for B200); BF16-equivalent MFU = MFU × 2.

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 1592 | NV defaults | SUCCESS | 21.40 |  7.46* |   335.5 | 1,340 | **prior best (cf=1.25)** |
|  8 | 1593 | NV defaults | ✗ OOM | -- | -- | -- | -- | 107.70 GiB |
|  8 | 4412 | NV + overlap4 + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR |
|  9 | 4413 | NV + mem97 | SUCCESS | 26.35 |  7.79* | **350.4** | **1,399** | **mem97 unlocks pdbs=9 — FP8 cf=1.25 best (+4.4% Tok/s/dev vs 1592 1,340)** |
| 10 | 1594 | NV defaults | ✗ OOM | -- | -- | -- | -- | 122.93 GiB |

**Core observations (Tok/s/dev–primary):**

- **FP8 cf=1.25 + mem97 + pdbs=9 (Job 4413) pushes the NV defaults baseline to 1,399 Tok/s/dev** (+4.4% vs 1592's 1,340; auxiliary 350.4 TFLOP/s/dev).
- FP8 cf=1.25 ceiling = pdbs=9 (pdbs=10 under NV defaults OOMs at 122.93 GiB; pdbs=8 + overlap4 + mem97 trips an IB error).
- Vs the BF16 cf=1.25 same-pdbs=7 baseline (Job 1524 = 1,183 Tok/s/dev), FP8 cf=1.25 pdbs=7 (Job 1592 = 1,340) = **+13.3% Tok/s/dev**.

### `dense-cf1` (FP8, `capacity_factor=1.0`, `quantization=fp8`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  7 | 4161 | NV defaults | SUCCESS | 19.46 |  8.20* |   369.0 | 1,473 | cf=1.0 +10.0% Tok/s/dev vs cf=1.25 (1592 1,340) |
|  8 | 4170 | NV + `slop_factor=95` | SUCCESS | 21.55 |  8.46* | **380.9** | **1,521** | **OVERALL BEST FP8** — slop95 unlocks pdbs=8 (+3.3% Tok/s/dev over pdbs=7) |
|  9 | 4178 | NV + `slop_factor=95` | ✗ OOM | -- | -- | -- | -- | 112.39 GiB |
|  9 | 4409 | NV + mem97 | FAILED | -- | -- | -- | -- | IBV_WC_RETRY_EXC_ERR |
| 12 | 1241 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 118.74 GiB (~15% less than BF16 bs=12) |
| 16 | 1242 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 147.38 GiB |

**Core observations (overall FP8 best; Tok/s/dev–primary):**

- **`fp8 cf=1.0 + slop95 + pdbs=8` (Job 4170) is the overall best throughput for DS3-671B on B200: 1,521 Tok/s/dev** (auxiliary 380.9 TFLOP/s/dev / FP8 MFU 8.46% = BF16-eq MFU 16.92%). Vs the BF16 best (Job 1208 cf=1.0 pdbs=8 = 1,242 Tok/s/dev) = **+22.5% Tok/s/dev**; vs the BF16 + mem97 best (Job 4405 pdbs=9 = 1,272 Tok/s/dev) = **+19.6%**.
- **FP8 cf=1.0 vs FP8 cf=1.25 at the same pdbs=7: +10.0% Tok/s/dev** (Job 4161 = 1,473 vs Job 1592 = 1,340) — on par with the cf=1.0 gain on BF16 (+11.8%).
- **`slop_factor=95` is the key enabler to unlock pdbs=8 on FP8 cf=1.0** (under the default `.93`, 1593 / 4178 both OOM) — same pattern as BF16 cf=1.25.
- pdbs=9 fails under both slop95 / mem97 (4178 OOM / 4409 IBV error), **pdbs=8 is the final ceiling for FP8 cf=1.0 on this hardware**.

### FP8 configs not covered (`dense-cf2` / `dense-cf4` / `sparse_matmul`)

The purpose of this B200 sweep is **to be an apples-to-apples mirror of the MI355 reference**, so only those (precision, config) combinations that ran successfully and produced comparable numbers on MI355 were run on B200. These three rows **did not run / have no comparable FP8 data in the MI355 reference**, so they were skipped on B200 as well:

| FP8 config | Reason skipped | MI355 reference status |
|---|---|---|
| `dense-cf2` (FP8) | No MI355 FP8 cf=2.0 baseline to mirror | The MI355 reference has no FP8 columns at all (the report focuses on BF16 + sparse-gmm-deepep kernel analysis); cf=2.0 only has BF16 rows |
| `dense-cf4` (FP8) | Same as above | Same as above |
| `sparse_matmul` (FP8) | BF16 sparse_matmul OOMs at pdbs=1 with 112 GiB on B200 — the failure happens in the XLA planner's shape-based feasibility check and is dtype-independent (dispatch routing uses an index dtype that FP8 cannot compress). MI355 only runs because it switches to the `sparse-gmm` / `sparse-gmm-deepep-v3` Primus-Turbo kernel path, which is not carried in this B200 image `nvcr.io/nvidia/jax:26.03-maxtext-py3` | The MI355 reference also only has BF16 data in this column; FP8 dispatch is natively supported by the DeepEP kernel (see the "where DeepEP actually shines" section in the reference), but there is no equivalent kernel entry under B200 + JAX 26.03 |

**Summary:** FP8 on B200 is only viable for `dense-cf1.25` / `dense-cf1` (which are also the two configs comparable in the MI355 reference; the MI355 dense-cf1.25 vs dense-cf1 corresponds to the 1592 / 4161 / 4170 series here). The other three (`cf=2.0` / `cf=4.0` / `sparse_matmul`) have no FP8 data on MI355 either, so any cross-platform comparison would be meaningless — no test resources were invested on B200. If the MI355 reference later fills in those FP8 columns, we will revisit B200 to fill in the corresponding runs.

---

## Overall best summary

Sorted by **Tok/s/dev** (primary metric) descending; TFLOP/s/dev / MFU / Step are auxiliary.

| Rank | Precision | Config | pdbs | **Tok/s/dev** | Δ vs BF16 best | Job | XLA Flags | Step (s) | TFLOP/s/dev | MFU (%) |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| **1** | **FP8** | **dense-cf1** | **8** | **1,521** | **+19.6%** | **4170** | **NV + slop95** | **21.55** | **380.9** | **8.46\*** |
| 2 | FP8 | dense-cf1.25 | 9 | 1,399 | +10.0% | 4413 | NV + mem97 | 26.35 | 350.4 | 7.79* |
| 3 | BF16 | dense-cf1.25 | 7 | 1,289 | +1.3% | 4238 | NV + overlap4 | 22.24 | 322.8 | 14.35 |
| **4** | **BF16** | **dense-cf1** | **9** | **1,272** | **0** (baseline) | **4405** | **NV + mem97** | **28.99** | **318.4** | **14.15** |
| 5 | BF16 | dense-cf1.25 | 8 | 1,167 | −8.3% | 4407 | NV + overlap4 + mem97 | 28.09 | 292.1 | 12.98 |
| 6 | BF16 | dense-cf2 | 7 | 913 | −28.2% | 4423 | NV + overlap4 + mem97 | 31.40 | 228.7 | 10.16 |
| 7 | BF16 | dense-cf4 | 3 | 466 | −63.4% | 4419 | NV + overlap4 + mem97 | 26.39 | 116.6 | 5.18 |
| — | BF16 | sparse_matmul | — | **infeasible** | — | — | — | — | — | — |

\* FP8 MFU% is relative to FP8 peak (4,500 TFLOP/s); BF16-equivalent MFU = MFU × 2 = 16.92% (Job 4170) / 15.58% (Job 4413). The "Δ vs BF16 best" column uses the BF16 overall best Job 4405 (1,272 Tok/s/dev) as baseline.

---

## Key takeaways

> All "+X% / −X%" figures use **Tok/s/dev** (primary metric); TFLOP/s/dev and MFU are shown as auxiliary references.

1. **B200 overall best = FP8 cf=1.0 pdbs=8 + slop95 (Job 4170): 1,521 Tok/s/dev** (auxiliary 380.9 TFLOP/s/dev, FP8 MFU 8.46% = BF16-eq MFU 16.92%). Vs the B200 BF16 overall best (Job 4405 cf=1.0 pdbs=9 + mem97 = **1,272 Tok/s/dev**) = **+19.6%**; vs the original B200 BF16 baseline (Job 1208 cf=1.0 pdbs=8 = 1,242) = **+22.5%**. **However, it is still −4.8% below the MI355 BF16 cf=1.0 pdbs=16 peak of 1,598** — FP8 on B200 can only beat the MI355 BF16 cf=1.25 peak (1,416), not the MI355 BF16 cf=1.0 peak. FP8 is a net gain on DS3-671B; on Kimi-K2-1T it is a net loss (4168 BF16 = 248 vs 4169 FP8 = 226, **−8.9%**) — both models go through the dense_matmul path, and the sign flip's root cause has not yet been profile-localized (candidates include the relative weight of FP8 quantize/dequantize vs GEMM speedup under different MoE topologies, and the amplification of per-step quant overhead by layer count × number of experts).
2. **`cf=1.0` is the single largest BF16 optimization: +11.8% Tok/s/dev** (cf=1.0 bs=7 = 1,209 vs cf=1.25 bs=7 = 1,081) — reduces dispatch padding while saving memory and compute; recommended as the default for dense BF16.
3. **The pdbs jump unlocked by `mem97` is the second-largest BF16 tuning knob**: cf=1.25 ceiling pdbs=7 (1,183 Tok/s/dev) → pdbs=8 (1,167 + overlap4 = 1,167 per-device / +14% global batch); cf=1.0 ceiling pdbs=8 (1,242) → pdbs=9 (**1,272**, +2.4% per-device / +12% global); cf=2.0 ceiling pdbs=5 (864) → pdbs=7 (**913**, +5.7%); cf=4.0 ceiling pdbs=2 (432) → pdbs=3 (**466**, +7.9%). But `mem97 + overlap4 + large cf` triggers IB errors (4412 / 4416 / 4424 / 4409), the B200 cluster's network edge under buffer pressure.
4. **`xla_gpu_experimental_parallel_collective_overlap_limit=4` is the single largest BF16 XLA flag gain**: Job 4238 = **1,289 Tok/s/dev = +9.0% vs NV defaults 1,183** (also the per-pdbs best across BF16 cf=1.25); raising to `overlap=8` (Job 4239) regresses to 1,228 (+3.8%).
5. **`capacity_factor` strongly dominates both max pdbs and Tok/s/dev**: cf=1.0 best = 1,272 → cf=1.25 best = 1,289 (slightly edges thanks to pdbs=7 + overlap4) → cf=2.0 best = 913 (−29% vs cf=1.0) → cf=4.0 best = 466 (−63% vs cf=1.0). Every doubling of cf roughly halves the pdbs ceiling and halves Tok/s/dev.
6. **NV defaults vs AMD-parity gives a stable +7~10% Tok/s/dev**: cf=1.25 = +9.4% (1524 1,183 vs 1200 1,081), cf=2.0 = +7.7% (4164 864 vs 4163 802), cf=4.0 = +7.7% (4176 432 vs 1596 401) — flag-set differences are uniform across cf.
7. **`sparse_matmul + shardy` is infeasible on 8N B200** (even pdbs=1 falls ~10 GiB short); the failure happens in the XLA planner's shape-based feasibility check, and all runtime allocator-level workarounds (slop95 / mem95) fail byte-identically. Tok/s/dev = unmeasurable / `infeasible`. **The `sparse-gmm-*` / DeepEP variants discussed in the MI355 reference are not carried under this image**, so they are not a B200-comparable dimension. Production sparse MoE on B200 + JAX 26.03 first requires resolving the Shardy buffer-planning pathology.
8. **Several "memory savers" are no-ops or reversed on DS3-671B**: `optimizer_memory_host_offload=True` makes memory larger (1203), `shard_exp_on_fsdp=True` pushes args to 492 GiB > limit (1204), `grad_dtype=bfloat16` saves nothing (1237), `num_vocab_tiling=4` is dtype-incompatible with the model (1236, 1238), `remat_policy=minimal_flash` blows up to 744 GiB (1211). None unlocks higher pdbs, all are zero / negative Tok/s/dev; `mem97` is the only stable pdbs-extending knob.
9. **Combine threshold trends on B200 are opposite to some AMD literature**: Tok/s/dev from 256 MiB → 4 GiB monotonically regresses −1.1% → −8.2% (4240 1,170 → 4241 1,141 → 4242 1,139 → 4243 1,128 → 4244 1,086, all vs NV defaults 1,183); the 256 B equivalent (NV defaults) is optimal. The 8 GiB AMD-parity default is the global worst point on this path (−8.2% Tok/s/dev).
10. **Triton GEMM is unusable**: `xla_gpu_enable_triton_gemm=true` drops Tok/s/dev from 1,081 (Job 1200) to **285 (Job 1526) = −73.6%**, with step time spiking to 100s+; cuBLAS thoroughly beats Triton-generated kernels on DS3-671B's GEMM shapes. **Never use**.

---

## Comparison with the MI355 reference sweep

References:
- [`deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md) (MI355 main pdbs sweep, image-default XLA) — provides full pdbs lines for cf=1.25 / cf=2.0 / cf=4.0 / cf=1.0.
- [`pp-vs-fsdp-deepseek3-671b.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/pp-vs-fsdp-deepseek3-671b.zh.md) (MI355 PP vs FSDP + XLA flag tuning) — provides best tuned recipes for dense-cf1.25 and sgd-v3 under both **EP+FSDP (FSDP=8)** and **PP+EP (PP=8)** DCN topologies.

Each cf block presents: (1) the B200 BF16 best at that cf, (2) MI355 results at the **same pdbs** across multiple tiers (untuned + EP+FSDP tuned + PP+EP tuned, where available), and (3) the MI355 cf peak. All MI355 numbers are BF16, 1-node/proc launcher.

> The MI355 pdbs grid is `{1, 2, 4, 5, 6, 7, 8, 16}`, **skipping 3 and 9** — on B200 these two gaps are hit at `dense-cf4 pdbs=3` and `dense-cf1 pdbs=9` respectively, and those rows are marked *n/a*.
>
> **PP+EP / EP+FSDP tuning only covers `dense-cf1.25` and `sgd-v3` (sparse-gmm-deepep-v3)**. MI355 has no tuned data for cf=2.0 / cf=4.0, so those rows can only be compared against the untuned row.

### `dense-cf1.25` (BF16, `capacity_factor=1.25`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**                              | B200  | **1,289** |  7 | NV + overlap4 (Job 4238)                                       | 322.8 | 14.35 | baseline |
| MI355 same pdbs=7 (image default XLA)      | MI355 |   1,086   |  7 | FSDP=8 + image-default XLA, 1-node/proc                        | 272.1 | 10.88 | **−15.7%** |
| **MI355 same pdbs=7 (EP+FSDP XLA tuned)** ⭐ | MI355 | **1,208** |  7 | FSDP=8 + `ag_combine=1 GiB` XLA flag (Job 14629)              | ≈302.6 | ≈12.10 | **−6.3%** |
| **MI355 same pdbs=7 (PP+EP tuned)** ⭐⭐    | MI355 | **1,224** |  7 | PP=8 + `overlap_limit=2` (Jobs 14672/14673 mean, n=2)         | ≈306.6 | ≈12.27 | **−5.0%** |
| **MI355 peak** (untuned)                   | MI355 | **1,416** | 16 | FSDP=8 + image-default XLA, 1-node/proc                        | 354.7 | 14.19 | **+9.9%** |

- **B200 same-pdbs advantage over MI355 untuned = +18.7% (1,289 vs 1,086); after MI355 EP+FSDP XLA tuning, it narrows to +6.7% (1,289 vs 1,208); after MI355 PP+EP tuning, it narrows further to +5.3% (1,289 vs 1,224).** MI355 tuning eats most of the "+19% throughput gap" — B200 still leads at same pdbs, but the gap shrinks from ~one tier (19%) to ~jitter (5%).
- Key enabler comparison:
  - **MI355 EP+FSDP tuning knob = `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`** (splits the image-default 8 GiB single all-gather into ~4–5 1 GiB chunks, recovering ~3 sec/step of exposed comms; a single flag gives **+20.5% Tok/s/dev**: 1,002.4 → 1,207.9 on top of the FSDP=8 image-default baseline).
  - **MI355 PP+EP tuning knob = `--xla_gpu_experimental_parallel_collective_overlap_limit=2`** + **dropping the EP+FSDP `ag=1 GiB` override** (back to ag=8 GiB image-default; under the PP=8 topology, ICI all-gather + DCN collective-permute run concurrently on the two fabrics; a single flag on top of the PP=8 image-default gives **+5.8% Tok/s/dev**: 1,156.9 → 1,224.1. Vs the PP=8 baseline-ag1G 1,080.2 = +13.3%).
  - **B200's corresponding tuning knob = `overlap_limit=4`** (+9.0% Tok/s/dev on top of NV defaults: 1,183 → 1,289, Job 4238) — same XLA flag in spirit (raising the count of concurrent in-flight collectives), but **different sweet spot** (MI355 PP=8 = 2, B200 = 4), reflecting the two platforms' different ICI + DCN fabric utilization characteristics; on MI355 FSDP=8, `overlap_limit=2/4/8` all regress −4~−5% (ICI is already saturated by ag), exactly the opposite sign of the +5.8% on PP=8.
- **PP+EP vs EP+FSDP on MI355 dense-cf1.25: PP=8 edges out marginally**: 1,224 (PP+EP) > 1,208 (EP+FSDP) = +1.34%. **But the order reverses on MI355 sgd-v3**: FSDP=8 tuned 1,135.7 > PP=8 tuned 999.8 = +13.6%, the gap coming from DeepEP per-microbatch overhead (~1.3s/step) + nn.scan carry-dep forcing collective-permute `is_pipelined=false` + pipeline bubble = 7/63 = 11.1%. I.e. "PP+EP beats FSDP" holds only on the dense_matmul branch; the sparse_matmul-DeepEP branch always favors FSDP.
- B200 vs MI355 peak (pdbs=16 untuned) is still −9.0% — the entire peak gap is the pdbs ceiling (B200 max=8 vs MI355 max=16), unrelated to per-device throughput. **MI355 at pdbs=16 + tuning** has not been measured (the PP-vs-FSDP sweep fixes pdbs=7 throughout), so this ceiling cannot be quantified.
- Per-device MFU 14.35% (B200) > 14.19% (MI355 peak) — B200's BF16 peak (2,250) is 10% below MI355's (2,500), but its actual TFLOP/s utilization is slightly higher, in the same MFU tier (MI355 pdbs=7 tuned MFU is only ~12.27%, so B200 leads by ~+17% MFU at the same pdbs).

### `dense-cf1` (BF16, `capacity_factor=1.0`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Notes |
|---|---|---:|---:|---|---:|---:|---|
| **B200 best**                    | B200  | **1,272** |  9 | NV + mem97 (Job 4405)                       | 318.4 | 14.15 | mem97 unlocks pdbs=9, highest BF16 MFU overall |
| **MI355 peak (cf=1.0)** ⭐       | MI355 | **1,598** | 16 | FSDP=8 + image-default XLA, 1-node/proc     | **400** | **16.01** | **B200 cf=1 best (1,272) −20.4%** vs MI355 cf=1.0 peak |
| MI355 same pdbs=16, cf=1.25 baseline (reference) | MI355 |   1,418   | 16 | FSDP=8 + image-default XLA                  | 354   | 14.20 | cf=1.0 vs cf=1.25 at same pdbs=16: +12.7% Tok/s/dev / +12.8% MFU |

- **MI355 cf=1.0 measured at pdbs=16 reaches 1,598 Tok/s/dev**, MFU 16.01% — the highest known MFU for DS3-671B BF16 MoE.
- **B200 cf=1 best (1,272 @ pdbs=9) vs MI355 cf=1.0 peak (1,598 @ pdbs=16) = −20.4%**. The peak gap comes both from the pdbs ceiling (B200 max=9 vs MI355 max=16) and from per-device MFU lag (14.15% vs 16.01%).
- **B200 cf=1 best MFU (14.15%) is below MI355 cf=1.0 peak MFU (16.01%)**, a −1.86 pp gap — the **only** single-kernel utilization dimension where B200 trails MI355 on the cf=1.0 path (B200 leads MI355 at same pdbs on cf=1.25 / cf=2.0 / cf=4.0). MI355 cf=1.0 only has a pdbs=16 datapoint; no small-pdbs sweep was run, so a strict same-pdbs per-device comparison is not possible.
- For contrast, FP8: B200 FP8 cf=1.0 pdbs=8 (Job 4170) = 1,521 Tok/s/dev, **−4.8% below MI355 cf=1.0 peak 1,598** (FP8 BF16-eq MFU 16.92% slightly exceeds MI355 cf=1.0 16.01%, but pdbs is still 2 tiers behind: B200=8 vs MI355=16) — **FP8 still does not let B200 surpass the MI355 BF16 cf=1.0 peak**; the root cause remains the HBM-bound pdbs ceiling.

### `dense-cf2` (BF16, `capacity_factor=2.0`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**          | B200  | **913**   |  7 | NV + overlap4 + mem97 (Job 4423) | 228.7 | 10.16 | baseline |
| MI355 same pdbs=7      | MI355 |   884     |  7 | 1-node/proc                      | 221.3 |  8.85 | **−3.2%** |
| **MI355 peak**         | MI355 | **968**   | 16 | 1-node/proc                      | 242.4 |  9.70 | **+6.0%** |

- At same pdbs=7, B200 is +3.3% faster than MI355 (913 vs 884, MFU 10.16% vs 8.85%) — B200 still leads per-device, but the advantage shrinks substantially from cf=1.25's +18.7%. MFU drops on both platforms (B200 14.35% → 10.16%, −4.19 pp; MI355 10.88% → 8.85%, −2.03 pp), and the drop is larger on B200; on the dense_matmul path, cf only affects the dispatch / combine einsum and the dense GEMM capacity dimension and does not introduce all-to-all. A plausible (unprofiled) reason MI355 narrows the gap is that its 1.6× HBM capacity leaves more overlap window for the compiler when the cf=2.0 dispatch buffer balloons; pinning this down requires a separate cf=1.25 vs cf=2.0 profile comparison.
- MI355 peak (utilizing larger pdbs=16) reverses to +6.0%; B200 is already at the ceiling at pdbs=7 (pdbs=8 trips IB errors even with mem97+overlap4).

### `dense-cf4` (BF16, `capacity_factor=4.0`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**          | B200  | **466**   | 3 | NV + overlap4 + mem97 (Job 4419) | 116.6 | 5.18 | baseline |
| MI355 same pdbs=3      | MI355 | *n/a*     | 3 | *n/a*                            | —     | —    | MI355 grid skips from 2 to 4 |
| MI355 adjacent pdbs=2  | MI355 |   374     | 2 | 1-node/proc                      | 93.6  | 3.74 | B200 same pdbs=2 (Job 4176) = 432 → **+15.5%** vs MI355 same pdbs |
| MI355 adjacent pdbs=4  | MI355 |   500     | 4 | 1-node/proc                      | 125.2 | 5.01 | B200 pdbs=4 OOMs; MI355 +7.3% vs B200 best |
| **MI355 peak**         | MI355 | **566**   | 8 | 1-node/proc                      | 141.7 | 5.67 | **+21.5%** vs B200 best |

- At same pdbs=2, B200 is +15.5% faster than MI355 (432 vs 374) — B200's NV-defaults flag advantage persists at cf=4.
- But on cf=4, MI355 max pdbs = 8 while B200 max pdbs = 3 (with mem97), a 2.7× ceiling gap — at peak, MI355 leads by +21.5%.
- cf=4 is the dense config most severely HBM-constrained on B200 (dispatch tensor ~ pdbs × cf linear blow-up).

### Cross-cf BF16 summary

| Config | B200 best (pdbs) | MI355 same pdbs untuned (TGS) | MI355 same pdbs **tuned** (TGS, recipe) | MI355 peak (pdbs, TGS) | B200 vs MI355 untuned same pdbs | B200 vs MI355 **tuned same pdbs** | B200 vs MI355 peak |
|---|---|---|---|---|---:|---:|---:|
| `dense-cf1.25` | **1,289** (7) | 1,086 (pdbs=7) | **1,224** (pdbs=7, PP=8+overlap2) ⭐ / 1,208 (pdbs=7, FSDP=8+ag1G) | 1,416 (pdbs=16) | **+18.7%** | **+5.3% (PP+EP) / +6.7% (EP+FSDP)** | **−9.0%** |
| `dense-cf1`    | **1,272** (9) | n/a | n/a | **1,598** (pdbs=16, FSDP=8 image default) ⭐ | — | — | **−20.4%** |
| `dense-cf2`    | **913** (7)   | 884 (pdbs=7) | n/a | 968 (pdbs=16) | **+3.3%** | — | **−5.7%** |
| `dense-cf4`    | **466** (3)   | adj. 374 (pdbs=2), 500 (pdbs=4) | n/a | 566 (pdbs=8) | **+15.5%** (vs pdbs=2) | — | **−17.7%** |

Dropless (`sparse_matmul`) supplementary comparison:

| Path | B200 | MI355 best Tok/s/dev (pdbs, config) | Notes |
|---|---|---|---|
| `sparse-gmm-deepep-v3` (sgd-v3) | **infeasible** (sparse_matmul OOMs at pdbs=1; image does not carry DeepEP/Primus-Turbo) | **1,135.7** (pdbs=7, FSDP=8 + `ag_combine=1 GiB` XLA tuning, Job 14602); 999.8 (pdbs=7, PP=8 + overlap2+async) | MI355 EP+FSDP tuned reaches the current best for the dropless path; PP=8 on sgd-v3 structurally loses −12% (DeepEP per-microbatch + nn.scan carry + bubble 11.1%) |
| `sparse-gmm-fixed` (sgf) | **infeasible** (same as above) | **OOM** (`ragged_all_to_all` materializes a num_ranks × tokens × hidden receive buffer; 217 GiB temp at pdbs=7) | On MI355, sgf is already replaced by sgd-v3 as an unusable path — B200 and MI355 agree this column is unavailable |

**Key observations:**

1. **At same pdbs, B200 vs MI355 on cf=1.25 leads +18.7% untuned and shrinks to +5% after tuning; on cf=2.0 same pdbs=7 it shrinks to +3.3%; on cf=4.0 same pdbs=2 it still leads +15.5%.** On cf=1.0, MI355 only has a pdbs=16 datapoint (1,598, MFU 16.01% — the highest known DS3-671B BF16 MoE MFU); no small-pdbs sweep is available, so a strict same-pdbs comparison is impossible — but MI355 cf=1.0 peak MFU (16.01%) > B200 cf=1 best MFU (14.15%), making this the cf config where B200 **clearly trails** MI355 on single-kernel utilization on the BF16 path.
2. **MI355 peak exceeds B200 peak across every cf**: cf=1.25 B200 −9.0%, cf=2.0 B200 −5.7%, cf=4.0 B200 −17.7%, **cf=1.0 B200 −20.4%** (B200 cf=1 best 1,272 vs MI355 cf=1.0 peak 1,598) — cf=1.0 is the worst-trailing cf for B200 on the BF16 path. The peak gap is mostly driven by the pdbs ceiling (B200 max=9 vs MI355 max=16; HBM 179 vs 288 GiB), but on cf=1.0 the "MI355 also wins on per-device MFU" effect stacks on top, amplifying the peak gap that would otherwise be mostly pdbs-driven.
3. **FP8 is B200's only lever to chase the MI355 peak — but after the cf=1.0 datapoint was published, even FP8 does not fully overtake the MI355 BF16 cf=1.0 peak**: B200 FP8 cf=1.0 pdbs=8 (Job 4170) = **1,521 Tok/s/dev vs MI355 BF16 cf=1.0 peak 1,598 = −4.8%**; but still higher than MI355 BF16 cf=1.25 tuned peak (1,224, +24.2%) and MI355 BF16 cf=1.25 untuned peak (1,416, +7.4%). FP8 BF16-eq MFU 16.92% is still B200's single highest BF16-eq MFU, **but absolute Tok/s/dev is still throttled by the HBM ceiling** (B200 FP8 cf=1.0 max pdbs=8 vs MI355 BF16 cf=1.0 max pdbs=16, a 2× global-batch gap). **Revised conclusion: FP8 lets B200 surpass the MI355 cf=1.25 peak but is not yet enough to overtake the MI355 cf=1.0 peak**; truly beating MI355's BF16 path requires B200 to scale to pdbs ≥ 12 as well (current FP8 cf=1.0 max pdbs=8 is the hard ceiling; pdbs=9 fails under both slop95 / mem97).
4. **B200 lacks a dropless path vs MI355**: on MI355, sgd-v3 with EP+FSDP tuning reaches **1,135.7 Tok/s/dev** (Job 14602), the currently recommended production recipe on the MI355 + JAX MoE path. On B200, `sparse_matmul` OOMs at pdbs=1 (XLA planner shape check fails), and this image does not carry the Primus-Turbo / DeepEP path, so **B200 has no comparable dropless column**. In production scenarios that require dropless (numerically convergence-sensitive), B200 + JAX 26.03 is not yet a viable platform.
5. **On MI355, the PP=8 vs FSDP=8 topology choice is path × branch dependent**: on dense-cf1.25, PP=8 tuned (1,224) **beats** FSDP=8 tuned (1,208) by ~+1.3%; but on sgd-v3, PP=8 (999.8) lags FSDP=8 (1,135.7) by −12.0% (the structural gap stacks DeepEP per-microbatch + nn.scan carry + bubble 11.1%). PP=8 topology was not swept on B200, but B200's IB already trips RDMA errors under mem97 + large buffers (Jobs 4412 / 4416 / 4424 / 4409), suggesting B200 InfiniBand may be more fragile than MI355 Pensando in the EP > ici regime — whether PP=8 holds on B200 and whether the sign flip reproduces is an open dimension for future sweeps.
6. **Equivalent key-enabler mapping (updated)**:

   | Platform / path | Key enabler 1 | Key enabler 2 | Single-enabler Tok/s/dev gain |
   |---|---|---|---|
   | MI355 dense FSDP=8 | `ag_combine_threshold=1 GiB` (splits the 8 GiB default) | *(single flag is near the ceiling)* | **+20.5%** (1,002.4 → 1,207.9) |
   | MI355 dense PP=8   | `overlap_limit=2` (2-fabric concurrency, ICI+DCN p2p) + dropping the EP+FSDP `ag=1 GiB` | *(other flags within jitter)* | **+5.8%** vs PP image default (1,156.9 → 1,224.1); +13.3% vs PP baseline-ag1G (1,080.2) |
   | MI355 sgd-v3 FSDP=8 | `ag_combine_threshold=1 GiB`        | `MAXTEXT_PATCH_BRANCH=…-v3` (eliminates the `input_scatter_fusion_*.kd` main-stream blocker) | ag-flag **+11.6%** (1,017.7 → 1,135.7); v1→v3 patch **+63%** dropless |
   | MI355 sgd-v3 PP=8   | `overlap_limit=2 + async_priority`  | *(structural bubble 11.1% + DeepEP per-microbatch are not removable)* | **+4.7%** PP=8 baseline-ag1G → 999.8 |
   | B200 BF16 dense   | `overlap_limit=4` (on top of NV defaults) | `mem97` (unlocks +1 pdbs, slight per-device drop but +14% global batch) | **+9.0%** pdbs=7 (1,183 → 1,289) / +14% global batch |
   | B200 FP8 dense    | `quantization=fp8` | `slop_factor=95` unlocks pdbs=8 | FP8 **+22.5%** vs BF16 baseline / slop95 +3.3% pdbs=7→8 |



---

## How to reproduce

```bash
# Overall best: FP8 cf=1.0 + pdbs=8 + slop95
./submit.sh deepseek3-671b::fp8-bs8-cf1-nv-slop95 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=8 capacity_factor=1.0 quantization=fp8 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=95'

# BF16 single-flag best: cf=1.25 + pdbs=7 + overlap_limit=4
./submit.sh deepseek3-671b::bf16-tune-overlap4 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=7 capacity_factor=1.25 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_experimental_parallel_collective_overlap_limit=4'

# BF16 cf=1.0 + large pdbs unlock: pdbs=9 + mem97
./submit.sh deepseek3-671b::bf16-bs9-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=9 capacity_factor=1.0 \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# BF16 cf=2.0 max pdbs: mem97 + overlap4
./submit.sh deepseek3-671b::bf16-cf2-bs7-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=7 capacity_factor=2.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_EXTRA_XLA_FLAGS=--xla_gpu_experimental_parallel_collective_overlap_limit=4' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'
```

---

## References

- Raw data: [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) (logged in Run / Job chronological order; this doc reorganizes by (precision, config) slices)
- MI355 parallel sweep: [`deepseek3-671b-pdbs-sweep.zh.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/deepseek3-671b-pdbs-sweep.zh.md) (source of structure and terminology; this doc follows its dense-cf{1.25, 1, 2, 4} + sparse five-way split)
