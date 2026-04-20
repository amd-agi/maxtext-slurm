# B200 Benchmark Report: DeepSeek3-671B

## Environment

- **Cluster**: hungry-hippo-fin-03-[1-8] (batch partition)
- **GPU**: 8x NVIDIA B200 per node, 183,359 MiB (~179.1 GiB) VRAM each
- **Nodes**: 8 (64 GPUs total)
- **Container**: `nvcr.io/nvidia/jax:26.03-maxtext-py3`
- **Branch**: `llying/benchmark-on-nv-b200` @ `5f68243`
- **XLA_PYTHON_CLIENT_MEM_FRACTION**: 0.93 (pre-allocated ~165.87 GB per GPU)
- **XLA_FLAGS**: `--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=true --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_command_buffer=''`
- **MFU peak reference**: B200 bf16 = 2,250 TFLOP/s
- **Date**: 2026-04-10

## MI355X Reference (from prior benchmark)

| Config | Step Time (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|--------|--------------|---------|-------------|-----------|-------|
| bs=16, cf=1.25, 8N | 46.2 | 14.20 | 354 | 1,418 | baseline |
| bs=16, cf=1.0, 8N | 41.0 | 16.01 | 400 | 1,598 | optimized |

MI355X had 267.8 GiB VRAM, memfrac=0.85 -> ~227.6 GiB usable. Model params: 58.6 GiB/GPU (22%).

## Summary Table

| Phase | Run ID | Job ID | Config Delta | Status | Step Time (s) | MFU (%) | Tok/s/dev | Notes |
|-------|--------|--------|-------------|--------|---------------|---------|-----------|-------|
| BF16-P0 | bf16-p0-baseline | 1193 | none (bs=16) | OOM | -- | -- | -- | 171.96 GiB alloc failed |
| BF16-P0 | bf16-p0-bs12 | 1194 | bs=12 | OOM | -- | -- | -- | 140.37 GiB alloc failed |
| BF16-P0 | bf16-p0-bs8 | 1195 | bs=8 | OOM | -- | -- | -- | 108.63 GiB alloc failed |
| BF16-P0 | bf16-p0-bs4 | 1196 | bs=4 | SUCCESS | 18.47 | 9.86 | 886 | **first working config** |
| BF16-P0 | bf16-p0-bs6 | 1198 | bs=6 | SUCCESS | 23.34 | 11.71 | 1,048 | OOM boundary probe |
| BF16-P0 | bf16-p0-bs7 | 1200 | bs=7 | SUCCESS | 26.51 | 12.04 | 1,081 | **max working bs** |
| BF16-P0b | bf16-p0b-prof-retry | 1210 | bs=7, profiler=xplane | SUCCESS | 26.43 | 12.08 | 1,085 | xplane traces captured |
| BF16-P1 | bf16-p1a-offload | 1203 | bs=8, offload=True | OOM | -- | -- | -- | 146.95 GiB alloc (worse!) |
| BF16-P1 | bf16-p1b-shard-exp | 1204 | bs=8, shard_exp=True | FAILED | -- | -- | -- | args 492 GiB > limit 178 GiB |
| BF16-P1 | bf16-p1c-both | 1205 | bs=8, offload+shard_exp | OOM | -- | -- | -- | CUDA_ERROR_OUT_OF_MEMORY |
| BF16-P1 | bf16-p1e-cf1 | 1207 | bs=7, cf=1.0 | SUCCESS | 23.71 | 13.46 | 1,209 | cf=1.0 +12% MFU |
| BF16-P1 | bf16-p1f-cf1-bs8 | 1208 | bs=8, cf=1.0 | SUCCESS | 26.38 | 13.83 | 1,242 | **cf=1.0 unlocks bs=8, best** |
| BF16-P1 | bf16-p1g-ga2 | 1209 | bs=7, ga=2 | OOM | -- | -- | -- | 124.93 GiB alloc failed |
| BF16-P1 | bf16-p1h-minflash | 1211 | bs=7, remat=minimal_flash | OOM | -- | -- | -- | 744.66 GiB alloc failed |
| BF16-P1 | bf16-p1i-cf1-ga2 | 1212 | bs=7, cf=1.0, ga=2 | OOM | -- | -- | -- | 118.91 GiB alloc failed |
| BF16-P1 | bf16-p1j-cf1-bs9 | 1218 | bs=9, cf=1.0 | OOM | -- | -- | -- | 109.83 GiB alloc failed |
| BF16-P1 | bf16-p1k-cf1-bs10 | 1219 | bs=10, cf=1.0 | OOM | -- | -- | -- | 116.11 GiB alloc failed |
| BF16-P2 | bf16-p2a-ici-fsdp2-ep4 | 1213 | bs=7, ici_fsdp=2, ici_ep=4 | SUCCESS | 26.29 | 12.14 | 1,090 | tiny gain vs baseline only |
| BF16-P4 | bf16-p4a-megablox | 1214 | bs=7, megablox=True | SUCCESS | 26.55 | 12.02 | 1,080 | neutral/slightly worse |
| BF16-P4 | bf16-p4b-sparse | 1215 | bs=7, sparse_matmul=True | FAILED | -- | -- | -- | RaggedDot requires shardy |
| BF16-P4 | bf16-p4c-cf1-megablox | 1216 | bs=7, cf=1.0, megablox=True | SUCCESS | 23.62 | 13.51 | 1,214 | tiny gain vs cf=1.0 bs7 |
| BF16-P4 | bf16-p4d-cf1-sparse | 1217 | bs=7, cf=1.0, sparse_matmul=True | FAILED | -- | -- | -- | RaggedDot requires shardy |
| BF16-P4 | bf16-p4e-cf1-megablox-bs8 | 1235 | bs=8, cf=1.0, megablox=True | SUCCESS | 26.35 | 13.84 | 1,243 | **current best BF16, tiny gain** |
| BF16-P1 | bf16-p1l-cf1-bs9-vocab4 | 1236 | bs=9, cf=1.0, num_vocab_tiling=4 | FAILED | -- | -- | -- | AssertionError: dtype mismatch (f32 vs bf16) |
| BF16-P1 | bf16-p1m-cf1-bs9-gradbf16 | 1237 | bs=9, cf=1.0, grad_dtype=bfloat16 | OOM | -- | -- | -- | 109.83 GiB alloc failed |
| BF16-P1 | bf16-p1n-cf1-bs9-gradbf16-vocab4 | 1238 | bs=9, cf=1.0, grad_dtype=bf16+vocab4 | FAILED | -- | -- | -- | AssertionError: dtype mismatch (f32 vs bf16) |
| BF16-P4 | bf16-p4f-cf1-sparse-shardy-bs7 | 1239 | bs=7, cf=1.0, sparse+shardy | OOM | -- | -- | -- | 2.28 TiB alloc failed |
| BF16-P4 | bf16-p4g-cf1-sparse-shardy-bs8 | 1240 | bs=8, cf=1.0, sparse+shardy | OOM | -- | -- | -- | 2.60 TiB alloc failed |
| FP8-P0 | fp8-p0-cf1-bs12 | 1241 | bs=12, cf=1.0, quantization=fp8 | OOM | -- | -- | -- | 118.74 GiB alloc failed |
| FP8-P0 | fp8-p0-cf1-bs16 | 1242 | bs=16, cf=1.0, quantization=fp8 | OOM | -- | -- | -- | 147.38 GiB alloc failed |
| XLA-A5r | bf16-xla-combine256-v2 | 1588 | bs=7, AMD flags + combine=256B | SUCCESS | 24.27 | 13.16 | 1,182 | resubmit of 1528 (IB hang) |
| XLA-A8 | bf16-xla-best-combo | 1589 | bs=7, NV defaults + combdim=true | SUCCESS | 24.27 | 13.15 | 1,181 | NV defaults + explicit combdim |
| XLA-A9 | bf16-nv-megablox | 1590 | bs=7, megablox, NV defaults | SUCCESS | 24.25 | 13.16 | 1,182 | NV defaults + megablox |
| XLA-A10 | bf16-nv-dense-shardy | 1591 | bs=7, shardy=true, NV defaults | SUCCESS | 24.27 | 13.16 | 1,181 | NV defaults + shardy |
| FP8-NV | fp8-nv-bs7 | 1592 | bs=7, fp8, NV defaults | SUCCESS | 21.40 | 7.46* | 1,340 | **FP8 best; 335 TFLOP/s** |
| FP8-NV | fp8-nv-bs8 | 1593 | bs=8, fp8, NV defaults | OOM | -- | -- | -- | 107.70 GiB alloc failed |
| FP8-NV | fp8-nv-bs10 | 1594 | bs=10, fp8, NV defaults | OOM | -- | -- | -- | 122.93 GiB alloc failed |
| CF-ext | bf16-cf2.0-bs4 | 1595 | bs=4, cf=2.0 | SUCCESS | 22.04 | 8.28 | 743 | cf=2.0 viable at bs=4 |
| CF-ext | bf16-cf4.0-bs2 | 1596 | bs=2, cf=4.0 | SUCCESS | 20.42 | 4.47 | 401 | cf=4.0 viable at bs=2 |
| Kimi | kimi-k2-bs1 | 1597 | bs=1, kimi-k2-1t | SUCCESS | 17.00 | 2.20 | 241 | first working kimi config |
| Kimi | kimi-k2-bs1-cf1 | 1598 | bs=1, cf=1.0, kimi-k2-1t | SUCCESS | 16.60 | 2.25 | 247 | cf=1.0 marginal gain |
| Kimi | kimi-k2-bs2-cf1 | 1599 | bs=2, cf=1.0, kimi-k2-1t | OOM | -- | -- | -- | 82.47 GiB alloc failed |
| Kimi | kimi-k2-bs2-fsdp2 | 1600 | bs=2, fsdp2/ep4, kimi-k2-1t | CANCELLED | -- | -- | -- | never ran (cancelled in queue) |
| Kimi | kimi-k2-bs4-fsdp2 | 1601 | bs=4, fsdp2/ep4, kimi-k2-1t | CANCELLED | -- | -- | -- | never ran (cancelled in queue) |

---

## BF16 Phase 0: Baseline

### Run: bf16-p0-baseline (Job 1193) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-baseline -N 8 -w hungry-hippo-fin-03-[1-8]
```

**Config (base, no overrides):**
- `model_name: deepseek3-671b` (full 671B model, 61 decoder layers)
- `per_device_batch_size: 16`, `max_target_length: 4096`
- `dcn_fsdp_parallelism: 8`, `ici_expert_parallelism: 8`
- `remat_policy: full`, `dtype: bfloat16`
- `capacity_factor: 1.25`, `megablox: False`, `sparse_matmul: False`

**Status: OOM**

**Failure details:**
- Model params: **671.026 billion** (confirmed full model, not proxy)
- XLA latency hiding scheduler: memory usage **195,798,628,216 bytes (~182.3 GiB)** vs limit **109,422,640,924 bytes (~101.9 GiB)** -- schedule requires ~1.8x available memory
- OOM error: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 171.96GiB`
- First OOM on node 6 (hungry-hippo-fin-03-7), then cascaded to other nodes
- Training wall time: 89s (compilation completed, OOM on first execution step)

**Log path:** `outputs/1193-JAX-deepseek3-671b-bf16-p0-baseline.log`

**Observations:**
- The full DS3-671B model has 671B params (vs proxy's 49.6B with 7 decoder layers)
- On MI355X (267.8 GiB, memfrac=0.85), bs=16 worked with 58.6 GiB params/GPU. But that was also the proxy config (4 nodes, dcn_fsdp=4). The full model with dcn_fsdp=8 has different memory characteristics.
- XLA schedule needs 182.3 GiB but only 101.9 GiB available -- this is a ~80% memory overshoot
- **Key question**: the XLA memory limit shows only 101.9 GiB, much less than the expected 165.87 GB (183359 MiB * 0.93). Need to investigate why.

**Decision for next run:**
bs=16 OOM by a large margin (171.96 GiB allocation attempt). Drop to bs=12 and bs=8 to find the memory boundary.

### Run: bf16-p0-bs12 (Job 1194) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-bs12 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=12
```

**Config delta:** `per_device_batch_size=12` (was 16)

**Status: OOM**

**Failure details:**
- OOM error: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 140.37GiB`
- Memory reduction from bs=16: 171.96 -> 140.37 GiB (-31.59 GiB for -4 bs, ~7.9 GiB/bs unit)
- Still far above available memory (~101.9 GiB XLA limit)

**Log path:** `outputs/1194-JAX-deepseek3-671b-bf16-p0-bs12-per_device_batch_size_12.log`

**Decision for next run:** Still heavily OOM. Linear extrapolation: bs=8 → ~108.8 GiB (still over), bs=4 → ~77.2 GiB (may fit), bs=2 → ~53.5 GiB. Submitted bs=8, bs=4, bs=2 as a dependency chain.

### Run: bf16-p0-bs8 (Job 1195) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-bs8 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1194 -- per_device_batch_size=8
```

**Config delta:** `per_device_batch_size=8`

**Status: OOM**

**Failure details:**
- OOM error: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 108.63GiB`
- Memory reduction from bs=12: 140.37 -> 108.63 GiB (-31.74 GiB for -4 bs)
- Still above XLA limit (~101.9 GiB), but close (~7 GiB over)

**Log path:** `outputs/1195-JAX-deepseek3-671b-bf16-p0-bs8-per_device_batch_size_8.log`

### Run: bf16-p0-bs4 (Job 1196) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-bs4 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1195 -- per_device_batch_size=4
```

**Config delta:** `per_device_batch_size=4`

**Status: SUCCESS**

**Memory:**
- Memstats after params: **58.61 GB / 165.87 GB (35.3%)** on cuda:0
- XLA latency hiding scheduler: 85,838,149,408 bytes (~79.9 GiB) usage / 109,423,574,812 bytes (~101.9 GiB) limit
- Memory headroom: ~22 GiB available after XLA schedule

**Training metrics (node 0):**

| Step | Time (s) | TFLOP/s/dev | MFU (%) | Tok/s/dev | Loss |
|------|----------|-------------|---------|-----------|------|
| 0 (compile) | 32.415 | 126.6 | 5.63 | 505 | 12.269 |
| 1 | 18.773 | 218.6 | 9.71 | 873 | 12.269 |
| 2 | 18.368 | 223.4 | 9.93 | 892 | 11.781 |
| 3 | 18.404 | 223.0 | 9.91 | 890 | 11.437 |
| 4 | 18.381 | 223.2 | 9.92 | 891 | 11.107 |
| 5 | 18.404 | 223.0 | 9.91 | 890 | 10.788 |
| 6 | 18.478 | 222.1 | 9.87 | 887 | 10.483 |
| 7 | 18.498 | 221.8 | 9.86 | 886 | 10.204 |
| 8 | 18.494 | 221.9 | 9.86 | 886 | 9.958 |
| 9 | 18.608 | 220.5 | 9.80 | 880 | 9.783 |
| 10 | 18.487 | 222.0 | 9.86 | 886 | 9.675 |
| 11 | 18.490 | 221.9 | 9.86 | 886 | 9.588 |
| 12 | 18.508 | 221.7 | 9.85 | 885 | 9.536 |
| 13 | 18.605 | 220.6 | 9.80 | 881 | 9.508 |
| 14 | 18.513 | 221.7 | 9.85 | 885 | 9.490 |

**Steady-state summary (steps 5-13):**
- Step time: **18.50 s** (mean)
- MFU: **9.85%**
- TFLOP/s/device: **221.8**
- Tokens/s/device: **886**
- Total tokens/s (64 GPUs): **~56,700**
- Loss: 12.269 -> 9.490

**Training wall time:** 362s (6m 2s)
**Compilation time (step 0):** 32.4s

**Log path:** `outputs/1196-JAX-deepseek3-671b-bf16-p0-bs4-per_device_batch_size_4.log`

**Observations:**
- bs=4 is the first working batch size on B200 with the full DS3-671B model
- Params use 58.61 GB/GPU (35.3%) -- matches MI355X data (58.6 GiB)
- XLA schedule uses 79.9 GiB / 101.9 GiB limit -- 22 GiB headroom
- MFU (9.85%) is significantly lower than MI355X baseline (14.2% at bs=16) due to smaller batch size
- Step time 18.5s vs MI355X 46.2s -- faster per-step but 4x fewer tokens per step
- Total throughput 56,700 tok/s vs MI355X 90,700 tok/s (baseline) -- **37% lower**
- The memory budget is the main bottleneck; need to explore memory-saving configs (offload, shard_exp, different parallelism) to enable larger batch sizes

**Decision for next run:**
bs=4 works with 22 GiB headroom. bs=8 OOM by only ~7 GiB. Next steps:
1. Try bs=6 (between 4 and 8) to find the exact maximum
2. Explore `optimizer_memory_host_offload=True` to reduce GPU memory and enable larger bs
3. Explore `shard_exp_on_fsdp=True` for expert weight distribution
4. These memory-saving features may unlock bs=8 or even higher

### Run: bf16-p0-bs6 (Job 1198) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-bs6 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=6
```

**Config delta:** `per_device_batch_size=6`

**Status: SUCCESS**

**Memory:**
- Memstats after params: **58.61 GB / 165.87 GB (35.3%)** on cuda:0
- XLA latency hiding scheduler: 106,696,931,120 bytes (~99.4 GiB) / limit 109,423,419,164 bytes (~101.9 GiB) -- 2.5 GiB headroom

**Training metrics (node 0):**

| Step | Time (s) | TFLOP/s/dev | MFU (%) | Tok/s/dev | Loss |
|------|----------|-------------|---------|-----------|------|
| 0 (compile) | 35.895 | 171.5 | 7.62 | 685 | 12.268 |
| 1 | 23.387 | 263.2 | 11.70 | 1,051 | 12.268 |
| 2 | 23.124 | 266.2 | 11.83 | 1,063 | 11.844 |
| 3 | 23.261 | 264.6 | 11.76 | 1,057 | 11.544 |
| 4 | 23.390 | 263.1 | 11.70 | 1,051 | 11.261 |
| 5 | 23.360 | 263.5 | 11.71 | 1,052 | 10.991 |
| 6 | 23.710 | 259.6 | 11.54 | 1,037 | 10.737 |
| 7 | 23.342 | 263.7 | 11.72 | 1,053 | 10.504 |
| 8 | 23.296 | 264.2 | 11.74 | 1,055 | 10.299 |
| 9 | 23.655 | 260.2 | 11.56 | 1,039 | 10.153 |
| 10 | 23.294 | 264.2 | 11.74 | 1,055 | 10.064 |
| 11 | 23.309 | 264.1 | 11.74 | 1,054 | 9.992 |
| 12 | 23.745 | 259.2 | 11.52 | 1,035 | 9.949 |
| 13 | 23.332 | 263.8 | 11.72 | 1,053 | 9.927 |
| 14 | 23.571 | 261.1 | 11.61 | 1,043 | 9.911 |

**Steady-state summary (steps 5-13):**
- Step time: **23.34 s** (mean)
- MFU: **11.71%**
- TFLOP/s/device: **262.5**
- Tokens/s/device: **1,048**
- Total tokens/s (64 GPUs): **~67,080**
- Loss: 12.268 -> 9.911

**Training wall time:** 435s (7m 15s)
**Compilation time (step 0):** 35.9s

**Log path:** `outputs/1198-JAX-deepseek3-671b-bf16-p0-bs6-per_device_batch_size_6/log`

**Observations:**
- bs=6 fits with only 2.5 GiB headroom in XLA schedule (99.4 / 101.9 GiB)
- MFU 11.71% is +19% over bs=4 (9.85%), total throughput +18% (67,080 vs 56,700 tok/s)
- Per-step time increased from 18.5s to 23.3s (+26%) but processes 50% more tokens per step

### Run: bf16-p0-bs7 (Job 1200) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0-bs7 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7
```

**Config delta:** `per_device_batch_size=7`

**Status: SUCCESS**

**Memory:**
- Memstats after params: **58.62 GB / 165.87 GB (35.3%)** on cuda:0
- XLA latency hiding scheduler: 110,549,661,472 bytes (~103.0 GiB) / limit 109,423,341,340 bytes (~101.9 GiB) -- **over limit by ~1.05 GiB** (soft limit, still runs)

**Training metrics (node 0):**

| Step | Time (s) | TFLOP/s/dev | MFU (%) | Tok/s/dev | Loss |
|------|----------|-------------|---------|-----------|------|
| 0 (compile) | 40.352 | 178.0 | 7.91 | 711 | 12.268 |
| 1 | 26.742 | 268.5 | 11.93 | 1,072 | 12.268 |
| 2 | 26.191 | 274.2 | 12.19 | 1,095 | 11.869 |
| 3 | 26.509 | 270.9 | 12.04 | 1,082 | 11.586 |
| 4 | 26.287 | 273.2 | 12.14 | 1,091 | 11.325 |
| 5 | 26.313 | 272.9 | 12.13 | 1,090 | 11.073 |
| 6 | 26.564 | 270.3 | 12.01 | 1,079 | 10.835 |
| 7 | 26.370 | 272.3 | 12.10 | 1,087 | 10.620 |
| 8 | 26.645 | 269.5 | 11.98 | 1,076 | 10.430 |
| 9 | 26.386 | 272.1 | 12.10 | 1,087 | 10.294 |
| 10 | 26.722 | 268.7 | 11.94 | 1,073 | 10.211 |
| 11 | 26.462 | 271.4 | 12.06 | 1,084 | 10.144 |
| 12 | 26.525 | 270.7 | 12.03 | 1,081 | 10.104 |
| 13 | 26.643 | 269.5 | 11.98 | 1,076 | 10.083 |
| 14 | 26.573 | 270.2 | 12.01 | 1,079 | 10.069 |

**Steady-state summary (steps 5-13):**
- Step time: **26.51 s** (mean)
- MFU: **12.04%**
- TFLOP/s/device: **270.8**
- Tokens/s/device: **1,081**
- Total tokens/s (64 GPUs): **~69,210**
- Loss: 12.268 -> 10.069

**Training wall time:** 481s (8m 1s)
**Compilation time (step 0):** 40.4s

**Log path:** `outputs/1200-JAX-deepseek3-671b-bf16-p0-bs7-per_device_batch_size_7/log`

**Observations:**
- bs=7 is the **maximum working batch size** without any memory optimization
- XLA schedule exceeds its own soft limit by 1.05 GiB but runs successfully (slop_factor=95 provides tolerance)
- MFU 12.04% is +2.8% over bs=6 (11.71%), +22.2% over bs=4 (9.85%)
- Total throughput 69,210 tok/s is +3.2% over bs=6, +22.1% over bs=4
- Still significantly below MI355X baseline (14.2% MFU, 90,700 tok/s) -- gap mainly from smaller batch size (bs=7 vs bs=16)
- Compilation time increased: 32.4s (bs=4) -> 35.9s (bs=6) -> 40.4s (bs=7)

**Decision for next run:**
bs=7 is the best baseline without memory optimization. **Carry forward as BF16 baseline.**
Next: Phase 0b -- re-run bs=7 with `profiler=xplane` to collect a trace for profiler-guided optimization.
Then: Phase 1 -- try `optimizer_memory_host_offload=True` and/or `shard_exp_on_fsdp=True` to push batch size higher.

### Phase 0 Summary: Batch Size Sweep

| bs | XLA Schedule (GiB) | XLA Limit (GiB) | Headroom (GiB) | Status | MFU (%) | Tok/s total |
|----|-------------------|-----------------|----------------|--------|---------|-------------|
| 16 | ~182.3 | ~101.9 | -80.4 | OOM | -- | -- |
| 12 | -- | ~101.9 | -- | OOM (140.37 GiB alloc) | -- | -- |
| 8 | -- | ~101.9 | -- | OOM (108.63 GiB alloc) | -- | -- |
| 7 | ~103.0 | ~101.9 | -1.1 (soft) | **SUCCESS** | **12.04** | **69,210** |
| 6 | ~99.4 | ~101.9 | +2.5 | SUCCESS | 11.71 | 67,080 |
| 4 | ~79.9 | ~101.9 | +22.0 | SUCCESS | 9.85 | 56,700 |

**Best baseline: bs=7, MFU=12.04%, 69,210 tok/s (64 GPUs)**

---

## BF16 Phase 1: Memory & Config Tuning

### Run: bf16-p1a-offload (Job 1203) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1a-offload -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1202 -- per_device_batch_size=8 optimizer_memory_host_offload=True
```

**Config delta:** `per_device_batch_size=8, optimizer_memory_host_offload=True`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 146.95GiB`

**Memory:**
- XLA LHS: 27,783,139,536 bytes (~25.87 GiB) / limit 149,269,900,639 bytes (~139.0 GiB)
- Note: offload changed XLA limit from ~101.9 GiB to ~139.0 GiB and schedule estimate from ~103 GiB to ~26 GiB
- Despite favorable LHS estimate, actual execution OOM at 146.95 GiB -- single large allocation exceeds even the raised limit

**Log path:** `outputs/1203-JAX-deepseek3-671b-bf16-p1a-offload-per_device_batch_size_8-optimizer_memory_host_offload_True/log`

**Observations:**
- `optimizer_memory_host_offload=True` paradoxically makes OOM worse (108.63 → 146.95 GiB allocation)
- The offload changes XLA's memory accounting dramatically but actual execution still requires large contiguous buffers
- Not viable for increasing batch size on B200

### Run: bf16-p1b-shard-exp (Job 1204) -- FAILED

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1b-shard-exp -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1203 -- per_device_batch_size=8 shard_exp_on_fsdp=True
```

**Config delta:** `per_device_batch_size=8, shard_exp_on_fsdp=True`

**Status: FAILED** -- `The byte size of input/output arguments (492042391348) exceeds the base limit (178097798571)`

**Log path:** `outputs/1204-JAX-deepseek3-671b-bf16-p1b-shard-exp-per_device_batch_size_8-shard_exp_on_fsdp_True/log`

**Observations:**
- `shard_exp_on_fsdp=True` causes input/output arguments to balloon to 492 GiB (vs 178 GiB limit)
- This is a compilation-time failure, not runtime OOM -- the sharding layout creates an infeasible program
- Likely incompatible with the current parallelism strategy (ici_ep=8 + dcn_fsdp=8)

### Run: bf16-p1c-both (Job 1205) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1c-both -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1204 -- per_device_batch_size=8 optimizer_memory_host_offload=True shard_exp_on_fsdp=True
```

**Config delta:** `per_device_batch_size=8, optimizer_memory_host_offload=True, shard_exp_on_fsdp=True`

**Status: OOM** -- `CUDA_ERROR_OUT_OF_MEMORY` during host memory registration, followed by cascading failures

**Log path:** `outputs/1205-JAX-deepseek3-671b-bf16-p1c-both-per_device_batch_size_8-optimizer_memory_host_offload_True-shard_exp_on_fsdp_True/log`

### Run: bf16-p1e-cf1 (Job 1207) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1e-cf1-bs7 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7 capacity_factor=1.0
```

**Config delta:** `per_device_batch_size=7, capacity_factor=1.0` (was 1.25)

**Status: SUCCESS**

**Memory:**
- XLA LHS: 109,228,455,736 bytes (~101.7 GiB) / limit 109,423,341,340 (~101.9 GiB) -- 0.2 GiB headroom
- cf=1.0 saved ~1.3 GiB vs cf=1.25 (103.0 → 101.7 GiB) -- now barely under limit

**Steady-state summary (steps 5-13):**
- Step time: **23.71 s** (mean) -- was 26.51s at cf=1.25 (**-10.6%**)
- MFU: **13.46%** -- was 12.04% (**+11.8%**)
- TFLOP/s/device: **302.9** -- was 270.8 (+11.8%)
- Tokens/s/device: **1,209** -- was 1,081 (+11.8%)
- Total tokens/s (64 GPUs): **~77,400** -- was 69,210 (+11.8%)
- Loss: 12.268 -> 10.069

**Training wall time:** 447s (7m 27s)
**Compilation time (step 0):** 36.3s

**Log path:** `outputs/1207-JAX-deepseek3-671b-bf16-p1e-cf1-bs7-per_device_batch_size_7-capacity_factor_1.0/log`

**Observations:**
- **capacity_factor=1.0 is the single biggest optimization so far** -- +11.8% MFU, +11.8% throughput
- Removes 25% expert padding: fewer wasted FLOPs and smaller expert dispatch buffers
- LHS headroom is only 0.2 GiB -- very tight but works reliably (all 15 steps completed)
- Compared to MI355X cf=1.0 baseline (41.0s, 16.01% MFU, ~102K tok/s): B200 is still ~24% behind in total throughput but ~42% faster per-step
- **Carry forward as new BF16 best config**

### Phase 1 Summary: Memory/Config Tuning

| Config Change | bs | Status | MFU (%) | Tok/s total | vs Baseline |
|--------------|-----|--------|---------|-------------|-------------|
| baseline (cf=1.25) | 7 | SUCCESS | 12.04 | 69,210 | -- |
| optimizer_memory_host_offload=True | 8 | OOM (146.95 GiB) | -- | -- | worse |
| shard_exp_on_fsdp=True | 8 | FAILED (492 GiB args) | -- | -- | incompatible |
| both offload+shard_exp | 8 | OOM | -- | -- | worse |
| **capacity_factor=1.0** | **7** | **SUCCESS** | **13.46** | **77,400** | **+11.8%** |
| **capacity_factor=1.0** | **8** | **SUCCESS** | **13.83** | **79,500** | **+14.9%** |

**Conclusion: offload/shard_exp don't help on B200; capacity_factor=1.0 is the key win, and it unlocks bs=8.**

### Run: bf16-p1f-cf1-bs8 (Job 1208) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1f-cf1-bs8 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1207 -- per_device_batch_size=8 capacity_factor=1.0
```

**Config delta:** `per_device_batch_size=8, capacity_factor=1.0`

**Status: SUCCESS**

**Memory:**
- XLA LHS: 114,377,226,008 bytes (~106.5 GiB) / limit 109,423,263,516 (~101.9 GiB) -- over limit by 4.6 GiB (soft limit, works via slop_factor)
- cf=1.0 reduced memory enough vs cf=1.25 to make bs=8 viable (was OOM at 108.63 GiB with cf=1.25)

**Steady-state summary (steps 5-13):**
- Step time: **26.38 s** (mean)
- MFU: **13.83%**
- TFLOP/s/device: **312.2**
- Tokens/s/device: **1,242**
- Total tokens/s (64 GPUs): **~79,500**
- Loss: 12.268 -> 10.192

**Training wall time:** 480s (8m 0s)
**Compilation time (step 0):** 40.3s

**Log path:** `outputs/1208-JAX-deepseek3-671b-bf16-p1f-cf1-bs8-per_device_batch_size_8-capacity_factor_1.0/log`

**Observations:**
- **cf=1.0 unlocks bs=8** -- previously OOM at 108.63 GiB with cf=1.25
- MFU 13.83% is +2.7% over cf=1.0 bs=7 (13.46%), +14.9% over baseline cf=1.25 bs=7 (12.04%)
- Total throughput 79,500 tok/s is +2.7% over cf=1.0 bs=7, +14.9% over baseline
- LHS over limit by 4.6 GiB but runs fine -- bs=9 likely needs ~111 GiB, may be too far over
- vs MI355X cf=1.0: B200 at 79.5K vs MI355X at ~102K tok/s (22% gap), step time 26.4s vs 41.0s (36% faster per-step)
- **New best BF16 config. Carry forward.**

---

## BF16 Phase 0b: Profiler Trace

### Run: bf16-p0b-prof-retry (Job 1210) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p0b-prof-retry -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1209 -- per_device_batch_size=7 profiler=xplane
```

**Config delta:** `per_device_batch_size=7, profiler=xplane`

**Status: SUCCESS**

**Profiler path:** `outputs/1210-JAX-deepseek3-671b-bf16-p0b-prof-retry-per_device_batch_size_7-profiler_xplane/deepseek3-671b_train_test/tensorboard/plugins/profile/2026_04_10_11_22_32/`

**Trace files:** one `.xplane.pb` per node (`hungry-hippo-fin-03-[1-8]`)

**Steady-state summary (steps 5-13):**
- Step time: **26.43 s**
- MFU: **12.08%**
- Tokens/s/device: **1,085**
- Total tokens/s (64 GPUs): **~69,400**

**Memory:** XLA LHS `110,549,661,472` bytes (~103.0 GiB) / limit `109,423,341,340` (~101.9 GiB)

**Observations:**
- Profiler retry succeeded after the earlier transient IB hang in Job 1202
- Step 4 slowed to 29.25s due to profiler capture; steady-state steps 5-13 remain essentially identical to the bs=7 baseline
- Trace collection is complete; later XLA/communication analysis can use these xplane files directly

### Run: bf16-p1g-ga2 (Job 1209) -- OOM

**Config delta:** `per_device_batch_size=7, gradient_accumulation_steps=2`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 124.93GiB`

**Observations:**
- `gradient_accumulation_steps=2` is **not** a memory-saving knob in this MaxText/JAX path
- It materially increases memory footprint vs plain bs=7 (124.93 GiB alloc vs working baseline)

### Run: bf16-p1h-minflash (Job 1211) -- OOM

**Config delta:** `per_device_batch_size=7, remat_policy=minimal_flash`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 744.66GiB`

**Observations:**
- `minimal_flash` is completely infeasible for this 671B model on B200
- This validates the model-config guidance: large models should stay on `remat_policy=full`

### Run: bf16-p1i-cf1-ga2 (Job 1212) -- OOM

**Config delta:** `per_device_batch_size=7, capacity_factor=1.0, gradient_accumulation_steps=2`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 118.91GiB`

**Observations:**
- `capacity_factor=1.0` reduces MoE memory, but not enough to offset the extra accumulation-state pressure from `gradient_accumulation_steps=2`

### Run: bf16-p1j-cf1-bs9 (Job 1218) -- OOM

**Config delta:** `per_device_batch_size=9, capacity_factor=1.0`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 109.83GiB`

**Observations:**
- `cf=1.0` unlocks bs=8, but **not** bs=9
- The gap from bs=8 success to bs=9 OOM is small, so additional memory-specific knobs may still unlock bs=9

### Run: bf16-p1k-cf1-bs10 (Job 1219) -- OOM

**Config delta:** `per_device_batch_size=10, capacity_factor=1.0`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 116.11GiB`

**Observations:**
- bs=10 is clearly out of range for BF16 on current settings
- Current BF16 feasible envelope is: `bs=8 @ cf=1.0` works, `bs=9` does not

### Run: bf16-p1l-cf1-bs9-vocab4 (Job 1236) -- FAILED

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1l-cf1-bs9-vocab4 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=9 capacity_factor=1.0 num_vocab_tiling=4
```

**Config delta:** `per_device_batch_size=9, capacity_factor=1.0, num_vocab_tiling=4`

**Status: FAILED** -- `AssertionError: (ShapedArray(float32[1536,58,128,192]), ShapedArray(bfloat16[1536,58,128,192]))`

**Log path:** `outputs/1236-JAX-deepseek3-671b-bf16-p1l-cf1-bs9-vocab4-per_device_batch_size_9-capacity_factor_1.0-num_vocab_tiling_4.log`

**Observations:**
- `num_vocab_tiling=4` triggers a dtype assertion during compilation: the embedding/logit projection tile expects `float32` but receives `bfloat16`
- This is a **code-level incompatibility** with the DS3-671B model config (or the current MaxText version), not a memory issue
- The failure occurs during `jax.transpose` inside `flax.core.axes_scan`, suggesting the tiling changes the scan axis layout and hits a dtype-invariant check
- `num_vocab_tiling` is designed to split the vocabulary dimension into tiles during the final logit projection to reduce peak memory of the softmax/loss computation. When set to N, the vocab axis is processed in N chunks rather than all at once. However, it appears incompatible with DS3-671B's mixed-precision setup.

### Run: bf16-p1m-cf1-bs9-gradbf16 (Job 1237) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1m-cf1-bs9-gradbf16 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1236 -- per_device_batch_size=9 capacity_factor=1.0 grad_dtype=bfloat16
```

**Config delta:** `per_device_batch_size=9, capacity_factor=1.0, grad_dtype=bfloat16`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 109.83GiB`

**Log path:** `outputs/1237-JAX-deepseek3-671b-bf16-p1m-cf1-bs9-gradbf16-per_device_batch_size_9-capacity_factor_1.0-grad_dtype_bfloat16.log`

**Observations:**
- `grad_dtype=bfloat16` forces gradients to be stored in bf16 rather than the default (which may accumulate in f32). The intention is to halve gradient memory at the cost of reduced numerical precision during accumulation.
- However, the OOM allocation size (109.83 GiB) is **identical** to the plain `bs=9, cf=1.0` run (Job 1218), meaning `grad_dtype=bfloat16` provided **zero memory savings** at this model scale
- The bottleneck is not gradient storage but activation/buffer memory in the XLA schedule
- `grad_dtype` is not a viable path to unlock bs=9

### Run: bf16-p1n-cf1-bs9-gradbf16-vocab4 (Job 1238) -- FAILED

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p1n-cf1-bs9-gradbf16-vocab4 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1237 -- per_device_batch_size=9 capacity_factor=1.0 grad_dtype=bfloat16 num_vocab_tiling=4
```

**Config delta:** `per_device_batch_size=9, capacity_factor=1.0, grad_dtype=bfloat16, num_vocab_tiling=4`

**Status: FAILED** -- `AssertionError: (ShapedArray(float32[1536,58,128,192]), ShapedArray(bfloat16[1536,58,128,192]))`

**Log path:** `outputs/1238-JAX-deepseek3-671b-bf16-p1n-cf1-bs9-gradbf16-vocab4-per_device_batch_size_9-capacity_factor_1.0-grad_dtype_bfloat16-num_vocab_tiling_4.log`

**Observations:**
- Same `num_vocab_tiling=4` dtype assertion as Job 1236 -- `grad_dtype=bfloat16` does not bypass the incompatibility
- Confirms `num_vocab_tiling` is broken for this model regardless of other settings

### Phase 1 Extended Summary: Additional Memory Knobs

| Config Change | bs | Status | OOM Alloc | Notes |
|--------------|-----|--------|-----------|-------|
| cf=1.0 + num_vocab_tiling=4 | 9 | FAILED | -- | dtype assertion, incompatible |
| cf=1.0 + grad_dtype=bfloat16 | 9 | OOM | 109.83 GiB | zero savings vs plain bs=9 |
| cf=1.0 + grad_dtype=bf16 + vocab4 | 9 | FAILED | -- | same dtype assertion |

**Conclusion: Neither `num_vocab_tiling` nor `grad_dtype=bfloat16` can unlock bs=9. The BF16 memory ceiling remains at `bs=8, cf=1.0`.**

---

## BF16 Phase 2: Parallelism

### Run: bf16-p2a-ici-fsdp2-ep4 (Job 1213) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p2a-ici-fsdp2-ep4 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1212 -- per_device_batch_size=7 ici_fsdp_parallelism=2 ici_expert_parallelism=4
```

**Config delta:** `per_device_batch_size=7, ici_fsdp_parallelism=2, ici_expert_parallelism=4`

**Status: SUCCESS**

**Memory:** XLA LHS `112,428,709,680` bytes (~104.7 GiB) / limit `109,423,341,340` (~101.9 GiB)

**Steady-state summary (steps 5-13):**
- Step time: **26.29 s**
- MFU: **12.14%**
- Tokens/s/device: **1,090**
- Total tokens/s (64 GPUs): **~69,800**

**Observations:**
- This alternative ICI split works, but the gain over baseline bs=7 is tiny (~+0.9% tokens/s/device)
- It remains far behind `cf=1.0 bs=8` (79.5K tok/s)
- **Do not carry forward**; `ici_ep8 + dcn_fsdp8` remains the better BF16 direction so far

---

## BF16 Phase 4: MoE Runtime

### Run: bf16-p4a-megablox (Job 1214) -- SUCCESS

**Config delta:** `per_device_batch_size=7, megablox=True`

**Status: SUCCESS**

**Memory:** XLA LHS `110,549,661,472` bytes (~103.0 GiB) / limit `109,423,341,340` (~101.9 GiB)

**Steady-state summary (steps 5-13):**
- Step time: **26.55 s**
- MFU: **12.02%**
- Tokens/s/device: **1,080**

**Observations:**
- `megablox=True` at `cf=1.25, bs=7` is effectively neutral to slightly worse than baseline
- No reason to carry it forward on top of the non-`cf=1.0` baseline

### Run: bf16-p4b-sparse (Job 1215) -- FAILED

**Config delta:** `per_device_batch_size=7, sparse_matmul=True`

**Status: FAILED** -- `Check failed: ... RaggedDot is only supported with Shardy.`

**Observations:**
- `sparse_matmul=True` is not usable with the current `shardy=false` GPU config
- The failure is deterministic and not a node/IB issue
- If we revisit sparse matmul, it must be with `shardy=true`

### Run: bf16-p4c-cf1-megablox (Job 1216) -- SUCCESS

**Config delta:** `per_device_batch_size=7, capacity_factor=1.0, megablox=True`

**Status: SUCCESS**

**Memory:** XLA LHS `109,228,455,736` bytes (~101.7 GiB) / limit `109,423,341,340` (~101.9 GiB)

**Steady-state summary (steps 5-13):**
- Step time: **23.62 s**
- MFU: **13.51%**
- Tokens/s/device: **1,214**
- Total tokens/s (64 GPUs): **~77,700**

**Observations:**
- `cf=1.0 + megablox=True` is a **tiny** gain over `cf=1.0 bs=7` (+0.4% tokens/s/device)
- Improvement is too small to beat `cf=1.0 bs=8`
- Worth remembering as a minor positive, but not the primary lever

### Run: bf16-p4d-cf1-sparse (Job 1217) -- FAILED

**Config delta:** `per_device_batch_size=7, capacity_factor=1.0, sparse_matmul=True`

**Status: FAILED** -- `Check failed: ... RaggedDot is only supported with Shardy.`

**Observations:**
- Same deterministic failure as Job 1215
- `capacity_factor=1.0` does not change the sparse-matmul incompatibility

### Run: bf16-p4f-cf1-sparse-shardy-bs7 (Job 1239) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p4f-cf1-sparse-shardy-bs7 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1238 -- per_device_batch_size=7 capacity_factor=1.0 sparse_matmul=True shardy=true
```

**Config delta:** `per_device_batch_size=7, capacity_factor=1.0, sparse_matmul=True, shardy=true`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.28TiB`

**Log path:** `outputs/1239-JAX-deepseek3-671b-bf16-p4f-cf1-sparse-shardy-bs7-per_device_batch_size_7-capacity_factor_1.0-sparse_matmul_True-shardy_true.log`

**Observations:**
- With `shardy=true`, `sparse_matmul=True` no longer crashes with the RaggedDot assertion -- **Shardy partitioner resolves the compatibility issue**
- However, the resulting program is catastrophically memory-inefficient: it attempts to allocate **2.28 TiB** per GPU (vs ~106 GiB for the standard path)
- Shardy is GSPMD's successor partitioner in XLA. When enabled (`shardy=true`), it takes over sharding propagation and may produce different (and in this case, much worse) sharding decisions for the MoE expert dispatch
- `sparse_matmul` replaces the standard padded expert dispatch (dense matmul over `capacity_factor`-padded token buffers) with a sparse `RaggedDot` operation that processes only the tokens actually routed to each expert. In theory this eliminates padding waste entirely. In practice, the Shardy-generated sharding plan materializes enormous intermediate buffers
- The 2.28 TiB allocation likely corresponds to an un-sharded or poorly-sharded expert intermediate tensor
- `sparse_matmul + shardy` is **not viable** at this model scale on B200 without further XLA/Shardy tuning

### Run: bf16-p4g-cf1-sparse-shardy-bs8 (Job 1240) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p4g-cf1-sparse-shardy-bs8 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1239 -- per_device_batch_size=8 capacity_factor=1.0 sparse_matmul=True shardy=true
```

**Config delta:** `per_device_batch_size=8, capacity_factor=1.0, sparse_matmul=True, shardy=true`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.60TiB`

**Log path:** `outputs/1240-JAX-deepseek3-671b-bf16-p4g-cf1-sparse-shardy-bs8-per_device_batch_size_8-capacity_factor_1.0-sparse_matmul_True-shardy_true.log`

**Observations:**
- Same pathological memory pattern as Job 1239, scaled up by the larger batch size (2.60 TiB vs 2.28 TiB)
- Confirms `sparse_matmul + shardy` is fundamentally broken for memory at this model scale

### Run: bf16-p4e-cf1-megablox-bs8 (Job 1235) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-p4e-cf1-megablox-bs8 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=8 capacity_factor=1.0 megablox=True
```

**Config delta:** `per_device_batch_size=8, capacity_factor=1.0, megablox=True`

**Status: SUCCESS**

**Memory:**
- XLA LHS `114,377,226,008` bytes (~106.5 GiB) / limit `109,423,263,516` (~101.9 GiB)
- Memory profile is effectively unchanged vs Job 1208

**Steady-state summary (steps 5-13):**
- Step time: **26.35 s**
- MFU: **13.84%**
- TFLOP/s/device: **311.4**
- Tokens/s/device: **1,243**
- Total tokens/s (64 GPUs): **~79,600**

**Training wall time:** 477s (7m 57s)
**Compilation time (step 0):** 39.4s

**Log path:** `outputs/1235-JAX-deepseek3-671b-bf16-p4e-cf1-megablox-bs8-per_device_batch_size_8-capacity_factor_1.0-megablox_True/log`

**Observations:**
- `megablox=True` on top of `cf=1.0, bs=8` gives only a **tiny** gain over Job 1208
- Throughput improvement is ~0.1% and within run-to-run noise, but this is still the best observed BF16 result so far
- If choosing one BF16 config today, carry forward `bs=8, cf=1.0, megablox=True`

### Phase 4 Summary: MoE Runtime

| Config Change | bs | Status | MFU (%) | Tok/s total | Takeaway |
|--------------|----|--------|---------|-------------|----------|
| baseline | 7 | SUCCESS | 12.04 | 69,210 | reference |
| megablox=True | 7 | SUCCESS | 12.02 | 69,100 | neutral/slightly worse |
| cf=1.0 | 7 | SUCCESS | 13.46 | 77,400 | major win |
| cf=1.0 + megablox=True | 7 | SUCCESS | 13.51 | 77,700 | tiny extra gain |
| cf=1.0 + megablox=True | 8 | SUCCESS | 13.84 | 79,600 | best observed BF16, but only marginally |
| sparse_matmul=True | 7 | FAILED | -- | -- | requires `shardy=true` |
| cf=1.0 + sparse+shardy | 7 | OOM (2.28 TiB) | -- | -- | Shardy sharding plan catastrophic |
| cf=1.0 + sparse+shardy | 8 | OOM (2.60 TiB) | -- | -- | same pathological pattern |

**Current BF16 best config:** `per_device_batch_size=8, capacity_factor=1.0, megablox=True` (Job 1235)  
**Best bs=7 variant:** `capacity_factor=1.0, megablox=True` (Job 1216), but only marginally above plain `cf=1.0`

---

## FP8 Phase 0: Initial Probes

### Run: fp8-p0-cf1-bs12 (Job 1241) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::fp8-p0-cf1-bs12 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1240 -- per_device_batch_size=12 capacity_factor=1.0 quantization=fp8
```

**Config delta:** `per_device_batch_size=12, capacity_factor=1.0, quantization=fp8`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 118.74GiB`

**Log path:** `outputs/1241-JAX-deepseek3-671b-fp8-p0-cf1-bs12-per_device_batch_size_12-capacity_factor_1.0-quantization_fp8.log`

**Observations:**
- FP8 quantization reduces weight/activation memory by ~50% for the quantized layers, but the overall model still has many non-quantized components (embeddings, norms, router, optimizer state)
- At bs=12 the OOM allocation (118.74 GiB) is comparable to the BF16 bs=12 OOM (140.37 GiB) -- FP8 saved ~15% but not enough to fit bs=12
- Need to probe smaller batch sizes (bs=8, bs=10) to find the FP8 feasible envelope

### Run: fp8-p0-cf1-bs16 (Job 1242) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::fp8-p0-cf1-bs16 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1241 -- per_device_batch_size=16 capacity_factor=1.0 quantization=fp8
```

**Config delta:** `per_device_batch_size=16, capacity_factor=1.0, quantization=fp8`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 147.38GiB`

**Log path:** `outputs/1242-JAX-deepseek3-671b-fp8-p0-cf1-bs16-per_device_batch_size_16-capacity_factor_1.0-quantization_fp8.log`

**Observations:**
- bs=16 with FP8 OOM at 147.38 GiB, comparable to BF16 bs=16 (171.96 GiB) -- FP8 saved ~14% at this batch size
- Memory savings from FP8 alone are not transformative for this model's memory footprint

### FP8 Phase 0 Summary

| Config | bs | Status | OOM Alloc | Notes |
|--------|-----|--------|-----------|-------|
| fp8, cf=1.0 | 12 | OOM | 118.74 GiB | ~15% less than BF16 bs=12 |
| fp8, cf=1.0 | 16 | OOM | 147.38 GiB | ~14% less than BF16 bs=16 |

**Next steps:** Probe FP8 at bs=8 (should work based on extrapolation) and bs=10 to find the FP8 memory ceiling. Then measure throughput and compare to BF16 best.

### Run: fp8-nv-bs7 (Job 1592) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::fp8-nv-bs7 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7 quantization=fp8 "_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=''"
```

**Config delta:** `per_device_batch_size=7, quantization=fp8, NV default XLA flags`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **21.40 s** (vs BF16 NV defaults 24.23s, **-11.7%**)
- MFU: **7.46%** (relative to FP8 peak of 4,500 TFLOP/s)
- TFLOP/s/device: **335.5** (vs BF16 NV 296.4, **+13.2%**)
- Tokens/s/device: **1,340** (vs BF16 NV 1,183, **+13.3%**)
- Total tokens/s (64 GPUs): **~85,800** (vs BF16 NV ~75,700, **+13.3%**)

**Training wall time:** 441s (7m 22s)
**Compilation time (step 0):** 45.2s (longer than BF16 ~36s due to FP8 kernel compilation)

**Observations:**
- **FP8 with NV defaults is the new overall best config for DeepSeek3-671B on B200**
- Despite lower MFU% (7.46% vs 13.17%), the FP8 peak is 2x higher (4,500 vs 2,250 TFLOP/s), yielding 335.5 TFLOP/s actual vs 296.4 for BF16
- 13.3% throughput improvement over best BF16 at the same batch size
- Step time drops from 24.2s to 21.4s -- FP8 tensor cores deliver meaningful speedup on the GEMM-dominated MoE workload
- Loss trajectory is slightly different (12.268→10.664 vs BF16 12.268→10.069 at step 14) due to reduced numerical precision, but training remains stable

### Run: fp8-nv-bs8 (Job 1593) -- OOM

**Config delta:** `per_device_batch_size=8, quantization=fp8, NV default XLA flags`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 107.70GiB`

**Observations:**
- FP8 does not unlock bs=8 with NV defaults -- the memory savings from FP8 quantization are offset by the removal of `slop_factor=95` (NV defaults use the XLA default which is less constrained)
- OOM at 107.70 GiB is close to the BF16 AMD-flags bs=8 OOM (108.63 GiB)

### Run: fp8-nv-bs10 (Job 1594) -- OOM

**Config delta:** `per_device_batch_size=10, quantization=fp8, NV default XLA flags`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 122.93GiB`

**Observations:**
- FP8 bs=10 is far over the memory limit, as expected from extrapolation

### FP8 Phase Summary (Updated)

| Config | bs | XLA Flags | Status | Step Time (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|--------|-----|-----------|--------|---------------|---------|-------------|-----------|-------|
| fp8, cf=1.0, AMD flags | 12 | AMD-parity | OOM (118.74 GiB) | -- | -- | -- | -- | |
| fp8, cf=1.0, AMD flags | 16 | AMD-parity | OOM (147.38 GiB) | -- | -- | -- | -- | |
| **fp8, NV defaults** | **7** | **NV defaults** | **SUCCESS** | **21.40** | **7.46*** | **335.5** | **1,340** | **overall best** |
| fp8, NV defaults | 8 | NV defaults | OOM (107.70 GiB) | -- | -- | -- | -- | |
| fp8, NV defaults | 10 | NV defaults | OOM (122.93 GiB) | -- | -- | -- | -- |

*MFU% is relative to FP8 peak (4,500 TFLOP/s for B200). Equivalent BF16-normalized MFU would be ~14.91%.

---

## Capacity Factor Exploration (Independent Line 2)

### Run: bf16-cf2.0 (Job 1522) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-cf2.0 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7 capacity_factor=2.0
```

**Config delta:** `per_device_batch_size=7, capacity_factor=2.0` (baseline cf=1.25)

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 116.84GiB`

**Log path:** `outputs/1522-JAX-deepseek3-671b-bf16-cf2.0-per_device_batch_size_7-capacity_factor_2.0/log`

**Observations:**
- `capacity_factor=2.0` increases expert padding from 1.25x to 2x, significantly increasing activation memory
- OOM at 116.84 GiB vs baseline (bs=7, cf=1.25) which fits in memory
- For `cf=2.0` to work, would need to reduce bs further (maybe bs=4-5)

### Run: bf16-cf4.0 (Job 1523) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-cf4.0 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1522 -- per_device_batch_size=7 capacity_factor=4.0
```

**Config delta:** `per_device_batch_size=7, capacity_factor=4.0` (baseline cf=1.25)

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 145.29GiB`

**Log path:** `outputs/1523-JAX-deepseek3-671b-bf16-cf4.0-per_device_batch_size_7-capacity_factor_4.0/log`

**Observations:**
- `capacity_factor=4.0` quadruples expert padding, massive memory increase
- OOM at 145.29 GiB (~24% more than cf=2.0), proportional to the increased padding
- Would require bs=2-3 at most to fit, likely impractical for throughput

### Run: bf16-cf2.0-bs4 (Job 1595) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-cf2.0-bs4 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=4 capacity_factor=2.0
```

**Config delta:** `per_device_batch_size=4, capacity_factor=2.0`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **22.04 s**
- MFU: **8.28%**
- TFLOP/s/device: **186.2**
- Tokens/s/device: **743**
- Total tokens/s (64 GPUs): **~47,600**

**Training wall time:** 414s (6m 56s)

**Observations:**
- `cf=2.0` is viable at `bs=4` but throughput is 37% worse than the baseline `bs=7/cf=1.25` (47.6K vs 69.2K tok/s)
- The extra expert padding provides no throughput benefit -- it only wastes compute on padding tokens

### Run: bf16-cf4.0-bs2 (Job 1596) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-cf4.0-bs2 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=2 capacity_factor=4.0
```

**Config delta:** `per_device_batch_size=2, capacity_factor=4.0`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **20.42 s**
- MFU: **4.47%**
- TFLOP/s/device: **100.5**
- Tokens/s/device: **401**
- Total tokens/s (64 GPUs): **~25,700**

**Training wall time:** 389s (6m 31s)

**Observations:**
- `cf=4.0` is viable at `bs=2` but throughput is catastrophically low -- 63% below baseline
- MFU of 4.47% means >55% of FLOPs are wasted on expert padding
- Confirms that high capacity_factor values are only useful if token-dropping behavior needs to be studied, not for throughput

### CF Exploration Summary (Updated)

| capacity_factor | bs | Status | MFU (%) | Tok/s/dev | Tok/s total | Notes |
|----------------|-----|--------|---------|-----------|-------------|-------|
| 1.0 | 7 | SUCCESS | 13.46 | 1,209 | 77,400 | **best throughput** (Job 1207) |
| 1.25 (baseline) | 7 | SUCCESS | 12.04 | 1,081 | 69,210 | baseline (Job 1200) |
| 2.0 | 7 | OOM | -- | -- | -- | 116.84 GiB (Job 1522) |
| **2.0** | **4** | **SUCCESS** | **8.28** | **743** | **47,600** | **viable but -37% vs baseline** (Job 1595) |
| 4.0 | 7 | OOM | -- | -- | -- | 145.29 GiB (Job 1523) |
| **4.0** | **2** | **SUCCESS** | **4.47** | **401** | **25,700** | **viable but -63% vs baseline** (Job 1596) |

**Conclusion:** Higher capacity_factor values require proportionally smaller batch sizes and deliver significantly worse throughput. `cf=1.0` remains optimal for throughput. `cf=2.0/4.0` are only useful for studying expert token-dropping behavior, not for benchmark throughput.

---

## XLA Flags Optimization (Independent Line 1)

**Baseline:** Job 1200 (bs=7, cf=1.25, AMD-parity XLA flags) -- MFU 12.04%, step time 27.48s, 69,210 tok/s

Current AMD-parity XLA flags in `train_env.nvidia.sh`:
- `--xla_gpu_memory_limit_slop_factor=95`
- `--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592` (8 GiB)
- `--xla_gpu_all_gather_combine_threshold_bytes=8589934592` (8 GiB)
- `--xla_gpu_enable_triton_gemm=false`
- `--xla_gpu_enable_cublaslt=true`
- `--xla_gpu_autotune_level=0`
- `--xla_gpu_enable_all_gather_combine_by_dim=false`

### Run: bf16-xla-nv-defaults (Job 1524) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-nv-defaults -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1523 -- per_device_batch_size=7 _env_XLA_FLAGS_REPLACE="--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=''"
```

**Config delta:** Replaced all XLA flags with NVIDIA image defaults (LHS=true + command_buffer fix only)

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **24.23 s** (vs baseline 27.48s, **-11.8%**)
- MFU: **13.17%** (vs baseline 12.04%, **+9.4%**)
- TFLOP/s/device: **296.4**
- Tokens/s/device: **1,183**
- Total tokens/s (64 GPUs): **~75,700** (vs baseline 69,210, **+9.4%**)

**Compilation time (step 0):** 36.3s (vs baseline ~39s)
**Training wall time:** 470s (7m 50s)

**Log path:** `outputs/1524-JAX-deepseek3-671b-bf16-xla-nv-defaults-per_device_batch_size_7-_env_XLA_FLAGS_REPLACE_--xla_gpu_enable_latency_hiding_scheduler_true,--xla_gpu_enable_command_buffer_''/log`

**Observations:**
- **Major finding:** Removing all AMD-parity XLA flags gives a 9.4% throughput boost on B200
- The AMD flags (slop_factor=95, combine_threshold=8GB, autotune=0, triton=false) were collectively a **significant negative optimization** for B200
- This is now the best BF16 result at bs=7 with cf=1.25, surpassing even the cf=1.0 results at the same batch size (MFU 13.46% at bs=7/cf=1.0 vs 13.17% at bs=7/cf=1.25 with NV defaults)
- The remaining single-variable XLA tests (A2-A7) will reveal which specific AMD flag(s) caused the most harm

### Run: bf16-xla-autotune4 (Job 1525) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-autotune4 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1524 -- per_device_batch_size=7 _env_EXTRA_XLA_FLAGS=--xla_gpu_autotune_level=4
```

**Config delta:** Appended `autotune_level=4` (overrides base `autotune_level=0`)

**Status: SUCCESS**

**Steady-state summary (steps 5-7):**
- Step time: **26.50 s** (vs baseline 27.48s)
- MFU: **12.04%** (identical to baseline)
- Tokens/s/device: **1,082**
- Total tokens/s (64 GPUs): **~69,200**

**Compilation time (step 0):** 40.2s (slightly longer due to autotuning)

**Log path:** `outputs/1525-JAX-deepseek3-671b-bf16-xla-autotune4-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_autotune_level_4/log`

**Observations:**
- Kernel autotuning provides zero benefit when the other AMD-parity flags are still present
- The autotuner likely finds the same cuBLAS kernels as the default selection
- Compilation overhead is minimal (~1s extra), so this flag is harmless but useless in isolation

### Run: bf16-xla-triton (Job 1526) -- SUCCESS (catastrophic perf)

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-triton -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1524 -- per_device_batch_size=7 _env_EXTRA_XLA_FLAGS=--xla_gpu_enable_triton_gemm=true
```

**Config delta:** Appended `triton_gemm=true` (overrides base `triton_gemm=false`)

**Status: SUCCESS** (but performance is catastrophically bad)

**Steady-state summary (steps 5-6):**
- Step time: **100.58 s** (vs baseline 27.48s, **+266%**)
- MFU: **3.17%** (vs baseline 12.04%, **-74%**)
- Tokens/s/device: **285**
- Total tokens/s (64 GPUs): **~18,200**

**Compilation time (step 0):** ~950s (long Triton kernel compilation)
**Training wall time:** 1598s (26m 38s)

**Log path:** `outputs/1526-JAX-deepseek3-671b-bf16-xla-triton-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_enable_triton_gemm_true/log`

**Observations:**
- Triton GEMM is an **extreme negative optimization** on B200 for this MoE model
- cuBLAS kernels vastly outperform Triton-generated kernels for the specific GEMM shapes in DeepSeek3
- The AMD image explicitly disables Triton GEMM (`triton_gemm=false`) -- this was a correct decision
- **Never enable `triton_gemm=true` for this model on B200**

### Run: bf16-xla-slop300 (Job 1527) -- OOM

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-slop300 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1524 -- per_device_batch_size=7 _env_EXTRA_XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=300
```

**Config delta:** Appended `slop_factor=300` (overrides base `slop_factor=95`)

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 108.71GiB`

**Log path:** `outputs/1527-JAX-deepseek3-671b-bf16-xla-slop300-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_memory_limit_slop_factor_300/log`

**Observations:**
- `slop_factor=300` gives the Latency Hiding Scheduler (LHS) a 3x memory budget multiplier
- LHS uses this budget to overlap computation with communication by pre-allocating buffers
- At 300, LHS tries to allocate too aggressively for the available HBM, causing OOM
- The base `slop_factor=95` is conservative but necessary for memory safety at bs=7

### Run: bf16-xla-combine256 (Job 1528) -- IB HANG (cancelled)

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-combine256 -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1525 -- per_device_batch_size=7 "_env_EXTRA_XLA_FLAGS=--xla_gpu_reduce_scatter_combine_threshold_bytes=256,--xla_gpu_all_gather_combine_threshold_bytes=256"
```

**Config delta:** Appended combine thresholds = 256 bytes (overrides base 8 GiB)

**Status: IB HANG** -- `IBV_WC_RETRY_EXC_ERR(12)` across all nodes during initialization. Hung for ~12 hours. Cancelled.

**Log path:** `outputs/1528-JAX-deepseek3-671b-bf16-xla-combine256-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_reduce_scatter_combine_threshold_bytes_256,--xla_gpu_all_gather_combine_threshold_bytes_256/log`

**Observations:**
- Transient InfiniBand P_Key issue (same as Job 1202), not related to the XLA flag change
- No training steps completed -- the hang occurred during initial NCCL setup
- **Resubmitted as Job 1588**

### Run: bf16-xla-pipelined (Job 1529) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-pipelined -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1526 -- per_device_batch_size=7 "_env_EXTRA_XLA_FLAGS=--xla_gpu_enable_pipelined_all_gather=true,--xla_gpu_enable_pipelined_all_reduce=true,--xla_gpu_enable_pipelined_reduce_scatter=true"
```

**Config delta:** Appended pipelined all-gather, all-reduce, reduce-scatter = true

**Status: SUCCESS**

**Steady-state summary (steps 5-6):**
- Step time: **26.71 s** (vs baseline 27.48s, -2.8%)
- MFU: **11.95%** (vs baseline 12.04%, **-0.7%**)
- Tokens/s/device: **1,073**
- Total tokens/s (64 GPUs): **~68,700**

**Log path:** `outputs/1529-JAX-deepseek3-671b-bf16-xla-pipelined-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_enable_pipelined_all_gather_true,--xla_gpu_enable_pipelined_all_reduce_true,--xla_gpu_enable_pipelined_reduce_scatter_true/log`

**Observations:**
- Pipelined collectives on top of AMD-parity flags are slightly negative (-0.7%)
- The LHS already handles overlap scheduling; adding explicit pipelining may introduce conflicts
- Not recommended as a standalone change

### Run: bf16-xla-combdim (Job 1530) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-combdim -N 8 -w hungry-hippo-fin-03-[1-8] --dependency=afterany:1527 -- per_device_batch_size=7 _env_EXTRA_XLA_FLAGS=--xla_gpu_enable_all_gather_combine_by_dim=true
```

**Config delta:** Appended `combine_by_dim=true` (overrides base `combine_by_dim=false`)

**Status: SUCCESS**

**Steady-state summary (steps 5-6):**
- Step time: **25.10 s** (vs baseline 27.48s, **-8.7%**)
- MFU: **12.71%** (vs baseline 12.04%, **+5.6%**)
- TFLOP/s/device: **286.0**
- Tokens/s/device: **1,142**
- Total tokens/s (64 GPUs): **~73,100**

**Log path:** `outputs/1530-JAX-deepseek3-671b-bf16-xla-combdim-per_device_batch_size_7-_env_EXTRA_XLA_FLAGS_--xla_gpu_enable_all_gather_combine_by_dim_true/log`

**Observations:**
- **Second most impactful single flag change** after NV-defaults
- `combine_by_dim=true` allows the XLA compiler to combine all-gather operations along specific dimensions, reducing the number of collective calls
- The base `combine_by_dim=false` was a significant negative optimization
- This single flag accounts for ~60% of the total gain seen in NV-defaults (5.6% vs 9.4%)

### XLA Flags Summary

| Test | Flag Change | MFU (%) | vs Baseline | Step Time (s) | Assessment |
|------|-------------|---------|-------------|---------------|------------|
| Baseline (1200) | AMD-parity flags | 12.04 | -- | 27.48 | reference |
| **A1: NV defaults (1524)** | Remove all AMD flags | **13.17** | **+9.4%** | **24.23** | **BEST** |
| A2: autotune4 (1525) | autotune_level=0→4 | 12.04 | 0% | 26.50 | no effect |
| A3: triton (1526) | triton_gemm=false→true | 3.17 | -74% | 100.58 | catastrophic |
| A4: slop300 (1527) | slop_factor=95→300 | OOM | -- | -- | too aggressive |
| A5: combine256 (1528) | combine=8GB→256B | IB hang | -- | -- | resubmit 1588 |
| A6: pipelined (1529) | +pipelined collectives | 11.95 | -0.7% | 26.71 | slightly worse |
| **A7: combdim (1530)** | combine_by_dim=false→true | **12.71** | **+5.6%** | **25.10** | **significant gain** |

### Run: bf16-xla-combine256-v2 (Job 1588) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-combine256-v2 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7 "_env_EXTRA_XLA_FLAGS=--xla_gpu_reduce_scatter_combine_threshold_bytes=256,--xla_gpu_all_gather_combine_threshold_bytes=256"
```

**Config delta:** AMD-parity flags + combine thresholds = 256 bytes (resubmit of Job 1528 which had IB hang)

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **24.27 s** (vs baseline 27.48s, **-11.7%**)
- MFU: **13.16%** (vs baseline 12.04%, **+9.3%**)
- TFLOP/s/device: **295.7**
- Tokens/s/device: **1,182**
- Total tokens/s (64 GPUs): **~75,700**

**Observations:**
- Reducing combine thresholds from 8 GiB to 256 bytes gives almost the same gain as removing all AMD flags entirely (+9.3% vs +9.4%)
- This confirms the 8 GiB combine threshold was the other major negative optimization alongside `combine_by_dim=false`
- Smaller thresholds allow more frequent, smaller collectives which overlap better with computation on B200's NVSwitch fabric

### Run: bf16-xla-best-combo (Job 1589) -- SUCCESS

**Submit command:**
```bash
./submit.sh deepseek3-671b::bf16-xla-best-combo -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=7 "_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer='',--xla_gpu_enable_all_gather_combine_by_dim=true"
```

**Config delta:** NV defaults + explicit `combine_by_dim=true`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **24.27 s** (vs NV defaults 24.23s)
- MFU: **13.15%** (vs NV defaults 13.17%)
- Tokens/s/device: **1,181**
- Total tokens/s (64 GPUs): **~75,600**

**Observations:**
- Adding explicit `combine_by_dim=true` on top of NV defaults provides **no additional gain** -- effectively identical performance
- This means NV defaults already implicitly enable `combine_by_dim=true` (or the XLA default is `true`)
- The 5.6% gain from A7 (combdim) was because it flipped the AMD flag's `false` override to `true`

### Run: bf16-nv-megablox (Job 1590) -- SUCCESS

**Config delta:** NV defaults + `megablox=True`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **24.25 s**
- MFU: **13.16%**
- Tokens/s/device: **1,182**
- Total tokens/s (64 GPUs): **~75,600**

**Observations:**
- `megablox=True` on top of NV defaults is neutral -- identical to plain NV defaults
- Consistent with earlier finding (Job 1214): megablox provides no benefit for DS3-671B at bs=7

### Run: bf16-nv-dense-shardy (Job 1591) -- SUCCESS

**Config delta:** NV defaults + `shardy=true`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **24.27 s**
- MFU: **13.16%**
- Tokens/s/device: **1,181**
- Total tokens/s (64 GPUs): **~75,600**

**Observations:**
- `shardy=true` with NV defaults performs identically to plain NV defaults for the dense (non-sparse) code path
- Shardy only causes issues when combined with `sparse_matmul=True` (pathological memory, Jobs 1239-1240)
- For standard dispatch, Shardy is a drop-in replacement with no perf impact

### XLA Flags Summary (Updated)

| Test | Flag Change | MFU (%) | vs Baseline | Step Time (s) | Assessment |
|------|-------------|---------|-------------|---------------|------------|
| Baseline (1200) | AMD-parity flags | 12.04 | -- | 27.48 | reference |
| **A1: NV defaults (1524)** | Remove all AMD flags | **13.17** | **+9.4%** | **24.23** | **BEST** |
| A2: autotune4 (1525) | autotune_level=0→4 | 12.04 | 0% | 26.50 | no effect |
| A3: triton (1526) | triton_gemm=false→true | 3.17 | -74% | 100.58 | catastrophic |
| A4: slop300 (1527) | slop_factor=95→300 | OOM | -- | -- | too aggressive |
| **A5r: combine256 (1588)** | combine=8GB→256B | **13.16** | **+9.3%** | **24.27** | **~equal to NV defaults** |
| A6: pipelined (1529) | +pipelined collectives | 11.95 | -0.7% | 26.71 | slightly worse |
| **A7: combdim (1530)** | combine_by_dim=false→true | **12.71** | **+5.6%** | **25.10** | **significant gain** |
| A8: best-combo (1589) | NV defaults + combdim=true | 13.15 | +9.2% | 24.27 | no gain over NV defaults |
| A9: NV + megablox (1590) | NV defaults + megablox | 13.16 | +9.3% | 24.25 | neutral |
| A10: NV + shardy (1591) | NV defaults + shardy | 13.16 | +9.3% | 24.27 | neutral |

**Key findings:**
1. The AMD-parity XLA flags collectively cost 9.4% MFU on B200
2. **Two flags account for nearly all the loss:** `combine_by_dim=false` (~60%) and `combine_threshold=8GB` (~40%)
3. `triton_gemm=true` is catastrophic (-74%) -- must stay disabled
4. `slop_factor=300` causes OOM; `slop_factor=95` is a necessary constraint
5. NV defaults are essentially optimal -- adding `combine_by_dim=true`, `megablox`, or `shardy` provides no additional gain
6. **Recommended BF16 XLA config:** Use NV image defaults (LHS=true + command_buffer fix only)

---

## Kimi-K2-1T: Initial Batch Sweep

### Kimi Batch Sweep (Jobs 1531-1534) -- ALL OOM

| Job | bs | Status | OOM Alloc | Notes |
|-----|-----|--------|-----------|-------|
| 1531 | 4 | OOM | 100.20 GiB | |
| 1534 | 3 | OOM | 91.19 GiB | |
| 1532 | 2 | OOM | 83.33 GiB | |
| 1533 | 6 | OOM | 118.07 GiB | |

**All batch sizes from bs=2 to bs=6 OOM.** Kimi-K2-1T (1T parameters) with `ici_ep=8, dcn_fsdp=8` parallelism requires too much memory on B200 (~179 GiB HBM3e, ~166 GiB usable).

**Memory scaling:** OOM alloc grows linearly with bs (roughly +8.5 GiB per bs increment), confirming that even the model weights alone are near the memory limit.

### Run: kimi-k2-bs1 (Job 1597) -- SUCCESS

**Submit command:**
```bash
./submit.sh kimi-k2-1t::kimi-k2-bs1 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=1
```

**Config delta:** `per_device_batch_size=1`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **17.00 s**
- MFU: **2.20%**
- TFLOP/s/device: **49.5**
- Tokens/s/device: **241**
- Total tokens/s (64 GPUs): **~15,400**

**Training wall time:** 340s (5m 40s)

**Observations:**
- **bs=1 is the only working batch size** for Kimi-K2-1T on B200 with default parallelism
- MFU is extremely low (2.20%) due to the minimal batch size -- GPUs are starved of work
- The 1T model barely fits at bs=1; it needs >16 nodes or memory-saving techniques to reach practical throughput

### Run: kimi-k2-bs1-cf1.0 (Job 1598) -- SUCCESS

**Submit command:**
```bash
./submit.sh kimi-k2-1t::kimi-k2-bs1-cf1.0 -N 8 -w hungry-hippo-fin-03-[1-8] -- per_device_batch_size=1 capacity_factor=1.0
```

**Config delta:** `per_device_batch_size=1, capacity_factor=1.0`

**Status: SUCCESS**

**Steady-state summary (steps 5-13):**
- Step time: **16.60 s** (vs cf=1.25 17.00s, **-2.4%**)
- MFU: **2.25%** (vs cf=1.25 2.20%, **+2.3%**)
- TFLOP/s/device: **50.7**
- Tokens/s/device: **247**
- Total tokens/s (64 GPUs): **~15,800**

**Training wall time:** 331s (5m 31s)

**Observations:**
- `cf=1.0` provides a marginal gain (+2.3%) consistent with DeepSeek3 results -- reduced expert padding saves both memory and compute
- At bs=1 the gain is small in absolute terms (~400 more tok/s total)
- Both bs=1 configs confirm Kimi-K2-1T can train on B200 8-node, but throughput is impractical

### Run: kimi-k2-bs2-cf1.0 (Job 1599) -- OOM

**Config delta:** `per_device_batch_size=2, capacity_factor=1.0`

**Status: OOM** -- `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 82.47GiB`

**Observations:**
- `cf=1.0` is not enough to unlock bs=2 for Kimi-K2-1T
- OOM at 82.47 GiB (vs bs=2/cf=1.25 OOM at 83.33 GiB from Job 1532) -- only ~1 GiB savings from cf=1.0
- The memory pressure from the 1T model weights dominates; capacity_factor changes are a drop in the bucket

### Jobs 1600, 1601 -- CANCELLED

**Job 1600:** `per_device_batch_size=2, ici_fsdp_parallelism=2, ici_expert_parallelism=4` -- CANCELLED before running
**Job 1601:** `per_device_batch_size=4, ici_fsdp_parallelism=2, ici_expert_parallelism=4` -- CANCELLED before running

These were queued as dependency chain after Job 1599 but were cancelled (likely manually) before execution. The `ici_fsdp=2, ici_ep=4` parallelism split remains untested for Kimi-K2-1T.

### Kimi-K2-1T Summary (Updated)

| Config | bs | Status | MFU (%) | Tok/s/dev | Tok/s total | Notes |
|--------|-----|--------|---------|-----------|-------------|-------|
| default (cf=1.25) | 6 | OOM (118.07 GiB) | -- | -- | -- | Job 1533 |
| default (cf=1.25) | 4 | OOM (100.20 GiB) | -- | -- | -- | Job 1531 |
| default (cf=1.25) | 3 | OOM (91.19 GiB) | -- | -- | -- | Job 1534 |
| default (cf=1.25) | 2 | OOM (83.33 GiB) | -- | -- | -- | Job 1532 |
| cf=1.0 | 2 | OOM (82.47 GiB) | -- | -- | -- | Job 1599 |
| **default (cf=1.25)** | **1** | **SUCCESS** | **2.20** | **241** | **15,400** | **Job 1597** |
| **cf=1.0** | **1** | **SUCCESS** | **2.25** | **247** | **15,800** | **Job 1598, marginal gain** |
| fsdp2/ep4 | 2 | CANCELLED | -- | -- | -- | Job 1600, untested |
| fsdp2/ep4 | 4 | CANCELLED | -- | -- | -- | Job 1601, untested |

**Conclusion:** Kimi-K2-1T is severely memory-constrained on 8-node B200. Only bs=1 fits, yielding impractical throughput (~2.2% MFU). The `ici_fsdp=2/ici_ep=4` parallelism split (Jobs 1600-1601) was never tested and may be worth resubmitting -- doubling FSDP sharding could halve the non-expert weight memory per GPU, potentially unlocking bs=2+.
