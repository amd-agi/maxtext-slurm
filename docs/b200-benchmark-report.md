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

**Current BF16 best config:** `per_device_batch_size=8, capacity_factor=1.0, megablox=True` (Job 1235)  
**Best bs=7 variant:** `capacity_factor=1.0, megablox=True` (Job 1216), but only marginally above plain `cf=1.0`
