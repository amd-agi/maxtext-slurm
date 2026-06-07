# Llama-3.1-405B 8-Node FP8 Training Run — Report

> **TL;DR** — 8-node / **64x MI355X** Llama-3.1-405B **FP8** training, 3 steps, profiled end-to-end with rocprofv3 (GPU trace + kernel statistics). **Steady-state MFU ~32.5%** (~38 s/step, ~644 tokens/s/device, ~1626 TFLOP/s/device). Compute is FP8-GEMM-dominated (~58% of GPU time); RCCL communication is ~21% (with blocking collectives up to ~3.2 s -> overlap opportunity); attention ~10%. Run COMPLETED cleanly; loss descended 12.26 -> 11.91. RDMA ran over the ionic NICs via the surgical libionic mount + `NCCL_IB_QPS_PER_CONNECTION=1`.

Run identifier: SLURM job **15697** (2026-06-03). Related docs: [`RCCL_UPGRADE.md`](RCCL_UPGRADE.md), [`EMPTY_STREAM_REPORT.md`](EMPTY_STREAM_REPORT.md), [`ROCPROF_TRACE_REPORT.md`](ROCPROF_TRACE_REPORT.md).

---

## 1. Hardware & software

| Component | Version / spec |
|-----------|----------------|
| GPUs | 8 nodes x 8 = **64x AMD Instinct MI355X** (gfx950 / CDNA4) |
| Interconnect | AMD Pensando "ionic" NICs, RoCE/IB (RDMA); intra-node XGMI |
| ROCm | **7.1.1** (build 7.1.1-38), HIP 7.1.52802 |
| RCCL | **2.27.7-HEAD:185e78a** (ROCm/rccl PR #2063, "use one side stream per process") |
| Framework | MaxText on JAX/XLA (`rocm/jax-training:maxtext-v26.2` base, RCCL rebuilt with PR #2063) |

## 2. Model configuration (Llama-3.1-405B)

| Parameter | Value |
|-----------|-------|
| Decoder layers | 126 |
| Embedding dim | 16384 |
| MLP dim | 53248 |
| Query / KV heads | 128 / 8 (GQA) |
| Head dim | 128 |
| Sequence length (`max_target_length`) | 8192 |
| Precision | FP8 (Tensile F8 GEMM kernels), BF16 attention |

## 3. Parallelism & batch

- **64-way pure FSDP** (ZeRO-3 style), no tensor parallelism:
  - `ici_fsdp_parallelism = -1` -> 8 (intra-node), `dcn_fsdp_parallelism = -1` -> 8 (inter-node) => 8 x 8 = 64.
  - tensor / sequence parallelism = 1.
- `per_device_batch_size = 3` -> **global batch = 192** (64 x 3); `gradient_accumulation_steps = 1`.
- Tokens/step = 192 x 8192 = **1,572,864**.

## 4. How it was run

```bash
NODELIST=<8 nodes> STEPS=3 \
  ./run_llama3_1_405b.sh \
    _env_NCCL_IB_QPS_PER_CONNECTION=1 \
    _env_ROCPROF_TRACE=1 \
    _env_ROCPROF_TRACES=kernel,rccl,marker
```

Launch chain: `run_llama3_1_405b.sh` -> `submit.sh` -> `_job.sbatch` (SLURM, 8 nodes) -> `_container.sh` (Docker) -> `_train.sh` -> MaxText.

Key runtime settings (resolved from the log):

| Setting | Value | Why |
|---------|-------|-----|
| `NCCL_IB_QPS_PER_CONNECTION` | **1** | ionic NIC QP table is small (`max_qp=4096`); the default 4 QPS x 4 channels overflows it at 8-node scale (`ibv_create_qp: No space left on device`). QPS=1 fits. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | **0.90** | leaves HBM for RCCL/GDR buffers once RDMA is active (0.97 OOMs with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`). |
| `RCCL_MSCCL_ENABLE` / `RCCL_MSCCLPP_ENABLE` | **0 / 0** | avoid `ncclCommSplit` issues with few channels. |
| `NCCL_NCHANNELS_PER_NET_PEER` | 4 | |
| `NVTE_FUSED_ATTN` | 1 | CK fused attention |
| ionic provider | surgical host-`libionic` mount (`USE_HOST_IONIC_PROVIDER_ONLY=true`) | RDMA over ionic without clobbering container glibc |
| rocprofv3 | `--kernel-trace --rccl-trace --marker-trace` | lean domains keep the `.pftrace` under the 1GB perfetto buffer with GPU kernels intact (see `ROCPROF_TRACE_REPORT.md`) |

## 5. Results — performance

| Step | Step time | TFLOP/s/device | MFU | Tokens/s/device | Loss |
|------|-----------|----------------|-----|------------------|------|
| 0 (warmup) | 67.9 s | 914 | 18.27% | 362 | 12.262 |
| 1 | 37.71 s | 1645 | 32.90% | 651.8 | 12.052 |
| 2 | 38.62 s | 1606 | 32.12% | 636.4 | 11.906 |

- **Steady-state (steps 1-2): MFU ~32.5%, ~38 s/step, ~644 tokens/s/device, ~1626 TFLOP/s/device.**
- Step 0 is the first-iteration warmup (one-time runtime overhead after compile); steps 1-2 are representative.
- Loss descends monotonically (12.26 -> 12.05 -> 11.91). Status: **COMPLETED** (exit 0).
- Profiling overhead is negligible: non-profiled 405B runs measured ~31.6-31.9% MFU.

## 6. Profiling artifacts

Job directory: `$JOB_WORKSPACE/15697-JAX-llama3.1-405b-run8n-steps_3-...-_env_ROCPROF_TRACES_kernel,rccl,marker/`

```
log                                              # training log (per-step metrics, resolved config, env)
rocprof/<host>/<pid>/<host>/<pid>_results.pftrace # GPU trace, 8 files (~750-766 MB each), 1 per node
                                                  #   -> open in https://ui.perfetto.dev
rocprof/<host>/<pid>/<host>/<pid>_kernel_trace.csv # raw per-kernel records (8 files)
rocprof/kernel_stats.csv                          # aggregated kernel statistics (304 kernels, 7.58M dispatches)
```

Regenerate the aggregate: `python3 utils/rocprof_kernel_stats.py <job>/rocprof --shorten --top 30`.

## 7. Kernel breakdown (cluster-wide, share of GPU time)

| Category | Share | Detail |
|----------|-------|--------|
| **FP8 GEMM** (`Cijk_*` Tensile) | **~58%** | F8BS 25.4% + F8B8BS 21.1% + B8F8 11.5% (mostly MT256x256x128) |
| **RCCL** (`ncclDevKernel`) | **~21%** | individual instances up to **3.2 s** -> exposed/blocking communication |
| **Attention** (`aiter::fmha_*`) | **~10%** | bwd 6.5% + fwd 3.9% (hd128, causal, BF16) |
| XLA fusions / transposes / converts | ~9% | `input_convert_transpose`, `input_reduce`, `wrapped_transpose`, ... |
| BF16 GEMM (`Cijk_*BBS`) | ~0.6% | |

Totals: 304 unique kernels, 7,584,562 dispatches, 7503 s of summed GPU time across the 64 GPUs over 3 steps.

## 8. Observations & next steps

- **RCCL ~21% with multi-second blocking collectives** points to exposed communication. Inspect the Perfetto timeline for compute/comm gaps; candidate levers: collective/compute overlap (latency-hiding scheduler, pipelined collectives), channel/QPS tuning, larger combine thresholds.
- Compute is **FP8-GEMM-bound (~58%)**, as expected for 405B FP8 on MI355X.
- This run reproduces the validated stack end-to-end: RCCL PR #2063 (no empty-stream churn), RDMA over ionic (surgical mount + `QPS=1`), and a lean rocprof trace that retains GPU kernels.
