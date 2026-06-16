# Llama-3.1-405B 8-Node FP8 Training Run (MaxText) — Throughput Baseline

> **TL;DR** — 8-node / **64x MI355X** Llama-3.1-405B **FP8** training on MaxText (JAX/XLA), run
> **without any profiler** to establish a clean throughput baseline for later tuning. Run
> **COMPLETED cleanly (exit 0)**, 10 steps. **Steady-state MFU ~32.5%** (~38.1 s/step,
> ~645 tokens/s/device, ~1626 TFLOP/s/device). RDMA over the AMD Pensando "ionic" NICs via the
> surgical libionic mount + `NCCL_IB_QPS_PER_CONNECTION=4` (pristine default). **Profiling
> (jax.profiler / xplane) is intentionally NOT run here — it is a separate follow-up task.**

Cluster: SMCI355 / DLC (partition `Compute-DCPT`). Run identifier: SLURM job **16784** (2026-06-16, 10 steps, no profiler).
Run dir: `/perf_apps/xuefjian/llama405b_20260616_133330/maxtext/`.

**Nodes (8, explicitly pinned, shared with the Primus baseline run):**
`smci355-ccs-aus-n01-33, n02-21, n03-33, n04-21, n04-25, n04-29, n04-33, n05-33`

---

## 1. Hardware & software

| Component | Version / spec |
|-----------|----------------|
| GPUs | 8 nodes x 8 = **64x AMD Instinct MI355X** (gfx950 / CDNA4) |
| Interconnect | AMD Pensando "ionic" NICs, RoCE/RDMA; intra-node XGMI |
| RCCL | **2.27.7** + ROCm/rccl PR #2063 (one side stream per process) |
| Framework | MaxText on JAX/XLA |
| Image | **`rocm/pyt-megatron-lm-jax-nightly-private:maxtext-v26.2-rccl-pr2063`** (AINIC-enabled) |
| Profiler | **none** (this is a throughput baseline; xplane profiling is a later task) |
| Kernels | Tensile FP8 GEMM, CK fused attention (BF16) |

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
JOB_WORKSPACE=/perf_apps/xuefjian/llama405b_20260616_133330/maxtext \
STEPS=10 NODES=8 TAG=baseline \
NODELIST="smci355-ccs-aus-n01-33,smci355-ccs-aus-n02-21,smci355-ccs-aus-n03-33,smci355-ccs-aus-n04-21,smci355-ccs-aus-n04-25,smci355-ccs-aus-n04-29,smci355-ccs-aus-n04-33,smci355-ccs-aus-n05-33" \
  ./run_llama3_1_405b.sh
```

No profiler flags are passed (MaxText runs profiler-free by default).
Launch chain: `run_llama3_1_405b.sh` -> `submit.sh` -> `_job.sbatch` (SLURM, 8 nodes) -> `_container.sh` (Docker) -> `_train.sh` -> MaxText.

Key runtime settings (pristine; resolved from the log):

| Setting | Value | Why |
|---------|-------|-----|
| `NCCL_IB_QPS_PER_CONNECTION` | **4** | pristine default (no QPS=1 override). The 8-node run fit the ionic QP table fine at QPS=4. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | **0.90** | leaves HBM for RCCL/GDR buffers once RDMA is active (0.97 OOMs with RDMA on). |
| `RCCL_MSCCL_ENABLE` / `RCCL_MSCCLPP_ENABLE` | **0 / 0** | avoid `ncclCommSplit` issues with few channels. |
| `xla_gpu_autotune_level` | **0** | heuristic kernel selection (faster compile; pristine config). |
| `NVTE_FUSED_ATTN` / `NVTE_FUSED_ATTN_CK` | 1 / 1 | CK fused attention |
| ionic provider | surgical host-`libionic` mount (`USE_HOST_IONIC_PROVIDER_ONLY=true`) | RDMA over ionic without clobbering container glibc |

## 5. Results — performance (job 16784, no profiler)

| Step | Step time | TFLOP/s/device | MFU | Tokens/s/device | Loss |
|------|-----------|----------------|-----|------------------|------|
| 0 (warmup) | 83.75 s | 740.7 | 14.81% | 293.5 | 12.262 |
| 1 | 36.74 s | 1688.2 | 33.76% | 668.9 | 12.262 |
| 2 | 37.73 s | 1644.1 | 32.88% | 651.4 | 12.008 |
| 3 | 37.94 s | 1635.1 | 32.70% | 647.8 | 11.887 |
| 4 | 38.09 s | 1628.5 | 32.57% | 645.2 | 11.832 |
| 5 | 38.17 s | 1625.1 | 32.50% | 643.9 | 11.803 |
| 6 | 38.21 s | 1623.4 | 32.47% | 643.2 | 11.785 |
| 7 | 38.22 s | 1623.1 | 32.46% | 643.1 | 11.778 |
| 8 | 38.19 s | 1624.1 | 32.48% | 643.5 | 11.773 |
| 9 | 38.21 s | 1623.3 | 32.47% | 643.1 | 11.771 |

- **Steady-state (steps 2-9): MFU ~32.5%, ~38.1 s/step, ~645 tokens/s/device, ~1626 TFLOP/s/device.**
- Global throughput ~645 x 64 ≈ **~41,300 tokens/s**.
- Step 0 is the first-iteration warmup (one-time runtime overhead after compile). Because there is **no profiler this time, step 2 is clean** (~37.7 s) — in the earlier xplane run step 2 was inflated to ~43 s by the trace dump.
- Loss descends monotonically (12.262 -> 11.771). Status: **COMPLETED** (exit 0), Wall 9m32s.

## 6. Consistency check (clean baseline)

- This run reproduces the historical MaxText good-node numbers (**~33% MFU / ~646 tokens/s/device**) within variance -> established as the **throughput baseline** for subsequent tuning.
- Stack reproduced end-to-end: RCCL PR #2063 (no empty-stream churn), RDMA over ionic (surgical mount), pristine config (QPS=4 default, autotune level 0, no pipelined flags).

## 7. Profiling — deferred to a follow-up task

Profiling was intentionally **not** run here to keep this a clean throughput baseline. The
xplane (`jax.profiler`) capture is a separate next task; rerun with
`profiler=xplane skip_first_n_steps_for_profiler=1 profiler_steps=1` when profiling is needed.

Related: Primus side [`LLAMA31_405B_RUN_REPORT.md`](../Primus/LLAMA31_405B_RUN_REPORT.md) (same 8 nodes, same baseline run).
