# RCCL "Empty Stream" Problem — Verification Report

> **TL;DR** — The RCCL upgrade (ROCm/rccl [PR #2063](https://github.com/ROCm/rccl/pull/2063), "Use one side stream per process") **fixes** the empty-stream problem. A controlled A/B on an 8-node Llama-3.1-405B run (only `DOCKER_IMAGE` changed) shows the pre-upgrade image creates **3,853 HIP streams per node and destroys 3,598** (thousands of transient `memcpy`-only streams), while the upgraded image creates **248 and destroys 0** (a fixed, long-lived pool). On the GPU side the kernel trace drops from **3,666 streams (98.6% with only 1-2 kernels)** to **82 streams**. Net: **~45x fewer streams, churn eliminated**, plus a small throughput gain (MFU 30.51% -> 31.62%).

---

## 1. What the "empty stream" problem is

Symptom (observed on the pre-upgrade image): **several thousand GPU streams per process, each containing only one or two kernels, and those kernels are `memcpy`/fill operations.**

Root cause: stock RCCL 2.27.7 creates and destroys a fresh "side stream" for collective bookkeeping work instead of reusing a single persistent one. Over a training run this produces thousands of short-lived streams (create -> issue 1-2 copy kernels -> destroy). Besides the allocation churn, this **interferes with HIP graph capture** (CUDA-graph-equivalent), which is the motivation for the upstream fix.

Fix: ROCm/rccl [PR #2063](https://github.com/ROCm/rccl/pull/2063) "Use one side stream per process" (merge commit `185e78a8`). Our upgraded image rebuilds RCCL with this patch; see [`docker/RCCL_BUILD_AND_VERIFY.md`](docker/RCCL_BUILD_AND_VERIFY.md) and [`RCCL_UPGRADE.md`](RCCL_UPGRADE.md).

| Image | RCCL version | PR #2063 |
|-------|--------------|----------|
| `rocm/jax-training:maxtext-v26.2` (pre-upgrade) | `2.27.7` (stock) | **no** |
| `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (upgraded; rebuilt from the base, see [`docker/RCCL_BUILD_AND_VERIFY.md`](docker/RCCL_BUILD_AND_VERIFY.md)) | `2.27.7-HEAD:185e78a` | **yes** |

---

## 2. Experiment design (single-variable A/B)

Both jobs are identical except for the one variable under test, `DOCKER_IMAGE`:

- Model / scale: Llama-3.1-405B FP8, **8 nodes x 8 MI355X = 64 GPUs**, `per_device_batch_size=3`, `steps=10`.
- `NCCL_IB_QPS_PER_CONNECTION=1` (keeps the ionic NIC's 4096-QP table from overflowing at 8-node scale).
- Surgical ionic-provider mount (`USE_HOST_IONIC_PROVIDER_ONLY=true`) so RCCL runs over RoCE/IB on both images.
- Profiling: `_env_ROCPROF_TRACE=1` (rocprofv3, traces `kernel,hip,rccl,marker`).
- 7 of 8 nodes are common to both runs; **one common node is used for the head-to-head numbers below** (1 JAX process per node drives all 8 local GPUs, so each per-node count covers 8 GPUs).

| Run | Job ID | Image | MFU | Result |
|-----|--------|-------|-----|--------|
| Upgraded (B) | `15687` | `...rccl-pr2063` | **31.62%** | COMPLETED |
| Pre-upgrade (A) | `15689` | `jax-training:maxtext-v26.2` | **30.51%** | COMPLETED |

Loss was identical (11.772) at the last step, confirming the two runs differ only in the RCCL build.

---

## 3. Results

### 3.1 HIP stream allocation (`hip_api_trace.csv`, one node = 8 GPUs)

| Metric | Pre-upgrade (A) | Upgraded (B) | Ratio |
|--------|-----------------|--------------|-------|
| `hipStreamCreateWithFlags` | 3,821 | 216 | |
| `hipStreamCreateWithPriority` | 32 | 32 | |
| **Total stream creates** | **3,853** | **248** | **~15.5x** |
| **`hipStreamDestroy`** | **3,598** | **0** | churn vs none |
| Streams created per GPU | ~481 | ~31 | |

The pre-upgrade image **creates and destroys thousands of streams** (3,853 created / 3,598 destroyed = continuous churn). The upgraded image creates a **fixed pool of 248 (~31/GPU) and never destroys any** — they are allocated once at startup and live for the whole run.

> Note: ~31 streams/GPU on the upgraded image matches the expected PJRT `LocalDeviceState` + XLA `StreamPool` architecture (the 32 `WithPriority` creates = 4/GPU high-priority streams).

### 3.2 GPU-side view (`kernel_trace.csv`, one node = 8 GPUs)

This is the direct reproduction of the reported symptom (distinct `Stream_Id` that dispatched at least one kernel):

| Metric | Pre-upgrade (A) | Upgraded (B) |
|--------|-----------------|--------------|
| Distinct streams (Agent, Stream) | **3,666** | **82** |
| Streams per GPU | ~458 | ~10 |
| Streams with <= 2 kernels | **3,614 (98.6%)** | 15 (18.3%) |
| &nbsp;&nbsp;- with exactly 1 kernel | 3,606 | 8 |
| Top kernels in the <=2-kernel streams | `__amd_rocclr_fillBufferAligned` (3,612), `__amd_rocclr_copyBuffer` (9) | same kernels, but only ~15 streams |

The pre-upgrade run reproduces the symptom exactly: **~3,666 streams, 98.6% of them carrying only 1-2 kernels, and those kernels are `fillBufferAligned`/`copyBuffer` (i.e. memcpy/fill)**. The upgraded run concentrates work into a small set of long-lived streams (24 streams with >1000 kernels each); its ~15 tiny streams are normal one-off init/copy, not pathological churn.

### 3.3 Throughput impact

MFU **30.51% (A) -> 31.62% (B)**, ~1.1 percentage points (~3.5% relative). In this configuration HIP command buffers / graph capture are disabled (`--xla_gpu_enable_command_buffer=''`), so the gain here is only the saved stream-management overhead. The larger benefit of PR #2063 is unblocking HIP graph capture, which is not exercised here.

---

## 4. How to detect / reproduce

The signal is captured by rocprofv3 (`_env_ROCPROF_TRACE=1`, see [`_train.sh`](_train.sh)). After a run, per-PID traces land under `<job_dir>/rocprof/<host>/<pid>/`.

**Count stream creation / destruction** (headline metric, one process = 8 GPUs):

```bash
HIP=<job_dir>/rocprof/<host>/<pid>/<host>/<pid>_hip_api_trace.csv
# creates (all variants) and destroys
awk -F, 'NR>1 && $2 ~ /StreamCreate/  {n++} END{print "creates :", n+0}' "$HIP"
awk -F, 'NR>1 && $2 ~ /StreamDestroy/ {n++} END{print "destroys:", n+0}' "$HIP"
# full hipStream* breakdown
awk -F, 'NR>1 && $2 ~ /[Ss]tream/ {c[$2]++} END{for(k in c) printf "%10d  %s\n",c[k],k}' "$HIP" | sort -rn
```

Healthy (upgraded): creates ~= 31/GPU, destroys = 0.
Buggy (pre-upgrade): creates and destroys both in the thousands.

**GPU-side stream/kernel histogram** — use [`utils/empty_stream_check.py`](utils/empty_stream_check.py), which counts distinct `(Agent_Id, Stream_Id)` pairs in `kernel_trace.csv`, buckets them by kernels-per-stream, and lists the kernel names in the <=2-kernel streams:

```bash
python3 utils/empty_stream_check.py '<job_dir>/rocprof/*<node>*/*/*/*_kernel_trace.csv'
```

A buggy run shows thousands of 1-kernel streams running `fillBufferAligned`/`copyBuffer`; a healthy run shows ~10 streams/GPU concentrated in a few long-lived ones.

> Caveat: `kernel_trace.csv` only contains streams that dispatched a kernel; pure DMA-copy streams need `--memory-copy-trace` (add `memory-copy` to `_env_ROCPROF_TRACES`). For the empty-stream bug both the HIP-API count and the kernel-trace view agree.

---

## 5. Appendix — artifact locations

```
Upgraded (B):    $JOB_WORKSPACE/15687-JAX-llama3.1-405b-...-_env_ROCPROF_TRACE_1/rocprof/
Pre-upgrade (A): $JOB_WORKSPACE/15689-JAX-llama3.1-405b-...-_env_ROCPROF_TRACE_1/rocprof/
```

Each `rocprof/<host>/<pid>/<host>/` holds `*_results.pftrace` (Perfetto GPU trace), `*_kernel_trace.csv`, `*_hip_api_trace.csv`, `*_rccl_api_trace.csv`, `*_marker_api_trace.csv`.

Verdict: **the RCCL upgrade is effective** — the empty-stream churn is eliminated (thousands -> a fixed pool with zero destroys), confirmed by both the HIP-API allocation count and the GPU kernel trace.
