# rocprofv3 GPU Trace for 405B — Buffer-Overflow Fix & Setup

> **TL;DR** — Capturing a usable Perfetto GPU trace (`.pftrace`) for an 8-node Llama-3.1-405B run failed at first: the trace had **no GPU kernels** (GEMM/RCCL/attention), only host-side markers. Root cause: rocprofv3's **default 1 GB perfetto buffer overflows** at 405B scale (a whole-run, all-domain trace is > 1 GB), and the highest-volume domain — kernel dispatches — is silently dropped (the per-domain CSVs are unaffected). Fix: **trace only the lean per-domain set `kernel,rccl,marker`** instead of the full `--runtime-trace`. The trace then fits the buffer and keeps all GPU kernels, for the whole run. Verified: 405B steps=3 -> **731 MB pftrace with GEMM + attention + RCCL `ncclDevKernel`**. This is now the default in [`_train.sh`](_train.sh).

Reference: [rocprofv3 docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

---

## 1. Goal

For an 8-node 405B run, produce both: (a) a Perfetto **GPU trace** for compute/communication overlap analysis, and (b) per-kernel **statistics**. Both come from rocprofv3, enabled with `_env_ROCPROF_TRACE=1` (the wrapper lives in [`_train.sh`](_train.sh)).

## 2. Symptom

At 405B scale the `.pftrace` contained only host-side ROCTx markers (`nvte_flash_attn_*`, `nvte_populate_rng_state_*`) — **no GPU kernels**. The exact same flags at 8B scale *did* show kernels, which is the key clue.

## 3. Root cause: perfetto buffer overflow

rocprofv3 writes trace records into an in-process perfetto buffer that defaults to **1 GB** (`--perfetto-buffer-size`, per the docs). If the trace exceeds it, records are dropped — and the **highest-volume domain (kernel dispatches) is what gets dropped**, while the low-volume marker domain survives. The `kernel_trace.csv` is written independently of this buffer, so kernel **statistics** were always complete; only the Perfetto **timeline** lost its kernels.

| scale | pftrace size | GPU kernels in pftrace |
|-------|--------------|------------------------|
| 8B, whole run | 226 MB (< 1 GB) | present |
| 405B, whole run | 1189 MB (capped at ~1 GB) | **dropped** |

## 4. Dead-ends (what did not work)

- **Bigger buffer (8 GB)** — perfetto rejected it: `Failed to allocate tracing buffers: Invalid buffer sizes` (exceeds perfetto's cap). Tracing never started and the run spun with no output.
- **Windowed capture** (`--collection-period 330:80:1` to skip compile and grab ~2 steady steps) — produced a **completely empty trace** (no files at all). `--collection-period` is an opt-in time-window feature (triplet `start_delay:collection_time:repeat`, default unit = seconds); on rocprofv3 v1.0.0 a *delayed* window yields no trace output in this wrapper. The docs' application-tracing examples never use `--collection-period`; forcing it was the source of the empty trace.

## 5. Fix: trace a lean per-domain set

`--runtime-trace` records HIP-API + Marker + RCCL + memory ops + kernel dispatches. The **HIP-API and memory domains are huge** (HIP-API is ~one host call per kernel), so they dominate the trace volume. Dropping them shrinks the trace below the buffer limit while keeping everything needed for a GPU timeline:

```
ROCPROF_TRACES=kernel,rccl,marker     ->   --kernel-trace --rccl-trace --marker-trace
```

- `--kernel-trace` already holds **all GPU kernels**: GEMM (`Cijk_*`), attention (`aiter::fmha_*`), **and RCCL (`ncclDevKernel`)** — RCCL collectives are GPU kernel dispatches.
- `--rccl-trace` adds the host-side RCCL API; `--marker-trace` adds ROCTx region labels.
- No `--collection-period` (canonical tracing; the whole run is kept, not a window).

## 6. Verification

Same model/scale/parallelism; only the trace config changed:

| Run | Trace config | pftrace | GPU kernels? | MFU |
|-----|--------------|---------|--------------|-----|
| 405B steps=5 | `--runtime-trace`, whole run | 1189 MB | **no** (buffer overflow) | 31.9% |
| 405B steps=3 | **`kernel,rccl,marker`** | **731 MB** | **yes** (GEMM + attn + RCCL) | 32.1% |

Strings confirmed in the 731 MB trace: `Cijk_Ailk_Bjlk...` (GEMM), `aiter::fmha_fwd/bwd_hd128...` (attention), `ncclDevKernel_Generic_2` (RCCL). Throughput was Tokens/s/device ~636 (MFU 32.1%), i.e. profiling overhead is negligible vs the ~31.6-31.9% non-profiled baseline.

## 7. Config changes (`_train.sh`)

- `ROCPROF_TRACES` default is now **`kernel,rccl,marker`** (was the all-domain `runtime` preset).
- `--collection-period` is now **opt-in**: it is only added when `_env_ROCPROF_DELAY` or `_env_ROCPROF_DURATION` is set. Default = none (avoids the empty-trace failure above).

## 8. How to run and use it

```bash
# rocprofv3 with the lean domains (now the default); steps kept small for size
STEPS=3 ./run_llama3_1_405b.sh _env_ROCPROF_TRACE=1
```

Per-node, per-PID outputs land under:

```
$JOB_WORKSPACE/<JOB_ID>-.../rocprof/<host>/<pid>/<host>/
    <pid>_results.pftrace      # GPU trace (one per node, ~750 MB). Open in https://ui.perfetto.dev
    <pid>_kernel_trace.csv      # per-kernel records -> statistics
    <pid>_rccl_api_trace.csv    # host RCCL API
    <pid>_marker_api_trace.csv  # ROCTx region labels
```

Aggregate kernel statistics across all nodes:

```bash
python3 utils/rocprof_kernel_stats.py "$JOB_WORKSPACE/<JOB_ID>-.../rocprof" --shorten --top 30
```

## 9. Kernel breakdown (perf-analysis starting point)

From an 8-node 405B run (cluster-wide, share of GPU time):

- **FP8 GEMM** (`Cijk_*` Tensile kernels): ~57%
- **RCCL** (`ncclDevKernel`): ~23%
- **FlashAttention** (`aiter::fmha_*`): ~10%
- XLA fusions / transposes / copies: ~10%

The high RCCL share (with individual `ncclDevKernel` instances up to ~3 s) points to exposed/blocking communication — a prime target for compute/communication overlap analysis in the Perfetto timeline.

## 10. Caveats and tuning

- **Trace size scales ~linearly with steps.** `kernel,rccl,marker` is 731 MB at steps=3, so ~1.2 GB at steps=5 (would overflow again). For more steps: drop to just `kernel` (RCCL kernels are already in `--kernel-trace`), reduce steps, or raise the perfetto buffer to a *valid* size (a few GB, not 8 GB).
- `--collection-period` (windowing) is opt-in but produced empty traces with a non-zero delay on rocprofv3 v1.0.0 — prefer reducing domains/steps over windowing.
- Helper scripts: [`utils/rocprof_kernel_stats.py`](utils/rocprof_kernel_stats.py) (kernel stats), [`utils/empty_stream_check.py`](utils/empty_stream_check.py) (per-stream kernel histogram).
