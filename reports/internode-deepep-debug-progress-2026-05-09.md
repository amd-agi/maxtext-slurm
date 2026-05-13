# Internode DeepEP Debug Progress Report

Date: 2026-05-09

Goal: run `deepseek3-671b-proxy` with internode DeepEP on 4 nodes. Current status: the original bootstrap/data-exchange failures are localized and worked around in the standalone repro; the remaining crash is the first real internode DeepEP GPU dispatch kernel.

## Current Conclusion

The problem has been narrowed substantially:

- `use_turbo_grouped_gemm=True` without DeepEP is known-good from earlier jobs (`use_deepep_dispatch=False`).
- 1-node standalone DeepEP repro passes dispatch and combine.
- 2-node standalone DeepEP repro now completes DeepEP bootstrap after replacing the broken JAX `process_allgather` with a JAX coordinator KV-store gather.
- The remaining failure is only in the first real inter-node GPU dispatch kernel: all 16 ranks queue `moe_dispatch`, then node-1 GPUs fault on address `(nil)` before `moe_dispatch GPU-DONE`.

Therefore the next best direction is direct Primus-Turbo / DeepEP instrumentation and one rebuild, not more external host-side probes.

## Important Task Log

| Task / Job | Command | Exec reason | Result | Next direction |
|---|---|---|---|---|
| Confirm non-DeepEP baseline | Historical 4N runs: `use_turbo_grouped_gemm=True use_deepep_dispatch=False` | Prove model + sharding + grouped GEMM are healthy without internode DeepEP. | Baseline completed step 0 with healthy loss. | Keep focus strictly on `use_deepep_dispatch=True`. |
| 2N canonical DeepEP repro | `bash submit.sh deepseek3-671b-proxy -N 2 -p amd-rccl -t ... -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 use_deepep_dispatch=True ...` | Check whether the 4N failure reproduces with fewer nodes. | Same `hipError_t(9)` + `Memory access fault on (nil)` as 4N. | Use 2N as the faster debug target. |
| Unsupported XLA path check (`23268`) | `bash submit.sh ... -N 4 ... -- use_turbo_grouped_gemm=False shardy=False use_deepep_dispatch=True ...` | Verify user hypothesis that `tgmm=False` requires Shardy. | Compile-time fail: `RaggedDot is only supported with Shardy.` | Do not use `tgmm=False, shardy=False`; not relevant to DeepEP runtime crash. |
| Shardy ragged-dot OOM (`23262`) | `bash submit.sh ... -- use_turbo_grouped_gemm=False shardy=True use_deepep_dispatch=True ...` | Test the only legal non-TGMM ragged-dot path. | OOM allocating ~682 GiB, caused by bad Shardy lowering for this path. | Not a useful route for fixing internode DeepEP. |
| ROCm verbose logging (`23281`) | `bash submit.sh ... -N 2 ... -- _env_AMD_LOG_LEVEL=3 _env_AMD_LOG_MASK=0x82 ...` | Try to name the failing kernel. | Same crash; 1.7 GB log mostly rocBLAS/Tensile lookup noise, no useful launch name. | ROCm LOG_KERN/LOG_CMD is a dead end for this issue. |
| XLA HLO dump attempt (`23282`) | `bash submit.sh ... -- _env_XLA_FLAGS="--xla_dump_to=... --xla_dump_hlo_as_text"` | Correlate failing op with HLO. | Bad env quoting through launcher; XLA did not get the intended flags. | Only retry if needed, with safer quoted/encoded flags. |
| rocSHMEM heap-size hypothesis (`23283`, `23284`) | `bash submit.sh ... -- _env_ROCSHMEM_HEAP_SIZE=4294967296 ...` for 2N and 4N | Test whether default rocSHMEM heap exhaustion causes `(nil)` fault. | Both failed with the same `hipError_t(9)` + `(nil)` fault. | Heap exhaustion rejected. |
| HIP launch blocking (`23285`, `23314`) | `bash submit.sh ... -- _env_HIP_LAUNCH_BLOCKING=1 ...` | Try to pin the failing kernel synchronously. | Too slow; timed out before useful point, same pattern as earlier blocking attempts. | Do not use blocking for this workload. |
| Build standalone repro | Added `utils/repro_internode_deepep.py`; enabled via `_env_REPRO_INTERNODE_DEEPEP=1` in `utils/mfu_tracker.py` | Avoid full MaxText compile/train path; directly exercise `moe_dispatch` + `moe_combine`. | Repro gives 30-60 sec Python-side iteration after image load and precise phase markers. | Continue using this repro for DeepEP runtime experiments. |
| Bootstrap localization (`23287`) | `bash submit.sh deepseek3-671b-proxy -N 2 -p amd-rccl -t 00:15:00 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=repro-K2` | Determine whether crash is bootstrap, dispatch, or combine. | `_bootstrap_per_process` failed before any dispatch kernel. C++ errors at `deep_ep.cpp:917` (`hipIpcOpenMemHandle invalid argument`) and `:922` (self IPC handle mismatch). | Instrument handle allgather. |
| IPC-handle hash diagnostic (`23288`) | `bash submit.sh ... -N 2 ... -- _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=repro-K3-hash` | Check whether gathered IPC handles match each rank's local handle. | One gathered slot was `sha1(64 zero bytes)`, not the rank's handle. | Test whether JAX `process_allgather` itself drops data. |
| Raw JAX allgather pre-test (`23290`) | `bash submit.sh ... -N 2 ... -- _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K4-pretest` | Isolate JAX allgather from DeepEP/HIP allocation. | Each rank contributed `(rank+1)*ones(64)`; gathered result had one slot zeroed (`[1,2,3,4,0,6,...]`). | JAX `multihost_utils.process_allgather` is corrupting/dropping one rank's contribution. |
| KV-store gather fix (`23302`) | `bash submit.sh ... -N 2 ... -- _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K5-fixA` | Compare raw allgather, `block_until_ready`, and JAX coordinator KV-store gather. | Raw dropped a slot; `block_until_ready` duplicated another rank into that slot; KV-store gather was correct. Bootstrap succeeded on all 16 ranks, and `moe_dispatch` queued for all ranks. | Replace bootstrap's allgather with KV-store gather in real runtime. Remaining fault moves to GPU dispatch. |
| Early-return + dispatch sync (`23310`) | `bash submit.sh ... -N 2 ... -- _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K6-fixA-noredo` | Avoid double-bootstrap and place `block_until_ready()` inside dispatch/combine wrappers. | Bootstrap exactly once per rank, then early-return on lazy dispatch call. All 16 ranks queued `moe_dispatch`; 0 reached `moe_dispatch GPU-DONE`; node-1 GPUs faulted on `(nil)`. | The remaining bug is inside the first inter-node dispatch GPU kernel. |
| 1-node sanity (`23313`) | `bash submit.sh deepseek3-671b-proxy -N 1 -p amd-rccl -t 00:15:00 -- _env_ONE_GPU_PER_PROCESS=true _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K7-1node-sanity` | Verify standalone repro and intranode path are healthy. | Completed successfully: all 8 ranks reached `moe_dispatch GPU-DONE`, `moe_combine GPU-DONE`, `iter=0 === step OK`, and `PASS`. | Confirms remaining problem is only internode rocSHMEM path. |
| ctypes rocSHMEM state probes (`23315`, `23318`) | `bash submit.sh ... -N 2 ... -- _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K9-rsstate / K10-rsstate-v2` | Try external host-side rocSHMEM state inspection after bootstrap. | Bootstrap still succeeded and dispatch still faulted. ctypes could not resolve exported `rocshmem_my_pe` / `rocshmem_n_pes` symbols through standard paths or `RTLD_DEFAULT`; no useful peer-state table produced. | Stop external probing; instrument DeepEP/rocSHMEM integration directly and rebuild once. |

## Key Evidence

### JAX allgather corruption

The minimal pre-test contributed `(rank+1)` from each rank. Expected:

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

Observed with raw `multihost_utils.process_allgather`:

```text
[1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

Observed with `block_until_ready()` before allgather:

```text
[1, 2, 3, 4, 9, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

Observed with JAX coordinator KV-store gather:

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

Conclusion: `block_until_ready()` is not sufficient; the bootstrap handle exchange must bypass XLA/JAX collective allgather.

### Standalone 1-node success

`23313` proves intranode path is healthy:

```text
moe_dispatch GPU-DONE   (8/8 ranks)
moe_combine GPU-DONE    (8/8 ranks)
iter=0 === step OK      (8/8 ranks)
PASS                    (8/8 ranks)
```

### Remaining 2-node failure

`23310` / `23318` after KV bootstrap:

```text
_bootstrap_per_process EXIT OK      (16/16 ranks)
_bootstrap_per_process EARLY-RETURN (16/16 ranks)
moe_dispatch QUEUED                 (16/16 ranks)
moe_dispatch GPU-DONE               (0/16 ranks)
Memory access fault on address (nil) on node-1 GPUs
```

Conclusion: bootstrap and Python/JAX setup are no longer the blocker. The remaining fault is in the first inter-node DeepEP GPU dispatch kernel.

## Recommended Next Step

Instrument Primus-Turbo / DeepEP directly and rebuild once. External diagnostics have served their purpose; further ctypes/host-side checks are lower signal because a host-side rocSHMEM API success would not prove the GPU-side `rocshmem_int_put_nbi()` path is correct.

Add minimal logging / assertions in:

1. `Primus-Turbo/csrc/jax/deep_ep/deep_ep.cpp`
   - `Buffer::SyncFromIPCHandles()`
   - Print `rank_`, `nvl_rank_`, `rdma_rank_`, `num_ranks_`, `num_nvl_ranks_`, `num_rdma_ranks_`.
   - Print `buffer_ptrs_[i]`, `barrier_signal_ptrs_[i]`, `buffer_ptrs_gpu_`, `barrier_signal_ptrs_gpu_`.
   - Print `rdma_buffer_ptr_`, `num_rdma_bytes_`, `num_nvl_bytes_`.
   - Null-check `internode::alloc()` before `hipMemset`; fail with a clear error if null.

2. `Primus-Turbo/csrc/kernels/deep_ep/runtime.hip`
   - `internode::init()`
   - Print input `rank`, `num_ranks`, `low_latency_mode`, unique-id hash.
   - After `rocshmem_init_attr`, print `rocshmem_my_pe()` and `rocshmem_n_pes()`.
   - If available in this rocSHMEM build, print heap base / heap size / team / ctx internals or wrapper-visible equivalents.

3. `Primus-Turbo/csrc/kernels/deep_ep/internode.hip`
   - Right before the first dispatch launch (`notify_dispatch` and the main dispatch kernel launch), print:
     - `rank`, `nvl_rank`, `rdma_rank`, destination PE(s), `num_ranks`, `num_rdma_ranks`.
     - `rdma_buffer_ptr`, `buffer_ptrs`, `buffer_ptrs[i]`, `barrier_signal_ptrs[i]`.
     - token counts: `num_tokens`, `num_worst_tokens`, `num_tokens_per_rdma_rank`.
   - Add host-side `PRIMUS_TURBO_CHECK` assertions for null pointers before launching the kernel.

Expected outcome:

- If a pointer is already null before launch, root cause is host-side setup / pointer table / symmetric heap mapping.
- If all pointers and PE metadata are valid before launch, root cause is inside the GPU-side rocSHMEM put/barrier path. Then instrument the device kernel around the first `rocshmem_int_put_nbi()` / barrier site or temporarily narrow the dispatch kernel to a minimal GPU rocSHMEM put repro.

## Immediate Action Plan

1. Port the KV-store bootstrap gather into `primus_turbo/jax/deep_ep/runtime.py` cleanly.
2. Add the C++/HIP logging above.
3. Rebuild the DeepEP extension / Docker image once.
4. Run the 2-node standalone repro first:

```bash
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm
bash submit.sh deepseek3-671b-proxy \
  -N 2 -p amd-rccl -t 00:20:00 \
  -- \
  _env_ONE_GPU_PER_PROCESS=true \
  _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
  _env_REPRO_INTERNODE_DEEPEP=1 \
  exp_tag=direct-deepep-instrumentation
```

5. After the standalone repro passes dispatch/combine on 2 nodes, bind-mount or bake the fixed runtime into the container and rerun the full 4-node `deepseek3-671b-proxy` internode DeepEP job.

## Instrumentation Pass 1 (2026-05-09)

Goal: maximise per-rebuild signal because rebuilding Primus-Turbo and rebaking the
container image takes long. Instead of narrowing down with iterative probes, we
spray markers across every plausible failure site so the next single 2N run can
either pinpoint the fault or eliminate large parts of the search space.

### Files modified (source of truth = `*.cu` / `*.cpp`; hipify regenerates `*.hip`)

| File | PT_ markers | Coverage |
|---|---:|---|
| `Primus-Turbo/csrc/kernels/deep_ep/runtime.cu` | 22 | rocSHMEM lifecycle host wrappers (`get_unique_id` / `init` / `alloc` / `free` / `barrier` / `finalize`) |
| `Primus-Turbo/csrc/jax/deep_ep/deep_ep.cpp` | 46 | `Buffer` ctor, `Sync`, `SyncFromIPCHandles`, `InternodeDispatch` (incl. CPU sync loop with per-second progress), `InternodeCombine`, `Destroy`; null checks for `rdma_buffer_ptr_` / `buffer_ptrs_gpu_` / `barrier_signal_ptrs_gpu_` before kernel launches |
| `Primus-Turbo/csrc/kernels/deep_ep/internode.cu` | 42 | All 4 host wrappers (`notify_dispatch` / `dispatch` / `cached_notify` / `combine`) print params + `hipGetLastError()` after launch; all 4 device kernels carry ENTER / EXIT markers; `notify_dispatch` carries 14 stage markers covering every rocSHMEM / barrier / clean / counter-write step in SM-0 plus per-thread `put_nbi PRE` / `POST` lines |

### Log prefix legend (grep cheat sheet)

| Prefix | Meaning | Source |
|---|---|---|
| `[PT-RS dev=N] ...` | rocSHMEM API wrapper | `runtime.cu` |
| `[PT-DEP dev=N] ...` | `Buffer` lifecycle / Internode{Dispatch,Combine} host | `deep_ep.cpp` |
| `[PT-IN-H dev=N] ...` | Internode kernel host launch wrappers | `internode.cu` host |
| `[PT-K rank=R sm=S tid=T] ...` | Device-side guarded `printf` | `internode.cu` device |

### Highest-signal probes (decision tree)

1. **`[PT-RS] init: rocshmem_init_attr OK arg_rank=R arg_num=N -> my_pe=X n_pes=Y`** — if any PE shows `n_pes != arg_num`, root cause is rocSHMEM bootstrap, ignore everything below.
2. **`[PT-RS] alloc: rocshmem_ptr(P, peer=K) = Q`** — printed for every peer right after `rocshmem_malloc`. **Any `Q == (nil)` is the same `(nil)` the GPU later faults on.**
3. **`[PT-DEP] InternodeDispatch ENTER` then `[PT-IN-H] notify_dispatch HOST: launch returned hipGetLastError=...`** — if exit code is non-zero, the launch itself was rejected (parameter or grid-size issue).
4. **`[PT-K rank=R sm=0 tid=0] notify_dispatch DEV stage=...`** — 14 ordered stage markers. The last marker before silence locates the fault to a single basic block in SM-0.
5. **`[PT-K] put_nbi PRE dst_pe=K dst=P src=Q nelem=N`** vs. matching **`POST`** — only thread 0..kNumRDMARanks-1 of block 0 emit these (≤ 2 PRE + 2 POST per GPU per launch). Missing POST → fault is in `rocshmem_int_put_nbi`; cross-reference `dst_pe` against the `[PT-RS] alloc` `rocshmem_ptr(...,peer=dst_pe)` line to decide between symmetric-heap miss and PE-index error.
6. **`[PT-DEP] InternodeDispatch: CPU sync waiting/READY/TIMEOUT`** — emitted once per second; lets us tell whether `*moe_recv_counter_mapped` was ever written by the GPU side. The matching device-side line is `[PT-K] notify_dispatch DEV stage=post-set-counter sum=X`.
7. **`[PT-K] dispatch DEV ENTER`** is printed **before** `rocshmem_wg_ctx_create`, so even if `wg_ctx_create` itself crashes, ENTER will appear and `post-wg_ctx_create` will not — that combination uniquely identifies a rocSHMEM context-creation fault.

### Notes on safety / overhead

- All host probes are `fprintf(stderr, ...)` + `fflush(stderr)` so they survive an immediate process abort.
- All device probes are guarded to `blockIdx.x == 0 && (threadIdx.x == 0 || thread_id < kNumRDMARanks)`, keeping per-GPU-per-launch printf volume to ≲ 25 lines. Device printf in HIP is buffered, but log volume is small enough that loss-on-crash is unlikely.
- `notify_dispatch`'s SM-1+ branch (the per-`dst_rdma_rank` token-counting block) only carries ENTER + EXIT markers because it is plain compute with no rocSHMEM calls; if pass-1 fingers SM-1+, pass-2 will add inner stage markers there.
- `combine` and `cached_notify` device kernels only carry ENTER/EXIT because the current bug never reaches them; full instrumentation is deferred to pass-2 if dispatch starts succeeding.

### Build / run plan for this pass

```bash
cd /home/liyingli/workspace/jax-deepep/Primus-Turbo
# Rebuild + rebake the container image once (root-owned `*.hip` are regenerated
# by hipify from the writable `*.cu` source files).

# Then:
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm
bash submit.sh deepseek3-671b-proxy \
  -N 2 -p amd-rccl -t 00:20:00 \
  -- \
  _env_ONE_GPU_PER_PROCESS=true \
  _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
  _env_REPRO_INTERNODE_DEEPEP=1 \
  exp_tag=instrumented-pass1

# Triage:
rg "^\[PT-(RS|DEP|IN-H|K)" slurm-<JOBID>.out | head -300
```

Expected outcomes feeding pass-2:

- Most likely: a single `[PT-K] put_nbi PRE` line on a node-1 GPU with no matching `POST`, plus a `[PT-RS] alloc: rocshmem_ptr(... peer=<that dst_pe>) = (nil)` line above. → fix at the rocSHMEM symmetric heap layer (PE / heap / team setup) and re-run.
- Less likely: `n_pes != num_rdma_ranks_` mismatch in `[PT-RS] init`. → fix at the unique-id / `rocshmem_init_attr` level.
- Least likely: all rocSHMEM probes look healthy, fault occurs after `notify_dispatch DEV stage=EXIT-sm0` and before `dispatch DEV ENTER`. → fault is between kernel launches, look at the host-side stream / hipMemset / counter-mapped pages.

## Pass-1 actual result (Job 23457)

`23457` was Pass-1 with the full marker spray. Outcome:

- `[PT-RS] init` (16/16 PEs): every PE entered `rocshmem_init_attr` and the host returned matching `my_pe`/`n_pes=2` for its 2-PE world. Per-process pairing (node0/dev_i ↔ node1/dev_i) confirmed.
- `[PT-RS] alloc rocshmem_ptr(P, peer=K)`: each PE printed two lines; for the local peer `Q` was non-NULL, for the cross-node peer `Q == (nil)` — that is the correct, expected behavior of rocSHMEM under the IPC-only backend. **Not** the `(nil)` the GPU later faults on.
- `[PT-DEP] SyncFromIPCHandles`: NVL IPC handles for all 8 peers opened on every PE; pointer tables uploaded to GPU.
- `[PT-IN-H] notify_dispatch HOST: launch returned hipGetLastError=0` (16/16): all `notify_dispatch` launches succeeded host-side.
- `[PT-K] notify_dispatch DEV stage=`...
  - All observed PEs printed `pre-barrier1(rdma)` and then `post-barrier1(rdma) pre-barrier1(nvl)`.
  - **None** printed `post-barrier1(nvl)`.
  - All 8 GPUs on node 1 simultaneously reported `Memory access fault on address (nil)`.

### Mistake made and corrected (transcript record)

A first read of the markers above led to: *"`rocshmem_barrier_all` succeeded; the fault is in `barrier_block`."* This is **wrong**. `PT_K_BLOCK0_LOG` only fires on `threadIdx.x == 0`; the rocSHMEM barrier executes on `threadIdx.x == kWarpSize == 64`. Thread 0 takes the `if (thread_id == kWarpSize)` false branch and falls through to the next marker without waiting for thread 64 to finish. Therefore the markers do **not** distinguish whether `rocshmem_barrier_all` or the subsequent `barrier_block` is the actual crash site.

A separate "cross-fatbin `ROCSHMEM_CTX_DEFAULT` is `nullptr` in the Primus-Turbo fatbin" hypothesis was also stated as deterministic. That is **not justified** under the current build: `setup.py` links rocSHMEM as `-l:librocshmem.a -fgpu-rdc --hip-link`, which produces a single fatbin with a single `ROCSHMEM_CTX_DEFAULT` symbol; init-time `hipMemcpyToSymbol` writes the only copy. Pass-2 carries the bridge code defensively (it is a no-op under static-link; see below) but it is no longer the asserted root cause.

### Pass-1 → Pass-2 questions still open

- Is the device-side `rocshmem_barrier_all()` (in `nvshmem_barrier_with_same_gpu_idx`, `internode.cu` L143) the crash site, or is it the immediately-following `barrier_block` (NVL IPC atomic add/sub on peer-mapped addresses)?
- The PyTorch DeepEP at Primus-Turbo uses the same `internode::*` kernels and is reported to have run inter-node DeepEP successfully under Megatron previously. So the kernel itself is not categorically broken; some build- or runtime-context delta between that scenario and the JAX path is the explanatory variable.

## Instrumentation Pass 2 (2026-05-09)

Goal: disambiguate the Pass-1 ambiguity (rocSHMEM barrier vs NVL barrier), and at the same time take the JAX-side `process_allgather` workaround that the standalone repro had been monkey-patching and bake it into Primus-Turbo's runtime so the full `deepseek3-671b-proxy` training path can drive the same code paths and confirm the fault is not an artifact of the repro harness.

### Files modified (Pass 2)

| File | Probe / change | Purpose |
|---|---|---|
| `Primus-Turbo/csrc/kernels/deep_ep/runtime.cu` | `init`: defensive `hipMemcpyFromSymbol`/`hipMemcpyToSymbol` of `rocshmem::ROCSHMEM_CTX_DEFAULT` after `rocshmem_init_attr`, with `[PT-RS] init: bridge READ/WRITE` log lines including `ctx_opaque` / `team_opaque`. | If `ctx_opaque == nullptr` after init, the next device kernel using `ROCSHMEM_CTX_DEFAULT` will (nil)-fault; the readback prints it explicitly per PE, and the writeback bridges any cross-fatbin instance (no-op under static-link single fatbin). |
| `Primus-Turbo/csrc/kernels/deep_ep/internode.cu` | `notify_dispatch` SM-0: PRE/POST `printf` around the `nvshmem_barrier_with_same_gpu_idx` call on `thread_id == kWarpSize` (probe **A**). | Distinguishes "rocSHMEM barrier completed" from "rocSHMEM barrier crashed". `PT_K_BLOCK0_LOG` thread-0 markers do not. |
| `Primus-Turbo/csrc/kernels/deep_ep/utils.cuh` | `barrier_block`: PRE-atomic and POST-atomic and POST-spin `printf` per thread `0..kNumRanks-1` of `blockIdx.x == 0`, including `barrier_signal_ptrs[rank]+tid`, `barrier_signal_ptrs[tid]+rank`, and the source `barrier_signal_ptrs[*]` (probe **B**). | Distinguishes "NVL barrier crashed at the cross-write atomic" from "stuck in spin loop"; surfaces any NULL or stale peer pointer. |
| `Primus-Turbo/primus_turbo/jax/deep_ep/runtime.py` | `_get_root_rocshmem_unique_id` and `_bootstrap_per_process` now allgather rocSHMEM unique IDs and IPC handles via a new `_kv_allgather_bytes` (JAX coordinator KV-store) instead of `multihost_utils.process_allgather`. | The standalone repro confirmed `process_allgather` zero-fills one rank's slot under XLA/RCCL; baking the workaround into the real runtime lets the full `deepseek3-671b-proxy` job drive the same dispatch path the repro drives. |

### Probe-A / Probe-B grep cheat sheet

- `[PT-K rank=R sm=0 tid=64] PRE  rocshmem_barrier_all team=...` and `POST rocshmem_barrier_all OK` — PRE without POST = rocSHMEM barrier crashed.
- `[PT-K-BB rank=R sm=0 tid=T] PRE atomicAdd/Sub self_slot=... peer_slot=... (peer_self=... peer_tid=...)` — values printed before the cross-write; PRE without `POST atomicAdd/Sub OK` = NVL atomic crashed (most likely a NULL `peer_tid` pointer).
- `[PT-K-BB rank=R sm=0 tid=T] EXIT spin loop done` — full `barrier_block` completed.

### Bridge log cheat sheet

- `[PT-RS] init: bridge READ ROCSHMEM_CTX_DEFAULT (Primus-Turbo fatbin) ctx_opaque=... team_opaque=...` — printed once per PE right after `rocshmem_init_attr`.
  - `ctx_opaque != nullptr` (expected under static-link): bridge writeback is a no-op; cross-fatbin null-deref is ruled out for that PE.
  - `ctx_opaque == nullptr`: a `WARNING` line is emitted; means the device-side default ctx in Primus-Turbo's fatbin was never written by init — investigate cross-fatbin layout.

### Build / run plan for this pass

```bash
cd /home/liyingli/workspace/jax-deepep/Primus-Turbo
# Rebuild Primus-Turbo (.cu sources are authoritative; hipify regenerates *.hip)
# and rebake the container image once.

# Step 1: full deepseek3-671b-proxy with REPRO disabled, using the in-runtime
# KV-store allgather. This validates that the (nil) fault still reproduces from
# the real MaxText path (not from the repro harness).
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm
bash submit.sh deepseek3-671b-proxy \
  -N 2 -p amd-rccl -t 00:30:00 -x useocpm2m-097-039 \
  -- \
  _env_ONE_GPU_PER_PROCESS=true \
  _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
  exp_tag=pass2-fullrun-no-repro

# Step 2 (optional, if pass-2 needs reproduction with the repro harness):
bash submit.sh deepseek3-671b-proxy \
  -N 2 -p amd-rccl -t 00:20:00 -x useocpm2m-097-039 \
  -- \
  _env_ONE_GPU_PER_PROCESS=true \
  _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
  _env_REPRO_INTERNODE_DEEPEP=1 \
  exp_tag=pass2-repro
```

Triage with the new probes:

```bash
# rocSHMEM CTX bridge (1 line per PE).
grep -F "[PT-RS" <log> | grep -F "bridge"

# Probe A: rocSHMEM barrier on thread 64.
grep -E "PRE  rocshmem_barrier_all|POST rocshmem_barrier_all" <log>

# Probe B: NVL barrier_block.
grep -F "[PT-K-BB" <log>
```

Decision tree:

- bridge READ shows `ctx_opaque == nullptr` → cross-fatbin null-deref is the root cause; investigate why static-link did not put the symbol in the path that init writes (link ordering, multiple `librocshmem.a` copies, accidental dlopen of `librocshmem.so`).
- Probe-A `PRE` without `POST` → fault is inside `rocshmem_barrier_all` (device-side rocSHMEM); compare ROCm/rocSHMEM versions vs. the PyTorch+Megatron runs that worked.
- Probe-A both `PRE` and `POST` printed, Probe-B `PRE atomicAdd/Sub` without `POST atomicAdd/Sub OK` → fault is the NVL cross-write; inspect the printed `peer_tid=...` value (NULL ⇒ IPC table not loaded for some peer; otherwise stale mapping).
- Probe-B reaches `EXIT spin loop done` and the next marker (`post-barrier1(nvl)`) appears → the (nil) fault has moved further into `notify_dispatch` and the next stage marker is the new boundary.
