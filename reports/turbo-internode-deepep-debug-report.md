# Turbo Internode DeepEP — Debug Report

**Date:** 2026-05-07  
**Owner:** liyingli  
**Goal:** Run `use_deepep_dispatch=True` (Primus-Turbo internode DeepEP) successfully on a 4-node DeepSeek-V3-671B proxy MaxText workload.  
**Outcome:** Root cause of the 4-node DeepEP internode crash isolated to a **rocSHMEM TCP-bootstrap timeout**. Fix candidates submitted; verification pending.

---

## 1. TL;DR

Across SLURM jobs **23116 → 23153** we removed three earlier obstacles (dirty node, wrong Docker image, fused-attention crash) and then nailed the actual internode-DeepEP failure. The 4-node crash on every "NVL slot 5 of non-root RDMA peer" rank is **not** a GPU/HIP-kernel bug. It is rocSHMEM's TCP bootstrap (`Socket::connect`) hitting its default **5-second** budget and calling `abort()` from `ERROR("connect timeout\n")` (`rocSHMEM/src/bootstrap/socket.cpp:501`). The fix candidates are:

```
ROCSHMEM_BOOTSTRAP_TIMEOUT=60
ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=eth0     # same iface NCCL uses successfully
ROCSHMEM_DEBUG_LEVEL=INFO                 # for self-diagnosing fallback
```

…and they are exercised by job **23153** (still pending due to amd-rccl queue priority).

---

## 2. Job timeline

| Job | Partition | Purpose / config delta | Outcome |
|----|----|----|----|
| 23116 | amd-rccl | First 4N attempt | Failed: 53 D-state `rccl-tests` procs on `useocpm2m-097-069` blocked GPU release in preflight; bad node, but `preflight.sh` *ignored* the failure and let the job burn time. |
| — | — | Patched `preflight.sh` (fail-fast on `release_gpu.sh` non-zero) and excluded the node. | — |
| 23118 | amd-rccl | Resubmitted | Failed: pyconfig `ValidationError: use_deepep_dispatch requires the primus_turbo package`. Cause: `DOCKER_IMAGE` env unset → defaulted to public `rocm/jax-training` (no `primus_turbo`). |
| — | — | Patched `container_env.sh` to default `DOCKER_IMAGE` to local DeepEP tarball. | — |
| 23122 | amd-rccl | Resubmitted with right image | Crashed in `p_train_step`: TE fused-attention `cudnn_flash_te` → `PopulateRngStateAsync: CUDA Error: invalid configuration argument` → `Memory access fault`. CK fused-attention bug for these shapes on ROCm. |
| 23123 | amd-rccl | Same nodes, `attention=dot_product` | Could not start: 23122 still holding nodes (post-crash). |
| — | — | `scancel 23122 23123`; waited for nodes idle. | — |
| **23124** | amd-rccl | Pinned to original 4 nodes (080/132/135/137), `attention=dot_product`, DeepEP enabled | **Reference crash.** ~6.5 min in, **3 ranks die simultaneously** at the first training step: rank 13/21/29 = **local GPU 5 on nodes 1, 2, 3**. Three coredumps, ~28 GB each, in the job output dir. |
| 23125 | amd-rccl | Same setup but `use_deepep_dispatch=False` | **Successful diagnostic.** Bug isolated to DeepEP internode dispatch/combine (not GSPMD/Shardy, not turbo grouped GEMM, not `dot_product`). |
| 23138 / 23140 | amd-arad | 1-node "gdb-on-coredump" debug job | 23138 failed (apt could not install gdb on offline compute nodes). 23140 used `/opt/rocm/bin/rocgdb` → **succeeded.** Backtrace produced. |
| — | — | Source-corroborated root cause via `/home/liyingli/rocSHMEM/`. | — |
| 23150 | amd-arad | Fix run on amd-arad | Failed *upstream* of rocSHMEM: container saw 0 AMD GPUs on amd-arad → JAX fell back to **CPU backend**; 32 "CPU devices" → multi-host op went through **Gloo**, which timed out at 30 s. amd-arad is unusable for this workload. |
| **23153** | amd-rccl | Fix run, 30-min walltime, `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`, `IFNAME=eth0`, `DEBUG_LEVEL=INFO` | **Pending (Priority).** Verification target. |

---

## 3. Patches applied

### 3.1 `utils/preflight.sh` — fail-fast on dirty GPUs

```bash
if ! bash "$SCRIPT_DIR/utils/release_gpu.sh"; then
    echo "FATAL: $HOST | preflight: GPU cleanup failed (likely D-state procs from prior job)"
    echo "FATAL: $HOST |   -> ask cluster admin to reboot $HOST, or resubmit with -x $HOST"
    exit 1
fi
```

Combined with `srun --kill-on-bad-exit=1`, this kills the job at preflight on any node that still has stuck GPU procs, instead of running 6 minutes of training and dying mid-collective.

### 3.2 `container_env.sh` — DeepEP image as default

```bash
DOCKER_IMAGE="${DOCKER_IMAGE:-/home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar}"
```

Stops `use_deepep_dispatch=True` configs from silently using the public `rocm/jax-training` image and failing pyconfig validation.

### 3.3 `utils/debug_bt_23124.sh` — new helper

A standalone script that submits a 1-node SLURM job, runs the same DeepEP container, mounts the 23124 coredump dir read-only at `/cores`, mounts the Primus-Turbo source at `/primus`, and dumps a non-interactive gdb backtrace to a writeable output dir. Detailed in §4.

---

## 4. Coredump debugging (detailed)

This is the section that turned a vague "tasks 13/21/29 stop heartbeating" into "rocSHMEM bootstrap times out at 5 s".

### 4.1 Constraints we had to work around

| Constraint | Implication |
|---|---|
| User has **no `sudo`** on the cluster. | Cannot reboot nodes, cannot read root-owned files directly, cannot drain nodes. |
| Coredumps were written by **root inside the container** to NFS, ending up `mode 0600 root:root`. | A non-root host user (`liyingli`) cannot `cat`/`gdb` them on the head node. |
| **Head node is a jump host** — heavy ops (`docker`, large `gdb` sessions) are forbidden there. | All container/gdb work must run inside a SLURM job. |
| **Compute nodes have no internet.** | `apt-get install gdb` fails — no DNS to `archive.ubuntu.com`. |
| Coredumps are **27 GB each**, on shared NFS. | Reading the whole core is slow; we need `gdb -batch` to dump just metadata + frames, not load the heap. |
| Job 23124 produced **3** cores (one per dying rank), all bit-equivalent in stack origin. | One core is enough to expose the crash site. |

### 4.2 Methodology

We treated this as a **post-mortem dump-only debug**, no live process. The plan was:

1. **Open a coredump in `gdb` inside the same container that wrote it**, so symbols (libc, libprimus_turbo_kernels, libxla_rocm_plugin) match the running process exactly.
2. **Run `gdb -batch`** with explicit commands to write a backtrace to a file we can read. No interactive shell — the SLURM job runs to completion.
3. **Cross-reference the backtrace with rocSHMEM source** (`/home/liyingli/rocSHMEM/`) and the 23124 stdout to identify the failing line and the relevant env knob.

The container being `--privileged` and running as **root inside** is what bypasses the host's `0600 root:root` permissions — root in any user namespace with access to the inode can read the file.

### 4.3 Step 1: locate the coredumps

```bash
ls -lh /home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs/23124-*/core.*
# core.23124.<epoch>.1.useocpm2m-097-132.python3.320  <- rank 13 (task 1, GPU 5)
# core.23124.<epoch>.2.useocpm2m-097-135.python3.320  <- rank 21
# core.23124.<epoch>.3.useocpm2m-097-137.python3.320  <- rank 29
```

The naming pattern is `core.<jobid>.<epoch>.<task>.<host>.<exe>.<pid>` — we can read the **task index** and **PID** straight off the filename, which lets us correlate to the JAX process index.

### 4.4 Step 2: design the in-container gdb job

Three things make this delicate:

1. **Image identity matters.** We must use the same `jax-deepep-1p1g:v0.2` image the job ran with — otherwise gdb will warn about library mismatches and could resolve symbols incorrectly.
2. **Mounts.**
   - `-v <23124-output-dir>:/cores:ro` so the cores are visible inside the container.
   - `-v /home/liyingli/workspace/jax-deepep/Primus-Turbo:/primus:ro` so a future interactive session has source for line-level debugging.
   - `-v <writable-out-dir>:/output` for `bt.txt`.
3. **`--user 0 --privileged`** so the in-container UID is root and can `open(2)` the `0600 root:root` cores. (Without `--user`, the container would normally run as the image's default user, which in this image is root anyway, but we make it explicit.)

### 4.5 Step 3: the script (`utils/debug_bt_23124.sh`)

Key fragment:

```bash
docker run --rm --user 0 --privileged --ipc=host --network=host \
    -v "$CORE_DIR":/cores:ro \
    -v "$PRIMUS_DIR":/primus:ro \
    -v "$OUT_DIR":/output \
    -e CORE_BASENAME="$CORE_BASENAME" \
    jax-deepep-1p1g:v0.2 \
    bash -lc '
        # Compute nodes have no internet -> apt install fails. Prefer the
        # gdb that ROCm ships (rocgdb wraps gdb with AMDGPU extensions but
        # works on plain CPU coredumps), then fall back to any gdb on PATH.
        GDB=""
        for cand in /opt/rocm/bin/rocgdb /opt/rocm/lib/llvm/bin/gdb \
                    /opt/rocm/llvm/bin/gdb $(command -v gdb 2>/dev/null) \
                    $(command -v rocgdb 2>/dev/null); do
            [[ -x "$cand" ]] && { GDB="$cand"; break; }
        done
        PYBIN=$(readlink -f /opt/venv/bin/python3 || true)
        [[ -z "$PYBIN" ]] && PYBIN=$(which python3)

        echo "==== gdb -batch: thread apply all bt 50 ====" > /output/bt.txt
        "$GDB" -batch -nx \
            -ex "set print thread-events off" \
            -ex "set logging redirect on" \
            -ex "set logging file /output/bt.txt" \
            -ex "set logging enabled on" \
            -ex "set pagination off" \
            -ex "set print frame-arguments scalars" \
            -ex "set print address on" \
            -ex "info threads" \
            -ex "echo \n==== full backtrace, all threads ====\n" \
            -ex "thread apply all bt 50" \
            -ex "echo \n==== shared libraries (deep_ep / rocshmem / primus) ====\n" \
            -ex "info shared deep_ep" \
            -ex "info shared rocshmem" \
            -ex "info shared primus" \
            -ex "echo \n==== signal info ====\n" \
            -ex "info signals" \
            -ex "echo \n==== current frame ====\n" \
            -ex "bt 30" \
            -ex "set logging enabled off" \
            -ex "quit" \
            "$PYBIN" "/cores/$CORE_BASENAME" 2>&1 | tee -a /output/bt.txt || true
        chmod a+r /output/bt.txt
    '
```

Submit:

```bash
sbatch -N 1 -p amd-arad -t 01:00:00 -J gdb-23124-bt \
       -o $OUT_DIR/slurm-%j.out  utils/debug_bt_23124.sh
```

### 4.6 Why each gdb option

| Option | Reason |
|---|---|
| `-batch` | run-to-completion; no interactive prompt — required for SLURM batch. |
| `-nx` | ignore `~/.gdbinit` so user/system rc files can't change behavior. |
| `set logging redirect on` + `set logging file /output/bt.txt` + `set logging enabled on` | tee gdb output to a file in the writable mount; lets us capture even if stdout is lost. |
| `set pagination off` | prevent gdb from blocking for "more?" on multi-thousand-thread output. |
| `set print thread-events off` | suppresses the noise from thread-creation events when listing 1230 threads. |
| `set print frame-arguments scalars` | print scalar args (ints/longs) but skip large struct dereferences — keeps output bounded. |
| `info threads` | enumerate all LWPs (rank had ~1230 threads). |
| `thread apply all bt 50` | 50-frame backtrace per thread; 50 is enough to get from `_start` through Python into the C++ crash site without runaway expansion of recursive frames. |
| `info shared <pat>` | list loaded shared objects matching `deep_ep` / `rocshmem` / `primus` so we can confirm symbols are resolved. |
| `bt 30` (current frame) | the canonical "what crashed" frame, without the all-threads noise. |

### 4.7 Failures and fixes during the gdb run itself

**Attempt 1 (job 23139):** the script blindly ran `apt-get install -y gdb`. Compute nodes can resolve repo.radeon.com but cannot reach `archive.ubuntu.com` (no internet). The install failed; gdb was never available; the script exited.

**Fix:** scan the image for a pre-installed gdb. ROCm ships **`/opt/rocm/bin/rocgdb`** (it's "GNU gdb (rocm-rel-7.1-38) 16.3" with AMDGPU extensions, but it's a perfectly normal gdb for plain CPU cores). The script now tries `rocgdb` first, then `/opt/rocm/lib/llvm/bin/gdb`, then `gdb` on PATH.

**Attempt 2 (job 23140):** completed in 22 s with a 1.2 MB `bt.txt`. The job finishing in 22 s is fine — gdb only had to read ELF section headers and a few thread-stack regions of the core, not the heap.

### 4.8 Step 4: interpreting `bt.txt`

The diagnostic banner first:

```
Core was generated by `/opt/venv/bin/python3 -u .../mfu_tracker.py ... attention=dot_product'.
Program terminated with signal SIGABRT, Aborted.
[Current thread is 1 (Thread 0x7f50e2dae140 (LWP 320))]
```

`SIGABRT`, **not** `SIGSEGV` — already a strong signal. `SIGABRT` means the program voluntarily called `abort()` (assertion failure, `__cxa_throw` of an unhandled exception, or a hand-rolled `ERROR()` macro). This **rules out** kernel-side memory-access faults and most HIP/GPU bugs.

The "current frame" output is decisive:

```
#0  syscall ()                                               libc.so.6
#1  SignalHandler(int, siginfo_t*, void*)                    xla_rocm_plugin.so
#2  <signal handler called>
#3  pthread_kill ()                                          libc.so.6
#4  raise ()                                                 libc.so.6
#5  abort ()                                                 libc.so.6                  <-- abort
#6  rocshmem::Socket::connect(long)                          libprimus_turbo_kernels.so <-- caller of abort
#7  rocshmem::TcpBootstrap::Impl::establishConnections(long)
#8  rocshmem::TcpBootstrap::Impl::initialize(uniqueid, long)
#9  rocshmem::TcpBootstrap::initialize(uniqueid, long)
#10 rocshmem::rocshmem_init_attr(...)
#11 primus_turbo::deep_ep::internode::init(uniqueid, num_ranks, rank, low_latency)
#12 primus_turbo::jax::deep_ep::Buffer::SyncFromIPCHandles  csrc/jax/deep_ep/deep_ep.cpp:941
#13..17 pybind11 dispatcher
#18..30 Python interpreter -> _start
```

Reading the stack top-down:

- The signal was delivered (frame 1: XLA's `SignalHandler` printed/handled it before the dump).
- We re-entered libc through `pthread_kill → raise → abort` (frames 3–5). This is the **textbook signature of a `abort()`-via-assertion-style failure**.
- Frame **6** is the caller of `abort()`: `rocshmem::Socket::connect`. So whatever logic that function uses to escalate "I can't connect" calls `abort()`.
- Frames 7–11 are the rocSHMEM bootstrap chain.
- Frame 12 is the Primus-Turbo entry point in `deep_ep.cpp:941`. We then have line numbers.

### 4.9 Step 5: source corroboration

`/home/liyingli/rocSHMEM/src/bootstrap/`:

- **`utils.hpp:34`**:
  ```cpp
  #define ERROR(...) { fprintf(stderr, __VA_ARGS__); abort(); }
  ```
  Confirms how rocSHMEM converts string errors into `SIGABRT`.

- **`socket.cpp:474..511` — `Socket::connect(int64_t timeout)`**:
  ```cpp
  state_ = SocketStateConnecting;
  do {
    progressState();
    if (timeout > 0 && timer.elapsed() > timeout) {
      ERROR("connect timeout\n");        // <-- the abort site we hit
      return;
    }
  } while (...);
  ```
  Confirms the connect path aborts via `ERROR("connect timeout\n")`.

- **`utils.cpp:239`**: `Timer::elapsed()` returns **microseconds**.

- **`bootstrap.cpp:467..480` — `TcpBootstrap::Impl::establishConnections(timeoutSec)`**:
  ```cpp
  const int64_t connectionTimeoutUs = timeoutSec * 1000000;
  ```
  Confirms that the "5 second" we'll see below is converted to 5 000 000 µs and passed down to `Socket::connect`.

- **`envvar.hpp:139`** and **`envvar.cpp:55`**:
  ```cpp
  template <> inline constexpr const char* prefix<tag::BOOTSTRAP> = "ROCSHMEM_BOOTSTRAP";
  ...
  const var<int64_t> timeout("TIMEOUT", "", 5);
  ```
  → **the env var is `ROCSHMEM_BOOTSTRAP_TIMEOUT`, default `5` (seconds).** This is the lever to bump.

- **`socket.cpp:331..350` — `FindInterfaces`**: when `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` is unset, the iface is auto-picked from the first non-`docker`/non-`lo` device. With multiple network interfaces present, this can pick a route that is unreachable from peers.

### 4.10 Step 6: log corroboration

Cross-checking the 23124 stdout:

```
$ rg -n 'connect timeout|rocshmem|TcpBootstrap|IFNAME' \
     outputs/23124-*.log | sed -n '1,30p'

  ...
   871: 0: NCCL INFO useocpm2m-097-080: NCCL_SOCKET_IFNAME=eth0
  19699: 3: connect timeout
  19700: 2: connect timeout
  19701: 1: connect timeout
```

Lines 19699–19701 are **exactly** the literal string from `ERROR("connect timeout\n")`, emitted by every non-root rdma-rank participating in the **NVL slot 5** rocSHMEM communicator. Three independent ranks → three writes to the shared stdout → three identical lines. Node 0 (the rdma-root for slot 5) doesn't print "connect timeout" because it's the listener side.

Time-stamp shows the failure window is `09:15:22 → 09:15:31` — a ~9-second gap that exactly fits "5-second `Socket::connect` budget + a few seconds of teardown".

### 4.11 Why slot 5 specifically

Each NVL slot (0..7, since `NUM_MAX_NVL_PEERS=8`) builds its own rocSHMEM communicator with its own TCP rendezvous. Slots 0,1,2,3,4,6,7 finish their rendezvous within 5 s; slot 5 doesn't. Two leading hypotheses:

- **Tight timeout race.** Eight rendezvous in parallel; the 5th one happens to lose. Bumping `ROCSHMEM_BOOTSTRAP_TIMEOUT=60` makes it irrelevant.
- **Wrong auto-picked interface.** With `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` unset, rocSHMEM auto-picks; if the chosen iface for slot 5's listener is unreachable from peers, no timeout will help. Pinning to the same `eth0` NCCL uses (`NCCL_SOCKET_IFNAME=eth0` in the same log) eliminates this.

The fix run sets **both**, so the failure mode tells us which one was the cause:
- still aborts with "connect timeout" but later than 5 s → it's the auto-pick / unreachable iface;
- runs cleanly → at least one of the two was the cause and the bug is unblocked.

### 4.12 Reusable artifacts

- `utils/debug_bt_23124.sh` — generic enough to be repointed at any future coredump by editing the `CORE_DIR` / `CORE_FILE` glob. Useful checklist for the next coredump:
  1. Same image as the crashing run.
  2. Container as root (`--user 0 --privileged`).
  3. `rocgdb` (or any pre-installed gdb) inside the image — never depend on apt.
  4. `gdb -batch` with `info threads`, `thread apply all bt`, `info shared` and `info signals`.
  5. Output to a host-writable mount.

---

## 5. Root cause statement

The DeepEP-internode crash on every "NVL-slot-5 of non-root RDMA peer" rank in job 23124 was caused by **rocSHMEM's TCP bootstrap `Socket::connect` aborting via `ERROR("connect timeout\n")` after the default 5-second `ROCSHMEM_BOOTSTRAP_TIMEOUT` budget elapsed.** The abort propagates up through `TcpBootstrap::initialize → rocshmem_init_attr → primus_turbo::deep_ep::internode::init → Buffer::SyncFromIPCHandles` and terminates the process via `SIGABRT`.

This is corroborated by:
1. gdb backtrace from the actual coredump (frame 6 = `rocshmem::Socket::connect`).
2. The literal "connect timeout" message in the job's stdout (3 ranks).
3. Source: `socket.cpp:501` is the only `connect timeout` printer in rocSHMEM.
4. Source: default `ROCSHMEM_BOOTSTRAP_TIMEOUT = 5 (seconds)` in `envvar.cpp:55`.

It is **not**:
- A `cudnn_flash_te` / fused-attention bug (that one was 23122 and was fixed by `attention=dot_product`; 23124 also used `dot_product`).
- A `turbo_grouped_gemm` bug (23125 succeeded with DeepEP off but turbo grouped GEMM on).
- A Shardy-vs-GSPMD partitioner issue (`shardy=False` was used throughout).
- A node-specific HW fault (deterministic across runs and node sets within amd-rccl).

---

## 6. Fix candidates (under verification by 23153)

| Env var | Value | Why |
|---|---|---|
| `ROCSHMEM_BOOTSTRAP_TIMEOUT` | `60` | 12× the default, eliminates the tight 5 s race. |
| `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` | `eth0` | Same iface NCCL has been using successfully on these nodes (per the 23124 log); avoids any auto-pick mismatch. |
| `ROCSHMEM_DEBUG_LEVEL` | `INFO` | If anything still fails, we get rocSHMEM bootstrap diagnostics (chosen iface, IP, listen port, peer addresses). |
| `ONE_GPU_PER_PROCESS` | `true` (existing) | Required so `auto_detect_mode()` selects `per_process` DeepEP mode. |
| `attention` | `dot_product` (existing) | Workaround for the unrelated TE fused-attention crash from 23122. |

---

## 7. Lessons / gotchas

1. **amd-arad cannot host this workload.** Job 23150 confirmed: container saw zero AMD GPUs there (driver/HW class mismatch with our ROCm 7.1.1 image), JAX silently fell back to **CPU backend** (`device_kind: cpu` in `device_info.json`), 32 "CPU devices" routed cross-host ops through **Gloo**, which in turn timed out at 30 s on amd-arad's TCP layout. amd-arad is fine for **gdb-on-coredump** (pure file I/O) but not for any GPU/network repro.
2. **Compute nodes have no internet.** Any in-container debug must rely on what the image already has. ROCm images provide `rocgdb`; use it.
3. **`--privileged --user 0` containers can read host `0600 root:root` files** — the cleanest way to gdb root-owned cores when you have no host `sudo`.
4. **Length budget for `JOB_NAME`.** Each `_env_KEY=VALUE` becomes part of `EXP_TAG`, and the SLURM `--output` filename has a 255-byte segment cap. Plan envs with this in mind, or compress with the `model:alias` spec.
5. **A "successful" diagnostic is the cheapest signal of all.** 23125 (DeepEP-off) finishing cleanly was what proved the bug lives strictly in DeepEP internode and removed several false leads (Shardy, turbo grouped GEMM, partitioner).

---

## 8. Open items

- [ ] Verify 23153 reaches the first DeepEP dispatch and proceeds to `step=0` / step counter > 0. Pending backfill on amd-rccl.
- [ ] Determine which of the two knobs was actually the cause:
  - if 23153 succeeds, run a 4N ablation with **only** `ROCSHMEM_BOOTSTRAP_TIMEOUT=60` (no IFNAME) — if it still works, the cause was the tight default; if it fails again with "connect timeout" or similar, IFNAME was the cause.
- [ ] If the fix holds, propose a small Primus-Turbo / launcher PR that sets `ROCSHMEM_BOOTSTRAP_TIMEOUT` and `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` from the same `NCCL_SOCKET_IFNAME` the user already configures, so other DeepEP-internode users don't hit the same wall.

---

## Appendix A — Reproducing the gdb run

```bash
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm
mkdir -p outputs/23124-debug-bt && chmod a+w outputs/23124-debug-bt
sbatch -N 1 -p amd-arad -t 01:00:00 -J gdb-23124-bt \
       -o outputs/23124-debug-bt/slurm-%j.out \
       utils/debug_bt_23124.sh

# After it finishes:
less outputs/23124-debug-bt/bt.txt
```

The script is partition-agnostic: works on amd-arad (idle, fast) or amd-rccl. Only file I/O and gdb are needed; no GPUs are touched.

## Appendix B — Re-running the verification (or the next iteration)

```bash
DOCKER_IMAGE=/home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar \
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:fix \
    -N 4 -p amd-rccl -t 00:30:00 -- \
    _env_ONE_GPU_PER_PROCESS=true \
    attention=dot_product \
    _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
    _env_ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=eth0 \
    _env_ROCSHMEM_DEBUG_LEVEL=INFO
```
