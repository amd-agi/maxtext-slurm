---
name: job-log-triage
description: "Triage MaxText training jobs from log files — classify status (failed, hanging, running, completed), identify failure root cause from log signatures, project training progress, and recommend targeted next steps. Use when the user asks why a job failed, wants to diagnose an error, sees a crash, hang, timeout, OOM, NCCL error, heartbeat timeout, wants to understand a job's status, or asks about bad/low/dropping TGS or throughput."
---

# Job Log Triage

Classify a job's status and failure mode from its log file and recommend targeted next steps. Works on any job — `RAY=0` or `RAY=1`, finished or running, Slurm or local.

## Workflow

1. **Locate the log file and job directory.** The user may provide a log file, a job directory, or a Slurm job ID. Always resolve both:
   - **Given a job directory** → follow the `log` symlink inside it to find the log file.
   - **Given a `.log` file** → the job directory is the sibling directory with the same name minus `.log` (e.g., `outputs/7877-FOO.log` → `outputs/7877-FOO/`).
   - **Given a Slurm ID** → look for `outputs/<id>-*` (directory) or `outputs/<id>-*.log` (log file).
   - **Given a k8s job ID** (e.g., `k8s-20260310-080957-475e`) → same pattern: `outputs/<id>-*`.

   Having the job directory gives access to `ray_logs/`, `prometheus/`, xplane profiles, and other per-job artifacts needed for deeper diagnosis.

   **Directory layout:** The `outputs/` folder contains:
   - **Job directories** — always have a `log` symlink (pointing to `../<dirname>.log`). Named `<slurm_id>-<config>`, `k8s_<timestamp>-<config>`, or `local_<timestamp>-<config>`.
   - **Log files** — `<dirname>.log` files, siblings of their job directories.
   - **Per-rank logs (k8s only)** — `rank-N.log` files inside the job directory. The primary `.log` file contains only rank 0's output. For node-specific failures on k8s jobs, check `outputs/<id>-<name>/rank-N.log` for the failing rank's full output. These per-rank logs are NOT present in Slurm jobs (Slurm puts all ranks in one file via `srun -l`).
   - **Shared checkpoint directories** — hold checkpoint files and TensorBoard data, shared across runs. **No `log` symlink.** Created when `enable_checkpointing=true`.

   When triaging all jobs in `outputs/`, skip directories that have no `log` symlink — they are shared checkpoint dirs, not jobs.

2. **Read the tail of the log** (last 200 lines) — this is where the JOB SUMMARY and final errors appear. Then read the head (first 80 lines) for the header block (env vars, stage timeouts, node list).

3. **Determine job status** using two signals: the JOB SUMMARY block (if present) and **training step progress** (not log mtime — see warning below).

   | Log pattern | Status |
   |-------------|--------|
   | `JOB SUMMARY` + `Status: SUCCESS (exit 0)` | completed |
   | `JOB SUMMARY` + `Status: FAILED (exit N)` (N not 130/143) | failed |
   | `JOB SUMMARY` + exit 130 or 143 | cancelled — but always check for a preceding hang or failure |
   | No `JOB SUMMARY` + training steps actively advancing | running |
   | No `JOB SUMMARY` + training steps stopped + **job still alive** (Slurm RUNNING) | **hanging** (see hang diagnosis below) |
   | No `JOB SUMMARY` + training steps stopped + job ended (or Slurm state unavailable) | unknown-death (SIGKILL / OOM-kill / preemption) |

   **Do not rely on log mtime to detect hangs or determine if a job is running.** A hung job can produce non-training output (Ray buffered C++ messages, system warnings, topology logs) that updates the file mtime without advancing training. The reliable indicator is whether the **last `completed step:` line** is recent. Use the training progress projection (step 5) to compare the last step against where training should be.

   **`RAY=1` Slurm log truncation.** For `RAY=1` jobs, the Slurm log may show **fewer training steps than actually completed**. Ray actors write output to internal buffers that are forwarded asynchronously to the driver's stdout (which becomes the Slurm log). When the job finishes, remaining buffered output may not flush before the process exits. **Always cross-check the Slurm log's last step against `ray_logs/<head_node>/worker*.out`** — these files are written directly by the actor and contain the authoritative training progress. A job that appears to have stopped at step 33 in the Slurm log may have actually completed all 100 steps per the worker log. Failure to check this can cause misclassification (e.g., labeling a completed job as "unknown-death").

   To distinguish a hang from a death when there is no JOB SUMMARY: check Slurm job state (`scontrol show job <id>`) if the Slurm ID is known. If the job is still RUNNING, it's a hang. If the job has ended, it's an unknown-death.

4. **Classify the failure** by scanning the log for signatures in the table below. Scan bottom-up — the most diagnostic error is usually near the end.

5. **Project training progress.** Steps are **0-indexed**: `completed step: N` means step N is done, and `steps=T` in config means the job runs steps 0 through T-1 (T steps total). A job is complete when `last_step == T - 1`.

   Parse from the log (and from `ray_logs` for `RAY=1` jobs — the Slurm log may be truncated; see the `RAY=1` truncation warning in step 3):
   - **Step time:** extract the `seconds:` field from recent `completed step:` lines (use the steady-state average, skip warmup steps 0–4 relative to the first step).
   - **Total steps:** from `steps=N` in `PASSTHROUGH_ARGS` (log header). The final step number will be N-1.
   - **Checkpoint period:** from `Config param checkpoint_period: N` lines in the log (printed by MaxText during config dump). If `enable_checkpointing=true` is in `PASSTHROUGH_ARGS` but no explicit period, the default is 200.
   - **First completed step:** the first `completed step: N` in the log. If N > 0, the job restored from a checkpoint at step N-1 (restore skips the checkpoint step and starts training at N). Report this as "restored from checkpoint step N-1".
   - **Confirming restore vs fresh start:** For `RAY=1` jobs, check `ray_logs/*/worker*.out` for:
     - `No existing checkpoints found, not restoring checkpoint.` → fresh start
     - `restoring from this run's directory step N` → restored from checkpoint step N
   - A fresh start with `enable_checkpointing=true` saves an initial checkpoint at step 0 before training, so `first_step=0`. A restore from that checkpoint produces `first_step=1`.
   - **Last completed step** and its approximate wall-clock time.
   - **Steps completed this run:** `last_step - first_step + 1`. The total progress including prior runs is `last_step + 1`.

   Then compute:
   - **Expected step now:** `last_step + (now - last_step_time) / step_time`. If this is significantly ahead of the last logged step, the job is stalled. This is the primary hang detection signal.
   - **Total training time vs wall time:** Sum all `seconds:` values from every `completed step:` line to get total training time. **Actually compute the sum** — do not estimate by multiplying average step time by step count, as this hides checkpoint overhead variance and can mask a 40-60 minute hang inside "expected overhead." For `RAY=1` jobs, sum from a single worker log (authoritative); for `RAY=0`, filter to one task ID to avoid double-counting. Compare against the job's wall time (from the JOB SUMMARY). A significant gap (wall time >> total training time + expected setup/compilation overhead of ~15-30 min) signals unaccounted time that warrants investigation. Common causes: RCCL hang (all GPUs idle/spinning before the job was killed), slow checkpoint writes beyond what is included in step times, data loading stalls, XLA recompilation pauses, or delay between the last step and job termination due to any failure (OOM, NCCL timeout, etc.). Do not assume the gap is a hang — cross-reference with the failure classification (step 4) and, for `RAY=1` jobs, the TSDB to determine what happened during the gap. This check is especially useful for finished jobs where "expected step now" is not applicable.
   - **Progress lost on failure:** `last_step - last_checkpoint_step` steps of unrecoverable work. The last checkpoint step is the highest multiple of `checkpoint_period` that is <= `last_step`. For runs that never reached `checkpoint_period`, all training steps are lost (the initial step-0 checkpoint is the starting state, not a training milestone).
   - **Estimated time remaining:** `(total_steps - 1 - last_step) * step_time` (steps remaining until the final step T-1).
   - **Last periodic checkpoint saved by this run:** the highest multiple of `checkpoint_period` reached by `last_step`. If `last_step < checkpoint_period`, this run saved no periodic checkpoint — report "none". Do not count the initial step-0 checkpoint as a periodic checkpoint; it is just the starting state for fresh runs.

   Include these projections in the report — they make stalls obvious (expected step 2000 but last step is 316 = hung for hours) and quantify the cost of the failure.

   **TGS trend check (proactive).** When extracting step times, also check for TGS degradation — even if the user didn't ask about it. Compare the TGS of the last 10 steps against the steady-state average (steps 5-15). If the recent TGS is >10% below the early average, flag it as `tgs-degradation` in the report's "Additional findings" section, note the magnitude and step range of the drop, and include the TGS degradation next-step template in the recommendations. A job can "succeed" (complete all steps, exit 0) while having run 20-30% slower than it should have — catching this proactively saves significant GPU-hours on subsequent runs.

6. **For `RAY=1` jobs:** Search the log for the SSH tunnel command (look for `ssh -L` near the start of training). Extract the head node hostname and Prometheus port from the tunnel command (e.g., `ssh -L ...:HOST:PORT`; the port defaults to 9190 but may differ). Include the tunnel command, hostname, and port in the report.
   - **Job still live** (running or hanging): the live Prometheus is at `http://<head_host>:<port>` — query it directly from the head node's network. The port defaults to 9190 but may auto-increment if occupied; check the log for the actual port (look for `[Prometheus] Started on port` or the SSH tunnel command). Do **not** use `localhost:9090`, which may be a different Prometheus (e.g., cluster-level monitor). The SSH tunnel command in the log is for the **user's laptop** to reach the head node through a jump host — it binds ports on the user's local machine, not on the head node. Ask the user if they want you to set up port forwarding to access the Ray Dashboard (8265), TensorBoard (6006), and Prometheus on their local machine.
   - **Job already ended** (completed, failed, or cancelled): live dashboards are gone — do not attempt to query them. For post-hoc analysis, use `utils/prometheus.sh view <job_dir>/prometheus` to start a read-only Prometheus against the persisted TSDB. If you gathered evidence from live queries earlier in the conversation, include those results.

7. **Report findings** in the structured format described in "Output format" below.

## Failure classification table

Scan the log for these signatures, in priority order (first match wins the primary classification, but report all matches found).

### Infrastructure failures (before training starts)

| Class | Log signature(s) | Stage | What happened |
|-------|------------------|-------|---------------|
| **prolog-kill-no-log** | Empty `outputs/<id>-…/` dir with no `.log` sibling, OR user reports a `[ERROR] Job <id> died in prolog before writing any log` message from `submit.sh`, OR `sacct`/`squeue -t all` shows `FAILED` with `Reason=RaisedSignal:53(Real-time_signal_19)` and `RunTime=00:00:01`. Entry point is **not** the log file — scan `squeue -t all` / `sacct` instead, since the usual `outputs/` walk finds nothing. | Slurm prolog | slurmd killed the job before the batch script could run — usually because `--output` path exceeds ext4's 255-byte per-path-segment limit (long `JOB_NAME`), or a partition-level prolog script failed. `submit.sh` catches most of these pre-submit (length check in `parse_job_args.sh`) and the rest at t+3s (`squeue -t all` poll + cleanup). If this signature still appears on a fresh submit, the cause is partition-side — wait for the partition to recover, check slurmd logs on allocated nodes, or try a different partition. |
| **container-pull-fail** | `[ERROR] Pull failed for`, `[ERROR] Authenticated pull failed`, `[ERROR] Login to ... failed` | Docker pull | Image pull or registry auth failed |
| **container-load-fail** | `[ERROR] Unable to determine image name or ID from docker load output` | Docker pull | Tarball load failed |
| **no-gpu** | `WARNING: No GPU devices detected` | Container start | No GPU devices visible |
| **nccl-nic-fail** | `NCCL FATAL ... Failed to auto-detect NCCL_SOCKET_IFNAME; ABORTING` | Container start | Multi-node: no suitable NIC for NCCL |
| **port-fail** | `FATAL: Could not find a free port for JAX coordinator`, `FATAL: Could not find a free port for Ray` | Job start | Port allocation failed |
| **model-not-found** | `!!! Unknown model:`, `!!! Model name resolution failed` | Training start | Config file missing or ambiguous name |
| **patch-branch-fail** | `[FAIL] Failed to check out` | Container start | MaxText hotfix branch checkout failed |
| **ray-start-fail** | `[Ray] HEAD failed to start`, `[Ray] HEAD timeout`, `[Ray] WORKER failed` | Ray init | Ray cluster bootstrap failed (falls back to non-Ray) |

### Stage timeouts

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **preflight-timeout** | `== Preflight TIMEOUT` | Preflight checks hung (stale GPU processes, NFS, NUMA) |
| **pull-timeout** | `== Docker pull TIMEOUT` | Image pull took too long (slow registry or large image) |
| **ecc-timeout** | `== ECC check TIMEOUT` | ECC memory check hung (GPU driver issue) |
| **train-timeout** | `== Training TIMEOUT` | Training exceeded the configured wall-clock limit |

### Training failures (during training)

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **hang** | Training steps stopped advancing (last `completed step:` far behind expected), job still RUNNING in Slurm, no error before the stall | Collective communication deadlock (NCCL/RCCL all-reduce/all-gather hang) — all nodes waiting on each other |
| **heartbeat-timeout** | `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats)`, `The tasks have crashed` | JAX coordination heartbeat timeout — **known bug** with documented root cause (see diagnosis below) |
| **oom-host** | `Killed` (from OOM killer), `oom-kill`, `Out of memory` | Host OOM: process killed by Linux OOM killer |
| **oom-gpu** | `OUT_OF_MEMORY`, `XLA_ERROR`, `ResourceExhausted`, `RESOURCE_EXHAUSTED`, `out of memory` | GPU VRAM exhausted during compilation or execution |
| **nccl-timeout** | `NCCL WARN Timeout`, `NCCL error`, `NCCL WARN` (during training), `Timeout waiting for`, `ncclSystemError` | NCCL/RCCL collective timeout — network or GPU issue |
| **xla-compile-fail** | `INTERNAL: Failed to compile`, `XLA compilation failed`, `HloModule` + `error` | XLA/GPU compiler failure |
| **python-exception** | `Traceback (most recent call last):` | Unhandled Python exception (read the traceback for details) |
| **signal-kill** | `Training subprocess killed by` | Training process killed by signal (SIGSEGV, SIGABRT, etc.) |
| **subprocess-fail** | `Training subprocess exited with code` | Training process exited non-zero (read preceding output) |
| **actor-fail** | `Actor failed:` | Ray actor exception (includes traceback) |
| **checkpoint-fs-error** | `Training stopped: Checkpointing failed`, `[Errno 2] No such file or directory: 'manifest.ocdbt.__lock'` | Checkpoint write failed due to NFS/storage filesystem error — **but the checkpoint may be intact** (see checkpoint filesystem error diagnosis below) |

### Training performance issues (job running but underperforming)

| Class | Detection method | What happened |
|-------|-----------------|---------------|
| **tgs-degradation** | TGS drops >10% below early steady-state average and stays low, or TGS steadily declines over time. Detected from `completed step:` lines in worker logs — not a log error signature. | Network (RDMA retransmit), resource contention, or hardware degradation slowing collective communication. The job runs without errors but significantly underperforms. |

### Job-level status

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **cancelled** | `CANCELLED (scancel / SIGTERM)`, exit 130 or 143 | User or scheduler cancelled the job — **but always check for a preceding hang or failure** (see below) |
| **node-fail** | `NODE_EXIT host=... exit=` (non-zero) | One or more nodes exited with errors |
| **unknown-death** | No JOB SUMMARY, training steps stopped, job no longer in Slurm RUNNING state | Process killed externally (SIGKILL, OOM-kill, preemption) with no chance to write summary |
| **stage-fail** | `== ... FAILED (exit=` | A non-timeout stage failure (check exit code) |

## Detailed Diagnosis Guides

For in-depth diagnosis procedures for specific failure modes, see [references/diagnosis-guides.md](references/diagnosis-guides.md):

| Guide | When to use |
|-------|-------------|
| Hang diagnosis | Job RUNNING but training stalled |
| Heartbeat timeout | "tasks are unhealthy" — known JAX bug, usually false positive |
| GPU OOM | `RESOURCE_EXHAUSTED` — check `XLA_PYTHON_CLIENT_MEM_FRACTION` first |
| Checkpoint filesystem error | `manifest.ocdbt.__lock` — checkpoint may be intact |
| TGS degradation | TGS drops >10% — constant, phased, gradual, or one-time patterns |
| Unknown-death | No JOB SUMMARY, job gone — check dmesg for OOM-kill |
| Node failures | `NODE_EXIT` with non-zero exit codes |

## Output format

Report findings in this structure:

```
## Job triage: <log_file_path>

**Status:** <completed | failed | cancelled | running | hanging | unknown>
**Primary failure:** <class name from table above>
**Stage:** <which stage failed, if identifiable>

### What happened
<1–3 sentence plain-English explanation>

### Evidence
<Relevant log lines, quoted verbatim with line context>

### Training progress projection
| Metric | Value |
|--------|-------|
| Start | fresh / restored from ckpt <N> |
| Steps completed (this run) | <last - first + 1> steps (step <first> through <last>) |
| Overall progress | <last+1> / <total> (<pct>%) |
| Steady-state step time | <X>s |
| Expected step by now | <N> (based on step time and run start) |
| Last periodic ckpt this run | step <N> (or "none — didn't reach checkpoint_period") |
| Progress lost | <N> steps (<time>) since last periodic ckpt (or "all — no periodic ckpt") |
| Estimated time remaining | <time> (from last step, if job were healthy) |

### Additional findings
<Any secondary signatures found (e.g., warnings, non-fatal errors)>

### Live dashboards (RAY=1 jobs only)
<If SSH tunnel command found in log, show it here>
<If job is still live (running or hanging), ask:
 "Want me to set up port forwarding so I can access the Ray Dashboard / Prometheus / TensorBoard?">

### Recommended next steps
<Numbered list of specific actions>
```

### Next-step templates by failure class

| Class | Recommended next steps |
|-------|----------------------|
| **container-pull-fail** | 1. Verify image name in `container_env.sh`. 2. Check registry credentials. 3. Try manual `docker pull`. |
| **no-gpu** | 1. Check that the node has GPUs (`rocm-smi` or `nvidia-smi`). 2. Check container device flags in `_container.sh`. |
| **nccl-nic-fail** | 1. Run `ip link show` on compute nodes to verify NIC availability. 2. Check `choose_nccl_socket_ifname.sh` logic. |
| **model-not-found** | 1. List available configs: `ls configs/*.gpu.yml`. 2. Create a new config per `docs/model-configs.md`. |
| **preflight-timeout** | 1. Check for stuck GPU processes: `rocm-smi` / `nvidia-smi`. 2. Increase timeout: `STAGE_TIMEOUTS="preflight:1800"`. |
| **pull-timeout** | 1. Check network/registry speed. 2. Pre-pull the image. 3. Increase timeout: `STAGE_TIMEOUTS="pull:1800"`. |
| **train-timeout** | 1. Increase timeout: `STAGE_TIMEOUTS="train:<seconds>"`. 2. Verify it's not a hang (check if training was progressing). |
| **hang** | 1. If `RAY=1` and job is still live: check Ray Dashboard (port 8265) for actor status and stack traces to confirm RCCL hang and identify the stuck collective. 2. Query Prometheus TSDB at hang time for GPU util (check per-GPU per-node for partial utilization patterns), power watts, TCP retransmits (both rate and absolute totals across nodes), RDMA counters. 3. Kill the job: `scancel <id>`. 4. **Retry on the same nodes first** — init-phase hangs (before step 0) are often transient RCCL race conditions. If the retry succeeds, no further action needed; note TCP retransmit outliers for awareness only. 5. Only if the hang **recurs**: exclude suspect nodes (`--exclude=<nodes>` targeting TCP retransmit or GPU pattern outliers), add `_env_NCCL_DEBUG=INFO`, and use `slurm_job_monitor.sh -j <id>` for early detection. |
| **heartbeat-timeout** | 1. Known bug — almost certainly a false positive (see `docs/jax-heartbeat-false-positive-postmortem.md`). 2. Increase `jax_distributed_heartbeat_timeout_seconds` to several hours (e.g., 14400). 3. Use `slurm_job_monitor.sh` for independent hang detection. 4. Follow the heartbeat diagnosis checklist above to confirm. |
| **oom-host** | 1. Reduce `per_device_batch_size`. 2. Enable `remat_policy=full`. 3. Check for checkpoint memory spike (DP replica #0 pattern). |
| **oom-gpu** | **First check `XLA_PYTHON_CLIENT_MEM_FRACTION`** — see GPU OOM diagnosis below. If the fraction is too low for the model size, increase it (e.g., `.85` → `.93`). Only after ruling that out: 1. Reduce `per_device_batch_size` or `max_target_length`. 2. Try `remat_policy=full`. 3. Check XLA buffer assignment for memory usage. |
| **nccl-timeout** | 1. Check network health (`ip link`, `ethtool`, RDMA counters). 2. Run with `_env_NCCL_DEBUG=INFO` for detailed NCCL logs. 3. Check if specific nodes are consistently failing. |
| **xla-compile-fail** | 1. Check XLA flags in `train_env.sh` for conflicting settings. 2. Try `_env_ENABLE_XLA_DUMP=1` to capture the failing HLO. 3. Reduce model complexity to isolate the issue. |
| **python-exception** | 1. Read the full traceback. 2. Check if it's a known MaxText issue. 3. Verify config parameters. |
| **signal-kill** | 1. Check for core dumps in the coredump path candidates: `<job_dir>/core*`, `<outputs_root>/core*`, and paths from `COREDUMP_EXTRA_DIRS` in `container_env.sh`. 2. Inspect with `gdb python3 <core_file>` inside the Docker container. 3. If logs show `NCCL_DMABUF_ENABLE=1`, check for the runtime warning `Forcing NCCL_DMABUF_ENABLE=0` (missing `/boot` kernel metadata safeguard). If that warning is absent on older commits, verify `/boot/*$(uname -r)*` availability in the container. 4. See `docs/debugging.md`. |
| **cancelled** | Cancellation is the mechanism, not the root cause. 1. Check training progress projection — if the last step is far behind the expected step, the real issue is a **hang** killed by scancel. 2. Check for preceding errors (NCCL, OOM, heartbeat). Report the underlying cause as primary failure. If no underlying issue, no action needed. |
| **node-fail** | 1. Identify which nodes failed. 2. Read their task output. 3. For exit 137: likely OOM. For exit 134/139: check core dumps. |
| **unknown-death** | 1. Check `dmesg` for OOM kills. 2. Check Slurm state: `scontrol show job <id>`. 3. If recurring: run with `RAY=1` for TSDB diagnostics. |
| **tgs-degradation** | 1. Extract TGS timeline from worker logs (see TGS degradation diagnosis above). 2. Identify the degradation pattern (constant drop, phased, gradual, periodic). 3. For `RAY=1` jobs: query TSDB — Playbook 6 (Network Health) for RDMA retransmits per host, Playbook 7 (Training Stability) contention checklist. **Constant drops point to RDMA issues; gradual increases point to resource leaks.** 4. Identify the offending nodes from per-host RDMA retransmit rates. 5. Resubmit with `--exclude=<bad_nodes>` targeting nodes with RDMA retry exhaustion or sustained RDMA retransmits (see node exclusion prioritization in TSDB skill). |
| **checkpoint-fs-error** | 1. **Check other workers' logs first** — the checkpoint may be intact despite the error (see checkpoint filesystem error diagnosis below). 2. If checkpoint is intact: resubmit restoring from that checkpoint; zero progress lost. 3. If checkpoint is corrupt: resubmit restoring from the previous periodic checkpoint. 4. For `RAY=1` jobs: query TSDB for I/O pressure (`hw_io_pressure_full_pct`) and TCP retransmits (`rate(hw_tcp_retransmits_total[5m])`) on checkpoint-writing nodes during the failure window. Compare against previous successful checkpoints to identify NFS degradation. 5. Check VAST/NFS storage health. |
| **ray-start-fail** | Non-critical — training falls back to non-Ray mode. If observability is needed: 1. Check port conflicts. 2. Check Ray logs in job dir. |

## Known-harmless log entries

Common patterns that appear in normal, healthy jobs — do not classify as failures. See [references/known-harmless.md](references/known-harmless.md) for the full list.

## Multi-failure jobs

Some failures cascade. When multiple signatures are found:

1. **Report all of them** in the "Additional findings" section.
2. **Identify the root cause** — the earliest error in the log is usually the primary failure. Later errors (heartbeat timeouts, node exits) are often consequences.
3. **Common cascades:**
   - OOM on one node → NCCL timeout on other nodes (waiting for the dead node) → heartbeat timeout
   - NCCL network error → all-reduce hang → training timeout or heartbeat timeout
   - One node dies silently → remaining nodes hang on the next collective (training steps stop, no error)
   - XLA compilation failure → Python exception → subprocess exit code 1
   - **RCCL hang (all nodes spinning in busy-wait) → gRPC channel deadlock on one task after extended hang (30+ min) → heartbeat timeout declares that task dead → all tasks killed.** The heartbeat timeout is the *kill mechanism*, not the root cause. The RCCL hang is the primary failure. This cascade is the most commonly misdiagnosed — the heartbeat error is prominent in the log (all 24 tasks report it), while the hang leaves no log signature (training simply stops advancing). Always check TSDB GPU power/utilization before the heartbeat error to detect this.
