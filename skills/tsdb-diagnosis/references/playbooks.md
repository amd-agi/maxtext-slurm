# Diagnostic Playbooks

## Diagnostic Playbooks

Each playbook is triggered by a specific scenario (from triage output or user request). Run the listed queries, apply the interpretation rules, and report findings.

### Correlation is not causation

When an incident occurs, many metrics move at once. A single underlying event — a network blip, a thermal excursion, an I/O stall — ripples through the system and causes correlated changes across GPU power, clocks, utilization, throughput, step time, and more. Most of what you see in the TSDB are **symptoms**, not root causes. Do not report a correlated symptom as the diagnosis.

**The diagnostic principle: find the earliest anomaly.** The root cause is the metric that deviates *first*. Everything that follows is a consequence. When you spot any anomaly:

1. **Note the timestamp** of the anomaly.
2. **Query all other metric families** at the same timestamp and in the minutes preceding it — both **training metrics** (`tb_*`) and **system metrics** (`hw_*`, `ray_*`).
3. **Find which metric deviated first.** That is the root cause candidate. Everything that moved after it is a correlated effect.
4. **Verify the causal direction.** The root cause should logically explain the downstream symptoms.

**Root causes can be in either domain — training or system.** The TSDB contains both training-level metrics (loss, grad norms, LR, MoE load balance loss, throughput) and system-level metrics (GPU power/clocks, network, I/O) on the same timeline. Do not assume the root cause is always a system event. Examples:

- **System → training:** A network retransmit burst stalls a collective → GPU power drops → step time spikes → throughput drops. System event is the root cause.
- **Training → system:** An MoE model's routing changes dramatically → `tb_learning_moe_lb_loss` spikes → some experts are overloaded while others are idle → GPU utilization becomes uneven → TGS drops. The mathematical event (routing shift) is the root cause; the GPU utilization change is a passive symptom.
- **Training → system:** A grad norm explosion → NaN values → recompilation or fallback → throughput collapse. The training instability is the root cause.
- **Training → system:** Learning rate warmup ends → larger weight updates → checkpoint sizes grow → I/O pressure during saves → periodic step time spikes. The LR schedule change is the root cause.

Always query both domains. When system metrics (power, clocks, utilization) change, check whether a training metric (`tb_learning_*`, `tb_perf_*`) shifted first — the system may be passively responding to a change in the model's computational behavior.

Always report the full chain: root cause → intermediate effects → observed symptoms. This is the core value of having all metrics in a single time-aligned TSDB.

### Trace suspicious findings to source code

Metrics tell you *what* is happening; source code tells you *why*. When the TSDB reveals an anomaly that can't be explained by external factors (network, hardware, thermal), trace the causal chain into the code to find the mechanism. This is especially valuable when the anomaly is persistent (not transient) and uniform across all nodes (not a single-node hardware issue) — these patterns point to a software-level root cause.

**When to investigate code:**
- A metric delta between two jobs with identical configs that isn't explained by hardware, network, or thermal differences (e.g., `hw_procs_running` is uniformly higher in one job).
- A persistent anomaly that starts at a specific event (checkpoint restore, profiler activation, config change) and never recovers.
- The TSDB points to a specific subsystem (checkpointing, data pipeline, collective communication) but the metric alone doesn't explain the mechanism.

**How to trace:**
1. **Start from the triggering event.** The triage report and metric timeline tell you *when* the anomaly starts. Identify what code executed at that point (e.g., checkpoint restore at step N, profiler hook at step M).
2. **Follow the call chain.** Read the relevant MaxText entry point (e.g., `checkpointing.py:load_state_if_possible` for restore issues), then trace into the libraries it calls (Orbax, JAX, XLA). Use semantic search and grep to find the code paths.
3. **Look for resource creation without cleanup.** The most common source of persistent anomalies is resources (communicators, threads, file handles, caches) that are created during a transient operation but never released. Check whether the code path creates anything that outlives the function call.
4. **Check for alternative code paths.** Libraries often have multiple implementations of the same operation (e.g., dispatcher-based vs legacy path in Orbax). If one path leaks resources, the other may not — this informs the fix direction.
5. **Verify with logs.** After forming a hypothesis from the code, confirm it in the job's logs (ray worker `.out` files, stderr). Look for initialization messages, warnings, or resource creation events that match the code path you identified.

**Accessible code locations:**
- MaxText source: `/workspace/maxtext/src/MaxText/` (checkpointing, training loop, configs)
- Orbax (checkpoint library): `/opt/venv/lib/python3.12/site-packages/orbax/checkpoint/`
- JAX: `/opt/venv/lib/python3.12/site-packages/jax/`
- XLA C++ headers (for understanding communicator/backend behavior): search under `/opt/venv/` or use `python3 -c "import jaxlib; print(jaxlib.__path__)"` to locate

**Example from real diagnosis:** TSDB showed `hw_procs_running` was ~17 higher per host in a restored job (with `enable_single_replica_ckpt_restoring=true`) vs a fresh-start baseline. Code tracing revealed: MaxText `checkpointing.py` → Orbax `SingleReplicaArrayHandler.deserialize()` → `multislice.broadcast_one_replica_to_all()` → `_merge_globalized_replicas()` which uses `jax.jit(jnp.sum)` → XLA all-reduce collective → new RCCL communicators with unique `GpuCliqueKey` → cached permanently in C++ clique cache → background polling threads increase `hw_procs_running`. Without code tracing, the TSDB alone could only say "more processes are running" — the code trace identified the exact mechanism and pointed to a fix.

### Checkpointing interference

Checkpointing is one of the most disruptive periodic events in training. During a checkpoint save, the system behavior changes dramatically — and many metrics that look anomalous are actually normal checkpoint effects. Always check whether an anomaly coincides with a checkpoint step before diagnosing it as a problem.

**What happens during a checkpoint save:**
- GPU utilization drops (GPUs idle while waiting for D2H transfer and I/O to complete)
- GPU power drops (idle GPUs draw less power)
- Host memory spikes (model parameters copied from GPU to host RAM, especially on DP replica #0 nodes)
- I/O pressure rises (`hw_io_pressure_full_pct`, `hw_mem_dirty_bytes`) as parameters are written to storage
- Step time spikes (the checkpoint step takes much longer than a normal training step)
- `hw_procs_blocked` increases (processes waiting on I/O)
- Network traffic may drop (no collectives during the checkpoint window)

**How to identify checkpoint steps:** Parse `checkpoint_period` from the log (from `Config param checkpoint_period: N`). Checkpoint saves occur at steps that are multiples of this period. If `async_checkpointing=true` is in `PASSTHROUGH_ARGS`, the save happens in the background and may overlap with subsequent training steps, spreading the I/O impact over multiple steps.

**Rules:**
- When analyzing steady-state performance, **exclude steps around checkpoint boundaries** (the checkpoint step and 1-2 steps after for async checkpointing).
- When diagnosing an anomaly, **always check whether the timestamp falls on a checkpoint step** before investigating further. A step time spike at step 200 with `checkpoint_period=200` is expected, not an incident.
- When comparing metrics across jobs with different `checkpoint_period` values, normalize by excluding checkpoint steps from both.
- Memory spikes during checkpoint saves are not OOM precursors unless they push the host to its limit. DP replica #0 nodes use ~2x model size in host RAM during saves — this is expected.

**Expected NFS congestion during checkpoint writes:**

Checkpoint-writing nodes (one per FSDP replica) write large model parameters to shared storage (VAST/NFS) simultaneously. This routinely causes:
- **TCP retransmit rates of 500–3000/s** on checkpoint-writing nodes — this is NFS retransmitting during heavy writes and is **normal, not alarming**. Do not flag TCP retransmits during checkpoint windows as a network issue unless they also appear outside checkpoint windows.
- **I/O pressure (`hw_io_pressure_full_pct`) up to 80%** — transient, lasts the duration of the checkpoint write (typically 5–8 minutes).
- **10–20+ blocked processes (`hw_procs_blocked`)** per checkpoint-writing node — processes waiting on NFS I/O.
- Non-checkpoint-writing nodes (remaining FSDP replicas) show near-zero I/O pressure and TCP retransmits during the same window.

**Diagnosing checkpoint filesystem errors (`checkpoint-fs-error` from triage):**

When a checkpoint write fails with an NFS-related error (e.g., `manifest.ocdbt.__lock` ENOENT), compare the I/O metrics at the failed checkpoint against previous successful checkpoints to assess whether NFS conditions were worse:

```bash
# For each checkpoint step, query peak I/O pressure and TCP retransmit rate:
for ts in <ckpt_2600_ts> <ckpt_2800_ts>; do
  end=$((ts+1800))
  curl -s "http://localhost:<port>/api/v1/query_range?query=hw_io_pressure_full_pct&start=$ts&end=$end&step=30s"
  curl -s "http://localhost:<port>/api/v1/query_range?query=rate(hw_tcp_retransmits_total%5B5m%5D)&start=$ts&end=$end&step=30s"
done
```

Compare peak values per node across checkpoints. A significant increase in TCP retransmits or I/O pressure at the failing checkpoint (vs. previous successes) points to NFS congestion as the trigger. Also check `hw_procs_blocked` — the head node (task 0) is especially vulnerable because it runs additional I/O-intensive services (Prometheus TSDB, Ray GCS, dashboard) alongside the checkpoint writer.

**Single-replica checkpoint restore and RCCL communicator leaks:**

When `enable_single_replica_ckpt_restoring=true` and the job restores from a checkpoint, the restore code path leaks RCCL communicators and their background polling threads. This manifests as a **persistent increase in `hw_procs_running`** (~17 extra runnable threads per host on a 24-node / 3-replica MoE run) that appears immediately after restore and never recovers. The leaked threads create constant CPU contention — competing with data pipeline threads, RCCL coordination, and the Python runtime — causing a persistent throughput (TGS) drop of ~1%.

**Root cause (code-level):** During single-replica restore, Orbax's `SingleReplicaArrayHandler` broadcasts the restored parameters from one replica to all others. The non-dispatcher code path in `orbax/.../multislice.py` (`_merge_globalized_replicas`) uses `jax.jit(lambda: jnp.sum(axis=0), out_shardings=...)` to perform this broadcast. The `jnp.sum` across the replica axis compiles to an **XLA all-reduce collective**, which causes RCCL to initialize new communicators via XLA's `AcquireGpuClique()`. Because the restore uses a different mesh/sharding configuration (single-replica subset mesh) than training, the resulting `GpuCliqueKey` is unique — so RCCL creates brand-new communicators instead of reusing the training ones. These communicators and their background polling threads are cached permanently in XLA's C++ GPU clique cache (`GpuCliqueKey → LockableGpuClique`).

**Why cleanup attempts fail:**
- `jax.clear_caches()` only clears Python-level JIT compilation caches. It does **not** touch the C++ communicator cache. Confirmed ineffective by testing.
- `jax.clear_backends()` would clear communicators but tears down the entire JAX runtime — unusable mid-training.
- There is no Python API to destroy individual RCCL communicators. The C++ clique cache has no eviction mechanism or public release API.
- MaxText's `_restore_original_array_handler()` re-registers the original `ArrayHandler` but cannot affect the C++ communicator state.

**Log-level confirmation:** The ray worker `.out` logs show an extra NCCL communicator initialization wave during restore. Look for `init.cc:2095 NCCL WARN MSCCL++` messages — a fresh-start job has 2 waves (startup + training), while a restored job has 3 waves (startup + **restore broadcast** + training). The extra wave corresponds to the `_merge_globalized_replicas` all-reduce creating new communicators.

**Verified fix:** Two patches on the `yihuang/fix-rccl-thread-leak-single-replica-restore` branch address this:
1. Replace Orbax's JIT-based broadcast with a direct RCCL broadcast via ctypes that explicitly destroys communicators after use (`ncclCommInitRank` / `ncclBroadcast` / `ncclCommDestroy` + `gc.collect()`). This eliminates the leaked threads and the ~1% TGS drop.
2. Replace Orbax's `jax.jit(create_zeros)` on non-primary hosts with `numpy.zeros` + `jax.device_put`, eliminating extra XLA compilations during restore that degrade steady-state performance.

**How to detect:** When comparing a restored job against a fresh-start baseline with identical config:
- Check `hw_procs_running` — a uniform delta across all hosts (not a spike, not on specific nodes) after restore is the signature.
- The delta is present from the first training step after restore and does not recover.
- Training-level metrics (loss, grad norms, LR) will be identical between the two jobs — this is purely system-level contention.
- A fresh start with the same config but no actual restore (even if `enable_single_replica_ckpt_restoring=true`) does not trigger the leak — the restore code path must actually execute.
- Check ray worker `.out` logs for the extra NCCL init wave (3 waves vs 2) as a definitive confirmation.
- If both jobs ran with `_env_ENABLE_XLA_DUMP=1`, compare the number of compiled modules in `xla_dump/` — a restored job may have extra `jit_create_zeros` entries from Orbax's non-dispatcher path. These extra XLA compilations are a secondary performance penalty beyond the leaked threads.

---

### Playbook 1: RCCL/NCCL Hang

**Trigger:** Triage reports `hang` or `cancelled` with underlying hang. User says "why did the job hang" or "confirm the RCCL hang."

**Goal:** Confirm the hang mechanism (RCCL busy-wait vs. dead node vs. network partition) and identify the trigger.

**Queries** (at the hang time window — from last successful training step to job end):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization` | Per-host GPU utilization |
| 2 | `hw_gpu_power_watts` | Per-GPU power draw |
| 3 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate per host |
| 4 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate per device |
| 5 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts per device |
| 6 | `hw_rdma_port_state` | RDMA port state (1=ACTIVE) |
| 7 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notification packets |

**Interpretation:**

| GPU util | Power | Network | Diagnosis |
|----------|-------|---------|-----------|
| High (variable per GPU) | Idle-level (~260–300W MI355X) | Clean | **Confirmed RCCL busy-wait hang.** GPUs in the RCCL polling loop report high utilization, but power is at idle/standby. **Important:** utilization may not be uniform — in partial deadlocks, only GPUs that entered the collective show high utilization (e.g., 2 of 8 GPUs at 100%, rest at 0%). Always check power as the ground truth. |
| ~100% all nodes | Idle-level | Retransmit spike before hang | **Network-triggered RCCL hang.** A transient network event caused a collective to stall, then all nodes entered busy-wait. |
| ~100% most nodes | Mixed (some idle, some active) | Clean | **Possible straggler.** One or more nodes may have fallen behind, causing others to wait. Check per-node power to identify the slow node. |
| 0% on some nodes | 0W on some nodes | N/A | **Node death.** Some nodes died — remaining nodes hung waiting for them. Check `hw_dmesg_gpu_errors_total` and `hw_gpu_ras_*` on the dead nodes. |
| ~100% all nodes | Active-level (~900W MI355X) | Clean | **Not a hang** — training was still running. Re-check the triage classification. |

**Key thresholds (MI355X):**
- Active training power: ~900W per GPU
- RCCL busy-wait power: ~300W per GPU (only ~42W above idle — nearly indistinguishable from standby)
- Idle/standby power: ~260W per GPU

---

### Playbook 2: Heartbeat False-Positive

**Trigger:** Triage reports `heartbeat-timeout`. User wants TSDB confirmation that the accused tasks were healthy.

**Goal:** Prove or disprove that the accused tasks were alive and training when the heartbeat mechanism killed them.

**Setup:** From the log, extract:
- Crash time (the heartbeat error timestamp)
- Heartbeat timeout value (`jax_distributed_heartbeat_timeout_seconds`)
- Accused task IDs and their host mappings (from `NNODES`, `NODE_LIST`, task-to-host mapping in log)
- Compute heartbeat-stop time: `crash_time - heartbeat_timeout`

**Queries** (at the heartbeat-stop time — this is when the heartbeats actually failed):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization{host=~"<accused_hosts>"}` | Were accused tasks training? |
| 2 | `ray_node_mem_used{host=~"<accused_hosts>"}` | Memory on accused hosts |
| 3 | `hw_io_pressure_full_pct{host=~"<accused_hosts>"}` | I/O pressure (checkpoint stall?) |
| 4 | `rate(hw_tcp_retransmits_total{host=~"<accused_hosts>"}[5m])` | Network health |
| 5 | `hw_tcp_listen_drops_total{host=~"<coordinator_host>"}` | gRPC coordinator overloaded? |
| 6 | `hw_mem_dirty_bytes{host=~"<accused_hosts>"}` | Dirty page pressure |

**Interpretation:**

| GPU util on accused | Network | I/O pressure | Diagnosis |
|---------------------|---------|--------------|-----------|
| High (active training) | Clean | Low | **Confirmed false positive.** Tasks were healthy. The heartbeat mechanism failed (known gRPC bug — see `docs/jax-heartbeat-false-positive-postmortem.md`). |
| High | Retransmit spike | Low | **Likely false positive** triggered by transient network issue blocking the heartbeat gRPC. Tasks were alive but heartbeat RPCs couldn't reach the coordinator. |
| High | Clean | High | **Likely false positive** triggered by checkpoint I/O pressure. Heavy writeback blocked the heartbeat thread. |
| Low/zero | Any | Any | **Possibly a true positive.** Tasks may have actually died. Check for OOM, GPU errors, or crashes on those hosts. |

**Report template for confirmed false positive:**
> The TSDB confirms this is a heartbeat false-positive kill. At the heartbeat-stop time (<time>), all accused hosts showed GPU utilization at <X>%, TCP retransmit rate near zero, and no I/O pressure. The tasks were alive and actively training when the heartbeat mechanism declared them dead. Root cause: shared gRPC channel bug documented in `docs/jax-heartbeat-false-positive-postmortem.md`.

---

### Playbook 3: Host OOM

**Trigger:** Triage reports `oom-host`. User wants to understand the memory trajectory.

**Goal:** Trace the memory growth pattern, identify what caused the OOM, and determine if it's reproducible.

**Queries** (over the full job duration, or last 30 minutes before the OOM):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_mem_used` | Per-host memory over time |
| 2 | `increase(hw_oom_kills_total[1h])` | OOM kill events |
| 3 | `hw_mem_dirty_bytes` | Dirty page accumulation |
| 4 | `hw_mem_writeback_bytes` | Writeback pressure |
| 5 | `hw_io_pressure_full_pct` | I/O stall percentage |
| 6 | `hw_procs_blocked` | Processes blocked on I/O |
| 7 | `hw_gpu_vram_used_bytes` | GPU VRAM (to correlate D2H transfers) |

**Interpretation:**
- **Gradual ramp** → memory leak (Python objects, data pipeline buffers, or XLA compilation cache growing unbounded).
- **Sudden spike** → checkpoint save (D2H copies all parameters to host RAM), data loading burst, or XLA recompilation allocating new buffers.
- **Periodic spikes that recover** → checkpoint saves. If the peak exceeds available RAM on one cycle, it's a checkpoint OOM.
- **Only DP replica #0 nodes OOM** → checkpoint save pattern. MaxText saves checkpoints from DP replica 0 only, which requires ~2x model size in host RAM (one copy in GPU VRAM, one on host for the save).
- **All nodes OOM simultaneously** → likely a data pipeline issue or XLA recompilation event.

---

### Playbook 4: Node Failure / Hardware Issue

**Trigger:** Triage reports `node-fail`, `signal-kill`, or `unknown-death`. User wants to identify hardware problems.

**Goal:** Identify hardware errors (ECC, PCIe, XGMI) or driver issues that caused the failure.

**Queries** (over full job duration — hardware errors can accumulate before causing a crash):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `increase(hw_gpu_ras_umc_ue_total[1h])` | HBM uncorrectable ECC errors (fatal) |
| 2 | `increase(hw_gpu_ras_umc_ce_total[1h])` | HBM correctable ECC errors (accumulating = concern) |
| 3 | `increase(hw_gpu_ras_xgmi_ue_total[1h])` | XGMI/WAFL link uncorrectable errors |
| 4 | `increase(hw_gpu_ras_xgmi_ce_total[1h])` | XGMI/WAFL link correctable errors |
| 5 | `increase(hw_gpu_ras_gfx_ue_total[1h])` | Compute engine errors |
| 6 | `increase(hw_gpu_pcie_fatal_total[1h])` | PCIe fatal errors |
| 7 | `increase(hw_gpu_pcie_nonfatal_total[1h])` | PCIe non-fatal errors |
| 8 | `increase(hw_dmesg_gpu_errors_total[1h])` | GPU/driver errors in kernel log |
| 9 | `hw_rdma_port_state` | RDMA port went down? |
| 10 | `hw_gpu_temperature_celsius` | Thermal excursion before crash? |

**Interpretation:**
- **Any uncorrectable error (UE) > 0** → hardware fault. That GPU/link is bad. The node should be drained for maintenance.
- **High correctable errors (CE)** → degrading hardware. Not immediately fatal but indicates the component is failing.
- **PCIe fatal** → PCIe link reset or card disconnect. Likely unrecoverable without node restart.
- **RDMA port state → 0** → network link went down. Check physical connectivity and switch.
- **dmesg GPU errors increasing** → driver-level GPU fault detected by the kernel.
- **Temperature spike before crash** → thermal throttling or shutdown. Check cooling.
- **Compare across nodes** — healthy nodes should have zero RAS/PCIe errors. Any node with non-zero values is the problem.

**GPU driver fault / D-state hang (non-fatal but degrading):**

Not all GPU hardware issues crash the job. A common pattern on AMD Instinct nodes is a **kernel bug in the RAS sysfs reporting path** that doesn't affect GPU compute but makes the monitoring interface unreliable. Diagnosis:

1. **Detect:** `hw_dmesg_gpu_errors_total` > 0 on one node, 0 on all others. `hw_scrape_duration_seconds` alternates between ~12s and <3s on the affected node (timeout vs success). `hw_gpu_power_watts` has intermittent gaps.

2. **Confirm via dmesg (Ray Jobs API):** Submit a Ray job to read `dmesg --level=err,warn -T` on the affected node. Look for `amdgpu` kernel oops, especially in the CPER (Common Platform Error Record) or RAS sysfs paths:
   - `amdgpu_cper_ring_get_ent_sz` / `amdgpu_cper_ring_write` — CPER ring buffer corruption
   - `amdgpu_ras_aca_sysfs_read` / `aca_sysfs_read` — RAS sysfs read hang
   - `amdgpu_cper_generate_ce_records` — correctable error record generation crash

3. **Check D-state process accumulation (Ray Jobs API):** Submit a Ray job to run `ps axo pid,stat,wchan:30,cmd` and count processes in D-state. Hundreds of D-state processes stuck in `amdgpu_cper_ring_write` confirms the driver bug is active and processes are accumulating (unkillable without reboot).

4. **Assess impact:**
   - **Training:** Unaffected. GPU compute uses `/dev/kfd` (KFD command submission), which is a completely separate code path from the RAS sysfs interface. The kernel bug is in the error-reporting path, not the compute path.
   - **Monitoring:** GPU hardware metrics (`hw_gpu_power_watts`, `hw_gpu_temperature_celsius`, `hw_gpu_vram_used_bytes`, RAS counters) are intermittently unavailable. Host and training metrics are unaffected.
   - **D-state accumulation:** ~6 unkillable processes per hour. Not immediately dangerous (Linux PID max = 4,194,304) but indicates the node needs a reboot.

5. **Recovery:** Not possible without a node reboot. D-state processes are stuck in kernel space and cannot be signaled. The CPER ring corruption persists in kernel memory. Schedule a reboot at the next job boundary.

6. **Report for cluster admin:** Include the node hostname, kernel version (`uname -r`), amdgpu module version (`/sys/module/amdgpu/version`), exact dmesg call stack, `hw_dmesg_gpu_errors_total` count, D-state process count, onset time (from dmesg timestamps and `hw_dmesg_gpu_errors_total` timeline), and confirmation that all other nodes are clean.

---

### Playbook 5: GPU Health Check

**Trigger:** User says "check GPU health", "are the GPUs OK", "check thermals", or proactive monitoring of a running job.

**Goal:** Assess GPU health across the cluster — temperatures, power, clocks, VRAM, error counters.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `hw_gpu_temperature_celsius` | Junction temperature | >90C = concern, >100C = throttling |
| 2 | `hw_gpu_power_watts` | Power draw | Variance >50W across GPUs = investigate |
| 3 | `hw_gpu_clock_mhz{type="sclk"}` | Core clock | Sudden drop = investigate |
| 4 | `hw_gpu_vram_used_bytes / hw_gpu_vram_total_bytes` | VRAM utilization | >95% = risk of GPU OOM |
| 5 | `hw_gpu_ras_umc_ce_total` | HBM correctable errors | Any >0 = accumulating hardware issue |
| 6 | `hw_gpu_ras_umc_ue_total` | HBM uncorrectable errors | Any >0 = bad GPU, drain node |
| 7 | `hw_gpu_ras_xgmi_ce_total` | XGMI correctable errors | Any >0 = inter-GPU link degrading |
| 8 | `hw_gpu_pcie_correctable_total` | PCIe correctable errors | Steady increase = link issue |
| 9 | `ray_node_gpus_utilization` | GPU utilization | <80% during training = underutilization |

**Report format for health check:**

```
GPU Health Summary (<N> nodes, <G> GPUs)

Temperature:  min <X>C  max <X>C  avg <X>C  (threshold: 90C)
Power:        min <X>W  max <X>W  avg <X>W  spread: <X>W
Core clock:   min <X>MHz  max <X>MHz  (nominal: <X>MHz)
VRAM:         <X>% used  (<X> GB / <X> GB)
Utilization:  min <X>%  max <X>%  avg <X>%

RAS Errors:   <N> correctable, <N> uncorrectable
PCIe Errors:  <N> correctable, <N> non-fatal, <N> fatal

Anomalous GPUs: <list of host:gpu with any non-zero errors or outlier metrics>
```

---

### Playbook 6: Network Health Check

**Trigger:** User says "check network", "is the network OK", "RDMA health", or investigating intermittent NCCL timeouts.

**Goal:** Assess network health — TCP retransmits, RDMA errors, congestion, port state.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate | >10/s sustained = concern |
| 2 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate | Any sustained > 0 = concern |
| 3 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts | Any > 0 = link or switch issue |
| 4 | `hw_rdma_port_state` | Port state (1=ACTIVE) | Any 0 = port down |
| 5 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notifications | Sustained = network congestion |
| 6 | `rate(hw_rdma_req_tx_retry_excd_err_total[5m])` | Retry exhaustion | Any > 0 = packets lost |
| 7 | `rate(hw_rdma_req_rx_cqe_err_total[5m])` | CQE errors | Any > 0 = RDMA errors |
| 8 | `rate(hw_net_rx_errors_total[5m])` | NIC RX errors | Any > 0 = NIC issue |
| 9 | `rate(hw_net_tx_errors_total[5m])` | NIC TX errors | Any > 0 = NIC issue |
| 10 | `rate(hw_tcp_abort_on_timeout_total[5m])` | TCP connections aborted | Any > 0 = severe |

**Interpretation:**
- **TCP retransmits only, no RDMA errors** → IP/TCP path issue (NFS, coordinator gRPC), not the NCCL/RCCL data path. **Critical: high TCP retransmits alone do NOT degrade training TGS.** TCP carries NFS (checkpoint I/O, data loading) and gRPC (coordinator heartbeats), but RCCL collectives use RDMA. A job can have thousands of TCP retransmits/sec and still achieve full TGS if RDMA is clean.
- **RDMA retransmits + ACK timeouts on specific devices** → bad cable, bad port, or switch issue on that link. Identify the device and port labels. **This is the #1 cause of unexplained TGS degradation.** Even a few nodes with sustained RDMA retransmits can cause 20-30% TGS degradation cluster-wide because all-to-all collectives are synchronous — the slowest RDMA link bounds every node in every step.
- **Congestion notifications (CNP) across many nodes** → switch-level congestion. May need ECN tuning or traffic engineering.
- **Retry exhaustion** → packets permanently lost. Indicates a hard link failure, not transient congestion. Nodes with retry exhaustion should be top priority for exclusion.
- **Port state 0** → RDMA link is down. Physical layer issue. Check cable and switch port.
- **Correlate with training events** — if retransmits spike at the same time as a step time increase or hang, the network event caused the training issue.

**RDMA degradation signature in training metrics:**
- RDMA-induced TGS drops appear as a **constant step-time increase** (not variable spikes), because every step's collective communication is uniformly slowed by the degraded link. This distinguishes RDMA issues from transient events (which cause isolated spikes) and checkpoint saves (which are periodic).
- **Phase correlation:** Map RDMA retransmit bursts on specific nodes to TGS degradation phases. If TGS drops when RDMA retransmits spike on node X, recovers when node X stops retransmitting, and drops again when node Y starts retransmitting, this proves a causal link. Use `rate(hw_rdma_tx_retx_pkts_total[5m])` per host alongside `tb_perf_per_device_tokens_per_sec` over the same time window.
- **Recovery test:** When RDMA retransmits cease on the offending nodes, TGS should recover to baseline within 1-2 steps. If TGS does not recover after retransmits stop, there is an additional or different root cause.

**Node exclusion prioritization** (when spare nodes are limited):

| Priority | Condition | Rationale |
|----------|-----------|-----------|
| 1 (highest) | RDMA retry exhaustion (`hw_rdma_req_tx_retry_excd_err_total` > 0) | Permanent packet loss — hard link failure |
| 2 | Sustained RDMA retransmits + ACK timeouts | Active link degradation slowing every collective |
| 3 | High CNP (congestion notifications) | Network congestion — may be switch-level, not node-level |
| 4 (lowest) | TCP retransmits only | No RDMA impact — does not affect training TGS |

Nodes with only TCP retransmit issues should **not** be excluded unless they also show RDMA problems. If spare capacity is limited, focus exclusions on priority 1-2 nodes.

---

### Playbook 7: Training Stability

**Trigger:** User says "check training", "is training stable", "loss looks weird", "throughput dropping", or proactive monitoring.

**Goal:** Assess training health — loss convergence, gradient norms, throughput, step time consistency.

**Important:** Filter out synthetic anti-staleness fills. Only use data points where `tb_metrics_plugin_staleness_fill == 0`. The `tb_*` bridge is best-effort and may have gaps — if critical steps are missing, fall back to the raw TensorBoard event file (see "Recovering from `tb_*` gaps" in the Troubleshooting section).

**Queries** (range over full job or recent window):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `tb_learning_loss and tb_metrics_plugin_staleness_fill == 0` | Training loss (real data only) |
| 2 | `tb_learning_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Gradient norm |
| 3 | `tb_learning_raw_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Pre-clipping gradient norm |
| 4 | `tb_perf_step_time_seconds and tb_metrics_plugin_staleness_fill == 0` | Step time |
| 5 | `tb_perf_per_device_tokens_per_sec and tb_metrics_plugin_staleness_fill == 0` | Throughput |
| 6 | `tb_learning_current_learning_rate and tb_metrics_plugin_staleness_fill == 0` | Learning rate |
| 7 | `tb_step` | Current step (for progress tracking) |
| 8 | `tb_learning_moe_lb_loss and tb_metrics_plugin_staleness_fill == 0` | MoE load balance loss (if MoE model) |

**Interpretation — training metrics:**
- **Loss divergence** (sudden increase or NaN) → learning rate too high, data issue, or numerical instability. Check if grad norm spiked at the same time.
- **Gradient norm spikes** → unstable training. If `raw_grad_norm >> grad_norm`, gradient clipping is active and may be too aggressive.
- **MoE load balance loss spike** → routing changed dramatically, causing expert load imbalance. This is a training-level root cause that will show up as uneven GPU utilization and TGS drop.
- **Learning rate anomaly** → verify the LR schedule matches expectations. A flat LR when warmup should be active (or vice versa) indicates a config issue.

**Interpretation — step time and throughput:**
- **Step time periodic spikes** → likely checkpoint saves. Correlate with `checkpoint_period` from the job config. Spikes of 10-30s every N steps are normal.
- **Throughput regression** → >10% drop from steady-state average warrants investigation. Run the contention checklist below.
- **Step time gradual increase** → resource contention worsening over time. Run the contention checklist below.

**Contention checklist.** When throughput drops or step time increases, systematically check all contention sources at the same timestamp. Any of these can silently degrade performance:

| Contention source | Metrics to check | What to look for |
|--------------------|-----------------|------------------|
| **CPU** | `hw_procs_running`, `hw_procs_blocked`, `hw_context_switches_total`, `ray_node_cpu_utilization`, `hw_gpu_user_processes` | High runnable count (oversubscribed), high blocked count (I/O wait), excessive context switching. **For multi-job comparison:** a constant but higher process count in one job creates a constant baseline CPU tax — extra processes compete with data pipeline threads, NCCL coordination, and Python runtime. This won't show as a spike but as a persistent throughput gap. Compare `hw_procs_running` across jobs, not just within a single job. **Known cause:** `enable_single_replica_ckpt_restoring=true` leaks RCCL communicator polling threads on restore (~17 threads/host on a 3-replica run) because Orbax's `_merge_globalized_replicas` creates all-reduce collectives with unique `GpuCliqueKey`s cached permanently in XLA's C++ clique cache; `jax.clear_caches()` does not help (see "Checkpointing interference" section). |
| **Network (TCP)** | `rate(hw_tcp_retransmits_total[5m])`, `rate(hw_tcp_estab_resets_total[5m])`, `rate(hw_tcp_abort_on_timeout_total[5m])` | Retransmit bursts slow NFS and gRPC; resets and aborts indicate connection failures |
| **Network (RDMA)** | `rate(hw_rdma_tx_retx_pkts_total[5m])`, `rate(hw_rdma_tx_ack_timeout_total[5m])`, `rate(hw_rdma_rx_cnp_pkts_total[5m])` | RDMA retransmits slow NCCL/RCCL collectives; CNP indicates switch congestion |
| **Storage I/O** | `hw_io_pressure_full_pct`, `hw_io_pressure_some_pct`, `hw_mem_dirty_bytes`, `hw_mem_writeback_bytes` | I/O pressure stalls data loading and checkpointing; dirty page buildup indicates NFS/storage backlog |
| **Memory** | `ray_node_mem_used`, `hw_oom_kills_total`, `hw_procs_blocked` | Memory pressure causes swapping (blocked procs); approaching limit risks OOM |
| **GPU thermal** | `hw_gpu_temperature_celsius`, `hw_gpu_clock_mhz{type="sclk"}`, `hw_gpu_power_watts` | Temperature rise → clock throttle → power drop → throughput drop (all correlated symptoms of thermal issue) |
| **GPU hardware** | `hw_gpu_ras_*`, `hw_gpu_pcie_*_total`, `hw_dmesg_gpu_errors_total` | Accumulating errors degrade performance before causing a crash |
| **Training-level** | `tb_learning_moe_lb_loss`, `tb_learning_grad_norm`, `tb_learning_loss` | Model behavior changes (routing shifts, grad spikes, loss instability) that alter computational load |

Check all rows — contention sources often compound (e.g., I/O pressure from checkpointing + network congestion from RDMA traffic = amplified step time spike).

**Cross-domain correlation queries** (the power of a unified TSDB):

```promql
# Did a temperature spike cause a throughput drop?
# Query both in the same time window and compare timestamps
tb_perf_per_device_tokens_per_sec{host="node001"}
hw_gpu_temperature_celsius{host="node001"}

# Did network issues cause step time spikes?
tb_perf_step_time_seconds{host="node001"}
rate(hw_tcp_retransmits_total{host="node001"}[5m])

# Did I/O pressure from checkpointing cause a loss spike?
tb_learning_loss{host="node001"}
hw_io_pressure_full_pct{host="node001"}
```
