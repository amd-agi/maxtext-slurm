---
name: tsdb-diagnosis
description: "Diagnose training job incidents and check cluster health using the per-job Prometheus TSDB. Use when the user asks to diagnose a failure root cause, check GPU/network health, query Prometheus metrics, investigate a hang, or when the triage skill recommends deeper TSDB analysis."
---

# TSDB Diagnosis

Query the per-job Prometheus TSDB to diagnose incident root causes and assess cluster health. Works on any `RAY=1` job — finished or running.

**Relationship to other skills:**
- **job-log-triage** identifies *what* failed (from logs). This skill identifies *why* using time-series metrics.
- **performance-analysis** identifies *why training is slow* (from xplane/HLO traces). This skill identifies *why metrics are anomalous* at the system level.
- The triage skill's next-step templates often say "Query Prometheus TSDB" — this skill automates that.

## Prerequisites

This skill requires a Prometheus TSDB, which is only available for `RAY=1` jobs. If the job was launched with `RAY=0`, no TSDB exists — report this and suggest re-running with `RAY=1`.

## Remote Execution via Ray Jobs API

For `RAY=1` live jobs, the **Ray Jobs API** (`http://<head_host>:8265/api/jobs/`) is a universal remote execution mechanism. It runs arbitrary Python inside the job's Docker containers — no SSH, no Slurm CLI required. As long as the Ray head node is reachable over HTTP, you can inspect and operate on any node in the cluster.

**Why this matters:** In many environments, the diagnosis machine has no SSH access to compute nodes and no Slurm CLI. The Ray Jobs API is the only way to reach inside the running containers. It is used throughout this skill for process inspection, dmesg reading, file delivery, and observability stack management.

**Safety boundary — what you can and cannot do:**

| Safe (decoupled from training) | Dangerous (touches training) |
|-------------------------------|------------------------------|
| Read files, copy files, inspect sysfs | Kill any process in the training process tree |
| `ps`, `dmesg`, `uname`, system inspection | `pkill python3` or broad pattern kills |
| Kill/restart the metrics exporter (watchdog auto-restarts) | Send signals to JAX/XLA processes |
| Overwrite plugin scripts (picked up on next poll cycle) | Modify training config or data files |
| Overwrite `metrics_exporter.sh` + kill exporter | Write to GPU sysfs control files |
| Query Prometheus, read logs | Anything that could cause a training process to exit |

**Critical rule:** Never kill a training process. In a distributed job, killing one trainer on one node causes all other nodes to hang waiting for the dead peer, eventually crashing the entire job (heartbeat timeout or NCCL timeout across all N nodes). The observability stack (metrics exporter, plugins, Prometheus) is fully decoupled — you can tear it down and rebuild it without any training impact.

**Submitting a job:**

```bash
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{"entrypoint": "python3 -c \"<python_code>\"", "runtime_env": {}}'
```

**Polling for results:**

```bash
# Check status
curl -s 'http://<head_host>:8265/api/jobs/<job_id>'
# Read stdout/stderr
curl -s 'http://<head_host>:8265/api/jobs/<job_id>/logs'
```

**Node targeting:** By default, Ray jobs run on the head node. To inspect a specific worker node, the Ray job's Python code must explicitly discover and connect to that node's resources, or you can use node-local information (like hostname) to verify which node the job landed on. For operations that must run on a specific node (e.g., reading that node's dmesg), you may need to submit multiple jobs and check which node each lands on, or use Ray's scheduling hints.

This API is referenced throughout the skill — in "Metrics Exporter Operations & Live Hotfix" for deploying fixes, in Playbook 4 for GPU driver fault diagnosis, and in the troubleshooting section for process inspection.

## Workflow

0. **Triage first.** Before querying the TSDB, run the **job-log-triage** skill on the job (or confirm triage has already been done earlier in the conversation). Triage establishes critical context that shapes the entire diagnosis:
   - **Job state** — running, completed, failed, cancelled, or crashed (determines how to connect to Prometheus in step 2)
   - **Fresh start vs checkpoint restore** — affects expected baseline (restored jobs may leak RCCL resources; see "Checkpointing interference")
   - **Failure class** — hang, heartbeat-timeout, OOM, etc. (determines which playbook to run in step 5)
   - **Step range and timing** — which steps are comparable, where anomalies occurred
   - **Config parameters** — `PASSTHROUGH_ARGS`, `NNODES`, `checkpoint_period`, etc.

   Without triage, you risk misinterpreting metrics (e.g., querying post-hang idle metrics as if they were training metrics, or missing that a checkpoint restore caused resource leaks). For proactive health checks on a live job where no failure has occurred, triage is still useful to confirm the job is running and extract the head node hostname/port.

1. **Locate the TSDB.** Resolve the job directory (same rules as triage: given a log file, job dir, Slurm ID, or k8s job ID). For k8s jobs, the primary log (`outputs/<id>-<name>.log`) contains rank 0's output; per-rank logs are at `outputs/<id>-<name>/rank-N.log`. Verify `<job_dir>/prometheus/` exists **and contains data** — it should have subdirectories with ULID names (e.g., `01KHV6MFN61MJKZ3ZSYYRYGDGX`) and/or a `wal/` directory. An empty or near-empty `prometheus/` directory means Prometheus failed to start or never scraped — check the observability stack logs (see "Troubleshooting Missing or Incomplete TSDB Data"), report what went wrong, and skip to log-only diagnosis.

2. **Connect to Prometheus.** First determine whether the job is still running, then choose the appropriate access method.

   **Determine job state** (from triage, or verify directly): The triage report from step 0 tells you whether the job is running or finished. If you need to verify independently:
   - `squeue -j <id> -h -o %T 2>/dev/null` — if it returns `RUNNING`, the job is live → **Case A**.
   - If `squeue` is unavailable or the job isn't found, check the log: a `JOB SUMMARY` block means the job is finished → **Case B**. No summary and no advancing steps → likely crashed → also **Case B** (the live Prometheus is gone).

   **Critical rule: never delete a TSDB lock file, and never run `prometheus.sh view` against a running job's TSDB.** The lock is held by the live Prometheus instance. Deleting it or starting a second instance against the same TSDB risks data corruption. For finished/crashed jobs, `prometheus.sh view` handles stale lock files automatically (Prometheus detects and replaces them).

   **Case A — Live job (still running):** Prometheus is already running on the job's head node (task 0). Find the head node hostname and port from the SSH tunnel command in the job log (`ssh -L ... <head_host>:<port>`). The port is usually 9190 but may differ if 9190 was occupied at job start (the startup script auto-increments: 9191, 9192, ...). Also check for `[Prometheus] WARNING: port 9190 was occupied; using <port> instead` in the log. Query at `http://<head_host>:<port>` — **not** `localhost:<port>`. The head node hostname comes from the log (e.g., `chi2816`); `localhost:9090` on the machine you are running from may be a completely different Prometheus (e.g., a cluster-level monitor). The SSH tunnel command in the log is for the **user's machine** to reach the head node through a jump host — it binds the port on the user's machine (which may be a laptop or the same cluster node you are on), not on the head node. If the user has set up such a tunnel on this machine, `localhost:<port>` already points to the live Prometheus — do not shadow it by starting another instance. Do **not** start a second Prometheus instance for a live job.

   **Case A fallback — Live job but Prometheus is unreachable:** If the head node's Prometheus doesn't respond (connection timeout, firewall, network partition between your machine and the head node), you can still access the TSDB data by copying the immutable blocks (not the WAL or lock file) to a temporary directory and running a read-only Prometheus against the copy. **Run the port scan first** (same as Case B — `ss -tlnp | grep 919`) to find a free port:
   ```bash
   mkdir /tmp/prom_<jobid>_readonly
   rsync -a --exclude='wal' --exclude='chunks_head' --exclude='lock' <job_dir>/prometheus/ /tmp/prom_<jobid>_readonly/
   utils/prometheus.sh view /tmp/prom_<jobid>_readonly -p <port> &
   PROM_FALLBACK_PID=$!
   ```
   This gives you access to all compacted data but **not** the most recent uncompacted samples still in the WAL. Expect a gap of up to 2 hours at the end of the data. For a long-running job, this is usually sufficient. Clean up in step 8: kill the process first (`kill $PROM_FALLBACK_PID`), then remove the data copy (`rm -rf /tmp/prom_<jobid>_readonly`).

   **Case B — Finished job (completed, failed, or cancelled):** Before starting a read-only Prometheus, **always scan for occupied ports first**. The user may have SSH tunnels on this machine forwarding to live Prometheus instances on remote head nodes (e.g., `ssh -L 9190:chi2816:9190 ...` makes the live job's Prometheus available at `localhost:9190`). Starting a read-only Prometheus on the same port would shadow the live database and break the user's monitoring.
   ```bash
   ss -tlnp | grep 919
   ```
   Check the process column to classify each occupant:
   - **`ssh`** — an SSH tunnel the user set up to reach a live job's Prometheus on a remote head node. **Never kill, never shadow.** This is the most common occupant to watch for.
   - **`prometheus`** — either a live job's Prometheus (if this machine is a head node) or a stale read-only instance from a previous diagnosis session. To distinguish: inspect the cmdline with `cat /proc/<pid>/cmdline | tr '\0' ' '` and check whether `--storage.tsdb.path` points to a running job's directory (live — never kill) or a finished job's directory (stale — safe to kill with `kill <pid>`).
   - **Anything else** — a legitimate process. Don't touch it.

   Pick a port that is **not occupied by any process**. Start at 9190 and increment past all occupied ports. **Always capture the PID** so you can reliably kill it later (do not rely on bash `%1` job control — it breaks if any other command is backgrounded in between):
   ```bash
   utils/prometheus.sh view <job_dir>/prometheus -p <port> &
   PROM_PID=$!
   ```
   Wait for "Open http://localhost:" in stdout (typically <3 seconds). Kill the process when done (step 8).

   **Case C — Multi-job comparison:** At least one job must have a Prometheus TSDB. **Run the port scan once** (`ss -tlnp | grep 919`) to find all occupied ports before starting any read-only instances. Assign each finished job a different free port, skipping all occupied ports. Capture each PID:
   ```bash
   utils/prometheus.sh view <job_A_dir>/prometheus -p <free_port_1> &
   PROM_PID_A=$!
   utils/prometheus.sh view <job_B_dir>/prometheus -p <free_port_2> &
   PROM_PID_B=$!
   ```
   If one of the jobs is still live, query its existing Prometheus at `http://<head_host>:<port>` (Case A) alongside the read-only instances on localhost. Jobs without a TSDB (`RAY=0`) can only be compared using log-based data from the triage skill. Query each Prometheus on its own address/port and compare results side by side.

3. **Always query all metrics first.** Before any diagnostic work, discover what the TSDB contains. This tells you which metric families were collected, how many nodes are present, and what time range is covered — essential context for choosing the right queries and interpreting results.

   In the examples below, replace `<host>:<port>` with the address from step 2: `<head_host>:<port>` for live jobs (Case A), or `localhost:<port>` for read-only instances (Cases B/C).

   **For read-only instances (Case B/C), always start with `api/v1/status/tsdb`** to find the time range before any data queries. This endpoint doesn't need a timestamp and tells you the min/max times the TSDB covers. Without this, you won't know what `&time=` values to use for instant queries.

   ```bash
   # All available metric names — understand what the observability stack captured
   curl -s 'http://localhost:<port>/api/v1/label/__name__/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   names = sorted(data.get('data', []))
   print(f'Total metrics: {len(names)}')
   for prefix in ['hw_gpu_', 'hw_tcp_', 'hw_rdma_', 'hw_io_', 'hw_mem_', 'hw_oom_', 'hw_net_', 'hw_procs_', 'hw_dmesg_', 'ray_node_', 'tb_']:
       group = [n for n in names if n.startswith(prefix)]
       if group: print(f'  {prefix}*: {len(group)} metrics')
   other = [n for n in names if not any(n.startswith(p) for p in ['hw_', 'ray_', 'tb_', 'up', 'scrape_'])]
   if other: print(f'  other: {other}')
   "

   # All hosts in the TSDB
   curl -s 'http://localhost:<port>/api/v1/label/host/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   hosts = sorted(data.get('data', []))
   print(f'Hosts ({len(hosts)}): {', '.join(hosts)}')
   "

   # Time range of available data — use min/max timestamps from the TSDB.
   # For live jobs (Case A), omit &time to use current time.
   # For read-only instances (Case B/C), first find the TSDB time range:
   curl -s 'http://localhost:<port>/api/v1/status/tsdb' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   d = data.get('data', {})
   min_t = d.get('minTime', 0) / 1000
   max_t = d.get('maxTime', 0) / 1000
   print(f'TSDB range: {datetime.datetime.fromtimestamp(min_t)} to {datetime.datetime.fromtimestamp(max_t)}')
   print(f'  min_ts={min_t:.0f}  max_ts={max_t:.0f}')
   "
   # Then query 'up' at the last known timestamp to see which targets were scraped:
   curl -s 'http://localhost:<port>/api/v1/query?query=up&time=<max_ts>' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       ts = float(r['value'][0])
       print(f'{r[\"metric\"].get(\"job\",\"?\")} last_scrape={datetime.datetime.fromtimestamp(ts)}')
   "
   ```

   This step is not optional — always run it. The output tells you:
   - Whether GPU metrics (`hw_gpu_*`), network metrics (`hw_tcp_*`, `hw_rdma_*`), and training metrics (`tb_*`) are all present.
   - How many nodes were scraped (should match the job's `NNODES`).
   - The time range covered, confirming the TSDB has data for the period of interest.

   **Verify you are querying the correct TSDB.** After discovering metrics, cross-check a data point against the job log to confirm the database belongs to the expected job:

   ```bash
   # Query the last recorded step and loss (use a timestamp from the time range discovered above)
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_step&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  step={r[\"value\"][1]}')
   "
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_learning_loss&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  loss={r[\"value\"][1]}')
   "
   ```

   Compare these values against the corresponding `completed step:` line in the job log. The step number and loss should match exactly (loss may differ by rounding in the last decimal). If they don't match, you are querying the wrong Prometheus — re-check the address/port. This is especially important in multi-job comparisons (Case C) to avoid mixing up databases.

4. **Determine the time window.** The queries need a start/end time (Unix timestamps):
   - **From triage:** use the crash time or hang-start time identified in the triage report.
   - **From log timestamps:** parse the timestamp from the last `completed step:` line and the first training line.
   - **For health checks:** use the full job duration, or "last N minutes" for live jobs.
   - **Heartbeat timeout:** compute `crash_time - heartbeat_timeout_seconds` to get when heartbeats actually stopped.

   **Map training steps to wall-clock timestamps** using `tb_step`. This is essential for multi-job comparisons and for querying system metrics at a specific training step:

   ```bash
   # Find the wall-clock time range of the TSDB and map key steps to timestamps.
   # Use a wide range (e.g., 24h before the last known scrape) to cover the full job.
   curl -s 'http://localhost:<port>/api/v1/query_range?query=tb_step&start=<start>&end=<end>&step=60s' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   results = data.get('data', {}).get('result', [])
   if not results: print('No tb_step data'); sys.exit()
   vals = [(float(v[0]), float(v[1])) for v in results[0]['values'] if v[1] != 'NaN']
   print(f'Step range: {vals[0][1]:.0f} to {vals[-1][1]:.0f}')
   print(f'Time range: {datetime.datetime.fromtimestamp(vals[0][0])} to {datetime.datetime.fromtimestamp(vals[-1][0])}')
   # Find timestamps for specific steps — replace with your target steps
   for target in [200, 300, 500, 1000]:
       for ts, step in vals:
           if step >= target:
               print(f'  step {target}: ts={ts:.0f} ({datetime.datetime.fromtimestamp(ts)})')
               break
   "
   ```

   Use the resulting timestamps to query system metrics (`hw_*`, `ray_*`) at the wall-clock times corresponding to specific training steps.

5. **Run the appropriate playbook** (see Diagnostic Playbooks below). Each playbook specifies the PromQL queries, how to interpret results, and what to conclude.

6. **Deepen the diagnosis if needed.** If the playbook identifies a suspicious metric but the root cause mechanism isn't clear:
   - **Check ray worker logs** — look for NCCL init waves, XLA recompilations, or warnings that correlate with the metric anomaly. See the "Ray Worker Logs" section for patterns and locations.
   - **Trace into source code** — when the anomaly is persistent, uniform across nodes, and not explained by hardware/network/thermal factors. See the "Trace suspicious findings to source code" diagnostic principle for the full methodology.

7. **Report findings** in the structured format (see Output Format below).

8. **Cleanup — stop read-only instances immediately after each diagnosis.** Kill all read-only Prometheus instances you started in step 2 as soon as you've finished querying them (i.e., after reporting findings in step 7 — do not leave them running "in case you need them later"). Use the PIDs captured in step 2:
   ```bash
   kill $PROM_PID 2>/dev/null        # single instance (Case B)
   kill $PROM_PID_A $PROM_PID_B 2>/dev/null   # multi-job (Case C)
   kill $PROM_FALLBACK_PID 2>/dev/null         # Case A fallback
   ```
   If the user asks a follow-up question that needs the TSDB again, restart the read-only instance at that point (repeating the port scan from step 2). Restarting is cheap (<3 seconds) — leaving instances running leaks ports and risks shadowing SSH tunnels or live Prometheus instances that the user sets up mid-conversation.

   Do **not** kill the live Prometheus of a running job.

## Querying Prometheus

All queries use the Prometheus HTTP API via `curl`. Parse responses with inline Python.

### Instant query (single point in time)

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>'
```

**Always specify `&time=<unix_ts>` for read-only instances (Case B/C).** Without it, Prometheus defaults to "now" — which is past the end of the data for finished jobs, returning empty results. Use a timestamp from within the job's time range (discovered in step 3).

### Range query (time series)

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<unix_ts>&end=<unix_ts>&step=<interval>'
```

Use `step=30s` for high resolution (short windows), `step=60s` for medium, `step=300s` for long durations.

### URL encoding for `rate()` and `increase()`

When using `rate()` or `increase()` in curl URLs, the square brackets must be URL-encoded:

```bash
# WRONG — brackets break the URL:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total[5m])'

# CORRECT — URL-encode the brackets:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total%5B5m%5D)'
```

`[` = `%5B`, `]` = `%5D`. This applies to all PromQL queries containing range selectors.

### Standard query + tabular output pattern

Use this pattern for all queries — it extracts the metric labels and values into a readable table:

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    val = r['value'][1]
    print(f'  {host:<20} gpu={gpu:<4} {val}')
"
```

For range queries, use this to extract per-host min/max/avg:

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<start>&end=<end>&step=30s' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    vals = [float(v[1]) for v in r['values'] if v[1] != 'NaN']
    if vals:
        print(f'  {host:<20} gpu={gpu:<4} min={min(vals):.1f}  max={max(vals):.1f}  avg={sum(vals)/len(vals):.1f}')
"
```

### Filtering by host

Narrow queries to specific hosts (e.g., the accused nodes from triage):

```promql
hw_gpu_power_watts{host=~"node001|node002"}
ray_node_gpus_utilization{host="node003"}
```

### Rate queries for counters

Counters (names ending in `_total`) must use `rate()` or `increase()`:

```promql
rate(hw_tcp_retransmits_total[5m])
increase(hw_oom_kills_total[1h])
```

## Ray Worker Logs

Ray worker logs are the second source of truth alongside TSDB metrics. They contain NCCL/RCCL initialization messages, XLA compilation events, Python tracebacks, and framework-level warnings that the TSDB cannot capture. Use them to confirm hypotheses formed from metric analysis.

**Location:** `<job_dir>/ray_logs/<hostname>/worker-*-<pid>-<pid>.out` (stdout) and `.err` (stderr). Each host has a subdirectory; each Ray worker has a pair of `.out`/`.err` files.

**Common patterns to search for:**
- `init.cc:2095 NCCL WARN MSCCL++` — NCCL communicator initialization. Count the number of "waves" (clusters of these messages separated by time gaps) to detect unexpected communicator creation (e.g., extra wave from single-replica restore broadcast).
- `Compilation of` or `Compiled` — XLA compilation events. Unexpected recompilations mid-training indicate shape changes or cache misses.
- `NCCL WARN` or `RCCL WARN` — collective communication warnings.
- Python tracebacks — stack traces from exceptions (may be caught and logged without crashing).
- `SingleReplicaArrayHandler` or `broadcast_one_replica_to_all` — checkpoint restore broadcast messages (relevant to RCCL leak diagnosis).

**Practical tips:**
- Worker logs can be very large. Use grep to search, don't read entire files.
- The head node (task 0) typically has the most informative logs — start there.
- When comparing two jobs, grep for the same pattern in both and diff the output (e.g., count NCCL init waves in each).

## Troubleshooting Missing or Incomplete TSDB Data

When the TSDB is empty, has gaps, or is missing metric families, see [references/troubleshooting.md](references/troubleshooting.md) for the full diagnostic workflow covering observability stack components (Prometheus, metrics exporter, plugins), recovery procedures, and live hotfix instructions.

## Diagnostic Playbooks

Seven playbooks for specific failure modes. Each specifies PromQL queries, interpretation, and conclusions.

| # | Playbook | When to use |
|---|----------|-------------|
| 1 | RCCL/NCCL Hang | Hang confirmed by triage; need GPU power/util/TCP evidence |
| 2 | Heartbeat False-Positive | Heartbeat timeout; need to confirm tasks were healthy |
| 3 | Host OOM | OOM-kill suspected; need memory timeline |
| 4 | Node Failure / Hardware | Node exit or RAS errors; need GPU/thermal/dmesg evidence |
| 5 | GPU Health Check | Proactive health check; no specific failure |
| 6 | Network Health Check | TGS degradation or network-related failures |
| 7 | Training Stability | TGS drift, contention, or resource leaks |

See [references/playbooks.md](references/playbooks.md) for full playbook details including PromQL queries, interpretation tables, and diagnostic principles (correlation vs causation, source code tracing, checkpointing interference).

## Multi-Job Comparison

When comparing metrics across two or more jobs (e.g., "why is job B slower than job A?"), follow these rules:

### 0. Triage all jobs first

Run the **job-log-triage** skill on **each** job being compared (per workflow step 0). For multi-job comparison, triage is especially critical because you need to know:
- Which jobs are running vs finished (determines Case A vs B for each)
- Which jobs are fresh starts vs checkpoint restores (restored jobs may have resource leaks — see "Checkpointing interference")
- The step range and checkpoint period for each job (needed to identify overlapping steps and exclude checkpoint boundaries)
- Any failures or hangs that affect which steps are comparable (e.g., don't compare post-hang idle metrics against active training)

### 1. Isolate config differences first

Before looking at metrics, compare the jobs' configurations. Parse `PASSTHROUGH_ARGS` from the header of each log file (first ~30 lines) and diff them. Common config parameters that affect performance:
- `per_device_batch_size`, `max_target_length`, `steps`
- `remat_policy`, `quantization`, `attention`
- `ici_*_parallelism`, `dcn_*_parallelism` (parallelism strategy)
- `enable_checkpointing`, `checkpoint_period`, `async_checkpointing`
- `load_balance_loss_weight` (MoE)
- `_env_*` flags (XLA flags, NCCL tuning)
- Number of nodes (`NNODES`), GPUs per node

If configs differ, the performance difference may be **expected** — the metric comparison should account for the config change, not treat it as an anomaly.

### 2. Align by training step, not wall clock

Jobs run at different times and speeds. Compare metrics at the **same training step range**, not the same wall-clock time. Use `tb_step` to map between step number and timestamp in each job's TSDB, then query system metrics at the corresponding wall-clock times.

### 3. Compare only overlapping steps

If job A ran steps 0-500 and job B ran steps 200-700, only compare metrics during the overlapping range (steps 200-500). Metrics outside the overlap are not comparable — warmup behavior (early steps) and long-run behavior (late steps) differ inherently.

**Checkpoint restore awareness:** If one job is a fresh start and the other restored from a checkpoint, the first few steps after restore may have different behavior (XLA recompilation, data pipeline warmup, potential resource leaks). Identify which job restored and what step it resumed from (from the triage report or log). Compare steady-state behavior after both jobs have fully warmed up — typically 5-10 steps after the later job's first step.

### 4. Compare runtime environment, not just config

Even with identical `PASSTHROUGH_ARGS`, the runtime environment can differ: different observability stacks (`RAY=1` vs lighter monitoring), different background processes, different node allocations. Check `hw_procs_running` across jobs — a higher process count means more CPU contention. A constant overhead (not a spike) creates a persistent throughput gap that is easy to misattribute to other causes.

### 5. Control for transient events

A throughput difference caused by a one-time network blip in job B is not a systematic issue. Use steady-state averages (skip warmup steps) and look at variance — a persistent gap indicates a real difference, while a spike indicates a transient event.

### 6. Systematic metric sweep

Run the **contention checklist** from Playbook 7 on both jobs at their overlapping step range. Compare each contention source side by side — CPU, network (TCP and RDMA), storage I/O, memory, GPU thermal, GPU hardware, and training-level metrics. The root cause is often a single metric family that differs between jobs while all others are identical.

**Key metrics for multi-job TGS comparison:**
- `tb_perf_per_device_tokens_per_sec` — the TGS metric itself. Compare steady-state averages and variance.
- `tb_perf_step_time_seconds` — step time. Inverse of TGS but more sensitive to outliers (checkpoint saves, profiler hooks).
- `rate(hw_rdma_tx_retx_pkts_total[5m])` per host — **the most impactful differentiator for identical-config jobs on different node sets.** RDMA retransmits directly slow RCCL collectives, and even a few bad nodes cause cluster-wide TGS degradation (20-30%). Use the phased correlation technique from Playbook 6 to map RDMA bursts on specific nodes to TGS degradation phases. Also check `sum(hw_rdma_req_tx_retry_excd_err_total) by (host)` for permanent packet loss.
- `hw_procs_running` — most common differentiator for identical-node-set comparisons. A uniform delta across all hosts points to a software-level resource leak (see "Checkpointing interference"). A per-node delta points to background processes or different node allocations.
- `rate(hw_tcp_retransmits_total[5m])` — TCP retransmits. **Important: high TCP retransmits alone do NOT explain TGS differences** (TCP carries NFS/gRPC, not RCCL). A job with higher TCP retransmits can still have higher TGS if its RDMA is clean. Do not chase TCP retransmits as a TGS root cause unless RDMA is also affected.

### 7. Deepen with logs and source code

If the metric sweep identifies a suspicious delta but doesn't explain the mechanism:
1. **Compare ray worker logs** between the two jobs — grep for NCCL init waves, XLA compilation events, or warnings and diff the counts.
2. **Trace into source code** if the delta is persistent, uniform, and correlates with a specific event (e.g., checkpoint restore). Follow the "Trace suspicious findings to source code" principle.

### 8. Report structure for comparison

```
## Multi-Job Comparison: <job_A> vs <job_B>

### Config differences
| Parameter | Job A | Job B |
|-----------|-------|-------|
| ... | ... | ... |

### Overlapping step range: <start> to <end>

### Metric comparison (steady-state averages over overlapping steps)
| Metric | Job A | Job B | Delta | Likely cause |
|--------|-------|-------|-------|--------------|
| ... | ... | ... | ... | ... |

### Root cause
<Traced from the earliest diverging metric, accounting for config differences>
```

## Common Pitfalls

Hard-won lessons from real diagnosis sessions. Avoid these mistakes:

1. **Empty results from read-only Prometheus.** If an instant query returns empty `result: []`, you almost certainly forgot the `&time=` parameter. Read-only Prometheus defaults to "now" which is past the end of the data. Use `api/v1/status/tsdb` to find the TSDB time range (step 3), then always include `&time=<max_ts>` for instant queries.

2. **Querying the wrong Prometheus.** `localhost:9090` on the head node may be a cluster-level Prometheus, not the job's Prometheus. The job's Prometheus runs on `<head_host>:9190` (or the auto-incremented port). Always verify the TSDB by cross-checking `tb_step`/`tb_learning_loss` against the job log. If the metrics don't match (wrong metric families, wrong step range), you're querying the wrong database.

3. **Mixing up databases in multi-job comparison.** When running multiple read-only Prometheus instances on different ports, it's easy to send a query to the wrong port. Label each port clearly (e.g., "9190 = job 7879, 9191 = job 7882") and verify each with the `tb_step` cross-check before starting diagnostic work.

4. **Diagnosing symptoms as root causes.** GPU power drops, clock drops, utilization drops, and throughput drops are usually symptoms, not causes. Always trace back to the *earliest* anomaly across all metric families — it may be a training-level event (MoE routing shift, grad spike) or a system event (network retransmit, I/O stall), but not the GPU metric itself.

5. **Ignoring checkpoint steps.** A 10x step time spike at step 200 with `checkpoint_period=200` is expected, not an incident. Always identify checkpoint steps before diagnosing step time anomalies.

6. **Comparing jobs without understanding their start conditions.** A job that restored from a checkpoint may have RCCL resource leaks, XLA recompilation overhead, or different data pipeline warmup compared to a fresh start. Always check whether a job is a fresh start or restore before comparing metrics.

7. **Assuming host memory contention affects GPU training.** For GPU training, the GPU compute path is largely independent of host memory usage. A spike in host memory (e.g., from checkpoint saves) does not directly slow GPU computation unless it triggers OOM or swapping. Focus on CPU contention (`hw_procs_running`), network contention, and GPU-level metrics.

8. **Treating high absolute RDMA counters as proof of current degradation.** `hw_rdma_tx_retx_pkts_total` and similar RDMA counters are cumulative from boot (or driver reload), not from job start. A node showing 90M cumulative retransmits may have accumulated them over weeks of previous jobs and have zero retransmits during the current job. Always use `rate()` queries within the job's time window to assess current impact. For quick triage, compare `sum(hw_rdma_tx_retx_pkts_total) by (host)` at job end vs job start — the delta is what matters, not the absolute value.

9. **Assuming `jax.clear_caches()` cleans up RCCL communicators.** `jax.clear_caches()` only clears Python-level JIT compilation caches (traced in `jax/_src/api.py`). RCCL/NCCL communicators live in XLA's C++ GPU clique cache (`GpuCliqueKey → LockableGpuClique` in `xla/backends/gpu/collectives/gpu_cliques.h`), which has no eviction mechanism and no Python-accessible release API. The only way to destroy them is `jax.clear_backends()`, which tears down the entire JAX runtime and is unusable mid-training. When diagnosing RCCL resource leaks (e.g., from single-replica restore), do not recommend `jax.clear_caches()` — it has been tested and confirmed ineffective.

## Metric Reference

Four metric families collected by the observability stack: GPU (`hw_gpu_*`), host/network (`hw_*`), Ray (`ray_node_*`), and training (`tb_*`). See [references/metric-reference.md](references/metric-reference.md) for the complete reference with metric names, labels, types, and sources.

## Output Format

```
## TSDB Diagnosis: <job_dir>

**Analysis type:** <hang | heartbeat | oom | hardware | gpu-health | network-health | training-stability | multi-job-comparison>
**Time window:** <start_time> to <end_time> (<duration>)
**Nodes:** <N> (<host1, host2, ...>)

### Findings
<Plain-English summary: what the metrics show, what the root cause is>

### Metric evidence

<For each key metric queried, show a table with per-host/per-GPU values.
Use the actual query results — do not fabricate data.>

| Host | GPU | Value | Interpretation |
|------|-----|-------|----------------|
| ... | ... | ... | ... |

### Anomalies detected
<Any nodes/GPUs/metrics that deviate from cluster norm. "None" if all healthy.>

### Log evidence (if applicable)
<Ray worker log findings that confirm the metric-based hypothesis.
E.g., "3 NCCL init waves in job 7882 vs 2 in job 7879, confirming extra
communicator creation during checkpoint restore.">

### Source code trace (if applicable)
<Code-level root cause chain when the diagnosis traced into source code.
E.g., "checkpointing.py → SingleReplicaArrayHandler → _merge_globalized_replicas
→ jax.jit(jnp.sum) → XLA all-reduce → leaked RCCL communicators.">

### Correlation with triage
<If triggered from a triage report, confirm or refute the failure hypothesis.
Example: "Triage classified this as an RCCL hang. TSDB confirms: all 192 GPUs
showed 100% utilization at 310W (idle-level) during the hang window, with zero
network errors. This is a confirmed RCCL busy-wait deadlock.">

### Recommended next steps
<Numbered list of specific actions based on the diagnosis>
```

## Integration with Triage

When the triage skill identifies a failure that benefits from TSDB analysis, it includes "Query Prometheus TSDB" in its recommended next steps. The handoff works as follows:

1. **Triage provides:** failure class, crash time, accused hosts/tasks, job directory.
2. **This skill takes over:** starts Prometheus against the persisted TSDB, runs the appropriate playbook, and reports metric-level evidence.
3. **Key handoff scenarios:**

| Triage class | Diagnosis playbook | What TSDB adds |
|--------------|-------------------|----------------|
| `hang` | Playbook 1 (RCCL Hang) | Confirms busy-wait signature, identifies network trigger |
| `heartbeat-timeout` | Playbook 2 (Heartbeat) | Proves false positive — tasks were healthy |
| `oom-host` | Playbook 3 (OOM) | Memory trajectory, checkpoint correlation |
| `node-fail` / `signal-kill` | Playbook 4 (Hardware) | RAS errors, PCIe faults, thermal excursion |
| `nccl-timeout` | Playbook 6 (Network) | Network health at failure time |
| `unknown-death` | Playbook 3 + 4 | OOM evidence or hardware faults |

For proactive use (no triage handoff), the user triggers directly with health-check requests.
