# Troubleshooting Missing or Incomplete TSDB Data

## Troubleshooting Missing or Incomplete TSDB Data

When the TSDB is empty, has gaps, or is missing metric families, the problem is in the observability stack itself. The stack has three components, each with its own persistent log:

| Component | Log location | What it does |
|-----------|-------------|--------------|
| **Prometheus** | `<job_dir>/prometheus/prometheus.log` | Scrapes targets, stores time series |
| **Metrics exporter** | `<job_dir>/metrics_exporter/<hostname>.log` | Runs plugins (`gpu_metrics_plugin.sh`, `host_metrics_plugin.sh`, `tb_metrics_plugin.sh`), serves metrics on port 9400 |
| **Ray runtime** | `<job_dir>/ray_logs/<hostname>/` (`gcs_server.out`, `raylet.out`, `dashboard.log`, etc.) | Cluster formation, built-in metrics exporter on port 55080 |

All three are persisted in the job directory and survive job termination. The watchdog (`_run_with_watchdog`) restarts Prometheus and the metrics exporter on crash, logging restart events with timestamps.

**Diagnosis by symptom:**

| Symptom | Which log to check | What to look for |
|---------|-------------------|------------------|
| TSDB directory empty or missing | Job log (main stdout) | `[Prometheus]` messages — did it start? Did `install_prometheus` fail? Port bind failures? |
| TSDB has data but with time gaps | `prometheus/prometheus.log` | Crash messages followed by watchdog restarts (`Watchdog: exited with code ... restart #N`). Each restart replays the WAL — gap duration ≈ crash-to-restart time (~5s) |
| No `hw_*` metrics (but `ray_*` and `tb_*` present) | `metrics_exporter/<hostname>.log` | Plugin errors — `gpu_metrics_plugin.sh` or `host_metrics_plugin.sh` failing. Exit code 99 means permanently skipped |
| No `tb_*` metrics | `metrics_exporter/<hostname>.log` | `tb_metrics_plugin.sh` failing — usually because TensorBoard event files haven't been written yet (normal during compilation), or the event file path is wrong |
| `tb_*` metrics have gaps (missing steps) | `prometheus/prometheus.log` + `metrics_exporter/<hostname>.log` | The `tb_metrics_plugin.sh` bridge is **best-effort** — gaps can occur when the GPU metrics plugin hangs (blocking the exporter), the exporter crashes and restarts, or Prometheus rejects samples whose `wall_time` is older than its `minValidTime` after TSDB compaction. Check Prometheus log for `"samples that are too old"` warnings and exporter log for plugin timeouts. **When `tb_*` data has gaps, use the raw TensorBoard event file as ground truth** (see "Recovering from `tb_*` gaps" below) |
| No `ray_*` metrics | `ray_logs/<hostname>/` (GCS, raylet, dashboard) | Ray didn't start or its metrics exporter on port 55080 failed. Check `gcs_server.out` and `raylet.out` for errors |
| Some hosts missing entirely | `prometheus/prometheus.log` | Scrape errors for those hosts — `context deadline exceeded` or `connection refused`. Means Prometheus couldn't reach the exporter on those nodes (node down, network issue, or exporter crashed before Prometheus could scrape) |
| `hw_scrape_duration_seconds` is high (>8s) | `metrics_exporter/<hostname>.log` | A plugin is running slow or timing out. ~12s usually means the GPU plugin is hitting its timeout (10s) — check for `timed out` in the exporter log. A flatline (same value for >5 min) means the exporter's collection loop is hung and all `hw_*` data is **ghost data** (see below) |
| `hw_*` metrics present but **flatline** (constant value for extended period) | `metrics_exporter/<hostname>.log` | **Ghost data** — the exporter's collection loop was stuck (e.g., a plugin hung before the timeout fix was deployed), but the HTTP server kept serving the stale cached metrics file. Prometheus assigned fresh scrape timestamps to the unchanged values, making them look "present" but they carry zero information. To detect: check `hw_scrape_duration_seconds` — if it flatlines at the same time, the data is ghost. Also check `hw_gpu_power_watts` — a GPU under training load never holds the exact same wattage for hours. Compare the suspect host against a healthy host in the same job for confirmation |
| All metrics present but stale (not updating) | `prometheus/prometheus.log` + `metrics_exporter/<hostname>.log` | Prometheus may be alive but the exporter is hung. Or Prometheus itself is hung (rare — check for WAL corruption messages) |
| `hw_gpu_power_watts` discontiguous (intermittent gaps) on one node | `metrics_exporter/<hostname>.log` + Prometheus TSDB | **GPU driver fault** causing intermittent sysfs hangs. Check `hw_scrape_duration_seconds` — alternating ~12s (timeout) and <3s (success) confirms the GPU plugin times out intermittently. Query `hw_dmesg_gpu_errors_total` — non-zero on the affected node (zero on all others) confirms a kernel-level GPU error. Use Ray Jobs API to check for D-state processes stuck in `amdgpu_cper_ring_write` (see "D-state Process Accumulation" and Playbook 4) |
| Exporter restarting repeatedly (exit code 137) | `metrics_exporter/<hostname>.log` | Watchdog log shows `exited with code 137, restart #N`. Exit code 137 = SIGKILL. Could be OOM killer, or a manual kill during hotfix deployment. Check for `OSError: Address already in use` after restarts — means the HTTP server from the previous instance hasn't released port 9400 yet |

**Checking the logs:**

```bash
# Prometheus crash/restart history
grep -i 'watchdog\|error\|fatal\|panic\|restart' <job_dir>/prometheus/prometheus.log

# Metrics exporter plugin failures (all hosts at once)
grep -i 'error\|skip\|fail\|exit' <job_dir>/metrics_exporter/*.log

# Ray runtime issues on head node
head_host=$(ls <job_dir>/ray_logs/ | head -1)
grep -i 'error\|fail\|exception' <job_dir>/ray_logs/$head_host/gcs_server.out
```

**Important:** These logs explain why the TSDB is broken — they don't replace the TSDB for diagnosing the training job. Once you've identified and understood the observability gap, report it alongside whatever partial TSDB data is available, and note which time ranges or metric families should not be trusted.

### Recovering from `tb_*` gaps

The `tb_metrics_plugin.sh` bridge is **best-effort**. It can lose data when:
- A plugin (e.g., `gpu_metrics_plugin.sh`) hangs in a D-state, blocking the exporter cycle
- The exporter crashes and the watchdog restarts it, but Prometheus has advanced its `minValidTime` past the pending steps
- The plugin drops stale steps per Rule 4 (freshness filtering) to avoid Prometheus rejection

**The raw TensorBoard event file is always the ground truth.** It is written directly by the training process and is never subject to Prometheus's timeline constraints. When `tb_*` metrics have gaps in the TSDB:

1. **Locate the event file:** `<job_dir>/tensorboard/events.out.tfevents.*` (or check the `--tensorboard_dir` flag in the job config).
2. **Read it programmatically** (Python):
   ```python
   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   ea = EventAccumulator("<path_to_event_dir>")
   ea.Reload()
   for tag in ea.Tags()["scalars"]:
       for e in ea.Scalars(tag):
           print(f"step={e.step}  wall_time={e.wall_time}  tag={tag}  value={e.value}")
   ```
3. **Cross-reference with TSDB:** Compare the steps present in the event file against what Prometheus has. The event file will have every step; the TSDB may be missing ranges.

**Trade-off:** Raw event files require programmatic access and cannot be queried with PromQL or visualized in Grafana dashboards. For one-off investigations this is fine. For systematic dashboarding, the Prometheus `tb_*` metrics remain the primary source — gaps are expected to be rare with the current exporter hardening (plugin timeouts, freshness filtering, mid-training restart fallback).

### Metrics Exporter Operations & Live Hotfix

The metrics exporter runs inside a Docker container on each node, managed by a watchdog that restarts it on crash. During a live job, you may need to deploy fixes to plugins or to the exporter itself without restarting the job. The two cases are fundamentally different.

**Architecture recap:**

```
watchdog (_run_with_watchdog)
  └─ metrics_exporter.sh  (long-running; poll loop every ~10s)
       ├─ gpu_metrics_plugin.sh   (re-sourced each cycle)
       ├─ host_metrics_plugin.sh  (re-sourced each cycle)
       ├─ tb_metrics_plugin.sh    (re-sourced each cycle)
       └─ HTTP server (serves cached metrics file on port 9400)
```

**Case 1 — Plugin hotfix (gpu_metrics_plugin.sh, host_metrics_plugin.sh, tb_metrics_plugin.sh):**

Plugins are re-read from disk on every poll cycle (~10s). To deploy a fix:
1. Overwrite the plugin file at the artifact path inside the container.
2. The next poll cycle picks up the new code automatically. No process restart needed.

**File delivery caveat:** The plugin files live at `<artifact_dir>/utils/` inside the container. The artifact directory is bind-mounted from the host NFS, but **NFS attribute caching** can delay visibility of changes by 30-60s or indefinitely. To guarantee immediate delivery, use the **Ray Jobs API** to copy the file inside the container:

```bash
# From any machine that can reach the Ray head node:
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import shutil; shutil.copy2(\"/path/on/nfs/plugin.sh\", \"/artifact_dir/utils/plugin.sh\")\"",
    "runtime_env": {}
  }'
```

The Ray job runs inside the container, bypassing NFS attribute cache issues.

**Case 2 — Exporter hotfix (metrics_exporter.sh):**

The exporter is a long-running process — overwriting the file alone does nothing until the process restarts. To deploy:
1. Deliver the updated `metrics_exporter.sh` to `<artifact_dir>/utils/` using the Ray Jobs API (same as above).
2. Kill the running exporter process. The watchdog detects the exit and restarts it with the updated code.

```bash
# Kill the exporter (via Ray Jobs API, targeting a specific node):
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess, os; subprocess.run([\\\"pkill\\\", \\\"-f\\\", \\\"metrics_exporter.sh\\\"])\"",
    "runtime_env": {}
  }'
```

**Pitfalls:**
- **Port conflict on restart:** When the exporter is killed, its HTTP server child process may linger briefly, holding port 9400. The watchdog restarts the exporter immediately, and the new instance gets `OSError: [Errno 98] Address already in use`. The watchdog will retry on the next crash cycle. If this persists, kill the orphaned HTTP server process explicitly before restarting.
- **Self-termination of Ray jobs:** When using `pkill -f <pattern>` inside a Ray job, the Ray job's own entrypoint command string may match the pattern, causing the job to kill itself. Use narrow patterns or filter out the current PID: `pkill -f "metrics_exporter" --signal TERM` with a pattern that won't match the Ray entrypoint string.
- **Multi-node deployment:** Ray jobs run on the head node by default. To target a specific worker node, you need to submit a job that explicitly connects to that node's Ray worker, or submit separate jobs per node. For bulk deployment across all nodes, iterate over the host list.
- **Watchdog restart logging:** Each restart is logged with a timestamp and restart count in `<job_dir>/metrics_exporter/<hostname>.log` (e.g., `[Metrics Exporter] Watchdog: exited with code 137, restart #3`). Monitor this log to confirm the restart succeeded.

**Case 3 — Remote process inspection and management:**

For any ad-hoc operation inside a container (checking process state, reading dmesg, killing specific processes), use the Ray Jobs API (see "Remote Execution via Ray Jobs API" section above). Common diagnostic commands:

```bash
# Check for D-state processes on the cluster
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; r=subprocess.run([\\\"ps\\\",\\\"axo\\\",\\\"pid,stat,wchan:30,cmd\\\"], capture_output=True, text=True); d=[l for l in r.stdout.splitlines() if l.split()[1:2] and l.split()[1][0]==\\\"D\\\"]; print(f\\\"D-state: {len(d)}\\\"); [print(l.strip()) for l in d[:20]]\"",
    "runtime_env": {}
  }'

# Read GPU-related dmesg errors
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; r=subprocess.run([\\\"dmesg\\\",\\\"--level=err,warn\\\",\\\"-T\\\"], capture_output=True, text=True); lines=[l for l in r.stdout.splitlines() if \\\"amdgpu\\\" in l.lower() or \\\"drm\\\" in l.lower() or \\\"cper\\\" in l.lower()]; print(f\\\"GPU dmesg lines: {len(lines)}\\\"); [print(l) for l in lines[-30:]]\"",
    "runtime_env": {}
  }'

# Get kernel and driver version
curl -s -X POST 'http://<head_host>:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; print(subprocess.run([\\\"uname\\\",\\\"-r\\\"], capture_output=True, text=True).stdout.strip()); print(open(\\\"/sys/module/amdgpu/version\\\").read().strip())\"",
    "runtime_env": {}
  }'
```

### D-state Process Accumulation from Plugin Timeouts

When a plugin (typically `gpu_metrics_plugin.sh`) enters kernel D-state (uninterruptible sleep), the exporter's timeout mechanism kills the plugin's bash wrapper process with `SIGKILL`. However, the actual child process stuck in D-state **cannot be killed by any signal** — it remains as an unkillable zombie consuming a PID and kernel task_struct memory.

Over time, these accumulate. At ~6 per hour (one per timeout cycle that hits D-state), a node can accumulate hundreds. While Linux's default PID max (4,194,304) provides ample headroom, a large accumulation indicates a persistent kernel-level fault.

**Detection:**
```promql
# Intermittent hw_scrape_duration_seconds alternating between ~12s (timeout)
# and <3s (success) indicates the plugin hangs intermittently.
# Full timeout (all cycles >=12s) means the plugin hangs every time.
hw_scrape_duration_seconds{host="<suspect_host>"}
```

**Confirmation via Ray Jobs API** (see "Remote Execution via Ray Jobs API" and "Case 3 — Remote process inspection" for full curl examples):
```bash
# Submit a Ray job to count D-state processes and show their kernel wait channel:
# amdgpu_cper_ring_write = GPU driver CPER ring bug
# amdgpu_ras_* = RAS sysfs read hang
# Use: ps axo pid,stat,wchan:30,cmd | filter for D-state
# Also submit: uname -r and cat /sys/module/amdgpu/version for the admin report
```

**Resolution:** D-state processes cannot be cleared without a **node reboot**. They are stuck in a kernel code path that will never return. Schedule a reboot at the next job boundary or maintenance window.
