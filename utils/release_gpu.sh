#!/bin/bash

# release_gpu.sh - Stop containers and kill GPU processes with retries.
#
# Two modes:
#   Full (no args):       Stop ALL running containers, kill GPU processes,
#                         wait for GPU memory release.  Used by preflight.
#   Targeted (--container NAME):  Kill a specific container immediately
#                         (SIGKILL, no graceful stop).  Used by scancel
#                         trap where KillWait budget is tight.
#
# Device-agnostic: detects AMD (rocm-smi) or NVIDIA (nvidia-smi) GPUs.
#
# Zombie handling:  Zombie processes (Z state) can't be killed — they're
# already dead but hold GPU memory until the kernel reaps them.  We handle
# this by (1) killing the zombie's parent so init adopts and reaps it,
# and (2) giving the container runtime enough time after docker stop to
# fully tear down PID namespaces and release KFD/GPU memory.

# ── Argument parsing ─────────────────────────────────────────────────────────

TARGET_CONTAINER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) TARGET_CONTAINER="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Docker binary ────────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/docker_utils.sh"
DOCKER_BIN="$(get_docker_bin)" || { echo "ERROR: no docker/podman found" >&2; exit 1; }
read -ra DOCKER_CMD <<< "$DOCKER_BIN"

# ── Configuration ────────────────────────────────────────────────────────────

STOP_TIMEOUT="${STOP_TIMEOUT:-20}"  # seconds to wait for graceful container stop
SETTLE_WAIT="${SETTLE_WAIT:-40}"    # seconds after container stop for runtime/KFD cleanup
KILL_WAIT=5                        # seconds between SIGTERM and SIGKILL
POLL_INTERVAL=20                   # seconds between GPU-free checks
MAX_RETRIES="${MAX_RETRIES:-10}"

# ── GPU process detection (device-agnostic) ──────────────────────────────────

_get_gpu_pids() {
    local pids=""
    # AMD: rocm-smi --showpids lists PIDs at the start of each line
    if command -v rocm-smi &>/dev/null; then
        pids+=" $(rocm-smi --showpids 2>/dev/null | grep -oE '^[0-9]+' | grep -v '^$')"
    fi
    # NVIDIA: nvidia-smi --query-compute-apps=pid --format=csv,noheader
    if command -v nvidia-smi &>/dev/null; then
        pids+=" $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    fi
    echo "$pids" | xargs  # deduplicate whitespace
}

# ── Zombie detection and parent-kill ─────────────────────────────────────────

_is_zombie() {
    local pid="$1"
    [[ "$(ps -p "$pid" -o stat= 2>/dev/null)" == Z* ]]
}

_kill_zombie_parent() {
    # Kill the parent of a zombie so init adopts and reaps it, releasing
    # GPU memory held by the KFD driver.
    local pid="$1"
    local ppid
    ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
    if [[ -n "$ppid" && "$ppid" != "1" && "$ppid" != "0" ]]; then
        echo "  PID $pid is zombie (parent=$ppid) — killing parent"
        kill -9 "$ppid" 2>/dev/null || sudo kill -9 "$ppid" 2>/dev/null || true
    elif [[ "$ppid" == "1" ]]; then
        # Parent is init — zombie should be reaped shortly; just wait.
        echo "  PID $pid is zombie (parent=init) — waiting for reap"
    fi
}

# ── GPU status on exit (device-agnostic, full mode only) ─────────────────────

if [[ -z "$TARGET_CONTAINER" ]]; then
    _show_gpu_status() {
        echo "--- GPU status after cleanup ---"
        command -v rocm-smi  &>/dev/null && { rocm-smi --showpids 2>&1 || true; }
        command -v amd-smi   &>/dev/null && { amd-smi 2>&1 || true; }
        command -v nvidia-smi &>/dev/null && { nvidia-smi 2>&1 || true; }
    }
    trap _show_gpu_status EXIT
fi

# =============================================================================
# Targeted mode: kill a specific container (scancel / post-flight)
# =============================================================================
# GPU processes ignore SIGTERM (deep in KFD/CUDA), so docker stop -t N wastes
# N seconds waiting.  Go straight to SIGKILL via docker kill, then rm -f.
# GPU memory release is the next job's preflight responsibility.
# Budget: kill (~1 s) + rm -f (~1 s) ≈ 2 s, well within Slurm's KillWait.

if [[ -n "$TARGET_CONTAINER" ]]; then
    "${DOCKER_CMD[@]}" kill "$TARGET_CONTAINER" 2>&1 || true
    "${DOCKER_CMD[@]}" rm -f "$TARGET_CONTAINER" 2>&1 || true
    exit 0
fi

# =============================================================================
# Full mode: stop all containers + clean GPU processes (preflight)
# =============================================================================

echo "=== Pre-flight GPU cleanup ==="

START_TIME=$(date +%s)

# ── Phase 0: Stop all running containers (with retries) ──────────────────────
# Safe: nodes are --exclusive (sole access), so any running container is stale.
# This catches training containers (GPU) AND coordination containers (Ray/CPU)
# that the GPU-PID check alone would miss.
#
# We retry docker stop because a single attempt may not fully tear down the
# container's PID namespace and KFD device references.  The SETTLE_WAIT after
# each attempt gives the container runtime + KFD driver time to release GPU
# memory — critical for zombie processes that hold ~255 GB/GPU.

mapfile -t STALE < <("${DOCKER_CMD[@]}" ps -q 2>/dev/null)
if [[ ${#STALE[@]} -gt 0 && -n "${STALE[0]}" ]]; then
    echo "[container] Stopping ${#STALE[@]} container(s): ${STALE[*]}"
    retry=0
    while true; do
        "${DOCKER_CMD[@]}" stop -t "$STOP_TIMEOUT" "${STALE[@]}" 2>/dev/null || true

        # Give the container runtime time to fully tear down PID namespaces
        # and release KFD/GPU memory.  Without this sleep, zombie processes
        # from the previous job survive and hold GPU memory indefinitely.
        sleep "$SETTLE_WAIT"

        remaining=$("${DOCKER_CMD[@]}" ps -q 2>/dev/null)
        if [[ -z "$remaining" ]]; then
            echo "[container] All containers stopped"
            break
        fi

        retry=$((retry + 1))
        if [[ $retry -ge $MAX_RETRIES ]]; then
            echo "[container] WARNING: containers still running after $MAX_RETRIES retries: $remaining"
            break
        fi
        echo "[container] Retry $retry/$MAX_RETRIES — still running: $remaining"
        mapfile -t STALE <<< "$remaining"
    done
fi

# ── Phase 1: Detect remaining GPU processes ──────────────────────────────────

GPU_PIDS=$(_get_gpu_pids)
if [[ -z "$GPU_PIDS" ]]; then
    echo "No GPU processes found - GPUs are free"
    exit 0
fi

echo "Found GPU processes after container stop: $GPU_PIDS"

# ── Phase 2: Kill remaining GPU processes ────────────────────────────────────
# Check each PID: zombies need parent-kill, live processes get SIGTERM→SIGKILL.

for PID in $GPU_PIDS; do
    if _is_zombie "$PID"; then
        _kill_zombie_parent "$PID"
    else
        kill -TERM "$PID" 2>/dev/null || sudo kill -TERM "$PID" 2>/dev/null || true
    fi
done

sleep "$KILL_WAIT"

# Re-read: any survivors get SIGKILL (skip zombies — already handled above)
GPU_PIDS=$(_get_gpu_pids)
if [[ -n "$GPU_PIDS" ]]; then
    echo "SIGTERM didn't clear all processes, sending SIGKILL: $GPU_PIDS"
    for PID in $GPU_PIDS; do
        if _is_zombie "$PID"; then
            _kill_zombie_parent "$PID"
        else
            kill -9 "$PID" 2>/dev/null || sudo kill -9 "$PID" 2>/dev/null || true
        fi
    done
fi

# ── Phase 3: Wait for GPUs to be truly freed ─────────────────────────────────

echo "Waiting for GPUs to be freed..."
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    sleep "$POLL_INTERVAL"
    REMAINING=$(_get_gpu_pids)

    if [[ -z "$REMAINING" ]]; then
        END_TIME=$(date +%s)
        echo "GPU cleanup successful - GPUs are now free (elapsed: $((END_TIME - START_TIME))s)"
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Retry $RETRY_COUNT/$MAX_RETRIES - GPUs still occupied: $REMAINING"

    # On each retry, re-check for zombies and kill their parents
    for PID in $REMAINING; do
        if _is_zombie "$PID"; then
            _kill_zombie_parent "$PID"
        fi
    done
done

# ── Cleanup failed ───────────────────────────────────────────────────────────

END_TIME=$(date +%s)
echo "ERROR: Failed to free GPUs (elapsed: $((END_TIME - START_TIME))s)"
echo "Remaining GPU processes:"
for PID in $REMAINING; do
    echo "  PID $PID:"
    ps -p "$PID" -o pid,user,stat,ppid,cmd 2>/dev/null || echo "    (process info unavailable)"
done

exit 1
