#!/bin/bash

# preflight.sh — Per-host pre-flight checks (run via srun).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=$(hostname -s)

# ---- Container + GPU cleanup ----
# Fail the job (via srun --kill-on-bad-exit=1) if the node still has stuck
# GPU processes after release_gpu.sh's retry budget. These are typically
# D-state procs leaked from a previous job and can only be cleared by an
# admin reboot — pressing on burns minutes of training before a silent
# rank death at the first collective. Better to fail in preflight and let
# the user resubmit with -x <bad node>.
if ! bash "$SCRIPT_DIR/utils/release_gpu.sh"; then
    echo "FATAL: $HOST | preflight: GPU cleanup failed (likely D-state procs from prior job)"
    echo "FATAL: $HOST |   -> ask cluster admin to reboot $HOST, or resubmit with -x $HOST"
    exit 1
fi

# ---- Disk cleanup ----
disk_before=$(df -BG / 2>/dev/null | awk 'NR==2 {print $4}')
# Stale temp files from previous jobs (older than 1 day, includes core dumps)
find /tmp -maxdepth 1 -user "$(id -u)" -mtime +1 -exec rm -rf {} + 2>/dev/null || true
find /var/tmp -maxdepth 1 -user "$(id -u)" -mtime +1 -exec rm -rf {} + 2>/dev/null || true
# Stopped containers and dangling (untagged) images — preserves cached tagged images
source "$SCRIPT_DIR/utils/docker_utils.sh"
if _DOCKER_BIN="$(get_docker_bin 2>/dev/null)"; then
    read -ra _DOCKER_CMD <<< "$_DOCKER_BIN"
    "${_DOCKER_CMD[@]}" container prune -f 2>/dev/null || true
    "${_DOCKER_CMD[@]}" image prune -f 2>/dev/null || true
fi
disk_after=$(df -BG / 2>/dev/null | awk 'NR==2 {print $4}')
echo "$HOST | disk free: $disk_before -> $disk_after (cleanup)"

# ---- NUMA auto-balancing ----
numa_before=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "N/A")
# Uncomment next 3 lines to disable NUMA auto-balancing:
#if [ "$numa_before" = "1" ]; then
#  echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1
#fi
numa_after=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "N/A")
echo "$HOST | numa_balancing: before=$numa_before after=$numa_after"

# ---- Leaked semaphores (safe: --exclusive gives us sole access to the node) ----
sem_before=$(ls -1 /dev/shm/sem.* 2>/dev/null | wc -l)
rm -f /dev/shm/sem.* 2>/dev/null
sem_after=$(ls -1 /dev/shm/sem.* 2>/dev/null | wc -l)
echo "$HOST | semaphores: before=$sem_before after=$sem_after deleted=$((sem_before - sem_after))"

# ---- Transparent Huge Pages ----
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)
echo "$HOST | THP before | enabled: $thp_enabled | defrag: $thp_defrag"
{ echo never > /sys/kernel/mm/transparent_hugepage/enabled && \
  echo never > /sys/kernel/mm/transparent_hugepage/defrag; } 2>/dev/null || \
{ echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled \
                         /sys/kernel/mm/transparent_hugepage/defrag > /dev/null 2>&1; }
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)
echo "$HOST | THP after  | enabled: $thp_enabled | defrag: $thp_defrag"


# ---- Network checks (multi-node only) ----
_num_nodes="${NNODES:-1}"
if [[ "$_num_nodes" -le 1 ]]; then
    echo "$HOST | [NETWORK_CHECKS] skipped (single-node job)"
else
    # ---- IPv6 routing rules check ----
    # Pensando/ionic RoCE clusters may need one fd-prefix IPv6 policy routing
    # rule per NIC. Other clusters often have no such rule, so the default is
    # observability-only. Set EXPECTED_IPV6_FD_RULES=8 (or another count) to
    # make this a fatal check.
    EXPECTED_IPV6_FD_RULES=${EXPECTED_IPV6_FD_RULES:-auto}
    _ip6_rules=$(ip -6 rule 2>/dev/null || sudo -n ip -6 rule 2>/dev/null || true)
    _ipv6_fd_rules=$(echo "$_ip6_rules" | grep -c fd) || _ipv6_fd_rules=0
    if [[ "$_ipv6_fd_rules" -eq 0 ]]; then
        echo "$HOST | [CHECK_IPV6_RULES] skipped (IPv4 cluster)"
    elif [[ "$EXPECTED_IPV6_FD_RULES" == "auto" ]]; then
        echo "$HOST | [CHECK_IPV6_RULES] observed $_ipv6_fd_rules fd-prefix rules (no expected count configured)"
    elif [[ "$EXPECTED_IPV6_FD_RULES" == "0" ]]; then
        echo "$HOST | [CHECK_IPV6_RULES] skipped by EXPECTED_IPV6_FD_RULES=0 (observed $_ipv6_fd_rules)"
    elif [[ "$_ipv6_fd_rules" -ne "$EXPECTED_IPV6_FD_RULES" ]]; then
        echo "FATAL: $HOST | [CHECK_IPV6_RULES] $_ipv6_fd_rules (expected $EXPECTED_IPV6_FD_RULES)"
        exit 1
    else
        echo "$HOST | [CHECK_IPV6_RULES] OK ($_ipv6_fd_rules)"
    fi
    unset _ip6_rules

    # ---- DCQCN congestion control check ----
    # DCQCN is a Pensando/ionic-specific operational check here. Keep it
    # observability-only unless the expected count is explicitly configured.
    EXPECTED_DCQCN_COUNT=${EXPECTED_DCQCN_COUNT:-auto}
    _dcqcn_out=$(nicctl show dcqcn 2>/dev/null || sudo -n nicctl show dcqcn 2>/dev/null || true)
    if [[ -z "$_dcqcn_out" ]]; then
        echo "$HOST | [CHECK_DCQCN] skipped (nicctl not available or no output)"
    else
        _dcqcn_enabled=$(echo "$_dcqcn_out" | grep -ic enable) || _dcqcn_enabled=0
        if [[ "$EXPECTED_DCQCN_COUNT" == "auto" ]]; then
            echo "$HOST | [CHECK_DCQCN] observed $_dcqcn_enabled enabled NICs (no expected count configured)"
        elif [[ "$EXPECTED_DCQCN_COUNT" == "0" ]]; then
            echo "$HOST | [CHECK_DCQCN] skipped by EXPECTED_DCQCN_COUNT=0 (observed $_dcqcn_enabled)"
        elif [[ "$_dcqcn_enabled" -ne "$EXPECTED_DCQCN_COUNT" ]]; then
            echo "WARNING: $HOST | [CHECK_DCQCN] $_dcqcn_enabled enabled (expected $EXPECTED_DCQCN_COUNT)"
            echo "WARNING: Training may experience network stalls without DCQCN on all NICs."
        else
            echo "$HOST | [CHECK_DCQCN] OK ($_dcqcn_enabled)"
        fi
    fi
fi
