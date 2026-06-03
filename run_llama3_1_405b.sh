#!/usr/bin/env bash
set -euo pipefail

# One-command Llama-3.1 405B FP8 run on MI355X Slurm nodes.
#
# Defaults use the conservative 8-node setting proven in prior experiments:
# per_device_batch_size=3. To try the config-file default, run BATCH=5.
#
# Usage:
#   ./run_llama3_1_405b.sh
#   STEPS=1 ./run_llama3_1_405b.sh
#   BATCH=5 ./run_llama3_1_405b.sh
#   EXCLUDE=nodeA,nodeB ./run_llama3_1_405b.sh
#   ./run_llama3_1_405b.sh _env_NCCL_DEBUG=INFO _env_NCCL_DEBUG_SUBSYS=INIT,NET,ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DLC cluster
# JOB_WORKSPACE="${JOB_WORKSPACE:-/perf_apps/xuefjian_maxtext}"
# PARTITION="${PARTITION:-Compute-DCPT}"
# Vultr cluster
JOB_WORKSPACE="${JOB_WORKSPACE:-/mnt/vast/xuefei_maxtext}"
PARTITION="${PARTITION:-k8s}"
NODES="${NODES:-8}"
TAG="${TAG:-run8n}"
STEPS="${STEPS:-15}"
BATCH="${BATCH:-3}"
HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-3600}"
NODELIST="${NODELIST:-}"
EXCLUDE="${EXCLUDE:-}"

SBATCH_ARGS=(-N "$NODES" -p "$PARTITION")
if [[ -n "$NODELIST" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
elif [[ -n "$EXCLUDE" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

PASSTHROUGH=(
  "steps=$STEPS"
  "per_device_batch_size=$BATCH"
  "_env_jax_distributed_heartbeat_timeout_seconds=$HEARTBEAT_TIMEOUT"
)

exec env JOB_WORKSPACE="$JOB_WORKSPACE" \
  "$SCRIPT_DIR/submit.sh" "llama3.1-405b:$TAG" \
  "${SBATCH_ARGS[@]}" \
  -- "${PASSTHROUGH[@]}" "$@"
