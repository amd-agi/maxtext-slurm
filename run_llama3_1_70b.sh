#!/usr/bin/env bash
set -euo pipefail

# One-command Llama-3.1 70B FP8 run on MI355X Slurm nodes.
#
# Usage:
#   ./run_llama3_1_70b.sh
#   STEPS=5 ./run_llama3_1_70b.sh
#   EXCLUDE=nodeA,nodeB ./run_llama3_1_70b.sh
#   ./run_llama3_1_70b.sh _env_NCCL_DEBUG=INFO _env_NCCL_DEBUG_SUBSYS=INIT,NET,ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_WORKSPACE="${JOB_WORKSPACE:-/perf_apps/xuefjian_maxtext}"
PARTITION="${PARTITION:-Compute-DCPT}"
NODES="${NODES:-4}"
TAG="${TAG:-run4n}"
STEPS="${STEPS:-50}"
BATCH="${BATCH:-6}"
HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-1800}"
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
  "$SCRIPT_DIR/submit.sh" "llama3.1-70b:$TAG" \
  "${SBATCH_ARGS[@]}" \
  -- "${PASSTHROUGH[@]}" "$@"
