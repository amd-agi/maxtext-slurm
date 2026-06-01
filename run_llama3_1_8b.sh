#!/usr/bin/env bash
set -euo pipefail

# One-command Llama-3.1 8B smoke run on MI355X Slurm nodes.
#
# Usage:
#   ./run_llama3_1_8b.sh
#   STEPS=5 ./run_llama3_1_8b.sh
#   NODELIST=smci355-ccs-aus-n01-21 ./run_llama3_1_8b.sh
#   ./run_llama3_1_8b.sh max_target_length=8192 _env_NCCL_DEBUG=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_WORKSPACE="${JOB_WORKSPACE:-/perf_apps/xuefjian_maxtext}"
PARTITION="${PARTITION:-Compute-DCPT}"
NODES="${NODES:-1}"
TAG="${TAG:-run}"
STEPS="${STEPS:-30}"
BATCH="${BATCH:-8}"
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
  "$SCRIPT_DIR/submit.sh" "llama3.1-8b:$TAG" \
  "${SBATCH_ARGS[@]}" \
  -- "${PASSTHROUGH[@]}" "$@"
