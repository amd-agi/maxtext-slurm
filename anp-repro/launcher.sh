#!/bin/bash
# Runs inside the Docker container on ONE compute node.
# Env expected:
#   NODE_RANK     0 or 1 (from Slurm SLURM_NODEID)
#   WORLD_SIZE    16 (total ranks across both nodes)
#   LOCAL_WORLD   8  (GPUs per node; we launch this many processes here)
#   UID_FILE      shared-filesystem path for ncclUniqueId exchange
#   WITH_ANP      0 or 1 (just for logging; actual plugin toggle is via
#                 NCCL_NET_PLUGIN env inherited from the outer script)
#   JOB_ID        tag for the UID file
#   REPRO_BIN     absolute path inside the container to ./anp_repro
#   REPRO_OPS, REPRO_SIZES, REPRO_ITERS, REPRO_WARMUP   optional passthrough
set -eo pipefail

: "${NODE_RANK:?NODE_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD:?LOCAL_WORLD not set}"
: "${UID_FILE:?UID_FILE not set}"
: "${REPRO_BIN:?REPRO_BIN not set}"

echo "[launcher] node_rank=$NODE_RANK  world=$WORLD_SIZE  local_world=$LOCAL_WORLD  uid_file=$UID_FILE  plugin=${NCCL_NET_PLUGIN:-(none)}"

# Clean up any stale UID file on the writer side
if [[ "$NODE_RANK" == "0" ]]; then
    rm -f "$UID_FILE" "$UID_FILE.tmp"
fi

declare -a PIDS
for LOCAL_RANK in $(seq 0 $((LOCAL_WORLD - 1))); do
    GLOBAL_RANK=$((NODE_RANK * LOCAL_WORLD + LOCAL_RANK))
    LOG="/outputs/rank_${GLOBAL_RANK}.log"
    (
        export GLOBAL_RANK
        export LOCAL_RANK
        export WORLD_SIZE
        export UID_FILE
        # Let every process see all 8 GPUs; anp_repro calls hipSetDevice(LOCAL_RANK)
        # internally. Setting HIP_VISIBLE_DEVICES=$LOCAL_RANK would renumber the
        # visible GPU to index 0, breaking the hipSetDevice call.
        # Every rank sees the same NCCL env; the ANP plugin toggle is via
        # NCCL_NET_PLUGIN which the outer script set (or didn't).
        exec "$REPRO_BIN" >"$LOG" 2>&1
    ) &
    PIDS[$LOCAL_RANK]=$!
done

echo "[launcher] Launched ${#PIDS[@]} ranks on node $NODE_RANK, pids=${PIDS[*]}"

fail=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[launcher] local rank $i (pid ${PIDS[$i]}) failed"
        fail=1
    fi
done

# Rank-0-node dumps its rank 0 log to stdout for Slurm capture
if [[ "$NODE_RANK" == "0" ]]; then
    echo "==================== rank 0 log ===================="
    cat /outputs/rank_0.log || true
fi

exit $fail
