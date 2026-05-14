#!/bin/bash
#SBATCH --job-name=anp-repro
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --exclusive
set -eo pipefail

# SCRIPT_DIR must be passed in from submit.sh — sbatch copies _sbatch.sh to
# /var/spool/slurmd/, so BASH_SOURCE here doesn't point back to the original
# repo checkout. submit.sh exports REPRO_SCRIPT_DIR for us.
SCRIPT_DIR="${REPRO_SCRIPT_DIR:?REPRO_SCRIPT_DIR must be set by submit.sh}"

export JOB_ID="${SLURM_JOB_ID:-local_$$}"
WITH_AROCE="${WITH_AROCE:-0}"
if   [[ "$WITH_AROCE" == "1" ]]; then TAG=AROCE
elif [[ "$WITH_ANP"   == "1" ]]; then TAG=ANP
else                                  TAG=noANP
fi
export OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/runs/$JOB_ID-$TAG}"
export UID_FILE="$OUT_DIR/ncclUniqueId.$JOB_ID"
mkdir -p "$OUT_DIR"

echo "[sbatch] JOB_ID=$JOB_ID  WITH_ANP=${WITH_ANP}  WITH_AROCE=${WITH_AROCE}  OUT_DIR=$OUT_DIR  NODES=$SLURM_JOB_NODELIST"
echo "[sbatch] IMAGE=${DOCKER_IMAGE:-rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.2_jax_0.8.2_20260427}"

# Run once per node (ntasks-per-node=1). NODE_RANK comes from SLURM_NODEID,
# which srun exports into every task's environment.
export JOB_ID WITH_ANP WITH_AROCE OUT_DIR UID_FILE DOCKER_IMAGE
export REPRO_OPS="${REPRO_OPS:-ag,rs,ar}"
export REPRO_SIZES="${REPRO_SIZES:-}"
export REPRO_ITERS="${REPRO_ITERS:-30}"
export REPRO_WARMUP="${REPRO_WARMUP:-5}"

srun --kill-on-bad-exit=1 --output="$OUT_DIR/node-%n.log" \
    bash -c "NODE_RANK=\$SLURM_NODEID bash \"$SCRIPT_DIR/run_node.sh\""

echo "[sbatch] done  OUT_DIR=$OUT_DIR"
