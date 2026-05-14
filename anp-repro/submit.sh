#!/bin/bash
# User-facing entry point.  Submits up to THREE Slurm jobs: noANP, ANP, AROCE.
# After they complete, call ./parse.sh <noANP_dir> <ANP_dir> [<AROCE_dir>] to diff.
#
# Usage (from a login node):
#   ./submit.sh                         # default: 3-way on k8s 2-node R72.1
#   NODELIST=chi[X,Y] PARTITION=pp ./submit.sh
#   WHICH=anp ./submit.sh               # only submit ANP variant
#   WHICH=aroce ./submit.sh             # only submit AROCE variant
#   DOCKER_IMAGE=/path/to/img ./submit.sh
set -eo pipefail

REPRO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$REPRO_SCRIPT_DIR"
export REPRO_SCRIPT_DIR

NODELIST="${NODELIST:-chi[2766,2798]}"
PARTITION="${PARTITION:-k8s}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.2_jax_0.8.2_20260427}"
WHICH="${WHICH:-all3}"   # all3 | noanp | anp | aroce | both (legacy: noanp+anp)
export DOCKER_IMAGE REPRO_OPS REPRO_SIZES REPRO_ITERS REPRO_WARMUP

submit_one() {
    local mode="$1"   # noanp | anp | aroce
    local with_anp=0
    local with_aroce=0
    local tag
    case "$mode" in
        noanp) tag=noANP ;;
        anp)   with_anp=1; tag=ANP ;;
        aroce) with_aroce=1; tag=AROCE ;;
        *) echo "[submit] unknown mode: $mode" >&2; exit 1 ;;
    esac
    local jobname="anp-repro-$tag"
    echo "[submit] sbatch $jobname  (WITH_ANP=$with_anp WITH_AROCE=$with_aroce  partition=$PARTITION  nodes=$NODELIST)"
    sbatch \
        --partition="$PARTITION" \
        --nodelist="$NODELIST" \
        --job-name="$jobname" \
        --output="$SCRIPT_DIR/runs/slurm-%j-$tag.log" \
        --export=ALL,WITH_ANP="$with_anp",WITH_AROCE="$with_aroce" \
        "$SCRIPT_DIR/_sbatch.sh"
}

mkdir -p "$SCRIPT_DIR/runs"

case "$WHICH" in
    all3)
        submit_one noanp
        submit_one anp
        submit_one aroce
        ;;
    both)
        submit_one noanp
        submit_one anp
        ;;
    noanp) submit_one noanp ;;
    anp)   submit_one anp ;;
    aroce) submit_one aroce ;;
    *) echo "WHICH must be one of: all3, both, noanp, anp, aroce"; exit 1 ;;
esac

echo ""
echo "[submit] Done. Watch with:  squeue --me"
echo "[submit] Outputs will appear under $SCRIPT_DIR/runs/"
echo "[submit] After both finish, compare with:"
echo "           $SCRIPT_DIR/parse.sh $SCRIPT_DIR/runs/<noANP_dir> $SCRIPT_DIR/runs/<ANP_dir>"
