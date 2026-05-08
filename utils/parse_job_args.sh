#!/bin/bash

# parse_job_args.sh — Parse model spec, resolve model name, build JOB_NAME
#
# Sourced (not executed) by callers. Expects:
#   SCRIPT_DIR — path to the repo root (set by caller)
#   $@         — caller's positional args (inherited via source)
#
# After sourcing, callers have access to:
#   MODEL_NAME, MODEL_NAME_ALIAS, EXP_TAG
#   SBATCH_ARGS, PASSTHROUGH_ARGS, SCRIPT_ARGS
#   JOB_NAME

source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"

source "$SCRIPT_DIR/utils/parse_model_spec.sh"
parse_model_spec

source "$SCRIPT_DIR/utils/resolve_model_name.sh"
if ! MODEL_NAME=$(resolve_model_name "$SCRIPT_DIR/configs" "$MODEL_NAME"); then
    echo "!!! Model name resolution failed. Exiting..." >&2
    exit 1
fi

JOB_NAME="JAX-${MODEL_NAME}${EXP_TAG:+-$EXP_TAG}"

# Cap JOB_NAME so that submit.sh's `<jobid>-<JOB_NAME>.log` output filename
# stays within the filesystem NAME_MAX (255 bytes on ext4/xfs). Slurm jobids
# are up to 8 digits + ".log" + "-" → reserve ~14 bytes; cap to 200 keeps a
# safe margin. When truncating, append a short md5 hash so unique configs
# still produce unique log files / job dirs.
_JOB_NAME_MAX=200
if (( ${#JOB_NAME} > _JOB_NAME_MAX )); then
    _hash=$(printf '%s' "$JOB_NAME" | md5sum | head -c 8)
    JOB_NAME="${JOB_NAME:0:$((_JOB_NAME_MAX - 9))}-${_hash}"
fi
unset _JOB_NAME_MAX _hash
