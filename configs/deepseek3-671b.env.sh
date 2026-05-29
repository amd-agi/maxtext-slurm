# Per-model environment overrides for DeepSeek-V3 671B (`deepseek3-671b`).
# Sourced after train_env.sh, before CLI _env_ overrides.
#
# The optimal XLA collective flag is topology-dependent: the FSDP and
# pipeline-parallel (PP) regimes prefer OPPOSITE settings, so a single static
# flag cannot serve both.
#   - FSDP: a smaller all-gather combine threshold (1 GiB) splits the large
#           weight all-gather into chunks the latency-hiding scheduler can
#           overlap with compute. This setting regresses PP.
#   - PP:   keep the image-default threshold and raise the parallel-collective
#           overlap limit to 2 so two collectives run concurrently on the
#           independent ICI and DCN fabrics. For the dropless sparse_matmul /
#           DeepEP path only, also raise async-stream priority (it helps the
#           skewed sparse path but regresses the dense capacity-factor path, so
#           it is gated on sparse_matmul below).
#
# This file auto-selects the recipe per submission. It mirrors MaxText's own
# config resolution -- the gpu.yml value is the base, a CLI passthrough arg (in
# PASSTHROUGH_ARGS, populated by _train.sh before this is sourced) overrides it --
# so it is correct whether topology is set on the CLI OR edited into the gpu.yml.

# Effective value of a MaxText config key: gpu.yml base, overridden by a CLI arg.
_ds_yml="${BASH_SOURCE[0]%.env.sh}.gpu.yml"
_ds_cfg() {  # $1=key  $2=fallback-if-unset-everywhere
    local key="$1" val="" a
    if [[ -f "$_ds_yml" ]]; then
        val="$(sed -n "s/^[[:space:]]*${key}[[:space:]]*:[[:space:]]*\([^#]*\).*/\1/p" "$_ds_yml" | tail -n1)"
        val="${val%"${val##*[![:space:]]}"}"   # rstrip
    fi
    for a in "${PASSTHROUGH_ARGS[@]:-}"; do
        [[ "$a" == "${key}="* ]] && val="${a#*=}"
    done
    printf '%s' "${val:-$2}"
}

# Topology = max pipeline degree across DCN/ICI; >1 means PP.
_ds_pp=1
for _ds_k in dcn_pipeline_parallelism ici_pipeline_parallelism; do
    _ds_v="$(_ds_cfg "$_ds_k" 1)"
    [[ "$_ds_v" =~ ^[0-9]+$ && "$_ds_v" -gt "$_ds_pp" ]] && _ds_pp="$_ds_v"
done
# Branch = dropless sparse routing (ragged_dot / DeepEP), case-insensitive bools.
_ds_sparse=false
for _ds_k in sparse_matmul use_turbo_deepep_dispatch; do
    [[ "$(_ds_cfg "$_ds_k" false)" =~ ^[Tt][Rr][Uu][Ee]$ ]] && _ds_sparse=true
done

if [[ "${_ds_pp:-1}" -le 1 ]]; then
    # FSDP (no pipeline) path: smaller all-gather combine threshold lets the
    # scheduler overlap the weight all-gather with compute.
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_all_gather_combine_threshold_bytes=1073741824"
else
    # PP (pipeline) path: keep the image-default threshold; raise collective overlap to 2.
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    if [[ "$_ds_sparse" == "true" ]]; then
        # Dropless sparse_matmul / DeepEP only -- async-stream priority helps the
        # skewed sparse path; it regresses the dense capacity-factor path, so it is gated here.
        XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_highest_priority_async_stream=true"
    fi
fi
export XLA_FLAGS
unset -f _ds_cfg
unset _ds_yml _ds_pp _ds_sparse _ds_k _ds_v
