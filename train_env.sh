#!/bin/bash

# Training environment configuration.
#
# Execution order:
#   1. Vendor-specific env (train_env.amd.sh / train_env.nvidia.sh)
#      — sets base XLA_FLAGS and vendor-specific vars (HIP/HSA, NVTE, etc.)
#      — AMD: base XLA_FLAGS come from the Docker image, .amd.sh adds non-XLA vars
#      — NVIDIA: .nvidia.sh explicitly sets base XLA_FLAGS to match AMD image defaults
#   2. Common additions below (appended on top of vendor base)
#
# Sourced by _train.sh before launching training.
# Per-run overrides: pass _env_KEY=VALUE after -- in submit.sh.

_TRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Vendor-specific environment (auto-detected) — MUST run first to set base XLA_FLAGS
# ============================================================================
if [[ -e /dev/kfd ]]; then
    echo "[train_env] GPU vendor: AMD (detected /dev/kfd)"
    source "$_TRAIN_ENV_DIR/train_env.amd.sh"
elif command -v nvidia-smi &>/dev/null; then
    echo "[train_env] GPU vendor: NVIDIA (detected nvidia-smi)"
    source "$_TRAIN_ENV_DIR/train_env.nvidia.sh"
else
    echo "[train_env] WARNING: No GPU vendor detected — no vendor-specific env loaded"
fi

# ============================================================================
# Common XLA_FLAGS additions
# ============================================================================

# NOTE: the entire build logic is commented out
#       to use the Docker image's default XLA_FLAGS!
: <<'BLOCK_COMMENT_TO_USE_DOCKER_IMAGE_DEFAULT_XLA_FLAGS'
# ---- Build XLA_FLAGS safely with clear structure ----
XLA_FLAGS=""

# === Core compiler and dump options ===
XLA_FLAGS+=" --xla_gpu_enable_cublaslt=true"
XLA_FLAGS+=" --xla_gpu_graph_level=0"
XLA_FLAGS+=" --xla_gpu_autotune_level=0"
# === GEMM and codegen behavior ===
XLA_FLAGS+=" --xla_gpu_enable_triton_gemm=false"
XLA_FLAGS+=" --xla_gpu_triton_gemm_any=false"
XLA_FLAGS+=" --xla_gpu_enable_command_buffer=''"   # Leave empty to disable explicit command buffer use
# === Collective combination / decomposition ===
XLA_FLAGS+=" --xla_gpu_enable_all_gather_combine_by_dim=false"
#XLA_FLAGS+=" --xla_gpu_enable_reduce_scatter_combine_by_dim=false"
#XLA_FLAGS+=" --xla_gpu_all_gather_combine_threshold_bytes=8589934592"   # Fix OOM for llama3.1-405b (dcn_fsdp=8, ici_fsdp=8)
#XLA_FLAGS+=" --xla_gpu_all_reduce_combine_threshold_bytes=1073741824"
#XLA_FLAGS+=" --xla_gpu_collective_permute_decomposer_threshold=1073741824"
#XLA_FLAGS+=" --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824"
# === Overlapping and pipelining ===
#XLA_FLAGS+=" --xla_gpu_enable_highest_priority_async_stream=true"
XLA_FLAGS+=" --xla_gpu_enable_latency_hiding_scheduler=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_all_gather=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_all_reduce=true"
#XLA_FLAGS+=" --xla_gpu_enable_pipelined_p2p=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_reduce_scatter=true"
#XLA_FLAGS+=" --xla_gpu_enable_while_loop_double_buffering=true"  # May cause OOM for llama3.1-405b (dcn_fsdp=8, ici_fsdp=8) even setting --xla_gpu_all_gather_combine_threshold_bytes=8589934592
#XLA_FLAGS+=" --xla_gpu_experimental_parallel_collective_overlap_limit=2"  # May conflict with latency-hiding scheduler (LHS=true)
# === Misc. ===
#XLA_FLAGS+=" --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=true"

# ---- Finalize and export XLA_FLAGS ----
export XLA_FLAGS
BLOCK_COMMENT_TO_USE_DOCKER_IMAGE_DEFAULT_XLA_FLAGS

# ---- XLA dump (enable via _env_ENABLE_XLA_DUMP=1 in PASSTHROUGH_ARGS) ----
ENABLE_XLA_DUMP="${ENABLE_XLA_DUMP:-${EXTRACTED_ENV_MAP[ENABLE_XLA_DUMP]:-0}}"
if [[ "${ENABLE_XLA_DUMP,,}" =~ ^(1|y|yes|true)$ ]]; then
    echo "[XLA dump] Enabled (ENABLE_XLA_DUMP=$ENABLE_XLA_DUMP)"
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_hlo_as_text"
    XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_module_re=^jit_train_step$"
    XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_pipeline_re='(?i)gpu'"
    XLA_FLAGS="$XLA_FLAGS --xla_dump_to=${OUTPUT_PATH}/xla_dump"
    export XLA_FLAGS
    echo "[XLA dump] XLA_FLAGS=$XLA_FLAGS"
fi

# ---- Fix for JAX-0.8.2 ----
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=''"
export XLA_FLAGS

# ---- Append extra XLA flags from _env_EXTRA_XLA_FLAGS=flag1,flag2,... ----
# Use commas to separate flags; they are converted to spaces here. Applied AFTER
# all base XLA_FLAGS construction so they reliably override or supplement.
if [[ -v 'EXTRACTED_ENV_MAP[EXTRA_XLA_FLAGS]' ]]; then
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }${EXTRACTED_ENV_MAP[EXTRA_XLA_FLAGS]//,/ }"
    export XLA_FLAGS
    echo "[train_env] XLA_FLAGS APPENDED via _env_EXTRA_XLA_FLAGS"
fi

# ============================================================================
# Common NCCL / runtime settings
# ============================================================================
export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=2

# ---- Memory fraction ----
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93

export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=512

# ---- Multi-rail network optimization ----
#export NCCL_CROSS_NIC=2  # For multi-rail networks
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=8

# ---- InfiniBand tuning ----
export NCCL_IB_QPS_PER_CONNECTION=4
#export NCCL_IB_RETRY_CNT=7
#export NCCL_IB_TIMEOUT=23

# ---- Auto-detected NCCL network settings (IB HCA, QoS, socket interface) ----
source "$_TRAIN_ENV_DIR/utils/detect_nccl_env.sh"

# ---- Protocol and algorithm selection ----
#export NCCL_ALGO=Ring,Tree  # Hybrid algorithm selection
#export NCCL_PROTO=Simple  # Better for large messages in MoE

# ---- Buffer management ----
#export NCCL_BUFFSIZE=8388608  # 8MB buffers
# Larger buffer sizes for massive models (e.g. 300B+ parameters)
#export NCCL_BUFFSIZE=16777216  # 16MB

# ---- GPU compute settings ----
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- Compilation cache settings ----
#export JAX_COMPILATION_CACHE_DIR="$OUTPUT_PATH/../jax_cache"
#export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

# ---- PGLE (Profile-Guided Layout Optimization) - uncomment after first run ----
#export JAX_ENABLE_PGLE=true
#export JAX_PGLE_AGGREGATION_PERCENTILE=90
#export JAX_PGLE_PROFILING_RUNS=5

# DMABUF default: enabled for performance, with runtime safety fallback below.
# If /boot kernel metadata is unavailable in the container, this file
# automatically forces NCCL_DMABUF_ENABLE=0 to avoid known SIGSEGV cases.
export NCCL_DMABUF_ENABLE=1
# Safety guard for direct sourcing and non-container launch paths.
if [[ "${NCCL_DMABUF_ENABLE:-}" == "1" ]]; then
    _kernel_release="$(uname -r 2>/dev/null || true)"
    _has_boot_kernel_metadata=false
    if [[ -n "$_kernel_release" && -d /boot ]] && compgen -G "/boot/*${_kernel_release}*" >/dev/null; then
        _has_boot_kernel_metadata=true
    fi
    if [[ "$_has_boot_kernel_metadata" != "true" ]]; then
        echo "[WARN] NCCL_DMABUF_ENABLE=1 but /boot lacks host kernel metadata for kernel '$_kernel_release'."
        echo "[WARN] Forcing NCCL_DMABUF_ENABLE=0 (mount /boot read-only to keep DMABUF enabled)."
        export NCCL_DMABUF_ENABLE=0
    fi
fi

# ---- Common NCCL IB settings ----
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PXN_DISABLE=0

unset _TRAIN_ENV_DIR
