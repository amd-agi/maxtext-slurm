#!/bin/bash

# Training environment configuration.
# Edit this file to tune XLA, NCCL, ROCm, and other runtime settings.
#
# Sourced by _train.sh before launching training.
# Per-run overrides: pass _env_KEY=VALUE after -- in submit.sh.

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
# Scope dump to global rank 0 only to avoid concurrent writers racing on the
# same xla_dump/ directory (all SPMD ranks compile identical HLO — rank 0's
# dump is representative and keeps xla_dump/ a flat dir for downstream tools
# like analyze_job.py and IRLens).
ENABLE_XLA_DUMP="${ENABLE_XLA_DUMP:-${EXTRACTED_ENV_MAP[ENABLE_XLA_DUMP]:-0}}"
if [[ "${ENABLE_XLA_DUMP,,}" =~ ^(1|y|yes|true)$ ]]; then
    _dump_rank="${GLOBAL_RANK:-${NODE_RANK:-0}}"
    if [[ "$_dump_rank" == "0" ]]; then
        echo "[XLA dump] Enabled on rank 0 (ENABLE_XLA_DUMP=$ENABLE_XLA_DUMP)"
        XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_hlo_as_text"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_module_re=^jit_train_step$"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_pipeline_re='(?i)gpu'"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_to=${OUTPUT_PATH}/xla_dump"
        export XLA_FLAGS
        echo "[XLA dump] XLA_FLAGS=$XLA_FLAGS"
    fi
    unset _dump_rank
fi

# ---- Disable XLA's in-process one-shot ragged-all-to-all kernel (default OFF) ----
# Controls --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel.
# Default 0 (kernel disabled) → we append --...=false, so XLA's ragged thunk falls
# back to its kNccl lowering — the same runtime path 1-GPU/proc gets automatically.
# For sparse MoE (sparse_matmul=true use_turbo_grouped_gemm=true) on 1-node/proc
# this is a ~3x TGS speedup at equal HBM budget; verified no-op on dense configs,
# on sparse-gmm-deepep, and on 1-GPU/proc.
# Set _env_ENABLE_RAGGED_ONESHOT_KERNEL=1 to restore XLA's one-shot kernel (debug only).
# Appends to XLA_FLAGS so the image's default tuning flags are preserved.
ENABLE_RAGGED_ONESHOT_KERNEL="${ENABLE_RAGGED_ONESHOT_KERNEL:-${EXTRACTED_ENV_MAP[ENABLE_RAGGED_ONESHOT_KERNEL]:-0}}"
if [[ "${ENABLE_RAGGED_ONESHOT_KERNEL,,}" =~ ^(0|n|no|false)$ ]]; then
    echo "[ENABLE_RAGGED_ONESHOT_KERNEL=0] Disabling --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel"
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false"
    export XLA_FLAGS
fi

# ---- Fix for JAX-0.8.2 ----
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=''"
export XLA_FLAGS

export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=WARN
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1  # For debugging if needed
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
_TRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_TRAIN_ENV_DIR/utils/detect_nccl_env.sh"
unset _TRAIN_ENV_DIR

# ---- Protocol and algorithm selection ----
#export NCCL_ALGO=Ring,Tree  # Hybrid algorithm selection
#export NCCL_PROTO=Simple  # Better for large messages in MoE

# ---- Buffer management ----
#export NCCL_BUFFSIZE=8388608  # 8MB buffers
# Larger buffer sizes for massive models (e.g. 300B+ parameters)
#export NCCL_BUFFSIZE=16777216  # 16MB

# ---- GPU compute settings ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPU_MAX_HW_QUEUES=2

# ---- AMD-specific optimizations ----
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1

# ---- rocSHMEM GDA settings ----
# The PyTorch Primus-Turbo internode DeepEP benchmark that passes on this
# cluster uses rocSHMEM GDA with the mlx5 provider.  Ablation showed that
# explicitly setting ROCSHMEM_GDA_PROVIDER=mlx5 is the key fix for:
#   Failed to lock memory pool ((nil)): 0x1001
# Keep the full set of successful benchmark defaults here, while preserving
# per-run _env_* overrides.
ROCSHMEM_MLX5_GDA_DEFAULTS="${ROCSHMEM_MLX5_GDA_DEFAULTS:-1}"
if [[ "${ROCSHMEM_MLX5_GDA_DEFAULTS,,}" =~ ^(1|y|yes|true)$ ]]; then
    if compgen -G "/sys/class/infiniband/mlx5_*" >/dev/null; then
        export ROCSHMEM_BACKEND="${ROCSHMEM_BACKEND:-gda}"
        export ROCSHMEM_GDA_PROVIDER="${ROCSHMEM_GDA_PROVIDER:-mlx5}"
        export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME="${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}}"
        export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-2147483648}"
        export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-64}"
        echo "[INFO] rocSHMEM mlx5/GDA defaults: ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND} ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER} ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME} ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE} ROCSHMEM_MAX_NUM_CONTEXTS=${ROCSHMEM_MAX_NUM_CONTEXTS}"
    fi
fi

# ---- Transformer Engine optimizations ----
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_FRAMEWORK=jax
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_FUSED_ATTN_CK=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_ROCM=1

# ---- Composable Kernel optimizations ----
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_CK_HOW_V3_BF16_CVT=2
# Forces FP32 precision for atomic accumulation in CK V3 GEMM output writes.
# Critical for MoE convergence: BF16 atomics (=0) cause visibly slower loss
# descent vs FP32 atomics (=1) due to accumulated rounding errors across many
# experts and layers. Use default value from the docker image (likely =1).
#export NVTE_CK_IS_V3_ATOMIC_FP32=1

# ---- Compilation cache settings ----
#export JAX_COMPILATION_CACHE_DIR="$OUTPUT_PATH/../jax_cache"
#export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

# ---- PGLE (Profile-Guided Layout Optimization) - uncomment after first run ----
#export JAX_ENABLE_PGLE=true
#export JAX_PGLE_AGGREGATION_PERCENTILE=90
#export JAX_PGLE_PROFILING_RUNS=5

if [[ -z "${IONIC_LOCKFREE:-}" ]]; then
    _TRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$_TRAIN_ENV_DIR/utils/detect_ainic_nccl_ib_tc.sh"
    if is_pensando; then
        export IONIC_LOCKFREE=all
    fi
    unset _TRAIN_ENV_DIR
fi

_kernel_config_has() {
    local option="$1"
    local kernel_release="$2"
    local config_file="/boot/config-${kernel_release}"

    if [[ -r "$config_file" ]]; then
        grep -q "^${option}=y" "$config_file"
        return $?
    fi

    if [[ -r /proc/config.gz ]] && command -v zgrep >/dev/null 2>&1; then
        zgrep -q "^${option}=y" /proc/config.gz
        return $?
    fi

    return 1
}

_kernel_supports_nccl_dmabuf() {
    local kernel_release
    kernel_release="$(uname -r 2>/dev/null || true)"
    [[ -n "$kernel_release" ]] || return 1
    _kernel_config_has CONFIG_DMABUF_MOVE_NOTIFY "$kernel_release" &&
        _kernel_config_has CONFIG_PCI_P2PDMA "$kernel_release"
}

# DMABUF default: auto. Enable only when the running kernel advertises the
# required P2P DMABUF options; otherwise keep it off to avoid known SIGSEGVs.
NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-auto}"
if [[ "${NCCL_DMABUF_ENABLE}" == "auto" ]]; then
    if _kernel_supports_nccl_dmabuf; then
        export NCCL_DMABUF_ENABLE=1
    else
        export NCCL_DMABUF_ENABLE=0
        echo "[INFO] NCCL_DMABUF_ENABLE=auto resolved to 0 (kernel lacks CONFIG_DMABUF_MOVE_NOTIFY/CONFIG_PCI_P2PDMA)."
    fi
elif [[ "${NCCL_DMABUF_ENABLE:-}" == "1" ]]; then
    if ! _kernel_supports_nccl_dmabuf; then
        echo "[WARN] NCCL_DMABUF_ENABLE=1 but the kernel lacks CONFIG_DMABUF_MOVE_NOTIFY/CONFIG_PCI_P2PDMA."
        echo "[WARN] Forcing NCCL_DMABUF_ENABLE=0."
        export NCCL_DMABUF_ENABLE=0
    fi
else
    export NCCL_DMABUF_ENABLE
fi
unset -f _kernel_config_has _kernel_supports_nccl_dmabuf

export NCCL_GDRCOPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_ECE_ENABLE=0
# NOTE: NCCL_IB_TC and NCCL_IB_FIFO_TC are auto-detected above (see utils/detect_nccl_env.sh).
[[ -n "${NCCL_IB_GID_INDEX:-}" ]] && export NCCL_IB_GID_INDEX
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_USE_INLINE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PXN_DISABLE=0
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=1
# RCCL 2.27.7's default MI300X MSCCL path fails on this cluster during
# ncclCommSplit with "number of channels available (28) less than required (32)".
export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
export RCCL_MSCCLPP_ENABLE="${RCCL_MSCCLPP_ENABLE:-0}"

#export HSA_DISABLE_CACHE=1
#export IB_PCI_RELAXED_ORDERING=1
#export NCCL_IB_QPS=2
#export NCCL_IB_SL=0
#export NCCL_IB_SPLIT_DATA_ON_QPS=0
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_OOB_NET_IFNAME=enp81s0f1.2026
#export NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt
#export UCX_LOG_LEVEL=INFO
