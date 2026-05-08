#!/bin/bash

# NVIDIA-specific training environment configuration.
# Sourced by train_env.sh BEFORE common additions.
#
# Sets base XLA_FLAGS to match the AMD Docker image defaults so that
# cross-platform benchmark comparisons start from an identical XLA baseline.
# Common flags (command_buffer, XLA dump, etc.) are appended by train_env.sh.

# ---- Base XLA_FLAGS (matching AMD image: rocm/jax-training:maxtext-v26.2) ----
# The NVIDIA JAX image only ships enable_latency_hiding_scheduler=true.
# We explicitly set the remaining flags that the AMD image provides by default.
#
# Override via _env_XLA_FLAGS_REPLACE='--flag1=v1,--flag2=v2' to replace this
# entire AMD-parity block (e.g. for "NV image defaults" experiments). The final
# fix-up flag (--xla_gpu_enable_command_buffer='') in train_env.sh runs AFTER
# this replacement, so include it in the replacement value if you want it kept.
if [[ -v 'EXTRACTED_ENV_MAP[XLA_FLAGS_REPLACE]' ]]; then
    XLA_FLAGS="${EXTRACTED_ENV_MAP[XLA_FLAGS_REPLACE]//,/ }"
    echo "[train_env.nvidia] XLA_FLAGS REPLACED via _env_XLA_FLAGS_REPLACE"
else
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }"
    XLA_FLAGS+="--xla_gpu_memory_limit_slop_factor=95 "
    XLA_FLAGS+="--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 "
    XLA_FLAGS+="--xla_gpu_all_gather_combine_threshold_bytes=8589934592 "
    XLA_FLAGS+="--xla_gpu_enable_triton_gemm=false "
    XLA_FLAGS+="--xla_gpu_enable_cublaslt=true "
    XLA_FLAGS+="--xla_gpu_autotune_level=0 "
    XLA_FLAGS+="--xla_gpu_enable_all_gather_combine_by_dim=false"
fi
export XLA_FLAGS

# ---- Transformer Engine (CUDA backend) ----
export NVTE_FRAMEWORK=jax
export NVTE_FUSED_ATTN=1

# ---- NVLink / NVSwitch (Blackwell and Hopper) ----
# NVLS (NVLink SHARP) enables in-network reductions over NVSwitch.
# Beneficial on DGX B200 / H100 systems with NVSwitch fabric.
#export NCCL_NVLS_ENABLE=1

# ---- IB Service Level ----
# NCCL_IB_SL sets the InfiniBand Service Level for QP traffic.
# Only use SL>0 when the fabric's Subnet Manager has QoS / SL-to-VL
# mapping configured (e.g. DGX-managed UFM clusters). On clusters
# without QoS, SL>0 maps to unconfigured Virtual Lanes, causing
# silent packet drops and IBV_WC_RETRY_EXC_ERR(12) on every rail.
#export NCCL_IB_SL=1

# ---- IB Partition Key fix ----
# On fabrics where the default P_Key (index 0) is 0x7fff (limited member),
# two limited-member endpoints cannot perform RDMA, causing
# IBV_WC_RETRY_EXC_ERR(12). NCCL_IB_PKEY is broken in NCCL 2.29.7 on
# native IB (it corrupts the GID selection via DOCA). Instead, use an
# LD_PRELOAD shim that patches ibv_modify_qp to swap pkey_index 0 for
# the first full-member index.
source "${BASH_SOURCE[0]%/*}/utils/ib_pkey_fix.sh"

# ---- NCCL tuning (NVIDIA-specific values) ----
# Most NCCL IB settings are in the common section of train_env.sh.
# Add NVIDIA-specific overrides here if needed.
#export NCCL_NET_GDR_LEVEL=5       # GPU Direct RDMA level for Mellanox IB
#export NCCL_IB_HCA=mlx5            # Explicit HCA selection (auto-detected by default)
