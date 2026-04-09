#!/bin/bash

# AMD-specific training environment configuration.
# Sourced by train_env.sh when AMD GPUs are detected (/dev/kfd present).

# ---- AMD GPU compute settings ----
export GPU_MAX_HW_QUEUES=2
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1  # For debugging if needed

# ---- AMD HIP / HSA runtime ----
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1

# ---- Transformer Engine (ROCm backend) ----
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_FRAMEWORK=jax
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_FUSED_ATTN_CK=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_ROCM=1

# ---- Composable Kernel (CK) optimizations ----
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_CK_HOW_V3_BF16_CVT=2
# Forces FP32 precision for atomic accumulation in CK V3 GEMM output writes.
# Critical for MoE convergence: BF16 atomics (=0) cause visibly slower loss
# descent vs FP32 atomics (=1) due to accumulated rounding errors across many
# experts and layers. Use default value from the docker image (likely =1).
#export NVTE_CK_IS_V3_ATOMIC_FP32=1

# ---- Pensando AINIC (AMD DLC clusters) ----
export IONIC_LOCKFREE=all

# ---- NCCL / RCCL advanced tuning (AMD-specific values) ----
export NCCL_IB_GID_INDEX=1             # RoCEv2 GID index (verify with show_gids)
export NCCL_GDRCOPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_ECE_ENABLE=0
export NCCL_IB_USE_INLINE=1
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=1
export RCCL_MSCCLPP_ENABLE=1

#export HSA_DISABLE_CACHE=1
#export IB_PCI_RELAXED_ORDERING=1
#export NCCL_IB_QPS=2
#export NCCL_IB_SL=0
#export NCCL_IB_SPLIT_DATA_ON_QPS=0
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_OOB_NET_IFNAME=enp81s0f1.2026
#export NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt
#export UCX_LOG_LEVEL=INFO
