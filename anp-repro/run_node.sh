#!/bin/bash
# Runs on one compute node (spawned by srun). Starts a single Docker
# container that owns all 8 local GPUs, then invokes launcher.sh inside.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${JOB_ID:?JOB_ID not set}"
: "${WITH_ANP:?WITH_ANP not set (0 or 1)}"
: "${NODE_RANK:?NODE_RANK not set}"
: "${OUT_DIR:?OUT_DIR not set}"
WITH_AROCE="${WITH_AROCE:-0}"

IMAGE="${DOCKER_IMAGE:-rocm/pyt-megatron-lm-jax-nightly-private:jax_rocm7.2_jax_0.8.2_20260427}"
if [[ "$IMAGE" == *.tar ]]; then
    if ! docker image inspect rocm/jax-training:maxtext-v26.2 >/dev/null 2>&1; then
        echo "[run_node.sh node $NODE_RANK] Loading image tar: $IMAGE"
        docker load -i "$IMAGE"
    fi
    IMAGE="rocm/jax-training:maxtext-v26.2"
fi

# Variant toggles: ANP loads the external plugin; AROCE keeps in-tree RCCL
# but turns on the AINIC RoCEv2 code path (CTS Inline + Offload + GDR-Flush-off).
# Note: AROCE and ANP are mutually exclusive at the dispatcher (plugin/net.cc:250).
PLUGIN_ENV=()
if [[ "$WITH_ANP" == "1" ]]; then
    PLUGIN_ENV=(-e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so)
fi
AROCE_ENV=()
if [[ "$WITH_AROCE" == "1" ]]; then
    AROCE_ENV=(-e RCCL_AINIC_ROCE=1)
fi

# Ionic / RoCE defaults (copied from train_env.sh key subset).  These are the
# env that made the non-ANP path reach 172 GB/s per GPU in the profiled run.
RCCL_ENV=(
    -e NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    -e NCCL_IB_GID_INDEX=1
    -e NCCL_IB_PCI_RELAXED_ORDERING=1
    -e NCCL_IB_ECE_ENABLE=0
    -e NCCL_IB_USE_INLINE=1
    -e NCCL_IB_QPS_PER_CONNECTION=4
    -e NCCL_NCHANNELS_PER_NET_PEER=4
    -e NCCL_GDRCOPY_ENABLE=1
    -e NCCL_GDR_FLUSH_DISABLE=1
    -e NCCL_IGNORE_CPU_AFFINITY=1
    -e NCCL_PXN_DISABLE=0
    -e NCCL_DMABUF_ENABLE=0   # /boot not mounted here; stay safe
    -e NCCL_DEBUG=INFO
    -e NCCL_DEBUG_SUBSYS=INIT,NET
    -e RCCL_LL128_FORCE_ENABLE=1
    -e RCCL_MSCCLPP_ENABLE=0   # avoid MSCCLPP noise — not relevant to ANP
    -e NET_OPTIONAL_RECV_COMPLETION=1
    -e HIP_FORCE_DEV_KERNARG=1
    -e HSA_ENABLE_IPC_MODE_LEGACY=1
    -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e HSA_NO_SCRATCH_RECLAIM=1
    -e IONIC_LOCKFREE=all
)

# Bind-mount host RoCE driver (matches USE_DOCKER_IMAGE_AINIC_DRIVER=false
# in container_env.sh — required on deepep-a77 where the container's
# libionic is older than the a-77 firmware).
IB_MOUNTS=(
    -v /etc/libibverbs.d:/etc/libibverbs.d:ro
    -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro
)

CONTAINER_NAME="anp-repro-${JOB_ID}-node${NODE_RANK}"
mkdir -p "$OUT_DIR"

# Match _container.sh device/capability setup: need --privileged and
# /dev/infiniband passthrough for ibv_get_device_list() to find the
# host's ionic NICs (otherwise NCCL falls back to TCP sockets).
IB_DEVICE=()
if [[ -e /dev/infiniband ]]; then IB_DEVICE=(--device /dev/infiniband); fi

echo "[run_node.sh node $NODE_RANK] starting container $CONTAINER_NAME"
docker run --rm \
    --name "$CONTAINER_NAME" \
    --label anp_repro=1 \
    --network host \
    --ipc host \
    --privileged \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit "nofile=1048576:1048576" \
    --shm-size 32G \
    --device /dev/kfd \
    --device /dev/dri \
    "${IB_DEVICE[@]}" \
    --group-add video \
    -v "$SCRIPT_DIR:/work" \
    -v "$OUT_DIR:/outputs" \
    "${IB_MOUNTS[@]}" \
    "${RCCL_ENV[@]}" \
    "${PLUGIN_ENV[@]}" \
    "${AROCE_ENV[@]}" \
    -e NODE_RANK="$NODE_RANK" \
    -e WORLD_SIZE=16 \
    -e LOCAL_WORLD=8 \
    -e UID_FILE="/outputs/ncclUniqueId.$JOB_ID" \
    -e REPRO_BIN=/work/anp_repro \
    -e REPRO_OPS="${REPRO_OPS:-ag,rs,ar}" \
    -e REPRO_SIZES="${REPRO_SIZES:-}" \
    -e REPRO_ITERS="${REPRO_ITERS:-30}" \
    -e REPRO_WARMUP="${REPRO_WARMUP:-5}" \
    -e WITH_ANP="$WITH_ANP" \
    -e WITH_AROCE="$WITH_AROCE" \
    -e JOB_ID="$JOB_ID" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    /work/launcher.sh
