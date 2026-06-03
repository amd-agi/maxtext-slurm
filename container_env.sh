#!/bin/bash

# Container environment configuration.
# Edit this file to switch images, paths, or deployment-specific settings.
# Sourced by _container.sh before launching the container
# and by in_container_run.sh for MAXTEXT_REPO_DIR and MAXTEXT_PATCH_BRANCH.
# All variables can be overridden from the command line, e.g.:
#   DOCKER_IMAGE=my/image:tag ./run_local.sh model_name -- ...

# ── Registry credentials (private images only) ────────────────────────────────
# For private images, copy the template and fill in your credentials:
#   cp container_env.local.template container_env.local.sh
# container_env.local.sh is gitignored — credentials are never committed.
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
if [[ -f "${BASH_SOURCE[0]%/*}/container_env.local.sh" ]]; then
    source "${BASH_SOURCE[0]%/*}/container_env.local.sh"
    echo "[INFO] Loaded registry credentials from container_env.local.sh"
fi
# ── end Registry credentials ──────────────────────────────────────────────────

# ── Docker image ──────────────────────────────────────────────────────────────
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pyt-megatron-lm-jax-nightly-private:maxtext-v26.2-rccl-pr2063}"
USE_DOCKER_IMAGE_AINIC_DRIVER="${USE_DOCKER_IMAGE_AINIC_DRIVER:-true}"    # true = use the container's own IB libs (no broad host /usr/lib mount). On DLC the host glibc (2.35) is older than the image (2.39), so the broad =false mount clobbers glibc and the container won't start; keep true and use USE_HOST_IONIC_PROVIDER_ONLY below instead. (On Vultr, where host glibc matches, =false also works.)
USE_HOST_IONIC_PROVIDER_ONLY="${USE_HOST_IONIC_PROVIDER_ONLY:-true}"     # Surgical ABI fix: bind-mount ONLY the host's libionic provider .so over the container's (host kernel verbs ABI=1 vs image rdma-core ABI=4) so RCCL uses RoCE over the ionic NICs WITHOUT dragging in host glibc. Pairs with USE_DOCKER_IMAGE_AINIC_DRIVER=true.
MAXTEXT_REPO_DIR="${MAXTEXT_REPO_DIR:-/workspace/maxtext}"  # MaxText location inside the container
MAXTEXT_PATCH_BRANCH="${MAXTEXT_PATCH_BRANCH:-}"            # Global patch branch (empty = image default); per-model .env.sh can override
# ── end Docker image ──────────────────────────────────────────────────────────

# ── Host paths to mount ───────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-/mnt/vast/datasets}"            # Host path to datasets (mounted read-only as /datasets inside the container)
# Extra coredump directories to probe (beyond JOB_WORKSPACE).
# First entry with >500GB free space wins.
# CLI override: comma-separated string, e.g. COREDUMP_EXTRA_DIRS="/path1,/path2"
if [[ -n "${COREDUMP_EXTRA_DIRS:-}" ]]; then
    IFS=',' read -ra COREDUMP_EXTRA_DIRS <<< "$COREDUMP_EXTRA_DIRS"
else
    COREDUMP_EXTRA_DIRS=(
        "/perf_apps/maxtext_coredump"           # DLC cluster
        # "/mnt/vast/xuefei_maxtext/coredump"       # Vultr cluster
    )
fi
# ── end Host paths to mount ───────────────────────────────────────────────────
