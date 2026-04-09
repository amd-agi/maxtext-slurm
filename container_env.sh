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
if [[ -f "${BASH_SOURCE[0]%/*}/container_env.local.sh" ]]; then
    source "${BASH_SOURCE[0]%/*}/container_env.local.sh"
    echo "[INFO] Loaded registry credentials from container_env.local.sh"
fi
# ── end Registry credentials ──────────────────────────────────────────────────

# ── Docker image ──────────────────────────────────────────────────────────────
# Auto-detect GPU vendor and set default registry, image, and paths.
# Pull reference = $DOCKER_REGISTRY/$DOCKER_IMAGE (assembled in _container.sh).
# Override: DOCKER_REGISTRY=nvcr.io DOCKER_IMAGE=nvidia/jax:tag ./submit.sh ...
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
    if [[ -e /dev/kfd ]]; then
        : "${DOCKER_REGISTRY:=docker.io}"
        DOCKER_IMAGE="rocm/jax-training:maxtext-v26.2"
        : "${DOCKER_IMAGE_HAS_AINIC:=true}"
    else
        : "${DOCKER_REGISTRY:=nvcr.io}"
        DOCKER_IMAGE="nvidia/jax:26.03-maxtext-py3"
        : "${DOCKER_IMAGE_HAS_AINIC:=false}"
    fi
else
    : "${DOCKER_REGISTRY:=docker.io}"
    : "${DOCKER_IMAGE_HAS_AINIC:=true}"
fi
# MaxText location inside the container (varies by image).
# AMD (rocm/jax-training):    /workspace/maxtext
# NVIDIA (nvcr.io/nvidia/jax): /opt/maxtext
if [[ -z "${MAXTEXT_REPO_DIR:-}" ]]; then
    if [[ -e /dev/kfd ]]; then
        MAXTEXT_REPO_DIR="/workspace/maxtext"
    else
        MAXTEXT_REPO_DIR="/opt/maxtext"
    fi
fi
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
        "/perf_apps/maxtext_coredump"             # DLC cluster
    )
fi
# ── end Host paths to mount ───────────────────────────────────────────────────
