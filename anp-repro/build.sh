#!/bin/bash
# Build anp_repro.cc inside the training container and drop the binary
# next to the source so both nodes (sharing the repo) can find it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${DOCKER_IMAGE:-rocm/jax-training:maxtext-v26.2}"

# If a tarball path was given (matching the pattern used in container_env.sh),
# load it into the local docker daemon first so we can "docker run" by tag.
if [[ "$IMAGE" == *.tar ]]; then
    if ! docker image inspect rocm/jax-training:maxtext-v26.2 >/dev/null 2>&1; then
        echo "[build.sh] Loading image tar: $IMAGE"
        docker load -i "$IMAGE"
    fi
    IMAGE="rocm/jax-training:maxtext-v26.2"
fi

echo "[build.sh] Building anp_repro using $IMAGE ..."
docker run --rm \
    -v "$SCRIPT_DIR:/work" \
    -w /work \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c 'set -e
        export PATH=/opt/rocm/bin:$PATH
        echo "[build] hipcc version:"
        hipcc --version | head -2
        echo "[build] Compiling anp_repro.cc ..."
        hipcc -std=c++17 -O2 -Wall \
              -I/opt/rocm/include \
              -o anp_repro anp_repro.cc \
              -L/opt/rocm/lib -lrccl
        echo "[build] Done: $(ls -la anp_repro)"
    '

echo "[build.sh] Binary: $SCRIPT_DIR/anp_repro"
file "$SCRIPT_DIR/anp_repro"
