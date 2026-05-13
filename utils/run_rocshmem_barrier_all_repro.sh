#!/usr/bin/env bash
#
# Build and run the minimal rocSHMEM device-side barrier_all repro inside the
# existing maxtext-slurm container environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/rocshmem_barrier_all_repro.cu"

: "${NODE_RANK:?NODE_RANK must be set by _container.sh}"
: "${NNODES:?NNODES must be set by _container.sh}"
: "${OUTPUT_PATH:?OUTPUT_PATH must be set by _train.sh}"

LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-$(python3 -c "import hip; print(hip.hipGetDeviceCount()[1])" 2>/dev/null || echo 8)}"
export LOCAL_WORLD_SIZE

BUILD_ROOT="${OUTPUT_PATH}/rocshmem_barrier_all_repro_build"
BUILD_DIR="${BUILD_ROOT}/node_${NODE_RANK}"
EXCHANGE_DIR="${OUTPUT_PATH}/rocshmem_barrier_all_repro_exchange_${JOB_ID:-manual}"
mkdir -p "$BUILD_DIR" "$EXCHANGE_DIR"

HIPCC="${HIPCC:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}"
if [[ ! -x "$HIPCC" ]]; then
  HIPCC="$(command -v hipcc || true)"
fi
if [[ -z "$HIPCC" || ! -x "$HIPCC" ]]; then
  echo "[RS-BARRIER-RUN] ERROR: hipcc not found. Set HIPCC=/path/to/hipcc." >&2
  exit 2
fi

ROCSHMEM_PREFIX="${ROCSHMEM_HOME:-}"
if [[ -z "$ROCSHMEM_PREFIX" ]]; then
  for p in /opt/rocshmem /workspace/rocshmem /opt/rocm /opt/rocm-7.1.1 /usr/local; do
    if [[ -f "$p/include/rocshmem/rocshmem.hpp" ]]; then
      ROCSHMEM_PREFIX="$p"
      break
    fi
  done
fi
if [[ -z "$ROCSHMEM_PREFIX" || ! -f "$ROCSHMEM_PREFIX/include/rocshmem/rocshmem.hpp" ]]; then
  echo "[RS-BARRIER-RUN] ERROR: could not find rocSHMEM headers. Set ROCSHMEM_HOME." >&2
  exit 3
fi

ROCSHMEM_LIB_DIR=""
for d in "$ROCSHMEM_PREFIX/lib" "$ROCSHMEM_PREFIX/lib64" /opt/rocm/lib /opt/rocm-7.1.1/lib; do
  if compgen -G "$d/librocshmem.so*" >/dev/null || compgen -G "$d/librocshmem.a" >/dev/null; then
    ROCSHMEM_LIB_DIR="$d"
    break
  fi
done
if [[ -z "$ROCSHMEM_LIB_DIR" ]]; then
  echo "[RS-BARRIER-RUN] ERROR: could not find librocshmem under $ROCSHMEM_PREFIX." >&2
  exit 4
fi

BIN="$BUILD_DIR/rocshmem_barrier_all_repro"
echo "[RS-BARRIER-RUN] node=$NODE_RANK/$NNODES local_world_size=$LOCAL_WORLD_SIZE"
echo "[RS-BARRIER-RUN] HIPCC=$HIPCC"
echo "[RS-BARRIER-RUN] ROCSHMEM_PREFIX=$ROCSHMEM_PREFIX"
echo "[RS-BARRIER-RUN] ROCSHMEM_LIB_DIR=$ROCSHMEM_LIB_DIR"
echo "[RS-BARRIER-RUN] EXCHANGE_DIR=$EXCHANGE_DIR"

"$HIPCC" -std=c++17 -O0 -g \
  "$SRC" \
  -I"$ROCSHMEM_PREFIX/include" \
  -L"$ROCSHMEM_LIB_DIR" \
  -Wl,-rpath,"$ROCSHMEM_LIB_DIR" \
  -lrocshmem \
  -o "$BIN"

echo "[RS-BARRIER-RUN] built $BIN"

export ROCSHMEM_BARRIER_REPRO_DIR="$EXCHANGE_DIR"
export ROCSHMEM_BARRIER_REPRO_ITERS="${ROCSHMEM_BARRIER_REPRO_ITERS:-1}"
export ROCSHMEM_BARRIER_REPRO_UID_TIMEOUT="${ROCSHMEM_BARRIER_REPRO_UID_TIMEOUT:-120}"
export ROCSHMEM_BARRIER_REPRO_TIMEOUT="${ROCSHMEM_BARRIER_REPRO_TIMEOUT:-180}"

# Keep defaults close to the failing DeepEP jobs, but allow caller override.
export ROCSHMEM_BOOTSTRAP_TIMEOUT="${ROCSHMEM_BOOTSTRAP_TIMEOUT:-60}"

echo "[RS-BARRIER-RUN] launching $LOCAL_WORLD_SIZE local processes"
PIDS=()
for ((i = 0; i < LOCAL_WORLD_SIZE; i++)); do
  (
    export LOCAL_RANK="$i"
    export GLOBAL_RANK=$((NODE_RANK * LOCAL_WORLD_SIZE + i))
    echo "[RS-BARRIER-RUN] start GLOBAL_RANK=$GLOBAL_RANK LOCAL_RANK=$LOCAL_RANK NODE_RANK=$NODE_RANK"
    timeout --signal=TERM "$ROCSHMEM_BARRIER_REPRO_TIMEOUT" "$BIN"
  ) &
  PIDS+=("$!")
done

exit_code=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    :
  else
    rc=$?
    if [[ "$exit_code" -eq 0 ]]; then
      exit_code="$rc"
    fi
  fi
done

if [[ "$exit_code" -eq 0 ]]; then
  echo "[RS-BARRIER-RUN] PASS all local processes"
else
  echo "[RS-BARRIER-RUN] FAIL exit=$exit_code" >&2
fi
exit "$exit_code"
