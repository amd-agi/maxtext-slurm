#!/bin/bash
# Standalone debug script: open 23284's coredump (rank 1, node 097-030) in
# gdb inside the same DeepEP container that crashed it, dump a backtrace, and
# write it to a path the user can read on the head node.
#
# Job 23284 crashed with:
#     hipError_t(9) + "Memory access fault by GPU node-X on address (nil)"
# during DeepEP per_process internode dispatch/combine.  We expect the host
# stack to show either a rocSHMEM bootstrap call (cf. 23124's
# Buffer::SyncFromIPCHandles -> rocshmem_init_attr -> Socket::connect) or
# a deep_ep dispatch FFI custom_call running with nullptr buffers.
#
# Submit via:
#   sbatch -N 1 -p amd-rccl -t 01:00:00 \
#          -x useocpm2m-097-038,useocpm2m-097-069,useocpm2m-097-084 \
#          -o /home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs/23284-debug-bt/slurm-%j.out \
#          /home/liyingli/workspace/jax-deepep/maxtext-slurm/utils/debug_bt_23284.sh
set -e

CORE_DIR=/home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs/23284-JAX-deepseek3-671b-proxy-internode-smoke-4n-full-bI-_env_ONE_GPU_PER_PROCESS_true-attention_dot_product-_env_ROCSHMEM_BOOTSTRAP_TIMEOUT_60-_env_ROCSHMEM_HEAP_SIZE_4294967296-per_device_batch_size_1
CORE_FILE=$(ls "$CORE_DIR"/core.23284.*.useocpm2m-097-030.python3.* 2>/dev/null | head -1)
PRIMUS_DIR=/home/liyingli/workspace/jax-deepep/Primus-Turbo
DOCKER_IMAGE_TAR=/home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar
OUT_DIR=/home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs/23284-debug-bt
mkdir -p "$OUT_DIR"; chmod a+w "$OUT_DIR"

echo "[DEBUG] Host: $(hostname)"
echo "[DEBUG] Core file: $CORE_FILE"
echo "[DEBUG] Out dir:   $OUT_DIR"
[[ -f "$CORE_FILE" ]] || { echo "ERROR: core file not found"; exit 2; }

if ! docker image inspect jax-deepep-1p1g:v0.2 >/dev/null 2>&1; then
    echo "[DEBUG] Loading image $DOCKER_IMAGE_TAR ..."
    docker load -i "$DOCKER_IMAGE_TAR"
fi

CORE_BASENAME=$(basename "$CORE_FILE")

# Run gdb -batch inside the container (root inside, so it can read root:600 cores).
docker run --rm \
    --user 0 \
    --privileged \
    --ipc=host \
    --network=host \
    -v "$CORE_DIR":/cores:ro \
    -v "$PRIMUS_DIR":/primus:ro \
    -v "$OUT_DIR":/output \
    -e CORE_BASENAME="$CORE_BASENAME" \
    jax-deepep-1p1g:v0.2 \
    bash -lc '
        set -e
        # Compute nodes have no internet -> apt install fails. Prefer the
        # gdb that ROCm ships (rocgdb wraps gdb with AMDGPU extensions but
        # works on plain CPU coredumps), then fall back to any gdb on PATH.
        GDB=""
        for cand in /opt/rocm/bin/rocgdb /opt/rocm/lib/llvm/bin/gdb /opt/rocm/llvm/bin/gdb $(command -v gdb 2>/dev/null) $(command -v rocgdb 2>/dev/null); do
            if [[ -x "$cand" ]]; then GDB="$cand"; break; fi
        done
        if [[ -z "$GDB" ]]; then
            echo "[ERROR] no gdb binary found in image; tried rocgdb / llvm gdb / system gdb"
            ls -la /opt/rocm/bin/ /opt/rocm/lib/llvm/bin/ 2>/dev/null | head -40
            exit 3
        fi
        echo "[gdb] using: $GDB"
        $GDB --version | head -2

        PYBIN=$(readlink -f /opt/venv/bin/python3 || true)
        [[ -z "$PYBIN" ]] && PYBIN=$(which python3)
        echo "[gdb] python binary: $PYBIN"
        echo "[gdb] core: /cores/$CORE_BASENAME"
        echo "[gdb] core size: $(stat -c%s /cores/$CORE_BASENAME) bytes"

        # python-gdb.py auto-load gives us py-bt to read CPython frames.  It
        # ships next to the matching libpython; tolerate it being absent.
        PYGDB=""
        for cand in /usr/share/gdb/auto-load/usr/bin/python3.12-gdb.py \
                    /usr/share/gdb/auto-load/usr/lib/python3.12/python-gdb.py \
                    /usr/share/gdb/auto-load/opt/venv/bin/python3.12-gdb.py \
                    $(find /usr/share/gdb/auto-load -name "python*-gdb.py" 2>/dev/null) \
                    $(find /usr/lib/debug -name "python*-gdb.py" 2>/dev/null); do
            if [[ -f "$cand" ]]; then PYGDB="$cand"; break; fi
        done
        echo "[gdb] python-gdb.py: ${PYGDB:-<not found>}"

        cd /

        echo "==== gdb -batch: thread apply all bt 50 (job 23284) ====" > /output/bt.txt
        EXTRA_PY_BT=()
        if [[ -n "$PYGDB" ]]; then
            EXTRA_PY_BT+=(-ex "source $PYGDB"
                          -ex "echo \n==== py-bt (Python frames) ====\n"
                          -ex "py-bt"
                          -ex "echo \n==== py-list (current Python source) ====\n"
                          -ex "py-list")
        fi

        "$GDB" -batch -nx \
            -ex "set print thread-events off" \
            -ex "set logging redirect on" \
            -ex "set logging file /output/bt.txt" \
            -ex "set logging enabled on" \
            -ex "set pagination off" \
            -ex "set print frame-arguments scalars" \
            -ex "set print address on" \
            -ex "set solib-search-path /opt/venv/lib/python3.12/site-packages/primus_turbo/lib:/opt/rocm/lib:/opt/rocm/lib64:/usr/lib/x86_64-linux-gnu" \
            -ex "directory /primus" \
            -ex "info threads" \
            -ex "echo \n==== full backtrace, all threads ====\n" \
            -ex "thread apply all bt 50" \
            -ex "echo \n==== shared libraries (deep_ep / rocshmem / primus / hsa / hip) ====\n" \
            -ex "info shared deep_ep" \
            -ex "info shared rocshmem" \
            -ex "info shared primus" \
            -ex "info shared hsa" \
            -ex "info shared hip" \
            -ex "echo \n==== signal info ====\n" \
            -ex "info signals SIGSEGV SIGABRT SIGBUS" \
            -ex "echo \n==== current frame ====\n" \
            -ex "bt 30" \
            -ex "echo \n==== locals/args of crashing frame ====\n" \
            -ex "info args" \
            -ex "info locals" \
            "${EXTRA_PY_BT[@]}" \
            -ex "set logging enabled off" \
            -ex "quit" \
            "$PYBIN" "/cores/$CORE_BASENAME" 2>&1 | tee -a /output/bt.txt || true

        chmod a+r /output/bt.txt || true
    '

echo "[DEBUG] Done. Backtrace at $OUT_DIR/bt.txt"
ls -lh "$OUT_DIR"
