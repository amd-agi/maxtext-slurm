"""Standalone internode DeepEP dispatch/combine repro.

This is a minimal driver that exercises **only** the Primus-Turbo
``moe_dispatch`` + ``moe_combine`` round-trip — no MaxText, no model, no
optimizer, no JAX/XLA tracing of a full ``p_train_step``.  Goal: reproduce the
``hipError_t(9) + Memory access fault on (nil)`` we see in MaxText with the
smallest possible Python program so we can iterate fast and add prints.

Invocation (under maxtext-slurm/_train.sh, which already preinits
``jax.distributed`` from SLURM env when ``NPROCS`` is set):

    python3 -u utils/mfu_tracker.py --repro-internode-deepep \
        [num_tokens=4096] [hidden=7168] [num_topk=8] [experts_per_rank=32] \
        [iters=1]

Same dims as ``Primus-Turbo/tests/jax/test_internode_dispatch_combine.py``
and the DeepSeek3-671B-proxy MoE config — chosen so this either reproduces
the MaxText crash exactly or rules out a Primus-Turbo-only cause.
"""

from __future__ import annotations

import os
import sys
import time
import traceback


def _parse_kv(argv):
    out = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _dump_rocshmem_state_via_ctypes(rank: int, world_size: int) -> None:
    """Inspect post-bootstrap, pre-dispatch rocSHMEM runtime state.

    The hypothesis after 23310: bootstrap succeeded but the kernel hits (nil)
    only on node-1 GPUs.  This dumps host-callable rocSHMEM API values per
    rank, then KV-gathers them to rank 0 for a cross-node comparison.  If any
    of {my_pe, n_pes, barrier_all, putmem round-trip} disagrees with what
    bootstrap was supposed to set up, we have a localized rocSHMEM-state bug.

    Designed to be a single-shot diagnostic that fails-soft: any error in here
    just gets printed and execution continues to the dispatch loop.
    """
    import ctypes
    import ctypes.util

    # 1. Find an open handle to the rocSHMEM library.  Primus-Turbo's
    # _C.deep_ep already dlopen'd it, so it's in the process's global symbol
    # table — using ctypes.CDLL(None) (a.k.a. RTLD_DEFAULT) is the most
    # reliable way to find symbols regardless of installation path.  Fall
    # back to explicit paths only if needed.
    # First, find the actual mapped path by inspecting /proc/self/maps.
    # Primus-Turbo's _C.deep_ep extension is already loaded and has pulled
    # librocshmem.so into our address space — use that exact path so we are
    # talking to the same library/instance.
    mapped_paths = []
    try:
        with open("/proc/self/maps") as f:
            seen = set()
            for ln in f:
                low = ln.lower()
                if "rocshmem" in low or "roc_shmem" in low:
                    parts = ln.split()
                    if len(parts) >= 6:
                        p = parts[5]
                        if p not in seen:
                            seen.add(p)
                            mapped_paths.append(p)
    except Exception:
        pass
    if mapped_paths:
        print(f"[PT-DBG:{rank}] rocshmem mapped libs in process: {mapped_paths}", flush=True)

    rs = None
    rs_source = ""
    # Strategy 1: CDLL on the actually-mapped paths (most reliable: same .so).
    for p in mapped_paths:
        try:
            rs = ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            rs_source = p
            print(f"[PT-DBG:{rank}] rocshmem ctypes loaded from mapped path {p}", flush=True)
            break
        except OSError as e:
            print(f"[PT-DBG:{rank}] CDLL({p}) failed: {e!r}", flush=True)
    # Strategy 2: process global symbol table.
    if rs is None:
        try:
            rs = ctypes.CDLL(None)
            rs_source = "RTLD_DEFAULT (process global)"
            print(f"[PT-DBG:{rank}] rocshmem ctypes via {rs_source}", flush=True)
        except OSError as e:
            print(f"[PT-DBG:{rank}] CDLL(None) failed: {e!r}; trying explicit paths", flush=True)

    if rs is None:
        candidates = []
        for prefix in ("/opt/rocm-7.1.1/lib", "/opt/rocm/lib", "/usr/lib", "/usr/lib64",
                       "/usr/lib/x86_64-linux-gnu"):
            for soname in ("librocshmem.so", "librocshmem.so.0", "librocshmem.so.1",
                           "libroc_shmem.so", "libroc_shmem.so.0"):
                candidates.append(f"{prefix}/{soname}")
        found_lib = ctypes.util.find_library("rocshmem") or ctypes.util.find_library("roc_shmem")
        if found_lib:
            candidates.append(found_lib)
        for path in candidates:
            try:
                rs = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                rs_source = path
                print(f"[PT-DBG:{rank}] rocshmem ctypes loaded from {path}", flush=True)
                break
            except OSError:
                continue
        if rs is None:
            # Last resort: print which librocshmem the process has actually
            # mapped, so the next iteration can target it precisely.
            try:
                with open("/proc/self/maps") as f:
                    rs_lines = [
                        ln.strip() for ln in f.readlines()
                        if ("rocshmem" in ln.lower() or "roc_shmem" in ln.lower())
                    ]
                # Take unique paths
                seen = set()
                paths = []
                for ln in rs_lines:
                    parts = ln.split()
                    if len(parts) >= 6:
                        p = parts[5]
                        if p not in seen:
                            seen.add(p)
                            paths.append(p)
                print(
                    f"[PT-DBG:{rank}] rocshmem ctypes load FAILED; "
                    f"/proc/self/maps shows: {paths[:6]}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[PT-DBG:{rank}] rocshmem ctypes load FAILED and could not read maps: {e!r}",
                    flush=True,
                )
            return

    def _resolve(rs_lib, *names, restype=ctypes.c_int, argtypes=None):
        for n in names:
            try:
                fn = getattr(rs_lib, n)
                fn.restype = restype
                fn.argtypes = argtypes or []
                return fn, n
            except Exception:
                continue
        return None, None

    sym_my_pe, n1   = _resolve(rs, "rocshmem_my_pe", "roc_shmem_my_pe")
    sym_n_pes, n2   = _resolve(rs, "rocshmem_n_pes", "roc_shmem_n_pes")
    sym_query, n3   = _resolve(rs, "rocshmem_query_thread", "roc_shmem_query_thread",
                                argtypes=[ctypes.POINTER(ctypes.c_int)])
    sym_barrier, n4 = _resolve(rs, "rocshmem_barrier_all", "roc_shmem_barrier_all",
                                restype=None)

    if sym_my_pe is None or sym_n_pes is None:
        # Try to enumerate all symbols matching pe/n_pes for sanity
        try:
            import subprocess
            for path in candidates:
                try:
                    out = subprocess.check_output(
                        ["nm", "-D", "--defined-only", path], text=True, stderr=subprocess.DEVNULL,
                    )
                    matches = [ln for ln in out.splitlines() if "my_pe" in ln or "n_pes" in ln]
                    if matches:
                        print(f"[PT-DBG:{rank}] nm({path}) matches: {matches[:6]}", flush=True)
                        break
                except Exception:
                    continue
        except Exception:
            pass
        print(
            f"[PT-DBG:{rank}] rocshmem symbol resolution FAILED "
            f"(my_pe={n1} n_pes={n2}); cannot continue diagnostic",
            flush=True,
        )
        return

    state = {"loaded_from": "ok"}
    try:
        state["my_pe"] = int(sym_my_pe())
        state["n_pes"] = int(sym_n_pes())
    except Exception as e:
        state["my_pe_err"] = repr(e)
        print(f"[PT-DBG:{rank}] rocshmem my_pe/n_pes call CRASHED: {e!r}", flush=True)
        return

    if sym_query is not None:
        try:
            qt = ctypes.c_int(-1)
            sym_query(ctypes.byref(qt))
            state["thread_level"] = int(qt.value)
        except Exception as e:
            state["thread_level_err"] = repr(e)

    print(
        f"[PT-DBG:{rank}] rocshmem state: my_pe={state.get('my_pe')} "
        f"n_pes={state.get('n_pes')} thread_level={state.get('thread_level')} "
        f"(symbols used: my_pe={n1}, n_pes={n2})",
        flush=True,
    )

    # KV-gather state from every rank, dump comparison table from rank 0.
    try:
        from jax._src.distributed import global_state as _gs
        client = _gs.client
        import json as _json
        client.key_value_set(f"rs_state:{rank}", _json.dumps(state))
        if rank == 0:
            print("[PT-DBG:0] === rocshmem cross-rank state table ===", flush=True)
            for r in range(world_size):
                v = client.blocking_key_value_get(f"rs_state:{r}", 60_000)
                rdma_rank = r // 8
                nvl_rank = r % 8
                print(
                    f"[PT-DBG:0]   rank={r} (rdma_rank={rdma_rank} nvl_rank={nvl_rank}) "
                    f"state={v}",
                    flush=True,
                )
    except Exception as e:
        print(f"[PT-DBG:{rank}] rs_state KV-gather failed: {e!r}", flush=True)

    # Host-side barrier check: should never hang in a healthy 2-PE rocSHMEM
    # world.  If it does, IB peering between this rank and its mirror is
    # broken even though init reported success.  Wrap with a stderr breadcrumb
    # so a hang here is immediately visible.
    if sym_barrier is not None:
        print(f"[PT-DBG:{rank}] calling rocshmem_barrier_all() ...", flush=True)
        try:
            sym_barrier()
            print(f"[PT-DBG:{rank}] rocshmem_barrier_all() returned OK", flush=True)
        except Exception as e:
            print(f"[PT-DBG:{rank}] rocshmem_barrier_all CRASHED: {e!r}", flush=True)


def _install_primus_turbo_traces(rank: int) -> None:
    """Wrap the Primus-Turbo runtime hot paths with print() before/after.

    Primus-Turbo lives inside the docker image, not the host artifact, so any
    edits to its source on the host are inert.  Monkey-patching from here is
    the cheapest way to get bootstrap + dispatch + combine visibility without
    rebuilding the image.
    """
    import hashlib
    import jax
    import jax.numpy as jnp
    import numpy as np
    import primus_turbo.jax.deep_ep.runtime as _rt
    import primus_turbo.jax.lax.moe as _moe
    from primus_turbo.jax.deep_ep.runtime import NUM_MAX_NVL_PEERS  # type: ignore

    # KV-store-based bytes allgather to bypass the JAX/XLA collective
    # data-loss bug (one random rank's slot comes back as zeros).
    _kv_call_id = {"value": 0}

    def _kv_allgather_bytes(label, payload, num_ranks, my_rank):
        from jax._src.distributed import global_state as _gs
        client = _gs.client
        _kv_call_id["value"] += 1
        cid = _kv_call_id["value"]
        prefix = f"primus_turbo_kvgather:{label}:c{cid}"
        client.key_value_set(f"{prefix}:{my_rank}", bytes(payload).hex())
        out = []
        for r in range(num_ranks):
            hexstr = client.blocking_key_value_get(f"{prefix}:{r}", 60_000)
            out.append(bytearray.fromhex(hexstr))
        return out

    # Replace _bootstrap_per_process entirely with a corrected version that
    # uses _kv_allgather_bytes instead of multihost_utils.process_allgather.
    # Mirrors the original control flow at runtime.py:207-311 but with the
    # broken collectives swapped out.
    def _bootstrap_fixed(*, hidden_bytes, config):
        print(
            f"[PT-DBG:{rank}] _bootstrap_per_process ENTER "
            f"hidden_bytes={hidden_bytes} num_sms={config.num_sms}",
            flush=True,
        )
        try:
            dep = _rt._get_c_deep_ep()
            num_ranks = jax.process_count()
            internode = num_ranks > NUM_MAX_NVL_PEERS

            num_nvl_bytes = dep.get_nvl_buffer_size_hint(
                hidden_bytes, num_ranks, config.num_sms,
                config.num_max_nvl_chunked_send_tokens,
                config.num_max_nvl_chunked_recv_tokens,
                config.num_max_rdma_chunked_send_tokens,
                config.num_max_rdma_chunked_recv_tokens,
            )
            num_rdma_bytes = 0
            if internode:
                num_rdma_bytes = dep.get_rdma_buffer_size_hint(
                    hidden_bytes, num_ranks, config.num_sms,
                    config.num_max_nvl_chunked_send_tokens,
                    config.num_max_nvl_chunked_recv_tokens,
                    config.num_max_rdma_chunked_send_tokens,
                    config.num_max_rdma_chunked_recv_tokens,
                )

            # Mirror upstream: early-return if buffer already exists and is
            # large enough, else destroy-then-recreate.  Avoiding double-
            # bootstrap matters: rocSHMEM teams established by the first
            # sync_per_process_buffer must not be re-created underneath them.
            if (
                dep.is_per_process_buffer_ready()
                and _rt._per_process_nvl_bytes >= num_nvl_bytes
                and _rt._per_process_rdma_bytes >= num_rdma_bytes
            ):
                print(
                    f"[PT-DBG:{rank}] _bootstrap_per_process EARLY-RETURN "
                    f"(already ready: nvl_have={_rt._per_process_nvl_bytes}>={num_nvl_bytes}, "
                    f"rdma_have={_rt._per_process_rdma_bytes}>={num_rdma_bytes})",
                    flush=True,
                )
                return
            if dep.is_per_process_buffer_ready():
                print(
                    f"[PT-DBG:{rank}] bootstrap: GROW (destroy old "
                    f"nvl={_rt._per_process_nvl_bytes}->{num_nvl_bytes} "
                    f"rdma={_rt._per_process_rdma_bytes}->{num_rdma_bytes})",
                    flush=True,
                )
                dep.destroy_per_process_buffer()
                _rt._per_process_nvl_bytes = 0
                _rt._per_process_rdma_bytes = 0

            print(
                f"[PT-DBG:{rank}] bootstrap: nvl={num_nvl_bytes} rdma={num_rdma_bytes} "
                f"internode={internode}",
                flush=True,
            )

            # rocSHMEM unique-ID gather.  Mirrors upstream:  every process
            # calls dep.get_unique_id(); only the entry at slot==nvl_rank is
            # actually consumed (that's the rdma_rank==0 process for this
            # NVL slot).  Other slots are gathered just to keep a uniform
            # shape — same convention as the JAX collective version.
            root_uid = None
            if internode:
                rdma_rank = rank // NUM_MAX_NVL_PEERS
                nvl_rank = rank % NUM_MAX_NVL_PEERS
                uid_bytes = bytes(dep.get_unique_id())
                all_uids = _kv_allgather_bytes("uid", uid_bytes, num_ranks, rank)
                root_global_rank = nvl_rank  # rdma_rank==0 on same NVL slot
                root_uid = bytes(all_uids[root_global_rank])
                print(
                    f"[PT-DBG:{rank}] bootstrap: rocSHMEM root_uid acquired "
                    f"(rdma_rank={rdma_rank} nvl_rank={nvl_rank} "
                    f"root_global_rank={root_global_rank})",
                    flush=True,
                )

            local_ipc_handle = dep.create_per_process_buffer(
                rank, num_ranks, num_nvl_bytes, num_rdma_bytes,
            )
            local_h = hashlib.sha1(bytes(local_ipc_handle)).hexdigest()[:16]
            print(
                f"[PT-DBG:{rank}] bootstrap: local_ipc_handle sha1={local_h}",
                flush=True,
            )

            ipc_handles_list = _kv_allgather_bytes(
                "ipc", bytes(local_ipc_handle), num_ranks, rank,
            )
            gathered_h_self = hashlib.sha1(bytes(ipc_handles_list[rank])).hexdigest()[:16]
            match = (gathered_h_self == local_h)
            print(
                f"[PT-DBG:{rank}] bootstrap: kv-gather ipc handles match[rank]={match} "
                f"local={local_h} gathered[{rank}]={gathered_h_self}",
                flush=True,
            )
            if rank == 0:
                for i, h in enumerate(ipc_handles_list):
                    print(
                        f"[PT-DBG:0]   ipc gathered[{i}] sha1="
                        f"{hashlib.sha1(bytes(h)).hexdigest()[:16]}",
                        flush=True,
                    )

            if root_uid is None:
                dep.sync_per_process_buffer(ipc_handles_list)
            else:
                dep.sync_per_process_buffer(ipc_handles_list, root_uid)

            _rt._per_process_nvl_bytes = num_nvl_bytes
            _rt._per_process_rdma_bytes = num_rdma_bytes
            print(
                f"[PT-DBG:{rank}] _bootstrap_per_process EXIT OK "
                f"nvl_bytes={num_nvl_bytes} rdma_bytes={num_rdma_bytes} "
                f"internode={internode}",
                flush=True,
            )
        except BaseException as e:
            print(f"[PT-DBG:{rank}] _bootstrap_per_process FAILED: {e!r}", flush=True)
            try:
                dep.destroy_per_process_buffer()
                _rt._per_process_nvl_bytes = 0
                _rt._per_process_rdma_bytes = 0
                print(f"[PT-DBG:{rank}] _bootstrap_per_process cleanup OK", flush=True)
            except BaseException as cleanup_e:
                print(
                    f"[PT-DBG:{rank}] _bootstrap_per_process cleanup FAILED: {cleanup_e!r}",
                    flush=True,
                )
            raise

    _rt._bootstrap_per_process = _bootstrap_fixed

    # Hash-instrument the IPC-handle exchange to test the allgather-order
    # hypothesis: rank R contributes a 64-byte hipIPC handle via
    # create_per_process_buffer().  After multihost_utils.process_allgather we
    # expect ipc_handles_list[R] to equal what R contributed.  If hashes don't
    # match, the gather has reordered handles relative to process_index, and
    # the C++ side (which indexes by rdma_rank*8 + nvl_rank == global rank)
    # will open the wrong remote handle and assert-fail.
    try:
        dep = _rt._get_c_deep_ep()
    except Exception as e:
        print(f"[PT-DBG:{rank}] could not get _C.deep_ep: {e!r}", flush=True)
        return

    _local_handle_hash = {"value": None}

    if hasattr(dep, "create_per_process_buffer"):
        _orig_create = dep.create_per_process_buffer

        def _create_traced(rk, num_ranks, num_nvl_bytes, num_rdma_bytes):
            handle = _orig_create(rk, num_ranks, num_nvl_bytes, num_rdma_bytes)
            try:
                hb = bytes(handle)
                h = hashlib.sha1(hb).hexdigest()[:16]
                _local_handle_hash["value"] = h
                print(
                    f"[PT-DBG:{rank}] create_per_process_buffer rk={rk} "
                    f"num_ranks={num_ranks} nvl={num_nvl_bytes} rdma={num_rdma_bytes} "
                    f"local_handle_len={len(hb)} local_handle_sha1={h}",
                    flush=True,
                )
            except Exception as e:
                print(f"[PT-DBG:{rank}] create hash failed: {e!r}", flush=True)
            return handle

        dep.create_per_process_buffer = _create_traced

    if hasattr(dep, "sync_per_process_buffer"):
        _orig_sync = dep.sync_per_process_buffer

        def _sync_traced(handles_list, *args, **kwargs):
            try:
                hashes = []
                for i, h in enumerate(handles_list):
                    if h is None:
                        hashes.append("NONE")
                    else:
                        hashes.append(hashlib.sha1(bytes(h)).hexdigest()[:16])
                local_h = _local_handle_hash["value"]
                gathered_at_rank = hashes[rank] if rank < len(hashes) else "OOB"
                match = (local_h is not None and local_h == gathered_at_rank)
                print(
                    f"[PT-DBG:{rank}] sync_per_process_buffer ENTER  "
                    f"local_sha1={local_h} gathered[{rank}]_sha1={gathered_at_rank} match={match}",
                    flush=True,
                )
                if rank == 0:
                    for i, hh in enumerate(hashes):
                        print(f"[PT-DBG:0]   gathered[{i}]_sha1={hh}", flush=True)
            except Exception as e:
                print(f"[PT-DBG:{rank}] sync hashing failed: {e!r}", flush=True)
            try:
                ret = _orig_sync(handles_list, *args, **kwargs)
            except Exception as e:
                print(f"[PT-DBG:{rank}] sync_per_process_buffer FAILED: {e!r}", flush=True)
                raise
            print(f"[PT-DBG:{rank}] sync_per_process_buffer EXIT OK", flush=True)
            return ret

        dep.sync_per_process_buffer = _sync_traced

    if hasattr(_moe, "moe_dispatch"):
        _orig_dispatch = _moe.moe_dispatch
        _orig_combine  = _moe.moe_combine

        def _dispatch_traced(x, topk_idx, topk_weights, num_experts, *args, **kwargs):
            print(
                f"[PT-DBG:{rank}] moe_dispatch BEFORE "
                f"x.shape={tuple(x.shape)} topk_idx.shape={tuple(topk_idx.shape)} "
                f"num_experts={num_experts}",
                flush=True,
            )
            out = _orig_dispatch(x, topk_idx, topk_weights, num_experts, *args, **kwargs)
            print(
                f"[PT-DBG:{rank}] moe_dispatch QUEUED (returned {len(out)}-tuple)",
                flush=True,
            )
            try:
                jax.block_until_ready(tuple(o for o in out if hasattr(o, "shape")))
            except Exception as e:
                print(f"[PT-DBG:{rank}] moe_dispatch GPU-FAULT: {e!r}", flush=True)
                raise
            print(f"[PT-DBG:{rank}] moe_dispatch GPU-DONE", flush=True)
            return out

        def _combine_traced(recv_x, handle, *args, **kwargs):
            print(
                f"[PT-DBG:{rank}] moe_combine BEFORE recv_x.shape={tuple(recv_x.shape)}",
                flush=True,
            )
            out = _orig_combine(recv_x, handle, *args, **kwargs)
            print(f"[PT-DBG:{rank}] moe_combine QUEUED", flush=True)
            try:
                jax.block_until_ready(out)
            except Exception as e:
                print(f"[PT-DBG:{rank}] moe_combine GPU-FAULT: {e!r}", flush=True)
                raise
            print(f"[PT-DBG:{rank}] moe_combine GPU-DONE", flush=True)
            return out

        _moe.moe_dispatch = _dispatch_traced
        _moe.moe_combine  = _combine_traced


def main(argv):
    kv = _parse_kv(argv)
    num_tokens       = int(kv.get("num_tokens", 4096))
    hidden           = int(kv.get("hidden", 7168))
    num_topk         = int(kv.get("num_topk", 8))
    experts_per_rank = int(kv.get("experts_per_rank", 32))
    iters            = int(kv.get("iters", 1))

    os.environ.setdefault("PRIMUS_TURBO_JAX_DEEPEP_MODE", "per_process")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import primus_turbo.jax  # noqa: F401  (registers ops)

    rank        = jax.process_index()
    world_size  = jax.process_count()
    num_experts = experts_per_rank * world_size

    print(
        f"[repro:{rank}] rank={rank}/{world_size} num_tokens={num_tokens} "
        f"hidden={hidden} topk={num_topk} num_experts={num_experts}",
        flush=True,
    )

    _install_primus_turbo_traces(rank)
    print(f"[repro:{rank}] PT-DBG traces installed", flush=True)

    print(f"[repro:{rank}] calling primus_turbo.jax.initialize() ...", flush=True)
    primus_turbo.jax.initialize()
    print(f"[repro:{rank}] primus_turbo.jax.initialize() OK", flush=True)

    # PRE-TEST: 23290 confirmed JAX's multihost_utils.process_allgather drops
    # one random rank's contribution (slot R == sha1(64 zero bytes)).  Each
    # rank contributes (rank+1)*ones(64); slot R should be (R+1) everywhere.
    # We test 3 candidate fixes side-by-side:
    #   (a) raw call (the broken baseline)
    #   (b) block_until_ready before the collective (forces materialization)
    #   (c) KV-store gather via the JAX distributed coordinator (no XLA collective)
    from jax.experimental import multihost_utils as _mhu

    def _check(label, gathered_np):
        first_bytes = [int(gathered_np[r, 0]) for r in range(world_size)]
        ok = all(int(gathered_np[r, 0]) == (r + 1) for r in range(world_size))
        print(
            f"[repro:{rank}] pre-test [{label}] first_bytes={first_bytes} OK={ok}",
            flush=True,
        )
        return ok

    try:
        contribution_a = jnp.full((64,), rank + 1, dtype=jnp.uint8)
        out_a = _mhu.process_allgather(contribution_a)
        _check("a-raw", np.asarray(out_a).reshape(world_size, -1))
    except Exception:
        print(f"[repro:{rank}] pre-test [a-raw] CRASH", flush=True)
        traceback.print_exc()

    try:
        contribution_b = jnp.full((64,), rank + 1, dtype=jnp.uint8)
        contribution_b.block_until_ready()
        out_b = _mhu.process_allgather(contribution_b)
        _check("b-block", np.asarray(out_b).reshape(world_size, -1))
    except Exception:
        print(f"[repro:{rank}] pre-test [b-block] CRASH", flush=True)
        traceback.print_exc()

    try:
        # KV-store gather via JAX coordinator: completely bypasses XLA.
        from jax._src.distributed import global_state as _gs  # noqa
        client = _gs.client
        payload = bytes([rank + 1] * 64)
        client.key_value_set(f"rkv_pretest:{rank}", payload.hex())
        gathered_c = []
        for r in range(world_size):
            v = client.blocking_key_value_get(f"rkv_pretest:{r}", 60_000)
            gathered_c.append(bytes.fromhex(v))
        gathered_c_np = np.array(
            [list(g) for g in gathered_c], dtype=np.uint8
        ).reshape(world_size, -1)
        _check("c-kv", gathered_c_np)
    except Exception:
        print(f"[repro:{rank}] pre-test [c-kv] CRASH", flush=True)
        traceback.print_exc()

    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch, warmup  # noqa: E402

    # DeepEP per_process bootstrap is a *host-side collective*: it does
    # multihost_utils.process_allgather of the rocSHMEM root unique ID and the
    # per-process IPC handles, then calls sync_per_process_buffer() so the C++
    # side can rocshmem_init and map peer buffers.  Inside _moe_dispatch_impl
    # this only fires when x is not a Tracer, so under jax.jit it would be
    # silently skipped and the FFI custom call would launch with nullptr
    # buffers (==> hipError(9) "Memory access fault on (nil)" for the wrong
    # reason).  Even in eager mode it's cleaner to bootstrap up front: it
    # gives us an isolated collective barrier we can diagnose, and it
    # decouples bootstrap timing from the first dispatch call.  Must be
    # called collectively on every rank.
    hidden_bytes = hidden * max(jnp.dtype(jnp.bfloat16).itemsize, 2)
    print(
        f"[repro:{rank}] warmup(hidden_bytes={hidden_bytes}) BEFORE",
        flush=True,
    )
    try:
        warmup(hidden_bytes)
    except Exception:
        print(f"[repro:{rank}] warmup CRASH", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print(f"[repro:{rank}] warmup AFTER OK", flush=True)

    # Per-rank rocSHMEM runtime-state diagnostic (post-bootstrap, pre-dispatch).
    # We don't rebuild the docker image; instead we ctypes-load librocshmem.so
    # (already pulled into the process by Primus-Turbo) and call host-side
    # rocSHMEM API directly to inspect what the kernel will see.  Cross-rank
    # comparison via the JAX coordinator KV store (which we've proven works).
    _dump_rocshmem_state_via_ctypes(rank, world_size)

    key          = jax.random.PRNGKey(rank)
    x            = jnp.ones((num_tokens, hidden), dtype=jnp.bfloat16) * rank
    scores       = jnp.abs(jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)) + 1
    topk_idx     = jax.lax.top_k(scores, num_topk)[1].astype(jnp.int32)
    topk_weights = jnp.ones((num_tokens, num_topk), dtype=jnp.float32) * rank

    # NOTE: We deliberately do NOT wrap dispatch/combine in jax.jit so that
    # each call is dispatched eagerly and we can place a block_until_ready
    # barrier between them.  This pins the crash to a specific call.
    for i in range(iters):
        t0 = time.time()
        print(f"[repro:{rank}] iter={i} === step start ===", flush=True)

        try:
            print(f"[repro:{rank}] iter={i} BEFORE moe_dispatch", flush=True)
            recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
                x, topk_idx, topk_weights, num_experts
            )
            jax.block_until_ready((recv_x, recv_topk_idx, recv_topk_weights))
            print(
                f"[repro:{rank}] iter={i} AFTER  moe_dispatch  recv_x.shape={tuple(recv_x.shape)} "
                f"recv_topk_idx.shape={tuple(recv_topk_idx.shape)}",
                flush=True,
            )
        except Exception:
            print(f"[repro:{rank}] iter={i} CRASH inside moe_dispatch", flush=True)
            traceback.print_exc()
            sys.exit(1)

        try:
            print(f"[repro:{rank}] iter={i} BEFORE moe_combine", flush=True)
            combined = moe_combine(recv_x, handle)
            jax.block_until_ready(combined)
            print(
                f"[repro:{rank}] iter={i} AFTER  moe_combine combined.shape={tuple(combined.shape)}",
                flush=True,
            )
        except Exception:
            print(f"[repro:{rank}] iter={i} CRASH inside moe_combine", flush=True)
            traceback.print_exc()
            sys.exit(1)

        try:
            checksum = float(np.asarray(jnp.sum(combined.astype(jnp.float32))))
        except Exception:
            checksum = float("nan")
        elapsed = time.time() - t0
        print(
            f"[repro:{rank}] iter={i} === step OK ({elapsed:.2f}s) checksum={checksum:.3e} ===",
            flush=True,
        )

    print(f"[repro:{rank}] PASS", flush=True)
    try:
        from primus_turbo.jax.deep_ep import runtime as _rt

        _rt.reset_runtime()
        print(f"[repro:{rank}] DeepEP runtime reset OK", flush=True)
    except Exception as e:
        print(f"[repro:{rank}] DeepEP runtime reset FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
