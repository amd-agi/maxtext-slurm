# InfiniBand P_Key Fix Report

**Cluster**: hungry-hippo-fin-03 (NVIDIA B200 × 8/node, ConnectX-7 × 8 400 Gbps native IB)
**Software**: NCCL 2.29.7 + CUDA 13.0, JAX + MaxText
**Date**: 2026-04-10

---

## 1. Symptom

Multi-node training (`-N 2`) hung during NCCL initialization with repeated errors:

```
NCCL WARN NET/IB: Got completion from peer ... with status=IBV_WC_RETRY_EXC_ERR(12) vendor_err=129
```

All 8 IB rails (`mlx5_0`–`mlx5_7`) were affected. Single-node training worked fine.

## 2. Root Cause

### 2.1 Eliminated red herrings

- **`NCCL_IB_SL=1`**: The SM had no QoS configuration; SL=1 mapped to an unconfigured Virtual Lane, causing silent packet drops. Commenting it out resolved this secondary issue but revealed the primary one.
- **`NCCL_IB_HCA` blanket enumeration**: Listing every device in `/sys/class/infiniband` could include unreachable HCAs. Changed to let NCCL auto-probe.

### 2.2 Core issue: limited-member default P_Key

The Subnet Manager configured the default P_Key (index 0) as a **limited member**:

```
pkeys/0 → 0x7fff  (bit 15 = 0 → limited member)
pkeys/1 → 0xf0dc  (bit 15 = 1 → full member)
```

The IB specification requires at least one full member in any RDMA communication pair. Two limited-member endpoints are rejected by hardware — manifesting as `IBV_WC_RETRY_EXC_ERR(12)`.

Verified with `ib_write_bw`:
- `--pkey_index=0`: connection failed
- `--pkey_index=1`: succeeded at 392 Gbps

### 2.3 Why NCCL_IB_PKEY didn't work

NCCL 2.29.7 uses a DOCA-based IB verbs layer. When `NCCL_IB_PKEY` is set on **native InfiniBand** (as opposed to RoCE), the GID selection logic fails — it cannot find a GID matching the non-default partition, returns index = -1, and `ibv_modify_qp` fails with `EINVAL`.

Setting `NCCL_IB_GID_INDEX=0` alongside `NCCL_IB_PKEY` had no effect; NCCL's internal code overrides the user-specified GID index when a P_Key is explicitly selected.

Loading the HPC-X RDMA SHARP external net plugin (`IBext_v11`) did not help either: NCCL's `ibvwrap.c` is shared infrastructure used by both the built-in and external plugins, so the GID = -1 bug persisted.

## 3. Solution

### Approach

Do **not** set `NCCL_IB_PKEY`. Let NCCL follow its default code path (correct GID selection, default pkey_index = 0). Use an `LD_PRELOAD` shim to transparently replace `pkey_index = 0` with the first full-member index at the `ibv_modify_qp` level.

### Key implementation detail

The initial shim used `dlsym(RTLD_NEXT, "ibv_modify_qp")` to find the real function. This **segfaulted** because NCCL loads libibverbs via `dlopen("libibverbs.so.1", RTLD_LOCAL)`, placing its symbols outside the global search scope — `RTLD_NEXT` returned NULL.

The fix uses explicit `dlopen` + `dlsym` with the library handle:

```c
static int (*resolve_real_fn(void))(...) {
    // Try RTLD_NEXT first (works when libibverbs is in global scope)
    fn_t f = dlsym(RTLD_NEXT, "ibv_modify_qp");
    if (f && f != ibv_modify_qp) return f;

    // Fallback: open libibverbs explicitly
    void *h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libibverbs.so.1", RTLD_NOW);
    if (h) {
        f = dlsym(h, "ibv_modify_qp");
        if (f && f != ibv_modify_qp) return f;
    }
    return NULL;
}
```

### Auto-detection (`utils/ib_pkey_fix.sh`)

1. Read `/sys/class/infiniband/<hca>/ports/1/pkeys/0`; check bit 15
2. If limited member, scan the P_Key table for the first full-member index
3. Compile the shim inside the container; export via `MAXTEXT_LD_PRELOAD`
4. No-op when the default P_Key is already a full member

### Files changed

| File | Change |
|------|--------|
| `utils/ib_pkey_fix.sh` | New. Auto-detect, compile, and load the shim |
| `train_env.nvidia.sh` | Source `ib_pkey_fix.sh`; comment out `NCCL_IB_SL=1` |
| `utils/detect_nccl_env.sh` | Remove blind `NCCL_IB_HCA` enumeration |

## 4. Validation

| Job | Configuration | Result |
|-----|---------------|--------|
| 1114 | Original (`NCCL_IB_SL=1`) | `IBV_WC_RETRY_EXC_ERR(12)` |
| 1119 | `NCCL_IB_PKEY=0xf0dc` | `ibv_modify_qp` failed, GID index -1 |
| 1124 | LD_PRELOAD shim v1 (`RTLD_NEXT`) | Segfault (NULL function pointer) |
| 1129 | HPC-X plugin + `NCCL_IB_PKEY` | Plugin loaded, but ibvwrap GID still -1 |
| **1133** | **LD_PRELOAD shim v2 (`dlopen`)** | **SUCCESS, exit 0, no IB errors** |

## 5. Scope and limitations

- **Applies to**: Native IB clusters where the default P_Key is limited-member, running NCCL 2.29.x (DOCA-based)
- **No-op on**: Clusters with full-member default P_Key (`0xffff`) — shim auto-skips
- **No impact on**: AMD / RoCE clusters (only sourced from `train_env.nvidia.sh`)
- **Requires**: `gcc` and `infiniband/verbs.h` inside the container (present in NVIDIA NGC images)
