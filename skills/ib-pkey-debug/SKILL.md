---
name: ib-pkey-debug
description: >-
  Diagnose and fix InfiniBand RDMA failures caused by P_Key misconfiguration.
  Use when multi-node training fails with IBV_WC_RETRY_EXC_ERR(12), NCCL IB
  retry exhaustion, ibv_modify_qp EINVAL, or GID index -1 errors. Also applies
  when ib_write_bw or other IB verbs tests fail between nodes.
---

# IB P_Key Debugging

Diagnose InfiniBand RDMA failures caused by Partition Key (P_Key) misconfiguration on the fabric. The typical symptom is `IBV_WC_RETRY_EXC_ERR(12)` in NCCL logs during multi-node training.

## Quick reference

| Concept | Description |
|---------|-------------|
| P_Key | 16-bit partition key for IB access isolation (like VLAN for Ethernet) |
| Full member | Bit 15 = 1. Can communicate with any member in the same partition |
| Limited member | Bit 15 = 0. Cannot communicate with another limited member |
| pkey_index | Position in the HCA's P_Key table (0 = default). Passed to `ibv_modify_qp` |
| Subnet Manager (SM) | Distributes P_Key tables to all HCAs on the fabric |

**IB spec rule**: RDMA requires at least one full member in the pair.
```
full    ↔ full     ✅
full    ↔ limited  ✅
limited ↔ limited  ❌  → IBV_WC_RETRY_EXC_ERR(12)
```

## Diagnostic workflow

### Step 1 — Identify the error signature

Look for these patterns in the job log:

| Pattern | Likely cause |
|---------|-------------|
| `IBV_WC_RETRY_EXC_ERR(12) vendor_err=129` | P_Key limited-member pair, OR `NCCL_IB_SL` targeting unconfigured VL |
| `ibv_modify_qp failed with 22 ... local GID index -1` | NCCL_IB_PKEY set on native IB → DOCA GID selection bug |
| `ib_write_bw` hangs or fails on default pkey | Same P_Key issue at verbs level |

### Step 2 — Check the P_Key table

Run on a cluster node (inside or outside the container):

```bash
for hca in /sys/class/infiniband/*/; do
    echo "=== $(basename $hca) ==="
    for f in "$hca"/ports/1/pkeys/*; do
        idx=$(basename "$f")
        val=$(cat "$f" 2>/dev/null)
        [[ "$val" == "0x0000" ]] && continue
        int_val=$(( val ))
        if (( int_val & 0x8000 )); then
            echo "  index $idx: $val (full member)"
        else
            echo "  index $idx: $val (LIMITED member)"
        fi
    done
done
```

**Healthy output**: index 0 should be `0xffff` (full member).
**Problematic output**: index 0 is `0x7fff` or similar (limited member).

### Step 3 — Verify with ib_write_bw

From one node to another:

```bash
# Server (node A):
ib_write_bw --pkey_index=0 -d mlx5_0

# Client (node B):
ib_write_bw --pkey_index=0 -d mlx5_0 <node_A_ip>
```

If this fails, try `--pkey_index=1` (or whatever index holds the full-member key). If that succeeds, the P_Key is confirmed as the root cause.

### Step 4 — Check NCCL_IB_SL

If `NCCL_IB_SL` is set to a non-zero value, verify the SM has QoS / SL-to-VL mapping configured. On clusters without QoS, any SL > 0 causes the same retry exhaustion symptom. Fix: unset or set to 0.

### Step 5 — Apply the fix

There are two paths depending on your access level:

#### Path A: SM-level fix (requires admin access)

Ask the fabric administrator to change the default P_Key at index 0 to full member (`0xffff`). This is the clean fix.

#### Path B: Application-level workaround (no admin access)

Use the `LD_PRELOAD` shim in `utils/ib_pkey_fix.sh`. It:
1. Auto-detects whether the default P_Key is limited-member
2. Finds the first full-member P_Key index
3. Compiles a C shim that intercepts `ibv_modify_qp` and swaps `pkey_index`
4. Loads via `MAXTEXT_LD_PRELOAD` (which becomes `LD_PRELOAD` in the training process)

Source it from the NVIDIA-specific env script:
```bash
source "${BASH_SOURCE[0]%/*}/utils/ib_pkey_fix.sh"
```

The shim is a no-op on clusters with a healthy default P_Key.

## Known pitfalls

### NCCL_IB_PKEY is broken on native IB (NCCL 2.29.x DOCA)

Do NOT set `NCCL_IB_PKEY` on native InfiniBand with NCCL 2.29.x. The DOCA-based IB layer has a bug: when a non-default P_Key is selected, the GID matching logic fails (returns index -1), causing `ibv_modify_qp` to fail with `EINVAL`. This affects both the built-in DOCA transport and external net plugins (they share `ibvwrap.c`).

### LD_PRELOAD shim must use dlopen, not RTLD_NEXT

NCCL loads libibverbs via `dlopen("libibverbs.so.1")` with local scope. A shim using `dlsym(RTLD_NEXT, "ibv_modify_qp")` will get NULL and segfault. The shim must explicitly `dlopen("libibverbs.so.1")` and `dlsym(handle, ...)` to find the real function. See `utils/ib_pkey_fix.sh` for the correct implementation.

### HPC-X external plugin does not bypass the bug

The HPC-X NCCL RDMA SHARP plugin (`libnccl-net.so`, `IBext_v11`) loads successfully and provides its own net transport. However, NCCL's `ibvwrap.c` (which handles QP attribute setup including P_Key and GID) is shared infrastructure. The GID = -1 bug still affects QP creation even with the external plugin loaded.

## Verification

After applying the fix, the job log should show:
```
[IB P_Key fix] Default P_Key is limited-member; patching pkey_index 0 -> 1 (P_Key=0xf0dc) via LD_PRELOAD
[ib_pkey_fix] shim active: pkey_index 0 -> 1
```

And there should be **no** `ibvwrap.c` errors or `IBV_WC_RETRY_EXC_ERR` messages.
