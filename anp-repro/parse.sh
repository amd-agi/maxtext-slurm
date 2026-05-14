#!/bin/bash
# Diff two run outputs. Usage:
#   ./parse.sh <noanp_out_dir> <anp_out_dir>
set -eo pipefail

NOANP="${1:?usage: parse.sh <noanp_dir> <anp_dir>}"
ANP="${2:?usage: parse.sh <noanp_dir> <anp_dir>}"

pick_log() {
    local d="$1"
    if [[ -f "$d/rank_0.log" ]]; then echo "$d/rank_0.log"; return; fi
    if [[ -f "$d/node-0.log" ]]; then echo "$d/node-0.log"; return; fi
    echo ""
}

L_NOANP=$(pick_log "$NOANP"); L_ANP=$(pick_log "$ANP")
[[ -z "$L_NOANP" ]] && { echo "No rank_0.log or node-0.log in $NOANP"; exit 2; }
[[ -z "$L_ANP"   ]] && { echo "No rank_0.log or node-0.log in $ANP";   exit 2; }

python3 - "$L_NOANP" "$L_ANP" <<'PY'
import sys, re

# Supports both the legacy format (op, per_rank, total, per_op, best, bw)
# and the new format with a leading scope column (dcn/ici/all).
RX_SCOPED = re.compile(
    r"^\s*(dcn|ici|all)\s+(ag|rs|ar)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)
RX_PLAIN = re.compile(
    r"^\s*(ag|rs|ar)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)

def parse(path):
    out = {}   # (scope, op, per_rank) -> dict
    with open(path) as f:
        for line in f:
            m = RX_SCOPED.match(line)
            if m:
                scope, op, per, tot, per_ms, best_ms, gbps = m.groups()
            else:
                m = RX_PLAIN.match(line)
                if not m: continue
                op, per, tot, per_ms, best_ms, gbps = m.groups()
                scope = "all"
            out[(scope, op, int(per))] = {
                "per_op_ms": float(per_ms),
                "best_ms":   float(best_ms),
                "bw_GBs":    float(gbps),
            }
    return out

a = parse(sys.argv[1]); b = parse(sys.argv[2])
scope_ord = {"dcn":0, "ici":1, "all":2}
op_ord    = {"ag":0, "rs":1, "ar":2}
keys = sorted(set(a) & set(b), key=lambda k: (scope_ord[k[0]], op_ord[k[1]], k[2]))

print(f"{'scope':>5s} {'op':>3s} {'per_rank_bytes':>15s}   "
      f"{'noANP per_op':>13s} {'ANP per_op':>12s} {'ratio':>7s}   "
      f"{'noANP GB/s':>10s} {'ANP GB/s':>9s}")
for k in keys:
    sc, op, per = k
    pm1, pm2 = a[k]["per_op_ms"], b[k]["per_op_ms"]
    gw1, gw2 = a[k]["bw_GBs"],    b[k]["bw_GBs"]
    ratio = pm2/pm1 if pm1 else float("nan")
    print(f"{sc:>5s} {op:>3s} {per:>15d}   "
          f"{pm1:>10.3f} ms {pm2:>9.3f} ms {ratio:>6.2f}x   "
          f"{gw1:>10.2f} {gw2:>9.2f}")

# Summary: where's the regression concentrated?
def band(scope, op, lo, hi):
    return sum(b[k]["per_op_ms"] - a[k]["per_op_ms"]
               for k in keys if k[0]==scope and k[1]==op and lo <= k[2] <= hi)

print()
print("Sum of (ANP - noANP) per-op time in ms, by scope and size band:")
for sc in ["dcn","ici","all"]:
    if not any(k[0]==sc for k in keys): continue
    for op in ["ag","rs","ar"]:
        if not any(k[0]==sc and k[1]==op for k in keys): continue
        large = band(sc, op, 1024*1024, 10**12)
        small = band(sc, op, 0,           256*1024 - 1)
        print(f"  scope={sc}  op={op}:  large(>=1MB)={large:+.2f} ms   small(<256KB)={small:+.2f} ms")
PY
