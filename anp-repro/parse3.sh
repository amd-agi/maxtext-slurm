#!/bin/bash
# 3-way diff: noANP vs ANP vs AROCE.
# Usage:
#   ./parse3.sh <noanp_dir> <anp_dir> <aroce_dir>
set -eo pipefail

NOANP="${1:?usage: parse3.sh <noanp_dir> <anp_dir> <aroce_dir>}"
ANP="${2:?usage: parse3.sh <noanp_dir> <anp_dir> <aroce_dir>}"
AROCE="${3:?usage: parse3.sh <noanp_dir> <anp_dir> <aroce_dir>}"

pick_log() {
    local d="$1"
    if [[ -f "$d/rank_0.log" ]]; then echo "$d/rank_0.log"; return; fi
    if [[ -f "$d/node-0.log" ]]; then echo "$d/node-0.log"; return; fi
    echo ""
}

L_NOANP=$(pick_log "$NOANP"); L_ANP=$(pick_log "$ANP"); L_AROCE=$(pick_log "$AROCE")
[[ -z "$L_NOANP" ]] && { echo "No rank_0/node-0 log in $NOANP" >&2; exit 2; }
[[ -z "$L_ANP"   ]] && { echo "No rank_0/node-0 log in $ANP" >&2;   exit 2; }
[[ -z "$L_AROCE" ]] && { echo "No rank_0/node-0 log in $AROCE" >&2; exit 2; }

python3 - "$L_NOANP" "$L_ANP" "$L_AROCE" <<'PY'
import re, sys

RX_SCOPED = re.compile(
    r"^\s*(dcn|ici|all)\s+(ag|rs|ar)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)
RX_PLAIN = re.compile(
    r"^\s*(ag|rs|ar)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def parse(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = RX_SCOPED.match(line)
            if m:
                scope, op, per, tot, per_ms, best_ms, gbps = m.groups()
            else:
                m = RX_PLAIN.match(line)
                if not m:
                    continue
                op, per, tot, per_ms, best_ms, gbps = m.groups()
                scope = "all"
            out[(scope, op, int(per))] = {
                "per_op_ms": float(per_ms),
                "best_ms": float(best_ms),
                "bw_GBs": float(gbps),
            }
    return out


a = parse(sys.argv[1])      # noANP
b = parse(sys.argv[2])      # ANP
c = parse(sys.argv[3])      # AROCE
scope_ord = {"dcn": 0, "ici": 1, "all": 2}
op_ord = {"ag": 0, "rs": 1, "ar": 2}
keys = sorted(set(a) & set(b) & set(c),
              key=lambda k: (scope_ord[k[0]], op_ord[k[1]], k[2]))

# Header
print(f"{'scope':>5s} {'op':>3s} {'per_rank_bytes':>15s}   "
      f"{'noANP_ms':>9s} {'ANP_ms':>9s} {'AROCE_ms':>9s}   "
      f"{'ANP/noANP':>9s} {'AROCE/noANP':>11s}   "
      f"{'noANP_GBs':>10s} {'ANP_GBs':>9s} {'AROCE_GBs':>10s}")
for k in keys:
    sc, op, per = k
    pa, pb, pc = a[k]["per_op_ms"], b[k]["per_op_ms"], c[k]["per_op_ms"]
    ga, gb, gc = a[k]["bw_GBs"], b[k]["bw_GBs"], c[k]["bw_GBs"]
    rb = pb / pa if pa else float("nan")
    rc = pc / pa if pa else float("nan")
    print(f"{sc:>5s} {op:>3s} {per:>15d}   "
          f"{pa:>9.3f} {pb:>9.3f} {pc:>9.3f}   "
          f"{rb:>8.2f}x {rc:>10.2f}x   "
          f"{ga:>10.2f} {gb:>9.2f} {gc:>10.2f}")


# Summary by scope+op+size-band
def band(d, scope, op, lo, hi):
    return sum(d[k]["per_op_ms"] - a[k]["per_op_ms"]
               for k in keys
               if k[0] == scope and k[1] == op and lo <= k[2] <= hi)


print()
print("Sum of (X - noANP) per-op time in ms, by scope/op/size-band:")
print(f"  {'cell':>5s}  {'scope':>5s} {'op':>3s}  {'large(>=1MB)':>14s} {'small(<256KB)':>14s}")
for cell, d in [("ANP", b), ("AROCE", c)]:
    for sc in ["dcn", "ici", "all"]:
        if not any(k[0] == sc for k in keys):
            continue
        for op in ["ag", "rs", "ar"]:
            if not any(k[0] == sc and k[1] == op for k in keys):
                continue
            large = band(d, sc, op, 1024 * 1024, 10**12)
            small = band(d, sc, op, 0, 256 * 1024 - 1)
            print(f"  {cell:>5s}  {sc:>5s} {op:>3s}  {large:>+13.2f} {small:>+14.2f}")
PY
