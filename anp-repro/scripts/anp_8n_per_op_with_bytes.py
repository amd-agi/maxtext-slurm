"""8N MoE: combine per-NCCL-op delta with HLO shape/comm-scope info.

Identifies which collectives drive the perf delta and labels each by:
  - communicator scope (global-64 / DCN-8 / ICI-intra-node)
  - payload bytes per op (input+output)
  - effective bandwidth per GPU
  - per-step contribution
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

from xprof.convert import _pywrap_profiler_plugin as xp


BASE_NOANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9501-JAX-ds-proxy-se0-e256-h4096-WithOUTANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_3155.723"
BASE_ANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9511-JAX-ds-proxy-se0-e256-h4096-WithANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_2538.112"
HLO_NOANP = os.path.join(BASE_NOANP, "xla_dump", "module_0142.jit_train_step.gfx950_gpu_after_optimizations.txt")

# Steady-state step timings from the logs (steps 1-14, excluding step 0 = compile).
STEP_TIME_NOANP = 31.30  # mean s/step (14 steps minus step 0)
STEP_TIME_ANP = 38.83
STEPS_PROFILED = 8  # n=512 events on 64 layers => 8 profiled iterations of the while body

DTYPE_BYTES = {"bf16": 2, "f32": 4, "s32": 4, "f8e4m3fn": 1}


def shape_bytes(shape_str: str) -> int:
    """Parse a single shape like 'bf16[32,512,2048]{2,0,1}' and return bytes."""
    m = re.match(r"\s*(\w+)\[([0-9,]+)\]", shape_str)
    if not m:
        return 0
    dt = m.group(1)
    dims = [int(d) for d in m.group(2).split(",") if d]
    n = 1
    for d in dims:
        n *= d
    return n * DTYPE_BYTES.get(dt, 0)


def parse_op_payload(rhs: str) -> int:
    """Extract every shape from the LHS tuple/struct and sum bytes."""
    total = 0
    for m in re.finditer(r"(\w+)\[([0-9,]+)\]", rhs):
        dt, dims = m.group(1), m.group(2)
        if dt not in DTYPE_BYTES:
            continue
        total += shape_bytes(f"{dt}[{dims}]")
    return total


def classify_comm_scope(replica_groups: str) -> str:
    """Identify the communicator scope from the replica_groups specifier."""
    rg = replica_groups.strip()
    if rg == "[1,64]<=[64]":
        return "GLOBAL-64 (cross-node, flat 64-rank)"
    if rg == "[8,8]<=[8,8]T(1,0)":
        return "DCN-8 (cross-node only, 1 GPU/node × 8 nodes)"
    if rg == "[8,8]<=[64]":
        return "ICI-8 (intra-node only, 8 GPUs in same node)"
    return f"UNKNOWN({rg})"


def parse_hlo_collectives(hlo_path: str) -> dict[str, dict]:
    """Map scheduling_name -> {scope, lhs_bytes_per_rank, replica_groups, op_kind}."""
    txt = open(hlo_path, encoding="utf-8", errors="ignore").read()
    info = {}
    pat = re.compile(
        r"^\s+(?:ROOT\s+)?%([A-Za-z0-9_.-]+)\s*=\s*(.*?)\s+(all-gather-start|all-gather|reduce-scatter|reduce-scatter-start|all-to-all|all-reduce-start|all-reduce)\((.*?)\),\s+channel_id=\d+,\s+replica_groups=(\S+)",
        re.MULTILINE | re.DOTALL,
    )
    for m in pat.finditer(txt):
        sched_var = m.group(1)
        lhs_shape = m.group(2)  # output shape(s)
        op_kind = m.group(3)
        rg = m.group(5)
        # The "scheduling_name" in the metadata block is what shows up in xplane args; it usually equals sched_var.
        # Extract scheduling_name from the trailing metadata if present (more reliable).
        # Look for scheduling_name="..." in the same statement (the metadata block follows replica_groups within ~500 chars).
        tail_start = m.end()
        tail = txt[tail_start: tail_start + 1500]
        sn = re.search(r'scheduling_name="([^"]+)"', tail)
        sched_name = sn.group(1) if sn else sched_var
        # output bytes -- shape immediately after '=' until next type or comma
        out_bytes = parse_op_payload(lhs_shape)
        info[sched_name] = {
            "scope": classify_comm_scope(rg),
            "out_bytes_total": out_bytes,
            "replica_groups": rg,
            "op_kind": op_kind,
            "var": sched_var,
        }
    return info


def find_xplane(run_dir: str, host: str) -> str:
    pat = os.path.join(run_dir, "ds-proxy-se0-e256-h4096", "tensorboard", "plugins", "profile", "*", f"{host}.xplane.pb")
    matches = glob.glob(pat)
    if not matches:
        raise FileNotFoundError(pat)
    return matches[0]


def aggregate_nccl_by_hlo(events):
    out = defaultdict(list)
    for e in events:
        name = e.get("name") or ""
        if "ncclDevKernel" not in name:
            continue
        if "dur" not in e:
            continue
        args = e.get("args") or {}
        hlo = args.get("hlo_op") or "<unknown>"
        out[hlo].append(float(e["dur"]))
    return out


def main():
    hlo = parse_hlo_collectives(HLO_NOANP)
    sys.stderr.write(f"Parsed {len(hlo)} collectives from HLO\n")

    runs = {
        "noANP_9501": find_xplane(BASE_NOANP, "chi2865"),
        "ANP_9511":   find_xplane(BASE_ANP,  "chi2865"),
    }
    summaries = {}
    for tag, path in runs.items():
        sys.stderr.write(f"[{tag}] loading {path}\n")
        evs = json.loads(xp.xspace_to_tools_data([path], "trace_viewer@", {})[0]).get("traceEvents", [])
        summaries[tag] = aggregate_nccl_by_hlo(evs)

    print("\n========================================================================")
    print("8N MoE: per-NCCL-op delta with HLO/comm-scope info  (chi2865, all profiled iters)")
    print("========================================================================")
    print(f"{'HLO op':<24s} | {'comm scope':<46s} | {'op':<12s} | {'n':>5} | {'out_bytes':>10} | {'noANP_ms':>8} | {'ANP_ms':>8} | {'Δ_ms':>8} | {'ratio':>5}")
    print("-" * 160)

    common = sorted(set(summaries["noANP_9501"]) | set(summaries["ANP_9511"]))
    rows = []
    for hop in common:
        a = summaries["noANP_9501"].get(hop, [])
        b = summaries["ANP_9511"].get(hop, [])
        a_sum_ms = sum(a) / 1e3
        b_sum_ms = sum(b) / 1e3
        a_mean = (a_sum_ms / len(a)) if a else 0
        b_mean = (b_sum_ms / len(b)) if b else 0
        info = hlo.get(hop, {})
        rows.append((hop, info, len(a), len(b), a_sum_ms, b_sum_ms, a_mean, b_mean))

    rows.sort(key=lambda r: -(r[5] - r[4]))  # sort by added time desc

    delta_by_scope = defaultdict(float)
    by_scope_total_noanp = defaultdict(float)
    by_scope_total_anp = defaultdict(float)

    for hop, info, na, nb, a_sum, b_sum, a_mean, b_mean in rows:
        scope = info.get("scope", "?")
        opk = info.get("op_kind", "?")
        out_bytes = info.get("out_bytes_total", 0)
        delta = b_sum - a_sum
        ratio = (b_sum / a_sum) if a_sum > 0 else float("inf")
        if a_sum < 1e-3 and b_sum < 1e-3:
            continue
        delta_by_scope[scope] += delta
        by_scope_total_noanp[scope] += a_sum
        by_scope_total_anp[scope] += b_sum
        out_str = f"{out_bytes/1e6:7.1f}MB" if out_bytes else "    n/a"
        print(f"{hop:<24s} | {scope:<46s} | {opk:<12s} | {nb:>5} | {out_str:>10} | "
              f"{a_mean:8.2f} | {b_mean:8.2f} | {delta:+8.1f} | {ratio:5.2f}x")

    print("\n--- AGGREGATED DELTA BY COMMUNICATOR SCOPE (chi2865, all profiled iters) ---")
    total_delta = sum(delta_by_scope.values())
    print(f"  {'scope':<48s} {'noANP_sum_ms':>14s} {'ANP_sum_ms':>14s} {'Δ_ms':>10s} {'pct_of_total_Δ':>14s}")
    for scope in sorted(delta_by_scope, key=lambda s: -delta_by_scope[s]):
        d = delta_by_scope[scope]
        a = by_scope_total_noanp[scope]
        b = by_scope_total_anp[scope]
        pct = 100.0 * d / total_delta if total_delta else 0
        print(f"  {scope:<48s} {a:14.1f} {b:14.1f} {d:+10.1f} {pct:13.1f}%")
    print(f"  {'TOTAL':<48s} {sum(by_scope_total_noanp.values()):14.1f} {sum(by_scope_total_anp.values()):14.1f} {total_delta:+10.1f}")

    print("\n--- PER-STEP IMPACT ---")
    print(f"  Wall-clock steady step time: noANP={STEP_TIME_NOANP:.2f}s, ANP={STEP_TIME_ANP:.2f}s, Δ={STEP_TIME_ANP-STEP_TIME_NOANP:+.2f}s ({(STEP_TIME_ANP/STEP_TIME_NOANP-1)*100:+.1f}%)")
    nccl_per_step_noanp = sum(by_scope_total_noanp.values()) / 1000.0 / STEPS_PROFILED  # s/step
    nccl_per_step_anp = sum(by_scope_total_anp.values()) / 1000.0 / STEPS_PROFILED
    print(f"  NCCL kernel time per step (one rank, summed over all collectives):")
    print(f"     noANP: {nccl_per_step_noanp:.2f}s   ANP: {nccl_per_step_anp:.2f}s   Δ_unmasked: {nccl_per_step_anp - nccl_per_step_noanp:+.2f}s")
    print(f"  Δ_NCCL_per_step / Δ_step = {(nccl_per_step_anp - nccl_per_step_noanp) / (STEP_TIME_ANP - STEP_TIME_NOANP) * 100:.1f}% accounted for by added NCCL kernel time on this rank")


if __name__ == "__main__":
    main()
