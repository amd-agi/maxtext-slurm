"""Validate that the noANP-vs-ANP NCCL delta pattern holds on multiple ranks.

For each of two hosts, compute the per-NCCL-op delta and the aggregated delta
by communicator scope. Confirms the analysis isn't an artifact of one rank.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

from xprof.convert import _pywrap_profiler_plugin as xp


BASE_NOANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9501-JAX-ds-proxy-se0-e256-h4096-WithOUTANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_3155.723"
BASE_ANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9511-JAX-ds-proxy-se0-e256-h4096-WithANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_2538.112"

# Scope tags by inspection of HLO module_0142.
SCOPE = {
    # GLOBAL-64 (replica_groups=[1,64]<=[64])
    "all-gather-start": "GLOBAL-64",
    "all-gather-start.2": "GLOBAL-64",
    "all-gather-start.4": "GLOBAL-64",
    "reduce-scatter.14": "GLOBAL-64",
    "reduce-scatter.16": "GLOBAL-64",
    "all-reduce-start": "GLOBAL-64",
    "all-reduce-start.1": "GLOBAL-64",
    "all-reduce-start.2": "GLOBAL-64",
    "all-reduce-start.3": "GLOBAL-64",
    "all-reduce-start.4": "GLOBAL-64",
    "all-reduce-start.5": "GLOBAL-64",
    # DCN-8 (replica_groups=[8,8]<=[8,8]T(1,0))
    "all-gather-start.1": "DCN-8",
    "all-gather-start.3": "DCN-8",
    "reduce-scatter.15": "DCN-8",
    # ICI-8 intra-node (replica_groups=[8,8]<=[64])
    "all-to-all.2.1": "ICI-8",
    "all-to-all.3.1": "ICI-8",
    "all-to-all.4.1": "ICI-8",
    "all-to-all.5.1": "ICI-8",
    "all-to-all.6.1": "ICI-8",
    "all-to-all.7.1": "ICI-8",
}

HOSTS = ["chi2865", "chi2870", "chi2880"]  # 1 from each "side" of the cluster, sample


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
    print("\n========================================================================")
    print("CROSS-RANK VALIDATION (per-host noANP vs ANP delta by communicator scope)")
    print("========================================================================")
    grand = defaultdict(lambda: defaultdict(list))  # grand[host][scope] -> list of (noanp_sum_ms, anp_sum_ms)
    for host in HOSTS:
        try:
            xa = find_xplane(BASE_NOANP, host)
            xb = find_xplane(BASE_ANP, host)
        except FileNotFoundError as e:
            sys.stderr.write(f"missing xplane for {host}: {e}\n")
            continue
        ea = json.loads(xp.xspace_to_tools_data([xa], "trace_viewer@", {})[0]).get("traceEvents", [])
        eb = json.loads(xp.xspace_to_tools_data([xb], "trace_viewer@", {})[0]).get("traceEvents", [])
        a = aggregate_nccl_by_hlo(ea)
        b = aggregate_nccl_by_hlo(eb)
        scope_a = defaultdict(float)
        scope_b = defaultdict(float)
        for hop in set(a) | set(b):
            scope = SCOPE.get(hop, "OTHER")
            scope_a[scope] += sum(a.get(hop, [])) / 1e3
            scope_b[scope] += sum(b.get(hop, [])) / 1e3

        print(f"\n--- {host} ---")
        print(f"  {'scope':<14s} {'noANP_ms':>12s} {'ANP_ms':>12s} {'Δ_ms':>10s} {'ratio':>8s}")
        for scope in sorted(set(scope_a) | set(scope_b)):
            sa, sb = scope_a[scope], scope_b[scope]
            d = sb - sa
            r = (sb / sa) if sa > 0 else float("inf")
            print(f"  {scope:<14s} {sa:12.1f} {sb:12.1f} {d:+10.1f} {r:7.2f}x")
        ta = sum(scope_a.values())
        tb = sum(scope_b.values())
        print(f"  {'TOTAL':<14s} {ta:12.1f} {tb:12.1f} {tb-ta:+10.1f} {(tb/ta if ta else 0):7.2f}x")


if __name__ == "__main__":
    main()
