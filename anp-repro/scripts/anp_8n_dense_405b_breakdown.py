"""Per-NCCL-op breakdown for the 8N full-126L llama3.1-405b NoANP profiled run.

Job 13803 (noANP, FP8, pdbs=5, MTL=8192). Only chi2766 was profiled.
We have NO matching ANP run for full 405b on 8N (every WithANP attempt crashed
at step 0). This script extracts the per-op call counts and per-op durations
from the noANP trace, so we can talk about which collectives a per-op-time
table would actually correspond to.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

from xprof.convert import _pywrap_profiler_plugin as xp


BASE_NOANP = "/mnt/vast/qiangh/run00/maxtext-slurm/outputs/13803-JAX-llama3.1-405b-profiler_xplane-steps_15-_env_ENABLE_XLA_DUMP_1-FP8-8N-pdbs5-TGS_668.201"
HOST = "chi2766"
HLO = os.path.join(BASE_NOANP, "xla_dump", "module_0059.jit_train_step.gfx950_gpu_after_optimizations.txt")


def find_xplane(run_dir, host):
    pat = os.path.join(run_dir, "llama3_405B_training", "tensorboard", "plugins", "profile", "*", f"{host}.xplane.pb")
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
        out[args.get("hlo_op") or "<unknown>"].append(float(e["dur"]))
    return out


def main():
    path = find_xplane(BASE_NOANP, HOST)
    sys.stderr.write(f"loading {path}\n")
    evs = json.loads(xp.xspace_to_tools_data([path], "trace_viewer@", {})[0]).get("traceEvents", [])
    sys.stderr.write(f"{len(evs)} events\n")
    by_hlo = aggregate_nccl_by_hlo(evs)

    print("\n========================================================================")
    print("PER-HLO-OP NCCL DURATIONS — 8N FULL llama3.1-405b NoANP (job 13803, chi2766)")
    print(f"  Step time (steady): 61.3 s/step, 668 TGS, MFU 33.7%")
    print(f"  Workload: 126 layers, FSDP-64, FP8, pdbs=5, MTL=8192")
    print("========================================================================")
    print(f"{'HLO op':<28s} {'n':>5s} {'sum_ms':>9s} {'mean_ms':>8s} {'min_ms':>8s} {'max_ms':>8s} {'cv%':>6s}")
    print("-" * 80)

    rows = sorted(by_hlo.items(), key=lambda kv: -sum(kv[1]))
    for hlo, durs_us in rows[:30]:
        n = len(durs_us)
        sum_ms = sum(durs_us) / 1e3
        mean_ms = sum_ms / n
        mn = min(durs_us) / 1e3
        mx = max(durs_us) / 1e3
        if n > 1:
            m = mean_ms
            var = sum((d/1e3 - m) ** 2 for d in durs_us) / (n - 1)
            std = var ** 0.5
            cv = 100.0 * std / m if m else 0.0
        else:
            cv = 0.0
        print(f"  {hlo:<28s} {n:5d} {sum_ms:9.1f} {mean_ms:8.2f} {mn:8.2f} {mx:8.2f} {cv:5.1f}")

    total = sum(sum(v) for v in by_hlo.values()) / 1e3
    n_total = sum(len(v) for v in by_hlo.values())
    print(f"\n  TOTAL: {total:.1f} ms across {n_total} kernel invocations")
    print(f"  Per profiled iteration (assuming N steps): see 'n' column / divide by N")


if __name__ == "__main__":
    main()
