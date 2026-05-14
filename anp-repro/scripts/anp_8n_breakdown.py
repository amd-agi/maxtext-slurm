"""Per-NCCL-op breakdown for the 8N MoE noANP vs ANP pair (jobs 9501 vs 9511).

Reads xplane.pb from one rank in each run, groups all `ncclDevKernel*` events by
the HLO op that emitted them, and prints aggregated time per HLO op so we can
see which collectives drive the perf delta.
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


def find_xplane(run_dir: str, host: str) -> str:
    pat = os.path.join(run_dir, "ds-proxy-se0-e256-h4096", "tensorboard", "plugins", "profile", "*", f"{host}.xplane.pb")
    matches = glob.glob(pat)
    if not matches:
        raise FileNotFoundError(pat)
    return matches[0]


def load_events(xplane_path: str):
    data, _ = xp.xspace_to_tools_data([xplane_path], "trace_viewer@", {})
    obj = json.loads(data)
    return obj.get("traceEvents", [])


def aggregate_nccl_by_hlo(events):
    """Returns dict[hlo_op] -> list of durations (us). Only ncclDevKernel events."""
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
    runs = {
        "noANP_9501": find_xplane(BASE_NOANP, "chi2865"),
        "ANP_9511":   find_xplane(BASE_ANP,  "chi2865"),
    }

    summaries = {}
    for tag, path in runs.items():
        sys.stderr.write(f"[{tag}] loading {path}\n")
        evs = load_events(path)
        sys.stderr.write(f"[{tag}] {len(evs)} events\n")
        summaries[tag] = aggregate_nccl_by_hlo(evs)

    print("\n========================================================================")
    print("PER-HLO-OP NCCL DURATIONS (one rank, all profiled steps)")
    print("Columns: hlo_op | n | sum_ms | mean_ms | min_ms | max_ms | cv%")
    print("========================================================================")
    for tag, by_hlo in summaries.items():
        print(f"\n--- {tag} ---")
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
                std_pct = 100.0 * std / m if m else 0.0
            else:
                std_pct = 0.0
            print(f"  {hlo:<60s} n={n:5d} sum={sum_ms:9.1f} mean={mean_ms:8.2f} min={mn:8.2f} max={mx:8.2f} cv={std_pct:5.1f}%")

    print("\n========================================================================")
    print("DELTA: ANP - noANP, sorted by aggregated added time")
    print("Columns: hlo_op | n_noANP n_ANP | sum_noANP sum_ANP delta(ms) ratio | mean_noANP mean_ANP delta_per_call(ms)")
    print("========================================================================")
    common = set(summaries["noANP_9501"]) | set(summaries["ANP_9511"])
    rows = []
    for hlo in common:
        a = summaries["noANP_9501"].get(hlo, [])
        b = summaries["ANP_9511"].get(hlo, [])
        a_sum = sum(a)/1e3
        b_sum = sum(b)/1e3
        rows.append((hlo, len(a), len(b), a_sum, b_sum, b_sum - a_sum))
    rows.sort(key=lambda r: -r[5])
    for hlo, na, nb, a_sum, b_sum, delta in rows[:30]:
        a_mean = (a_sum / na) if na else 0
        b_mean = (b_sum / nb) if nb else 0
        ratio = (b_sum / a_sum) if a_sum > 0 else float("inf")
        print(f"  {hlo:<60s} n=({na:4d},{nb:4d}) sum=({a_sum:8.1f}->{b_sum:8.1f}) Δ={delta:+8.1f} r={ratio:5.2f}x  mean=({a_mean:7.2f}->{b_mean:7.2f}) Δ={b_mean-a_mean:+6.2f}")

    print("\n=== TOTAL NCCL kernel time per rank (one rank, all profiled events) ===")
    for tag, by_hlo in summaries.items():
        total = sum(sum(v) for v in by_hlo.values()) / 1e3
        n = sum(len(v) for v in by_hlo.values())
        print(f"  {tag:<14s} : {total:9.1f} ms across {n} kernel invocations")


if __name__ == "__main__":
    main()
