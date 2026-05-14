"""Per-NCCL-op delta with HAND-CURATED byte counts and effective BW.

We previously derived bytes from the HLO regex, but the regex over-matched into
to_apply clones. Here we hard-code the per-op byte volumes from the HLO shapes
that I read out of module_0142, so the BW math is trustworthy.

All shapes come from /mnt/vast/.../9501-*/xla_dump/module_0142.jit_train_step.gfx950_gpu_after_optimizations.txt
(verified identical to 9511's HLO for the relevant collectives).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from xprof.convert import _pywrap_profiler_plugin as xp


BASE_NOANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9501-JAX-ds-proxy-se0-e256-h4096-WithOUTANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_3155.723"
BASE_ANP = "/mnt/vast/qiangh/clean/maxtext-slurm/outputs/9511-JAX-ds-proxy-se0-e256-h4096-WithANP-steps_15-dataset_type_synthetic-profiler_xplane-_env_ENABLE_XLA_DUMP_1-_env_NCCL_DEBUG_INFO-8N-pdbs24-TGS_2538.112"

# (scope, kind, full_output_bytes_per_op, per_rank_input_bytes, per_rank_output_bytes)
# bf16 = 2 bytes; values computed from the HLO shapes verbatim.
HLO_INFO = {
    # ---- GLOBAL-64 (replica_groups=[1,64]<=[64]) ----
    "all-gather-start": dict(scope="GLOBAL-64", kind="all-gather",
        # input per rank: bf16[64,32,128]+bf16[64,8,128]+bf16[64,8,128]+bf16[64,128,32]+bf16[64,256] = 2,654,208 B
        # output per rank: above × 64 = 169,869,312 B
        per_rank_in_b=2654208, per_rank_out_b=169869312, ranks=64,
        descr="FSDP weight AG (FWD), non-MoE attn weights"),
    "all-gather-start.2": dict(scope="GLOBAL-64", kind="all-gather",
        # same shapes as all-gather-start (BWD remat)
        per_rank_in_b=2654208, per_rank_out_b=169869312, ranks=64,
        descr="FSDP weight AG (BWD remat), non-MoE attn weights"),
    "all-gather-start.4": dict(scope="GLOBAL-64", kind="all-gather",
        # bf16[64,102400]×2 in -> bf16[4096,102400]×2 out
        per_rank_in_b=2 * 64*102400*2, per_rank_out_b=2 * 4096*102400*2, ranks=64,
        descr="Embedding weight AG (1× per step)"),
    "reduce-scatter.14": dict(scope="GLOBAL-64", kind="reduce-scatter",
        # output per rank: bf16[64,256]+bf16[64,128,8]+bf16[64,32,128]+bf16[64,128,32]+bf16[64,128,8] = 1,463,808 B
        # input per rank: 64x = 93,683,712 B
        per_rank_in_b=93683712, per_rank_out_b=1463808, ranks=64,
        descr="Gradient RS for non-MoE attn weights"),
    "reduce-scatter.16": dict(scope="GLOBAL-64", kind="reduce-scatter",
        # bf16[4096,102400]×2 in -> bf16[64,102400]×2 out
        per_rank_in_b=2 * 4096*102400*2, per_rank_out_b=2 * 64*102400*2, ranks=64,
        descr="Embedding gradient RS (1× per step)"),
    "all-reduce-start": dict(scope="GLOBAL-64", kind="all-reduce",
        per_rank_in_b=4, per_rank_out_b=4, ranks=64, descr="f32[] reduce_sum"),
    "all-reduce-start.1": dict(scope="GLOBAL-64", kind="all-reduce",
        # bf16[8192]
        per_rank_in_b=8192*2, per_rank_out_b=8192*2, ranks=64,
        descr="bf16[8192] all-reduce (16KB - boundary case)"),
    "all-reduce-start.2": dict(scope="GLOBAL-64", kind="all-reduce",
        per_rank_in_b=4, per_rank_out_b=4, ranks=64, descr="s32[] reduce_sum"),
    "all-reduce-start.3": dict(scope="GLOBAL-64", kind="all-reduce",
        per_rank_in_b=4096*2, per_rank_out_b=4096*2, ranks=64, descr="bf16[4096]"),
    "all-reduce-start.4": dict(scope="GLOBAL-64", kind="all-reduce",
        per_rank_in_b=10*4, per_rank_out_b=10*4, ranks=64, descr="f32[10]"),
    "all-reduce-start.5": dict(scope="GLOBAL-64", kind="all-reduce",
        per_rank_in_b=31*4, per_rank_out_b=31*4, ranks=64, descr="f32[31]"),
    # ---- DCN-8 (replica_groups=[8,8]<=[8,8]T(1,0)) ----
    "all-gather-start.1": dict(scope="DCN-8", kind="all-gather",
        # bf16[32,512,2048]×3 in (per rank) -> bf16[32,4096,2048]×3 out
        per_rank_in_b=3 * 32*512*2048*2, per_rank_out_b=3 * 32*4096*2048*2, ranks=8,
        descr="FSDP MoE expert weights AG (FWD), DCN-8 ring"),
    "all-gather-start.3": dict(scope="DCN-8", kind="all-gather",
        per_rank_in_b=3 * 32*512*2048*2, per_rank_out_b=3 * 32*4096*2048*2, ranks=8,
        descr="FSDP MoE expert weights AG (BWD remat), DCN-8 ring"),
    "reduce-scatter.15": dict(scope="DCN-8", kind="reduce-scatter",
        per_rank_in_b=3 * 32*4096*2048*2, per_rank_out_b=3 * 32*512*2048*2, ranks=8,
        descr="Gradient RS for MoE expert weights, DCN-8 ring"),
    # ---- ICI-8 intra-node (replica_groups=[8,8]<=[64]) ----
    "all-to-all.2.1": dict(scope="ICI-8", kind="all-to-all",
        # bf16[1,24,8,32,128,4096] = 1*24*8*32*128*4096 = 3,221,225,472 B per rank (3 GB)
        per_rank_in_b=1*24*8*32*128*4096*2, per_rank_out_b=1*24*8*32*128*4096*2, ranks=8,
        descr="MoE token dispatch (BWD remat), intra-node"),
    "all-to-all.3.1": dict(scope="ICI-8", kind="all-to-all",
        per_rank_in_b=1*24*4096*8*32*128*2, per_rank_out_b=1*24*4096*8*32*128*2, ranks=8,
        descr="MoE token combine (BWD), intra-node"),
    "all-to-all.4.1": dict(scope="ICI-8", kind="all-to-all",
        per_rank_in_b=1*8*24*32*128*4096*2, per_rank_out_b=1*8*24*32*128*4096*2, ranks=8,
        descr="MoE dispatch transpose (BWD), intra-node"),
    "all-to-all.5.1": dict(scope="ICI-8", kind="all-to-all",
        per_rank_in_b=32*1*8*24*128*4096*2, per_rank_out_b=32*1*8*24*128*4096*2, ranks=8,
        descr="MoE combine (BWD remat), intra-node"),
    "all-to-all.6.1": dict(scope="ICI-8", kind="all-to-all",
        per_rank_in_b=1*24*8*32*128*4096*2, per_rank_out_b=1*24*8*32*128*4096*2, ranks=8,
        descr="MoE token dispatch (FWD), intra-node"),
    "all-to-all.7.1": dict(scope="ICI-8", kind="all-to-all",
        per_rank_in_b=32*1*8*24*128*4096*2, per_rank_out_b=32*1*8*24*128*4096*2, ranks=8,
        descr="MoE combine (FWD), intra-node"),
}


def find_xplane(run_dir, host):
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
        out[args.get("hlo_op") or "<unknown>"].append(float(e["dur"]))
    return out


def main():
    runs = {
        "noANP": find_xplane(BASE_NOANP, "chi2865"),
        "ANP":   find_xplane(BASE_ANP,   "chi2865"),
    }
    summaries = {}
    for tag, path in runs.items():
        evs = json.loads(xp.xspace_to_tools_data([path], "trace_viewer@", {})[0]).get("traceEvents", [])
        summaries[tag] = aggregate_nccl_by_hlo(evs)

    print("\n========================================================================")
    print("PER-OP DELTA + EFFECTIVE BW (chi2865, all profiled iters)")
    print("Σ in cross-network bytes per rank uses the standard formula:")
    print("  AG: bytes = (R-1)/R * out_per_rank        (each rank receives R-1 shards)")
    print("  RS: bytes = (R-1)/R * in_per_rank         (each rank sends R-1 shards)")
    print("  AR: bytes = 2*(R-1)/R * msg_per_rank      (RS + AG combined)")
    print("  AlltoAll(intra-node): bytes = (R-1)/R * msg_per_rank  (over XGMI, not network)")
    print("========================================================================")
    print(f"{'HLO op':<22s} {'scope':<10s} {'kind':<14s} {'n':>4s} {'bytes/op':>10s} {'noANP_ms':>9s} {'noANP_GB/s':>12s} {'ANP_ms':>9s} {'ANP_GB/s':>10s} {'Δms':>9s} {'BWratio':>8s}")
    print("-" * 150)
    rows = []
    for hop, info in HLO_INFO.items():
        durs_a = summaries["noANP"].get(hop, [])
        durs_b = summaries["ANP"].get(hop, [])
        if not durs_a and not durs_b:
            continue
        n = len(durs_a)
        a_mean_ms = sum(durs_a)/len(durs_a)/1e3 if durs_a else 0
        b_mean_ms = sum(durs_b)/len(durs_b)/1e3 if durs_b else 0
        # bytes/rank that traverse the wire per op
        R = info["ranks"]
        if info["kind"] == "all-gather":
            bytes_per_op = (R - 1) / R * info["per_rank_out_b"]
        elif info["kind"] == "reduce-scatter":
            bytes_per_op = (R - 1) / R * info["per_rank_in_b"]
        elif info["kind"] == "all-reduce":
            bytes_per_op = 2 * (R - 1) / R * info["per_rank_in_b"]
        elif info["kind"] == "all-to-all":
            bytes_per_op = (R - 1) / R * info["per_rank_in_b"]
        else:
            bytes_per_op = 0
        a_bw = bytes_per_op / (a_mean_ms / 1000) / 1e9 if a_mean_ms > 0 else 0
        b_bw = bytes_per_op / (b_mean_ms / 1000) / 1e9 if b_mean_ms > 0 else 0
        bw_ratio = (a_bw / b_bw) if b_bw > 0 else float("inf")
        delta = (b_mean_ms - a_mean_ms)
        rows.append((hop, info, n, len(durs_b), bytes_per_op, a_mean_ms, a_bw, b_mean_ms, b_bw, delta, bw_ratio))
    rows.sort(key=lambda r: -r[9] * max(r[2], r[3]))  # by total added time (mean × n)
    for hop, info, na, nb, b_op, a_ms, a_bw, b_ms, b_bw, dms, br in rows:
        print(f"{hop:<22s} {info['scope']:<10s} {info['kind']:<14s} {nb:>4d} {b_op/1e6:>8.1f}MB {a_ms:>9.2f} {a_bw:>10.1f}GB/s {b_ms:>9.2f} {b_bw:>8.1f}GB/s {dms:>+9.2f} {br:>7.2f}x")

    print("\nNote: 'cross-network bytes' for GLOBAL-64 ops is the *logical* cross-rank traffic.")
    print("On 8 nodes the actual cross-NODE traffic is ~7/8 of that (intra-node phase eats 1/8).")
    print("On the DCN-8 rings, all 7 of 8 stages are cross-node.")


if __name__ == "__main__":
    main()
