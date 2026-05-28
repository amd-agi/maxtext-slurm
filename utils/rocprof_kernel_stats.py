#!/usr/bin/env python3
"""Aggregate rocprofv3 kernel_trace CSVs into per-kernel statistics.

Why this exists
---------------
rocprofv3 v1.0.0 in `rocm/jax-training:maxtext-v26.2` will NOT emit the
built-in `--stats --summary` tables and the per-domain `--<X>-trace` CSVs in
the same run (the former suppresses the latter). The wrapper in `_train.sh`
keeps per-domain traces because they are the ground truth for both Perfetto
visualization (`.pftrace`) and kernel statistics. This script derives kernel
statistics from the same `kernel_trace.csv` files by post-processing — no
second rocprofv3 run needed.

CSV schema (verified on job 14903 output)
-----------------------------------------
    Kind, Agent_Id, Queue_Id, Stream_Id, Thread_Id, Dispatch_Id, Kernel_Id,
    Kernel_Name, Correlation_Id, Start_Timestamp, End_Timestamp,
    LDS_Block_Size, Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count,
    Workgroup_Size_{X,Y,Z}, Grid_Size_{X,Y,Z}
Timestamps are nanoseconds.

Outputs
-------
- Prints a sorted table (top-N by total time) to stdout.
- Writes a single `kernel_stats.csv` per input scope (per file, per node, or
  cluster-wide depending on --group-by).

Usage
-----
    # All CSVs under one job's rocprof tree, cluster-wide aggregation:
    python3 utils/rocprof_kernel_stats.py /perf_apps/.../14903-.../rocprof

    # Per-node summaries (one stats CSV per hostname dir):
    python3 utils/rocprof_kernel_stats.py /perf_apps/.../14903-.../rocprof --group-by node

    # Single file:
    python3 utils/rocprof_kernel_stats.py path/to/<pid>_kernel_trace.csv

    # Top-50 with mangled rocBLAS Tensile names shortened:
    python3 utils/rocprof_kernel_stats.py <path> --top 50 --shorten
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# rocBLAS Tensile kernels look like
# `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x32_MI16x16x1_SN_...`
# with a hundred-char suffix encoding tile shapes / flags. For grouping, the
# `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs` prefix uniquely identifies the
# GEMM kernel family; the suffix is parameter tuning that we collapse.
_TENSILE_RE = re.compile(r"^(Cijk_[A-Za-z_]+_UserArgs)_.*$")
# C++-mangled stream_executor / XLA kernels include the full argument list,
# which clutters output but does not change identity. Strip "(...)" tail.
_CXX_ARGS_RE = re.compile(r"\(.*$")


def shorten_kernel_name(name: str) -> str:
    """Collapse the parameter-encoding tail of rocBLAS Tensile / mangled names."""
    m = _TENSILE_RE.match(name)
    if m:
        return m.group(1) + "_*"
    return _CXX_ARGS_RE.sub("", name)


def iter_kernel_csvs(path: Path):
    """Yield kernel_trace.csv files under `path` (or just `path` if it is one)."""
    if path.is_file():
        if path.name.endswith("kernel_trace.csv"):
            yield path
        return
    yield from sorted(path.rglob("*_kernel_trace.csv"))


def aggregate(files: list[Path], shorten: bool):
    """Return (per_name -> dict(count,total,min,max), grand_total_ns)."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total_ns": 0, "min_ns": None, "max_ns": 0}
    )
    grand_total_ns = 0
    for f in files:
        with f.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Defensive: rocprofv3 may emit non-KERNEL_DISPATCH rows in
                # future versions. Skip anything else.
                if row.get("Kind") and row["Kind"] != "KERNEL_DISPATCH":
                    continue
                try:
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                dur = end - start
                if dur < 0:  # corrupt row, skip
                    continue
                name = row.get("Kernel_Name", "<unknown>")
                if shorten:
                    name = shorten_kernel_name(name)
                s = stats[name]
                s["count"] += 1
                s["total_ns"] += dur
                s["min_ns"] = dur if s["min_ns"] is None else min(s["min_ns"], dur)
                s["max_ns"] = max(s["max_ns"], dur)
                grand_total_ns += dur
    return stats, grand_total_ns


def fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.3f} us"
    return f"{ns} ns"


def print_table(stats, grand_total, top, label):
    rows = sorted(stats.items(), key=lambda kv: kv[1]["total_ns"], reverse=True)
    if top > 0:
        rows = rows[:top]
    print(f"\n=== {label} ===")
    print(f"  files aggregated  : {label}")
    print(f"  unique kernels    : {len(stats)}")
    print(f"  total dispatches  : {sum(v['count'] for v in stats.values())}")
    print(f"  total GPU time    : {fmt_ns(grand_total)}\n")
    header = f"  {'%':>6}  {'count':>9}  {'total':>11}  {'avg':>10}  {'min':>10}  {'max':>10}  kernel"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, v in rows:
        pct = 100.0 * v["total_ns"] / grand_total if grand_total else 0.0
        avg = v["total_ns"] // v["count"] if v["count"] else 0
        print(
            f"  {pct:>5.2f}%  {v['count']:>9}  {fmt_ns(v['total_ns']):>11}  "
            f"{fmt_ns(avg):>10}  {fmt_ns(v['min_ns'] or 0):>10}  {fmt_ns(v['max_ns']):>10}  {name}"
        )


def write_csv(stats, grand_total, out_path: Path):
    rows = sorted(stats.items(), key=lambda kv: kv[1]["total_ns"], reverse=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["kernel_name", "count", "total_ns", "avg_ns", "min_ns", "max_ns", "pct_of_total"]
        )
        for name, v in rows:
            avg = v["total_ns"] // v["count"] if v["count"] else 0
            pct = 100.0 * v["total_ns"] / grand_total if grand_total else 0.0
            w.writerow([name, v["count"], v["total_ns"], avg, v["min_ns"] or 0, v["max_ns"], f"{pct:.4f}"])
    print(f"  wrote summary: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="kernel_trace.csv file, a node dir, or a job's rocprof root")
    ap.add_argument("--top", type=int, default=30, help="rows to print (0=all). Default 30.")
    ap.add_argument(
        "--shorten",
        action="store_true",
        help="collapse rocBLAS Tensile param suffix and C++ arg lists for grouping",
    )
    ap.add_argument(
        "--group-by",
        choices=("all", "node", "file"),
        default="all",
        help="all = one cluster-wide summary; node = per-hostname dir; file = per CSV",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="override output CSV path (only meaningful for --group-by all)",
    )
    args = ap.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(2)

    if args.group_by == "file":
        for f in iter_kernel_csvs(root):
            stats, total = aggregate([f], args.shorten)
            print_table(stats, total, args.top, str(f))
            write_csv(stats, total, f.with_name(f.stem.replace("kernel_trace", "kernel_stats") + ".csv"))
        return

    if args.group_by == "node":
        # Group files by the first dir component under `root` (= hostname dir).
        groups: dict[Path, list[Path]] = defaultdict(list)
        for f in iter_kernel_csvs(root):
            try:
                rel = f.relative_to(root)
                node_dir = root / rel.parts[0] if rel.parts else root
            except ValueError:
                node_dir = f.parent
            groups[node_dir].append(f)
        for node_dir, files in sorted(groups.items()):
            stats, total = aggregate(files, args.shorten)
            print_table(stats, total, args.top, f"{node_dir} ({len(files)} files)")
            write_csv(stats, total, node_dir / "kernel_stats.csv")
        return

    # group-by all (default)
    files = list(iter_kernel_csvs(root))
    if not files:
        print(f"error: no *_kernel_trace.csv found under {root}", file=sys.stderr)
        sys.exit(2)
    stats, total = aggregate(files, args.shorten)
    print_table(stats, total, args.top, f"{root} ({len(files)} files)")
    out = Path(args.out) if args.out else (root if root.is_dir() else root.parent) / "kernel_stats.csv"
    write_csv(stats, total, out)


if __name__ == "__main__":
    main()
