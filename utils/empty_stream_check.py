#!/usr/bin/env python3
"""Detect the RCCL 'empty stream' problem from a rocprofv3 kernel_trace.csv.

Symptom (old, pre-PR#2063): thousands of distinct GPU streams each carrying
only 1-2 kernels (typically copy/fill blit kernels), because RCCL created and
destroyed a fresh side stream per op. Fixed -> a small number of long-lived
streams, each with many kernels.

Analyzes per (Agent_Id, Stream_Id) since stream ids are per-GPU.
"""
import csv, sys, glob, collections

flags = {a for a in sys.argv[1:] if a.startswith("--")}
paths = []
for a in sys.argv[1:]:
    if a.startswith("--"):
        continue
    paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
if not paths:
    print("usage: empty_stream_check.py <kernel_trace.csv | glob>"); sys.exit(2)

csv.field_size_limit(1 << 30)
for path in paths:
    # (agent, stream) -> kernel count ; and name tally for small streams later
    per_stream = collections.Counter()
    agents = set()
    queues = set()
    rows = 0
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        cols = r.fieldnames
        for row in r:
            if row.get("Kind") and row["Kind"] != "KERNEL_DISPATCH":
                continue
            rows += 1
            ag = row.get("Agent_Id")
            sid = row.get("Stream_Id")
            per_stream[(ag, sid)] += 1
            agents.add(ag)
            queues.add((ag, row.get("Queue_Id")))

    n_streams = len(per_stream)
    n_agents = len(agents)
    # histogram of kernels-per-stream
    buckets = collections.Counter()
    def b(c):
        if c == 1: return "1"
        if c == 2: return "2"
        if c <= 10: return "3-10"
        if c <= 100: return "11-100"
        if c <= 1000: return "101-1000"
        return ">1000"
    for c in per_stream.values():
        buckets[b(c)] += 1
    n_le2 = sum(1 for c in per_stream.values() if c <= 2)

    print("="*78)
    print(f"FILE: {path}")
    print(f"  columns           : {cols}")
    print(f"  kernel rows        : {rows:,}")
    print(f"  distinct GPUs (Agent_Id)        : {n_agents}")
    print(f"  distinct HW queues (Agent,Queue): {len(queues)}")
    print(f"  distinct streams (Agent,Stream) : {n_streams:,}")
    print(f"  -> streams per GPU              : {n_streams/max(n_agents,1):.1f}")
    print(f"  streams with <=2 kernels        : {n_le2:,}  ({100*n_le2/max(n_streams,1):.1f}% of streams)")
    print(f"  kernels-per-stream histogram:")
    for k in ["1","2","3-10","11-100","101-1000",">1000"]:
        if buckets.get(k):
            print(f"      {k:>9} kernels : {buckets[k]:,} streams")
    # what do the tiny (<=2 kernel) streams run?
    if n_le2 and "--no-names" not in flags:
        tiny = {s for s,c in per_stream.items() if c <= 2}
        name_tally = collections.Counter()
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("Kind") and row["Kind"] != "KERNEL_DISPATCH":
                    continue
                if (row.get("Agent_Id"), row.get("Stream_Id")) in tiny:
                    nm = row.get("Kernel_Name","")
                    nm = nm.split("(")[0][:60]
                    name_tally[nm] += 1
        print(f"  top kernels in the <=2-kernel streams:")
        for nm, c in name_tally.most_common(8):
            print(f"      {c:>6}  {nm}")
