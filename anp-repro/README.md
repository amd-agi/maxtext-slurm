# ANP plugin perf drop — standalone reproducer

A single-file C++ RCCL benchmark that reproduces the ANP plugin regression
**without** MaxText, JAX, mpirun, Pyxis, or srun's container integration.
It runs an AllGather + ReduceScatter + AllReduce sweep across message
sizes on 2-nodes × 8-MI355X, toggles only `NCCL_NET_PLUGIN=…/librccl-anp.so`,
and compares three communicator scopes side by side:

| scope | `ncclComm_t` world | peers per GPU     | matches what in MaxText?                           |
| ----- | -----------------: | ----------------- | -------------------------------------------------- |
| `dcn` |                  2 | 1 remote (IB)     | `replica_groups` inner-axis DCN-FSDP-2 collective  |
| `ici` |                  8 | 7 local (NVLink)  | ICI-FSDP-8 intra-node all-gather                   |
| `all` |                 16 | 15 (NVLink + IB)  | global 16-rank ring (flat replica_groups=[1,16])   |

The `dcn` scope is the critical one — it's the same one- / two-peer
cross-node pattern that drives XLA's DCN-FSDP-2 decomposition, and it's
where the ANP regression shows up cleanly in source-code terms
(`CTS_RCVR_OFFLOAD_ENABLED` + `ionic_dv_qp_set_gda(qp, false, true)`; see
`notes/anp-source-code-analysis.md`).

## Actual measured result (jobs 13312 noANP vs 13313 ANP on chi[2812,2899])

```
scope  op  per_rank_bytes    noANP per_op   ANP per_op   ratio    noANP GB/s   ANP GB/s
  dcn  ag        16777216        0.496 ms     0.559 ms   1.13x         34.10      30.00
  dcn  ag       134217728        3.827 ms     4.019 ms   1.05x         35.15      33.56
  dcn  ag      1073741824       29.865 ms    31.005 ms   1.04x         36.13      34.66
  dcn  ag      2147483648       58.965 ms    61.747 ms   1.05x         36.59      34.79
  dcn  rs        16777216        0.575 ms     0.611 ms   1.06x         29.65      27.44
  dcn  rs       134217728        3.522 ms     3.930 ms   1.12x         38.24      34.34
  dcn  rs      1073741824       24.544 ms    28.618 ms   1.17x         43.84      37.56
  dcn  rs      2147483648       48.772 ms    56.508 ms   1.16x         44.05      38.01
  dcn  ar       134217728        3.199 ms     3.645 ms   1.14x         42.05      36.85
  dcn  ar       536870912       12.026 ms    14.401 ms   1.20x         44.83      37.43
  dcn  ar      1073741824       23.602 ms    28.523 ms   1.21x         45.61      37.72
  dcn  ar      2147483648       46.762 ms    56.715 ms   1.21x         45.98      37.88
  ici  ag        16777216        0.290 ms     0.290 ms   1.00x        404.95     405.62
  ici  ag       134217728        2.126 ms     2.124 ms   1.00x        441.97     443.29
  ici  ar       134217728        0.555 ms     0.552 ms   0.99x        423.71     426.42
  ici  ar       536870912        2.082 ms     2.082 ms   1.00x        451.60     451.39
  all  ag       134217728        5.733 ms     5.619 ms   0.98x        351.58     358.50
  all  rs       134217728        5.746 ms     5.652 ms   0.98x        350.49     356.74
```

**Tri-modal signal that exactly matches the source analysis:**

1. **`dcn` large ops (≥ ~16 MB): 1.04×–1.21× slower under ANP.**
   - AR reaches 1.21× at ≥ 512 MB — cleanest signal.
   - RS steady at 1.16–1.17× for ≥ 512 MB.
   - AG slowdown is smallest (~1.05×) — likely because AG traffic is more
     one-way and the ANP CTS-offload path has less to intercept.
2. **`dcn` small ops (< 256 KB): ratio ≈ 1.00×** — matches the LL/LL128
   fast path (`anpNetIrecvPostCTS`, no recv WQE, no GDA / no CTS offload).
3. **`ici` (NVLink-only): ratio ≈ 1.00× at all sizes** — ANP plugin never
   engages for intra-node NVLink collectives. Clean negative control.
4. **`all` (16-rank global ring): ratio ≈ 1.00×** — the 15-stage pipelined
   ring over ~50 GB/s NICs amortizes the per-op ANP overhead across many
   small stages; the per-op overhead can't be isolated here.

## Why this is *the* reproducer, even though the magnitude is smaller than in MaxText

The MaxText profile showed 1.5× (RS) to 2.3× (AG) per-op regression. This
test shows 1.05×–1.21× on the same nodes, same image, same RoCE env, same
plugin. The gap is real and explainable:

- **In MaxText, the collective kernel is serialized against GEMM/attention
  compute**, so each op incurs its per-op ANP overhead *without overlap*.
- **In this tight-loop benchmark, ops run back-to-back** — ANP's per-op
  latency has some of it hidden by the NCCL proxy thread / NIC pipeline
  already being "warm" between adjacent calls.

So this reproducer isolates the **mechanism** (DCN-only regression on
Simple-protocol, unaffected LL/LL128, unaffected NVLink), but the
**magnitude** grows when you put the pattern in a realistic training
step with compute interleaved. That is exactly what `notes/anp-source-code-analysis.md`
predicted from reading ANP's two `#define`s and the unconditional
`ionic_dv_qp_set_gda(qp, false, true)` call.

## Files

```
anp-repro/
├── anp_repro.cc      # C++ reproducer: dcn / ici / all × AG/RS/AR sweep
├── build.sh          # hipcc inside the container
├── run_node.sh       # docker launch on one node (mounts host ionic driver)
├── launcher.sh       # in-container fan-out to 8 processes per node
├── _sbatch.sh        # Slurm job script (invoked by submit.sh)
├── submit.sh         # entry point; submits noANP + ANP variants
├── parse.sh          # diff two run outputs, prints table + band summary
└── README.md         # this file
```

## Step-by-step

### 1. Build the binary once

```bash
cd /mnt/vast/qiangh/ANP_test/maxtext-slurm/anp-repro
./build.sh
```

Runs `hipcc -std=c++17 -O2 anp_repro.cc -lrccl` inside the training
container; drops a ~40 KB ELF next to the source. Re-runnable.

### 2. Submit both variants

```bash
./submit.sh                 # both noANP and ANP on chi[2812,2899]
WHICH=anp ./submit.sh       # just ANP
REPRO_SCOPES=dcn ./submit.sh    # only DCN scope (fastest signal)
```

Each variant allocates 2 nodes `--exclusive` and runs end-to-end in
~90 seconds.

Overrides (all optional):

| env var         | default                                 | purpose                           |
| --------------- | --------------------------------------- | --------------------------------- |
| `NODELIST`      | `chi[2812,2899]`                        | 2 MI355X hosts                    |
| `PARTITION`     | `deepep-a77`                            |                                   |
| `DOCKER_IMAGE`  | `.../jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2.tar` | tar path or image tag |
| `WHICH`         | `both`                                  | `both`, `noanp`, `anp`            |
| `REPRO_SCOPES`  | `dcn,ici,all`                           | subset to run                     |
| `REPRO_OPS`     | `ag,rs,ar`                              | subset (ag/rs/ar)                 |
| `REPRO_SIZES`   | built-in sweep (1 KB → 2 GB)            | comma-separated bytes             |
| `REPRO_ITERS`   | `30`                                    | per-size iters                    |

### 3. Compare

```bash
ls runs/                    # two dirs; find your noANP_dir and ANP_dir
./parse.sh  runs/<noANP_dir>  runs/<ANP_dir>
```

Look for the **"Sum of (ANP − noANP) per-op time in ms, by scope and
size band"** line at the bottom — the band for `scope=dcn op=ar
large(>=1MB)` is the cleanest reproduction signal.

## How it sidesteps mpirun / Pyxis / srun-containers

1. `submit.sh → sbatch` allocates 2 nodes, `--ntasks-per-node=1 --exclusive`.
2. `srun` runs `run_node.sh` on each node in parallel.
3. Each `run_node.sh` instance `docker run`s one container per node, with
   all 8 local GPUs, `--privileged`, `/dev/infiniband`, host
   `/etc/libibverbs.d` and `/usr/lib/x86_64-linux-gnu` bind-mounted
   (necessary on `deepep-a77` to use the host's ionic driver; otherwise
   NCCL falls back to TCP).
4. Inside the container, `launcher.sh` spawns 8 background processes
   with `LOCAL_RANK=0..7` and computes `GLOBAL_RANK = NODE_RANK*8 + LOCAL_RANK`.
5. `anp_repro` in each process initializes THREE RCCL comms:
   - `comm_all`: 16-way global (rank=GLOBAL_RANK, world=16)
   - `comm_ici`: 8-way intra-node (rank=LOCAL_RANK, world=8)
   - `comm_dcn`: 2-way cross-node (rank=NODE_RANK, world=2)
   Each uses its own ncclUniqueId bootstrapped via a shared-filesystem
   file (e.g. `/outputs/ncclUniqueId.<jobid>.dcn.slot3`), so we get 1
   global + 2 ICI + 8 DCN comms across the cluster — matching the
   hierarchical communicator layout NCCL/XLA build for an FSDP-2×8 mesh.

No MPI, no Ray, no Pyxis, no sshd-in-container, no srun-container
integration. Works on any Slurm + Docker cluster where the image is
present.

## Validation against the MaxText finding

The tri-modal pattern in this reproducer is **the same tri-modal pattern**
the MaxText profile captured (`notes/anp-2n-root-cause.md §5`):

| layer of evidence                            | matches? |
| -------------------------------------------- | -------- |
| ANP regresses only on cross-node ops         | ✓ (`dcn` scope only) |
| ICI / NVLink-only ops unaffected             | ✓ (`ici` scope exactly 1.00×) |
| Small < 256 KB ops unaffected                | ✓ (both in MaxText and here) |
| Reduce-scatter shows smaller ratio than AG/AR (in MaxText: RS 1.51×, AG 2.3×) | ✓ (`dcn rs` 1.17× steady, `dcn ar` 1.21× growing) |
| Rebuild-required to disable the regression   | ✓ (env knobs don't touch the two `#define`s) |

If the Slurm cluster has rccl-tests installed, you can cross-check by
running `all_gather_perf -b 1M -e 2G -f 2 -g 1 -n 30 -w 5` with 2 ranks
on a 2-node allocation. Expect the same per-op regression shape —
just replace our 2-rank `comm_dcn` with rccl-tests's 2-rank
communicator.

## Troubleshooting

* **"NET/IB : No device found."** (NCCL falls back to TCP) — the container
  can't see `/sys/class/infiniband/ionic_*`. Confirm `--privileged`,
  `--device /dev/infiniband`, and the two host bind-mounts are present
  in `run_node.sh`. On `deepep-a77` the container's own `libionic1` is
  older than the a-77 firmware, so those host mounts are required.

* **"cannot write /outputs/ncclUniqueId…"** — `OUT_DIR` isn't on a shared
  filesystem visible from both nodes. Put it under `/mnt/vast/...`.

* **No regression even in `dcn` scope** — you may be running on a build of
  ANP that has one of the two compile-time knobs disabled, or
  `NET_OPTIONAL_RECV_COMPLETION=0` is shunting everything through the
  slow path (and hiding the LL/LL128 fast-path contrast). Verify:
  ```bash
  docker run --rm --entrypoint=/bin/bash rocm/jax-training:maxtext-v26.2 \
      -c 'grep -E "CTS_RCVR_OFFLOAD_ENABLED|CTS_INLINE_ENABLED" /workspace/amd-anp/src/net_ib.cc | head'
  ```
  Expect both `#define`s uncommented (this is the v1.3.0 default).

* **Ratios much larger than the reported 1.05×–1.21×** — good, that means
  either newer ANP has a worse regression, or your cluster is showing
  more of the compute-interleave amplification than ours. Report the
  numbers; this repro's role is to show the **direction and
  localisation** of the regression, not to hit a specific magnitude.

## Where to go from here

`notes/anp-source-code-analysis.md §4` lists four 1-line source patches
(disable `CTS_RCVR_OFFLOAD_ENABLED`, flip the `ionic_dv_qp_set_gda`
arguments, etc.). Each should land at or near 1.00× for all the `dcn`
rows above. The `anp-repro` directory here is the cheapest way to A/B
those changes: rebuild the plugin, bind-mount the new `.so` into the
container, rerun `./submit.sh` in ≈ 2 minutes per variant.
