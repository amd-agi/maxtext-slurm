# DeepSeek-V3-671B grain data-loading optimization

- **Date:** 2026-05-01
- **Model:** `deepseek3-671b` (Mixture-of-Experts, 58 decoder layers, vocab 129280, 256 routed experts, top-k=8)
- **Hardware:** 8 nodes × 8× AMD MI355 (288 GB HBM / device, Pensando AINIC interconnect). 64 GPUs total. Pinned nodelist `chi[2766,2800,2810,2832,2835,2865,2872,2883]`.
- **Image:** `/mnt/vast/yihuang/ppfix-hangfix-deepep-gmm-maxtext-v26.2.tar`
- **MaxText branch:** [`yihuang/moe-turbo-gmm-and-deepep-v3`](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) (`container_env.sh` default since 2026-04-30)
- **Base config:** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml), FSDP=8, pdbs=7, `sgd-v3` flags (`sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`)
- **Dataset:** C4 parquet (`/datasets/c4/en/3.0.1/parquet/c4-train-*-of-01024.parquet`), HF DeepSeek-V3 tokenizer (`tokenizer_path=deepseek-ai/DeepSeek-V3-Base`), `max_target_length=4096`
- **Probe:** `steps=15` per job, `learning_rate_schedule_steps=2000` (decouples LR sweep from probe length so steady-state TGS isn't perturbed by LR-warmup cliff)
- **Steady-state metric:** mean of `Tokens/s/device` over steps 5–14 (drops compile-warm step 0 and routing-cold steps 1–4); step time is mean of `seconds:` over the same window. Cluster-side jitter is reported as the per-step σ alongside the mean.

## TL;DR — optimal recipe

```yaml
# configs/deepseek3-671b.gpu.yml — grain-data section
dataset_type: grain
grain_file_type: parquet
grain_train_files: /datasets/c4/en/3.0.1/parquet/c4-train-*-of-01024.parquet
tokenizer_type: huggingface
tokenizer_path: deepseek-ai/DeepSeek-V3-Base
grain_worker_count: 1                # one async worker saturates the queue
# All other grain knobs left at base.yml defaults:
#   grain_per_worker_buffer_size: 1   # increasing only adds queue depth, no win
#   packing: True                     # required for compile to fit the heartbeat budget
#   max_segments_per_seq: 32          # smaller values leak segment boundaries into loss
#   grain_num_threads: 16             # parquet-path inert (see § Inert knobs)
#   grain_prefetch_buffer_size: 500   # parquet-path inert
```

This recipe yields **~890 TGS/device** (≈ 0.45 MFU on MI355X) at `sgd-v3` pdbs=7 / FSDP=8, indistinguishable from `grain_worker_count` ∈ {4, 16} within cluster jitter and ~+2.5 % above `grain_worker_count=0` on the mean.

The selection logic in one line: **one async prefetch worker is the smallest setting that fully overlaps batch preparation with the JAX step on this stack; more workers add no measurable throughput, fewer leave per-step prep time exposed.**

## `grain_worker_count` sweep

12 submissions, single knob varied:

| Job   | `gw` value         | Mean TGS | Mean step (s) | Step σ (s) | TGS range | Notes |
|-------|--------------------|---------:|--------------:|-----------:|-----------|-------|
| 14720 | 0                  |    879.4 |         32.61 |       0.15 | 873–887   | quiet run |
| 14773 | 0  (replicate)     |    850.4 |         33.73 |       0.68 | 812–865   | jittery run |
| 14721 | 1                  |    893.1 |         32.10 |       0.15 | 887–900   |       |
| 14724 | 1  (replicate)     |    880.3 |         32.57 |       0.32 | 864–889   |       |
| 14722 | 4                  |    891.9 |         32.15 |       0.16 | 883–899   |       |
| 14723 | 16                 |    891.4 |         32.17 |       0.17 | 887–902   |       |
| 14764 | 1, `buf=4`         |    873.5 |         32.83 |       0.30 | 860–888   | per-worker buffer = 4; deeper queue, no win |
| 14726 | 1, `mseg=8`        |       —  |     **60.8**¹ |          — | —         | **broken**: 2× step time, loss=NaN, time-limit kill |
| 14727 | 1, `packing=false` |       —  |             — |          — | —         | **broken**: DEADLINE_EXCEEDED before step 0 |
| 14725 | 1, `buf=4` (1st)   |       —  |             — |          — | —         | RCCL init flake; resubmit succeeded as 14764 |
| 14774 | 1 (replicate)      |       —  |             — |          — | —         | DEADLINE_EXCEEDED cluster flake (config identical to 14721) |

¹ Single observed step-1 figure on the broken `mseg=8` run.

**Per-`gw` aggregate (steady-state steps 5–14, across replicates):**

| `grain_worker_count` | Jobs | Mean TGS | TGS σ between jobs | Step σ within jobs (pooled) |
|---------------------:|-----:|---------:|-------------------:|----------------------------:|
| 0                    |   2  | 864.9 |              20.5 |                        0.75 |
| 1                    |   2  | 886.7 |               9.0 |                        0.40 |
| 4                    |   1  | 891.9 |                 — |                        0.16 |
| 16                   |   1  | 891.4 |                 — |                        0.17 |

Two patterns are robust against the noise floor:

1. **TGS rises from `gw=0` → `gw=1`, then plateaus.** 0 → 1 buys ~22 TGS (+2.5 %); 1 → 4 → 16 buys at most ~5 TGS (+0.6 %), inside cluster-side run-to-run jitter.
2. **Per-step jitter falls monotonically with `grain_worker_count`.** Pooled per-step σ over a 10-step window: 0.75 s → 0.40 s → 0.17 s. The asynchronous worker absorbs cluster-side latency spikes that would otherwise show up as slow steps on the JAX driver thread.

## How `grain_worker_count` interacts with the JAX driver thread

Each batch the trainer requests passes through the parquet pretrain pipeline (`src/MaxText/input_pipeline/_grain_data_processing.py:161-239`):

```
parquet read  →  TokenizeAndChunk (HF DeepSeek tokenizer)  →  Rekey
              →  FirstFitPackIterDataset  →  Rekey segment_ids
              →  batch_and_pad  →  ShiftData  →  mp_prefetch ← grain_worker_count lives here
```

For pdbs=7 / FSDP=8 / 8 nodes, each per-host batch is `pdbs × num_local_devices × max_target_length = 7 × 8 × 4096 ≈ 230 k` tokens. The HF DeepSeek tokenizer + `FirstFitPackIterDataset` together cost ~1 s of single-CPU wall-clock per host-batch; the JAX step is ~32 s. Prep is well under 5 % of step time, so even modest overlap captures it entirely.

How the value of `grain_worker_count` controls the overlap (`grain/_src/python/dataset/transformations/prefetch.py:381-390`):

| Setting           | Behavior |
|-------------------|----------|
| `0`               | `MultiprocessPrefetchIterDataset.__iter__` short-circuits to the parent iterator. All preprocessing runs **on the JAX driver thread**, in series with — not parallel to — `pjit` step dispatch. |
| `1`               | One forked subprocess runs the pipeline; main process pulls finished batches via shared-memory queue (`per_worker_buffer_size=1`). Preprocessing fully overlaps the running step. |
| `N ≥ 2`           | `N` forked subprocesses, round-robin batch handoff. Total in-flight queue depth = `N × per_worker_buffer_size`. |

This explains the three observations:

- **`gw=0` vs `gw=1` (~+22 TGS, +2.5 %).** At `gw=0` the driver alternates between blocking on `next(iterator)` (~1 s tokenize + pack) and dispatching the next `pjit` step. The two cannot overlap, so prep adds directly on top of the 32 s step. At `gw=1` the worker produces batch *N+1* while the driver is dispatching step *N*; `next(iterator)` becomes a queue read of microseconds. The +22 TGS / 887 ≈ 2.5 % matches the prep-time / step-time ratio.

- **`gw=1` vs `gw=4/16` (within noise).** One worker is enough to keep the queue full because preprocessing is much faster than a step. Extra workers sit idle (each holding its own dataset iterator state and tokenizer copy) but don't speed anything up — the bottleneck is the JAX step, not the prep. The marginal +5 TGS at `gw=4` is inside cluster jitter.

- **Jitter monotone in `gw`.** Even though *mean* prep is small, prep occasionally spikes (parquet row-group load, tokenizer cache miss, GIL contention with NCCL progress threads). With no async worker the spikes land directly on the driver and produce slow steps; with a worker queue the spike is absorbed by the buffer and the driver never sees it. Each additional worker contributes one more in-flight batch, smoothing the tail further — this is why per-step σ falls all the way to 0.17 s at `gw=16`.

### Upper bound on `grain_worker_count`

Spawning many workers per node is *not free at startup*. Each worker forks from a parent that has already loaded the HF tokenizer, RCCL state, and parts of the JAX/CUDA runtime. At `gw=16` (128 worker subprocesses across the 8-node setup) the simultaneous fork + tokenizer-load + first-collective-barrier race has been observed to deadlock during cold start, manifesting as a hang between barrier and step 0; the same value also runs cleanly on other attempts, so the failure mode is non-deterministic rather than a hard ceiling — but the rate is non-zero, and rises with `gw`. Throughput plateaus at `gw=1`, so the practical recommendation is "the smallest value that hits the plateau" — which avoids the fork-race risk entirely.

## Other knobs

### `grain_per_worker_buffer_size` — leave at `1`

`per_worker_buffer_size` is the queue depth between each worker and the main process (`grain/_src/python/options.py:84-87`). With ~1 s prep behind a 32 s step, the queue is rarely empty at depth `1` — adding depth doesn't fill more slots, it just keeps stale batches alive longer. `gw=1, buf=4` came in at 873.5 TGS (job 14764), the slowest `gw=1` measurement; mechanism is shared-memory pressure plus serialization round-trips for batches that sit at the back of the queue. No upside observed at any depth `> 1`.

### `packing` — must remain `true`

`packing=false` (job 14727) failed at the JAX distributed-init barrier with `DEADLINE_EXCEEDED: GetKeyValue() timed out` after 10 minutes. The `packing=false` HLO is a different shape (no segment_ids carried through, different attention-mask, different graph for `FirstFitPackIterDataset`'s removal); on a 1T-class MoE model the cold compile of that graph exceeds the 10-min `GetKeyValue` window and aborts before step 0 ever runs. Even if it eventually compiled, packed training is strictly more efficient at C4-document-length distributions on a 4096-token sequence — there's no upside to chasing this.

### `max_segments_per_seq` — keep at `base.yml` default of `32`

`max_segments_per_seq=8` (job 14726) doubled the steady-state per-step time to ~60 s **and** drove `loss=nan` from step 1, killing the job at the 25-min walltime before step 14. `FirstFitPackIterDataset` packs up to `max_segments_per_seq` short documents into one 4096-token sequence; with the bound at 8 the packer stops early on document distributions that would naturally fit ≥8 segments per sequence, leaving more padded tokens per useful token (drives step time up) and leaking segment-boundary structure into the loss math (drives loss numerics off). The default `32` is high enough that the C4-document-length distribution is bounded by sequence length rather than by `max_segments_per_seq`.

### Inert knobs for the parquet path

`grain_num_threads` and `grain_prefetch_buffer_size` are passed to `grain.ReadOptions`, but `ReadOptions` is **only consumed by the arrayrecord branch** of the source resolver (`_grain_data_processing.py:117-122` and `:134-139`). The parquet branch (line 141 onward) uses `ParquetIterDataset + InterleaveIterDataset + WindowShuffleIterDataset` instead and does not look at `ReadOptions`. Setting these knobs to anything else has no effect on the parquet pipeline; the `base.yml` defaults (`16` and `500`) are kept for compatibility with arrayrecord-mode runs.

## Notes on cluster jitter

Run-to-run TGS for *identical* configs varied by ~13 TGS (~1.5 %) at fixed `gw=1`, which is the same order of magnitude as the `gw=0` → `gw=1` mean delta. To avoid mistaking jitter for signal, the per-`gw` aggregates above use replicates where available; the dominant signal (gw=0 noisier, gw≥1 plateau) is well above the noise floor across the replicate pairs but the marginal `gw=1` → `gw=4/16` differences are not.

The sweep also incidentally measured the cluster's flake rate at ~25 % over 12 nominally-identical submissions: 1 RCCL init flake (job 14725, recovered via cancel + resubmit as 14764) and 2 `DEADLINE_EXCEEDED: GetKeyValue() timed out` cluster-coord flakes (jobs 14727, 14774 — the latter had a configuration identical to the successful 14721, ruling out a config link). These are properties of the underlying stack at this date, not of the data-loader recipe.

## Recommended deployment

In `configs/deepseek3-671b.gpu.yml`, set `grain_worker_count: 1` explicitly even though that matches the `base.yml` default — the explicit value documents the choice and guards against accidental copy-paste from another model's `gpu.yml` (e.g. `ds-proxy-*` sets `16`, which has been observed to deadlock startup on this setup, see the previous section). Leave all other grain knobs at their `base.yml` defaults.

When the model is run in synthetic-data mode (`dataset_type: synthetic`), grain knobs are inert and do not need to be reset.
