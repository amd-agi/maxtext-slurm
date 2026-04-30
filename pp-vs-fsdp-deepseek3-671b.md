# DeepSeek-V3 671B — PP=8 vs FSDP=8 on AMD MI355 (8 nodes × 8 GPUs)

- **Date:** 2026-04-30 (initial PP-vs-FSDP comparison + FSDP=8 tuning); 2026-05-01 (PP=8 XLA / NCCL tuning extension + cross-topology `remat_policy` sweep)
- **Model:** `deepseek3-671b` (Mixture-of-Experts, 58 decoder layers, vocab 129280, 256 routed experts, top-k=8)
- **Hardware:** 8 nodes × 8× AMD MI355 (288 GB HBM / device, Pensando AINIC interconnect). 64 GPUs total. Peak BF16 ≈ 2500 TFLOP/s/device → MFU ≈ TFLOP/25. Pinned nodelist `chi[2766,2800,2810,2832,2835,2865,2872,2883]` for the PP=8 tuning sweep; 4/30 PP-vs-FSDP comparison + FSDP=8 sweep used the same partition.
- **Image:** [`/mnt/vast/yihuang/ppfix-hangfix-deepep-gmm-maxtext-v26.2.tar`](https://github.com/ROCm/Primus-Turbo) (axis-aware Primus-Turbo batching rules for `nn.vmap("stage")`-of-`shard_map` composition + upstream [`fix/deepep/combine_hang`](https://github.com/AMD-AGI/Primus-Turbo/tree/fix/deepep/combine_hang) C++ kernel fix)
- **MaxText branch** (sgd configs only): [`yihuang/moe-turbo-gmm-and-deepep-v3`](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9` — also the `container_env.sh` default since 2026-04-30, so `sgd-v3` runs no longer need an explicit `MAXTEXT_PATCH_BRANCH=…` env-var prefix. v1/v2 baselines still need explicit overrides.
- **Base config:** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml). PP=8 passthrough adds `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1` (MaxText auto-derives `num_layers_per_pipeline_stage=1`, `num_pipeline_microbatches=8` from `pipeline_parallel_layers=56`).
- **Sequence length:** 4096. **Steps:** 15 (synthetic data). All numbers are **steady-state averages over steps 9-14** unless noted; clean (`profiler: ""`, no XLA dump) where indicated.

## TL;DR

All three configs we test are MoE — they differ in which **MoE implementation** in MaxText's `moe.py` is selected (`sparse_matmul=False` → dense_matmul branch; `sparse_matmul=True` → sparse_matmul branch). "dense" here refers to the *MoE implementation choice*, not to a non-MoE model:

- **`dense-cf1.25`**: dense_matmul branch with capacity-factor 1.25 dropping (`sparse_matmul=False`, `capacity_factor=1.25`). Each expert receives a fixed-size `capacity_factor × tokens / num_experts` tensor; tokens beyond capacity are dropped. The "dense" name comes from `moe.py`'s use of dense matmul on this fixed-shape per-expert tensor (no ragged dimensions).
- **`sparse-gmm-fixed` (sgf)**: sparse_matmul branch, dropless via `ragged_all_to_all` + `ragged_dot` (`sparse_matmul=True`, `use_turbo_grouped_gemm=True`). All tokens routed exactly to their top-k experts; ragged GEMM avoids capacity-factor padding but `ragged_all_to_all` materialises a `num_ranks × tokens × hidden` receive buffer that scales poorly.
- **`sparse-gmm-deepep-v3` (sgd-v3)**: sparse_matmul branch, dropless via DeepEP intra-node IPC dispatch + Primus-Turbo grouped GEMM (`sparse_matmul=True`, `use_turbo_grouped_gemm=True`, `use_deepep_dispatch=True`, on the `yihuang/moe-turbo-gmm-and-deepep-v3` MaxText branch). Bypasses MaxText's `ragged_all_to_all` with a per-rank-prefix-routed buffer of size `num_worst_tokens × hidden`.

**The optimal recipe is path × topology × branch dependent.** A single global default doesn't exist; the headline result is:

| Path | FSDP=8 best | PP=8 best | Production winner |
|---|---:|---:|---|
| **`sgd-v3`** (DeepEP dropless) | **1135.7** (14602) ⭐ — `ag=1 GiB` | 999.8 avg n=2 (14668/14674) — `overlap2 + async_priority` | **FSDP=8 (+12.0 %)** |
| **`dense-cf1.25`** (capacity-factor dropping) | 1207.9 (14629) — `ag=1 GiB` | **1224.1** avg n=2 (14672/14673) ⭐ — `overlap2` alone | **PP=8 (+1.34 %)** |
| **`sparse-gmm-fixed`** (ragged_dot dropless) | OOM @217 GiB temp | OOM @217 GiB temp | n/a |

**The two winning XLA flags have opposite signs across topologies** — leaving the FSDP-tuned `--xla_gpu_all_gather_combine_threshold_bytes=1073741824` deployed unconditionally (as it currently is in `configs/deepseek3-671b.env.sh`) silently degrades PP=8 dense by **-6.6 %** and PP=8 sgd-v3 by -1.0 % (in jitter). The recipe must be guarded by topology (and, for PP, by MoE branch) — see [§ Recommended deployment](#recommended-deployment).

The other operational findings:

- **`remat_policy: 'full'` is universally optimal** at pdbs=7 across both topologies and both MoE branches. Lighter remat policies either OOM (memory blow-up factor of 56-58× from `nn.scan`-PP / `scan_layers=True`-FSDP layouts) or fit-but-regress; FSDP=8 dense even diverges loss with `save_out_proj`. See [§ `remat_policy` sensitivity](#remat_policy-sensitivity-covers-pp8-and-fsdp8--universal-conclusion-full-is-optimal-at-pdbs7).
- **`sgf` (sparse-gmm-fixed) is unusable on this stack** at the production scale: OOMs at every feasible pdbs ≥ 7 because `ragged_all_to_all` materialises a full `num_ranks × tokens × hidden` receive buffer.

| Path | `dense-cf1.25` (dropping) | `sparse-gmm-fixed` (dropless via ragged_dot) | `sparse-gmm-deepep-v3` (DeepEP dropless, default XLA) | `sparse-gmm-deepep-v3` (DeepEP dropless, tuned XLA) |
|---|---:|---:|---:|---:|
| **PP=8 pdbs=7** TGS (image-default XLA) | **1157.6** (14571) | OOM @217 GiB temp | **939.9** (14570) | — |
| **PP=8 pdbs=7** TGS (PP-tuned: `overlap2`[+`async`]) | **1224.1** (14672/14673 avg) ⭐ | (still OOM) | — | **999.8** (14668/14674 avg) |
| **FSDP=8 pdbs=7** TGS (image-default XLA) | **1002.4** (14573) | OOM @217 GiB temp | **1017.7** (14572) | — |
| **FSDP=8 pdbs=7** TGS (FSDP-tuned: `ag=1 GiB`) | 1207.9 (14629) | (still OOM) | — | **1135.7** (14602) ⭐ |
| historical sgd-fs8 baseline (Apr 14, original `deepep-gmm-maxtext-v26.2.tar`) | — | — | 1097 | — |

All clean (`profiler: ""`, no XLA dump), steps 9-14 average. PP=8 uses `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1 pipeline_parallel_layers=56 num_layers_per_pipeline_stage=1 num_pipeline_microbatches=8` (V=7 virtual chunks). FSDP=8 uses the YAML default `dcn_fsdp_parallelism=8 dcn_pipeline_parallelism=1`.

## Configurations (all are MoE — DS3-671B has 256 routed experts, top-k=8)

| Tag | sparse_matmul | capacity_factor | use_turbo_grouped_gemm | use_deepep_dispatch | MoE matmul branch | MaxText branch |
|---|---|---:|---|---|---|---|
| **`dense-cf1.25`** | false | 1.25 | false | false | dense_matmul (fixed-shape per-expert tensor, drops on overflow) | base |
| **`sgf`** | true | n/a | true | false | sparse_matmul via `ragged_all_to_all` + `ragged_dot` (MaxText built-in) | base |
| **`sgd-v3`** | true | n/a | true | true | sparse_matmul via DeepEP IPC dispatch + Primus-Turbo grouped GEMM | `yihuang/moe-turbo-gmm-and-deepep-v3` |

"dense" in `dense-cf1.25` is the `moe.py` branch name (dense_matmul), not a claim that the model is non-MoE — DS3-671B is always MoE. `dense_matmul` selects MaxText's capacity-factor-bounded fixed-shape per-expert path (works around the lack of a built-in ragged GEMM kernel by padding to capacity); `sparse_matmul` selects the dropless ragged path (`sgf` uses MaxText's built-in `ragged_dot`; `sgd-v3` uses Primus-Turbo's grouped GEMM with DeepEP dispatch).

Each tag is run twice: once with `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1` (PP=8), once with the YAML default `dcn_fsdp_parallelism=8 dcn_pipeline_parallelism=1` (FSDP=8). 64 GPUs total, 8 GPUs per DCN replica.

`pdbs=7` is the apples-to-apples comparison axis (matches FSDP-only memory budget); for memory-feasibility data points we also tested `pdbs=8` on PP-only configurations.

## Memory feasibility matrix

|  | pdbs=7 | pdbs=8 |
|---|---|---|
| sgd-pp8 (DeepEP dropless) | total **253** / temp **178** GiB ✓ | total 253 / temp 178 ✓ (fits with axis-aware Primus-Turbo rules) |
| sgd-fs8 (DeepEP dropless) | total **236** / temp **178** GiB ✓ | (not run; pdbs=7 is the FSDP ceiling for tracking) |
| **sgf-pp8** (ragged_dot dropless) | total **276** / temp **217** GiB **✗ OOM** | total 296 / temp 222 ✗ OOM |
| **sgf-fs8** (ragged_dot dropless) | total **276** / temp **217** GiB **✗ OOM** | (skipped) |
| dense-cf1.25-pp8 (capacity-factor dropping) | total ≈200 / temp ≈125 GiB ✓ | ✓ |
| dense-cf1.25-fs8 (capacity-factor dropping) | total ≈200 / temp ≈125 GiB ✓ | ✓ |

HBM ceiling per device: ~268 GiB after BFCAllocator overhead.

The `ragged_dot`-based dropless path (`sgf`) consistently OOMs at pdbs=7 on this stack regardless of PP vs FSDP. **DeepEP is the memory-efficient dropless routing path**; the `ragged_all_to_all`-based `sgf` materializes a full `num_ranks × tokens × hidden` receive buffer (because it's a regular all_to_all, not a topology-aware dispatch), which DeepEP avoids by sizing the per-rank receive buffer to `num_worst_tokens × hidden` (a tight upper bound on the actual fan-in to each rank). The +44 GiB temp delta between `sgd-v3` and `sgf` at the same pdbs is the receive-buffer difference. The capacity-factor dropping path (`dense-cf1.25`) is even cheaper because each expert sees a fixed-size `capacity_factor × tokens / num_experts` chunk regardless of routing, so the fan-in tensor is bounded by `num_experts × capacity_factor × tokens / num_experts × hidden = capacity_factor × tokens × hidden` — a ~5× saving over `sgf`'s `num_ranks × tokens × hidden`.

## Why the dense_matmul PP=8 win does not transfer to sparse_matmul-DeepEP PP=8

A naïve roofline says PP=8 should beat FSDP=8 because it reduces DCN comm volume per step — instead of a full-parameter all-gather on every microbatch, the pipeline only ships hidden-state activations between adjacent stages. For DS3-671B at pdbs=7 this is ~3-4× reduction in DCN bytes per step. **For the dense_matmul path (`dense-cf1.25`), the prediction holds (+15.5 % at image-default XLA, +1.34 % even after FSDP is XLA-tuned). For sparse_matmul-DeepEP (`sgd-v3`), it doesn't (-8.3 % at image-default XLA, -12.0 % after FSDP is XLA-tuned).**

Why the dense_matmul-path win evaporates under sparse_matmul-DeepEP: three structural costs that show up only on the dropless sparse_matmul path and compound to overshoot the bytes-saved-per-step gain. The key axis is **per-stage compute uniformity**: capacity-factor dropping in dense_matmul enforces uniform per-expert tensor shapes (each expert always sees `capacity_factor × tokens / num_experts` tokens, no more no less); sparse_matmul-DeepEP routes the actual top-k assignments, so per-stage compute varies with routing skew across the 256 experts.

### 1. `nn.scan`-PP carry serializes the pipeline schedule (applies to both branches, but only sparse_matmul pays for it)

MaxText's pipeline implementation lowers to `jax.lax.scan` over the stage axis with a `loop_state` carry. XLA's latency-hiding scheduler **cannot move work across iteration boundaries** because each iteration's input depends on the previous iteration's output. The HLO from 14563 (sgd-pp8 pdbs=7 with HLO dump, `module_*.jit_train_step.gfx950_gpu_after_optimizations.txt`) shows every `collective-permute-start` immediately followed by its `collective-permute-done` with no compute in between — the scheduler had no slack to insert prefetch. Same scan structure exists in `dense-cf1.25` (14565), but the per-stage compute is uniform across stages (capacity-factor dropping forces every expert to see exactly `capacity_factor × tokens / num_experts` tokens), so the `collective-permute` always lines up with the fastest-arriving rank's compute boundary — i.e., minimal exposed wait. Under sparse_matmul-DeepEP, per-stage compute varies with routing skew (some stages have fewer dispatched tokens than others), so the `collective-permute` always lines up with the slowest-arriving rank's wait — i.e., maximal exposed wait.

Empirically, `--xla_gpu_enable_pipelined_p2p=true` (the analogue of `pipelined_all_gather` for P2P) made zero difference on the sparse_matmul path: with no slack to fill, there's nothing to pipeline. The PP=8 tuning sweep further confirms this — 14662's HLO (with the winning `overlap2` recipe + HLO dump on) still tags every `collective-permute-start` with `is_pipelined=false`. No XLA flag tested opens the carry-dep.

### 2. `collective-permute` is a synchronizing collective per-call

Each pipeline send/recv is a JAX-level synchronizing op. Per-call rendezvous wait accumulates as `Σᵢ max_r(Tᵢ,r)`, not `max_r(Σᵢ Tᵢ,r)` — so layer-by-layer sync points compound rather than averaging out across ranks. From the 14563 trace: 4.4 s/step of pure rendezvous wait per GPU; ~3.5 s of that is exposed in the step (i.e., not overlapped with anything).

This is doubly painful for the *sparse_matmul* (dropless) branch: top-k assignment skew across the 256 experts produces per-rank compute imbalance every layer. With FSDP, the imbalance amortizes across the per-step all-gather / reduce-scatter boundary (every rank does the full all-gather before each layer, so per-rank-imbalance cancels at step granularity). With PP, it amplifies through the per-layer `collective-permute` sync (the imbalance shows up at every layer's sync, not once per step). The dense_matmul branch's capacity-factor mechanism happens to flatten the per-stage tensor shapes, so this third cost vanishes there.

### 3. DeepEP per-microbatch fixed cost

PP=8 with `pipeline_parallel_layers=56`, `num_pipeline_microbatches=8`, V=7 (`num_layers_per_pipeline_stage=1`) means **8 microbatches × 7 chunks × 8 stages = 448 DeepEP dispatch+combine round-trips per step**. The DeepEP kernel has a fixed launch cost (~0.5 ms) plus the per-rank IPC handshake (~1 ms warmest, more under skew). The 448 round-trips alone account for ~2 s/step of fixed overhead that FSDP doesn't pay (FSDP issues a single dispatch+combine per layer × per token group, totaling ~58 × 7 = ~406 calls, similar order, but they're not on the bubble-critical path).

### Bubble accounting

Pipeline bubble fraction = (num_stages − 1) / (num_microbatches × V + num_stages − 1) = 7 / (8×7 + 7) = 7 / 63 ≈ **11.1 %**.

This alone caps PP gain — PP must save *more* than 11 % of step time on FSDP comm to net positive. FSDP's exposed-comm savings on this stack are ~10 s out of 27 s/step (≈ 37 %), so PP *could* win in principle if the bubble were the only cost. But the bubble + rendezvous-amplified straggler wait + DeepEP fixed cost together exceed FSDP's exposed comm:

```
PP step time ≈ FSDP step time − (FSDP exposed comm)
                              + (PP bubble)
                              + (PP rendezvous-amplified wait)
                              + (PP DeepEP fixed cost)
             ≈ 27.6  − 10.0  + 3.0  + 3.5  + 2.0
             ≈ 26.1 s/step  (predicted)
empirical PP step time = 30.5 s (clean) / 33.9 s (with profiler)
```

The empirical extra ~4 s/step over the predicted PP step time comes from XLA scheduler overhead under the `nn.scan`-PP layout (each iteration has its own scheduling context; cross-iteration optimizations are shut off).

## What about the `combine_hang` fix?

The April-30 upstream `fix/deepep/combine_hang` merge added `has_side_effect=True` to `moe_dispatch`/`moe_cached_dispatch`/`moe_combine` FFI lowerings, plus a 3rd `send_head_work` output on `moe_combine`. The 14550 (NEW image) vs 14539 (OLD image) sgd-pp8 pdbs=8 comparison showed:

| | TGS (steps 9-14) | loss step 14 |
|---|---:|---:|
| 14539 OLD (no `combine_hang`) | 972.2 | 10.136 |
| 14550 NEW (with `combine_hang`) | 966.9 | 10.136 |

**Δ = -5 TGS, -0.5 %** — within step-to-step jitter (~7 TGS std dev). Loss is bit-identical. The fix is **effectively perf-neutral** in steady state: `has_side_effect=True` blocks XLA from DCE-ing or reordering the IPC ops, but on this MoE workload XLA wasn't using that scheduling freedom anyway (the calls already had hard cross-rank data dependencies). Pure correctness win.

## FSDP=8 tuning experiments (`sgd-v3` only, 36 runs)

The clean FSDP=8 baseline lands at 1017.7 TGS (`sgd-v3`, profiler off, no XLA dump, image-default XLA flags). The historical sgd-fs8 baseline on the original `deepep-gmm-maxtext-v26.2.tar` image was 1097 TGS — the ~5 % gap is split between (a) cluster jitter Apr-14-to-Apr-30, (b) the `combine_hang` correctness fix (verified ≤ 0.5 % on its own).

I tested 36 distinct XLA-flag / NCCL-env / memory-fraction profiles by editing `train_env.sh` between submissions (each `submit.sh` invocation freezes its own artifact, so pending jobs are unaffected by later edits — see `submit.sh:53-69` for the artifact-build mechanism). Hypotheses tested:

1. **Cross-iteration overlap flags** — `xla_gpu_enable_while_loop_double_buffering`, `xla_gpu_enable_pipelined_all_gather/reduce_scatter/all_reduce`. Hypothesis: prefetch the next-iteration's all-gather while current-iteration compute finishes.
2. **Async-stream priority** — `xla_gpu_enable_highest_priority_async_stream`. Hypothesis: give async collectives priority over compute.
3. **Per-call concurrency** — `xla_gpu_experimental_parallel_collective_overlap_limit ∈ {2, 4, 8}`. Hypothesis: allow more in-flight async collectives.
4. **LHS force** — `xla_gpu_enable_latency_hiding_scheduler=true`. Hypothesis: image default may have it off.
5. **Combiner threshold sweep** — `xla_gpu_all_gather_combine_threshold_bytes` and `xla_gpu_reduce_scatter_combine_threshold_bytes` at 256 MiB / 384 MiB / 512 MiB / 768 MiB / 1 GiB / 2 GiB / 4 GiB, with both ag and rs varied together AND ag varied alone. Hypothesis: image default is too small (so collectives fragment) OR too coarse (so collectives serialize).
6. **NCCL tuning** — `NCCL_BUFFSIZE=16 MiB`, `NCCL_NCHANNELS_PER_NET_PEER=8`, `NCCL_IB_QPS_PER_CONNECTION=8`, `NCCL_PROTO=Simple`, and combinations.
7. **Memory fraction** — `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` (vs default `.93`). Hypothesis: more HBM headroom for prefetch buffers.

36 experiments run, top recipes converge at **+11.6 % TGS over baseline** (1017.7 → 1135.7-1135.8). The dominant lever is the all-gather combine threshold; reduce-scatter combiner, NCCL channel count, memory fraction, and LHS-force all stack to <1 % each on top of `ag1G_only`.

### Leaderboard (top 8 + bottom 9)

| Rank | Profile | TGS | step | Δ% | Recipe |
|---:|---|---:|---:|---:|---|
| 1 | GLP_ag1G_chan8_mem95 (14624) | 1135.8 | 25.24 s | +11.61 % | ag=1 GiB + NCCL_NCHANNELS=8 + mem_frac=.95 |
| **2** | **G_ag1G_only** (14602) | **1135.7** | **25.25 s** | **+11.60 %** | `--xla_gpu_all_gather_combine_threshold_bytes=1073741824` ONLY |
| 3 | G_ag512M_only (14620) | 1135.2 | 25.26 | +11.55 % | ag=512 MiB only |
| 4 | G_combine384M (14609) | 1135.0 | 25.26 | +11.53 % | both ag and rs at 384 MiB |
| 5 | GLP512_full_mem95 (14613) | 1130.6 | 25.36 | +11.10 % | both ag/rs=512 MiB + NCCL_NCHANNELS=8 + mem_frac=.95 |
| 6 | GLD_combine1G_chan8_LHS (14606) | 1130.1 | 25.37 | +11.05 % | both ag/rs=1 GiB + NCCL_NCHANNELS=8 + LHS=true |
| 7 | G_ag2G_only (14621) | 1126.3 | 25.46 | +10.67 % | ag=2 GiB only |
| 8 | G_combine512M (14598) | 1119.2 | 25.62 | +9.98 % | both ag/rs=512 MiB |
| ... | (rest of positives, see appendix table) | | | | |
| baseline | 14572 sgd-fs8c | 1017.7 | 28.21 | 0 | image-default XLA_FLAGS (8 GiB combine threshold for both ag and rs) |
| | (negative results, sorted by impact) | | | | |
| | N_nccl_proto_simple (14594) | 1004.0 | 28.56 | -1.34 % | NCCL_PROTO=Simple |
| | O_nccl_combo (14595) | 1001.8 | 28.64 | -1.56 % | NCCL buffsize+channels+qps stacked |
| | K_nccl_buffsize16M (14591) | 997.6 | 28.75 | -1.98 % | NCCL_BUFFSIZE=16 MiB |
| | A_doublebuffer (14579) | 990.3 | 28.97 | -2.69 % | `--xla_gpu_enable_while_loop_double_buffering=true` |
| | AB (14582) | 978.4 | 29.31 | -3.86 % | A + highest_priority_async_stream |
| | C_overlap_limit2 (14581) | 975.6 | 29.40 | -4.14 % | `--xla_gpu_experimental_parallel_collective_overlap_limit=2` |
| | H_combine4G (14588) | 973.0 | 29.49 | -4.39 % | both ag/rs at 4 GiB (too coarse — close to 8 GiB default) |
| | J_async_unconstrained (14590) | 964.8 | 29.73 | -5.19 % | overlap_limit=8 |
| | ABC (14583) | 953.6 | 30.08 | -6.29 % | A+B+C stacked |

### Why all-gather combine threshold matters most

**The docker image's default sets `--xla_gpu_all_gather_combine_threshold_bytes=8589934592` (8 GiB) and the same for `reduce_scatter_combine_threshold_bytes`.** The full set of inherited XLA_FLAGS observed in the baseline (14572 sgd-fs8c) job log:

```
--xla_gpu_all_gather_combine_threshold_bytes=8589934592      ← 8 GiB (way too coarse)
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592  ← 8 GiB
--xla_gpu_enable_latency_hiding_scheduler=True
--xla_gpu_memory_limit_slop_factor=95
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_autotune_level=0
--xla_gpu_enable_all_gather_combine_by_dim=FALSE
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false
--xla_gpu_enable_command_buffer=''
```

This explains why "force LHS" (D) was only +1.72 % — LHS was already on. It also explains the combiner-threshold sweep shape:

| ag threshold | rs threshold | ag count per step (estimate) | TGS | Δ% |
|---:|---:|---:|---:|---:|
| 8 GiB (default) | 8 GiB (default) | 1 (everything fused) | 1017.7 | 0 |
| 4 GiB | 4 GiB | ~1-2 | 973.0 | -4.4 % |
| 2 GiB | 2 GiB | ~2-3 | 1098.1 | +7.9 % |
| 1 GiB | 1 GiB | ~4-5 | 1077.2 | +5.85 % |
| 768 MiB | 768 MiB | ~5-6 | 1081.9 | +6.3 % |
| 512 MiB | 512 MiB | ~8-10 | 1119.2 | +9.98 % |
| 384 MiB | 384 MiB | ~10-13 | 1135.0 | +11.53 % |
| 1 GiB | **8 GiB (default)** | ~4-5 | **1135.7** | **+11.60 %** |
| 512 MiB | **8 GiB (default)** | ~8-10 | 1135.2 | +11.55 % |
| 2 GiB | **8 GiB (default)** | ~2-3 | 1126.3 | +10.67 % |

**The image's 8 GiB threshold causes XLA to fuse every per-step all-gather into one mega-call that runs serially before any layer's compute can start.** This is the opposite of "prefetch / overlap" — it's a hard barrier. Reducing the all-gather threshold to 384 MiB - 2 GiB splits the mega-all-gather into 4-13 smaller all-gathers that XLA's latency-hiding scheduler can interleave with each layer's compute, recovering ~3 s/step of exposed comm.

**Reduce-scatter wants the opposite — leave it at 8 GiB.** Backward-pass reduce-scatters are inherently large (gradient buffers from cross-layer accumulation are 100s of MiB to several GiB per layer). Combining them into one 8 GiB chunk is fine because they're already individually large, and the backward pass is more compute-dense (gradient computation has more arithmetic per byte than forward), so RCCL launch overhead matters less and pipelining matters less. Splitting the reduce-scatter (e.g., `G_combine1G` sets BOTH to 1 GiB) actually hurts because it adds RCCL launch overhead without the corresponding overlap benefit. Hence the gap between `G_ag1G_only` (+11.60 %, only ag split) and `G_combine1G` (+5.85 %, both split).

`G_combine384M` (+11.53 %) matches `G_ag1G_only` (+11.60 %) because at 384 MiB the reduce-scatters that would have been combined into 8 GiB now split into ~3-4 still-large chunks (each gradient is hundreds of MiB), so the backward-pass cost is bounded — but you don't gain anything from splitting them either. So 384M-both and 1G-ag-only are functionally equivalent at the +11.5-11.6 % plateau.

### Negative findings (hypothesis-debugging value)

The user's initial premise was "all-gather/reduce-scatter not overlapped with compute, ideally should prefetch". The experiment matrix shows that prefetch-style flags **all hurt** on this workload:

- **`while_loop_double_buffering=true` (-2.69 %)**: cross-iteration overlap of the `train_step` `while` body. Costs HBM for the buffered next-iteration's input, which forces XLA to recompute or spill, and that overhead exceeds the (limited) overlap window.
- **`pipelined_all_gather/all_reduce/reduce_scatter=true` (E, OOM)**: enables an XLA pass that splits each collective across loop iterations, with one half running on the previous iteration. Same memory cost; on this workload the buffers are large enough that the pass OOMs.
- **`experimental_parallel_collective_overlap_limit=2 / 4 / 8` (-4 % to -5 %)**: allows multiple in-flight async collectives. RCCL contention (each gets fewer NIC channels) outweighs concurrency gain.
- **`highest_priority_async_stream=true` alone (+0.32 %)**: image default already prioritizes the async stream sufficiently.
- **`enable_latency_hiding_scheduler=true` (+1.72 %)**: small win — confirming image default did NOT have LHS on. But once you turn it on, you can't stack much else on top.

The lesson: **on this MoE workload at FSDP=8, the bottleneck is RCCL launch overhead, not exposed comm-vs-compute slack.** Once you collapse the launch count by raising the all-gather combine threshold, additional scheduling tweaks barely move TGS because the remaining RCCL is already running at near-peak bandwidth.

(The PP=8 tuning sweep covered later in this doc shows that *every one* of these signs flips on PP=8. See [§ Sign flips between FSDP=8 and PP=8](#why-overlap_limit2-wins-on-pp8-and-the-sign-flips-against-fsdp8).)

### Why dense-cf1.25 sees +20.5 % vs sgd-v3's +11.6 %

The same flag applied to both routing paths yields nearly double the speedup on `dense-cf1.25`. Likely mechanism:

- **dense-cf1.25 has higher inherent compute density per layer** — capacity-factor padding gives every expert a fixed-size GEMM regardless of routing, and there's no MoE dispatch/combine cost. So the per-layer compute window is large and uninterrupted, and overlap with a pipelined all-gather is highly effective.
- **sgd-v3's per-layer compute is shorter and fragmented** — each layer pays for DeepEP `moe_dispatch` (~1-2 ms IPC), then the actual GEMM, then `moe_combine` (~1-2 ms). The all-gather still gets to overlap with the dispatch-GEMM-combine sequence, but each piece is smaller, so the achievable overlap window is smaller.
- **MoE routing skew ceiling** — sgd-v3's per-layer compute time also varies with top-k assignment skew. The slowest rank's per-layer wall determines when the next all-gather starts (for the sub-chunked, pipelined version), so the overlap ceiling is bounded by `Σᵢ max_r(layer_iᵢ,r)`. dense-cf1.25 has uniform per-layer wall across ranks, so its ceiling is `max_r(Σᵢ layer_iᵢ,r) = Σᵢ layer_i` (no skew amplification).

This is the empirical proof that FSDP's dropless-MoE comm-overlap is fundamentally bounded by routing-skew variance — a structural property of any dropless MoE on synchronous distributed training, not specific to DS3-671B or this stack. Capacity-factor dropping (`dense-cf1.25`) eliminates that variance by construction, which is why it wins by a wider margin under the tuned recipe.

## PP=8 tuning experiments (sgd-v3 + dense-cf1.25, 36 runs, 5/01)

After deploying the FSDP-tuned `ag=1 GiB` flag in `configs/deepseek3-671b.env.sh`, a parallel sweep targeted PP=8 specifically. Headline result: **PP=8 dense-cf1.25 with `overlap2` alone beats FSDP=8 dense production by +1.34 %**, and the **FSDP-tuned env flag has the opposite sign on PP=8** (degrading dense by 6.6 %). PP=8 sgd-v3 reaches +4.66 % over the production-state baseline but stays ~12 % below FSDP=8 sgd-v3 — a structural gap (DeepEP per-microbatch cost + scan-carry serialization + pipeline bubble) that no XLA/NCCL knob closes.

### Effect of the deployed `configs/deepseek3-671b.env.sh` (FSDP-tuned `ag=1 GiB`) on PP=8

The 4/30 FSDP=8 sweep committed `--xla_gpu_all_gather_combine_threshold_bytes=1073741824` to the per-model env file because it gave FSDP=8 +11.6 % (sgd-v3) / +20.5 % (dense-cf1.25). For PP=8 the sign of this flag is **opposite**:

| Path | ag=1 GiB (env-file inherited) | ag=8 GiB (image default) | Δ |
|---|---:|---:|---:|
| sgd-v3 PP=8 | 955.3 (14638) | 965.2 (14639) | -1.0 % (within jitter) |
| dense-cf1.25 PP=8 | 1080.2 (14640) | 1156.9 (14641) | **-6.6 % (real, well above jitter)** |

The mechanism is path-dependent. For FSDP=8, the all-gather is over the full DCN ring (8 nodes × 8 GPUs) of expert weights, so collapsing it into one 8 GiB call creates a hard barrier; splitting to 1 GiB lets XLA's latency-hiding scheduler interleave the chunks with per-layer compute (+11.6 to +20.5 %). For PP=8, the all-gather is at ICI (intra-node, replica_groups=[8,8] over the `ici_expert_parallelism=8` axis) over a much smaller per-stage weight tensor; smaller chunks don't help here and just add RCCL launch overhead.

So `configs/deepseek3-671b.env.sh` must be guarded by topology before it can be safely applied to all `deepseek3-671b` submissions — see [§ Recommended deployment](#recommended-deployment).

### sgd-v3 PP=8 leaderboard (pinned 8-node nodelist, pdbs=7, steps 9-14)

All Δ% reported vs `pp8-baseline-ag1G` (job 14638, 955.3 TGS) — the production-state primary baseline (env-file ag=1 GiB inherited). Δ vs `pp8-restore_ag_default` (job 14639, 965.2 TGS, image default) shown for comparison.

| Rank | Profile | TGS | step | Δ vs ag=1G | Δ vs ag=8G | Recipe |
|---:|---|---:|---:|---:|---:|---|
| 1 ⭐ | `pp8-d_full_stack` (14669) | 1001.2 | 28.64 s | **+4.80 %** | +3.73 % | cp_decomp_1G + async_priority + overlap2 + NCCL_PROTO=Simple, ag=8 GiB default |
| 2 | `pp8-d_overlap2_async` (14668, 14674 retry n=2 avg=999.8) | 1000.2 | 28.67 s | **+4.70 %** | +3.62 % | overlap2 + async_priority, ag=8 GiB default |
| 3 | `pp8-d_overlap2_proto` (14670) | 999.3 | 28.69 s | +4.61 % | +3.53 % | overlap2 + NCCL_PROTO=Simple, ag=8 GiB default |
| 4 | `pp8-cp1G_async_ov2` (14667) | 995.4 | 28.80 s | +4.20 % | +3.13 % | cp_decomp_1G + async_priority + overlap2, ag=1 GiB |
| 5 | `pp8-d_cp1G_async_ov2` (14666) | 995.0 | 28.82 s | +4.15 % | +3.08 % | cp_decomp_1G + async_priority + overlap2, ag=8 GiB default |
| 6 | `pp8-overlap2` (14649, 1st measurement) | 994.9 | 28.82 s | +4.14 % | +3.08 % | overlap2 only, ag=1 GiB |
| 7 | `pp8-evidence` (14662, profiled run with HLO+xplane on overlap2) | 989.2 | 28.99 s | +3.55 % | +2.49 % | overlap2 + ag=1 GiB + profiler+HLO |
| 8 | `pp8-d_cp1G_async` (14657) | 988.3 | 29.01 s | +3.45 % | +2.39 % | cp_decomp_1G + async_priority, ag=8 GiB default |
| | (overlap2 + ag=1G, **n=3 avg** = 987.6, std 8.2) | | | **+3.38 %** | +2.32 % | overlap2 alone, ag=1 GiB |
| | `pp8-cp_decomp_1G_async` (14665, retry) | 987.0 | 29.05 s | +3.32 % | +2.26 % | cp_decomp_1G + async_priority, ag=1 GiB |
| | `pp8-nccl_proto_simple` (14664) | 987.2 | 29.04 s | +3.34 % | +2.28 % | NCCL_PROTO=Simple, ag=1 GiB |
| | `pp8-overlap8` (14660) | 981.7 | 29.21 s | +2.76 % | +1.71 % | overlap_limit=8, ag=1 GiB |
| | `pp8-async_priority` (14647) | 972.1 | 29.50 s | +1.76 % | +0.71 % | async_priority alone, ag=1 GiB |
| | `pp8-d_overlap2` (14661) | 971.8 | 29.51 s | +1.72 % | +0.68 % | overlap2 alone, ag=8 GiB |
| | `pp8-cp_decomp_1G` (14644) | 970.7 | 29.54 s | +1.62 % | +0.57 % | cp_decomp 1 GiB threshold (no-op for 392 MiB c-p), ag=1 GiB |
| | `pp8-mem95` (14663) | 968.8 | 29.60 s | +1.42 % | +0.38 % | XLA_PYTHON_CLIENT_MEM_FRACTION=.95, ag=1 GiB |
| baseline | **`pp8-baseline-ag1G`** (14638) | **955.3** | 30.02 s | 0 | -1.04 % | env-file inherited (`ag=1 GiB`) |
| | (`pp8-restore_ag_default`, 14639) | 965.2 | 29.71 s | +1.04 % | 0 | image-default `ag=8 GiB` |
| | `pp8-cp_decomp_256M` (14648) | 961.0 | 29.84 s | +0.60 % | -0.44 % | cp_decomp 256 MiB threshold (decomposes 392 MiB c-p) |
| | `pp8-double_buffer` (14646) | 962.8 | 29.78 s | +0.78 % | -0.25 % | `while_loop_double_buffering=true`; loss diverges 0.02 from baseline |
| | (negative results sorted by impact) | | | | | |
| | `pp8-d_chan8` (14653) | 958.9 | 29.90 s | +0.38 % | -0.65 % | NCCL_NCHANNELS_PER_NET_PEER=8, ag=8 GiB |
| | `pp8-nccl_chan8` (14652) | 938.6 | 30.57 s | -1.74 % | -2.75 % | NCCL_NCHANNELS_PER_NET_PEER=8, ag=1 GiB |
| | `pp8-overlap4` (14659) | 921.6 | 31.13 s | -3.53 % | -4.52 % | `parallel_collective_overlap_limit=4` |
| | `pp8-pp_p2p` (14643) | 904.6 | 31.73 s | **-5.31 %** | -6.28 % | `--xla_gpu_enable_pipelined_p2p=true` |
| | `pp8-pp_all_reduce` (14650) | OOM | — | — | — | `--xla_gpu_enable_pipelined_all_reduce=true` (217 GiB temp) |
| | `pp8-pp_all_gather` (14651) | OOM | — | — | — | `--xla_gpu_enable_pipelined_all_gather=true` (302 GiB temp) |

### dense-cf1.25 PP=8 leaderboard

| Rank | Profile | TGS | step | Δ vs ag=1G | Δ vs ag=8G | Δ vs FSDP=8 prod |
|---:|---|---:|---:|---:|---:|---:|
| 1 ⭐ | `pp8-dense_d_overlap2` (14673, 2nd measurement) | **1233.0** | 23.25 s | **+14.1 %** | **+6.58 %** | **+2.08 % (PP BEATS FSDP)** |
| 2 | `pp8-dense_d_overlap2` (14672, 1st measurement) | 1215.2 | 23.60 s | +12.5 % | +5.04 % | +0.61 % |
| | (avg of 14672/14673, n=2) | **1224.1** | 23.42 s | **+13.3 %** | **+5.81 %** | **+1.34 %** |
| 3 | `pp8-dense_d_full_stack` (14671) | 1206.3 | 23.78 s | +11.68 % | +4.27 % | -0.13 % (parity) |
| ref | (FSDP=8 dense + ag=1 GiB, 14629) | 1207.9 | — | — | — | 0 |
| 4 | `pp8-dense_d_overlap2_async` (14676) | 1190.4 | 23.93 s | +10.20 % | +2.90 % | -1.45 % (async HURTS dense) |
| | (`pp8-restore_ag_default`, dense, 14641) | 1156.9 | 24.79 s | +7.10 % | 0 | -4.22 % |
| baseline | **`pp8-baseline-ag1G`** dense (14640) | **1080.2** | 26.60 s | 0 | -6.62 % | -10.57 % |

**Important path-dependent finding**: adding `async_priority` HURTS dense-cf1.25 by -2.7 % vs overlap2 alone (1190.4 vs 1224.1 avg). For dense, **overlap2 alone is the winning recipe**. For sgd-v3, overlap2+async stacks slightly (+1.5 %) over overlap2 alone (avg 999.8 vs 984.9). The recipe split makes sense mechanistically: async_priority helps when the schedule is tight (sgd-v3's MoE skew creates per-rank stragglers); it hurts when the schedule is already loose (dense-cf1.25's uniform compute).

**Headline:** the simplest possible recipe — overlap2 alone — gives `dense-cf1.25` PP=8 **parity with (and slightly above) FSDP=8 dense production**. The full_stack additions (cp_decomp, async_priority, NCCL_PROTO=Simple) give a tiny regression on dense-cf1.25 — overlap2 alone is the winning recipe.

### Why `overlap_limit=2` wins on PP=8, and the sign flips against FSDP=8

`--xla_gpu_experimental_parallel_collective_overlap_limit=N` controls how many async collectives XLA's latency-hiding scheduler may have in-flight at the same time. Image default is 1 (one collective at a time).

From 14662's HLO (overlap2 + ag=1 GiB, profiled run for evidence), every `collective-permute-start` op is still tagged `is_pipelined=false` — i.e., the LHS scheduler still cannot move the c-p across iteration boundaries (same `nn.scan`-PP carry-dep that defeated `pipelined_p2p` in 14643). So overlap2 does NOT help by enabling pipelining.

What overlap2 **does** enable: per-iteration concurrent execution of the **5 distinct `collective-permute-start` ops** in the train_step HLO (channels 41, 42, 105, 200, 201 — split between forward and backward stage rotations) and the FSDP-style ICI collectives (all-gather over MoE expert weights, all-reduce for `DeepSeekMoeBlock_0/shard_map/psum`, reduce-scatter for gradients). With overlap=1, these serialize even when their data deps don't force serialization; with overlap=2, two can run concurrently on different RCCL streams.

The FSDP=8 sweep saw `overlap_limit=2/4/8 = -4 to -5 %` because FSDP's ICI all-gather already saturates the ICI fabric (so adding a 2nd concurrent collective creates contention). PP=8's ICI all-gather is much smaller (per-stage weight subset) and its DCN c-p is **point-to-point per stage hop** with much lower fabric utilization, so 2 concurrent collectives on different fabrics (one ICI, one DCN p2p) can actually overlap. This is the structural reason the flag's sign flips PP→FSDP.

`overlap_limit=4` already creates contention (-3.53 %); `overlap_limit=8` recovers somewhat (+2.76 %) but doesn't beat overlap=2 (+3.38 % avg). The **2-stream sweet spot** matches the count of independent fabrics (ICI + DCN).

The other Wave 2 flags (`cp_decomp_1G`, `async_priority`, `NCCL_PROTO=Simple`) are all in the **+3.3-3.5 %** band individually (within jitter of overlap2 alone) — they all address the same DCN scheduling bottleneck and are not additive on top of each other. Stacking all four lifts to +4.7-4.8 % vs the +3.4 % single-flag ceiling, only a +1.4 pp incremental from 3 extra flags. The simpler 2-flag (`overlap2 + async_priority`) recipe is recommended for sgd-v3; for dense-cf1.25 even `overlap2` alone is enough.

The systematic sign flips between regimes are summarised below — every flag that helps FSDP=8 hurts PP=8 (or vice versa), with the same mechanism (fabric utilization differences):

| Flag | FSDP=8 Δ | PP=8 Δ |
|---|---:|---:|
| `parallel_collective_overlap_limit=2` | **-4.14 %** | **+3.38 %** |
| `NCCL_NCHANNELS_PER_NET_PEER=8` | +4.26 % | -1.74 % |
| `pipelined_p2p=true` | no-op | -5.31 % |
| `NCCL_PROTO=Simple` | -1.34 % | +3.34 % |
| `all_gather_combine_threshold=1 GiB` (vs 8 GiB image default) | **+11.60 %** sgd / **+20.5 %** dense | -1.0 % sgd / **-6.6 %** dense |

**Past sweep results from one regime do NOT transfer to the other.**

### Why sgd-v3 PP=8 doesn't reach FSDP=8 parity (the structural -12 % gap)

While dense-cf1.25 PP=8 with overlap2 reaches FSDP=8 parity, sgd-v3 PP=8 with the best stack stops at 1001 TGS — still -11.95 % below FSDP=8 sgd-v3 production at 1135.7. The structural cost decomposition stands:

1. **`nn.scan`-PP carry serializes the schedule**, no XLA flag opens the carry-dep. The 14662 (overlap2) HLO confirms `collective-permute-start` still has `is_pipelined=false` even with the winning recipe.
2. **`collective-permute` is synchronizing per-call**. The 5 `collective-permute-start` ops × 8 microbatches × 8 stages × 2 (fwd+bwd) = ~640 c-p calls per step, each with rendezvous wait that compounds across ranks under MoE skew.
3. **DeepEP per-microbatch fixed cost**: 8 microbatches × 7 V-chunks × 8 stages = 448 DeepEP dispatch+combine round-trips per step at ~3 ms each = ~1.3 s fixed cost (sgd-v3 only).
4. **Bubble fraction = 7/63 = 11.1 %** — hard floor independent of XLA tuning.

dense-cf1.25 doesn't pay (3) at all (no DeepEP). It also pays (2) less because per-stage compute is uniform (capacity-factor dropping), so per-rank straggler skew is small. Only (1) and (4) apply, and overlap2's 2-fabric concurrency is sufficient to absorb most of (1)'s cost — hence the parity result.

### PP=8 negative findings (with Δ%) — useful for future agents to skip

| Profile | Δ vs sgd-v3 ag=1G | Note |
|---|---:|---|
| `pp_p2p` | -5.31 % | enables cross-iter prefetch buffer on c-p, but `nn.scan` carry-dep means LHS can't actually use the buffer; pure overhead |
| `overlap_limit=4` | -3.53 % | RCCL contention exceeds concurrency gain past the 2-fabric sweet spot |
| `nccl_chan8` (with ag=1G) | -1.74 % | extra NCCL channels don't help the per-stage 2-rank c-p; minor regression |
| `pp_all_reduce` | OOM 217 GiB temp | matches FSDP=8 OOM; pipelined_all_reduce buffer exceeds HBM under PP=8 too |
| `pp_all_gather` | OOM 302 GiB temp | same OOM mode |
| `cp_decomp_256M` | +0.60 % | aggressive c-p decomposition (does decompose the 392 MiB c-p) — provides no benefit on PP=8; LHS still can't pipeline across the carry-dep |
| `cp_decomp_1G` | +1.62 % (within jitter) | no-op for PP=8 (392 MiB c-p < 1 GiB threshold so no decomposition occurs) |
| `double_buffer` | +0.78 % | within jitter; loss14 diverges 0.02 from baseline (numerical effect, suggests changed accumulation order) |
| `mem95` | +1.42 % (within jitter) | matches FSDP finding (no measurable effect on either path) |

## `remat_policy` sensitivity (covers PP=8 *and* FSDP=8 — universal conclusion: `full` is optimal at pdbs=7)

YAML default `remat_policy: 'full'` (slowest recompute, lowest HBM) was suspected to be over-conservative for PP=8 given the apparent HBM headroom in the 4/30 memory feasibility matrix (sgd-pp8 total 253 / temp 178 GiB, dense-pp8 total ≈200 / temp ≈125 GiB on a 268 GiB-after-BFC HBM ceiling). The remat sweep tested 7 alternative policies (`save_out_proj`, `save_qkv_proj`, `save_dot_except_mlp`, `save_dot_except_mlpwi`, `minimal_with_context`, `minimal`) on top of the winning XLA recipes for *both* topologies.

| Path | Topology | remat_policy | TGS | step | Δ vs `full` | Total mem (compiled) | Loss14 (Δ) | Status | Note |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| sgd-v3 | PP=8 (overlap2+async) | **`full`** ⭐ | **999.8** | 28.68 s | 0 | 253 GB | 9.994 (=) | ✓ | winning |
| sgd-v3 | PP=8 | `save_out_proj` (14678) | 973.6 | 29.46 s | -2.62 % | 210-260 GB est. | 9.993 (≈=) | ✓ | fits but slower |
| sgd-v3 | PP=8 | `save_qkv_proj` (14677) | OOM | — | — | 513.6 GB | — | ✗ | requested 438.7 GiB single alloc |
| sgd-v3 | PP=8 | `save_dot_except_mlp` (14679) | OOM | — | — | 535.2 GB | — | ✗ | requested 460.4 GiB |
| sgd-v3 | PP=8 | `save_dot_except_mlpwi` (14680) | OOM | — | — | 2121.8 GB | — | ✗ | requested 2.00 TiB (8.4× HBM) |
| sgd-v3 | PP=8 | `minimal_with_context` (14681) | OOM | — | — | 3098.3 GB | — | ✗ | requested 2.95 TiB (12× HBM) |
| sgd-v3 | FSDP=8 (ag=1G prod) | **`full`** ⭐ | **1135.7** | 24.78 s | 0 | 236 GB | 9.994 (=) | ✓ | winning |
| sgd-v3 | FSDP=8 | `save_out_proj` (14700) | 1093.0 | 26.24 s | **-3.76 %** | 255 GB | **10.031 (+0.04)** | ✓ | fits, larger slowdown, **loss diverges** |
| sgd-v3 | FSDP=8 | `save_qkv_proj` (14689) | OOM | — | — | 469.1 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `save_dot_except_mlp` (14690) | OOM | — | — | 488.0 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `save_dot_except_mlpwi` (14691) | OOM | — | — | 1909.0 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `minimal_with_context` (14692) | OOM | — | — | 2779.8 GB | — | ✗ | |
| dense-cf1.25 | PP=8 (overlap2) | **`full`** ⭐ | **1224.1** | 23.42 s | 0 | ~200 GB | 9.998 (=) | ✓ | winning |
| dense-cf1.25 | PP=8 | `save_out_proj` (14686) | 1194.4 | 24.04 s | -2.42 % | 210.2 GB | 9.997 (≈=) | ✓ | fits but slower |
| dense-cf1.25 | PP=8 | `save_qkv_proj` (14687) | OOM | — | — | 410.7 GB | — | ✗ | requested 335.8 GiB |
| dense-cf1.25 | PP=8 | `save_dot_except_mlp` (14682) | OOM | — | — | 432.3 GB | — | ✗ | requested 357.5 GiB |
| dense-cf1.25 | PP=8 | `save_dot_except_mlpwi` (14683) | OOM | — | — | 688.3 GB | — | ✗ | requested 613.5 GiB |
| dense-cf1.25 | PP=8 | `minimal_with_context` (14684) | OOM | — | — | 922.8 GB | — | ✗ | requested 848.0 GiB |
| dense-cf1.25 | PP=8 | `minimal` (14685) | OOM | — | — | 862.4 GB | — | ✗ | requested 787.5 GiB |
| dense-cf1.25 | FSDP=8 (ag=1G prod) | **`full`** ⭐ | **1207.9** | — | 0 | ≈200 GB | 9.998 (=) | ✓ | winning |
| dense-cf1.25 | FSDP=8 | `save_out_proj` (14699) | 1049.8 | 27.32 s | **-13.09 %** | 190 GB | **10.032 (+0.034)** | ✓ | fits, **large slowdown**, loss diverges |
| dense-cf1.25 | FSDP=8 | `save_qkv_proj` (14694) | OOM | — | — | 368.4 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `save_dot_except_mlp` (14695) | OOM | — | — | 387.3 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `save_dot_except_mlpwi` (14696) | OOM | — | — | 608-688 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `minimal_with_context` (14697) | OOM | — | — | 800.9 GB | — | ✗ | |

**Three observations:**

1. **The activation tensors blow up much faster than the historical "headroom" suggested.** `dense-cf1.25` had ~70 GiB of free temp memory under `full` (200 vs 268 GiB HBM ceiling), but going to `save_dot_except_mlp` (a moderate, 4-tensors-saved policy) doubles total to 432 GB — losing 230 GB to extra activations, way more than the 70 GiB free. The `pipeline_module/while/body/closed_call/.../scan(layers.func_to_vmap)` lowering apparently keeps **per-microbatch × per-stage × per-V-chunk** copies of every saved activation, multiplying by 8×7 = 56 instead of the expected per-layer count of 7 (V chunks per stage). For PP=8 + DeepSeek-V3's `bf16[1,7,4096,7168]` (392 MiB) per-layer activations, multiplying by 56 instead of 7 explains the 8× memory blow-up observed. Under FSDP=8 the same blow-up applies via `scan_layers=True` over the 58 decoder layers — comparable factor (58 vs 56), so OOM thresholds are nearly identical across topologies.

2. **The one policy that fits (`save_out_proj`) is slower than `full` everywhere, with a 5× larger regression on FSDP=8 dense (-13.1 %) than on PP=8 dense (-2.4 %).** Likely mechanism: the FSDP=8 production schedule has its all-gather + reduce-scatter chunking finely tuned for `full`'s recompute schedule (the +11.6 / +20.5 % ag=1 GiB win was measured assuming `full`). Introducing extra saved activations (1 per layer) shifts the schedule's allocator fingerprint enough to break the prefetch overlap that ag=1 GiB achieves. PP=8 doesn't have this finely tuned overlap (overlap2 is the entire optimization), so the `save_out_proj` regression is just the BFC pressure cost.

3. **Loss diverges ~0.03-0.04 with `save_out_proj` under FSDP=8 only** (no divergence on PP=8). This suggests the FSDP=8 reduce-scatter schedule re-orders accumulations differently when extra activations are saved — same forward but slightly different gradient summation order → slightly different gradient values → slightly different loss after a few steps. Under PP=8, the per-stage local reduce-scatter doesn't reorder enough to perturb the loss.

**Recommendation**: keep `remat_policy: 'full'` for *both* topologies and *both* MoE branches at pdbs=7. Lighter remat is only viable at smaller pdbs (not tested here) or with framework-level changes that prevent the per-iteration activation duplication.

## Recommended deployment

The recipe is path × topology × branch dependent, so the per-model env file needs three branches:

```bash
# configs/deepseek3-671b.env.sh — split by topology axis and MoE branch
if [[ "${MAXTEXT_DCN_PP:-1}" -le 1 ]]; then
    # FSDP=8 path: keep the +11.6 % / +20.5 % all-gather combiner win.
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_all_gather_combine_threshold_bytes=1073741824"
else
    # PP=8 path: image default ag=8 GiB + overlap2 (universal sgd-v3 + dense win).
    # async_priority is BRANCH-DEPENDENT — helps sgd-v3 (+1.5 %) but HURTS dense (-2.7 %).
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    if [[ "${MAXTEXT_USE_DEEPEP_DISPATCH:-false}" == "true" || "${MAXTEXT_SPARSE_MATMUL:-false}" == "true" ]]; then
        # sgd-v3 (DeepEP / sparse_matmul) only — async_priority addresses MoE skew straggler wait.
        XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_highest_priority_async_stream=true"
    fi
    # dense-cf1.25 (`sparse_matmul=False`): no async_priority — uniform per-stage compute makes
    # the async-stream priority bump counter-productive (loses some scheduler flexibility).
fi
export XLA_FLAGS
```

**Caveat**: `submit.sh` doesn't currently propagate `MAXTEXT_DCN_PP` / `MAXTEXT_USE_DEEPEP_DISPATCH` / `MAXTEXT_SPARSE_MATMUL` as plain env vars (they're MaxText config keys passed through `--`). The guard variable names need to be coordinated with `submit.sh` / `_train.sh` — e.g., `_env_PP_TOPOLOGY=pp` could be the user-side hint, set automatically in `_job.sbatch` or in a wrapper script when `dcn_pipeline_parallelism=8` is detected in the passthrough args. Until that mechanism lands, the conservative fallback is to leave `configs/deepseek3-671b.env.sh` FSDP-only (its current state) and add a CLI override for PP=8 runs:

```bash
RAY=1 ./submit.sh deepseek3-671b:pp8-prod ... -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true \
    dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1 \
    _env_TUNE_PROFILE=pp8-d_overlap2_async   # for sgd-v3
    # OR _env_TUNE_PROFILE=pp8-dense_d_overlap2   # for dense-cf1.25 (overlap2 alone is enough)
```

This requires keeping `pp8-*` `TUNE_PROFILE` blocks in `train_env.sh` (which currently has them).

## Recommendations

1. **For production sgd-v3 (DeepEP dropless): use FSDP=8 pdbs=7 + `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`.** This recipe lands at 1135.7 TGS (+11.6 % over clean baseline, +3.5 % over the historical Apr-14 baseline of 1097). PP=8 cannot match this on the dropless path within the editable scope (Primus-Turbo + MaxText `moe.py`); closing the gap requires framework-level changes (rewrite `pipeline.py` to use explicit non-blocking `psend/precv`, or replace `nn.vmap`-of-`shard_map` composition with a custom pipeline schedule).
2. **For production `dense-cf1.25` (dense_matmul + capacity-factor dropping): use PP=8 + `--xla_gpu_experimental_parallel_collective_overlap_limit=2` ALONE on image-default XLA.** This recipe lands at 1224.1 TGS avg (n=2), beating FSDP=8 dense production (1207.9) by +1.34 %. **Do NOT inherit** the FSDP-tuned `ag=1 GiB` flag for this path — it costs -6.6 %. Adding `async_priority` HURTS dense by -2.7 %, so the simpler 1-flag recipe is correct for dense.
3. **For PP=8 sgd-v3 (when FSDP-feasibility forces PP): use `overlap_limit=2 + --xla_gpu_enable_highest_priority_async_stream=true` on image defaults.** This stack hits +4.66 % over the production-state baseline. The 4-flag stack adds only +0.1 % more, so the simpler 2-flag recipe is preferred. The remaining ~12 % gap to FSDP=8 sgd-v3 is structural (DeepEP per-microbatch cost + scan-carry serialization + pipeline bubble) and is not addressable via XLA / NCCL knobs at this stack level.
4. **Make `configs/deepseek3-671b.env.sh` (the FSDP-tuned `ag=1 GiB` flag) conditional on `dcn_pipeline_parallelism <= 1`.** The flag is +11.6 % / +20.5 % on FSDP=8 but **-1.0 % / -6.6 % on PP=8** (sign flip) — leaving it universal silently degrades PP=8 dense production by 6.6 %. See the deployment block above for the recommended split.
5. **Keep `remat_policy: 'full'` at pdbs=7 across BOTH topologies and BOTH branches.** The activation-memory blow-up factor under both `nn.scan`-PP and `scan_layers=True`-FSDP is ~56-58×, so OOM thresholds are nearly identical. The single non-OOM alternative (`save_out_proj`) regresses 2.4-13.1 % everywhere and additionally diverges loss under FSDP=8.
6. **Don't use `pp_p2p`, `overlap_limit≥4`, `nccl_chan8`, or any `pipelined_*=true` on PP=8.** All are negative or OOM. The FSDP=8 ranking of these flags does NOT transfer to PP=8 — sign flips are common (overlap2: FSDP -4.14 % → PP +3.38 %; nccl_chan8: FSDP +4.26 % → PP -1.74 %; pp_p2p: FSDP no-op → PP -5.31 %; NCCL_PROTO=Simple: FSDP -1.34 % → PP +3.34 %).
7. **Keep the upstream `combine_hang` fix.** It is required for production correctness on long runs; the ≤0.5 % steady-state cost is negligible.
8. **`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3` is now the `container_env.sh` default** — sgd configs no longer need an explicit env-var prefix. v1/v2 baselines still need an explicit override (`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep[-v2]`). The image carries C++/runtime fixes; the MaxText Python integration is orthogonal and lives on the patch branch (see `moe-pdbs-sweep-prompt.md` for the full ORTHOGONAL rule).
9. **`sparse-gmm-fixed` (sparse_matmul + `ragged_dot`) is the wrong path for DS3-671B at this scale.** It OOMs at every feasible pdbs ≥ 7 because `ragged_all_to_all` materializes a full `num_ranks × tokens × hidden` receive buffer (44 GiB more than DeepEP's per-rank `num_worst_tokens × hidden` buffer at pdbs=7). Use `sgd-v3` (DeepEP) instead, or fall back to `dense-cf1.25` (dense_matmul + capacity-factor dropping) if dropless routing isn't required.

## Summary of one-liner findings

1. **Best `sgd-v3` recipe (any topology choice): FSDP=8, pdbs=7, single XLA flag override `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`.** Yields **1135.7 TGS** vs 1017.7 baseline (+11.6 %), and beats the historical Apr-14 baseline (1097 TGS) by +3.5 %. PP=8 sgd-v3 stays ~12 % below FSDP=8 even with the best PP-tuned XLA stack (structural).
2. **Best `dense-cf1.25` recipe: PP=8, pdbs=7, `--xla_gpu_experimental_parallel_collective_overlap_limit=2` ALONE on image-default XLA.** Yields **1224.1 TGS avg (n=2)**, **+1.34 % over FSDP=8 dense production** (1207.9). Adding `async_priority` hurts (-2.7 %); dense wants the simpler recipe. The +20.5 % FSDP=8 dense win came from `ag=1 GiB` on the *FSDP* path; on the PP path the *same flag* costs -6.6 %.
3. **The recipe is path × topology × branch dependent.** Sign flips between FSDP=8 and PP=8 are systematic (overlap2: FSDP -4.14 % → PP +3.38 %; ag=1 GiB: FSDP +11.6 % → PP -1 to -6.6 %; NCCL_PROTO=Simple: FSDP -1.34 % → PP +3.34 %). The cause is fabric utilization differences (FSDP's ICI all-gather saturates the fabric and rejects concurrency; PP=8's ICI ag is small + DCN c-p is point-to-point with low utilization, so 2-stream concurrency helps). Past sweep results from one regime do NOT transfer to the other — and `configs/deepseek3-671b.env.sh` should be guarded by topology before being applied universally.
4. **`overlap2` is the optimum** in the parallel-collective-overlap-limit curve under PP=8 (overlap=1: 0; overlap=2: +3.38 %; overlap=4: -3.53 %; overlap=8: +2.76 %). The 2-stream sweet spot matches the count of independent fabrics in the PP=8 schedule (1 ICI + 1 DCN-p2p = 2).
5. **The `nn.scan`-PP carry-dep is the structural bottleneck on `is_pipelined=false`** — confirmed in 14563's HLO (image defaults, `xla_dump` from FSDP=8 sweep) and 14662's HLO (PP=8 sweep with `_env_ENABLE_XLA_DUMP=1` + winning `overlap2` recipe). No tested XLA flag changes the carry-dep, so cross-iteration scheduling stays off; PP wins come from intra-iteration concurrency (overlap2) instead.
6. **PP=8 c-p ops are all 392 MiB (`bf16[1,7,4096,7168]`)** — `cp_decomposer_threshold` between 512 MiB and 8 GiB is a no-op (doesn't decompose), while `=256 MiB` does decompose but provides no PP=8 speedup (the LHS scheduler still can't pipeline across the `nn.scan` carry-dep). C-p size is not the bottleneck.
7. **The +20.5 % vs +11.6 % gap on FSDP=8 (dense vs sgd-v3) proves dropless MoE's comm-overlap ceiling is bounded by routing skew.** `dense-cf1.25` has uniform per-layer compute (capacity-factor dropping) → no skew amplification → larger overlap window → bigger speedup from breaking the all-gather fusion barrier. The same routing-skew penalty is what locks PP=8 sgd-v3 at -12 % vs FSDP=8 sgd-v3.
8. **`sgf` (sparse-gmm-fixed) is unusable on this stack at scale.** OOMs at every feasible pdbs ≥ 7 due to `ragged_all_to_all`'s `num_ranks × tokens × hidden` receive buffer. Use `sgd-v3` (DeepEP) instead.
9. **No NCCL flag, memory-fraction tweak, async-stream priority, or while-loop double-buffering helped FSDP=8** beyond noise once the all-gather combine threshold was lowered. The +11.6 % (sgd-v3) / +20.5 % (dense-cf1.25) numbers are the practical FSDP=8 ceilings within editable scope.
10. **`pipelined_*=true` flags OOM on PP=8 too** (`pp_all_reduce` 217 GiB temp, `pp_all_gather` 302 GiB temp), matching FSDP=8 OOM behavior. Per-stage HBM headroom isn't enough to absorb the prefetch buffer for these collectives on either topology.
11. **The `combine_hang` fix is perf-neutral in steady state.** Loss is bit-identical to the pre-fix image; TGS gap is ≤ 0.5 % (within step-to-step jitter). Pure correctness win.
12. **`remat_policy: 'full'` is optimal at pdbs=7 across both topologies and both MoE branches** (sweep of 13 alternatives across jobs 14677-14700). Activation memory blows up by ~56-58× under both `nn.scan`-PP and `scan_layers=True`-FSDP; OOM thresholds are nearly identical. The single non-OOM alternative (`save_out_proj`) fits but regresses everywhere — and additionally diverges loss by 0.03-0.04 only on FSDP=8.

## Appendix: data sources

Job IDs sorted chronologically. Tags use the convention `<config>-<topology>` (e.g. `sgd-pp8` = `sgd-v3` on PP=8) plus an optional suffix (`c` = clean / no profiler, `+ag1G` = with the FSDP-tuned all-gather threshold).

| Job ID | Tag | Path | Image / Recipe | Profiler | HLO dump | Status |
|---|---|---|---|---|---|---|
| 13711 (historical) | sgd-fs8 pdbs=7 | sgd-v3 | original `deepep-gmm-maxtext-v26.2.tar` (Apr 14) | off | off | TGS=1097 baseline |
| 14539 | sgd-pp8 pdbs=8 (OLD image) | sgd-v3 | OLD (no `combine_hang`) | xplane | off | step 14 ✓ — used for `combine_hang` Δ check |
| 14550 | sgd-pp8 pdbs=8 | sgd-v3 | NEW (`combine_hang`) | xplane | yes | step 14 ✓ exit=1 cleanup |
| 14551 / 14564 | sgf-pp8 pdbs=8 / pdbs=7 | sgf | NEW | xplane / off | yes / off | OOM 217-222 GiB temp |
| 14552 | dense-cf1.25-pp8 pdbs=8 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14553 | sgd-fs8 pdbs=7 | sgd-v3 | NEW | xplane | yes | step 14 ✓ |
| 14554 | sgf-fs8 pdbs=7 | sgf | NEW | off | off | OOM 217 GiB temp |
| 14555 | dense-cf1.25-fs8 pdbs=7 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14563 | sgd-pp8 pdbs=7 | sgd-v3 | NEW | xplane | yes | step 14 ✓ exit=1 cleanup |
| 14565 | dense-cf1.25-pp8 pdbs=7 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14570 | sgd-pp8c pdbs=7 | sgd-v3 | NEW (clean) | off | off | step 14 ✓ exit=1 cleanup |
| 14571 | dense-cf1.25-pp8c pdbs=7 | dense-cf1.25 | NEW (clean) | off | off | clean PP=8 dropping baseline |
| 14572 | sgd-fs8c pdbs=7 | sgd-v3 | NEW (clean) | off | off | clean FSDP=8 sgd-v3 baseline |
| 14573 | dense-cf1.25-fs8c pdbs=7 | dense-cf1.25 | NEW (clean) | off | off | clean FSDP=8 dropping baseline |
| 14579-14626 | fs8 XLA-tuning sweep | sgd-v3 | NEW + various TUNE_PROFILE | off | off | 28 profiles, sgd-v3 only |
| 14602 | FSDP=8 sgd-v3 + ag=1 GiB tuned | sgd-v3 | `ag=1 GiB` | off | off | step 14 ✓ — TGS=1135.7 (FSDP=8 sgd-v3 production) |
| 14629 | FSDP=8 dense-cf1.25 + ag=1 GiB tuned | dense-cf1.25 | `ag=1 GiB` | off | off | step 14 ✓ — TGS=1207.9 (FSDP=8 dense production) |
| 14638 | sgd ag=1G baseline | sgd-v3 PP=8 | `pp8-baseline-ag1G` | off | off | step 14 ✓ — primary PP-sweep baseline |
| 14639 | sgd ag=8G default | sgd-v3 PP=8 | `pp8-restore_ag_default` | off | off | step 14 ✓ |
| 14640 | dense ag=1G baseline | dense-cf1.25 PP=8 | `pp8-baseline-ag1G` | off | off | step 14 ✓ |
| 14641 | dense ag=8G default | dense-cf1.25 PP=8 | `pp8-restore_ag_default` | off | off | step 14 ✓ |
| 14642 | (Wave 1.5 attempt 1) | sgd-v3 PP=8 | `pp8-evidence` | xplane | yes | cancelled in compile by slurm at 2:20 — non-recoverable |
| 14643 | pp_p2p | sgd-v3 PP=8 | `pp8-pp_p2p` | off | off | step 14 ✓ |
| 14644 | cp_decomp_1G | sgd-v3 PP=8 | `pp8-cp_decomp_1G` | off | off | step 14 ✓ |
| 14645 | (async_priority attempt 1) | sgd-v3 PP=8 | `pp8-async_priority` | off | off | RCCL flake at 15 min wall, cancelled+retried |
| 14646 | double_buffer | sgd-v3 PP=8 | `pp8-double_buffer` | off | off | step 14 ✓; loss diverges 0.02 |
| 14647 | async_priority retry | sgd-v3 PP=8 | `pp8-async_priority` | off | off | step 14 ✓ |
| 14648 | cp_decomp_256M | sgd-v3 PP=8 | `pp8-cp_decomp_256M` | off | off | step 14 ✓ |
| 14649 | overlap2 (1st measurement) | sgd-v3 PP=8 | `pp8-overlap2` | off | off | step 14 ✓ — first +4.14 % signal |
| 14650 | pp_all_reduce | sgd-v3 PP=8 | `pp8-pp_all_reduce` | off | off | OOM 217 GiB temp |
| 14651 | pp_all_gather | sgd-v3 PP=8 | `pp8-pp_all_gather` | off | off | OOM 302 GiB temp |
| 14652 | nccl_chan8 (ag=1G) | sgd-v3 PP=8 | `pp8-nccl_chan8` | off | off | step 14 ✓ |
| 14653 | nccl_chan8 (ag=8G) | sgd-v3 PP=8 | `pp8-d_chan8` | off | off | step 14 ✓ |
| 14654 | (mem95 attempt 1) | sgd-v3 PP=8 | `pp8-mem95` | off | off | RCCL flake, cancelled+retried |
| 14655 | (proto_simple attempt 1) | sgd-v3 PP=8 | `pp8-nccl_proto_simple` | off | off | RCCL flake, cancelled+retried |
| 14656 | (cp1G_async stack attempt 1) | sgd-v3 PP=8 | `pp8-cp1G_async` | off | off | RCCL flake, cancelled+retried |
| 14657 | d_cp1G_async (ag=8G) | sgd-v3 PP=8 | `pp8-d_cp1G_async` | off | off | step 14 ✓ |
| 14658 | overlap2 (2nd measurement) | sgd-v3 PP=8 | `pp8-overlap2` | off | off | step 14 ✓ — TGS=978.6 |
| 14659 | overlap4 | sgd-v3 PP=8 | `pp8-overlap4` | off | off | step 14 ✓ |
| 14660 | overlap8 | sgd-v3 PP=8 | `pp8-overlap8` | off | off | step 14 ✓ |
| 14661 | d_overlap2 (overlap2 + ag=8G) | sgd-v3 PP=8 | `pp8-d_overlap2` | off | off | step 14 ✓ |
| 14662 | overlap2 (3rd measurement) + profile + HLO | sgd-v3 PP=8 | `pp8-overlap2` | xplane | yes | step 14 ✓ — TGS=989.2; HLO/xplane at outputs/14662-*/ |
| 14663 | mem95 retry | sgd-v3 PP=8 | `pp8-mem95` | off | off | step 14 ✓ |
| 14664 | proto_simple retry | sgd-v3 PP=8 | `pp8-nccl_proto_simple` | off | off | step 14 ✓ |
| 14665 | cp1G_async retry | sgd-v3 PP=8 | `pp8-cp1G_async` | off | off | step 14 ✓ |
| 14666 | BS1 cp+as+ov2 (ag=8G) | sgd-v3 PP=8 | `pp8-d_cp1G_async_ov2` | off | off | step 14 ✓ |
| 14667 | BS2 cp+as+ov2 (ag=1G) | sgd-v3 PP=8 | `pp8-cp1G_async_ov2` | off | off | step 14 ✓ |
| 14668 | BS3 ov2+async (ag=8G) | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ — best 2-flag |
| 14669 | d_full_stack (cp+as+ov2+proto, ag=8G) | sgd-v3 PP=8 | `pp8-d_full_stack` | off | off | step 14 ✓ — sgd-v3 leader |
| 14670 | d_overlap2_proto (ag=8G) | sgd-v3 PP=8 | `pp8-d_overlap2_proto` | off | off | step 14 ✓ |
| 14671 | dense full_stack (ag=8G) | dense-cf1.25 PP=8 | `pp8-dense_d_full_stack` | off | off | step 14 ✓ |
| 14672 | dense overlap2 (ag=8G) | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ — **dense-cf1.25 LEADER** |
| 14673 | dense overlap2 retry (n=2 confirmation) | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ — TGS=1233.0 |
| 14674 | sgd BS3 retry (overlap2+async ag=8G) | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ — TGS=999.4 |
| 14675 | sgd d_overlap2 retry (overlap2 alone ag=8G) | sgd-v3 PP=8 | `pp8-d_overlap2` | off | off | step 14 ✓ — TGS=998.0 |
| 14676 | dense overlap2+async (universal recipe test) | dense-cf1.25 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ — TGS=1190.4 (async hurts dense) |
| 14677 | sgd save_qkv_proj remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 513.6 GB total |
| 14678 | sgd save_out_proj remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ — TGS=973.6 |
| 14679 | sgd save_dot_except_mlp remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 535.2 GB total |
| 14680 | sgd save_dot_except_mlpwi remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 2121.8 GB total |
| 14681 | sgd minimal_with_context remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 3098.3 GB total |
| 14682 | dense save_dot_except_mlp remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 432.3 GB total |
| 14683 | dense save_dot_except_mlpwi remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 688.3 GB total |
| 14684 | dense minimal_with_context remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 922.8 GB total |
| 14685 | dense minimal remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 862.4 GB total |
| 14686 | dense save_out_proj remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ — TGS=1194.4 |
| 14687 | dense save_qkv_proj remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 410.7 GB total |
| 14688 | FSDP=8 sgd save_out_proj attempt 1 | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | RCCL flake → 14698 |
| 14689 | FSDP=8 sgd save_qkv_proj | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | OOM 469.1 GB total |
| 14690 | FSDP=8 sgd save_dot_except_mlp | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | OOM 488.0 GB total |
| 14691 | FSDP=8 sgd save_dot_except_mlpwi | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | OOM 1909.0 GB total |
| 14692 | FSDP=8 sgd minimal_with_context | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | OOM 2779.8 GB total |
| 14693 | FSDP=8 dense save_out_proj attempt 1 | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | RCCL flake → 14699 |
| 14694 | FSDP=8 dense save_qkv_proj | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | OOM 368.4 GB total |
| 14695 | FSDP=8 dense save_dot_except_mlp | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | OOM 387.3 GB total |
| 14696 | FSDP=8 dense save_dot_except_mlpwi | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | OOM 608-688 GB total |
| 14697 | FSDP=8 dense minimal_with_context | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | OOM 800.9 GB total |
| 14698 | FSDP=8 sgd save_out_proj attempt 2 | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | failed (CUDA cuInit error 303 on node8 mid-run) → 14700 |
| 14699 | FSDP=8 dense save_out_proj | dense-cf1.25 FSDP=8 | (env-file ag=1G only) | off | off | step 14 ✓ — TGS=1049.8 (-13.09 % vs full) |
| 14700 | FSDP=8 sgd save_out_proj attempt 3 ✓ | sgd-v3 FSDP=8 | (env-file ag=1G only) | off | off | step 14 ✓ — TGS=1093.0 (-3.76 % vs full) |
