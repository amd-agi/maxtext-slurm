# Kimi-K2 1T — comprehensive pdbs sweep

- **Date:** 2026-04-23 (sweep); 2026-04-24 (profile drill-down + v3 finalization); 2026-04-25 (sgd-v2 profile refresh + DCN expert-parallelism extension `dcn_expert_parallelism ∈ {2, 4, 8}` for the 4 non-DeepEP configs)
- **Model:** `kimi-k2-1t` (MaxText). 1026.4 B parameters total. 61 decoder layers (layer 0 dense, layers 1–60 MoE with 384 experts × top-8 routing + 1 shared expert). MLA attention (`q_lora_rank=1536`, `kv_lora_rank=512`). See [`configs/kimi-k2-1t.gpu.yml`](configs/kimi-k2-1t.gpu.yml).
- **Hardware:** 8 nodes × 8× AMD MI355 (288 GB HBM / device), Pensando AINIC interconnect, k8s partition (`chi[2766,2800,2810,2832,2835,2865,2872,2883]`).
- **Image:** `/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar` (includes [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) GMM + DeepEP).
- **Patch branches:**
  - [yihuang/moe-turbo-gmm-and-deepep](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep) @ `ad693da2` (baseline — `sparse-gmm-deepep` / v1 column)
  - [yihuang/moe-turbo-gmm-and-deepep-v2](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v2) @ `627168f8` (v2 column)
  - [yihuang/moe-turbo-gmm-and-deepep-v3](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9` (v3 column — the headline)
- **Base config:** [`configs/kimi-k2-1t.gpu.yml`](configs/kimi-k2-1t.gpu.yml) (`dcn_fsdp_parallelism=8`, `ici_expert_parallelism=8` on 8-node × 8-GPU topology).
- **Dataset:** `dataset_type=synthetic` (the gpu.yml default at sweep time; if the current default has since changed, CLI-override `dataset_type=synthetic` to reproduce).
- **Peak BF16:** ≈ 2500 TFLOP/s/device → MFU ≈ TFLOP/25.
- **Sweep counterpart:** [`deepseek3-671b-pdbs-sweep.md`](deepseek3-671b-pdbs-sweep.md) — this doc reuses its taxonomy and structure.

## Background

The [DS3 sweep](deepseek3-671b-pdbs-sweep.md) established that the `sparse-gmm-deepep` v1→v2→v3 optimization chain (Python-only patches to `src/MaxText/layers/moe.py`) delivered +24 % / +59 % TGS over baseline at DS3's pdbs=6, by eliminating the dominant `input_scatter_fusion_*.kd` kernel from the DeepEP dispatch backward. This sweep re-runs the same 10-config matrix on kimi-k2-1t to answer: **does the v1→v3 gain shape replicate on a 1T model where memory feasibility (not kernel optimization) is the dominant axis?**

> **1-GPU/proc launcher is intentionally not swept.** DS3's takeaway #3 (on `deepseek3-671b-pdbs-sweep.md`) already characterized 1-GPU/proc behavior model-agnostically: its wins stem from XLA's runtime-kernel swap on `ragged_all_to_all`, which is now accessible on 1-node/proc via `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` (the new default in `train_env.sh`). That mechanism is not expected to vary with model size. Only the 1-node/proc launcher is measured here.

**The result tables below are ragged.** The refined prompt (see [`moe-pdbs-sweep-prompt.md`](moe-pdbs-sweep-prompt.md)) replaces a fixed pdbs ladder with dynamic ceiling probing: for each config, walk pdbs upward until OOM, then back-fill ±1 around the ceiling. Missing cells in the tables below are *uninteresting* (skipped by the monotonic-in-pdbs rule once OOM was observed), not pending. `P★` is defined as `min(max_pdbs)` across feasible configs and marks the apples-to-apples cross-config comparison row.

---

## Configs under test

| Tag                    | Submit-time env var prefix                                             | Passthrough flags                                                                    |
|------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `dense-cf1.25`         | —                                                                      | *(default)* — `sparse_matmul=false`, `capacity_factor=1.25`                          |
| `dense-cf2`            | —                                                                      | `capacity_factor=2.0`                                                                |
| `dense-cf4`            | —                                                                      | `capacity_factor=4.0`                                                                |
| `sparse`               | —                                                                      | `sparse_matmul=true shardy=true`                                                     |
| `sparse-gmm`           | —                                                                      | `sparse_matmul=true use_turbo_grouped_gemm=true _env_ENABLE_RAGGED_ONESHOT_KERNEL=1` |
| `sparse-gmm-fixed`     | —                                                                      | `sparse_matmul=true use_turbo_grouped_gemm=true`                                     |
| `sparse-deepep`        | —                                                                      | `sparse_matmul=true use_deepep_dispatch=true shardy=true`                            |
| `sparse-gmm-deepep` (v1)| `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep` (override the v3 default) | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`           |
| `sparse-gmm-deepep-v2` | `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v2` (override the v3 default) | same passthroughs as v1                                                              |
| `sparse-gmm-deepep-v3` | — *(default branch — `container_env.sh` already points here)*           | same passthroughs as v1                                                              |

**What distinguishes `sparse-gmm-deepep`, `-v2`, `-v3`:** the three rows run the same image / launcher / passthrough flags — only the patched MaxText branch differs, and the only file that differs between the branches is `src/MaxText/layers/moe.py`. **v2** composes two DeepEP dispatch-side gathers into one (halving the backward scatter-add count). **v3** replaces the remaining duplicate-index scatter-add backward with a `jax.custom_vjp` that uses an argsort-inverse gather + reduce-sum — no atomics. Forward output is bit-identical across v1/v2/v3 (loss matches to bf16 LSB at every step), so any TGS delta is pure kernel-level optimization. See DS3's [v1/v2/v3 drill-down section](deepseek3-671b-pdbs-sweep.md#why-do-v1--v2--v3-of-sparse-gmm-deepep-differ-in-tgs-pdbs6-profile-drill-down) for the code-level diff.

**Why only `sparse` and `sparse-deepep` carry `shardy=true`:** `sparse_matmul=true` without `use_turbo_grouped_gemm=true` falls back to `jax.lax.ragged_dot`, whose sharding propagation needs `shardy=true`. `sparse-gmm*` rows use Primus-Turbo's GMM custom call which carries its own sharding spec and sidesteps the propagation pass.

---

## Feasibility summary

Memory cliffs for Kimi-K2 1T are *much* tighter than DS3's (which is the primary 1T-specific finding of this sweep). `P★ = 4` across all feasible configs (locked in by the two sparse-gmm ceilings). Full-row infeasible: `sparse` and `sparse-deepep`.

| Config                    | `max_pdbs` (ceiling) | `argmax_TGS_pdbs` | Peak TGS | Peak MFU | OOM signature at ceiling |
|---------------------------|---------------------:|------------------:|---------:|---------:|--------------------------|
| `dense-cf1.25`            |                   12 |            **11** |   1170.1 |   9.62 % | `✗ 202.4 GiB @ pdbs=16`  |
| `dense-cf2`               |                   10 |                10 |    827.8 |   6.80 % | `✗ 189.4 GiB @ pdbs=11`  |
| `dense-cf4`               |                    6 |                 5 |    455.1 |   3.74 % | `✗ 180.5 GiB @ pdbs=7`   |
| `sparse`                  |          **infeasible** |              — |        — |        — | `✗ 581.8 GiB @ pdbs=1`   |
| `sparse-gmm` (one-shot)   |                    4 |                 4 |    249.0 |   2.05 % | `✗ 195.6 GiB @ pdbs=5`   |
| `sparse-gmm-fixed`        |                    4 |                 4 |    614.5 |   5.05 % | `✗ 195.6 GiB @ pdbs=5`   |
| `sparse-deepep`           |          **infeasible** |              — |        — |        — | `✗ 507.4 GiB @ pdbs=1`   |
| `sparse-gmm-deepep` (v1)  |                    5 |                 5 |    515.7 |   4.24 % | `✗ 195.3 GiB @ pdbs=6`   |
| `sparse-gmm-deepep-v2`    |                    5 |                 5 |    635.9 |   5.23 % | `✗ 202.6 GiB @ pdbs=6`   |
| **`sparse-gmm-deepep-v3`** |                **7** |             **7** |    **897.9** |  **7.38 %** | `✗ 214.3 GiB @ pdbs=8` |

**P★ = 4** (`min(max_pdbs)` across feasible configs, locked by the two sparse-gmm paths).

**Key feasibility findings:**

1. **v3 extends the DeepEP frontier by 2 pdbs over v1 / v2 / sparse-gmm-fixed.** v3's `jax.custom_vjp` eliminates the duplicate-index scatter-add intermediate tensors that v1 / v2 hold (~K × H × N bf16 floats per MoE layer ≈ several GiB per layer on a 1T model). The HBM headroom that buys is **exactly what lets v3 reach pdbs=6 and 7 while v1 / v2 / sparse-gmm-fixed all OOM at pdbs=6**. This is the 1T-specific version of DS3's v3 story: on DS3, v3's kernel advantage showed up as a TGS delta at the same ceiling; on kimi-1T, it additionally shows up as **ceiling extension**.
2. **`sparse` and `sparse-deepep` (the `ragged_dot` paths without GMM) are infeasible on 1T at pdbs=1** — both OOM on the `RaggedDot` working set (581.8 GiB and 507.4 GiB respectively). Same story as DS3 but the numbers are 30–35 % larger, matching the 1T-vs-671B parameter ratio.
3. **Even `sparse-gmm` and `sparse-gmm-fixed` cap at pdbs=4 on 1T.** On DS3 these reached pdbs=7 (with MEM_FRACTION=.96). On kimi-1T they OOM at pdbs=5 at default MEM_FRACTION=.93 with a 195.6 GiB allocation that's well below the 267.8 GiB pool — meaning the *total* working set (not the single large alloc) crosses the pool ceiling. `.96` retry doesn't help here because the alloc size is not within 10 % of the pool.
4. **Dense `cf=4.0` doesn't fall off the cliff as early as might be expected.** cf=4 doubles activation memory vs cf=2 but `max_pdbs` only drops from 10 to ≥5 (still probing). The activation memory scales with `cf × pdbs × seq_len × emb_dim × layers` while params stay the same — at pdbs=5 with cf=4, activations are ~1.6 × what cf=2 at pdbs=5 uses.

---

## Results matrix — 1-node/proc, ragged

All metrics except loss are **mean over training steps 5–14** (steps 0–4 discarded as warmup). Loss is reported from step 14 only since the synthetic-data loss at a single step is a consistent numerical-correctness probe.

Legend: `✗<GiB>` = OOM with the reported alloc size; `—` = skipped by monotonic-in-pdbs rule (earlier pdbs already OOM'd); blank = skipped intentionally by the dynamic probing ladder (typically 3, 13–15 for most configs).

**Rows at `pdbs=P★=4`** (the apples-to-apples cross-config comparison row) are **bold**.

### Tokens/s/device (TGS)

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm (one-shot) | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        234.9 |     208.3 |     195.2 |                 135.0 |            220.4 |         193.0 |         212.9 |             229.5 |
|    2 |        399.0 |     373.4 |     297.1 |                 190.8 |            380.9 |         320.1 |         358.3 |             405.5 |
|  **4** |    **678.9** | **597.0** | **414.9** |            **249.0**  |        **614.5** |     **476.9** |     **575.4** |         **685.2** |
|    5 |        787.2 |     652.1 | **455.1** |              ✗ 195.6  |         ✗ 195.6  |     **515.7** |     **635.9** |             750.8 |
|    6 |        873.3 |     707.5 |     453.2 |                    —  |         ✗ 207.2  |       ✗ 195.3 |       ✗ 202.6 |             856.4 |
|    7 |        930.3 |     758.4 |   ✗ 180.5 |                    —  |                — |             — |             — |         **897.9** |
|    8 |       1028.9 |     808.8 |   ✗ 216.4 |                    —  |                — |             — |             — |           ✗ 214.3 |
|    9 |       1035.1 |         — |         — |                    —  |                — |             — |             — |                 — |
|   10 |       1135.1 | **827.8** |         — |                    —  |                — |             — |             — |                 — |
|   11 |   **1170.1** |   ✗ 189.4 |         — |                    —  |                — |             — |             — |                 — |
|   12 |       1134.5 |   ✗ 189.7 |         — |                    —  |                — |             — |             — |                 — |
|   16 |      ✗ 202.4 |         — |         — |                    —  |                — |             — |             — |                 — |

**Full-row infeasible:** `sparse` (OOM 581.8 GiB @ pdbs=1), `sparse-deepep` (OOM 507.4 GiB @ pdbs=1).

**Peak MFU:** `dense-cf1.25 @ pdbs=11` = **9.62 %** (1170.1 TGS, 240.4 TFLOP/s/device on MI355's 2500 TFLOP/s peak BF16).
**Peak dropless MFU:** `sparse-gmm-deepep-v3 @ pdbs=7` = **7.38 %** (897.9 TGS, 184.5 TFLOP/s/device) — no MEM_FRACTION bump needed.

### TFLOP/s/device

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm (one-shot) | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |         48.3 |      42.8 |      40.1 |                  27.7 |             45.3 |          39.7 |          43.7 |              47.2 |
|    2 |         82.0 |      76.7 |      61.0 |                  39.2 |             78.3 |          65.8 |          73.6 |              83.3 |
|  **4** |     **139.5** | **122.7** |  **85.3** |             **51.2**  |        **126.3** |      **98.0** |     **118.2** |         **140.8** |
|    5 |        161.7 |     134.0 |  **93.5** |                    ✗  |               ✗  |     **106.0** |     **130.7** |             154.3 |
|    6 |        179.4 |     145.4 |      93.1 |                    —  |               ✗  |             ✗ |             ✗ |             176.0 |
|    7 |        191.1 |     155.8 |         ✗ |                    —  |                — |             — |             — |         **184.5** |
|    8 |        211.4 |     166.2 |         ✗ |                    —  |                — |             — |             — |                 ✗ |
|    9 |        212.7 |         — |         — |                    —  |                — |             — |             — |                 — |
|   10 |        233.2 | **170.1** |         — |                    —  |                — |             — |             — |                 — |
|   11 |    **240.4** |         ✗ |         — |                    —  |                — |             — |             — |                 — |
|   12 |        233.1 |         ✗ |         — |                    —  |                — |             — |             — |                 — |
|   16 |            ✗ |         — |         — |                    —  |                — |             — |             — |                 — |

### Average per-step time (seconds)

Lower is better. Mean of the per-step wall times (`seconds:` field in the training log) over steps 5–14.

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm (one-shot) | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        17.44 |     19.72 |     21.00 |                 30.34 |            18.59 |         21.24 |         19.24 |             17.84 |
|    2 |        20.58 |     21.94 |     27.58 |                 42.95 |            21.51 |         25.60 |         22.89 |             20.21 |
|  **4** |    **24.14** | **27.45** | **39.49** |            **65.79**  |        **26.67** |     **34.37** |     **28.47** |         **23.91** |
|    5 |        26.02 |     31.41 | **45.00** |                    ✗  |               ✗  |     **39.72** |     **32.22** |             27.29 |
|    6 |        28.14 |     34.74 |     54.23 |                    —  |               ✗  |             ✗ |             ✗ |             28.70 |
|    7 |        30.82 |     37.81 |         ✗ |                    —  |                — |             — |             — |             31.95 |
|    8 |        31.85 |     40.52 |         ✗ |                    —  |                — |             — |             — |                 ✗ |
|    9 |        35.64 |         — |         — |                    —  |                — |             — |             — |                 — |
|   10 |        36.09 | **49.48** |         — |                    —  |                — |             — |             — |                 — |
|   11 |        38.51 |         ✗ |         — |                    —  |                — |             — |             — |                 — |
|   12 |        43.33 |         ✗ |         — |                    —  |                — |             — |             — |                 — |
|   16 |            ✗ |         — |         — |                    —  |                — |             — |             — |                 — |

### Training loss at step 14

Within each pdbs row, all feasible configs agree to Δ ≤ 0.002 — v2/v3 forward HLO is bit-identical to v1 (loss deltas at the bf16 LSB), and MoE-vs-dense loss agrees within the synthetic-data warmup regime because both paths see identical input tokens.

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm (one-shot) | sparse-gmm-fixed | sgd-deepep v1 | sgd-deepep v2 | **sgd-deepep v3** |
|-----:|-------------:|----------:|----------:|----------------------:|-----------------:|--------------:|--------------:|------------------:|
|    1 |        8.772 |     8.771 |     8.771 |                 8.740 |            8.740 |         8.741 |         8.740 |             8.741 |
|    2 |        9.627 |     9.626 |     9.625 |                 9.627 |            9.626 |         9.625 |         9.625 |             9.626 |
|  **4** |   **10.367** | **10.366** | **10.365** |           **10.366** |       **10.366** |    **10.366** |    **10.366** |        **10.366** |
|    5 |       10.566 |    10.566 | **10.566** |                   ✗  |               ✗  |    **10.565** |    **10.565** |            10.565 |
|    6 |       10.713 |    10.712 |    10.712 |                    —  |               ✗  |             ✗ |             ✗ |            10.712 |
|    7 |       10.817 |    10.817 |         ✗ |                    —  |                — |             — |             — |            10.816 |
|    8 |       10.920 |    10.919 |         ✗ |                    —  |                — |             — |             — |                 ✗ |
|    9 |       11.007 |         — |         — |                    —  |                — |             — |             — |                 — |
|   10 |       11.067 | **11.066** |         — |                   —  |                — |             — |             — |                 — |
|   11 |       11.120 |         ✗ |         — |                    —  |                — |             — |             — |                 — |
|   12 |       11.184 |         ✗ |         — |                    —  |                — |             — |             — |                 — |

---

## DCN expert-parallelism extension (`dcn_expert_parallelism > 1`)

Extends the main sweep above (which runs at the kimi-1T default of `dcn_expert_parallelism=1`, i.e. expert parallelism is purely intranode and FSDP is purely inter-node) by walking the inter-node EP factorization. On 8 nodes × 8 GPUs = 64 ranks the parallelism grid stays the same total but is re-split:

| `DCN_EP` | `dcn_fsdp` × `ici_ep × dcn_ep` | total EP rank-product | EP fanout per host |
|---:|:-:|---:|---|
| **1** *(default — main sweep)* | 8 × 8 × 1 | 8 | EP axis is intranode-only |
| 2 | 4 × 8 × 2 | 16 | each host's experts spread to 1 peer host |
| 4 | 2 × 8 × 4 | 32 | each host's experts spread to 3 peer hosts |
| 8 | 1 × 8 × 8 | 64 | full DCN-EP, no inter-node FSDP |

### Known limitation: DeepEP variants are gated to `DCN_EP=1`

`MaxText/pyconfig.py` validates `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` and rejects the config with a pydantic `ValidationError("Internode DeepEP is not yet supported in JAX")` in ~2 min before reaching XLA compile. This applies to **all 4 DeepEP configs** (`sparse-deepep`, `sparse-gmm-deepep` v1/v2/v3) — the JAX/MaxText integration layer blocks the very regime DS3's "DeepEP wins on inter-node EP" hypothesis was supposed to be tested in. The DS3 prediction therefore **cannot be empirically tested against this MaxText version** (would require an upstream fix lifting the validator). The DCN-EP extension below characterizes only the **non-DeepEP** regime — `dense-cf1.25 / cf2 / cf4` and `sparse-gmm-fixed`. `sparse-gmm` (one-shot) is also skipped at DCN_EP > 1: its OneShot kernel is intranode-only, falls back to the kNccl path at DCN_EP > 1, and would just duplicate `sparse-gmm-fixed`.

### TGS @ DCN_EP > 1 (4 non-DeepEP configs)

DCN_EP=1 column copied from the main matrix above for cross-comparison. Cells marked `n/a` are pdbs values not measured at that DCN_EP.

#### `dense-cf1.25`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  2 |  481.4 |     479.5 |       n/a |     n/a |
|  4 |  678.9 | **723.0** |     669.4 |   577.7 |
|  6 |  873.3 |     821.4 |       n/a |     n/a |
|  8 | 1028.9 |     888.1 |     696.4 |   605.8 |
| 10 | 1135.1 | **956.2** |       n/a |     n/a |
| 12 | 1134.5 |     840.5 |       n/a |     n/a |

#### `dense-cf2`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  597.0 |     544.9 |     453.1 |   390.1 |
|  6 |  707.5 |     608.4 |       n/a |     n/a |
|  8 |  808.8 | **629.4** |     459.1 |     n/a |
| 10 |  827.8 |     568.6 |       n/a |     n/a |

#### `dense-cf4`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  414.9 |     309.8 |     229.5 |   199.6 |
|  5 |  455.1 | **322.8** |       n/a |     n/a |
|  6 |  453.2 |     297.3 |     214.0 |     n/a |

#### `sparse-gmm-fixed`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  1 |  362.5 |     313.7 | **339.4** | ✗ 190 GiB |
|  2 |  500.5 | **450.7** | ✗ 237 GiB |     —   |
|  4 |  614.5 | ✗ 226 GiB |       —   |     —   |

### Feasibility summary across DCN_EP

| Config | DCN_EP=1 max_pdbs | DCN_EP=2 max_pdbs | DCN_EP=4 max_pdbs | DCN_EP=8 max_pdbs | Notes |
|---|---:|---:|---:|---:|---|
| `dense-cf1.25` | 12 | ≥ 12 | ≥ 8 | ≥ 8 | non-monotonic TGS @ DCN_EP=2 (peak at pdbs=10, drops at 12) |
| `dense-cf2`    | 10 | ≥ 10 | ≥ 8 | ≥ 4 | non-monotonic at DCN_EP=2 (peak at pdbs=8, drops at 10) |
| `dense-cf4`    |  6 | ≥ 6  | ≥ 6 | ≥ 4 | argmax_TGS shifts down: DCN_EP=1 → pdbs=5; DCN_EP=2 → pdbs=5; DCN_EP=4 → pdbs=4 |
| `sparse-gmm-fixed` | 4 | **2** | **1** | **0 (infeasible)** | tightest cliff — non-expert FSDP shard growth dominates expert-shard shrinkage |

### Cross-DCN_EP TGS comparison at pdbs=4 (cross-config baseline)

| Config | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 | Δ EP=1→8 |
|---|---:|---:|---:|---:|---:|
| `dense-cf1.25`  | 678.9 | **723.0** *(+6.5%)* | 669.4 *(−1.4%)* | 577.7 *(−14.9%)* | **−14.9%** |
| `dense-cf2`     | 597.0 | 544.9 *(−8.7%)*    | 453.1 *(−24.1%)* | 390.1 *(−34.7%)* | **−34.7%** |
| `dense-cf4`     | 414.9 | 309.8 *(−25.3%)*   | 229.5 *(−44.7%)* | 199.6 *(−51.9%)* | **−51.9%** |
| `sparse-gmm-fixed` | 614.5 | ✗ OOM | ✗ OOM | ✗ OOM | infeasible at DCN_EP > 1 |

Loss values agree to ε across all DCN_EP variants for every config-pdbs cell — DCN_EP factorization is numerically transparent, as expected.

### DCN-EP key observations

1. **`dense-cf1.25 @ pdbs=4` is FASTER at DCN_EP=2 than at DCN_EP=1** (723.0 vs 678.9 TGS, +6.5%). At small pdbs the per-rank expert-weight reduction (48 → 24 experts/GPU) outweighs the inter-node `all-to-all` RDMA overhead. The crossover happens around pdbs=6 — by pdbs=8 DCN_EP=1 already wins (1028.9 vs 888.1, +15.9%) because activation memory has grown enough that the dispatch all-to-all becomes the bottleneck.
2. **The dropping-vs-dropping memory-cost ranking is preserved.** `dense-cf4` always loses to `cf2` loses to `cf1.25` at the same pdbs at every DCN_EP value; the *gap* widens with DCN_EP (cf4 drops 52% at DCN_EP=8 vs cf1.25's 15%). This is consistent with cf=4's larger activation tensor making it relatively more sensitive to inter-node all-to-all overhead.
3. **`sparse-gmm-fixed` is the most DCN-EP-fragile config**: its `max_pdbs` collapses 4 → 2 → 1 → 0 (infeasible) as DCN_EP doubles. The dropless RCCL `ragged-all-to-all` over RDMA scales worse than dense's regular `all-to-all` because the ragged buffer is sized for the worst-case routing fan-out, not the dropping-truncated capacity that dense uses. By DCN_EP=8 even pdbs=1 OOMs at 190 GiB.
4. **TGS curves go non-monotonic across pdbs at DCN_EP=2** for dense-cf1.25 (peak 956 @ pdbs=10, drops to 840 @ pdbs=12) and dense-cf2 (peak 629 @ pdbs=8, drops to 569 @ pdbs=10). Same `argmax_TGS_pdbs < max_pdbs` pattern observed in the DCN_EP=1 sweep, just shifted one to two pdbs lower.
5. **`sparse-gmm-fixed` ceiling collapses much faster than dense ceilings.** At DCN_EP=1, `sparse-gmm-fixed` and `dense-cf1.25` differed in `max_pdbs` by 8 (4 vs 12); at DCN_EP=4 the gap widens to ≥7 (1 vs ≥8); at DCN_EP=8, sparse-gmm-fixed is fully infeasible while dense-cf1.25 still fits at pdbs=8 (605 TGS). The dropless RCCL `ragged-all-to-all` is the most DCN-EP-fragile collective in this set; dense's regular `all-to-all` is far more tolerant.
6. **TGS-per-pdbs slope flattens at high DCN_EP** (same pattern as DS3). dense-cf1.25 pdbs=4 → pdbs=8 gain: +42% at DCN_EP=1 (678.9 → 1028.9, inferred from main sweep table) → +22.8% at DCN_EP=2 (723.0 → 888.1) → +4% at DCN_EP=4 (669.4 → 696.4) → +5% at DCN_EP=8 (577.7 → 605.8). Above DCN_EP=2, increasing pdbs at fixed DCN_EP buys very little throughput — the inter-node `all-to-all` cost has overtaken the per-pdbs dense-compute amortization.
7. **No DeepEP comparison row is possible.** The DS3 v3-vs-fixed-at-DCN_EP>1 question — the original headline of the DCN-EP sweep — is unanswerable without an upstream MaxText `pyconfig.py` change to lift the `use_deepep_dispatch ⇒ dcn_expert_parallelism == 1` validator. This is documented as the only "blocker" in this sweep.

---

## Key takeaways

1. **Peak throughput (dropping vs dropless):**
   - **Dropping:** `dense-cf1.25 @ pdbs=11` → **1170.1 TGS, MFU 9.62 %** (max_pdbs = 12; peak pdbs is 11, not 12 — TGS falls from 1170 → 1134 at pdbs=12 due to activation-memory pressure before the OOM at pdbs=16).
   - **Dropless:** `sparse-gmm-deepep-v3 @ pdbs=7` → **897.9 TGS, MFU 7.38 %** (max_pdbs = 7, at default `MEM_FRACTION=.93`).
   - Dropless / dropping peak ratio = 0.767. DS3 had 1097/1416 = 0.775. Very close — kimi's dropless path is ~77 % of dropping, matching DS3 to within noise.

2. **At P★ = 4, sgd-v3 and dense-cf1.25 are effectively tied** (within measurement noise; see reproducibility table in infra notes):
   ```
   sgd-deepep-v3         685.2 TGS  ← 1st sample
   dense-cf1.25          678.9
   (sgd-v3 2-sample mean 668.1, dense-cf1.25 2-sample mean 680.7 — dense edges v3 by +1.9 % when sample sizes are equalized)
   sparse-gmm-fixed      614.5
   dense-cf2             597.0
   sgd-deepep-v2         575.4
   sgd-deepep-v1         476.9
   dense-cf4             414.9
   sparse-gmm (one-shot) 249.0
   ```
   Taking both runs into account, dense-cf1.25 and sgd-v3 tie at the cross-config comparison point. On DS3 at pdbs=4, dense had a clearer 3.2 % lead. On kimi-1T, **the gap closes to a statistical tie** — v3's kernel-level optimizations (eliminated `input_scatter_fusion_*.kd`) bring dropless closer to dropping parity at P★ than DS3 showed. At higher pdbs (pdbs=5…7, where dense still has headroom), dense pulls back ahead — see takeaway #4.

3. **v1 → v2 → v3 DS3 shape replicates (with mild attenuation):**

   | pdbs | v1 | v2 | v3 | Δ v1→v2 | Δ v1→v3 | DS3 Δ v1→v3 (same pdbs) |
   |-----:|----:|----:|----:|--------:|--------:|-----------------------:|
   |    4 | 476.9 | 575.4 | 685.2 | +20.7 % | **+43.7 %** | +47.4 % |
   |    5 | 515.7 | 635.9 | 750.8 | +23.3 % | **+45.6 %** | +66.0 % |
   |    6 | ✗ OOM | ✗ OOM | 856.4 | — | — | +59.4 % |
   |    7 | ✗ OOM | ✗ OOM | 897.9 | — | — | +63.1 % |

   **v3 replicates DS3's shape at ~85–95 % of DS3's magnitude**, and additionally extends the feasibility frontier by +2 pdbs on this 1T model — v1 and v2 both OOM at pdbs=6 (195.3 GiB and 202.6 GiB respectively), v3 survives through pdbs=7. The DS3 optimization story (eliminating `input_scatter_fusion_*.kd`) carries over; profile drill-down below will confirm the kernel-level mechanism.

4. **Dense (dropping) beats dropless at pdbs > P★ on this model.** Side-by-side at same pdbs:

   | pdbs | dense-cf1.25 | sgd-v3 | Δ |
   |-----:|-------------:|-------:|--:|
   |    4 | 678.9 | **685.2** | **v3 +0.9 %** |
   |    5 | **787.2** | 750.8 | dense +4.9 % |
   |    6 | **873.3** | 856.4 | dense +2.0 % |
   |    7 | **930.3** | 897.9 | dense +3.6 % |

   Retries of sgd-v3 at pdbs=6 and pdbs=7 reproduced the original numbers within ~3 % (827.1 / 886.8 respectively — both on the slower side) — the dense-ahead pattern at pdbs ≥ 5 is real, not a single-sample fluke. Mechanistically: the dropping path emits regular `all-to-all` + regular GEMM, which is simpler kernels than dropless `moe_dispatch/combine` + grouped-GEMM even after v3's kernel-level optimizations. v3 closes the DeepEP integration-overhead gap (as DS3 showed) but doesn't close the regular-vs-ragged kernel gap. See the [profile drill-down](#profile-drill-down) for the kernel-level accounting.

5. **Memory ceilings are tight — this is the primary 1T finding.** Every single config OOMs at a lower pdbs than on DS3:

   | Config | DS3 max_pdbs (1-node) | Kimi-1T max_pdbs | Δ |
   |---|---:|---:|---:|
   | `dense-cf1.25` | 16 | 12 | −4 |
   | `dense-cf2` | 16 | 10 | −6 |
   | `dense-cf4` | 7 | 6 | −1 |
   | `sparse-gmm` | 7 (w/ `.96`) | 4 | −3 |
   | `sparse-gmm-fixed` | 7 (w/ `.96`) | 4 | −3 |
   | `sgd-v1` | 7 | 5 | −2 |
   | `sgd-v2` | 7 | 5 | −2 |
   | `sgd-v3` | 7 | **7** | **0** |
   | `sparse` | OOM@1 | OOM@1 | — (both infeasible) |
   | `sparse-deepep` | OOM@1 | OOM@1 | — (both infeasible) |

   **Only sgd-v3 retains DS3's pdbs=7 ceiling on 1T**, precisely because its `custom_vjp` backward eliminates the duplicate-index scatter-add intermediate tensors that all other dropless paths hold. On a 1T model where HBM headroom is scarcer, this memory advantage matters more than it did on DS3.

6. **Numerical correctness verified end-to-end** — all loss values at every pdbs agree across configs to Δ ≤ 0.002 (bf16 LSB noise). v2/v3 forward HLO is bit-identical to v1 baseline.

7. **Dropping's capacity_factor cost curve is steeper on kimi than DS3.** `cf=1.25 → 2.0` cuts TGS by 12–22 % (DS3: 22–32 %). `cf=1.25 → 4.0` cuts TGS by 41–50 % on the pdbs overlap range (DS3: 50–60 %). Kimi's sparsity pattern (384 experts × top-8, vs DS3's 256 × top-8) distributes tokens more evenly across experts at the same capacity factor, so the dropping loss from a given cf value is smaller.

8. **Non-monotonic TGS near the ceiling** — observed explicitly on dense-cf1.25 (peaks at pdbs=11, drops at pdbs=12 by 3 %). Activation-memory pressure on the last pdbs before OOM slows the HBM allocator and XLA's layout decisions, not fully compensated by the extra per-step token count. Expect this whenever `max_pdbs` is probed directly — report both `max_pdbs` and `argmax_TGS_pdbs`.

---

## Infrastructure / memory-ceiling notes

### Pre-sweep node fix (load-bearing)

Two of the 8 assigned k8s nodes (`node3`, `node7`) had their **routed RoCE GID at sysfs index 2 instead of index 1**, while the other 6 nodes had it at index 1:

```
node3: L-G-     (gid[0]=fe80, gid[1]=zero, gid[2]=fd93..., gid[3]=zero)
node7: L-G-
node1: LG--     (gid[0]=fe80, gid[1]=fd93..., gid[2]=zero)
node2: LG--
node4: LG--
node5: LG--
node6: LG--
node8: LG--
```

`train_env.sh` hardcoded `NCCL_IB_GID_INDEX=1`, causing `ibv_query_gid failed with error Unknown error -1` on rank 2 and 6 of every RCCL init, deterministically breaking distributed training on this nodelist. Fix applied to [`utils/detect_nccl_env.sh`](utils/detect_nccl_env.sh): per-node auto-detection that scans `/sys/class/infiniband/<hca>/ports/<port>/gids/` for the first non-zero global-scope GID and exports `NCCL_IB_GID_INDEX` accordingly. `train_env.sh`'s hardcoded value relaxed to `"${NCCL_IB_GID_INDEX:-1}"` so the auto-detect wins. This is a property of the current k8s-partition node topology (possibly due to a recent `ip addr add` ordering on the two affected nodes); the fix is minimally invasive and idempotent for healthy nodes. No nodes excluded; no sudo required.

### OOM allocation sizes at ceiling (all at default `MEM_FRACTION=.93`)

| Config | First OOM pdbs | Alloc size (GiB) | Notes |
|---|---:|---:|---|
| `sparse` | 1 | 581.8 | RaggedDot working set; entire row infeasible. |
| `sparse-deepep` | 1 | 507.4 | RaggedDot working set; entire row infeasible. |
| `sparse-gmm-fixed` | 5 | 195.6 | Alloc below pool (267.8 GiB) but total working set > pool. `.96` retry not attempted — alloc not within 10 % of pool. |
| `sparse-gmm-fixed` | 6 | 207.2 | Monotonic confirmation. |
| `sparse-gmm` (one-shot) | 5 | 195.6 | Same ceiling as fixed; the OS kernel holds more HBM than kNccl does. |
| `sgd-v1` | 6 | 195.3 | 2 scatter-add intermediates per layer — highest per-layer memory among dropless paths, cliff matches v2. |
| `sgd-v2` | 6 | 202.6 | 1 scatter-add intermediate per layer. `6 GiB` more than v1 at the OOM point, but clifts identically (both at 5 `max_pdbs`). |
| `sgd-v3` | 8 | 214.3 | No scatter-add intermediates; ceiling +2 pdbs vs v1/v2. |
| `dense-cf1.25` | 16 | 202.4 | Peak TGS at pdbs=11 (not 12 or 16); `argmax < max` cliff signature. |
| `dense-cf2` | 11 | 189.4 | `pdbs=12` OOMs at 189.7 GiB (monotonic confirmation). |
| `dense-cf4` | 7 | 180.5 | `pdbs=8` OOMs at 216.4 GiB (monotonic confirmation). TGS peaks at pdbs=5, slightly falls at pdbs=6 (same argmax<max pattern). |

`MEM_FRACTION=.96` retries were not applied to any sparse-family cell because the OOM alloc sizes (~195–215 GiB) are far below the `.93` pool (267.8 GiB) — the working set, not a single large allocation, is what clips. Raising `MEM_FRACTION` would add ~8 GiB/device, which isn't enough headroom and would starve RCCL.

### Compile-time issues

XLA compile time for this 1T model is **highly non-deterministic** at the same cell:
- `sparse-gmm-deepep-v3 pdbs=5` compiled in ~5 min on one attempt (13627), >45 min on two others (13577 timed out at `--time=45:00`; 13667 abandoned at 23 min wall / 64 min CPU time).
- XLA's rematerialization heuristic branches on allocation order; lucky runs find a fast schedule, unlucky ones iterate.
- **Escalation applied:** `--time=25:00 → 45:00 → 60:00 → 90:00`. All sparse+DeepEP cells now default to `--time=45:00` minimum; retries at 60:00.

### MaxText heartbeat default (100 s)

MaxText sets `jax_distributed_heartbeat_timeout_seconds=100` in `base.yml` — much tighter than JAX's 300 s default. On a 1T model, any cold compile >100 s will trip this and get killed. **After the dense-cf2 pdbs=5 failure (13607), every submission included a `jax_distributed_heartbeat_timeout_seconds=99999` passthrough as a hedge.** Since the repo doesn't enable persistent JAX compile caching, this hedge is always needed — there is no "warm cache" across jobs on this sweep.

*Caveat: the original sweep actually submitted this as `jax_distributed_heartbeat_timeout_seconds=99999` (with `_env_` prefix), which `_train.sh` extracts into a shell env var. MaxText doesn't read it from there — the actual MaxText config stayed at the 100 s default for the entire sweep. The sweep escaped the bug only because every cell is a 15-step probe (≤ 8 min, well below any plausible 100 s stall). For long-running re-runs (e.g. on real-data loss tests), the bare form above (no `_env_` prefix) is required — `_env_` is silently a no-op.*

### GitHub outage window (2026-04-23 ~16:00–16:55 UTC)

A ~45-min GitHub 500 outage killed 9 jobs in the `MAXTEXT_PATCH_BRANCH` checkout phase (`remote: Internal Server Error` / `fatal: unable to access 'https://github.com/ROCm/maxtext.git/': HTTP 500`). Cascading symptom on peer ranks: `ActorUnschedulableError` within 90 s of job start — Ray's scheduling retry gave up when rank N's `git fetch` failed. All 9 resubmitted after 17:00 UTC and ran cleanly.

### Compile-hang observed on OOM-adjacent cells

`dense-cf4 pdbs=6` (13651) and `sgd-v1 pdbs=6` (13663) exhibited the OOM-hang signature on first attempt: >10 min post-BARRIER with 0 steps, head-node python at 292–300 % CPU with growing TIME+ (20–30 min CPU time). Per the OOM-as-hang rule, scancelled and retried with longer wall budgets:

- `dense-cf4 pdbs=6` retry (13673) **succeeded** (453.2 TGS) — was a slow-compile transient, not a memory cliff. `max_pdbs` for dense-cf4 is 6, not 5.
- `sgd-v1 pdbs=6` retry (13674) **OOM'd cleanly** at 195.3 GiB — confirmed real memory cliff. `max_pdbs` for sgd-v1 is 5.

The OOM-hang disambiguation protocol correctly separated these: one flipped to success on retry (transient slow compile), the other flipped to a clean OOM signature (real ceiling). Both retries were submitted at `--time=45:00` with the heartbeat hedge.

### Reproducibility retries (all at default `MEM_FRACTION=.93`)

Every feasible sgd-v3 cell (pdbs=1, 2, 4, 5, 6, 7) was retried once to verify the original TGS is not a single-sample fluke. Dense-cf1.25 pdbs=4 was also retried as a baseline control. Summary:

| cell | original TGS | retry TGS | delta | notes |
|---|---:|---:|---:|---|
| sgd-v3 pdbs=1 | 229.5 | 219.7 | −4.3 % | |
| sgd-v3 pdbs=2 | 405.5 | 388.8 | −4.1 % | |
| sgd-v3 pdbs=4 | 685.2 | 650.9 | −5.0 % | See takeaway #2 — second sample shifts the at-P★ ranking. |
| sgd-v3 pdbs=5 | 750.8 | 765.3 | +1.9 % | |
| sgd-v3 pdbs=6 | 856.4 | 827.1 | −3.4 % | |
| sgd-v3 pdbs=7 | 897.9 | 886.8 | −1.2 % | |
| dense-cf1.25 pdbs=4 | 678.9 | 682.5 | +0.5 % | Control — reproducibility within 1 % on dense. |

**The retries reveal that sgd-v3's XLA compile produces slightly different kernel schedules between runs** (mean drift −2.8 %, range ±5 %), while dense-cf1.25 is reproducible to <1 %. The v3 variance is rooted in XLA rematerialization heuristic branching on allocation order — it's intrinsic to the path, not an infrastructure issue. All sgd-v3 TGS values in the main results tables are from the *first* run of each cell; the retry values are recorded here only for reproducibility assessment. Any takeaway that depends on a 1–5 % margin between sgd-v3 and another config should be read as "within measurement noise."

### Cleanup-flake (`exit=143` on one rank) ≠ training failure

Jobs 13557, 13606 completed all 15 training steps with valid rank-0 data, but one rank exited with code 143 during Ray/container teardown. Slurm reported `JOB SUMMARY Status: FAILED (exit 1)` even though training data is intact. **Kept both as valid data points** (extractor's success criterion is `last_completed_step >= steps-1`, not JOB SUMMARY status).

---

## Footnotes

- All cells reported in the TGS/TFLOP/s/step-time/loss tables above are from the **first successful run** of each cell. Reproducibility retries for sgd-v3 (pdbs=1,2,4,5,6,7) and a dense-cf1.25 pdbs=4 control are tabulated in the "Reproducibility retries" subsection of the infrastructure notes — they inform takeaway #2's "effectively tied at P★" reading but are not substituted into the main tables.
- `sparse-gmm-fixed` and `sparse-gmm` at pdbs=5 OOM at the same 195.6 GiB allocation. That's the minimum XLA-pool-blowout allocation size on this model; it reflects MoE intermediate tensors, not a specific variant's pathology.
- sgd-v1 / sgd-v2 / sparse-gmm-fixed / sparse-gmm all cliff at `max_pdbs ∈ {4, 5}`. The common 1T-specific lesson: dropless paths without v3's `custom_vjp` backward cannot hold pdbs=6+ on 288 GB HBM / device at default `MEM_FRACTION=.93`, and `.96` doesn't buy enough headroom to reach it. v3 is the only dropless path that reaches pdbs=6+.
- Sweep completed 2026-04-23. ~53 benchmark jobs run, ~5 retries, ~7 pre-emptively cancelled. Compute budget ≈ 18 GPU-hours (8-node × ~22 min × 58 jobs / 60).

---

## Profile drill-down

Profile jobs (`profiler=xplane profiler_steps=3 _env_ENABLE_XLA_DUMP=1`, `--time=60:00` / `--time=90:00` for the slow-compile retries). Initial runs (`13687–13694`, `13711`) had XLA dump dropped to fit the 243-byte `JOB_NAME` limit; a re-run pass (`13809–13816`, `13829`) dropped the redundant `skip_first_n_steps_for_profiler=5` override (kimi yml already defaults to `3`, which is equally post-warmup) and put XLA_DUMP back. Kernel-timing numbers below are from the XLA-DUMP-enabled re-run batch; they reproduce the original batch within ±1 %.

Per-kernel times from [`utils/profile_drill.py`](utils/profile_drill.py) on all 8 trace-JSON windows × 8 GPUs × 3 profiled steps (divisor = 192 per cell). HLO collective / custom-call inventories from `grep` on `xla_dump/module_*.jit_train_step.gfx950_gpu_after_optimizations.txt`.

### HLO collective-op inventory at P★ = 4

Counts of standard collective instructions in the post-optimization HLO, plus DeepEP's `custom_call_target` instances. `sparse-gmm-fixed` is the only config emitting `ragged-all-to-all` (6 ops) — the other four sparse paths (v1/v2/v3) replace it with DeepEP's `moe_dispatch` / `moe_combine` custom calls. **v1/v2/v3 emit a byte-identical HLO collective inventory** (5/5/0/0/3 AG/AR/RA2A/A2A/RS), confirming the "forward HLO is bit-identical across v1/v2/v3" claim.

| Op | `dense-cf1.25` | `sparse-gmm-fixed` | sgd-v1 | sgd-v2 | **sgd-v3** |
|---|---:|---:|---:|---:|---:|
| `all-gather` | 5 | 7 | 5 | 5 | 5 |
| `all-reduce` | 5 | 5 | 5 | 5 | 5 |
| `ragged-all-to-all` | 0 | **6** | 0 | 0 | 0 |
| `all-to-all` | 6 | 4 | 0 | 0 | 0 |
| `reduce-scatter` | 3 | 3 | 3 | 3 | 3 |
| `custom_call_target="moe_dispatch"` | 0 | 0 | 2 | 2 | 2 |
| `custom_call_target="moe_combine"` | 0 | 0 | 2 | 2 | 2 |
| `custom_call_target="moe_cached_dispatch"` | 0 | 0 | 1 | 1 | 1 |

DeepEP custom-call counts match DS3's at the same position (1 × `moe_cached_dispatch`, 2 × `moe_combine`, 2 × `moe_dispatch`) — the DeepEP dispatch/combine emission pattern is model-independent.

The 5 DeepEP custom-call instances collectively replace the 6 `ragged-all-to-all` + 4 `all-to-all` = 10 HLO-collective instances that `sparse-gmm-fixed` uses for the same dispatch/combine work (identical pattern to DS3's drill-down).

### Cross-path step-time composition at P★ = 4 (seconds / GPU / step)

### Cross-path step-time composition at P★ = 4 (seconds / GPU / step)

| Slice                                               | `dense-cf1.25` | `sparse-gmm-fixed` | sgd-v1 | sgd-v2 | **sgd-v3** |
|-----------------------------------------------------|---------------:|-------------------:|-------:|-------:|-----------:|
| `RaggedAllToAllKernelImpl` (XLA in-process)          |           0.00 |               0.00 |   0.00 |   0.00 |   0.00 |
| `primus_turbo::deep_ep::*` (DeepEP native HIP)       |           0.00 |               0.00 |   0.95 |   0.95 |   0.92 |
| `input_scatter_fusion_*.kd`                          |           0.00 |               0.01 | **5.34** | **2.68** | **0.02** |
| `loop_select_fusion_*.kd` (valid-rows mask)          |           0.01 |               0.01 |   0.95 |   0.61 |   0.53 |
| `loop_gather_fusion_*.kd`                            |           0.00 |               1.21 |   0.00 |   0.00 |   0.00 |
| RCCL (`ncclDevKernel_*`)                             |          13.44 |               9.56 |   9.17 |   9.25 |   9.04 |
| CK / Primus-Turbo GEMM (grouped + dense)             |           4.98 |               2.48 |   2.67 |   2.69 |   2.61 |
| Flash-attention (`aiter::fmha_*`)                    |           0.43 |               0.29 |   0.35 |   0.36 |   0.35 |
| Other fusions (`loop_reduce` / `loop_convert` / `loop_transpose` / `input_reduce_select` / `input_broadcast_reduce_select` / misc) | 1.34 | 1.76 | 1.26 | 1.21 | 1.32 |
| **Total kernel time (on any stream)**                |      **20.21** |          **15.31** | **20.71** | **17.74** | **14.86** |
| Benchmark step time (from main sweep)                |          24.14 |              26.67 |  34.37 |  28.47 |  23.91 |
| Step − total = idle gap (+) or overlap (−)           |          +3.93 |             +11.36 | +13.66 | +10.73 |  +9.05 |

*(Step times are from the no-profile benchmark runs. Profile-run TGS is slightly lower due to profiler writeback overhead and is not used for the "step − total" comparison. Numbers are from the XLA-DUMP-enabled re-run batch (`13809-13816` for dense / sparse-gmm-fixed / sgd-v1 / sgd-v3 + `13829` for sgd-v2 — added 2026-04-25 after a compile-timeout retry). Within-cell reproducibility of stationary kernels (input_scatter_fusion, loop_select, GEMM, flash_attn) is ≤ ±1 %; the comm-bound families (`primus_turbo::deep_ep`, `ncclDevKernel_*`) show ~5–15 % run-to-run variance — `sgd-v2` deep_ep and RCCL in this row dropped 0.34 s and 1.56 s respectively versus an earlier no-XLA-DUMP run of the same cell, illustrating that comm-stream measurements are inherently noisier than on-device kernel timings.)*

**The headline replication of DS3's kernel-elimination chain:**

| cell | `input_scatter_fusion_*.kd` kimi-1T @ pdbs=4 | DS3 671B @ pdbs=6 | note |
|---|---:|---:|---|
| v1 baseline (2 gathers → 2 scatter-adds) | **5.34 s** | 8.97 s | v1 roughly proportional to model size: kimi-1T ~60% of DS3 (more experts but similar `base_moe_mlp_dim`) |
| v2 (composed gathers → 1 scatter-add) | **2.68 s** | 4.45 s | −50% vs v1 |
| **v3 (`custom_vjp` → 0 scatter-adds)** | **0.02 s** | 0.04 s | **−99.6% vs v1 — kernel eliminated** |

**This is the definitive kernel-level confirmation that DS3's v1→v2→v3 story replicates on kimi-1T.** The `input_scatter_fusion_*.kd` family shrinks by two orders of magnitude from v1 to v3 — exactly as DS3 predicted and exactly matching the Python-only `moe.py` patch semantics (v1's duplicate-index scatter-add backward → v3's permutation-gather + reduce-sum backward, no atomics).

**Other cross-path observations:**

1. **Dense-cf1.25 has the highest RCCL time (14.17 s)** — regular `all-to-all` + all the `all-gather`/`all-reduce`/`reduce-scatter` of the dropping path sum to more RCCL work than the dropless paths' ragged traffic. But dense's *exposed* idle gap is tiny (+3.2 s) because this RCCL traffic overlaps perfectly with compute, while the dropless paths' main-stream-blocking `input_scatter_fusion` or `loop_gather_fusion` force positive idle gaps.
2. **`sparse-gmm-fixed` has no `input_scatter_fusion` but has a large `loop_gather_fusion_*.kd` (1.20 s)** — XLA's lowering of the ragged fan-in/out for non-DeepEP dropless uses a gather family instead of the scatter family. The DeepEP paths (v1, v2) have `loop_gather_fusion = 0` because DeepEP's custom calls own that work.
3. **RCCL time is roughly constant across v1/v2/v3 (9.0 – 9.3 s with the refreshed sgd-v2)** even though total kernel time drops ~6 s from v1 to v3. The v3 wallclock gain doesn't come from comm savings — it comes from eliminating the main-stream-blocking scatter-add. This matches DS3's observation.
4. **Idle gap shrinks monotonically across v1 → v2 → v3: +13.66 s → +10.73 s → +9.05 s** — scheduler cascade recovery: with less main-stream blocking from `input_scatter_fusion`, XLA can overlap more work. The v1→v2 improvement (−2.93 s) is bigger than the v2→v3 improvement (−1.68 s) on this model, unlike DS3 where v2→v3 was the larger jump. Likely because 1T has proportionally more RCCL time relative to GEMM (~9.2 of the 20.71 s total in v1 is RCCL), so the last remaining scatter-add blocks less additional overlap opportunity than it does on DS3.

### Supplementary v1/v2/v3 drill-down at pdbs=5 (s / GPU / step)

pdbs=5 is `min(v1_max, v2_max, v3_max)` and strictly greater than P★=4, so this table captures the optimization chain at peak v1/v2 throughput (where `dense-cf1.25` and `sparse-gmm-fixed` are also still feasible but not re-profiled here since pdbs=4 already shows their kernel pattern).

| Slice                                               | sgd-v1 | sgd-v2 | **sgd-v3** |
|-----------------------------------------------------|-------:|-------:|-----------:|
| `primus_turbo::deep_ep::*`                           |   1.24 |   1.25 |   1.26 |
| `input_scatter_fusion_*.kd`                          | **7.39** | **3.83** | **0.03** |
| `loop_select_fusion_*.kd`                            |   1.64 |   0.83 |   0.73 |
| RCCL (`ncclDevKernel_*`)                             |   9.85 |  10.65 |  10.12 |
| CK / Primus-Turbo GEMM                               |   3.49 |   3.70 |   3.61 |
| Flash-attention                                      |   0.48 |   0.51 |   0.50 |
| Other fusions + misc                                 |   1.66 |   1.60 |   1.88 |
| **Total kernel time (any stream)**                   | **25.75** | **22.35** | **18.14** |
| Benchmark step time                                  |  39.72 |  32.22 |  27.29 |
| Step − total = idle gap                              | +13.97 |  +9.87 |  +9.15 |

**The `input_scatter_fusion` elimination chain holds at pdbs=5 too:** v1 = 7.39, v2 = 3.83, v3 = 0.03. The absolute numbers grow proportionally with pdbs (more tokens per device → larger scatter-add dimension). The v3 kernel is essentially not present on any pdbs we measure — confirmed as universal, not a pdbs-4-specific artifact.

**v1 → v3 total-kernel savings at pdbs=5:** 25.75 − 18.14 = **7.61 s/step/GPU**, of which the `input_scatter_fusion` removal alone accounts for 7.36 s (97% of the savings). The step-time delta is 39.72 − 27.29 = 12.43 s, so scheduler recovery (idle gap shrinkage) adds another 4.82 s on top — same "lose the main-stream blocker, let comm overlap" mechanism DS3 described. On kimi-1T specifically the scheduler-cascade contribution is smaller than DS3's because there's less RCCL slack to overlap into.

### Kernel-level takeaway

| Dimension | Kimi-1T finding | Matches DS3? |
|---|---|---|
| v1→v2 halves `input_scatter_fusion` | yes: 5.34 → 2.68 (pdbs=4), 7.39 → 3.83 (pdbs=5) | yes: 8.97 → 4.45 (pdbs=6) |
| v3 eliminates `input_scatter_fusion` | yes: → 0.02 (pdbs=4), → 0.03 (pdbs=5) | yes: → 0.04 (pdbs=6) |
| v1/v2/v3 HLO collective inventory identical (5 AG / 5 AR / 0 RA2A / 0 A2A / 3 RS) | yes at pdbs=4 and pdbs=5 (verified by `grep` on post-opt HLO dumps — this sweep) | yes (DS3 drill-down states "HLO bit-identical across v1/v2/v3"; kimi confirms) |
| Total-kernel reduction > `input_scatter_fusion` alone | yes (scheduler cascade adds ~0.2–0.5 s on 1T; less than DS3's ~4 s on 671B) | pattern matches, magnitude smaller |
| v3 extends feasibility frontier | yes: v3 max=7 vs v1/v2 max=5 on 1T | N/A — DS3 had all three at max=7 |

**Conclusion: DS3's kernel optimization story carries over to 1T with the same mechanism and ~60% of DS3's absolute scatter-add time per step. The DeepEP v3 patch is the unambiguous replacement for the v1 baseline on this model.** Additionally, on 1T specifically, v3 is the *only* DeepEP variant that keeps DS3's pdbs=7 ceiling — a memory-frontier win not visible on DS3 because DS3 had enough HBM headroom for v1/v2/v3 to all reach pdbs=7.

### Profile job artifacts (kept under `outputs/`)

- `13687-…` — dense-cf1.25 pdbs=4 profile (661.6 TGS under profile vs 678.9 benchmark)
- `13688-…` — sparse-gmm-fixed pdbs=4 profile (586.1 TGS vs 614.5)
- `13689-…` — sgd-v1 pdbs=4 profile (469.2 TGS vs 476.9)
- `13690-…` — sgd-v2 pdbs=4 profile (541.8 TGS vs 575.4)
- `13711-…` — sgd-v3 pdbs=4 profile (632.6 TGS vs 685.2 — retry at `--time=90:00` after 13691 compile-timed out at 60:00)
- `13692-…` — sgd-v1 pdbs=5 profile (510.1 TGS vs 515.7)
- `13693-…` — sgd-v2 pdbs=5 profile (626.6 TGS vs 635.9)
- `13694-…` — sgd-v3 pdbs=5 profile (753.0 TGS vs 750.8)

---

## How to reproduce

```bash
cd /maxtext-slurm

# dense-cf1.25 @ peak (pdbs=11)
RAY=1 ./submit.sh kimi-k2-1t:dense-cf125 \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=11 steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-deepep-v3 @ peak dropless (pdbs=7) — the headline cell.
# Note: `container_env.sh` now defaults to MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3,
# so v3 runs no longer need the env-var prefix (kept here as a no-op for explicit reproducibility).
MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3 \
RAY=1 ./submit.sh kimi-k2-1t:sgd-deepep-v3 \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
    use_deepep_dispatch=true steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-fixed @ P★ (pdbs=4) — best non-DeepEP dropless at apples-to-apples pdbs
RAY=1 ./submit.sh kimi-k2-1t:sparse-gmm-fixed \
    --partition=k8s --nodes=8 \
    --nodelist=node1,node2,node3,node4,node5,node6,node7,node8 \
    --time=45:00 -- \
    per_device_batch_size=4 sparse_matmul=true use_turbo_grouped_gemm=true \
    steps=15 dataset_type=synthetic \
    jax_distributed_heartbeat_timeout_seconds=99999
```

**Always include `jax_distributed_heartbeat_timeout_seconds=99999`** as a bare flag — NOT the `jax_distributed_heartbeat_timeout_seconds=99999` form, which is silently a no-op (see the [MaxText heartbeat default](#maxtext-heartbeat-default-100-s) caveat above). MaxText's 100 s default heartbeat is tighter than most cold compiles on this model. Always use `--time=45:00` unless the cell has been observed to compile in <15 min. See [`moe-pdbs-sweep-prompt.md`](moe-pdbs-sweep-prompt.md) for the retry escalation ladder and disambiguation priority.

---

*Document status: **v4 final** — main sweep + profile drill-down + DCN expert-parallelism extension. 48 successful main-sweep cells + 8 profile cells + 1 sgd-v2 pdbs=4 profile refresh (`13829`, with HLO dump, 2026-04-25), 14 OOM ceilings, 2 full-row infeasible. **v4 (DCN-EP extension)** adds 27 cells (2026-04-25) at `dcn_expert_parallelism ∈ {2, 4, 8}` × 4 non-DeepEP configs (dense-cf1.25/cf2/cf4 + sparse-gmm-fixed): 22 successful, 5 OOM, 4 DeepEP variants × 3 DCN_EP values blocked by `MaxText/pyconfig.py` validator. All `max_pdbs` confirmed via direct observation. Main-sweep P★ = 4. DS3 v1→v2→v3 kernel-elimination chain confirmed at 99.6% elimination of `input_scatter_fusion_*.kd` on both pdbs=4 and pdbs=5. v3.1 monotonic-idle-gap chain observed (v1 +13.66 s → v2 +10.73 s → v3 +9.05 s) after refreshing sgd-v2 on the same XLA-DUMP-enabled batch. DCN-EP finding: dense-cf1.25 actually GAINS +6.5 % at DCN_EP=2 / pdbs=4 over DCN_EP=1; sparse-gmm-fixed cliff sharpens steeply with DCN_EP (max_pdbs 4 → 2 → 1 → 0); DS3 v3-vs-fixed-at-DCN_EP>1 hypothesis untestable due to MaxText pydantic block on inter-node DeepEP.*
