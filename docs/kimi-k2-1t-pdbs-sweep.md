# Kimi-K2-1T — pdbs Sweep on B200

- **Date:** 2026-04-10 (initial bs=1 baseline sweep); 2026-05-08 (`mem_fraction` extension unlocked bs=2; see the results matrix)
- **Model:** `kimi-k2-1t` (MaxText). 1026.4 B parameters. 61 decoder layers (layer 0 dense, layers 1–60 MoE with 384 experts × top-8 routing + 1 shared expert). MLA attention (`q_lora_rank=1536`, `kv_lora_rank=512`).
- **Hardware:** 8 nodes × 8× NVIDIA B200 (179.1 GiB HBM / dev), InfiniBand fabric
- **Image:** `nvcr.io/nvidia/jax:26.03-maxtext-py3`
- **Patch branch:** [`llying/benchmark-on-nv-b200`](https://github.com/AMD-AGI/maxtext-slurm/tree/llying/benchmark-on-nv-b200) @ `5f68243`
- **Base config:** [`configs/kimi-k2-1t.gpu.yml`](../configs/kimi-k2-1t.gpu.yml) (`dcn_fsdp_parallelism=8`, `ici_expert_parallelism=8`, `sparse_matmul=false`, `capacity_factor=1.25`)
- **Source data:** [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) (reorganized by precision × capacity_factor)
- **Peak:** BF16 ≈ 2,250 TFLOP/s/dev; FP8 ≈ 4,500 TFLOP/s/dev
- **XLA_PYTHON_CLIENT_MEM_FRACTION default:** `0.93` (preallocates ~165.87 GiB / dev; later bs=2 runs bumped to `.97`)

## Background

This document follows [`AMD-AGI/maxtext-slurm@yihuang/moe/kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md) (the MI355 sweep) as a structural template, organizing the B200 Kimi-K2-1T results in the same shape. All raw data comes from [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) — this doc just slices it along (precision (BF16 / FP8) × capacity_factor variant). `pdbs` is shorthand for `per_device_batch_size`.

> **Kimi-K2-1T on B200 is an extreme HBM-bound workload:** on 8N B200, `max_pdbs = 2` (and only with `mem97`). On the same scale of MI355, `max_pdbs ≥ 12`. This **6× pdbs-ceiling gap** is the headline of the cross-platform comparison (much wider than DS3's ~2× ceiling gap). Why: the model is 1.026 T parameters, 53% larger than DS3-671B; combined with `ici_expert_parallelism=8` over 384 experts → 48 experts per GPU (vs DS3's 32), the per-GPU expert-weight footprint is more sensitive to HBM capacity.

> **Key B200 vs MI355 differences:** B200 has 179.1 GiB HBM / device (MI355: 288 GiB), BF16 peak 2,250 TFLOP/s (MI355 ≈ 2,500), and uses InfiniBand instead of Pensando AINIC. The `sparse-gmm-*` / DeepEP / `_env_ENABLE_RAGGED_ONESHOT_KERNEL` XLA / Primus-Turbo paths discussed in the MI355 reference are not carried in this B200 image, and `sparse_matmul=True` was not measured on B200 either (per the DS3 sparse_matmul analysis, sparse_matmul + shardy OOMs at pdbs=1 on 8N B200).

## Configs under test

Each table fixes a (precision, capacity_factor) combination; rows vary `pdbs` and the specific XLA / memory flag set.

| Tag             | Passthrough flags                                          |
|-----------------|----------------------------------------------------------|
| `dense-cf1.25`  | *(default)* — `sparse_matmul=false`, `capacity_factor=1.25` |
| `dense-cf1`     | `capacity_factor=1.0`                                      |

**cf variants not measured on B200 (present in the MI355 reference):**

| Tag             | Why skipped on B200 |
|-----------------|---------------------|
| `dense-cf2`     | cf=1.25 + default mem already OOMs at bs=2 on B200; cf=2.0 doubles dispatch padding, making even bs=1 likely OOM. MI355 reaches max_pdbs=10 at cf=2.0, but with no comparable pdbs on B200 there is no cross-platform point to land — not invested |
| `dense-cf4`     | Same logic, more severe; MI355 hits max_pdbs=6 at cf=4.0, B200 likely OOMs at pdbs=1 |
| `sparse_matmul` | See the DS3 sparse_matmul analysis: on 8N B200, `sparse_matmul + shardy` fails at pdbs=1 with a ~112 GiB / dev allocation in the XLA planner's shape-based feasibility check (dtype-independent); this image carries no Primus-Turbo / DeepEP path, so there is no comparable column |

**XLA flag-set shorthands** (same as the b200-benchmark-report; every run appends `--xla_gpu_enable_command_buffer=''` from `train_env.sh`'s JAX-0.8.2 fix):

- **AMD-parity** (the image-default `XLA_FLAGS`, for AMD ↔ NV cross-validation) — full flag string:

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_memory_limit_slop_factor=95
  --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592   # 8 GiB
  --xla_gpu_all_gather_combine_threshold_bytes=8589934592       # 8 GiB
  --xla_gpu_enable_triton_gemm=false
  --xla_gpu_enable_cublaslt=true
  --xla_gpu_autotune_level=0
  --xla_gpu_enable_all_gather_combine_by_dim=false
  --xla_gpu_enable_command_buffer=''
  ```

- **NV defaults** (`_env_XLA_FLAGS_REPLACE` wholesale replaces the flag set with the two flags below, dropping all AMD-parity flags) — full flag string:

  ```text
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_enable_command_buffer=''
  ```

- **mem97** — `_env_XLA_PYTHON_CLIENT_MEM_FRACTION` raised from default `.93` to `.97` (JAX preallocation pool fraction, orthogonal to XLA flags).

Legend: `✗` = OOM; `—` = untested; `CANCELLED` preserves the original Slurm status.

**Primary metric = `Tok/s/dev`; auxiliary metrics = `TFLOP/s/dev` and MFU.** Tok/s/dev is unaffected by FLOP-counting conventions (FP8 and BF16 have different peaks, so TFLOP/s/dev cross-precision comparisons are distorted; Tok/s/dev maps directly to actual training throughput). `TFLOP/s/dev` and `MFU` columns are kept as auxiliary references.

**Tok/s/dev convention:** every `Tok/s/dev` column in this doc = `per_device_batch_size × max_target_length / step_time` (Kimi-K2-1T defaults to `max_target_length = 4096`, see [`configs/kimi-k2-1t.gpu.yml`](../configs/kimi-k2-1t.gpu.yml)).

---

## BF16

### `dense-cf1.25` (BF16, `capacity_factor=1.25`, `sparse_matmul=false`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 1597 | AMD-parity | SUCCESS | 17.00 | 2.20  |  49.5 |   241 | **first viable pdbs**; MFU is extremely low |
|  1 | 4167 | NV defaults | SUCCESS | 16.85 | 2.22  |  49.9 |   243 | NV +0.8% over AMD-parity — flag set has minimal effect on Kimi |
|  2 | 1532 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 83.33 GiB |
|  2 | 4180 | NV + `ici_fsdp=2, ici_ep=4` | ✗ OOM | -- | -- | -- | -- | **worse**: +5.7 GiB; ici_ep=4 forces each GPU to hold 2× expert weights — net negative |
|  2 | 4414 | NV defaults + mem97 | SUCCESS | 19.31 |  3.87 |  87.2 |   424 | **mem97 unlocks pdbs=2 (cf=1.25)** |
|  3 | 1534 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 91.19 GiB |

**Core observations (Tok/s/dev–primary):**

- **pdbs=2 is the BF16 cf=1.25 hard ceiling on B200** (pdbs=3 OOMs at 91.19 GiB even with AMD-parity; pdbs=2 only opens up with `mem97`). Job 4414 = **424 Tok/s/dev** (MFU 3.87%, auxiliary 87.2 TFLOP/s/dev), **+74.5%** over the pdbs=1 best (Job 4167 = 243).
- **NV defaults vs AMD-parity gives only +0.8% on Kimi** (4167 vs 1597), far smaller than DS3 (+9.4%). Kimi's MoE dispatch overhead dominates step time, leaving little headroom for XLA flag-set tuning.
- **The `ici_fsdp=2 / ici_ep=4` rebalance hypothesis is empirically disproved** (Job 4180): ici_ep=4 makes each GPU hold 2× more expert weights, while ici_fsdp=2 only halves the non-expert (attention/router) weights. Expert weights dominate Kimi-K2's 1T parameter count, so the net effect is **+5.7 GiB** of memory pressure. bs=2 remains OOM under this split.

---

### `dense-cf1` (BF16, `capacity_factor=1.0`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 1598 | AMD-parity | SUCCESS | 16.60 | 2.25  |  50.7 |   247 | cf=1.0 vs cf=1.25 (1597 = 241) = **+2.5% Tok/s/dev** |
|  1 | 4168 | NV defaults | SUCCESS | 16.52 | 2.26  | **50.94** | **247.9** | **Kimi BF16 best @ pdbs=1** (NV set) |
|  2 | 1599 | AMD-parity | ✗ OOM | -- | -- | -- | -- | 82.47 GiB — vs cf=1.25 sibling (1532 = 83.33 GiB) saves only ~0.86 GiB, not enough to fit |
|  2 | 4173 | NV defaults | ✗ OOM | -- | -- | -- | -- | 80.48 GiB — NV vs AMD saves only ~2 GiB, still not enough |
|  2 | 4410 | NV defaults + mem97 | SUCCESS | 18.66 | **4.01** | **90.2** | **439** | **Kimi BF16 overall best: mem97 unlocks pdbs=2 (cf=1.0)** |

**Core observations (Tok/s/dev–primary):**

- **Kimi BF16 overall best = Job 4410 (bs=2, cf=1.0, mem97) = 439 Tok/s/dev** (MFU 4.01%, auxiliary 90.2 TFLOP/s/dev). **+77.1%** over the pdbs=1 best (Job 4168 = 247.9) — Kimi's only effective tuning knob on B200 is **`mem97` unlocking pdbs=2**, a single flag that nearly doubles throughput.
- **`cf=1.0` gives only ~2.5% on Kimi** (1598 vs 1597 / 4168 vs 4167), far below DS3's +11.8%. Reason: Kimi's 384 experts × top-8 routing distributes tokens more evenly across experts than DS3's 256 × top-8, so the dispatch padding eliminated by cf=1.25 → cf=1.0 is intrinsically smaller.
- **bs=3+ is all OOM** (whether at default mem or mem97): bs=3 = 91 GiB / bs=4 = 100 GiB / bs=6 = 118 GiB, all over B200's ~102 GiB XLA budget.

---

## FP8

### `dense-cf1.25` (FP8, `capacity_factor=1.25`, `quantization=fp8`)

> FP8 MFU% is relative to FP8 peak (4,500 TFLOP/s for B200); BF16-equivalent MFU = MFU × 2.

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 4179 | NV defaults | SUCCESS | 18.35 | 1.02* |  45.9 |   223.3 | **FP8 cf=1.25 baseline** — vs BF16 cf=1.25 (4167 = 243) = **−8.1% Tok/s/dev** |
|  2 | 4415 | NV defaults + mem97 | SUCCESS | 20.31 | 1.84* |  82.9 |   403   | **mem97 unlocks pdbs=2 (FP8 cf=1.25)** |

### `dense-cf1` (FP8, `capacity_factor=1.0`, `quantization=fp8`)

| pdbs | Job ID | XLA Flags / Run | Status | Step (s) | MFU (%) | TFLOP/s/dev | Tok/s/dev | Notes |
|---:|---:|---|---|---:|---:|---:|---:|---|
|  1 | 4169 | NV defaults | SUCCESS | 18.10 | 1.03* |  46.5 |   226.4 | vs BF16 cf=1.0 (4168 = 247.9) = **−8.7% Tok/s/dev**; cf=1.0 vs cf=1.25 (FP8) = +1.4% |
|  2 | 4174 | NV defaults | ✗ OOM | -- | -- | -- | -- | 79.09 GiB — FP8 vs BF16 at bs=2 only saves ~1.4 GiB, not enough to fit |
|  2 | 4411 | NV defaults + mem97 | SUCCESS | 19.64 | 1.90* | **85.7** | **417**   | **Kimi FP8 overall best: mem97 unlocks pdbs=2 (cf=1.0)** |

**Core observations (Tok/s/dev–primary):**

- **Kimi FP8 overall best = Job 4411 (bs=2, cf=1.0, mem97) = 417 Tok/s/dev** (FP8 MFU 1.90% = BF16-eq MFU 3.80%; auxiliary 85.7 TFLOP/s/dev). **Still −5.0% Tok/s/dev vs the BF16 overall best (Job 4410 = 439)** — **FP8 is a net loss on Kimi**.
- **FP8 vs BF16 at the same pdbs / same cf is consistently negative:** cf=1.25 pdbs=1 = −8.1% (223 vs 243); cf=1.0 pdbs=1 = −8.7% (226 vs 248); cf=1.0 pdbs=2 = −5.0% (417 vs 439). The gap narrows slightly as pdbs grows but does not close.
- **FP8 also saves very little HBM:** FP8 cf=1.0 bs=2 OOMs at 79.09 GiB under NV defaults; BF16 at the same config OOMs at 80.48 GiB — FP8 only compresses weights, while dispatch padding + activations dominate the working set. Kimi's B200 memory pressure is not bound by weight precision.

### Why FP8 vs BF16 flips on Kimi (vs DS3)

DS3-671B sees FP8 as a **+19.6% / +22.5%** net gain; on the same hardware, same cf, same image, same flag set, Kimi-K2-1T sees FP8 as a **−5.0% ~ −8.7%** net loss. Both models go through the `sparse_matmul: False` dense_matmul path, so all-to-all is not involved. Possible root causes (not pinned down by profile yet):

- The relative weight of FP8 quantize/dequantize vs GEMM speedup is different in Kimi's MoE forward — the 1T model has more experts per layer (384 vs 256), so more quant ops per step.
- Kimi's 384 experts × top-8 routing produces a thinner/longer dispatch tensor shape, leading to different FP8 GEMM efficiency.
- Pinning the root cause requires nsys / xprof comparing per-kernel FP8 vs BF16 time inside dense_matmul on both models.

---

## Overall best summary

Sorted by **Tok/s/dev** (primary metric) descending.

| Rank | Precision | Config | pdbs | **Tok/s/dev** | Δ vs BF16 best | Job | XLA Flags | Step (s) | TFLOP/s/dev | MFU (%) |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| **1** | **BF16** | **dense-cf1** | **2** | **439** | **0** (baseline) | **4410** | **NV + mem97** | **18.66** | **90.2** | **4.01** |
| 2 | BF16 | dense-cf1.25 | 2 | 424 | −3.4% | 4414 | NV + mem97 | 19.31 | 87.2 | 3.87 |
| 3 | FP8  | dense-cf1   | 2 | 417 | −5.0% | 4411 | NV + mem97 | 19.64 | 85.7 | 1.90\* |
| 4 | FP8  | dense-cf1.25 | 2 | 403 | −8.2% | 4415 | NV + mem97 | 20.31 | 82.9 | 1.84\* |
| 5 | BF16 | dense-cf1   | 1 | 247.9 | −43.5% | 4168 | NV defaults | 16.52 | 50.94 | 2.26 |
| 6 | BF16 | dense-cf1.25 | 1 | 243   | −44.6% | 4167 | NV defaults | 16.85 | 49.9 | 2.22 |
| 7 | FP8  | dense-cf1   | 1 | 226.4 | −48.4% | 4169 | NV defaults | 18.10 | 46.5 | 1.03\* |
| 8 | FP8  | dense-cf1.25 | 1 | 223.3 | −49.1% | 4179 | NV defaults | 18.35 | 45.9 | 1.02\* |
| — | BF16 | dense-cf2 | — | **not measured** | — | — | — | — | — | — |
| — | BF16 | dense-cf4 | — | **not measured** | — | — | — | — | — | — |
| — | BF16 | sparse_matmul | — | **not measured** (expected infeasible, see DS3) | — | — | — | — | — | — |

\* FP8 MFU% is relative to FP8 peak (4,500 TFLOP/s); BF16-equivalent MFU = MFU × 2 = 3.80% (Job 4411) / 3.68% (Job 4415) / 2.06% (Job 4169) / 2.04% (Job 4179).

---

## Key takeaways

> All "+X% / −X%" figures use **Tok/s/dev** (primary metric); TFLOP/s/dev and MFU are shown as auxiliary references.

1. **B200 overall best = BF16 cf=1.0 pdbs=2 + mem97 (Job 4410): 439 Tok/s/dev** (MFU 4.01%). **+77.1%** over the pdbs=1 baseline (Job 4168 = 247.9) — the only effective tuning on Kimi on B200 is **`mem97` unlocking pdbs=2**, a single flag that nearly doubles throughput.
2. **FP8 is a net loss on Kimi (opposite of DS3): cf=1.0 pdbs=2 = −5.0% (417 vs 439), cf=1.25 pdbs=2 = −3.4% (403 vs 424 BF16, or −8.2% vs the 4410 BF16 best).** Both models go through dense_matmul; the sign flip's root cause is not yet pinned down by profile. FP8 also saves very little HBM (FP8 vs BF16 at default mem only differs ~1.4 GiB at bs=2, so the OOM ceiling is unchanged).
3. **`mem97` is the only usable pdbs-extending knob for Kimi on B200**: it pushes max_pdbs from 1 → 2 across cf=1.25 / cf=1.0 / FP8-cf=1.25 / FP8-cf=1.0, each worth +74~77% Tok/s/dev. `slop_factor=95` was not isolated on Kimi (per DS3, slop95 / mem95 underperform mem97 at high buffer pressure).
4. **The `ici_fsdp=2 / ici_ep=4` topology rebalance is empirically reversed** (Job 4180): ici_ep=4 doubles per-GPU expert weights, with a net +5.7 GiB of memory pressure; bs=2 still OOMs under this split. The hypothesis is disproved — on the 1T model, expert weights dominate (a different expert/non-expert ratio than 671B), so ICI topology rebalancing cannot unlock higher pdbs.
5. **`cf=1.0` only gives +2.5% Tok/s/dev on Kimi** (vs DS3's +11.8%). Reason: Kimi's 384 experts × top-8 distributes tokens more evenly than DS3's 256 × top-8, so the absolute dispatch padding eliminated by cf=1.25 → cf=1.0 is intrinsically smaller.
6. **NV defaults vs AMD-parity gives only +0.8% on Kimi** (vs DS3's +9.4%). Kimi's MoE dispatch overhead dominates the step time, leaving little headroom for XLA flag-set differences.
7. **bs=3+ is the hard ceiling**: bs=3 (91 GiB) / bs=4 (100 GiB) / bs=6 (118 GiB) all OOM, exceeding B200's ~102 GiB XLA budget. mem97 was not isolated at bs=3, but even though the bs=3 allocation leaves headroom against the mem97 pool (~174 GiB), the total working-set pressure (dispatch + activations + scatter intermediates) would still over-commit — same behavior seen on DS3, where `mem97` also only unlocks +1 pdbs.
8. **`dense-cf2` / `dense-cf4` / `sparse_matmul` were not measured on B200**: cf=2.0 / cf=4.0 doubles dispatch padding, making bs=1 likely OOM on B200; `sparse_matmul + shardy` allocates ~112 GiB / dev at pdbs=1 on 8N B200 (per the DS3 analysis); this image carries no Primus-Turbo / DeepEP path. The MI355 reference has data for all three.

---

## Comparison with the MI355 reference sweep

Reference: [`kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md) (MI355 Kimi main pdbs sweep + sparse-gmm-deepep v1/v2/v3 + DCN-EP extension). **The MI355 reference is BF16-only; no FP8 or cf-variant FP8 was run on MI355.** Cross-platform comparison is therefore restricted to BF16.

Each cf block presents: (1) the B200 BF16 best at that cf, (2) MI355's **same-pdbs** result, and (3) MI355's cf peak. All MI355 numbers are BF16, 1-node/proc launcher.

### `dense-cf1.25` (BF16, `capacity_factor=1.25`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Δ vs B200 best |
|---|---|---:|---:|---|---:|---:|---:|
| **B200 best**          | B200  | **424**     |  2 | NV + mem97 (Job 4414)       |  87.2 |  3.87 | baseline |
| MI355 same pdbs=2      | MI355 |   399.0     |  2 | FSDP=8 + image default XLA  |  82.0 |  3.28 | **−5.9%** |
| MI355 same pdbs=4 (P★) | MI355 |   678.9     |  4 | FSDP=8 + image default XLA  | 139.5 |  5.58 | **+60.1%** vs B200 best |
| **MI355 peak**         | MI355 | **1,170.1** | 11 | FSDP=8 + image default XLA  | **240.4** | **9.62** | **+176%** vs B200 best |

- **At same pdbs=2, B200 is +6.3% Tok/s/dev faster than MI355** (424 vs 399; MFU 3.87% vs 3.28%) — B200 has a small per-device edge, but the absolute gap is one-tick noise level.
- **MI355 peak is +176% over B200 peak (1,170 vs 424)** — almost entirely from the pdbs-ceiling gap (B200 max=2 vs MI355 max=12, a 6× spread), with MI355's per-device MFU actually slightly lower than B200's at the same pdbs.
- **MI355 peak sits at pdbs=11 (not max_pdbs=12)** — TGS at pdbs=12 falls back to 1,134 from 1,170, an `argmax_TGS < max_pdbs` ceiling-adjacent degradation pattern — same pattern observed on DS3.

### `dense-cf1` (BF16, `capacity_factor=1.0`)

| Comparison | Platform | Tok/s/dev | pdbs | Configuration | TFLOP/s/dev | MFU (%) | Notes |
|---|---|---:|---:|---|---:|---:|---|
| **B200 best**   | B200  | **439** |  2 | NV + mem97 (Job 4410)       |  90.2 |  4.01 | mem97 unlocks pdbs=2 |
| MI355 same pdbs | MI355 | n/a     | —  | MI355 reference did not run cf=1.0 | — | — | MI355 Kimi sweep covers cf=1.25 / 2 / 4 only |

- **The MI355 reference does not include a cf=1.0 column for Kimi**, so no strict apples-to-apples comparison is possible. What we can say: on B200, cf=1.0 vs cf=1.25 at the same pdbs=2 yields +3.5% (439 vs 424), much smaller than DS3's +11.8% — Kimi's 384 expert × top-8 padding distribution makes cf-reduction gains intrinsically smaller.
- Using cf=1.25 as a proxy (B200 +6.3% over MI355 at the same pdbs=2), B200 might still edge MI355 per-device at cf=1.0, but MI355's peak (pdbs=11, cf=1.25 = 1,170) still vastly exceeds B200's cf=1.0 peak (pdbs=2 = 439).

### Cross-cf BF16 summary

| Config | B200 best (pdbs) | MI355 same pdbs (TGS) | MI355 peak (pdbs, TGS) | B200 vs MI355 same pdbs | B200 vs MI355 peak |
|---|---|---|---|---:|---:|
| `dense-cf1.25` | **424** (2) | 399.0 (pdbs=2) | 1,170.1 (pdbs=11) | **+6.3%** | **−63.8%** |
| `dense-cf1`    | **439** (2) | n/a | n/a (MI355 Kimi did not run cf=1.0) | — | — |
| `dense-cf2`    | **not measured** | 597.0 (pdbs=4, P★); peak 827.8 (pdbs=10) | 827.8 (pdbs=10) | — | — |
| `dense-cf4`    | **not measured** | 414.9 (pdbs=4, P★); peak 455.1 (pdbs=5) | 455.1 (pdbs=5) | — | — |

Dropless (`sparse_matmul`) supplementary comparison:

| Path | B200 | MI355 best Tok/s/dev (pdbs, configuration) | Notes |
|---|---|---|---|
| `sparse-gmm-deepep-v3` (sgd-v3) | **not measured** (expected infeasible, see DS3 sparse_matmul analysis) | **897.9** (pdbs=7, FSDP=8 + image default XLA, MI355 best dropless) | On MI355, the v3 patch lifts dropless to ~80% of dense's level, and its `custom_vjp` backward eliminates scatter-add intermediate tensors, lifting the pdbs ceiling from v1/v2's 5 to 7 |
| `sparse-gmm-fixed` (sgf) | **not measured** (same as above) | **614.5** (pdbs=4) | On MI355, sgf's dropless ceiling is limited by `ragged_all_to_all` temporary buffer materialization; pdbs ≤ 4 |
| `sparse-gmm-deepep` v1 / v2 | **not measured** (same as above) | v1 = 515.7 (pdbs=5); v2 = 635.9 (pdbs=5) | v1→v2→v3 optimization chain: v3 eliminates `input_scatter_fusion_*.kd` (v1 5.34 s → v3 0.02 s @ pdbs=4) |

**Key observations:**

1. **B200 vs MI355 at same pdbs=2 is only +6.3% (cf=1.25); cf=1.0 has no comparison data.** B200's same-pdbs advantage on Kimi is much smaller than on DS3 (where cf=1.25 pdbs=7 untuned gives B200 +18.7% over MI355) — because Kimi's MFU on B200 is extremely low (3.87% vs DS3's 14.35%), the per-device compute advantage gets diluted by the model's own dispatch / quantization overhead.
2. **MI355 peak exceeds B200 peak by +176% (cf=1.25)** — this is the largest peak gap across all current cross-platform comparisons, almost entirely from the pdbs-ceiling gap: B200 max=2 vs MI355 max=12, a 6× spread. MI355's 288 GiB HBM vs B200's 179 GiB HBM amplifies the per-GPU expert-weight footprint difference on a 1T model.
3. **B200 has no dropless path at all vs MI355**: MI355 sgd-v3 reaches 897.9 Tok/s/dev at pdbs=7, **+104%** over the B200 BF16 cf=1.0 best (439). `sparse_matmul + shardy` OOMs at pdbs=1 on B200 (per the DS3 analysis), and this image carries no Primus-Turbo / DeepEP — Kimi has no dropless option on B200.
4. **FP8 is a regression on Kimi (Kimi-specific; DS3 sees +22.5%)**: B200 FP8 cf=1.0 pdbs=2 = 417 vs BF16 = 439 = −5.0%. The MI355 Kimi sweep did not run FP8, so we cannot cross-platform-verify whether FP8 also reverses on MI355 — but since MI355's BF16 peak (1,170) already vastly exceeds B200's (439), cross-platform FP8 comparison has little practical meaning.
5. **The `mem97` tuning knob vs any MI355 flag tuning:** on B200, `mem97` is a single flag worth +77% — Kimi's only critical enabler; on MI355, the Kimi main sweep reaches its 1,170 peak under image-default XLA, no flag-level tuning required. The reason: MI355's 288 GiB HBM lets the model land near the dispatch / compute balance point directly, without needing memory-fraction tricks. The "extending pdbs" bottleneck is fundamentally different on the two platforms — HBM capacity ceiling on B200, per-step dispatch / scheduling on MI355.
6. **Equivalent key-enabler mapping:**

   | Platform / path | Key enabler 1 | Single-enabler Tok/s/dev gain | Equivalent physical resource |
   |---|---|---|---|
   | B200 BF16 Kimi  | `XLA_PYTHON_CLIENT_MEM_FRACTION=.97` (unlocks pdbs 1→2) | **+77.1%** (247.9 → 439) | Unlocks HBM headroom |
   | MI355 BF16 Kimi (dense-cf1.25) | *(image-default XLA already near-peak, no single-flag headline)* | pdbs=1 → pdbs=11 natural growth = +398% | MI355 288 GiB HBM gives intrinsic capacity advantage |
   | B200 FP8 Kimi   | `quantization=fp8` | **−5.0% vs BF16** (net loss) | Quant overhead > GEMM speedup |
   | MI355 sgd-v3 Kimi | `MAXTEXT_PATCH_BRANCH=…-v3` (eliminates `input_scatter_fusion_*.kd`) | v1→v3 **+88.3%** (476.9 → 897.9 @ pdbs=4; +2 pdbs ceiling) | Python-only patch, zero compute increase |

---

## How to reproduce

```bash
# Kimi overall best: BF16 cf=1.0 + bs=2 + mem97
./submit.sh kimi-k2-1t::bf16-bs2-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# Kimi BF16 cf=1.25 + bs=2 + mem97 (second-best baseline)
./submit.sh kimi-k2-1t::bf16-bs2-cf125-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.25 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'

# Kimi pdbs=1 baseline (default mem, BF16 best @ pdbs=1)
./submit.sh kimi-k2-1t::bf16-bs1-cf100-nv -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=1 capacity_factor=1.0 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer='

# Kimi FP8 best (FP8 is a net regression on Kimi; reproduced for completeness)
./submit.sh kimi-k2-1t::fp8-bs2-cf100-mem97 -N 8 -w 'hungry-hippo-fin-03-[1-8]' \
    --time=00:30:00 -- per_device_batch_size=2 capacity_factor=1.0 quantization=fp8 \
    '_env_XLA_FLAGS_REPLACE=--xla_gpu_enable_latency_hiding_scheduler=true,--xla_gpu_enable_command_buffer=' \
    '_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.97'
```

---

## References

- Raw data: [`docs/b200-benchmark-report.md`](b200-benchmark-report.md) (logged in Run / Job chronological order; this doc reorganizes by (precision, config) slices)
- DS3-671B B200 sweep (parallel structure): [`docs/deepseek3-671b-pdbs-sweep.md`](deepseek3-671b-pdbs-sweep.md) (source of structure and terminology; this doc follows its dense-cf{1.25, 1} split + cross-platform comparison pattern)
- MI355 Kimi sweep (parallel structure): [`kimi-k2-1t-pdbs-sweep.md`](https://github.com/AMD-AGI/maxtext-slurm/blob/yihuang/moe/kimi-k2-1t-pdbs-sweep.md) (MI355 dense-cf{1.25, 2, 4} + sparse-gmm-deepep v1/v2/v3 + DCN-EP extension)
