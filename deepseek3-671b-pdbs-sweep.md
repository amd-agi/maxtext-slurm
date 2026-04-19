# DeepSeek-V3 671B — comprehensive pdbs sweep

- **Date:** 2026-04-17 (initial sweep); 2026-04-18 (post-sweep extension — `sparse-gmm-fixed` column and key-takeaway #8); 2026-04-22 (post-sweep extension — `sparse-gmm-deepep-v2` / `sparse-gmm-deepep-v3` columns and key-takeaway #9); 2026-04-25 (DCN expert-parallelism extension `dcn_expert_parallelism ∈ {2, 4, 8}` for the 4 non-DeepEP configs, see ["DCN expert-parallelism extension"](#dcn-expert-parallelism-extension-dcn_expert_parallelism--1) section after the main matrix)
- **Model:** `deepseek3-671b` (MaxText)
- **Hardware:** 8 nodes × 8× AMD MI355 (288 GB HBM / device), Pensando AINIC interconnect
- **Image:** `/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar` (includes [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) GMM + DeepEP)
- **Patch branches:**
  - [yihuang/moe-turbo-gmm-and-deepep](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep) @ `ad693da2` (baseline — `sparse-gmm-deepep` column)
  - [yihuang/moe-turbo-gmm-and-deepep-v2](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v2) @ `627168f8` (single commit on baseline — `sparse-gmm-deepep-v2` column)
  - [yihuang/moe-turbo-gmm-and-deepep-v3](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9` (two commits on baseline — `sparse-gmm-deepep-v3` column)
- **Base config:** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)
- **Peak BF16:** ≈ 2500 TFLOP/s/device → MFU ≈ TFLOP/25

## Background

The 1-GPU/proc launcher ([maxtext-slurm#111](https://github.com/AMD-AGI/maxtext-slurm/pull/111)) was originally added to enable [**mori-EP**](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md) (AMD's high-performance MoE dispatch/combine kernel library, similar in spirit to DeepEP, which requires one JAX process per GPU), which ultimately did not pan out on this model.  During stress-testing we accidentally noticed `sparse-gmm 1-GPU/proc` was running **~3× faster than `sparse-gmm 1-node/proc`** — a delta unrelated to anything mori-EP-specific.  Profiling (pdbs=6, jobs **12895 / 12916 / 12897**) traced the cause to XLA's `ragged-all-to-all` thunk: it picks the naive in-process kernel `RaggedAllToAllKernelImpl<8l>` when all EP ranks share a process (1-node/proc), and the much faster `kNccl` path when they don't (1-GPU/proc).  Once the mechanism was clear, the faster path turned out to be reachable on 1-node/proc too — via a single XLA flag — making the launcher switch unnecessary.  That fix shipped in [maxtext-slurm#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112); the `sparse-gmm-fixed (1-node)` column in the results matrix below measures its impact.  The detailed drill-down is in the [Why do the three sparse variants differ in TGS?](#why-do-the-three-sparse-variants-differ-in-tgs-pdbs6-profile-drill-down) section.

> **Default change (2026-04-18, [maxtext-slurm#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112)):** `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` is now the default in `train_env.sh` (was opt-in during the initial sweep).  Running `sparse-gmm` on 1-node/proc without any extra flags therefore produces the `sparse-gmm-fixed (1-node)` numbers below — the `sparse-gmm (1-node / 1-GPU)` column shows historical data from when XLA's in-process one-shot kernel was still in effect.  Verified no-op on dense / deepep / 1-GPU paths (takeaway #8); set `_env_ENABLE_RAGGED_ONESHOT_KERNEL=1` to restore XLA's one-shot kernel for debugging.

---

## Configs under test

| Tag                    | Passthrough flags                                                                           |
|------------------------|---------------------------------------------------------------------------------------------|
| `dense-cf1.25`         | *(default)* — `sparse_matmul=false`, `capacity_factor=1.25`                                |
| `dense-cf2`            | `capacity_factor=2.0`                                                                       |
| `dense-cf4`            | `capacity_factor=4.0`                                                                       |
| `sparse`               | `sparse_matmul=true shardy=true`                                                            |
| `sparse-gmm`           | `sparse_matmul=true use_turbo_grouped_gemm=true _env_ENABLE_RAGGED_ONESHOT_KERNEL=1`        |
| `sparse-deepep`        | `sparse_matmul=true use_deepep_dispatch=true shardy=true`                                   |
| `sparse-gmm-deepep`    | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true` (patch branch `yihuang/moe-turbo-gmm-and-deepep`) |
| `sparse-gmm-deepep-v2` | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true` (patch branch `yihuang/moe-turbo-gmm-and-deepep-v2`) |
| `sparse-gmm-deepep-v3` | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true` (patch branch `yihuang/moe-turbo-gmm-and-deepep-v3`) |
| `sparse-gmm-fixed`     | `sparse_matmul=true use_turbo_grouped_gemm=true`                                            |

**What distinguishes `sparse-gmm-deepep`, `-v2`, `-v3`:** the three rows run the same image / launcher / passthrough flags — only the patched MaxText branch differs, and the only file that differs between the branches is `src/MaxText/layers/moe.py`. **v2** (commit `627168f8`) composes the DeepEP fan-out gather with the sort permutation into a single gather (halves the dispatch-side scatter-add count in the backward). **v3** (commit `f59be3c9`) replaces the resulting duplicate-index scatter-add backward with a `jax.custom_vjp` that folds the top-K duplicates with an argsort-inverse gather + reduce-sum — no atomics. The forward output is bit-identical across all three (same loss at every step within bf16 LSB noise); the two changes together eliminate the dominant `input_scatter_fusion_*.kd` kernel from the dispatch backward.

**Why only `sparse` and `sparse-deepep` carry `shardy=true`:** `sparse_matmul=true` without `use_turbo_grouped_gemm=true` falls back to `jax.lax.ragged_dot` for the expert matmul, whose sharding propagation requires `shardy=true` (the Shardy framework, XLA's successor to GSPMD) — hence the extra flag on those two rows.  `sparse-gmm` and `sparse-gmm-deepep` don't need it because the Primus-Turbo GMM custom call carries its own sharding spec and sidesteps the propagation pass.  `use_deepep_dispatch=true` replaces `ragged_all_to_all` with Primus-Turbo's DeepEP intranode custom call and does *not* itself impose a shardy requirement — the shardy dependency is a property of the `ragged_dot` matmul path, orthogonal to the dispatch path.

**What `_env_ENABLE_RAGGED_ONESHOT_KERNEL` controls:** maps directly to XLA flag `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel`.  With **`=1`** (the `sparse-gmm` row above; XLA's original default, pre-[#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112)), 1-node/proc uses the in-process one-shot kernel `stream_executor::gpu::RaggedAllToAllKernelImpl<8l>` for `ragged-all-to-all`, producing the `sparse-gmm (1-node / 1-GPU)` column's numbers.  With **`=0`** (the `sparse-gmm-fixed` row above; now the default in `train_env.sh`), XLA's ragged thunk falls back to its `kNccl` path — the same RCCL-based lowering 1-GPU/proc gets automatically at runtime — producing the `sparse-gmm-fixed (1-node)` column.  `train_env.sh` wires this env var in as an append to the image's default `XLA_FLAGS` so the tuning defaults (`xla_gpu_enable_cublaslt`, `xla_gpu_enable_latency_hiding_scheduler`, etc.) are preserved.  Since `0` is now the repo default, plain `sparse-gmm` submissions reproduce the `sparse-gmm-fixed` column; the explicit `=1` override is needed only to reproduce the historical pre-flag column.

## Launcher modes

| Tag             | Passthrough flag                        | Notes                                                             |
|-----------------|-----------------------------------------|-------------------------------------------------------------------|
| `1-node/proc`   | *(default)*                             | One Python process per node; JAX sees 8 local devices             |
| `1-GPU/proc`    | `_env_ONE_GPU_PER_PROCESS=true`         | Eight processes per node; each JAX process owns exactly one GPU   |

> **These are process-granularity modes, not job sizes.**  Every cell in the results matrix below is measured on the **full 8-node × 64-GPU** topology regardless of launcher; the only difference is whether each node runs **one** JAX process (`1-node/proc`) or **eight** (`1-GPU/proc`).  Throughout the rest of the doc, shorthand column headers like `sparse-gmm (1-node / 1-GPU)` and `sparse-gmm-fixed (1-node)` refer to these launcher modes — **not** to 1-node or 1-GPU jobs.

---

## Feasibility summary

**Feasible (1-node / 1-GPU):**
- `dense-cf1.25` ✓ / ✓
- `dense-cf2` ✓ / ✓
- `dense-cf4` ✓ / ✓
- `sparse-gmm` ✓ / ✓
- `sparse-gmm-deepep` ✓ / ✗ (AssertionError — see below)
- `sparse-gmm-deepep-v2` ✓ / ✗ (same AssertionError as baseline)
- `sparse-gmm-deepep-v3` ✓ / ✗ (same AssertionError as baseline)
- `sparse-gmm-fixed` ✓ / — (1-node only; the flag is redundant on 1-GPU/proc since that launcher already lowers to the `kNccl` path automatically)

**Infeasible at pdbs=1 (category skipped):**

| Config            | 1-node                               | 1-GPU                                 |
|-------------------|--------------------------------------|---------------------------------------|
| `sparse`          | OOM 444 GiB (RaggedDot)              | OOM 444 GiB (RaggedDot)               |
| `sparse-deepep`   | OOM 375 GiB (RaggedDot)              | `AssertionError: EP ranks=1`          |
| `sparse-gmm-deepep` | *(feasible)*                       | `AssertionError: EP ranks=1` at every pdbs |

`AssertionError: Unsupported number of EP ranks: 1` comes from Primus-Turbo's DeepEP Python binding, which hardcodes `num_ranks = jax.local_device_count()`. In 1-GPU/proc that's always 1. Fundamental API incompatibility.

---

## Results matrix — `1-node / 1-GPU` per cell

All metrics except loss are **mean over training steps 5–14** (steps 0–4 discarded as warmup).  Loss is reported from step 14 only since the synthetic-data loss at a single step is a consistent numerical-correctness probe.

Legend: `✗` = OOM; `—` = skipped (earlier pdbs already OOM'd for the combo).

### Tokens/s/device (TGS)

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 333 / 326                     | 311 / 307                  | 265 / 259                  | 163 / 281                    | 305                       | 271                        | 294                           | **317**                       |
| 2    | 563 / 535                     | 497 / 471                  | 374 / 369                  | 224 / 496                    | 514                       | 415                        | 454                           | **545**                       |
| 4    | 867 / 822                     | 721 / 723                  | 500 / 493                  | 275 / 768                    | 782                       | 569                        | 676                           | **839**                       |
| 5    | 962 / 959                     | 796 / 775                  | 535 / 531                  | 288 / 872                    | 880                       | 614                        | 751                           | **948**                       |
| 6    | 1040 / 1043                   | 835 / 808                  | 543 / 548                  | 298 / 942                    | 949                       | 647                        | 806                           | **1030**                      |
| 7    | 1086 / 1080                   | 884 / 867                  | 560 / 540                  | 302ᵃ / ✗ᵇ                   | 989ᵃ                      | 673                        | 836                           | **1097**                      |
| 8    | 1191 / 1171                   | 918 / 928                  | 566 / 571                  | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 1416 / 1387                   | 968 / 966                  | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

### TFLOP/s/device

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 83.3 / 81.6                   | 77.9 / 76.8                | 66.2 / 64.9                | 40.9 / 70.5                  | 76.3                      | 67.8                       | 73.6                          | 79.3                          |
| 2    | 140.9 / 134.1                 | 124.4 / 117.9              | 93.6 / 92.4                | 56.1 / 124.3                 | 128.8                     | 103.8                      | 113.8                         | 136.5                         |
| 4    | 217.2 / 205.8                 | 180.7 / 181.2              | 125.2 / 123.5              | 68.9 / 192.5                 | 195.8                     | 142.5                      | 169.2                         | 210.1                         |
| 5    | 241.0 / 240.2                 | 199.4 / 194.1              | 133.9 / 133.1              | 72.1 / 218.5                 | 220.4                     | 153.8                      | 188.1                         | 237.5                         |
| 6    | 260.6 / 261.2                 | 209.2 / 202.5              | 135.9 / 137.2              | 74.6 / 236.0                 | 237.6                     | 162.2                      | 201.9                         | 257.8                         |
| 7    | 272.1 / 270.5                 | 221.3 / 217.0              | 140.1 / 135.3              | 75.7ᵃ / ✗ᵇ                  | 247.7ᵃ                    | 168.5                      | 209.3                         | **274.8**                     |
| 8    | 298.2 / 293.3                 | 230.0 / 232.3              | 141.7 / 143.0              | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 354.7 / 347.3                 | 242.4 / 242.0              | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

**Peak MFU:** `dense-cf1.25 @ pdbs=16, 1-node` = **14.19 %** (354.7 TFLOP/s/device, peak BF16 ≈ 2500 TFLOP/s/device on MI355). **Peak dropless MFU:** `sparse-gmm-deepep-v3 @ pdbs=7, 1-node` = **10.99 %** (274.8 TFLOP/s/device) — no `MEM_FRACTION` bump needed.

### Average per-step time (seconds)

Lower is better. Mean of the per-step wall times (`seconds:` field in the training log) over steps 5–14.

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 12.3 / 12.6                   | 13.2 / 13.4                | 15.5 / 15.8                | 25.1 / 14.6                  | 13.4                      | 15.1                       | 13.9                          | **12.9**                      |
| 2    | 14.6 / 15.3                   | 16.5 / 17.4                | 21.9 / 22.2                | 36.6 / 16.5                  | 15.9                      | 19.8                       | 18.1                          | **15.0**                      |
| 4    | 18.9 / 20.0                   | 22.7 / 22.7                | 32.8 / 33.2                | 59.6 / 21.3                  | 21.0                      | 28.8                       | 24.3                          | **19.5**                      |
| 5    | 21.3 / 21.4                   | 25.7 / 26.4                | 38.3 / 38.5                | 71.2 / 23.5                  | 23.3                      | 33.4                       | 27.3                          | **21.6**                      |
| 6    | 23.6 / 23.6                   | 29.4 / 30.4                | 45.3 / 44.9                | 82.5 / 26.1                  | 25.9                      | 38.0                       | 30.5                          | **23.9**                      |
| 7    | 26.4 / 26.5                   | 32.4 / 33.1                | 51.2 / 53.1                | 94.8ᵃ / ✗ᵇ                  | 29.0ᵃ                     | 42.6                       | 34.3                          | **26.1**                      |
| 8    | 27.5 / 28.0                   | 35.7 / 35.3                | 57.9 / 57.4                | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 46.3 / 47.3                   | 67.7 / 67.8                | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

### Training loss at step 14

Within each pdbs row, all configs/launchers agree to Δ ≤ 0.003 — launcher choice does not perturb numerics, and the v2/v3 `moe.py` changes are forward-bit-identical to baseline (deltas at the bf16 LSB). Sparse sits ~0.02 below dense only at pdbs=1 because sparse is dropless whereas dense drops whatever `capacity_factor` doesn't hold.

| pdbs | dense-cf1.25 (1-node / 1-GPU) | dense-cf2 (1-node / 1-GPU) | dense-cf4 (1-node / 1-GPU) | sparse-gmm (1-node / 1-GPU) | sparse-gmm-fixed (1-node) | sparse-gmm-deepep (1-node) | sparse-gmm-deepep-v2 (1-node) | sparse-gmm-deepep-v3 (1-node) |
|------|-------------------------------|----------------------------|----------------------------|------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|
| 1    | 7.714 / 7.715                 | 7.712 / 7.714              | 7.713 / 7.714              | 7.692 / 7.694                | 7.694                     | 7.694                      | 7.692                         | 7.693                         |
| 2    | 8.594 / 8.594                 | 8.594 / 8.594              | 8.592 / 8.592              | 8.592 / 8.591                | 8.592                     | 8.592                      | 8.592                         | 8.592                         |
| 4    | 9.439 / 9.439                 | 9.439 / 9.439              | 9.437 / 9.437              | 9.438 / 9.438                | 9.437                     | 9.437                      | 9.438                         | 9.438                         |
| 5    | 9.684 / 9.684                 | 9.682 / 9.682              | 9.682 / 9.682              | 9.682 / 9.681                | 9.681                     | 9.680                      | 9.680                         | 9.680                         |
| 6    | 9.884 / 9.884                 | 9.884 / 9.884              | 9.883 / 9.883              | 9.883 / 9.883                | 9.883                     | 9.883                      | 9.883                         | 9.883                         |
| 7    | 10.031 / 10.031               | 10.030 / 10.030            | 10.030 / 10.030            | 10.030ᵃ / ✗ᵇ                | 10.030ᵃ                   | 10.029                     | 10.029                        | 10.029                        |
| 8    | 10.157 / 10.157               | 10.157 / 10.157            | 10.156 / 10.156            | ✗ / ✗                        | ✗ᶜ                        | ✗                          | ✗                             | ✗                             |
| 16   | 10.821 / 10.821               | 10.820 / 10.820            | ✗ / ✗                      | —                            | ✗ᶜ                        | —                          | —                             | —                             |

---

## DCN expert-parallelism extension (`dcn_expert_parallelism > 1`)

*(Added 2026-04-25.)* Extends the main matrix above (which runs at the DS3 default of `dcn_expert_parallelism=1`, i.e. expert parallelism is purely intranode and FSDP is purely inter-node) by walking the inter-node EP factorization. On 8 nodes × 8 GPUs = 64 ranks the parallelism grid stays the same total but is re-split:

| `DCN_EP` | `dcn_fsdp` × `ici_ep × dcn_ep` | total EP rank-product | EP fanout per host |
|---:|:-:|---:|---|
| **1** *(default — main matrix)* | 8 × 8 × 1 | 8 | EP axis is intranode-only |
| 2 | 4 × 8 × 2 | 16 | each host's experts spread to 1 peer host |
| 4 | 2 × 8 × 4 | 32 | each host's experts spread to 3 peer hosts |
| 8 | 1 × 8 × 8 | 64 | full DCN-EP, no inter-node FSDP |

### Known limitation: DeepEP variants are gated to `DCN_EP=1`

`MaxText/pyconfig.py` validates `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` and rejects the config with a pydantic `ValidationError("Internode DeepEP is not yet supported in JAX")` in ~2 min before reaching XLA compile. This applies to **all 4 DeepEP configs** in the main matrix (`sparse-deepep`, `sparse-gmm-deepep` v1/v2/v3) — the JAX/MaxText integration layer blocks the very regime the [DeepEP today section's "Where DeepEP's design already wins" callout](#deepep-today-competitive-kernel-expensive-integration--and-where-it-would-shine) identifies as "Inter-node EP (RDMA-backed AllToAll)." The DCN-EP extension below characterizes only the **non-DeepEP** regime — the 3 dense-cf configs and `sparse-gmm-fixed`. `sparse-gmm` (one-shot) is also skipped at DCN_EP > 1: its OneShot kernel is intranode-only, falls back to the kNccl path at DCN_EP > 1, and would just duplicate `sparse-gmm-fixed`.

### TGS @ DCN_EP > 1 (4 non-DeepEP configs, 1-node/proc)

DCN_EP=1 column copied from the main matrix (1-node column) for cross-comparison. Cells marked `n/a` are pdbs values not measured at that DCN_EP.

#### `dense-cf1.25`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  867 |     799.7 |     649.3 |   537.8 |
|  8 | 1191 |     898.8 |     667.6 |   585.5 |

#### `dense-cf2`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  721 |     577.5 |     443.0 |   377.9 |
|  8 |  918 |     620.3 |     449.0 |   393.4 |

#### `dense-cf4`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  4 |  500 |     339.6 |     237.7 |   204.5 |
|  6 |  543 |     337.0 | ✗ OOM-hang | (skipped: pdbs=4 already at ceiling) |

#### `sparse-gmm-fixed`

| pdbs | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 |
|---:|---:|---:|---:|---:|
|  1 |  305 |     364.7 |     338.4 |   333.6 |
|  2 |  514 |     497.3 | ✗ 214.6 GiB |     n/a |
|  4 |  782 | ✗ 224 GiB | ✗ 332 GiB |     n/a |

### Cross-DCN_EP TGS comparison at pdbs=4 (cross-config baseline)

| Config | DCN_EP=1 | DCN_EP=2 | DCN_EP=4 | DCN_EP=8 | Δ EP=1→8 |
|---|---:|---:|---:|---:|---:|
| `dense-cf1.25`  | 867 | 799.7 *(−7.7%)*    | 649.3 *(−25.1%)*  | 537.8 *(−38.0%)* | **−38.0%** |
| `dense-cf2`     | 721 | 577.5 *(−19.9%)*   | 443.0 *(−38.6%)*  | 377.9 *(−47.6%)* | **−47.6%** |
| `dense-cf4`     | 500 | 339.6 *(−32.1%)*   | 237.7 *(−52.5%)*  | 204.5 *(−59.1%)* | **−59.1%** |
| `sparse-gmm-fixed` | 782 | ✗ OOM   | ✗ OOM    | ✗ OOM (implied) | infeasible at DCN_EP > 1 / pdbs=4 |

Loss values agree to ε across all DCN_EP variants for every config-pdbs cell — DCN_EP factorization is numerically transparent, as expected (cross-DCN_EP loss values for pdbs=4 are 9.439 / 9.439 / 9.439 for dense-cf1.25; same pattern for cf2 / cf4 / sparse-gmm-fixed).

### Feasibility summary across DCN_EP (sparse-gmm-fixed cliff)

| Config | DCN_EP=1 max_pdbs | DCN_EP=2 max_pdbs | DCN_EP=4 max_pdbs | DCN_EP=8 max_pdbs |
|---|---:|---:|---:|---:|
| `dense-cf1.25` | ≥ 16 | ≥ 8 | ≥ 8 | ≥ 8 |
| `dense-cf2`    | ≥ 16 | ≥ 8 | ≥ 8 | ≥ 8 |
| `dense-cf4`    | 7   | ≥ 6 | **4 or 5** *(pdbs=4 ✓ 237.7; pdbs=6 OOM-hang; pdbs=5 not probed)* | **4 or 5** *(pdbs=4 ✓ 204.5; pdbs=6 not probed)* |
| `sparse-gmm-fixed` | 7 | **2** *(pdbs=3 not probed; pdbs=4 OOM)* | **1** *(pdbs=2 OOM 214.6 GiB)* | **1** *(pdbs=1 → 333.6 TGS, pdbs=2/4 not probed)* |

### DCN-EP key observations

1. **DS3 dense-cf1.25 at pdbs=4 monotonically degrades with DCN_EP** (867 → 800 → 649 → 538, −7.7% / −18.7% / −17.2% per doubling). This is *qualitatively different* from the kimi-1T DCN-EP extension's [headline finding](kimi-k2-1t-pdbs-sweep.md#dcn-ep-key-observations) where dense-cf1.25 *gained* +6.5% at DCN_EP=2 over DCN_EP=1: kimi-1T has 384 experts (48 → 24 experts/GPU at DCN_EP=2 saves significant per-rank expert weight memory and compensates for the inter-node `all-to-all` RDMA cost), while DS3 has 256 experts (32 → 16 experts/GPU saves less per-rank memory and the RDMA cost dominates from the start). On smaller-expert MoE models, DCN_EP > 1 is purely a cost without an offsetting win; on larger-expert MoE models, there's a small-pdbs window where DCN_EP=2 wins.

2. **TGS gain from pdbs=4 → pdbs=8 collapses at DCN_EP > 1.** At DCN_EP=1 dense-cf1.25 gains +37% from pdbs=4 → pdbs=8 (867 → 1191); at DCN_EP=2 the gain shrinks to +12% (799.7 → 898.8); at DCN_EP=4 it's only +2.8% (649.3 → 667.6); at DCN_EP=8 it's +8.9% (537.8 → 585.5). dense-cf2 shows the same pattern (+27% / +7.4% / +1.4% / +4.1%). This means **the throughput-per-pdbs slope flattens hard as DCN_EP grows** — the inter-node `all-to-all` cost becomes the dominant per-step term, dwarfing the per-pdbs amortization of dense compute. Above a small threshold, increasing pdbs at high DCN_EP buys nothing.
3. **Capacity-factor sensitivity to DCN-EP grows steeply** (same shape as kimi-1T). dense-cf1.25 only loses 38% from EP=1→8; dense-cf4 loses 59%. cf=4's bigger activation tensor is much more punishing under inter-node `all-to-all`.
4. **`sparse-gmm-fixed` cliff sharpens steeply with DCN_EP**, similar to kimi-1T but at a smaller starting ceiling: max_pdbs collapses 7 → 2 → 1 → 1 as DCN_EP doubles. The dropless RCCL `ragged-all-to-all` over RDMA is the most DCN-EP-fragile collective in this set; dense's regular `all-to-all` is far more tolerant. At DCN_EP=8, sparse-gmm-fixed pdbs=1 still fits at 333.6 TGS (vs kimi-1T which goes infeasible at DCN_EP=8 even at pdbs=1), reflecting DS3's lower per-config memory pressure overall.
5. **`sparse-gmm-fixed` and dense configs at DCN_EP=4 have markedly long compile times**: sparse-gmm-fixed pdbs=1 took >45 min on the first attempt and was scancelled as a suspected hang; the `--time=90:00` retry compiled and ran cleanly to 338.4 TGS. Same pattern for `dense-cf1.25` at DCN_EP=4 / pdbs=4 (first attempt hit a `runaway-log` corruption, second attempt hung past 17 min, third attempt at `--time=90:00` succeeded at 649.3 TGS) and `dense-cf4` at DCN_EP=4 / pdbs=6 (50 min compile-hang, scancelled). For DS3 at DCN_EP=4 specifically, **default `--time=60:00` is sometimes insufficient** — bump to `--time=90:00` for any new probe at this DCN_EP.
6. **No DeepEP comparison row is possible** — the ["Inter-node EP (RDMA-backed AllToAll)" hypothesis](#deepep-today-competitive-kernel-expensive-integration--and-where-it-would-shine) (inter-node EP exercises DeepEP's RDMA-dispatch advantage over RCCL's all-to-all-over-RDMA) **remains untestable** against current MaxText. Lifting the `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` validator is the prerequisite. Until that change lands, `sparse-gmm-fixed` is the only dropless-config DCN-EP curve in this section.

---

## Key takeaways

1. **Peak throughput:**
   - **Dropping:** `dense-cf1.25 @ pdbs=16` → 1416 TGS, MFU 14.19 % (1-node/proc).
   - **Dropless:** `sparse-gmm-deepep-v3 @ pdbs=7` → **1097 TGS**, MFU **10.99 %** (1-node/proc, default `MEM_FRACTION=.93`). Beats the previous dropless peak `sparse-gmm-fixed @ pdbs=7` (989 TGS, needed `MEM_FRACTION=.96`) by +10.9 % TGS and without any memory-fraction bump.
2. **Launcher impact on dense is small** (≤ ±8 % at every pdbs, with no consistent winner) — in stark contrast to sparse-gmm (takeaway #3), where the launcher choice drives a 1.7–3.2× swing.
3. **Launcher impact on `sparse-gmm` is dramatic**: 1-GPU/proc is 1.7 × (pdbs=1) → 3.2 × (pdbs=6) faster than 1-node/proc. XLA lowers `ragged_all_to_all` to an in-process kernel when the EP axis is local (1-node), but to RCCL when the EP axis spans processes (1-GPU); RCCL is much faster for this collective.
4. **Best sparse path: `sparse-gmm-deepep-v3` on 1-node/proc** (post-2026-04-22; see takeaway #9).
   - **Beats every other sparse path at every pdbs**: 1030 vs 949 @ pdbs=6 (+8.5 % over `sparse-gmm-fixed`, the previous best), 1097 vs 989 @ pdbs=7 (+10.9 %).
   - **Runs at default `MEM_FRACTION=.93`** — pdbs=7 works out of the box, whereas `sparse-gmm-fixed @ pdbs=7` required `MEM_FRACTION=.96` (see footnote ᵃ) and 1-GPU/proc is permanently OOM at pdbs=7.
   - **Same deployment shape as `sparse-gmm-fixed`**: 1-node/proc launcher, same Docker image, same passthrough flags — only difference is the patched MaxText branch (`yihuang/moe-turbo-gmm-and-deepep-v3`, now the `container_env.sh` default since 2026-04-30). No new XLA flag, no launcher change. (At the time of this sweep the v3 branch had to be set explicitly via `MAXTEXT_PATCH_BRANCH=…`; new submissions inherit it as the default.)
   - **Strictly dominates the historical sparse-path rankings**: 3.5 × faster than stock `sparse-gmm 1-node` at pdbs=6 (1030 vs 298), 1.59 × faster than baseline `sparse-gmm-deepep` at pdbs=6 (1030 vs 647), and 1.08 × faster than `sparse-gmm-fixed` at pdbs=6 (1030 vs 949). Previous takeaways that `sparse-gmm-fixed` was the best dropless path and `sparse-gmm-deepep` had been "beaten by integration overhead" are both obsolete — a Python-only patch (two commits on top of the `sparse-gmm-deepep` baseline) closed most of that integration overhead.
5. **Higher capacity_factor is expensive**: `cf=1.25 → 2.0` cuts TGS by 22–32 %; `cf=1.25 → 4.0` cuts it by 50–60 %. `cf=4.0` OOMs at pdbs=16 on both launchers and was compile-flaky at pdbs=6 on 1-node (now resolved on retry).
6. **Dense (dropping) always beats sparse-gmm (dropless) on TGS at same pdbs on this hardware.** All three dense configs tested use dropping (cf=1.25/2.0/4.0), which fixes per-expert capacity and lets MaxText emit regular `all-to-all` + regular matmul instead of `ragged_all_to_all` + ragged matmul. That kernel simplification is the dominant source of the TGS gap at every pdbs — it's a consequence of dropping, not something sparse overcomes. This 15-step throughput sweep does not probe long-run loss convergence, which is where dropping's discarded tokens would normally show up as a model-quality cost; the ≤ 0.003 loss agreement within each pdbs is a numerical-correctness probe, not a quality comparison.
7. **Numerical correctness verified end-to-end** — all loss values agree to Δ ≤ 0.003 within each pdbs across launcher × config.
8. **The `sparse-gmm 1-node` pathology is fixable with an XLA flag, not just a launcher switch.** `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` (→ `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`) forces XLA's ragged-all-to-all thunk to use its `kNccl` lowering instead of the naive in-process one-shot kernel (`RaggedAllToAllKernelImpl<8l>`). The same 3.2 × speedup that 1-GPU/proc triggers automatically at runtime is now accessible on 1-node/proc at compile time, and `0` is the default so plain `sparse-gmm` submissions pick it up automatically. The change is **strictly additive** — verified empirically to be a no-op on configs that don't emit `ragged-all-to-all` HLO ops: `sparse-gmm-deepep` (HLO has 0 ragged ops; DeepEP's `moe_dispatch`/`moe_combine` custom calls absorbed them) shows a 647 → 642 TGS delta, and `dense-cf1.25` (never emits ragged_all_to_all) shows 1040 → 1026 TGS — both within the ≤ 1.3 % launcher-noise range.
9. **`sparse-gmm-deepep` has significant per-layer Python headroom — v2 and v3 realise +24 % and +59 % over baseline at pdbs=6 respectively, with no library changes.** The v3 dispatch-side backward no longer goes through a duplicate-index scatter-add (which on MI355 compiles to the main-stream-blocking `input_scatter_fusion_*.kd` kernel dominating baseline DeepEP's step time); a `jax.custom_vjp` that inverts the sort permutation and folds the top-K duplicates with a reduce-sum eliminates that kernel entirely. v2 is the intermediate step (composes the two dispatch-side gathers into one; halves the scatter-add count but doesn't remove atomics). Forward output is bit-identical to baseline at every step — losses match within bf16 LSB noise at every pdbs. Net effect: `sparse-gmm-deepep-v3` overtakes `sparse-gmm-fixed` as the best dropless path at every pdbs ∈ 1…7, and crucially extends the pdbs=7 frontier at default `MEM_FRACTION=.93` where `sparse-gmm-fixed` needed `.96` to fit. See the [DeepEP today: competitive kernel, expensive integration — and where it would shine](#deepep-today-competitive-kernel-expensive-integration--and-where-it-would-shine) section for the kernel-level accounting; v3 closes most of the "8.97 s `input_scatter_fusion_*.kd` main-stream overhead" identified as direction 1 in that section, from the Python side.

---

## Infrastructure / memory-ceiling notes

- **`sparse-gmm 1-node pdbs=7` requires `XLA_PYTHON_CLIENT_MEM_FRACTION=.96`** (default `.93` → silent RCCL-init hang). XLA's working set is 274.6 GiB; default pool is only 267.8 GiB. Submit with `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96`.
- **`sparse-gmm-fixed 1-node pdbs=7` likewise requires `MEM_FRACTION=.96`** (default `.93` → RESOURCE_EXHAUSTED when allocating 217.3 GiB). The `kNccl` path uses less memory than the one-shot kernel (275 → 217 GiB for the dominant allocation) but still exceeds the default pool. Once at `.96` (285 GiB pool), the job runs cleanly at 989 TGS. Per the sweep, this is the best dropless configuration observed — **989 TGS vs 942 TGS for `sparse-gmm 1-GPU pdbs=6`**, plus the larger pdbs.
- **`sparse-gmm-fixed 1-node pdbs=8, 16` remain infeasible** (OOM 242 GiB at pdbs=8, 367 GiB at pdbs=16, both at `MEM_FRACTION=.96`). The flag changes the collective lowering but does not shrink the MoE working set enough to clear the pdbs=7 feasibility ceiling on 288 GB HBM. To push beyond pdbs=7 on this model / hardware, a different axis is needed (FP8, DCN-EP, larger hardware).
- **`sparse-gmm 1-GPU pdbs=7` is hardware-infeasible on 288 GB MI355.** XLA needs 274.6 GiB; the remaining HBM (~13 GiB) is not enough for 1-GPU/proc's per-process RCCL peer-access buffers. Verified at `MEM_FRACTION ∈ {.96, .97, .98, .99}` — all fail with `Cuda failure 'out of memory'` in RCCL's `alloc.h:376`.
- **1-GPU/proc costs ~5–15 GiB/GPU extra HBM vs 1-node/proc** (runtime duplication per process, per-process RCCL channel buffers, IPC peer-access buffers via `hipIpcOpenMemHandle`, and higher allocator fragmentation across 8 independent allocators). At pdbs=7 we're on the wrong side of the knife edge for 1-GPU/proc.
- **`dense-cf4 1-node pdbs=6`**: the original job (12690) hit a compile-time OOM (XLA rematerialization could not reduce below 278 GiB). A plain rerun at default `MEM_FRACTION=.93` (job 12886) succeeded at 543 TGS — confirming the first run was a transient XLA-scheduling flake rather than a real ceiling (XLA's `hlo_rematerialization` is a heuristic and is not strictly monotonic in pdbs; the DAG produced for pdbs=6 happened to defeat it the first time while pdbs=7 and 8 compile into schedules that fit). The retry result is entered in all three tables above.
- **Pushing `MEM_FRACTION` too high starves RCCL**: a separate retry of the same cell at `MEM_FRACTION=.96` (job 12885) actually failed — at 96%, only ~11 GiB/GPU is left outside XLA's pool, which isn't enough for RCCL scratch. Two ranks got OOM-killed during collective init. Default `.93` leaves ~20 GiB/GPU free, which is the right amount for this model. Lesson: raise `MEM_FRACTION` only when the pre-OOM log shows XLA hitting an allocation limit (not when training dies silently).
- **RCCL-init hangs on 1-node/proc are flaky** — `dense-cf1.25 1-node pdbs=2` and `pdbs=6` each hung at RCCL init on fresh submits and required 1–2 retries. No deterministic root cause; retries resolved them.
- **External cancellations happened twice during the sweep** (cluster admin activity). All ~10 affected jobs were resubmitted and ran to completion or reached their natural OOM.

---

## Footnotes

- **ᵃ** `sparse-gmm 1-node pdbs=7` tested only with `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96`; default `.93` hangs at RCCL init.  `sparse-gmm-fixed 1-node pdbs=7` likewise needs `MEM_FRACTION=.96` (default `.93` OOMs at 217 GiB allocation; the kNccl path uses less memory than the one-shot kernel but still exceeds the default pool).
- **ᵇ** `sparse-gmm 1-GPU pdbs=7` is infeasible at every `MEM_FRACTION` tested (`.93 → .99`); RCCL OOM.
- **ᶜ** `sparse-gmm-fixed 1-node pdbs=8` OOMs with 242 GiB allocation and pdbs=16 OOMs with 367 GiB allocation, both at `MEM_FRACTION=.96`. The kNccl path's memory savings over the one-shot kernel aren't enough to push past the pdbs=7 feasibility ceiling on 288 GB HBM.

---

## Why do the three sparse variants differ in TGS? (pdbs=6 profile drill-down)

Profiling jobs: **12895** (`sparse-gmm 1-node`), **12916** (`sparse-gmm 1-GPU`), **12897** (`sparse-gmm-deepep 1-node`).  Each was rerun with `profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 _env_ENABLE_XLA_DUMP=1`.  Per-kernel times below are averaged across all 64 GPUs × 3 profiled steps (divisor = 192) from the xplane trace JSONs, using the same [`utils/profile_drill.py`](utils/profile_drill.py) script as the v1 / v2 / v3 drill-down (methodology documented in [`skills/profile-drill/SKILL.md`](skills/profile-drill/SKILL.md)).  Per-GPU variance is ≤ 3 % for every kernel (verified on `RaggedAllToAllKernelImpl`: min 27.5 s, p50 28.4 s, max 29.3 s / step).

### Step-time composition (per GPU per step, seconds, `pdbs=6`)

| Slice                                             | 1-node sparse-gmm (step 82.5 s) | 1-GPU sparse-gmm (step 26.1 s) | 1-node sparse-gmm-deepep (step 38.0 s) |
|---------------------------------------------------|--------------------------------:|-------------------------------:|---------------------------------------:|
| `RaggedAllToAllKernelImpl` (XLA in-process)       |                       **28.36** |                           0.00 |                                   0.00 |
| `primus_turbo::deep_ep::*` (DeepEP native HIP)    |                            0.00 |                           0.00 |                                   1.62 |
| `input_scatter_fusion_*.kd`                       |                            0.01 |                           0.02 |                               **8.97** |
| `loop_select_fusion_*.kd` (valid-rows mask)       |                            0.00 |                           0.01 |                                   1.52 |
| `loop_gather_fusion_*.kd`                         |                            1.79 |                           3.47 |                                   0.00 |
| RCCL (`ncclDevKernel_*`)                          |                            7.50 |                      **15.33** |                                   6.90 |
| CK / Primus-Turbo grouped GEMM + dense GEMM       |                            4.14 |                           8.21 |                                   4.80 |
| Flash-attention (`aiter::fmha_*`)                 |                            0.87 |                           1.83 |                                   1.22 |
| Other fusions (convert / reduce / transpose / …)  |                            2.45 |                           4.64 |                                   1.90 |
| **Total kernel time (on any stream)**             |                       **45.12** |                      **33.51** |                              **26.93** |
| Step − total kernel = idle gap (+) or overlap (−) |                           +37.4 |                           −7.4 |                                  +11.1 |

Numbers use `dur` from the raw trace events summed over `.kd` kernels, RCCL `ncclDevKernel_*`, and the `stream_executor::gpu::*` / `primus_turbo::*` / `ck_tile::*` / `aiter::*` / `Cijk_*` families, divided by auto-detected `64 GPUs × 3 profiled steps`.  The last row is **step − total** (not a kernel bucket):

- **+37.4 s on 1-node sparse-gmm** — `RaggedAllToAllKernelImpl` is launched on the main compute stream and reports its ~155 ms per-call `dur`, but during that time the SMs are mostly waiting on sequential xGMI peer transfers; downstream kernels cannot start, so the stream accumulates real wall-clock that isn't attributed to any kernel event.  This is the "blocking kernel stalls everything" signature.
- **−7.4 s on 1-GPU sparse-gmm** — total kernel time *exceeds* step time, i.e. kernels on the compute stream and the RCCL comm stream genuinely overlap.  The 15.33 s RCCL bucket is largely hidden behind the ~14 s of compute work.  Healthy.
- **+11.1 s on 1-node sparse-gmm-deepep** — smaller version of the 1-node pathology: `input_scatter_fusion_2.kd` (≈ 4.4 s) is also main-stream-blocking, but the stall budget it creates is much smaller than the 1-node one-shot kernel's.

Note the counterintuitive column deltas for GEMM and flash-attention (both ~2× higher in 1-GPU than 1-node).  The raw per-call GEMM latency is identical in both launchers — the 2× difference is an observation artefact: in 1-node/proc, multiple GEMMs from the 8 intra-process GPUs interleave via xGMI-aware scheduling, which XLA's profiler partially attributes to non-kernel overhead (folding into the +37.4 s idle gap above); in 1-GPU/proc each process observes its own GEMMs as fully-accounted kernel events.  The 1-GPU column is the more faithful per-GPU reading.

### HLO collective-op inventory (identical across both sparse-gmm variants)

| Op                  | sparse-gmm (both) | sparse-gmm-deepep |
|---------------------|-------------------|-------------------|
| `all-gather`        | 18                | 14                |
| `all-reduce`        | 12                | 12                |
| `ragged-all-to-all` | **6**             | 0                 |
| `all-to-all`        | 4                 | 0                 |
| `reduce-scatter`    | 4                 | 4                 |

The HLO emitted for `sparse-gmm 1-node` and `sparse-gmm 1-GPU` is bit-identical (same XLA lowering, same 6 `ragged-all-to-all` ops).  Only the **runtime lowering** of those collectives differs, because the `ici_expert_parallelism=8` axis is intra-process in the 1-node launcher and inter-process in the 1-GPU launcher.

The zero counts for `sparse-gmm-deepep` under `ragged-all-to-all` and `all-to-all` do **not** mean DeepEP has no dispatch communication.  `use_deepep_dispatch=true` replaces those HLO collectives with three XLA `custom_call` targets — counting instances directly in the HLO dump from job 12897:

| `custom_call_target`    | HLO instances |
|-------------------------|---------------|
| `moe_dispatch`          | 2             |
| `moe_combine`           | 2             |
| `moe_cached_dispatch`   | 1             |

These 5 custom-call instances collectively replace the 6 `ragged-all-to-all` + 4 `all-to-all` = 10 HLO-collective instances used by stock sparse-gmm for the same dispatch/combine work.  They're outside the collective-op inventory shown above because that table only counts standard HLO collectives, not custom-call ops.  At runtime the custom-call work shows up in two places in the per-kernel profile: **8.97 s/step/GPU as `input_scatter_fusion_*.kd`** (the XLA fusion that implements the custom call's surrounding token-permutation logic — the new dominant XLA-bucket kernel, see below) and **1.62 s/step/GPU as native `primus_turbo::deep_ep::intranode::*` HIP kernels** (`dispatch`, `combine`, `cached_notify_combine`, `cached_notify_dispatch`, `get_dispatch_layout`, `notify_dispatch`).  The `all-gather` drop (18 → 14) is similarly from DeepEP's combine custom call absorbing a few token-permutation all-gathers that the stock path expresses separately.

### The 28.4 s/step smoking gun: `RaggedAllToAllKernelImpl`

When the EP axis is **intra-process** (1-node/proc has 8 local devices in one JAX process), XLA lowers `ragged-all-to-all` to the in-process kernel `stream_executor::gpu::RaggedAllToAllKernelImpl<8l>`.  This is a naive "each device loops over every peer, copies the ragged segment via peer memory access" implementation — sequential across peers, not pipelined, not chunked.  At `pdbs=6` it runs **~155 ms per call × 183 calls per step = 28.4 s/step/GPU** — by far the single biggest kernel in the step (63 % of the 45.1 s total kernel time, 34 % of the 82.5 s wallclock), and it cannot overlap with compute because it lives on the main compute stream.  On top of the direct 28.4 s, the kernel's serialised peer transfers stall downstream kernels, opening the +37.4 s idle-gap row in the composition table above — i.e. for every second spent inside `RaggedAllToAllKernelImpl`, another ~1.3 s of wall-clock elapses while nothing else on the main stream can run.

When the EP axis is **inter-process** (1-GPU/proc has 1 local device per JAX process), XLA cannot use the in-process kernel and falls back to RCCL `AllToAll`.  That kernel vanishes (28.4 → 0 s), its work shows up as ~7.8 extra seconds on the RCCL bucket (7.50 → 15.33 s), and the RCCL work now runs on the comm stream overlapping with compute — flipping the step − kernel row from **+37.4 s idle** to **−7.4 s overlap**.

The net 3.2× TGS win (298 → 942 TGS at pdbs=6) — step 82.5 → 26.1 s, saving **56.4 s** — decomposes into:

| Component                                                             | Δ on the step |
|-----------------------------------------------------------------------|--------------:|
| `RaggedAllToAllKernelImpl` removed from main stream                   |       −28.4 s |
| Idle gap collapses once main-stream blocker is gone                   |       −44.8 s (from +37.4 to −7.4) |
| RCCL kernel grows (it now owns the ragged traffic, but on comm stream) |       +7.8 s |
| Other kernel deltas (GEMM +4.1, FA +1.0, loop_gather +1.7, other +2.2) |       +9.0 s |
| — (these last two cancel partially against the scheduler gain)         |               |
| **Net**                                                               |    **−56.4 s** |

**≈ 50 % is raw kernel removal (28.4 s), ≈ 50 % is scheduler cascade recovery** — the same "lose the main-stream blocker, let the rest overlap" mechanism that v2 → v3 of `sparse-gmm-deepep` later exhibits with a different kernel.  The kernel-quality piece alone is ~60× per-call: RCCL's AllToAll-over-IPC runs at **~2.6 ms per call** vs the one-shot kernel's **~155 ms per call**, purely because it's hand-tuned, chunked, and pipelined across peers rather than sequential.

**This is exactly the fix delivered by [#112](https://github.com/AMD-AGI/maxtext-slurm/pull/112).**  Since the speedup is purely a runtime-kernel swap and not a launcher property, the in-process one-shot kernel can also be disabled directly via the XLA flag `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` — the ragged thunk then picks `kNccl` at thunk-select time even when all EP ranks are intra-process.  The `sparse-gmm-fixed (1-node)` column in the results matrix measures this path: same runtime behavior as 1-GPU/proc's fallback, on the 1-node/proc launcher, accessed via env var `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` (now the default in `train_env.sh`; see takeaway #8).  The HLO stays bit-identical to stock `sparse-gmm` — only the thunk's implementation-type selection changes.

### Why RCCL is ~60× faster per call

The per-call speedup (155 ms → ~2.6 ms, 60×) stacks four hardware-utilization gaps:

| Aspect                | `RaggedAllToAllKernelImpl`                 | RCCL `AllToAll`                                                    |
|-----------------------|--------------------------------------------|--------------------------------------------------------------------|
| Peer concurrency      | Serial — one (src, dst) copy at a time     | All 7 peers' sends + receives in flight simultaneously             |
| Link utilization      | Single xGMI link active per iteration      | Multiple xGMI links driven in parallel via multiple channels       |
| Chunking / pipelining | None — whole peer-buffer per copy          | Per-peer buffer split into chunks that pipeline through the links  |
| SM utilization        | Coordination-limited (few thread-blocks)   | Many SMs driving DMA + copy + sync concurrently                    |

The naive kernel is **bandwidth-bounded by a single xGMI link at a time**, with per-peer launch and barrier overhead between iterations — the measured 155 ms/call is consistent with that serial path for this model's ragged-all-to-all shapes.  RCCL's grouped P2P path parallelises the same traffic across links and channels simultaneously, with per-peer transfers overlapping instead of serialising, approaching the intra-node xGMI aggregate bandwidth the MI355 topology affords — measured ~2.6 ms/call.

Ragged semantics are preserved by the op, not by padding: RCCL's `RaggedAllToAllThunk` issues a grouped `ncclSend`/`ncclRecv` pair per peer using the runtime-computed actual send/recv sizes (read from the HLO op's ragged-offset operands).  No bytes wasted on padding, no tokens dropped.

### DeepEP today: competitive kernel, expensive integration — and where it would shine

**The kernel itself is fine.**  Per-step per-GPU, DeepEP's native HIP kernels (`primus_turbo::deep_ep::intranode::*` — dispatch/combine/layout/notify) run at 1.62 s — same order as the ~0.9 s RCCL's grouped `ncclSend`/`ncclRecv` needs for the equivalent transport in `sparse-gmm 1-GPU`.  DeepEP's design intent (MoE-aware, fused, xGMI IPC, minimal launches) is delivering at the kernel level; the gap in raw transport time is small and dominated by stream-placement difference rather than kernel efficiency.

**What decides the contest is XLA's integration, not the kernel.**  `moe_dispatch` / `moe_combine` output tokens in a layout that doesn't match `grouped_gemm`'s expected per-expert grouping, so XLA emits `input_scatter_fusion_*.kd` to bridge them — 8.97 s/step of main-stream token permutation that the sparse-gmm RCCL path doesn't pay because its output shape feeds directly into GMM.  The total kernel time (any stream) tells the story directly:

| Family                                    | `sparse-gmm 1-GPU` | `sparse-gmm-deepep 1-node` |
|-------------------------------------------|-------------------:|---------------------------:|
| DeepEP HIP kernels (dispatch/combine/…)   |                  0 |                       1.62 |
| `input_scatter_fusion_*.kd`               |               0.02 |                   **8.97** |
| `loop_select_fusion_*.kd`                 |               0.01 |                       1.52 |
| `loop_gather_fusion_*.kd`                 |               3.47 |                       0.00 |
| RCCL (`ncclDevKernel_*`)                  |              15.33 |                       6.90 |
| CK / Primus-Turbo GEMM                    |               8.21 |                       4.80 |
| Flash-attention (`aiter::fmha`)           |               1.83 |                       1.22 |
| Other fusions + misc                      |               4.64 |                       1.90 |
| **Total kernel time (any stream)**        |          **33.51** |                  **26.93** |
| **Step time**                             |          **26.10** |                  **38.00** |
| Step − total kernel                       |             −7.4 s |                     +11.1 s |

DeepEP 1-node has an **11.1 s idle gap** on a 38.0 s step — about 29 % of wallclock is main-stream idle despite DeepEP being designed as a faster path.  That idle is forced by the main-stream-blocking `input_scatter_fusion_2.kd` (≈ 4.4 s, not atomics-free until v3).  `sparse-gmm 1-GPU` has no such blocker; its compute and RCCL streams overlap (step 26.1 s, negative idle gap −7.4 s).  That XLA-emitted fusion family, not DeepEP's own HIP kernels, is why 647 TGS (sparse-gmm-deepep 1-node) sits below 942 TGS (sparse-gmm 1-GPU) despite DeepEP's kernel being more specialised.  DeepEP still beats stock sparse-gmm 1-node (298 → 647) because `RaggedAllToAllKernelImpl` on that path (28.4 s kernel + 37.4 s idle gap = 65.8 s of main-stream wastage) is dramatically worse than DeepEP's 8.97 s scatter + 1.62 s native kernels + 11.1 s gap ≈ 21.7 s.  (Cross-launcher comparisons use step time and idle gap only — per-kernel totals on 1-node/proc are attribution-depressed, see the note above about GEMM / flash-attn on 1-node vs 1-GPU.)

**What would close the gap** — both are Primus-Turbo kernel-engineering projects, not JAX/XLA config changes (**partially addressed** from the Python side by `sparse-gmm-deepep-v3`, which eliminates the `input_scatter_fusion` kernels entirely via a `custom_vjp` — see the v1/v2/v3 drill-down below):

1. **Emit dispatch output in the GMM-compatible layout directly.**  Fuse the per-expert permutation into DeepEP's dispatch kernel so `input_scatter_fusion_*.kd` is no longer emitted.  v3 already approximates this from the Python side (Δ +59 % TGS vs v1); a kernel-level fix would let v1's Python code emit the same HLO as v3 and generalise to other MoE frontends.  **This is the real lever.**
2. **Wrap DeepEP custom calls as async-start / async-done.**  Gives XLA's latency-hiding scheduler a window to hide DeepEP's 1.62 s behind compute.  Blocked today by an upstream constraint: JAX emits StableHLO, whose legalisation pass rejects `mhlo.async_start` (verified empirically when we tried it — XLA's compiler rewrites collectives into async form internally but there's no stable-front-end path for user code to emit async custom calls).  Needs an XLA / StableHLO-level change, not a Primus-Turbo change.  Upside bounded at ~5 % TGS even if landed; not a substitute for direction 1.

**Where DeepEP's design already wins** (regimes this sweep does not exercise):

- **Inter-node EP (RDMA-backed AllToAll).**  RCCL's `AllToAll` over RDMA adds round-trip setup and ring/tree overhead that DeepEP's direct RDMA dispatch avoids.  Our sweep runs `ici_expert_parallelism=8` (purely intranode); a hypothetical `dcn_expert_parallelism>1` configuration would exercise this.
- **FP8 dispatch.**  DeepEP's kernels natively support FP8 input; the stock `ragged_all_to_all` path does not efficiently.  Halving wire bytes directly halves the transport time of the bucket DeepEP targets.
- **H800 / NVLink stacks where DeepEP was developed.**  NCCL on that hardware has different overhead characteristics than RCCL on MI355; DeepEP's kernel-fusion gains compound more there.

None of these are reachable through a `maxtext-slurm` config change alone — each requires different topology, precision, or hardware.

**The "unblock DeepEP 1-GPU/proc" shortcut is not worth pursuing alone.**  Primus-Turbo hardcodes `num_ranks = jax.local_device_count()`, yielding `AssertionError: Unsupported number of EP ranks: 1` under 1-GPU/proc.  Fixing that binding by itself wouldn't help: (a) the dispatch transport on `sparse-gmm 1-GPU` costs ~0.9 s/step/GPU and is already overlapped with compute — driving it to zero saves <1 % of step time; (b) the ~7.8 s/step of truly exposed comm is dominated by the 34 *non-dispatch* collectives per step (18 `all-gather` + 12 `all-reduce` + 4 `reduce-scatter`), which DeepEP replaces none of; (c) without direction 1 above landed first, a 1-GPU DeepEP port would inherit the same 8.97 s `input_scatter_fusion_*.kd` on main stream and likely be **slower** than plain `sparse-gmm 1-GPU`, not faster.  Prioritise direction 1; the launcher question becomes moot once DeepEP is competitive on 1-node.

### Takeaway

The entire 1-node → 1-GPU speedup on sparse MoE reduces to **getting off `RaggedAllToAllKernelImpl`**.  Three known ways:

1. **Use 1-GPU-per-process** so the EP axis is inter-process and falls back to RCCL `AllToAll` (overlapped, 3.2× faster).
2. **Use `use_deepep_dispatch=true`** on 1-node so the collective is replaced by Primus-Turbo's dedicated intranode kernels (2.2× faster than the stock path, but still inferior to option 1 for this model on MI355).
3. **Set `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` on 1-node** (new, added post-initial-sweep; now the default in `train_env.sh`). This passes `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` to XLA, disabling the in-process one-shot kernel and forcing the ragged thunk's `kNccl` code path — the same runtime lowering 1-GPU/proc triggers automatically. Beats option 1's TGS at every pdbs while keeping 1-node/proc's HBM headroom. **This is the new best path.**

Ranking by peak TGS @ pdbs=6 (MI355, EP=8, BF16):

| Option | 1-node TGS | 1-GPU TGS | Best for |
|---|---|---|---|
| Stock `sparse-gmm` (one-shot kernel) | 298 | 942 | — (superseded) |
| Option 2: `sparse-gmm-deepep` (baseline) | 647 | ✗ (EP ranks=1) | — (superseded by v3) |
| Option 1: `sparse-gmm 1-GPU` | — | 942 | pdbs ≤ 6 only; costs 5-15 GiB/GPU HBM |
| Option 3: `sparse-gmm-fixed` | 949 | — | pdbs=1…7 (pdbs=7 needs `MEM_FRACTION=.96`) |
| **Option 4: `sparse-gmm-deepep-v3`** | **1030** | — | **pdbs=1…7** (pdbs=7 at default `MEM_FRACTION=.93`) |

Option 4 now supersedes options 1, 2 and 3 on this workload: beats option 3 by +8.5 % TGS at pdbs=6 and +10.9 % at pdbs=7 (1097 vs 989), keeps the default HBM budget, and extends the peak-dropless row without any memory-fraction tuning. It shares the same deployment shape as option 3 (1-node/proc launcher, no extra Docker image, no extra env var — `yihuang/moe-turbo-gmm-and-deepep-v3` is the `container_env.sh` default since 2026-04-30).

For a detailed analysis of where DeepEP's cost goes on this workload and what Primus-Turbo-side changes would make it competitive, see the `DeepEP today` section above.

### Profiled job artifacts (kept under `outputs/`)

- `12895-…-TGS_292.811/` — 1-node sparse-gmm pdbs=6 profile
- `12916-…-TGS_907.556/` — 1-GPU sparse-gmm pdbs=6 profile
- `12897-…-TGS_631.150/` — 1-node sparse-gmm-deepep pdbs=6 profile

---

## Why do v1 / v2 / v3 of `sparse-gmm-deepep` differ in TGS? (pdbs=6 profile drill-down)

Profiling jobs — all three run with `profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 _env_ENABLE_XLA_DUMP=1` on the same 8 nodes:

- **12897** — v1 baseline `sparse-gmm-deepep`, branch `yihuang/moe-turbo-gmm-and-deepep` @ `ad693da2`
- **13412** — v2, branch `yihuang/moe-turbo-gmm-and-deepep-v2` @ `627168f8`
- **13382** — v3, branch `yihuang/moe-turbo-gmm-and-deepep-v3` @ `f59be3c9`

All three share image, launcher, passthrough flags, and HLO collective inventory — only `src/MaxText/layers/moe.py` differs between the branches.  Per-kernel times below are averaged across all 64 GPUs × 3 profiled steps (divisor = 192) from the trace JSONs.  Headline TGS / step-time numbers in the column headers come from the no-profiler runs **12897** (v1, TGS=647), **13292** (v2, TGS=806), and **13370** (v3, TGS=1030) that populate the main results matrix; the delta vs the profile-run step time is <1 s everywhere except v3 (profiler writeback slightly inflates its profiled steps).

### The code-level change

The three versions only differ in how DeepEP's received tokens are fanned out onto the sorted expert-dispatch layout.  All three produce the same `x` tensor fed into grouped-GEMM — the difference is in how many gather/scatter kernels XLA compiles the forward-backward chain into.

**v1** (`ad693da2`) — two chained gathers, two duplicate-index scatter-adds on backward:

```python
expanded_x = recv_x[token_indices]                 # fan-out gather (K-way dupes)
x          = expanded_x[sort_idx]                  # sort permute gather
x          = jnp.where(_deepep_valid_rows, x, 0)
```

**v2** (`627168f8`) — compose the two gathers into a single gather; one duplicate-index scatter-add on backward:

```python
composed_idx = token_indices[sort_idx]             # compose permutation
x            = recv_x[composed_idx]                # single K-way-dupe gather
x            = jnp.where(_deepep_valid_rows, x, 0)
```

**v3** (`f59be3c9`) — same forward as v2, but a `jax.custom_vjp` replaces the duplicate-index scatter-add backward with `argsort(sort_idx) + reshape + reduce-sum(axis=K)` — no atomics:

```python
x = _deepep_dispatch_fan_out(recv_x, sort_idx, num_topk)  # same forward output
x = jnp.where(_deepep_valid_rows, x, 0)
# Inside _deepep_dispatch_fan_out_bwd:
#   grad_fanned   = grad_x[argsort(sort_idx)]      # permutation gather, no atomics
#   grad_recv_x   = grad_fanned.reshape(N, K, H).sum(axis=1)   # reduction, no atomics
```

Forward values are bit-identical across v1 / v2 / v3 (loss at every step agrees to bf16 LSB); only the *backward* HLO shape changes.

### Main-stream `input_scatter_fusion_*.kd` — the dominant kernel that evaporates

The per-kernel breakdown shows the entire TGS progression is driven by XLA's `input_scatter_fusion_*.kd` family shrinking as the duplicate-index structure disappears from the backward HLO:

| variant | heavy scatter-add kernels | `input_scatter_fusion_*.kd` s / step / GPU |
|---|---|---|
| v1 baseline (2 gathers, 2 scatter-adds) | `_2.kd` **4.39 s** + `_3.kd` **4.54 s** | 4.39 + 4.54 + 0.04 = **8.97** |
| v2 (compose: 1 gather, 1 scatter-add)   | `_2.kd` **4.41 s** only              | 4.41 + 0.04 = **4.45** |
| v3 (custom_vjp: reduce-sum backward)    | *(none — all variants < 25 ms)*      | **0.04** |

Each heavy `input_scatter_fusion_*.kd` in v1/v2 is a duplicate-index atomic scatter-add on bf16[N*K, H] ≈ [1.57 M, 7168].  On MI355 those atomics stream through HBM one peer-word at a time and cannot overlap with the grouped-GEMM behind them — exactly the "main-stream-busy" bucket the earlier drill-down attributes the 1-node DeepEP penalty to.  v3's backward produces the same `grad_recv_x` via a cheap permutation gather (no atomics) followed by a contiguous reduce-sum over the top-K axis (no atomics), so XLA emits a handful of tiny fusion variants that each weigh in at < 25 ms / step / GPU — two orders of magnitude below v1 or v2.

### Step-time composition (per GPU per step, seconds, `pdbs=6`)

| Slice                                       | v1 (step 38.0 s) | v2 (step 30.5 s) | v3 (step 23.9 s) |
|---------------------------------------------|-----------------:|-----------------:|-----------------:|
| `input_scatter_fusion_*.kd`                 |         **8.97** |         **4.45** |         **0.04** |
| `loop_select_fusion_*.kd` (valid-rows mask) |             1.52 |             0.93 |             0.83 |
| RCCL (`ncclDevKernel_*`)                    |             6.90 |             6.89 |             8.18 |
| CK / Primus-Turbo grouped + dense GEMM      |             6.41 |             6.41 |             6.71 |
| Flash-attention (`aiter::fmha_*`)           |             1.22 |             1.22 |             1.27 |
| Other fusions (convert / reduce / transpose / select)  |  1.95 |             1.80 |             2.10 |
| **Total kernel time (on any stream)**       |        **26.93** |        **21.69** |        **19.13** |
| **Step time (TGS-derived steady state)**    |        **38.00** |        **30.50** |        **23.90** |
| Step − total kernel = scheduler gaps + overlap gap |      11.07 |             8.81 |             4.77 |

All rows computed from the same script ([`utils/profile_drill.py`](utils/profile_drill.py), methodology: [`skills/profile-drill/SKILL.md`](skills/profile-drill/SKILL.md)) over all 8 host trace JSONs × 8 GPUs × 3 profiled steps = 192 gpu-step samples.  The "Total kernel time" row sums across the main compute stream, the RCCL comm stream, and any auxiliary streams — so `step − total kernel` lower-bounds the pure idle time; the real "scheduler cannot overlap" gap is smaller because RCCL and compute share an execution timeline.

### Why both v1→v2 and v2→v3 save *more* wallclock than kernel time

Kernel-time accounting is super-linear for both transitions, but dramatically more so for v2 → v3:

| transition | Δ `input_scatter_fusion` | Δ total kernel time | Δ step time | step / kernel ratio |
|---|---:|---:|---:|---:|
| v1 → v2 | −4.52 s | −5.24 s | −7.50 s | **143 %** |
| v2 → v3 | −4.41 s | −2.56 s | −6.60 s | **258 %** |

**v1 → v2 (143 %).**  Removing one of the two heavy `input_scatter_fusion` kernels shaves 4.5 s directly off the main stream; the `loop_select_fusion` mask also shrinks (1.52 → 0.93 s) because one of the two gather/mask chains collapses.  The scheduler then recovers another ~2 s on top by overlapping more of the dispatch/combine RCCL traffic with the remaining backward compute — but only partially, because v1's *other* main-stream-blocking `input_scatter_fusion` is still in place after the first one goes.

**v2 → v3 (258 %).**  The last heavy `input_scatter_fusion` disappears.  Step time drops 6.6 s while net kernel time only drops 2.6 s — *less than half* of the wallclock saving is raw kernel removal.  The rest comes from scheduler cascade: with no main-stream atomic scatter-add left, XLA's latency-hiding scheduler is free to reorder the grouped-GEMM backward and push the RCCL dispatch/combine kernels into overlap slots that were previously blocked.  Evidence: RCCL *kernel* time actually *increases* from v2's 6.89 s to v3's 8.18 s (+1.29 s), i.e. XLA now dispatches more comm work overall — yet its *exposed* (non-overlapped) share of wallclock shrinks enough to erase that plus another ~5 s.

Conceptually this is the same mechanism the earlier `sparse-gmm 1-node → 1-GPU` drill-down attributed its 3.2 × win to (lose the main-stream-blocking kernel, let comm overlap).  Same phenomenon, different kernel: there the blocker was `RaggedAllToAllKernelImpl` (53 s/step/GPU, pure XLA-runtime kernel); here it's `input_scatter_fusion_2.kd` (4.4 s/step/GPU, XLA-emitted duplicate-index scatter-add).  The `input_scatter_fusion` removal is *exactly* direction 1 that the "DeepEP today" subsection predicted would close the DeepEP vs `sparse-gmm 1-GPU` gap — v3 closes it from the Python side via a 22-line `custom_vjp`, rather than via the Primus-Turbo kernel change that subsection originally called out.

### Takeaway

All three variants emit the same HLO collective inventory and feed the same forward tensor to the grouped-GEMM; what changes is how many atomic-heavy main-stream scatter-add kernels XLA's autodiff places between them.  v3 reduces that count from two (v1) / one (v2) / zero (v3).  Given the forward is bit-identical and the deployment shape is identical (same image, same flags, different patch branch), v3 is the unambiguous replacement for baseline `sparse-gmm-deepep` on this hardware.

| variant | patch branch | pdbs=6 TGS | pdbs=6 step time | dropless peak pdbs | dropless peak TGS |
|---|---|---|---|---|---|
| v1 (baseline) | `yihuang/moe-turbo-gmm-and-deepep` @ `ad693da2` | 647 | 38.0 s | 7 | 673 |
| v2 | `yihuang/moe-turbo-gmm-and-deepep-v2` @ `627168f8` | 806 (+25 %) | 30.5 s | 7 | 836 (+24 %) |
| **v3** | `yihuang/moe-turbo-gmm-and-deepep-v3` @ `f59be3c9` | **1030 (+59 %)** | **23.9 s** | **7** | **1097 (+63 %)** |

### Profiled job artifacts (kept under `outputs/`)

- `12897-…-TGS_631.150/` — v1 (baseline sparse-gmm-deepep) pdbs=6 profile
- `13412-…-dataset_type_synthetic-profiler_xplane…/` — v2 (`sparse-gmm-deepep-v2`) pdbs=6 profile
- `13382-…-dataset_type_synthetic-profiler_xplane…/` — v3 (`sparse-gmm-deepep-v3`) pdbs=6 profile

---

## How to reproduce

```bash
cd /maxtext-slurm
export DOCKER_IMAGE=/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar
# This sweep was run before container_env.sh defaulted to v3 (2026-04-30 change).
# To reproduce v1 baseline numbers, the patch-branch override below is now required:
export MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep

# Example: dense-cf1.25, 1-GPU/proc, pdbs=7
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=7 _env_ONE_GPU_PER_PROCESS=true

# Example: sparse-gmm-deepep (v1), 1-node/proc, pdbs=6
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true

# Example: sparse-gmm, 1-node/proc, pdbs=6 (post-default-change; reproduces the
# `sparse-gmm-fixed (1-node)` column since train_env.sh's default disables the
# one-shot kernel). Equivalent explicit form: add _env_ENABLE_RAGGED_ONESHOT_KERNEL=0.
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true

# Example: sparse-gmm, 1-node/proc, pdbs=7 — peak dropless (989 TGS).  Needs
# higher MEM_FRACTION because the kNccl lowering's working set at pdbs=7 exceeds
# the default .93 pool (~217 GiB allocation).
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96

# Example: reproduce the historical `sparse-gmm (1-node)` column (pre-flag, with
# XLA's in-process one-shot kernel active — e.g., 298 TGS @ pdbs=6, 302 TGS @ pdbs=7).
# Requires the explicit override since default is now kernel-disabled.
./submit.sh deepseek3-671b --partition=k8s --nodes=8 -- \
    per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true \
    _env_ENABLE_RAGGED_ONESHOT_KERNEL=1
```
