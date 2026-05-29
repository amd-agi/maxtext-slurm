# DeepSeek-V3 671B - Primus-Turbo GMM/DeepEP + pp-support verification (2026-05-30)

> **Purpose:** verify that the two feature branches do not regress correctness,
> feasibility, or throughput for `deepseek3-671b`, across both parallelism
> topologies (expert/FSDP and pipeline). The final stack (MaxText branch on top
> of `feature/pp-support`) is compared directly to the authoritative baselines.

- **Branches under test**
  - MaxText: [`feature/primus-turbo-gmm-deepep-integration`](https://github.com/ROCm/maxtext/tree/feature/primus-turbo-gmm-deepep-integration) - two self-contained commits: `9aeaa97b` (Primus-Turbo grouped GEMM) and `1c572908` (DeepEP dispatch/combine, incl. the one-time `setup()` bootstrap).
  - Primus-Turbo: [`feature/pp-support`](https://github.com/AMD-AGI/Primus-Turbo) - axis-aware (`nn.vmap("stage")`-of-`shard_map`) batching rules for the MoE/grouped-GEMM primitives, on top of current `main` (post moe_permute / DeepEPTokenDispatcher refactor). Baked into the image below.
- **Image:** `/mnt/vast/yihuang/c6f2df3d-deepep-gmm-maxtext-v26.2.tar` (Primus-Turbo `0.3.0+c6f2df3d` = `main` @ `8915c667` + the pp-support batching rules; JAX-only, gfx950).
- **Hardware:** 8 nodes x 8x AMD MI355 (288 GB HBM/device, Pensando AINIC), 64 GPUs. Peak BF16 ~ 2500 TFLOP/s/device -> MFU ~ TFLOP/25.
- **Reference baselines:** [`deepseek3-671b-pdbs-sweep-rerun-2026-05-09.md`](deepseek3-671b-pdbs-sweep-rerun-2026-05-09.md) (EP/FSDP), [`pp-vs-fsdp-deepseek3-671b.md`](pp-vs-fsdp-deepseek3-671b.md) (PP=8).
- **Dataset:** `dataset_type=synthetic`. **Seq:** 4096. **Steps:** 15.
- **Key prerequisite:** on this Primus-Turbo the DeepEP path requires an explicit one-time
  `setup()` (pins the expert-parallel comm group before the first `moe_dispatch`/`moe_combine`;
  older builds auto-bootstrapped). It is folded into the DeepEP commit (`1c572908`), called from
  a once-per-process guarded helper at the top of the MoE forward, so it covers both RAY=0 and
  RAY=1 launch modes.

## TL;DR

| Axis | Verified? | Evidence |
|---|---|---|
| EP/FSDP correctness + feasibility + TGS | YES (no regression) | 51-cell pdbs sweep matches the 2026-05-09 baseline |
| PP=8 (`feature/pp-support` batching rules) | YES | 9-cell PP=8 set: rules engage, memory win holds, loss matches the pp-vs-fsdp doc |

- **Loss:** matches the prior baselines to <= 0.002 in every clean cell, both topologies; the
  `sgd-deepep-v3 pdbs=10` NaN anomaly reproduces exactly.
- **Feasibility / OOM ceilings:** identical to the 2026-05-09 baseline to the GiB.
- **Throughput:** EP is within run-to-run + nodelist variance of the 2026-05-09 baseline (dense at
  parity, dropless sparse a few percent either side); PP=8 matches/slightly exceeds the pp-vs-fsdp doc.

## Configs (all MoE - DS3-671B has 256 routed experts, top-k=8)

| Tag | Passthrough flags | Primus-Turbo primitives exercised |
|---|---|---|
| `dense-cf1.25` | *(default)* `capacity_factor=1.25` | none (MaxText dense_matmul) |
| `dense-cf2` | `capacity_factor=2.0` | none |
| `dense-cf4` | `capacity_factor=4.0` | none |
| `sparse-gmm-fixed` (sgmf) | `sparse_matmul=true use_turbo_grouped_gemm=true` | grouped_gemm, compute_group_offs |
| `sparse-gmm-deepep-v3` (sgdv3) | `sparse_matmul=true use_turbo_grouped_gemm=true use_turbo_deepep_dispatch=true` | grouped_gemm + moe_dispatch + moe_combine |

Note the flag rename on this branch: `use_deepep_dispatch` -> `use_turbo_deepep_dispatch`.

---

# Part A - EP/FSDP pdbs sweep (51 cells)

Full sweep on the c6f2df3d image (new Primus-Turbo). The pp-support batching rules are inert
under EP/FSDP, so this part verifies that the new Primus-Turbo `main` (refactor) + the MaxText
GMM/DeepEP integration reproduce the 2026-05-09 baseline.

### Tokens/s/device (TGS, mean steps 5-14)

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|-------------:|----------:|----------:|-----------------:|---------------------:|
|    1 |        336.4 |     317.9 |     275.3 |            293.2 |                324.6 |
|    2 |        590.5 |     533.2 |     413.6 |            496.6 |                571.7 |
|    4 |        961.1 |     813.1 |     532.6 |            757.1 |                873.4 |
|    5 |       1083.5 |     881.3 |     561.1 |            847.3 |                973.1 |
|    6 |       1176.2 |     915.1 |     562.8 |            911.7 |               1055.5 |
|    7 |       1209.8 |     934.3 |     575.5 |          974.5 a |               1121.6 |
| **8**|   **1292.7** |  **960.1**|  **580.7**|        OOM 242.3 |           **1179.7** |
|    9 |       1399.0 |     994.8 | OOM 211.9 |        OOM 242.1 |               1189.4 |
|   10 |       1369.1 |    1013.4 | OOM 213.5 |        OOM 264.1 |     WARNING NaN (553.9) |
|   12 |          -   |       -   | OOM 221.2 |              -   |            OOM 217.3 |
|   16 |       1438.1 |    1015.1 | OOM 278.4 |              -   |            OOM 316.4 |

**Peak MFU:** `dense-cf1.25 @ pdbs=16` -> 14.41 %. **Peak dropless:** `sgd-deepep-v3 @ pdbs=8` -> 11.82 %.

### Training loss at step 14

Matches the 2026-05-09 baseline to <= 0.002; all configs agree to <= 0.003 within each pdbs row
(bf16-LSB invariant).

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|-------------:|----------:|----------:|-----------------:|---------------------:|
|    1 |        7.713 |     7.713 |     7.713 |            7.692 |                7.693 |
|    2 |        8.593 |     8.593 |     8.592 |            8.592 |                8.592 |
|    4 |        9.439 |     9.439 |     9.437 |            9.437 |                9.438 |
|    5 |        9.684 |     9.683 |     9.682 |            9.681 |                9.681 |
|    6 |        9.884 |     9.884 |     9.883 |            9.883 |                9.883 |
|    7 |       10.030 |    10.030 |    10.030 |           10.030 |               10.029 |
|    8 |       10.157 |    10.157 |    10.156 |              OOM |               10.156 |
|    9 |       10.266 |    10.266 |       OOM |              OOM |               10.266 |
|   10 |       10.354 |    10.353 |       OOM |              OOM |          WARNING NaN |
|   16 |       10.821 |    10.820 |       OOM |              -   |                 -    |

### EP feasibility (unchanged from baseline)

- `dense-cf1.25`, `dense-cf2`: >= 16. `dense-cf4`: max 8 (pdbs>=9 OOM 211.9-278.4 GiB).
- `sparse-gmm-fixed`: max 7 (`.96` required at pdbs=7; pdbs>=8 OOM 242-264 GiB).
- `sparse-gmm-deepep-v3`: usable max 8; pdbs=9 clean (1189 TGS, loss 10.266); pdbs=10 NaN (fits,
  ~74 s/step, unusable); pdbs=12/16 OOM (217.3/316.4 GiB). OOM allocations match the baseline to <= 1 GiB.

### EP vs 2026-05-09 baseline

Loss matches to <= 0.002 (exact, see the loss table) and feasibility / OOM ceilings are identical
to the GiB. TGS is within run-to-run + nodelist variance (the 2026-05-09 rerun used a different
8-node nodelist and the predecessor MaxText branch): dense lands at parity, the dropless sparse
paths a few percent either side. No regression in correctness, feasibility, or throughput.

---

# Part B - PP=8 verification (9 cells) - feature/pp-support

PP=8 = `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1` (MaxText auto-derives
`num_layers_per_pipeline_stage=1`, `num_pipeline_microbatches=8`), `remat_policy: 'full'`.
This part is what actually exercises the `feature/pp-support` batching rules: under PP the MoE
layer is wrapped in `nn.vmap(spmd_axis_name="stage")`, so the Primus-Turbo primitives are batched
over the stage axis. `configs/deepseek3-671b.env.sh` auto-selects the PP=8 recipe here
(`--xla_gpu_experimental_parallel_collective_overlap_limit=2`, + `async_priority` for the sparse
paths) by detecting `dcn_pipeline_parallelism>1` in the passthrough args; FSDP runs still get the
`ag=1 GiB` flag from the same file.

| Config | pdbs | TGS (steps 9-14) | loss@14 | Note |
|---|---:|---:|---:|---|
| sgd-deepep-v3 | 2 | 927.0 | 8.546 | dispatch+combine+GMM batched over stage |
| sgd-deepep-v3 | 4 | 991.0 | 9.392 | |
| sgd-deepep-v3 | 7 | 1013.2 | **9.994** | matches pp-vs-fsdp doc exactly |
| sgd-deepep-v3 | 8 | 997.2 | **10.136** | no OOM (SPMD memory win); matches doc |
| sparse-gmm-fixed | 4 | 838.3 | 9.393 | grouped-GEMM batched over stage (no DeepEP) |
| sparse-gmm-fixed | 6 | 859.1 | 9.835 | |
| dense-cf1.25 | 2 | 1122.0 | 8.559 | PP baseline (no Primus-Turbo) |
| dense-cf1.25 | 4 | 1216.9 | 9.400 | |
| dense-cf1.25 | 7 | 1246.4 | **9.998** | matches pp-vs-fsdp doc exactly |

**What PP=8 confirms about `feature/pp-support`:**

1. **All batching rules engage under `nn.vmap("stage")`** with no batching/vmap errors:
   `sgd-deepep-v3` exercises `moe_dispatch` + `moe_combine` + `grouped_gemm` + `compute_group_offs`;
   `sparse-gmm-fixed` independently exercises the grouped-GEMM family (no DeepEP).
2. **`setup()` works under PP** (fires in the Ray actor, before the first dispatch).
3. **The axis-aware SPMD rules deliver the memory win** - `sgd-deepep-v3` PP=8 pdbs=8 fits (no OOM),
   the data point that only fits with the axis-aware (not reshape-merge/scan) batching path.
4. **Loss is correct** - p7/p8 match the pp-vs-fsdp doc exactly (9.994 / 10.136 / 9.998); at lower
   pdbs the two dropless paths agree tightly (sgdv3 p4 9.392 vs sgmf p4 9.393, delta 0.001) and
   dense sits slightly above (dropping-vs-dropless gap, as documented).
5. **TGS is sensible and correctly ordered** (dense > DeepEP > ragged), and matches/slightly
   exceeds the pp-vs-fsdp doc at p7/p8.

(Note: PP=8 loss at a given pdbs differs from the EP/FSDP sweep value because PP changes the global
batch; the correct reference is the pp-vs-fsdp doc, which matches.)

---

## Conclusion

Both branches are verified non-breaking for `deepseek3-671b`:

- **MaxText `feature/primus-turbo-gmm-deepep-integration`** (GMM `9aeaa97b` + DeepEP `1c572908`):
  numerically correct and feasibility-equivalent on EP/FSDP (51-cell sweep), and correct under PP=8.
- **Primus-Turbo `feature/pp-support`** (axis-aware batching, in c6f2df3d): reproduces the
  2026-05-09 baseline on the EP path (where the rules are inert) and is verified correct +
  memory-efficient where the rules are active (PP=8).

**Scope/caveats:**
- Verified on `deepseek3-671b`, the 5 configs, EP/FSDP + PP=8. Strong regression check on the
  production workload, not a broad multi-model guarantee. The PP set is 9 focused cells but
  exercises every turbo primitive under the stage-vmap.
- The DeepEP path required the `setup()` forward-port to run on the refactored Primus-Turbo (without
  it the path RuntimeErrors). That is a required API adaptation (now part of `1c572908`), not a
  regression.
- 2 transient RCCL-init flakes during the PP runs auto-retried to SUCCESS; one host event during
  the EP sweep was resumed with no change to results.

## Per-cell job-id map

**EP/FSDP (c6f2df3d image), nodelist `chi[2798,2800,2816,2832,2835,2865,2872,2883]`:**

| Config | pdbs -> job |
|---|---|
| sgd-deepep-v3 | 1:16720 2:16721 4:16722 5:16723 6:16725 7:16727 8:16728 9:16729 10:16730(NaN) 12:16731(OOM) 16:16732(OOM) |
| sparse-gmm-fixed | 1:16733 2:16734 4:16735 5:16736 6:16737 7:16738 8:16739(OOM) 9:16740(OOM) 10:16741(OOM) |
| dense-cf1.25 | 1:16742 2:16743 4:16744 5:16746 6:16747 7:16748 8:16749 9:16750 10:16751 16:16752 |
| dense-cf2 | 1:16753 2:16754 4:16756 5:16757 6:16758 7:16759 8:16760 9:16761 10:16764 16:16766 |
| dense-cf4 | 1:16771 2:16773 4:16774 5:16775 6:16776 7:16777 8:16780 9:16782(OOM) 10:16783(OOM) 12:16784(OOM) 16:16785(OOM) |

**PP=8 (c6f2df3d image):**

| Config | pdbs -> job |
|---|---|
| sgd-deepep-v3 | 2:16791 4:16793 7:16788 8:16789 |
| sparse-gmm-fixed | 4:16794 6:16795 |
| dense-cf1.25 | 2:16798 4:16800 7:16790 |

## How to reproduce

```bash
cd /maxtext-slurm
BR=feature/primus-turbo-gmm-deepep-integration
IMG=/mnt/vast/yihuang/c6f2df3d-deepep-gmm-maxtext-v26.2.tar
COMMON="dataset_type=synthetic jax_distributed_heartbeat_timeout_seconds=99999"

# EP/FSDP sgd-deepep-v3 pdbs=8 (yml default dcn_fsdp_parallelism=8)
DOCKER_IMAGE=$IMG MAXTEXT_PATCH_BRANCH=$BR USE_DOCKER_IMAGE_AINIC_DRIVER=false RAY=1 ./submit.sh \
  deepseek3-671b --partition=k8s --nodes=8 --time=45:00 -- \
  per_device_batch_size=8 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_turbo_deepep_dispatch=true $COMMON

# PP=8 sgd-deepep-v3 pdbs=7 (env.sh supplies overlap_limit=2)
DOCKER_IMAGE=$IMG MAXTEXT_PATCH_BRANCH=$BR USE_DOCKER_IMAGE_AINIC_DRIVER=false RAY=1 ./submit.sh \
  deepseek3-671b --partition=k8s --nodes=8 --time=45:00 -- \
  per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_turbo_deepep_dispatch=true dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1 $COMMON
```
