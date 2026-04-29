# DeepEP One-GPU-Per-Process Loss Correctness Report

Date: 2026-04-28

Model: `deepseek3-671b-proxy`

Node: `chi2815`

Partition: `deepep-a66`

Scope: single-node training correctness check for `sparse_matmul=True`,
`use_turbo_grouped_gemm=True`, `use_deepep_dispatch=True`, and
`ONE_GPU_PER_PROCESS=true`.

Interactive curves:

- `reports/deepep-loss-comparison.html`

## Core Conclusion

The DeepEP sparse grouped-GEMM path is numerically consistent with the dense
baseline for this 15-step synthetic training check.

Across the six successful runs below, the final loss is `7.061`-`7.064`, and
the maximum per-step loss difference from the dense baseline is at most `0.003`.
This is small enough to treat the DeepEP path and the one-GPU-per-process
launcher as correct for this test setup.

DeepEP also improves throughput materially: at `per_device_batch_size=4`,
DeepEP reaches about `6530`-`6543` tokens/s/device, compared with about
`5265`-`5286` tokens/s/device for sparse grouped GEMM without DeepEP.

## Test Matrix

All runs used:

```bash
dcn_fsdp_parallelism=1
dcn_data_parallelism=1
per_device_batch_size=4
steps=15
dataset_type=synthetic  # from the model config
```

The run mapping is:

| Case | Job | Config |
|---|---:|---|
| Baseline | 14191 | dense MoE, `capacity_factor=1.25`, regular process mode |
| Case 1 | 14208 | dense MoE, `capacity_factor=1.25`, `_env_ONE_GPU_PER_PROCESS=true` |
| Case 2 | 14209 | `sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=False` |
| Case 3 | 14205 | Case 2 + `_env_ONE_GPU_PER_PROCESS=true` |
| Case 4 | 14206 | `sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True` |
| Case 5 | 14207 | Case 4 + `_env_ONE_GPU_PER_PROCESS=true` |

## Loss Summary

| Job | Config | Final loss | Max abs delta vs baseline | Avg TGS/device, steps 2-14 |
|---:|---|---:|---:|---:|
| 14191 | dense baseline | 7.064 | 0.000 | 6,227 |
| 14208 | dense + 1g1p | 7.064 | 0.001 | 6,237 |
| 14209 | sparse + tgmm | 7.061 | 0.003 | 5,264 |
| 14205 | sparse + tgmm + 1g1p | 7.061 | 0.002 | 5,285 |
| 14206 | sparse + tgmm + DeepEP | 7.062 | 0.002 | 6,525 |
| 14207 | sparse + tgmm + DeepEP + 1g1p | 7.061 | 0.003 | 6,535 |

Per-step losses:

| Step | 14191 baseline | 14208 case1 | 14209 case2 | 14205 case3 | 14206 case4 | 14207 case5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.265 | 12.265 | 12.265 | 12.265 | 12.265 | 12.265 |
| 1 | 12.265 | 12.265 | 12.265 | 12.265 | 12.265 | 12.265 |
| 2 | 11.487 | 11.487 | 11.486 | 11.486 | 11.486 | 11.486 |
| 3 | 10.816 | 10.816 | 10.815 | 10.815 | 10.815 | 10.815 |
| 4 | 10.148 | 10.149 | 10.147 | 10.147 | 10.147 | 10.147 |
| 5 | 9.506 | 9.506 | 9.505 | 9.505 | 9.505 | 9.505 |
| 6 | 8.908 | 8.908 | 8.906 | 8.906 | 8.906 | 8.906 |
| 7 | 8.401 | 8.401 | 8.399 | 8.399 | 8.399 | 8.399 |
| 8 | 7.935 | 7.935 | 7.934 | 7.933 | 7.933 | 7.934 |
| 9 | 7.551 | 7.551 | 7.549 | 7.549 | 7.549 | 7.549 |
| 10 | 7.381 | 7.381 | 7.378 | 7.378 | 7.378 | 7.378 |
| 11 | 7.240 | 7.240 | 7.238 | 7.238 | 7.238 | 7.238 |
| 12 | 7.135 | 7.136 | 7.133 | 7.133 | 7.133 | 7.133 |
| 13 | 7.095 | 7.095 | 7.092 | 7.092 | 7.092 | 7.092 |
| 14 | 7.064 | 7.064 | 7.061 | 7.061 | 7.062 | 7.061 |

## Key Findings

1. **DeepEP preserves training loss.** `use_deepep_dispatch=True` matches the
   non-DeepEP sparse grouped-GEMM loss curve step-for-step, with only
   `0.000`-`0.001` visible differences after rounding to 3 decimals.

2. **One-GPU-per-process does not change the loss curve.** The launcher mode
   does not introduce a correctness regression:
   - dense baseline vs dense + 1g1p: max delta `0.001`
   - sparse + tgmm vs sparse + tgmm + 1g1p: max delta `0.001`
   - DeepEP vs DeepEP + 1g1p: max delta `0.001`

3. **Sparse/dropless paths differ slightly from the dense `capacity_factor=1.25`
   baseline.** The maximum delta vs dense baseline is `0.002`-`0.003`. This is
   expected because the dense baseline uses capacity-limited routing while the
   sparse grouped-GEMM / DeepEP paths exercise the sparse dispatch path.

4. **DeepEP improves throughput while preserving loss.** At
   `per_device_batch_size=4`, DeepEP improves steady-state TGS/device from
   roughly `5.26k` to `6.53k`, about a `24%` gain. The one-GPU-per-process
   DeepEP run is slightly faster in this sample (`~6.54k` TGS/device).

5. **Use `per_device_batch_size=4` for this correctness matrix.** The
   sparse-matmul batch sweep shows non-DeepEP sparse grouped-GEMM OOMs at
   `pdbs=5`, while DeepEP can pass larger batches. `pdbs=4` is the common
   passing point across all correctness cases.

## Reproduction

Run these commands from `/mnt/vast/llying/maxtext-slurm`.

### Baseline

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh deepseek3-671b-proxy -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15
```

### Case 1: Dense baseline + one GPU per process

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh 'deepseek3-671b-proxy:loss-1-d1g1p' -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15 _env_ONE_GPU_PER_PROCESS=true
```

### Case 2: Sparse grouped GEMM, no DeepEP

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh 'deepseek3-671b-proxy:loss-2-sp-tg' -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15 sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=False
```

### Case 3: Sparse grouped GEMM, no DeepEP, one GPU per process

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh 'deepseek3-671b-proxy:loss-3-stg-1g1p' -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15 sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=False _env_ONE_GPU_PER_PROCESS=true
```

### Case 4: Sparse grouped GEMM + DeepEP

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh 'deepseek3-671b-proxy:loss-4-stg-dp' -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15 sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True
```

### Case 5: Sparse grouped GEMM + DeepEP + one GPU per process

```bash
cd /mnt/vast/llying/maxtext-slurm && ./submit.sh 'deepseek3-671b-proxy:loss-5-stg-dp-1g1p' -N 1 -p deepep-a66 -w chi2815 -- dcn_fsdp_parallelism=1 dcn_data_parallelism=1 per_device_batch_size=4 steps=15 sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True _env_ONE_GPU_PER_PROCESS=true
```

## How To Recompute The Loss Comparison

Extract loss lines from each log:

```bash
rg 'completed step.*loss:' /mnt/vast/llying/maxtext-slurm/outputs/14191-*.log /mnt/vast/llying/maxtext-slurm/outputs/14205-*.log /mnt/vast/llying/maxtext-slurm/outputs/14206-*.log /mnt/vast/llying/maxtext-slurm/outputs/14207-*.log /mnt/vast/llying/maxtext-slurm/outputs/14208-*.log /mnt/vast/llying/maxtext-slurm/outputs/14209-*.log
```

For `ONE_GPU_PER_PROCESS=true` runs, each local process may print the same
training step. Deduplicate or average by `(job, step)` before comparing loss
curves. In this run, duplicate per-rank loss values were identical after
rounding to 3 decimals.

