# DeepSeek3-671B Proxy Sparse Matmul Batch Sweep

Date: 2026-04-28

Model: `deepseek3-671b-proxy`

Node: `chi2815`

Partition: `deepep-a66`

Common overrides:

```bash
dcn_fsdp_parallelism=1
dcn_data_parallelism=1
steps=15
sparse_matmul=True
use_turbo_grouped_gemm=True
```

The model config used the 15-layer proxy setting:

```yaml
base_num_decoder_layers: 15
first_num_dense_layers: 3
```

Metrics below are averaged over completed steps 5-14. For `ONE_GPU_PER_PROCESS=true` runs, multiple process logs can emit the same step; results were averaged per step first, then averaged across steps 5-14.

## Configs Swept

| Label | Original pdbs=4 job | Key overrides |
|---|---:|---|
| `stg` | 14209 | `sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=False` |
| `stg-1g1p` | 14205 | `stg + _env_ONE_GPU_PER_PROCESS=true` |
| `stg-dp` | 14206 | `stg + use_deepep_dispatch=True` |
| `stg-dp-1g1p` | 14207 | `stg + use_deepep_dispatch=True + _env_ONE_GPU_PER_PROCESS=true` |

## Results

### `stg`

| pdbs | Job | Status | Avg step time | Avg TGS/device | Avg MFU | Notes |
|---:|---:|---|---:|---:|---:|---|
| 4 | 14209 | SUCCESS | 3.112s | 5,264.8 | 13.85% | best/only passing |
| 5 | 14222 | OOM | - | - | - | allocation 180.78 GiB |
| 6 | 14218 | OOM | - | - | - | allocation 203.19 GiB |
| 8 | 14214 | OOM | - | - | - | allocation 224.79 GiB |
| 16 | 14210 | OOM | - | - | - | allocation 369.31 GiB |

Recommendation: `per_device_batch_size=4`.

### `stg-1g1p`

| pdbs | Job | Status | Avg step time | Avg TGS/device | Avg MFU | Notes |
|---:|---:|---|---:|---:|---:|---|
| 4 | 14205 | SUCCESS | 3.099s | 5,285.9 | 13.90% | best/only passing |
| 5 | 14223 | OOM | - | - | - | allocation 180.78 GiB |
| 6 | 14219 | OOM | - | - | - | allocation 203.19 GiB |
| 8 | 14215 | OOM | - | - | - | allocation 224.79 GiB |

Recommendation: `per_device_batch_size=4`.

### `stg-dp`

| pdbs | Job | Status | Avg step time | Avg TGS/device | Avg MFU | Notes |
|---:|---:|---|---:|---:|---:|---|
| 4 | 14206 | SUCCESS | 2.509s | 6,530.8 | 17.18% | slightly higher TGS |
| 6 | 14220 | SUCCESS | 3.770s | 6,518.4 | 17.14% | larger batch, slightly lower TGS |
| 7 | 14283 | OOM | - | - | - | allocation 179.04 GiB |
| 8 | 14216, 14224 | OOM | - | - | - | allocation 184.38 GiB |

Recommendation: `per_device_batch_size=4` if optimizing pure TGS; `per_device_batch_size=6` if a larger global batch is preferred with negligible TGS loss.

### `stg-dp-1g1p`

| pdbs | Job | Status | Avg step time | Avg TGS/device | Avg MFU | Notes |
|---:|---:|---|---:|---:|---:|---|
| 4 | 14207 | SUCCESS | 2.504s | 6,542.8 | 17.21% | best TGS |
| 6 | 14221 | SUCCESS | 3.767s | 6,524.4 | 17.16% | larger batch, slightly lower TGS |
| 7 | 14284 | OOM | - | - | - | allocation 179.04 GiB |
| 8 | 14217, 14225 | OOM | - | - | - | allocation 184.38 GiB |

Recommendation: `per_device_batch_size=4` if optimizing pure TGS; `per_device_batch_size=6` if a larger global batch is preferred with negligible TGS loss.

## Summary

| Config | Best pdbs | Best job | Avg TGS/device | Avg MFU | Max passing pdbs | First OOM |
|---|---:|---:|---:|---:|---:|---:|
| `stg` | 4 | 14209 | 5,264.8 | 13.85% | 4 | 5 |
| `stg-1g1p` | 4 | 14205 | 5,285.9 | 13.90% | 4 | 5 |
| `stg-dp` | 4 | 14206 | 6,530.8 | 17.18% | 6 | 7 |
| `stg-dp-1g1p` | 4 | 14207 | 6,542.8 | 17.21% | 6 | 7 |

Key observations:

- DeepEP dispatch (`use_deepep_dispatch=True`) improves TGS by about 24% at `pdbs=4` compared with sparse grouped GEMM without DeepEP.
- `ONE_GPU_PER_PROCESS=true` has almost no throughput penalty in the successful DeepEP sparse matmul path: `stg-dp-1g1p` is slightly faster than `stg-dp` at `pdbs=4`.
- DeepEP also improves memory headroom: non-DeepEP configs OOM at `pdbs=5`, while DeepEP configs pass at `pdbs=6` and OOM at `pdbs=7`.
- For maximum TGS, use `pdbs=4` for all four sparse matmul configs. For DeepEP configs, `pdbs=6` is also viable when a larger global batch is more important than the small TGS difference.

Supplemental probe conclusions:

- Non-DeepEP sparse grouped GEMM configs (`stg` and `stg-1g1p`) cannot move beyond `pdbs=4`: `pdbs=5` already OOMs, so the sweep ceiling is confirmed at `pdbs=4`.
- DeepEP sparse grouped GEMM configs (`stg-dp` and `stg-dp-1g1p`) pass at `pdbs=6`, but their TGS is slightly lower than `pdbs=4`; `pdbs=7` OOMs, so `pdbs=4` remains the throughput-optimal setting and `pdbs=6` is only useful for larger global batch requirements.
- `ONE_GPU_PER_PROCESS=true` does not introduce a visible throughput penalty in this sparse + DeepEP path. The `stg-dp-1g1p` run at `pdbs=4` is the fastest result in this sweep.

## Recommended Commands

Best TGS, regular process mode:

```bash
./submit.sh deepseek3-671b-proxy -N 1 -p deepep-a66 -w chi2815 -- \
  dcn_fsdp_parallelism=1 dcn_data_parallelism=1 \
  per_device_batch_size=4 steps=15 \
  sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True
```

Best TGS, `ONE_GPU_PER_PROCESS` mode:

```bash
./submit.sh deepseek3-671b-proxy -N 1 -p deepep-a66 -w chi2815 -- \
  dcn_fsdp_parallelism=1 dcn_data_parallelism=1 \
  per_device_batch_size=4 steps=15 \
  sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True \
  _env_ONE_GPU_PER_PROCESS=true
```

Larger passing batch for DeepEP, with slight TGS loss:

```bash
./submit.sh deepseek3-671b-proxy -N 1 -p deepep-a66 -w chi2815 -- \
  dcn_fsdp_parallelism=1 dcn_data_parallelism=1 \
  per_device_batch_size=6 steps=15 \
  sparse_matmul=True use_turbo_grouped_gemm=True use_deepep_dispatch=True
```
