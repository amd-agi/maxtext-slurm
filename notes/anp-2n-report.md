# ANP Performance Delta at 2-Node Scale — 2026-04-21

## TL;DR

Enabling the ANP NCCL plugin (`librccl-anp.so`) on the `deepep-a77` partition **slows training by 23.4%** on a 2-node × 8-MI355X setup (1444 → 1106 TGS, +30.6% step time). The result is statistically ironclad (Welch's t = 57.5 across n=28 steady-state steps per variant, CV < 3%) and **reproduces the 8-node regression** previously observed in jobs 9501 (WithOUTANP, 3155 TGS) vs 9511 (WithANP, 2538 TGS) = **−19.6%**.

| | **WithOUTANP** (RoCE) | **WithANP** (plugin) | **Delta** |
|---|---|---|---|
| Mean TGS (tokens/s/device) | **1444.4** ± 6.4 | **1105.8** ± 30.5 | **−23.4%** |
| Mean step time | 5.67 s ± 0.03 | 7.40 s ± 0.16 | **+30.6%** |
| MFU (fp8) | 23.4% | 17.9% | −5.5 pp |
| Run-to-run stability (CV) | 0.44% | 2.76% | ANP is noisier |

Both paths successfully use the 8 `ionic_*` RoCE NICs per node at 4096 MTU. The delta is purely software: NCCL built-in RoCE transport vs the ANP plugin's network fabric, with everything else held constant.

---

## 1. Experimental setup

### Hardware
- **Partition**: `deepep-a77` (a-77 firmware, MI355X GPUs, 288 GB HBM3e each)
- **Nodes**: `chi[2812, 2899]` — 2 nodes × 8 MI355X = 16 GPUs
- **NICs**: 8 × ionic RoCE per node (`ionic_0` through `ionic_7`), `1.117.5-a-77` firmware, PORT_ACTIVE, 4096 MTU
- **Host RAM**: ~3 TB/node

### Software
- **Image**: `/mnt/vast/qiangh/docker_images/jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2.tar` (same image used by run0 job 12295 which ran DeepSeek-v3 671B at 8N successfully)
- **JAX**: 0.8.2 (PJRT C API, ROCm 7.1.1)
- **RCCL**: 2.27.7-HEAD:84d2752
- **MaxText**: v26.2 (pip-installed in container as `maxtext==0.1.1`)
- **ANP plugin**: `/workspace/amd-anp/build/librccl-anp.so` baked into the image

### Model (405B architecture, depth-scaled)
Scaled-down llama3.1-405b for 2N HBM fit without optimizer host-offload:
- Config: `configs/llama3.1-405b-40L-2n.gpu.yml`
- Depth: **40 decoder layers** (vs 126 for full 405B) → **131.7B params**
- **All width knobs unchanged** from 405B so per-layer FSDP byte volume and comm/compute ratio are preserved: `base_emb_dim=16384`, `base_mlp_dim=53248`, `base_num_query_heads=128`, `base_num_kv_heads=8`, `head_dim=128` (GQA 16:1), `decoder_block=llama2`, `attention=cudnn_flash_te`, `remat_policy=full`, `quantization=fp8`
- Parallelism: `dcn_fsdp_parallelism=-1`, `ici_fsdp_parallelism=-1` → pure FSDP (8-way within node, 2-way across nodes). Identical to the 8N reference runs (9501/9511/9512).
- Batch/seq: `per_device_batch_size=1`, `max_target_length=8192` → 131,072 tokens/step (global)
- Memory footprint per GPU (no offload): ~9 GB weights (fp8) + ~19 GB grads (bf16) + ~75 GB optimizer states (fp32 m+v) ≈ **103 GB state + activations**, comfortably in the 268 GB HBM budget.

### Why depth-scaling and not host offload?
Full 405B at 2N (pure FSDP-16) needs `optimizer_memory_host_offload=True` to fit — the optimizer states dominate (126 × 3.2 GB fp32 m+v / 16 = 200 GB/GPU alone). But offload makes every step CPU↔GPU-PCIe-bound (measured step = 380 s, MFU = 1.1%), which **hides** the ANP signal. Depth-scaling keeps per-layer comm volume identical while making state fit in HBM — so the 2N step is comm+compute-bound just like the 8N reference.

### Why the `jax_distributed_heartbeat_timeout_seconds=14400` override?
JIT compile for this 40L config takes ~90 s on MI355X. The default heartbeat timeout is ~60 s, which triggers a false-positive kill (see `docs/jax-heartbeat-false-positive-postmortem.md`). 14400 s (4 h) eliminates the race; has no effect on steady-state perf.

### Why `USE_DOCKER_IMAGE_AINIC_DRIVER=false`?
The container's baked-in `libionic1` is older than the a-77 host firmware. With the default `true`, the container's libibverbs can't enumerate host `ionic_*` devices, NCCL silently falls back to TCP sockets (→ 100 s+ per step, MFU 1.3%). Setting `false` bind-mounts host `/etc/libibverbs.d/ionic.driver` and `/usr/lib/x86_64-linux-gnu/libionic.so.*` into the container, restoring RoCE. (Job 13235 demonstrated the broken state; 13242 onward is on the fixed path.)

### Variant toggle (the *only* difference between WithOUTANP and WithANP)
In `_container.sh`, inside `IB_MOUNT_OPTIONS=( ... )`:
```
# WithOUTANP:
#   # -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so
# WithANP:
    -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so
```
Everything else (image, driver bind-mount, env vars, model, parallelism, batch, nodelist, reservation) is byte-identical across all 4 runs.

---

## 2. Results

### 2.1 Per-run steady-state (steps 1–14 after step-0 compile)

| Job | Variant | n | Mean step (s) | σ step (s) | Mean TGS | σ TGS | TGS range |
|---|---|---:|---:|---:|---:|---:|---|
| 13242 | WithOUTANP | 14 | 5.673 | 0.0245 | 1444.0 | 6.3 | 1437.1 – 1460.5 |
| 13250 | WithOUTANP | 14 | 5.670 | 0.0261 | 1444.9 | 6.7 | 1436.6 – 1460.6 |
| 13248 | WithANP | 14 | 7.281 | 0.148 | 1125.6 | 22.3 | 1070.6 – 1150.9 |
| 13249 | WithANP | 14 | 7.547 | 0.173 | 1086.0 | 24.3 | 1035.6 – 1113.1 |

### 2.2 Pooled by variant (n = 28 steady-state steps each)

| Variant | Mean TGS | σ | CV | TGS range |
|---|---:|---:|---:|---|
| WithOUTANP | **1444.4** | 6.4 | 0.44% | 1436.6 – 1460.6 |
| WithANP | **1105.8** | 30.5 | 2.76% | 1035.6 – 1150.9 |

- **Delta: −338.6 TGS, −23.4%**
- Welch's t = **57.5** (two-tailed p ≪ 10⁻³⁰)
- The WithANP distribution is also **≈5× noisier** (σ = 30.5 vs 6.4) — the ANP path introduces step-to-step jitter that the RoCE path does not.

### 2.3 Cross-scale consistency with 8-node reference

| Scale | Model | WithOUTANP TGS | WithANP TGS | Delta | Source |
|---|---|---:|---:|---:|---|
| 2N × 8 GPU | 405B-40L (131B) | 1444.4 | 1105.8 | **−23.4%** | This report |
| 8N × 8 GPU | ds-proxy-se0-e256-h4096 | 3155.7 | 2538.1 | **−19.6%** | Jobs 9501, 9511 |
| 8N × 8 GPU | ds-proxy-se0-e256-h4096 | (3155.7 baseline) | 2585.2 (PATCH) | −18.1% | Job 9512 |

The 2N regression is slightly larger than the 8N regression, which is consistent with the theory that ANP adds a fixed per-message cost: at 2N the DCN ring is narrower (2-way) so each cross-node RCCL op has fewer messages to amortize the overhead over.

---

## 3. What this means for ANP

1. **ANP is currently a perf regression, not a perf win**, at both 2N and 8N on the `deepep-a77` partition. The regression is reproducible, large (20–23% TGS), and statistically airtight.
2. **ANP also introduces step-time jitter** (5× stdev increase). For SLO-sensitive jobs this matters beyond just the mean slowdown.
3. **RoCE works fine** — the "no-ANP" path via host `ionic.driver` bind-mount hits 23.4% MFU at pdbs=1 on 2N, with 0.4% CV. That is the target ANP must meet or beat.
4. **The software toggle is the ONLY knob changed** between WithOUTANP and WithANP runs. Both paths use the same 8 ionic NICs, same RoCE fabric, same image, same JAX/RCCL, same host driver. The regression is entirely inside the ANP plugin vs RCCL's built-in IB transport.

---

## 4. How to reproduce (copy-paste)

```bash
cd /mnt/vast/qiangh/ANP_test/maxtext-slurm
git checkout anp-2n-repro

# WithOUTANP: ensure _container.sh line 298-304 has the NCCL_NET_PLUGIN line
# COMMENTED, then:
RAY=1 ./submit.sh 405b-40L-2n:WithOUTANP -N 2 \
    --partition=deepep-a77 --nodelist='chi[2812,2899]' -- \
    steps=15 dataset_type=synthetic \
    base_num_decoder_layers=40 \
    jax_distributed_heartbeat_timeout_seconds=14400 \
    _env_NCCL_DEBUG=INFO \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.93

# WithANP: uncomment the NCCL_NET_PLUGIN line in _container.sh, commit, then:
RAY=1 ./submit.sh 405b-40L-2n:WithANP -N 2 \
    --partition=deepep-a77 --nodelist='chi[2812,2899]' -- \
    steps=15 dataset_type=synthetic \
    base_num_decoder_layers=40 \
    jax_distributed_heartbeat_timeout_seconds=14400 \
    _env_NCCL_DEBUG=INFO \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
```

Each run takes ~5 minutes end-to-end (including ~2 min for preflight + JIT compile and ~1.5 min of training). Per-step TGS is visible in the job log; steady-state stabilizes at step 1.

---

## 5. Paths and artifacts

- **Workspace**: `/mnt/vast/qiangh/ANP_test/maxtext-slurm` (branch `anp-2n-repro`, not pushed)
- **New config**: `configs/llama3.1-405b-40L-2n.gpu.yml`
- **Job outputs**: `outputs/{13242,13250,13248,13249}-JAX-llama3.1-405b-40L-2n-...`
- **Per-run artifacts** (frozen copy of scripts + submit cmd): `outputs/<job>/artifact -> ../.artifacts/artifact_<ts>/`
- **RCCL transport confirmation** per run:
  ```
  grep 'NET/IB : Using' outputs/<job>-*.log
  ```
  All 4 runs show the same ionic devices; only NCCL_NET_PLUGIN differs.

## 6. Follow-up questions (for the ANP team)

1. **Is there an ANP tunable we're missing?** (QP count, completion-queue sizing, message-class routing). The 5× step-jitter in ANP suggests either a backpressure issue or a per-message overhead that RoCE's transport doesn't have.
2. **Does ANP expect different `NCCL_IB_HCA`/`NCCL_IB_GID_INDEX`/`NCCL_IB_TC` values?** We used the same settings (from `train_env.sh`) in both variants; ANP may want its own defaults.
3. **Which ANP build is this?** `/workspace/amd-anp/build/librccl-anp.so` inside the run0 training image. Timestamp / commit SHA would be useful to correlate with ANP team's build history.
