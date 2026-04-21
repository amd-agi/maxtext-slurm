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

### Hardware (as run)
- **Partition**: `deepep-a77` (a-77 firmware, MI355X GPUs, 288 GB HBM3e each)
- **Nodes**: `chi[2812, 2899]` — 2 nodes × 8 MI355X = 16 GPUs
- **NICs**: 8 × ionic RoCE per node (`ionic_0` through `ionic_7`), `1.117.5-a-77` firmware, PORT_ACTIVE, 4096 MTU
- **Host RAM**: ~3 TB/node

### Software (as run)
- **Image**: `/mnt/vast/qiangh/docker_images/jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2.tar` (same image used by run0 job 12295 which ran DeepSeek-v3 671B at 8N successfully). Single-file manifest tag: `jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2:latest`.
- **JAX**: 0.8.2 (PJRT C API, ROCm 7.1.1)
- **RCCL**: 2.27.7-HEAD:84d2752
- **MaxText**: v26.2 (pip-installed in container as `maxtext==0.1.1`; source under `/workspace/maxtext/src/MaxText/`)
- **ANP plugin**: `/workspace/amd-anp/build/librccl-anp.so` baked into the image

### Model (405B architecture, depth-scaled)
Scaled-down llama3.1-405b for 2N HBM fit without optimizer host-offload:
- Config: `configs/llama3.1-405b-40L-2n.gpu.yml`
- Depth: **40 decoder layers** (vs 126 for full 405B) → **131.7B params** (runtime-reported)
- **All width knobs unchanged** from 405B so per-layer FSDP byte volume and comm/compute ratio are preserved: `base_emb_dim=16384`, `base_mlp_dim=53248`, `base_num_query_heads=128`, `base_num_kv_heads=8`, `head_dim=128` (GQA 16:1), `decoder_block=llama2`, `attention=cudnn_flash_te`, `remat_policy=full`, `quantization=fp8`
- Parallelism: `dcn_fsdp_parallelism=-1`, `ici_fsdp_parallelism=-1` → pure FSDP (8-way within node, 2-way across nodes). Identical to the 8N reference runs (9501/9511/9512).
- Batch/seq: `per_device_batch_size=1`, `max_target_length=8192` → 131,072 tokens/step (global)
- Memory footprint per GPU (no offload): ~9 GB weights (fp8) + ~19 GB grads (bf16) + ~75 GB optimizer states (fp32 m+v) ≈ **103 GB state + activations**, comfortably in the 268 GB HBM budget.

### Why depth-scaling and not host offload?
Full 405B at 2N (pure FSDP-16) needs `optimizer_memory_host_offload=True` to fit — the optimizer states dominate (126 × 3.2 GB fp32 m+v / 16 = 200 GB/GPU alone). But offload makes every step CPU↔GPU-PCIe-bound (measured step = 380 s, MFU = 1.1%), which **hides** the ANP signal. Depth-scaling keeps per-layer comm volume identical while making state fit in HBM — so the 2N step is comm+compute-bound just like the 8N reference.

### Why the `jax_distributed_heartbeat_timeout_seconds=14400` override?
JIT compile for this 40L config takes ~90 s on MI355X. The default heartbeat timeout is ~60 s, which triggers a false-positive kill (see `docs/jax-heartbeat-false-positive-postmortem.md`). 14400 s (4 h) eliminates the race; has no effect on steady-state perf.

### Why `USE_DOCKER_IMAGE_AINIC_DRIVER=false`?
The container's baked-in `libionic1` is older than the a-77 host firmware. With the default `true`, the container's libibverbs can't enumerate host `ionic_*` devices, NCCL silently falls back to TCP sockets (→ 100 s+ per step, MFU 1.3%). Setting `false` bind-mounts host `/etc/libibverbs.d/ionic.driver` and `/usr/lib/x86_64-linux-gnu/libionic.so.*` into the container, restoring RoCE. (Job 13235 demonstrated the broken state; 13242 onward is on the fixed path.) **May not be needed on clusters where the container driver matches host firmware** — see the cross-cluster checklist below.

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

## 4. How to reproduce on THIS cluster (same nodes, same image)

Current repo state on the `anp-2n-repro` branch at `/mnt/vast/qiangh/ANP_test/maxtext-slurm` has ANP OFF (WithOUTANP variant). The ANP toggle lives inside `IB_MOUNT_OPTIONS=( ... )` near line 301 of `_container.sh`:

```bash
IB_MOUNT_OPTIONS=(
    # -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so  # uncomment for WithANP
)
```

### Run 1 — WithOUTANP (baseline)
```bash
cd /mnt/vast/qiangh/ANP_test/maxtext-slurm
# ensure ANP line is COMMENTED (default state on this branch)
sed -i 's|^    -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so$|    # -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so  # uncomment for WithANP|' _container.sh
RAY=1 ./submit.sh 405b-40L-2n:WithOUTANP -N 2 \
    --partition=deepep-a77 --nodelist='chi[2812,2899]' -- \
    steps=15 dataset_type=synthetic \
    base_num_decoder_layers=40 \
    jax_distributed_heartbeat_timeout_seconds=14400 \
    _env_NCCL_DEBUG=INFO \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
```

### Run 2 — WithANP
```bash
cd /mnt/vast/qiangh/ANP_test/maxtext-slurm
# ensure ANP line is UNCOMMENTED
sed -i 's|^    # -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so.*|    -e NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so|' _container.sh
RAY=1 ./submit.sh 405b-40L-2n:WithANP -N 2 \
    --partition=deepep-a77 --nodelist='chi[2812,2899]' -- \
    steps=15 dataset_type=synthetic \
    base_num_decoder_layers=40 \
    jax_distributed_heartbeat_timeout_seconds=14400 \
    _env_NCCL_DEBUG=INFO \
    _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
```

Each run takes ~5 minutes end-to-end (≈2 min preflight + JIT compile, ≈1.5 min of training, ≈1.5 min teardown). Per-step TGS appears in the job log; steady-state stabilizes at step 1.

### Pull numbers afterwards
```bash
# per-step breakdown
grep 'completed step' outputs/<JOB_ID>-*.log | head -n 20

# steady-state summary across multiple runs (edit JOB_IDS)
python3 - <<'PY'
import re, glob, statistics as s
JOBS = [('13242','WithOUTANP'), ('13250','WithOUTANP'),
        ('13248','WithANP'),    ('13249','WithANP')]
data = {}
for j, v in JOBS:
    logf = glob.glob(f'outputs/{j}-*.log')[0]
    seen, tgs = set(), []
    with open(logf) as f:
        for line in f:
            m = re.search(r'completed step: (\d+),.*Tokens/s/device: ([\d.]+)', line)
            if not m: continue
            step = int(m.group(1))
            if step in seen or step == 0 or step > 14: continue
            seen.add(step); tgs.append(float(m.group(2)))
    data.setdefault(v, []).extend(tgs)
    print(f"{j} {v}: mean TGS={s.mean(tgs):.1f}  σ={s.stdev(tgs):.2f}  n={len(tgs)}")
mo = s.mean(data['WithOUTANP']); ma = s.mean(data['WithANP'])
print(f"\nWithOUTANP pooled: {mo:.1f}  WithANP pooled: {ma:.1f}  delta: {(ma-mo)/mo*100:+.1f}%")
PY
```

---

## 5. Reproducing on a DIFFERENT cluster (colleague checklist)

This section is for a reviewer running the experiment on their own MI355X nodes.

### 5.1 Preflight — verify hardware compatibility

Run these on each target node (from the host, not inside the container) before anything else:

```bash
# 1. Confirm 8 ionic RDMA NICs, all PORT_ACTIVE
ls /sys/class/infiniband/   # expect ionic_0 .. ionic_7
ibv_devinfo | grep -E 'hca_id|state:|active_mtu|fw_ver'
# expect: 8 hca_ids (ionic_0..7), state=PORT_ACTIVE (4), active_mtu=4096, fw_ver=1.117.5-a-77 (or newer a-77)

# 2. Confirm MI355X (or compatible MI3xx) GPUs and HBM
rocm-smi --showproductname | head -n 20    # expect MI355X / gfx950
rocm-smi --showmeminfo vram | grep -E 'GPU|VRAM' | head -n 5   # expect ~268-288 GiB free per GPU

# 3. Confirm host IB driver files present (needed for USE_DOCKER_IMAGE_AINIC_DRIVER=false path)
ls /etc/libibverbs.d/ionic.driver         # required
ls /usr/lib/x86_64-linux-gnu/libionic.*   # required

# 4. Confirm Slurm can see both nodes as idle (or reserve them)
sinfo -p <your-partition> -N -o '%N %T'
```

If any of the above fails, stop and report which check failed. The experiment depends on functioning ionic RoCE and a-77 firmware.

### 5.2 Get the code

The `anp-2n-repro` branch is **local-only** (not pushed to GitHub). Ways to transfer:

**Option A — git bundle (recommended, self-contained)**

On the source cluster:
```bash
cd /mnt/vast/qiangh/ANP_test/maxtext-slurm
git bundle create /tmp/anp-2n-repro.bundle main..anp-2n-repro   # ~30 KB
# scp /tmp/anp-2n-repro.bundle <colleague>
```

On the target cluster:
```bash
git clone https://github.com/AMD-AGI/maxtext-slurm.git    # public
cd maxtext-slurm
git fetch /path/to/anp-2n-repro.bundle anp-2n-repro:anp-2n-repro
git checkout anp-2n-repro
```

**Option B — apply the branch as a patch**

```bash
# Source:
git format-patch main..anp-2n-repro --stdout > /tmp/anp-2n-repro.patch   # ~25 KB
# Target (on a fresh clone of main):
git checkout -b anp-2n-repro
git am /path/to/anp-2n-repro.patch
```

**Option C — by-hand copy of the 3 touched files**

Only 3 files matter: `configs/llama3.1-405b-40L-2n.gpu.yml` (new), `_container.sh` (edited), `container_env.sh` (edited). A `git diff main..anp-2n-repro` shows exactly what changed.

### 5.3 Get a compatible image

The tar at `/mnt/vast/qiangh/docker_images/jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2.tar` is a 77 GB Docker save. What you need in an equivalent image:

| Requirement | Path inside the image | Check command |
|---|---|---|
| MaxText installed | `/workspace/maxtext/src/MaxText/train.py` AND `pip show maxtext` returns a version | `docker run --rm --entrypoint sh $IMG -c 'pip show maxtext && ls /workspace/maxtext/src/MaxText/train.py'` |
| ANP plugin | `/workspace/amd-anp/build/librccl-anp.so` | `docker run --rm --entrypoint sh $IMG -c 'ls /workspace/amd-anp/build/librccl-anp.so'` |
| JAX on ROCm | `jax.devices()` returns `RocmDevice(...)` | `docker run --rm --entrypoint sh $IMG -c 'python -c "import jax; print(jax.devices())"'` |
| RCCL built-in IB transport | RCCL ≥ 2.27 (for ionic via `libibverbs`) | `docker run --rm --entrypoint sh $IMG -c 'strings /opt/rocm/lib/librccl.so \| grep -i "RCCL version" \| head'` |
| FSDP-compatible MaxText | `llama3.1-405b` in `/workspace/maxtext/src/MaxText/configs/models/` | `docker run --rm --entrypoint sh $IMG -c 'ls /workspace/maxtext/src/MaxText/configs/models/llama3.1-405b.yml'` |

If your cluster has the same image tar accessible (e.g. same NFS mount layout), just edit the path in `container_env.sh`. Otherwise, transfer the tar (~77 GB) or build/pull an equivalent image and set `DOCKER_IMAGE` to it.

### 5.4 Adjust cluster-specific knobs in the repo

Edit **one file** per cluster: `container_env.sh`:

```bash
# Line ~24: point to YOUR image tar (or registry tag)
DOCKER_IMAGE="${DOCKER_IMAGE:-/path/to/your/image.tar}"

# Line ~25: decide USE_DOCKER_IMAGE_AINIC_DRIVER
# Try false first; if preflight run hits TCP fallback (see Step 5.5 verification), flip to true.
USE_DOCKER_IMAGE_AINIC_DRIVER="${USE_DOCKER_IMAGE_AINIC_DRIVER:-false}"
```

Everything else (the 40L config, the submit command shape) is cluster-agnostic. Just change `--partition=` and `--nodelist=` in the submit commands to match your Slurm layout.

### 5.5 First-run verification (do this BEFORE trusting the TGS numbers)

After running the **WithOUTANP** job, inspect its log to confirm you have real RDMA and the right transport:

```bash
LOG=outputs/<JOB_ID>-*.log

# a) RDMA active: expect 8 ionic_* devices, RoCE transport (not NET/Socket)
grep 'NET/IB : Using' $LOG | head -n 2
#  → should show: NET/IB : Using [0]ionic_0:1/RoCE ... [7]ionic_7:1/RoCE [RO]
#    If you see "NET/Socket : Using ..." instead, you're on TCP fallback:
#    flip USE_DOCKER_IMAGE_AINIC_DRIVER (try the other value) and rerun.

# b) ANP plugin NOT loaded on WithOUTANP
grep NCCL_NET_PLUGIN $LOG
#  → should show NOTHING (env var not set). If it shows a path, the toggle is wrong.

# c) Model built with 40 layers (not 126)
grep 'number parameters' $LOG
#  → expect "number parameters: 131.712 billion"
#    If it says 405.856 billion, the base_num_decoder_layers=40 passthrough
#    didn't take effect — re-check the submit command.

# d) Step 1+ cadence
grep 'completed step' $LOG | tail -n 15
#  → expect step 0 = 30-60s (compile); steps 1-14 = 5-7s each.
#    If steps 1+ are > 20s, you may still be on TCP fallback (step a).
```

Then run **WithANP** and verify ANP engaged:

```bash
LOG=outputs/<JOB_ID>-*.log
grep NCCL_NET_PLUGIN $LOG
#  → expect: NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-anp.so (on BOTH nodes)
#    If missing, the _container.sh toggle didn't apply — check the diff.
grep 'completed step' $LOG | tail -n 15
#  → expect steady-state TGS 20-25% lower than the WithOUTANP run.
```

### 5.6 What "success" looks like

Reproduction is successful when:
1. Hardware preflight (5.1) passes on both nodes.
2. WithOUTANP log shows RoCE transport via 8 ionic NICs, 131.7 B params, step 1+ in the 4–7 s range.
3. WithANP log shows `NCCL_NET_PLUGIN` set, same RoCE device list, step 1+ ≈ 25–40% slower than WithOUTANP.
4. Pooled TGS delta is in the **−18% to −25%** band (this report got −23.4%; the 8N reference was −19.6%). Any delta in that range confirms the regression; outside it, share the numbers for investigation.

---

## 6. Commit history on `anp-2n-repro` (annotated)

For reviewers who want to understand the debugging journey:

| SHA | Subject | What it means |
|---|---|---|
| `b3d6439` | WithOUTANP baseline setup | First attempt config (full 405B). Abandoned after OOM. Superseded by `a189746`. |
| `0df7da5` | switch to primus-training image | First image swap — abandoned, image lacked MaxText. |
| `14633c9` | use local tar image from run0 | Final image choice (known-working MaxText + Primus-Turbo + ANP). |
| `662fbfa` | enable optimizer_memory_host_offload for 2N | Host offload attempt — abandoned after PCIe became the bottleneck. |
| `a189746` | add llama3.1-405b-40L variant | Final model config (depth-scaled 405B). |
| `ee5211b` | move base_num_decoder_layers override to CLI | Documents that YAML is ignored by MaxText's model registry. |
| `b14af99` | bind-mount host IB libs | Fixes the container/host libionic firmware mismatch. |
| `011d081` | enable ANP plugin for WithANP variant | First ANP-on attempt. |
| `29f0ccd` | pull ANP plugin toggle out of USE_DOCKER_IMAGE_AINIC_DRIVER branch | Fixes a scoping bug — ANP env was lost when the driver bind-mount path was taken. |
| `f5e46d0` | disable ANP plugin for WithOUTANP variance run | Toggle flip for the second run of each variant. |
| `<latest>` | add results report | This file. |

The sequence documents three non-obvious failure modes that any reproduction on similar hardware is likely to hit (image-driver mismatch, model-registry precedence, toggle scoping). Each failure mode has a one-commit fix that's preserved in the branch for easy cherry-pick.

---

## 7. Paths and artifacts (source cluster)

- **Workspace**: `/mnt/vast/qiangh/ANP_test/maxtext-slurm` (branch `anp-2n-repro`, not pushed)
- **Config**: `configs/llama3.1-405b-40L-2n.gpu.yml`
- **Job outputs**: `outputs/{13242,13250,13248,13249}-JAX-llama3.1-405b-40L-2n-...` (raw logs + per-run artifact snapshots)
- **Per-run artifact symlink**: `outputs/<job>/artifact -> ../.artifacts/artifact_<ts>/` — contains the exact code state + `submit_cmd.txt` for that run.
- **Git bundle for transfer**: `/mnt/vast/qiangh/ANP_test/anp-2n-repro.bundle` (regenerate with `git bundle create ... main..anp-2n-repro`)

## 8. Follow-up questions (for the ANP team)

1. **Is there an ANP tunable we're missing?** (QP count, completion-queue sizing, message-class routing). The 5× step-jitter in ANP suggests either a backpressure issue or a per-message overhead that RoCE's transport doesn't have.
2. **Does ANP expect different `NCCL_IB_HCA`/`NCCL_IB_GID_INDEX`/`NCCL_IB_TC` values?** We used the same settings (from `train_env.sh`) in both variants; ANP may want its own defaults.
3. **Which ANP build is this?** `/workspace/amd-anp/build/librccl-anp.so` inside the `jax-training-maxtext-v26.2-with-primus-turbo-conv-fix-v2` image. Timestamp / commit SHA would be useful to correlate with ANP team's build history.
