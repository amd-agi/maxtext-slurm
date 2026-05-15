# DeepSeek-V3 671B — pdbs throughput rerun (2026-05-09)

> **Scope:** synthetic-data throughput rerun of the **5 most-relevant configs only** from the original [deepseek3-671b-pdbs-sweep.md](deepseek3-671b-pdbs-sweep.md), on a fresh nodelist and the current production image. No loss tests, no profile drill-downs, no `.zh.md`. Original sweep remains the authoritative full-taxonomy reference; this rerun is a hardware-consistency check on the curves that matter most operationally and a memory-ceiling re-survey under the current production image.

- **Date:** 2026-05-09 (sweep launched 2026-05-10 02:58 UTC after waiting for prior loss-test job 15070 to clear; finished 2026-05-11 13:36 UTC after ~10.5 h of mostly-pipelined Slurm wall time)
- **Model:** `deepseek3-671b` (MaxText)
- **Hardware:** 8 nodes × 8× AMD MI355 (288 GB HBM/device), Pensando AINIC interconnect
- **Nodelist (frozen for every cell + retry):** `chi[2766,2798,2800,2816,2832,2835,2865,2872]` (partition `k8s`); `chi2798` replaced the originally-requested `chi2878` (the latter was reported bad before the sweep started)
- **Image:** `/mnt/vast/yihuang/ppfix-hangfix-deepep-gmm-maxtext-v26.2.tar` (current `container_env.sh` default — newer than the original sweep's `deepep-gmm-maxtext-v26.2.tar`; see "Cross-sweep comparability" below)
- **Patch branch:** `yihuang/moe-turbo-gmm-and-deepep-v3` (current `container_env.sh` default; the patch only enters the `sgd-deepep-v3` row's HLO via `use_deepep_dispatch=true` — the other 4 configs are unaffected)
- **Base config:** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)
- **Dataset:** `dataset_type=synthetic` passthrough on every job (the gpu.yml defaults to `grain`/c4 — synthetic is required for throughput consistency)
- **Peak BF16:** ≈ 2500 TFLOP/s/device → MFU ≈ TFLOP/25
- **Launcher:** 1-node/proc only (default `RAY=1`)

## Cross-sweep comparability

The image changed between the original sweep and this rerun (`deepep-gmm-maxtext-v26.2.tar` → `ppfix-hangfix-deepep-gmm-maxtext-v26.2.tar`). The "ppfix-hangfix" image is the current production default in `container_env.sh`; numbers in this rerun therefore reflect what an operator running today's production stack would see, **not** byte-comparable reproductions of the original sweep's measurements. Per-config TGS deltas vs the original (see "Vs original" rows in the matrix below) split into three regimes:

- **Dense paths gain +3–15 % TGS** at every pdbs (largest at mid-pdbs ~ 4–6).
- **`sparse-gmm-fixed` is ~2 % slower** at every pdbs (consistent slight regression on the kNccl ragged-all-to-all path).
- **`sparse-gmm-deepep-v3` gains +5–8 %** at every pdbs.

Loss-at-step-14 matches the original sweep to ≤ 0.001 in every row — numerical correctness is fully preserved across the image change. (The one exception is `sgd-deepep-v3 pdbs=10` which produced NaN on the new image — see takeaway #5.)

NUM_STREAMS=1 (8 nodes / 8 nodes-per-job ⇒ serial scheduler), so there's no calibration measurement to report and no per-stream factor.

## Configs under test

Same passthrough flags as the [original sweep](deepseek3-671b-pdbs-sweep.md#configs-under-test); 5 of the 10 rows are exercised:

| Tag                    | Submit env vars | Passthrough flags (after `--`)                                                              |
|------------------------|-----------------|---------------------------------------------------------------------------------------------|
| `dense-cf1.25`         | —               | *(default)* — `sparse_matmul=false`, `capacity_factor=1.25`                                |
| `dense-cf2`            | —               | `capacity_factor=2.0`                                                                       |
| `dense-cf4`            | —               | `capacity_factor=4.0`                                                                       |
| `sparse-gmm-fixed`     | —               | `sparse_matmul=true use_turbo_grouped_gemm=true`                                            |
| `sparse-gmm-deepep-v3` | —               | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`                   |

Every job adds the heartbeat hedge `jax_distributed_heartbeat_timeout_seconds=99999` and `dataset_type=synthetic` (required since the yml now defaults to `grain`). `sparse-gmm-fixed pdbs=7` additionally adds `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96` (matching the original sweep's footnote ᵃ).

## Feasibility summary

| Config                 | max_pdbs (rerun) | max_pdbs (orig) | Δ                                       |
|------------------------|-----------------:|----------------:|-----------------------------------------|
| `dense-cf1.25`         |             ≥ 16 |            ≥ 16 | unchanged                               |
| `dense-cf2`            |             ≥ 16 |            ≥ 16 | unchanged                               |
| `dense-cf4`            |                8 |               7 | **+1** (new image fits one more pdbs)   |
| `sparse-gmm-fixed`     |  7 (`.96` req'd) | 7 (`.96` req'd) | unchanged                               |
| `sparse-gmm-deepep-v3` |                8 |               7 | **+1 memory** / +0 usable (see ⚠ below) |

⚠ `sgd-deepep-v3 pdbs=10` *fits memory* (does not OOM) but produces `loss=NaN` from step 1 with anomalous step time (~73 s vs ~30 s expected). It is therefore not usable in practice. `pdbs=8` is the largest **usable** value. `pdbs=12` and `pdbs=16` OOM. See takeaway #5.

## Results matrix

All metrics except loss are **mean over training steps 5–14** (steps 0–4 discarded as warmup). Loss is reported from step 14 only. `✗` = OOM; `—` = skipped or not run.

### Tokens/s/device (TGS)

| pdbs |   dense-cf1.25 |     dense-cf2 |     dense-cf4 |        sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|---------------:|--------------:|--------------:|------------------------:|---------------------:|
|    1 |          344.3 |         325.0 |         280.9 |                   299.5 |                335.5 |
|    2 |          605.5 |         544.8 |         417.7 |                   505.8 |                583.9 |
|    4 |          982.3 |         822.0 |         535.4 |                   767.7 |                909.5 |
|    5 |         1105.5 |         890.9 |         561.0 |                   858.4 |               1011.6 |
|    6 |         1187.1 |         923.2 |         562.1 |                   923.1 |               1088.8 |
|    7 |         1224.6 |         941.7 |         574.2 |                  984.7ᵃ |               1151.4 |
| **8**|     **1303.7** |     **963.6** |     **578.3** |                     ✗ᶜ |           **1199.9** |
|    9 |         1408.0 |         997.8 |   ✗ 211.9 GiB |     ✗ 242.1 GiB at .96  |               1209.7 |
|   10 |         1368.8 |        1010.7 |   ✗ 213.5 GiB |     ✗ 264.1 GiB at .96  |   ⚠ NaN (see #5)     |
|   12 |          —     |         —     |   ✗ 221.2 GiB |              —          |          ✗ 217.3 GiB |
|   16 |         1435.5 |        1013.4 |   ✗ 278.4 GiB |              —          |          ✗ 316.4 GiB |

### TFLOP/s/device

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 |  sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|-------------:|----------:|----------:|------------------:|---------------------:|
|    1 |         86.2 |      81.4 |      70.4 |              75.0 |                 84.0 |
|    2 |        151.6 |     136.4 |     104.6 |             126.7 |                146.2 |
|    4 |        246.0 |     205.9 |     134.1 |             192.3 |                227.8 |
|    5 |        276.9 |     223.1 |     140.5 |             215.0 |                253.4 |
|    6 |        297.3 |     231.2 |     140.8 |             231.2 |                272.7 |
|    7 |        306.7 |     235.9 |     143.8 |            246.6ᵃ |                288.4 |
|    8 |        326.5 |     241.3 |     144.8 |               ✗ᶜ  |           **300.5** |
|    9 |        352.6 |     249.9 |   ✗       |               ✗   |                303.0 |
|   10 |        342.8 |     253.1 |   ✗       |               ✗   |   ⚠ NaN              |
|   16 |        359.5 |     253.8 |   ✗       |              —    |                 —    |

**Peak MFU:** `dense-cf1.25 @ pdbs=16` → 14.38 % (TFLOP 359.5 / 2500). **Peak dropless MFU:** `sgd-deepep-v3 @ pdbs=8` → **12.02 %** (TFLOP 300.5 / 2500), beating the original sweep's `sgd-deepep-v3 @ pdbs=7` peak of 10.99 %.

### Average per-step time (seconds)

Lower is better.

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 |  sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|-------------:|----------:|----------:|------------------:|---------------------:|
|    1 |        11.90 |     12.60 |     14.58 |             13.67 |                12.21 |
|    2 |        13.53 |     15.04 |     19.61 |             16.20 |                14.03 |
|    4 |        16.68 |     19.93 |     30.60 |             21.34 |                18.01 |
|    5 |        18.52 |     22.99 |     36.51 |             23.86 |                20.24 |
|    6 |        20.70 |     26.62 |     43.72 |             26.62 |                22.57 |
|    7 |        23.41 |     30.45 |     49.93 |            29.12ᵃ |                24.90 |
|    8 |        25.13 |     34.01 |     56.67 |               ✗ᶜ  |               27.31 |
|    9 |        26.18 |     36.95 |   ✗       |               ✗   |                30.47 |
|   10 |        29.92 |     40.53 |   ✗       |               ✗   |  ⚠ 73 (NaN)          |
|   16 |        45.65 |     64.67 |   ✗       |              —    |                 —    |

### Training loss at step 14

Within each pdbs row, all configs agree to Δ ≤ 0.003 — same numerical-correctness invariant as the original sweep. Sparse sits ~0.02 below dense at pdbs=1 because sparse is dropless, identical to the original-sweep observation.

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 |  sparse-gmm-fixed | sparse-gmm-deepep-v3 |
|-----:|-------------:|----------:|----------:|------------------:|---------------------:|
|    1 |        7.715 |     7.714 |     7.713 |             7.693 |                7.693 |
|    2 |        8.593 |     8.593 |     8.592 |             8.591 |                8.592 |
|    4 |        9.439 |     9.439 |     9.437 |             9.437 |                9.438 |
|    5 |        9.684 |     9.682 |     9.682 |             9.681 |                9.681 |
|    6 |        9.884 |     9.883 |     9.883 |             9.883 |                9.883 |
|    7 |       10.030 |    10.030 |    10.030 |            10.029 |               10.029 |
|    8 |       10.157 |    10.157 |    10.156 |               ✗ᶜ  |              10.156 |
|    9 |       10.267 |    10.266 |   ✗       |               ✗   |               10.266 |
|   10 |       10.354 |    10.353 |   ✗       |               ✗   |          ⚠ NaN       |
|   16 |       10.821 |    10.820 |   ✗       |              —    |                 —    |

## Vs original sweep — TGS deltas

Sign and magnitude of the new-image effect on each config. `Δ% = (rerun − orig) / orig × 100`.

| pdbs | cf1.25 Δ% | cf2 Δ% | cf4 Δ% | sgmf Δ% | sgdv3 Δ% |
|-----:|----------:|-------:|-------:|--------:|---------:|
|    1 |     +3.4  |  +4.5  |  +6.0  |   −1.8  |    +5.8  |
|    2 |     +7.5  |  +9.7  | +11.7  |   −1.6  |    +7.1  |
|    4 |    +13.3  | +14.0  |  +7.1  |   −1.8  |    +8.4  |
|    5 |    +14.9  | +12.0  |  +4.9  |   −2.5  |    +6.7  |
|    6 |    +14.1  | +10.6  |  +3.5  |   −2.7  |    +5.7  |
|    7 |    +12.7  |  +6.5  |  +2.5  |   −0.4  |    +5.0  |
|    8 |     +9.5  |  +5.0  | NEW    |    n/a  |    NEW   |
|   16 |     +1.6  |  +4.7  |  n/a   |    n/a  |    n/a   |

## Key takeaways

1. **Numerical correctness preserved across image change.** All 41 successful cells produce loss-at-step-14 within ≤ 0.001 of the original sweep's values where applicable (the original sweep's standard ladder didn't include pdbs=9 or pdbs=10, but those new rows are internally consistent: all clean configs agree to ≤ 0.001 within each pdbs row). Within each pdbs row of the original-ladder rows, all 5 configs agree to Δ ≤ 0.003 — exactly the bf16-LSB invariant the original sweep documented. The single exception is `sgd-deepep-v3 pdbs=10` — see takeaway #5.

2. **Dense paths consistently faster on the new image (+3–15 % TGS).** Largest gains at mid-pdbs (pdbs=4–6 across all three dense configs). dense-cf1.25 peak (pdbs=16) lands at 1436 TGS / MFU 14.38 % (vs orig 1416 / 14.19 %, +1.4 % TGS).

3. **`sparse-gmm-fixed` is ~2 % slower across the curve.** Consistent small regression on the kNccl `ragged_all_to_all` path. Peak dropless via this config: pdbs=7 with `MEM_FRACTION=.96` → **985 TGS / MFU 9.86 %** (vs orig 989 TGS / 9.91 %). pdbs=8 OOM ceiling unchanged (242 GiB allocation matches the original's `c` footnote).

4. **`sparse-gmm-deepep-v3` gains +5–8 % across the curve AND extends the usable ceiling by one pdbs.** Peak dropless on this rerun is `sgd-deepep-v3 @ pdbs=8` → **1200 TGS / MFU 12.02 %**, beating both (a) the original sweep's `sgd-deepep-v3 @ pdbs=7` peak of 1097 TGS by +9.4 %, and (b) `sparse-gmm-fixed @ pdbs=7 .96` on this rerun by +21.9 %. `sgd-deepep-v3` is the unambiguous best dropless path on the production image.

5. **`sgd-deepep-v3 pdbs=10` NaN is unique to that one (config × pdbs) cell.** Cross-config comparison at pdbs=9 and pdbs=10 isolates the failure tightly:

   | Config              | pdbs=8 | pdbs=9                 | pdbs=10                  | pdbs=12       | pdbs=16       |
   |---------------------|:------:|:----------------------:|:------------------------:|:-------------:|:-------------:|
   | `dense-cf1.25`      | 1304 ✓ | **1408 ✓**             | **1369 ✓**               | —             | 1436 ✓        |
   | `dense-cf2`         |  964 ✓ |  **998 ✓**             | **1011 ✓**               | —             | 1013 ✓        |
   | `dense-cf4`         |  578 ✓ | ✗ 211.9 GiB            | ✗ 213.5 GiB              | ✗ 221.2 GiB  | ✗ 278.4 GiB   |
   | `sparse-gmm-fixed`  | ✗ 242 GiB at .96 | **✗ 242.1 GiB at .96** | **✗ 264.1 GiB at .96** | —             | —             |
   | `sgd-deepep-v3`     | 1200 ✓ |  **1210 ✓ (loss 10.266)** | **⚠ NaN, 2.7× slowdown** | ✗ 217.3 GiB | ✗ 316.4 GiB  |

   The pdbs=9/10 row across the four other configs (4 new probes recorded as jobs 15131, 15132, 15133, 15137, 15138, 15139, plus the existing cf4 OOMs) shows the failure **is not a memory-pressure gradient and is not pdbs=10 generically**. Specifically:

   - Both dense-cf1.25 and dense-cf2 run pdbs=10 with normal loss (10.354 / 10.353) — the same-pdbs loss invariant holds across all clean configs.
   - `sgd-deepep-v3 pdbs=9` is fully clean (TGS 1210, loss 10.266, matching the pdbs=9 dense-row exactly to LSB) — so the NaN is *not* a gradual numerical breakdown as `sgd-deepep-v3` approaches its memory ceiling. pdbs=8 and pdbs=9 are both clean; pdbs=10 NaNs; pdbs=12 OOMs.
   - sgmf hits a hard wall at pdbs=8/9 (242 GiB allocation at both — the bottleneck is a non-pdbs-scaling tensor in the kNccl ragged-a2a path); pdbs=10 then jumps to 264 GiB. Memory ceiling unchanged at pdbs=7.

   Operationally, **`sgd-deepep-v3 pdbs=8` remains the max usable** and **pdbs=9 is now the new strict frontier in this rerun** (not in the standard ladder so not the headline). The pdbs=10 NaN is an XLA-level pathology specific to this exact (config × pdbs) cell — most plausibly a layout/rematerialization choice the scheduler picks only at this exact pdbs that interacts badly with the v3 `custom_vjp` backward (the `argsort + reduce-sum` permutation gather). Worth a per-pass HLO drill-down if/when this regression matters operationally; both `pdbs=8` and `pdbs=9` are clean alternatives.

6. **`dense-cf4` ceiling shifts +1 pdbs (7 → 8).** Original sweep had `dense-cf4 pdbs=8` OOM at 278 GiB; the new image fits pdbs=8 cleanly at 578 TGS, then OOMs at every pdbs ≥ 9 (211 → 222 GiB allocations). `pdbs=8` peak is +3.2 % TGS over `pdbs=7` — small, but it shifts the operationally-recommended max for very-high-capacity-factor runs. Note TGS is plateauing at this pdbs (562 → 562 → 574 → 578 across pdbs=5/6/7/8): `dense-cf4` is communication-bottlenecked above pdbs ≈ 5; further pdbs increases buy ≤ 3 % each before OOM.

7. **Memory-ceiling is image-sensitive in the dropless / high-capacity regime; cap-1.25/2 are slack.** The two ceilings that *did* shift (`cf4 7→8`, `sgdv3 7→8`) are the two that sit on the working-set cliff against 288 GB HBM in the original sweep. The image's hangfix package evidently saves a few GiB of working-set on those paths. `dense-cf1.25` and `dense-cf2` remained slack at pdbs=16 in both sweeps; `sparse-gmm-fixed` at pdbs=8 OOMs at 242 GiB on both, indicating the kNccl ragged-all-to-all path's working-set is dominated by collective-buffer geometry that's unchanged across image versions.

## Per-cell job-id and output-dir map

For traceability and any later drill-down. All jobs ran on the frozen nodelist `chi[2766,2798,2800,2816,2832,2835,2865,2872]`, partition `k8s`, with `--time=45:00`.

| Config                 |  pdbs | Job ID | Output dir suffix                                                                                                                                                                                                  | Notes                                              |
|------------------------|------:|-------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `dense-cf1.25`         |     1 |  15078 | `15078-…-dense-cf1.25-per_device_batch_size_1-…`                                                                                                                                                                  |                                                    |
| `dense-cf1.25`         |     2 |  15079 | `15079-…-dense-cf1.25-per_device_batch_size_2-…`                                                                                                                                                                  | cleanup-flake (training OK, exit 143 in teardown)  |
| `dense-cf1.25`         |     4 |  15080 | `15080-…-dense-cf1.25-per_device_batch_size_4-…`                                                                                                                                                                  |                                                    |
| `dense-cf1.25`         |     5 |  15081 | `15081-…-dense-cf1.25-per_device_batch_size_5-…`                                                                                                                                                                  |                                                    |
| `dense-cf1.25`         |     6 |  15082 | `15082-…-dense-cf1.25-per_device_batch_size_6-…`                                                                                                                                                                  | cleanup-flake                                      |
| `dense-cf1.25`         |     7 |  15083 | `15083-…-dense-cf1.25-per_device_batch_size_7-…`                                                                                                                                                                  |                                                    |
| `dense-cf1.25`         |     8 |  15084 | `15084-…-dense-cf1.25-per_device_batch_size_8-…`                                                                                                                                                                  | cleanup-flake                                      |
| `dense-cf1.25`         |     9 |  15135 | *(scancelled — RCCL-init hang, see infra notes)*                                                                                                                                                                  | superseded by 15138                                |
| `dense-cf1.25`         |     9 |  15138 | `15138-…-dense-cf1.25-per_device_batch_size_9-…`                                                                                                                                                                  | RCCL-init flake retry of 15135; cleanup-flake      |
| `dense-cf1.25`         |    10 |  15131 | `15131-…-dense-cf1.25-per_device_batch_size_10-…`                                                                                                                                                                 | NaN-isolation probe                                |
| `dense-cf1.25`         |    16 |  15085 | `15085-…-dense-cf1.25-per_device_batch_size_16-…`                                                                                                                                                                 |                                                    |
| `dense-cf2`            |     1 |  15086 | `15086-…-dense-cf2-per_device_batch_size_1-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     2 |  15087 | `15087-…-dense-cf2-per_device_batch_size_2-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     4 |  15088 | `15088-…-dense-cf2-per_device_batch_size_4-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     5 |  15089 | `15089-…-dense-cf2-per_device_batch_size_5-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     6 |  15090 | `15090-…-dense-cf2-per_device_batch_size_6-capacity_factor_2.0-…`                                                                                                                                                 | cleanup-flake                                      |
| `dense-cf2`            |     7 |  15091 | `15091-…-dense-cf2-per_device_batch_size_7-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     8 |  15092 | `15092-…-dense-cf2-per_device_batch_size_8-capacity_factor_2.0-…`                                                                                                                                                 |                                                    |
| `dense-cf2`            |     9 |  15136 | *(scancelled — RCCL-init hang, see infra notes)*                                                                                                                                                                  | superseded by 15139                                |
| `dense-cf2`            |     9 |  15139 | `15139-…-dense-cf2-per_device_batch_size_9-capacity_factor_2.0-…`                                                                                                                                                 | RCCL-init flake retry of 15136                     |
| `dense-cf2`            |    10 |  15132 | `15132-…-dense-cf2-per_device_batch_size_10-capacity_factor_2.0-…`                                                                                                                                                | NaN-isolation probe                                |
| `dense-cf2`            |    16 |  15093 | `15093-…-dense-cf2-per_device_batch_size_16-capacity_factor_2.0-…`                                                                                                                                                | cleanup-flake                                      |
| `dense-cf4`            |     1 |  15094 | *(scancelled — RCCL-init hang, see infra notes)*                                                                                                                                                                  | superseded by 15100                                |
| `dense-cf4`            |     1 |  15100 | `15100-…-dense-cf4-per_device_batch_size_1-capacity_factor_4.0-…`                                                                                                                                                 | RCCL-init flake retry of 15094                     |
| `dense-cf4`            |     2 |  15095 | `15095-…-dense-cf4-per_device_batch_size_2-capacity_factor_4.0-…`                                                                                                                                                 |                                                    |
| `dense-cf4`            |     4 |  15096 | `15096-…-dense-cf4-per_device_batch_size_4-capacity_factor_4.0-…`                                                                                                                                                 |                                                    |
| `dense-cf4`            |     5 |  15097 | `15097-…-dense-cf4-per_device_batch_size_5-capacity_factor_4.0-…`                                                                                                                                                 |                                                    |
| `dense-cf4`            |     6 |  15098 | `15098-…-dense-cf4-per_device_batch_size_6-capacity_factor_4.0-…`                                                                                                                                                 | cleanup-flake                                      |
| `dense-cf4`            |     7 |  15099 | `15099-…-dense-cf4-per_device_batch_size_7-capacity_factor_4.0-…`                                                                                                                                                 | cleanup-flake                                      |
| `dense-cf4`            |     8 |  15112 | `15112-…-dense-cf4-per_device_batch_size_8-capacity_factor_4.0-…`                                                                                                                                                 | new probe (fits!) cleanup-flake                    |
| `dense-cf4`            |     9 |  15121 | `15121-…-dense-cf4-per_device_batch_size_9-capacity_factor_4.0-…`                                                                                                                                                 | new probe — OOM 211.9 GiB                          |
| `dense-cf4`            |    10 |  15119 | `15119-…-dense-cf4-per_device_batch_size_10-capacity_factor_4.0-…`                                                                                                                                                | new probe — OOM 213.5 GiB                          |
| `dense-cf4`            |    12 |  15120 | `15120-…-dense-cf4-per_device_batch_size_12-capacity_factor_4.0-…`                                                                                                                                                | new probe — OOM 221.2 GiB                          |
| `dense-cf4`            |    16 |  15117 | `15117-…-dense-cf4-per_device_batch_size_16-capacity_factor_4.0-…`                                                                                                                                                | new probe — OOM 278.4 GiB                          |
| `sparse-gmm-fixed`     |     1 |  15101 | `15101-…-sparse-gmm-fixed-per_device_batch_size_1-…`                                                                                                                                                              |                                                    |
| `sparse-gmm-fixed`     |     2 |  15102 | `15102-…-sparse-gmm-fixed-per_device_batch_size_2-…`                                                                                                                                                              |                                                    |
| `sparse-gmm-fixed`     |     4 |  15103 | `15103-…-sparse-gmm-fixed-per_device_batch_size_4-…`                                                                                                                                                              |                                                    |
| `sparse-gmm-fixed`     |     5 |  15104 | `15104-…-sparse-gmm-fixed-per_device_batch_size_5-…`                                                                                                                                                              |                                                    |
| `sparse-gmm-fixed`     |     6 |  15105 | `15105-…-sparse-gmm-fixed-per_device_batch_size_6-…`                                                                                                                                                              |                                                    |
| `sparse-gmm-fixed`     |     7 |  15113 | `15113-…-sparse-gmm-fixed-per_device_batch_size_7-…-_env_XLA_PYTHON_CLIENT_MEM_FRACTION_.96-…`                                                                                                                    | ᵃ `MEM_FRACTION=.96` required                     |
| `sparse-gmm-fixed`     |     8 |  15114 | `15114-…-sparse-gmm-fixed-per_device_batch_size_8-…-_env_XLA_PYTHON_CLIENT_MEM_FRACTION_.96-…`                                                                                                                    | OOM-confirm — 242.3 GiB at .96, matches original ᶜ |
| `sparse-gmm-fixed`     |     9 |  15137 | `15137-…-sparse-gmm-fixed-per_device_batch_size_9-…-_env_XLA_PYTHON_CLIENT_MEM_FRACTION_.96-…`                                                                                                                    | NaN-isolation probe — OOM 242.1 GiB at .96        |
| `sparse-gmm-fixed`     |    10 |  15133 | `15133-…-sparse-gmm-fixed-per_device_batch_size_10-…-_env_XLA_PYTHON_CLIENT_MEM_FRACTION_.96-…`                                                                                                                   | NaN-isolation probe — OOM 264.1 GiB at .96        |
| `sparse-gmm-deepep-v3` |     1 |  15106 | *(scancelled — RCCL-init hang, see infra notes)*                                                                                                                                                                  | superseded by 15116                                |
| `sparse-gmm-deepep-v3` |     1 |  15116 | `15116-…-sgd-deepep-v3-per_device_batch_size_1-…`                                                                                                                                                                 | RCCL-init flake retry of 15106; cleanup-flake      |
| `sparse-gmm-deepep-v3` |     2 |  15107 | `15107-…-sgd-deepep-v3-per_device_batch_size_2-…`                                                                                                                                                                 | cleanup-flake                                      |
| `sparse-gmm-deepep-v3` |     4 |  15108 | `15108-…-sgd-deepep-v3-per_device_batch_size_4-…`                                                                                                                                                                 |                                                    |
| `sparse-gmm-deepep-v3` |     5 |  15109 | `15109-…-sgd-deepep-v3-per_device_batch_size_5-…`                                                                                                                                                                 | cleanup-flake                                      |
| `sparse-gmm-deepep-v3` |     6 |  15110 | `15110-…-sgd-deepep-v3-per_device_batch_size_6-…`                                                                                                                                                                 |                                                    |
| `sparse-gmm-deepep-v3` |     7 |  15111 | `15111-…-sgd-deepep-v3-per_device_batch_size_7-…`                                                                                                                                                                 | cleanup-flake                                      |
| `sparse-gmm-deepep-v3` |     8 |  15115 | `15115-…-sgd-deepep-v3-per_device_batch_size_8-…`                                                                                                                                                                 | new probe (fits!) cleanup-flake                    |
| `sparse-gmm-deepep-v3` |     9 |  15134 | `15134-…-sgd-deepep-v3-per_device_batch_size_9-…`                                                                                                                                                                 | NaN-isolation probe — clean (loss 10.266)          |
| `sparse-gmm-deepep-v3` |    10 |  15122 | `15122-…-sgd-deepep-v3-per_device_batch_size_10-…`                                                                                                                                                                | new probe — NaN loss + slowdown; cancelled at step 8 |
| `sparse-gmm-deepep-v3` |    12 |  15123 | `15123-…-sgd-deepep-v3-per_device_batch_size_12-…`                                                                                                                                                                | new probe — OOM 217.3 GiB                          |
| `sparse-gmm-deepep-v3` |    16 |  15118 | `15118-…-sgd-deepep-v3-per_device_batch_size_16-…`                                                                                                                                                                | new probe — OOM 316.4 GiB                          |

A machine-readable TSV with all metrics (TGS / TFLOP / step_s / MFU / loss14 / status) is at [`outputs/.rerun_2026_05_09/results.tsv`](outputs/.rerun_2026_05_09/results.tsv).

## Infrastructure / memory-ceiling notes

- **NUM_STREAMS = 1** (8 nodes / 8 nodes-per-job ⇒ serial sweep). No per-stream calibration.
- **Pre-sweep node health**: skipped formal `sinfo + ssh + rocm-smi + ibstat` round because job 15070 had been running cleanly on the same 8-node list for ≥ 18.5 h when this rerun was queued (de-facto health validation).
- **Queue head-of-line wait**: cell 1 (`dense-cf1.25 pdbs=1`) submitted at 2026-05-10 02:58 UTC and pended ~1 h until job 15070 completed and 15071/15072 (its `runafternotok` backups) auto-cancelled.
- **Image change vs original sweep**: see "Cross-sweep comparability" above.
- **Four RCCL-init flakes total** — same signature each time: BARRIER reached, then 11–25 min of log silence with no `completed step:` line, no GID-WARN, no OOM signature, no other error. Per the sweep prompt's protocol (each affected cell's pdbs is *not* > last_successful for that config → not OOM-hang suspect → genuine RCCL-init flake), each was scancelled and a fresh resubmit ran cleanly to completion. Cells affected: cf4 pdbs=1 / 15094 → 15100; sgdv3 pdbs=1 / 15106 → 15116; cf1.25 pdbs=9 / 15135 → 15138; cf2 pdbs=9 / 15136 → 15139. All four are at **the first compile of a new HLO shape** (cf4 pdbs=1 was the first cf4 cell; sgdv3 pdbs=1 was the first sgdv3 cell; cf1.25 pdbs=9 and cf2 pdbs=9 were both the first compile at pdbs=9 for those configs, added during the post-hoc NaN-isolation probing). Cells inside an already-warm-config-and-pdbs combination never flaked. This matches the original sweep's "RCCL-init hangs on 1-node/proc are flaky" observation; the new finer-grained pattern is "first compile of any new HLO shape, not just first compile of a new config".
- **`cleanup-flake` (training OK but JOB SUMMARY = FAILED exit 143/1)** affected ~10 cells across all configs. All such cells reached `completed step: 14`; the failure is a Docker / Ray teardown race after training completes successfully. Per `skills/job-log-triage`, training data is intact — recorded normally with no retry. No correlation with any specific node.
- **`sgd-deepep-v3 pdbs=10` numerical anomaly** (15122): step 0 produced normal warmup loss (12.27); from step 1 onward `loss=NaN` persisted, and step time stayed at 73 s — about **2.7× slower than the ~27 s/step `pdbs=8` measured on the same image, same nodelist, immediately before**. pdbs=12 (15123) OOMed cleanly at 217 GiB (memory-ceiling), so the pdbs=10 NaN is *not* a memory-pressure-induced numerical breakdown. Job was scancelled at step 8 to free the queue; this cell is recorded as `numerical-failure (NaN, deferred)`. The most plausible cause is an HLO layout/rematerialization choice that XLA picks at this specific batch size — worth a per-pass HLO drill-down if/when this regression matters operationally.
- **Memory-ceiling drill** for `dense-cf4` and `sgd-deepep-v3`: original sweep had both OOM at pdbs=8. On this image, both fit at pdbs=8. Probing higher: cf4 pdbs=9 OOMs (211.9 GiB), pdbs=10 OOMs (213.5 GiB), pdbs=12 OOMs (221.2 GiB), pdbs=16 OOMs (278.4 GiB) — confirming `cf4 max=8`. sgdv3 pdbs=9 fits cleanly (1210 TGS, loss 10.266), pdbs=10 hits the NaN anomaly above (memory fits but unusable), pdbs=12 OOMs (217.3 GiB), pdbs=16 OOMs (316.4 GiB) — confirming `sgdv3 max usable=8` for the standard ladder, with pdbs=9 as a non-standard but clean alternative.

- **NaN-isolation probes for the `sgd-deepep-v3 pdbs=10` anomaly** (added after initial sweep): probed pdbs=9 and pdbs=10 across all 5 configs to determine whether the NaN was sgdv3-specific, pdbs=10-generic, or memory-pressure-driven. Result: NaN is a single (config × pdbs) cell — only `sgd-deepep-v3 pdbs=10`. Specifically, `sgd-deepep-v3` produces clean loss at both adjacent pdbs (pdbs=8: 10.156, pdbs=9: 10.266), then NaN at pdbs=10, then OOM at pdbs=12. dense-cf1.25 and dense-cf2 both run pdbs=10 with normal loss (10.354 / 10.353). dense-cf4 OOMs at every pdbs ≥ 9 (so no comparison data point at pdbs=10 from that config). sparse-gmm-fixed OOMs at pdbs=9 with the *same* 242 GiB allocation as pdbs=8 — the kNccl ragged-a2a path's working-set is dominated by a non-pdbs-scaling tensor, so pdbs=9 is OOM-equivalent to pdbs=8 (memory ceiling for sgmf is firmly pdbs=7). pdbs=10 jumps the sgmf OOM allocation to 264 GiB. See takeaway #5 for the cross-config table and the sgdv3-NaN-cell-isolation conclusion.

## Footnotes

- **ᵃ** `sparse-gmm-fixed pdbs=7` requires `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96` (default `.93` OOMs allocating 217 GiB; the kNccl path's working set at pdbs=7 exceeds the default pool). Identical to the original sweep's footnote ᵃ.
- **ᶜ** `sparse-gmm-fixed pdbs=8` OOMs allocating 242.29 GiB even at `MEM_FRACTION=.96`. Identical to the original sweep's footnote ᶜ.
- All other cells use the default `MEM_FRACTION=.93` from `train_env.sh`.

## How to reproduce

```bash
cd /maxtext-slurm
NL='chi[2766,2798,2800,2816,2832,2835,2865,2872]'

# dense-cf1.25, pdbs=1
RAY=1 ./submit.sh deepseek3-671b:dense-cf1.25 \
  --partition=k8s --nodes=8 --nodelist="$NL" --time=45:00 -- \
  per_device_batch_size=1 dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999

# dense-cf2, pdbs=4
RAY=1 ./submit.sh deepseek3-671b:dense-cf2 \
  --partition=k8s --nodes=8 --nodelist="$NL" --time=45:00 -- \
  per_device_batch_size=4 capacity_factor=2.0 dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999

# dense-cf4, pdbs=8 (new max — was OOM in original sweep)
RAY=1 ./submit.sh deepseek3-671b:dense-cf4 \
  --partition=k8s --nodes=8 --nodelist="$NL" --time=45:00 -- \
  per_device_batch_size=8 capacity_factor=4.0 dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-fixed, pdbs=7 (needs MEM_FRACTION=.96)
RAY=1 ./submit.sh deepseek3-671b:sparse-gmm-fixed \
  --partition=k8s --nodes=8 --nodelist="$NL" --time=45:00 -- \
  per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true \
  dataset_type=synthetic _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96 \
  jax_distributed_heartbeat_timeout_seconds=99999

# sparse-gmm-deepep-v3, pdbs=8 (new max — peak dropless, no env-var prefix needed)
RAY=1 ./submit.sh deepseek3-671b:sgd-deepep-v3 \
  --partition=k8s --nodes=8 --nodelist="$NL" --time=45:00 -- \
  per_device_batch_size=8 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_deepep_dispatch=true dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999
```
