# MoE pdbs sweep — generic agent prompt

Copy everything below this line into a fresh agent session.  **Before starting, fill in the "Model parameters" section below** with the target model's details.

---

# Task: Comprehensive pdbs sweep for `<MODEL_TAG>` (1-node/proc only)

Produce a full performance sweep document for the `<MODEL_TAG>` MaxText model, modeled after the existing DeepSeek-V3 sweep at `@maxtext-slurm/deepseek3-671b-pdbs-sweep.md`. Same structure, same config taxonomy, same metrics tables, same analysis depth.

**The headline question this sweep answers:** does the DS3 v1→v2→v3 `sparse-gmm-deepep` optimization story (+24% → +59% TGS at pdbs=6, attributed to eliminating the `input_scatter_fusion_*.kd` kernel) replicate on this model? The v2/v3 rows are the most valuable single deliverable — treat them as first-class.

*(For non-bf16 sweeps, this restates slightly — see [Dtype / quantization](#dtype--quantization) point 6 and its "survives the fp8 conversion?" framing.)*

**Scope: 1-node/proc launcher only, one dtype per sweep.** The 1-GPU/proc launcher mode's behavior is already fully characterized in the DS3 sweep (takeaway #3) — its findings are model-agnostic. Skip it. **Dtype is a separate sweep axis** (a new quantization target → a new pair of deliverable docs; see [Dtype / quantization](#dtype--quantization)).

**`dcn_expert_parallelism` (DCN-EP) sweep is OPT-IN, not default.** A new model's first sweep runs **only at the model's default `dcn_expert_parallelism` (typically 1)** — i.e. the existing main-matrix structure with no DCN-EP rows. **Do NOT proactively sweep DCN_EP > 1 unless the user explicitly requests it.** This is because (a) DCN_EP > 1 takes a 7×–22× compute multiplier on top of the base sweep (each non-default DCN_EP value adds 4–7 configs × ~5 pdbs each), (b) DeepEP variants are validator-blocked at DCN_EP > 1 so the most interesting "DS3 v3-vs-fixed at inter-node EP" question can't be answered, and (c) the typical finding ("dropless ceiling collapses, dense throughput degrades monotonically") is now well-characterized by the kimi-1T and DS3 DCN-EP extensions and replicates predictably across new models — running it on every new sweep is rarely worth the GPU-hours.

**When the user *does* ask for DCN-EP coverage**, treat it as an in-doc column dimension (not a new deliverable doc) — non-default DCN_EP values go into the same per-dtype `<MODEL_TAG>[-<QUANTIZATION>]-pdbs-sweep.md` as extra rows in every results table. See [Parallelism axis: `dcn_expert_parallelism` (DCN-EP)](#parallelism-axis-dcn_expert_parallelism-dcn-ep) for the protocol.

## Model parameters (fill in before starting)

- **MODEL_TAG**: e.g. `kimi-k2-1t` (the `configs/<MODEL_TAG>.gpu.yml` name, also the `submit.sh <MODEL_TAG>:...` first arg)
- **CONFIG_FILE**: `@maxtext-slurm/configs/<MODEL_TAG>.gpu.yml`
- **MODEL_ENV_FILE** (optional): `@maxtext-slurm/configs/<MODEL_TAG>.env.sh` if one exists (some models like grok-2 have per-model env overrides)
- **QUANTIZATION**: `bf16` (default) | `fp8` | `nanoo_fp8`. This is a first-class sweep axis — **a non-bf16 sweep produces its own pair of deliverable docs**, not an extra column in the bf16 doc. See the [Dtype / quantization](#dtype--quantization) section below for the full set of behavioral differences.
- **DCN_EP_VALUES**: list of `dcn_expert_parallelism` values to sweep for DCN-EP-relevant configs. **Default: `[1]`** (only the model's default `dcn_expert_parallelism`, i.e. no DCN-EP extension). Set to e.g. `[1, 2, 4, 8]` **only when the user explicitly asks for a DCN-EP sweep** (see opt-in rule above). Each value factorizes as `ici_expert_parallelism=8` × `dcn_expert_parallelism=N` × `dcn_fsdp_parallelism=NODES_PER_JOB/N`. Non-default values land as additional rows in the per-config ladder within the same per-dtype deliverable doc — this is not a separate-doc axis like `QUANTIZATION`. See [Parallelism axis: `dcn_expert_parallelism` (DCN-EP)](#parallelism-axis-dcn_expert_parallelism-dcn-ep) for the row layout, which-configs-to-re-sweep rules, and the DS3 claim this axis was designed to test.
- **OUTPUT_DOC_EN**: `@maxtext-slurm/<MODEL_TAG>-pdbs-sweep.md` *(bf16)* OR `@maxtext-slurm/<MODEL_TAG>-<QUANTIZATION>-pdbs-sweep.md` when `QUANTIZATION != bf16` (e.g. `kimi-k2-1t-fp8-pdbs-sweep.md`). **All DCN_EP values land in this single doc** as extra rows per config.
- **OUTPUT_DOC_ZH**: `@maxtext-slurm/<MODEL_TAG>-pdbs-sweep.zh.md` *(bf16)* OR `@maxtext-slurm/<MODEL_TAG>-<QUANTIZATION>-pdbs-sweep.zh.md` when `QUANTIZATION != bf16`.
- **NODES_PER_JOB**: integer — minimum nodes required for one training instance. Read from the config's `dcn_fsdp_parallelism` × `dcn_expert_parallelism` × other dcn axes (the "world size" at the DCN granularity). For most MoE models on this cluster, it's 8 (dcn_fsdp=8). When sweeping DCN_EP > 1, NODES_PER_JOB stays the same (e.g. 8) — what changes is the **factorization**: `dcn_fsdp=8/dcn_ep=1` → `dcn_fsdp=4/dcn_ep=2` → `dcn_fsdp=2/dcn_ep=4` → `dcn_fsdp=1/dcn_ep=8`, all using the same 8-node topology.
- **PARTITION**: e.g. `k8s` or `mi355x`
- **AVAILABLE_NODES**: sorted list of currently-healthy node names in PARTITION, e.g. `node1,node2,node3,node4,node5,node6,node7,node8`. Run `sinfo -p <PARTITION>` up front to determine this.
- **NUM_STREAMS**: `floor(len(AVAILABLE_NODES) / NODES_PER_JOB)` — how many configs can run in parallel. If `NUM_STREAMS == 1`, the sweep is serial. If `≥ 2`, use the parallel-streams protocol below.

### Dtype / quantization

`QUANTIZATION` is orthogonal to the 10-config routing taxonomy. The same 10 configs, the same pdbs probing, the same decision rules — just a different dtype axis, producing its own pair of `.md` / `.zh.md` deliverables (linked from and cross-referencing the bf16 sweep's pair for the same model).

> **⚠️ MLA models currently unsupported at `QUANTIZATION != bf16`.** Models with `attention_type=mla` (deepseek3-671b, kimi-k2-1t, deepseek2-*, anything else with q/kv-LoRA projections) crash XLA's fp8 GEMM rewriter on the current `rocm-jaxlib-v0.8.2` build:
>
> ```
> F gemm_rewriter.cc:1382] Check failed: b_contracting_dims[0] == num_batch_dims || b_contracting_dims[0] == num_batch_dims + 1
> ```
>
> The CHECK is in `xla::gpu::GemmRewriterVisitor::CreateF8CustomCall()` — it's a layout precondition that some HLO pass between SPMD-partitioning and gemm-rewriter violates when MLA's q/kv-LoRA backward dots interact with fp8 quantization. The crash is a `F[ATAL]` absl CHECK that aborts the Python process unrecoverably (and Ray's stdout-capture inflates the SLURM log to multi-GB — the [runaway-log pattern](#crash--hang--flake-quick-reference) — within ~3 min).
>
> Verified scope (2026-04-26 diagnostic session, image `deepep-gmm-maxtext-v26.2.tar`, jaxlib `0.8.2.dev0+selfbuilt`):
>
> | Model | Attention | MoE? | fp8 status |
> |---|---|---|---|
> | `llama2-70b` | MHA | dense | ✅ works |
> | `llama3.1-405b` | GQA | dense | ✅ works |
> | `grok-1` | MHA | MoE 8e | ✅ works |
> | `deepseek3-671b` | **MLA** | MoE 256e | ❌ crashes |
> | `kimi-k2-1t` | **MLA** | MoE 384e | ❌ crashes |
>
> Workarounds tried that did NOT resolve it (kept here as a marker so future attempts don't re-tread the same ground):
>
> 1. Skip fp8 wrapping on MoE `combine` einsum (`moe.py / get_einsum`).
> 2. Skip fp8 on `out_projection` DenseGeneral (`attentions.py / init_out_w`).
> 3. Skip fp8 on **all** MLA DenseGenerals (`attention_mla.py`, q-LoRA + kv-LoRA + q-up + kv-up).
> 4. Disable `dot-merger` HLO pass via `--xla_disable_hlo_passes=dot-merger`.
>
> All 4 layered → post-SPMD HLO has 0 fp8 dots violating the CHECK in static parse, yet the GemmRewriter still crashes on a dot produced by some downstream pass (algebraic-simplifier / transpose-folding / dot-decomposer / gemm-broadcast-folding-rewriter — undetermined which one without per-pass HLO drill-down).
>
> **Action when the user requests `QUANTIZATION=fp8` on an MLA model:** TG-tell the user the bug exists, point at this section, and ask whether to (a) skip and document, (b) pivot to a non-MLA model, or (c) attempt the per-pass HLO drill-down to find the residual bad-dot-creating pass. Do not start the sweep — every probe will runaway-log.
>
> This block can be removed when MaxText (or the underlying XLA fork) ships a fix that makes `quantization=fp8` compile cleanly on MLA + MoE HLO. Sanity test: `RAY=1 ./submit.sh deepseek3-671b:fp8-test --partition=k8s --nodes=8 -- per_device_batch_size=1 quantization=fp8 quantize_kvcache=false jax_distributed_heartbeat_timeout_seconds=99999` should reach `completed step: 0` without crashing.

**Apply the following differences when `QUANTIZATION != bf16`:**

1. **Peak compute + MFU formula.** MI355's peak rate differs by dtype: BF16 ≈ 2500 TFLOP/s/device (MFU ≈ TFLOP/25); **FP8 ≈ 5000 TFLOP/s/device (MFU ≈ TFLOP/50)**. MaxText's `mfu_tracker.py` picks the correct peak from the `quantization` config, so the log's reported MFU is already against the right peak — no manual rescaling. But the output doc's header + reproduction examples must cite the fp8 peak and the fp8 MFU ceiling.

2. **Passthrough-arg injection.** Append `quantization=<QUANTIZATION>` (and for DS3/kimi-class models that set `base_emb_dim / base_mlp_dim / v_head_dim` to non-power-of-two values, also append `quantize_kvcache=false` unless benchmarking KV quantization separately) to **every row's passthrough flags** in the configs-to-sweep table. Do not modify the config file itself — this is a sweep-time override.

3. **Output doc filename.** Add `-<QUANTIZATION>-` before `pdbs-sweep` in both deliverable filenames. Cross-link from the new doc's header to the bf16 doc: `"See [\`<MODEL_TAG>-pdbs-sweep.md\`](<MODEL_TAG>-pdbs-sweep.md) for the bf16 reference sweep at the same cell set."`

4. **Memory-ceiling expectations.** Weight memory drops ~2× at fp8; activation memory drops ~1.5–2× (forward fp8, backward mostly fp16/bf16 for atomic ops). Expect `max_pdbs` to **shift up by 1.3–1.8×** vs the bf16 sweep of the same model. Extend the pdbs probing ladder to 20+ (or 32+ for dense configs on 1T) as needed — the dynamic probing rule handles this, but pre-emptively queuing pdbs=24 / 32 for dense configs saves compile-retry cycles.

5. **Loss-parity thresholds.** The existing Δ ≤ 0.003 "all configs agree" rule assumes bf16 LSB noise. For fp8 sweeps:
   - **Across dtypes** (fp8 vs bf16 at the same cell): expect Δ ≈ 0.05–0.2 at step 14 due to fp8 rounding noise. This is *expected*, not a numerical bug. Don't flag it as a correctness failure.
   - **Within the fp8 sweep** (v1 vs v2 vs v3 at the same cell): v2/v3's `moe.py` changes are dtype-agnostic — HLO is still forward-bit-identical across variants, so loss should match to Δ ≤ 0.01. Tighter than cross-dtype, looser than bf16 LSB.
   - Replace the "Δ ≤ 0.003" language in takeaway #6 with regime-dependent thresholds.

6. **The DS3 headline question needs restating.** Instead of "does the v1→v2→v3 story replicate?", the fp8 version is: **"does the v1→v2→v3 kernel-elimination chain *survive* the fp8 conversion?"** v3's `custom_vjp` fix replaces duplicate-index scatter-add (which requires bf16/fp32 atomics) with a permutation-gather + reduce-sum (dtype-agnostic). If XLA's fp8 lowering promotes v1's scatter-add to fp16 / bf16 / fp32 *anyway* (because fp8 atomics don't exist on MI355), the v1 baseline kernel may already be cheaper than bf16 — making v3's Python-patch savings smaller or zero. Either outcome is publishable; the answer is worth measuring.

7. **Profile-drill family predicates.** `utils/profile_drill.py`'s kernel-family list (`input_scatter_fusion`, `loop_*_fusion`, RCCL, CK+primus GEMM, flash_attn, …) is dtype-agnostic in naming, but the *distribution* shifts on fp8:
   - `loop_convert_fusion` inflates (fp8 ↔ bf16/fp32 conversions on every boundary)
   - Primus-Turbo GEMM may take a different `primus_turbo::fp8::*` path — the CK+primus predicate should be widened if names change
   - `input_scatter_fusion` may be replaced by or supplemented with `loop_reduce_fusion` if XLA rewrites the atomic into a non-atomic reduce-over-axis

   When drafting the profile drill-down section, first do a per-kernel top-10 listing (the tool already prints it) before committing to the cross-path composition table — the table's row names may need adjustment vs the bf16 doc.

8. **What does NOT change.** Retry policies, OOM-as-hang protocol, MaxText heartbeat hedge, `--time` escalation ladder, NIC / github-flake / cleanup-flake handling, resumability protocol, parallel-streams calibration, the 10-config routing taxonomy, v1/v2/v3 patch-branch names. All dtype-agnostic. **Only the 8 items above are dtype-specific.**

### Parallelism axis: `dcn_expert_parallelism` (DCN-EP)

> **OPT-IN axis, not a default.** Skip this entire section unless the user has explicitly requested a DCN-EP sweep. The default `DCN_EP_VALUES = [1]` (no DCN-EP rows in the deliverable doc) is the right behavior for any first sweep on a new model. See the scope rule in the prompt header for why.

`DCN_EP` is an **in-doc column dimension** within each per-dtype sweep, not a separate-doc axis. Every DCN_EP value for a given dtype lands in the same `<MODEL_TAG>[-<QUANTIZATION>]-pdbs-sweep.md`, as additional rows under each DCN-EP-relevant config.

The base config sets `ici_expert_parallelism=8` × `dcn_fsdp_parallelism=NODES_PER_JOB` × `dcn_expert_parallelism=1` — i.e. expert parallelism is purely intranode (8 GPUs share the EP axis), and FSDP is purely inter-node. Walking `DCN_EP > 1` factorizes a chunk of the inter-node parallelism out of FSDP and into EP, **putting expert dispatch on RDMA instead of FSDP gradient all-reduce**:

| `DCN_EP` | `dcn_fsdp_parallelism` | `dcn_expert_parallelism` | Total EP rank-product | EP fanout per host |
|---:|---:|---:|---:|---|
| **1** *(default)* | `NODES_PER_JOB` (e.g. 8) | 1 | 8 | EP axis is 8 intranode GPUs only |
| 2 | `NODES_PER_JOB/2` (e.g. 4) | 2 | 16 | each host's experts spread to 1 peer host (2-host EP groups) |
| 4 | `NODES_PER_JOB/4` (e.g. 2) | 4 | 32 | each host's experts spread to 3 peer hosts (4-host EP groups) |
| 8 | 1 | `NODES_PER_JOB` (e.g. 8) | 64 | full DCN-EP, no inter-node FSDP |

**What this axis can and can't measure**

The DS3 sweep called this regime out as "where DeepEP would shine but our sweep does not exercise":

> Inter-node EP (RDMA-backed AllToAll). RCCL's AllToAll over RDMA adds round-trip setup and ring/tree overhead that DeepEP's direct RDMA dispatch avoids. Our sweep runs `ici_expert_parallelism=8` (purely intranode); a hypothetical `dcn_expert_parallelism>1` configuration would exercise this.
> — [`deepseek3-671b-pdbs-sweep.md`](deepseek3-671b-pdbs-sweep.md), "Where DeepEP's design already wins"

**Known limitation: the current DeepEP integration in MaxText only supports intranode EP.** `MaxText/pyconfig.py` validates `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` and rejects the config with a pydantic `ValidationError("Internode DeepEP is not yet supported in JAX")` in ~2 min before reaching XLA compile. This applies to **all DeepEP configs** (`sparse-deepep`, `sparse-gmm-deepep` v1/v2/v3) and is a JAX/MaxText integration-layer constraint, not a Primus-Turbo or RCCL limitation. The DS3 v3-vs-fixed margin claim (DeepEP RDMA-dispatch beats RCCL all-to-all-over-RDMA) is therefore **not testable against current MaxText** — measuring it would require either an upstream MaxText fix lifting the validator or a hand-patched `pyconfig.py`.

**The DCN-EP sweep therefore deliberately characterizes only the non-DeepEP regime** (3 dense-cf + `sparse-gmm-fixed`) at DCN_EP > 1. This is still a useful ablation of how inter-node token routing scales with expert parallelism in the path that *does* go through XLA's RCCL all-to-all / ragged-all-to-all lowerings.

**Apply the following differences when `DCN_EP > 1`:**

1. **Passthrough-arg injection on every row's submit:**
   ```
   dcn_expert_parallelism=<DCN_EP> dcn_fsdp_parallelism=<NODES_PER_JOB / DCN_EP>
   ```
   These two flags together are ~50–60 chars in EXP_TAG; reserve passthrough headroom (drop yml-default redundancies — see the [Passthrough hygiene](#passthrough-hygiene--drop-cli-overrides-that-match-defaults) section).

2. **Configs to re-sweep at DCN_EP > 1.** Only 4 of the 10 routing configs can run at DCN_EP > 1 — DeepEP variants are pydantic-validated to require `dcn_expert_parallelism == 1` (see "What this axis can and can't measure" above for the validator wording and the upstream-fix dependency). The actually-feasible sweep at DCN_EP > 1 is therefore:

   | Config | Re-sweep at DCN_EP > 1? | Why |
   |---|---|---|
   | `dense-cf1.25` / `cf2` / `cf4` | YES | dropping path uses regular `all-to-all` for token routing across the EP axis; DCN_EP > 1 puts that on RDMA. Probably SLOWER than DCN_EP=1 since RCCL's RDMA `all-to-all` adds overhead. |
   | `sparse` | SKIP | infeasible at DCN_EP=1 already; won't fit at higher EP either. |
   | `sparse-gmm` (one-shot) | SKIP | OS kernel is intranode-only; meaningless at DCN_EP > 1 (would fall back to kNccl path, duplicating `sparse-gmm-fixed`). |
   | **`sparse-gmm-fixed`** | **YES** | the only dropless config that isn't DeepEP-gated — uses RCCL's `ragged_all_to_all` over RDMA. This was going to be the DeepEP baseline; now it's the *only* dropless DCN-EP row available. |
   | `sparse-deepep` | **BLOCKED (known)** | `use_deepep_dispatch=true` validator requires `dcn_expert_parallelism == 1`. |
   | `sparse-gmm-deepep` v1 | **BLOCKED (known)** | same DeepEP validator. |
   | `sparse-gmm-deepep` v2 | **BLOCKED (known)** | same DeepEP validator. |
   | `sparse-gmm-deepep` v3 | **BLOCKED (known)** | same DeepEP validator — the "DeepEP wins on inter-node EP" question requires upstream MaxText work to test. |

   = 4 DCN-EP-relevant configs × (`|DCN_EP_VALUES|` − 1) non-default values × ~5 pdbs typical ladder. For `DCN_EP_VALUES = [1, 2, 4, 8]` on a 10-config baseline, that's `10 + 4·3 = 22` row-families in the deliverable doc.

   **Implication for sweep planning**: do not queue DeepEP configs at DCN_EP > 1 — they will fail config-validation in ~2 min and waste a queue slot. Stick to the 4 feasible configs above for all non-default DCN_EP rows. The narrower question this sweep *can* answer is "how does non-DeepEP (dropping dense + RCCL-dispatch dropless `sparse-gmm-fixed`) scale with DCN_EP factorization?" — a valid characterization of cross-node token routing in the XLA/RCCL path.

3. **Doc layout inside `<MODEL_TAG>[-<QUANTIZATION>]-pdbs-sweep.md`.** Every results table (TGS / TFLOP/s / step-time / loss) gets a `dcn_ep / dcn_fsdp` column inserted immediately after the config name, before the pdbs columns. Sort order: primary by config (existing order), secondary by DCN_EP ascending. Example shape:

   ```
   | config               | dcn_ep / dcn_fsdp | pdbs=1 | pdbs=2 | pdbs=3 | pdbs=4 | pdbs=5 | pdbs=6 |
   |----------------------|-------------------|-------:|-------:|-------:|-------:|-------:|-------:|
   | dense-cf1.25         | 1/8 *(default)*   | 400    | 500    | …      | …      | …      | …      |
   | dense-cf1.25         | 2/4               | …      | …      | …      | …      | …      | —      |
   | dense-cf1.25         | 4/2               | …      | …      | —      | —      | —      | —      |
   | dense-cf1.25         | 8/1               | …      | —      | —      | —      | —      | —      |
   | sparse               | 1/8 *(default)*   | ✗ OOM  | —      | —      | —      | —      | —      |
   | sparse-gmm           | 1/8 *(default)*   | 249    | —      | —      | —      | —      | —      |
   | sparse-gmm-fixed     | 1/8 *(default)*   | 400    | …      | …      | 614    | —      | —      |
   | sparse-gmm-fixed     | 2/4               | …      | …      | …      | …      | —      | —      |
   | sparse-gmm-fixed     | 4/2               | …      | …      | —      | —      | —      | —      |
   | sparse-gmm-fixed     | 8/1               | …      | —      | —      | —      | —      | —      |
   | …                    | …                 |        |        |        |        |        |        |
   | sgd-v3               | 1/8 *(default)*   | 400    | …      | …      | 620    | 800    | 850    |
   | sgd-v3               | 2/4               | …      | …      | …      | …      | …      | —      |
   | sgd-v3               | 4/2               | …      | …      | —      | —      | —      | —      |
   | sgd-v3               | 8/1               | …      | —      | —      | —      | —      | —      |
   ```

   The feasibility-summary table gains a matching `dcn_ep / dcn_fsdp` column (one row per `(config, DCN_EP)` pair).

4. **Memory ceiling shifts AND TGS-at-fixed-pdbs depend on expert count.** When DCN_EP increases, two memory terms move in opposite directions: (a) **expert weights per rank shrink** — total EP product = `ici_ep × dcn_ep` grows, so each rank holds `num_experts / (ici_ep × dcn_ep)` experts; (b) **non-expert FSDP shard per rank grows** — `fsdp_factor = nodes / dcn_ep` shrinks, so each rank holds a larger attention / shared-expert / embedding / optimizer-state chunk. Net effect depends on the expert-count of the model:

   - **kimi-1T (384 experts):** expert shrinkage dominates. `max_pdbs` is roughly preserved at DCN_EP=2 (e.g. dense-cf1.25 max stays ≥ 12), and **dense-cf1.25 pdbs=4 actually GAINS +6.5% TGS at DCN_EP=2** vs DCN_EP=1 (per-rank expert weights drop from 48 → 24 experts/GPU, freeing HBM for activations and lowering GEMM time enough to outweigh the inter-node `all-to-all` cost).
   - **DS3-671B (256 experts):** expert shrinkage smaller, RDMA cost dominates from the start. dense-cf1.25 monotonically degrades EP=1→8 (867 → 800 → 649 → 538 TGS at pdbs=4 — no DCN_EP=2 bump).
   - **Smaller-expert / attention-heavy models** (e.g. llama3.1-405b MoE, grok-1): non-expert FSDP shard growth likely dominates → `max_pdbs` falls steeply at DCN_EP > 1.

   **Operational rule:** below ~300 experts, expect monotonic TGS degradation with DCN_EP and steeply collapsing `max_pdbs`; above ~300 experts, look for a small-pdbs window where DCN_EP=2 marginally beats DCN_EP=1. Probe each `(config, DCN_EP)` cell fresh; do not assume a direction.
   - Start each DCN_EP > 1 ladder at the DCN_EP=1 `max_pdbs` of the same config. If it fits, probe higher (pdbs+1, pdbs+2, ...) until OOM. If it OOMs, probe lower (pdbs-1, pdbs-2, ...).
   - Previously-infeasible rows (e.g. `sparse-deepep` full-row ✗ at DCN_EP=1) are worth **re-probing at DCN_EP > 1** — the expert-memory reduction may unlock them. (Note: DeepEP variants are validator-blocked at DCN_EP > 1 regardless — see point 2.)

   **Compile-time observations from DS3 + kimi-1T DCN-EP runs:**
   - `DCN_EP=2` cells compile in roughly the same wall time as `DCN_EP=1` (15–25 min typical).
   - **`DCN_EP=4` cells frequently need `--time=90:00`** — observed on DS3 at every (config, pdbs) combination that's near the memory frontier (sparse-gmm-fixed pdbs=1, dense-cf1.25 pdbs=4, dense-cf4 pdbs=6 all hung past 45 min on first attempt then ran cleanly at `--time=90:00`). Default to `--time=90:00` for `DCN_EP=4` probes on 671B-class and larger MoE.
   - `DCN_EP=8` cells compile faster than DCN_EP=4 in our data (the simpler `dcn_fsdp=1` factorization seems to give the rematerializer less to chew on). `--time=60:00` is usually enough.

5. **Diminishing TGS-per-pdbs return at high DCN_EP — adjust ceiling-probe priority.** Dense configs at DCN_EP=1 typically gain +20–40% TGS from pdbs=4 → pdbs=8; at DCN_EP=4–8 the same delta shrinks to +1–10% (DS3 measurements: dense-cf1.25 pdbs=4→8 gain is +37% / +12% / +2.8% / +8.9% at DCN_EP=1/2/4/8; dense-cf2 same shape at +27% / +7.4% / +1.4% / +4.1%). The inter-node `all-to-all` cost becomes the per-step bottleneck and additional pdbs amortizes only the dense compute portion, which is now a small fraction of the step time. **Implication for sweep planning:** at DCN_EP ≥ 4, do NOT spend the full 25-min compile cost on `pdbs=12 / 16` ceiling probes unless you specifically need the absolute maximum-pdbs throughput value. Two cells per `(config, DCN_EP)` — a low-pdbs (e.g. 4) and a mid-pdbs (e.g. 8) — typically capture the curve shape; chasing the ceiling adds compile cost without changing the qualitative picture. Reserve ceiling probes for the DCN_EP=1 baseline and the most operationally relevant DCN_EP value (typically 2).

6. **HLO collective inventory will change.** At DCN_EP > 1, the EP axis spans hosts → XLA emits cross-host `ragged_all_to_all` / `all_to_all` for `sparse-gmm-fixed`, and DeepEP's `moe_dispatch` / `moe_combine` now go RDMA. The drill-down section should carry **one HLO collective table per DCN_EP value** at a chosen reference pdbs (typically the smallest commonly-feasible cell across all DCN_EP values, e.g. pdbs=2 or pdbs=3).

7. **The DS3 v3-vs-fixed margin question cannot be answered at DCN_EP > 1** (see point 2 — DeepEP is pydantic-blocked). The DCN-EP portion of the doc instead documents **"how much does `sparse-gmm-fixed`'s RCCL-RDMA throughput degrade as DCN_EP grows?"** and **"does the dense-cf dropping path fare better or worse under inter-node all-to-all than the dropless RCCL-ragged-a2a path?"**. Report TGS(DCN_EP) curves for each of the 4 feasible configs at a shared feasible pdbs.

8. **Profile drill-down.** Add a supplementary drill-down sub-section for `sparse-gmm-fixed` vs `dense-cf1.25` at each DCN_EP (the dropless-vs-dropping RCCL comparison, at a shared-feasible pdbs). The HLO collective inventory changes at DCN_EP > 1 as the EP axis spans hosts — regenerate the inventory per DCN_EP value.

9. **What does NOT change.** Same 10-config routing taxonomy, same pdbs probing strategy, same retry rules, same OOM-as-hang protocol, same heartbeat hedge, same ceiling extension rule. Only `dcn_expert_parallelism` + `dcn_fsdp_parallelism` passthroughs are added for non-default DCN_EP rows.

**Combinatorial scope of deliverable docs** (DCN_EP collapses into columns, so only dtype × MODEL_TAG drives doc count):

| dtype | deliverable doc pair |
|---|---|
| bf16 | `<MODEL_TAG>-pdbs-sweep.md` / `.zh.md` *(contains all DCN_EP rows)* |
| fp8 | `<MODEL_TAG>-fp8-pdbs-sweep.md` / `.zh.md` *(contains all DCN_EP rows)* |

The user's request "sweep DCN_EP=2/4/8 for sparse-gmm-fixed and MoE configs" is an **in-place extension of the existing bf16 deliverable doc** (add rows, update summary, regenerate drill-down), not a new doc.

### Parallel-streams assignment + calibration (when NUM_STREAMS ≥ 2)

Pinning configs to fixed nodelists creates wall-time imbalance — configs have wildly different ladder lengths (dense-cf1.25 may probe 1 → 16, a DeepEP config may probe 1 → 6), and a pinned-long-ladder stream becomes the critical path while short-ladder streams idle. Instead: **calibrate up front, then schedule freely.**

**Step 1. Partition the cluster.** Split AVAILABLE_NODES into NUM_STREAMS disjoint nodelists `S0, S1, ..., S{NUM_STREAMS-1}`, each of size NODES_PER_JOB (drop any remainder — document excluded nodes in the infra notes).

**Step 2. Calibrate nodelist-to-nodelist variance.** Before starting the real sweep, run a short reference cell on every stream to measure how different the nodelists actually are:
- Reference cell: `dense-cf1.25 pdbs=1 steps=15` with heartbeat hedge. Fast (~6 min after warm cache), representative of compute + interconnect mix, trivially feasible on any node.
- Run the reference cell TWICE on each stream, back-to-back (total: `2 × NUM_STREAMS` jobs, plus one initial cold-compile run; ≈ 20–30 min).
- For each stream `Ss`, record: `tgs_i = step-5-to-14 mean of run i`. Derive `median_tgs[s]` = median of the 2+ runs on `Ss`; `intra_stream_std[s]` = stddev of those runs; `cluster_median_tgs` = median across all streams' medians; `per_stream_factor[s] = median_tgs[s] / cluster_median_tgs`.
- Compute `cross_stream_std = stddev(per_stream_factor[s])` and `typical_intra_std = median(intra_stream_std[s])`.

**Step 3. Classify the regime** based on cross-stream variance (expressed as percentage of TGS):

| `cross_stream_std` vs `typical_intra_std` | Regime | Scheduling strategy |
|---|---|---|
| `cross ≤ max(typical, 1 %)` | **Noise-only** | Fully parallel. Configs roam freely across streams. v1/v2/v3 may be split across streams (kernel-delta headline is not meaningfully biased). No per-cell normalization needed. |
| `cross ∈ (1 %, 5 %]` | **Small-discrepancy** | Fully parallel. Configs roam. BUT: (a) record `per_stream_factor[s]` in the doc's infrastructure notes; (b) for the v1/v2/v3 drill-down (the single most kernel-sensitive measurement), pin v1 + v2 + v3 + their profile runs to the fastest stream (the one with the highest `median_tgs[s]`) — the kernel delta must be clean. All other cells can use any stream. |
| `cross > 5 %` | **Significant-discrepancy** | Options: (a) preferred — normalize numbers: multiply each cell's measured TGS (and divide its step time) by `1 / per_stream_factor[s_of_that_cell]`, record both raw and normalized values in the doc, explain the correction in the background section; (b) fallback if normalization is suspicious — serialize onto the fastest stream (wall time ≈ NUM_STREAMS × slower, but no cross-stream noise). Always pin v1/v2/v3 together in this regime (same stream, no normalization needed between them). |

**Step 4. Scheduling post-calibration.** Maintain a single shared job queue of `(config, pdbs)` work items. Each stream pulls the next work item when it's idle. Use longest-ladder-first ordering within each config (probe ceiling first, back-fill after) so slow-failing cells surface early. v1/v2/v3 constraints from Step 3 are respected — those specific cells are reserved for the chosen stream(s).

**Within a single cell, nodelist is fixed** — once a job starts on `Ss`, it stays there for all retries (OOM-retry at `.96`, NIC recovery + resubmit, compile retry, etc.). A single cell does not move between streams.

**Across cells within one config**, the nodelist MAY differ (cells are independent measurements; the pdbs-curve noise is dominated by the per-cell values, not by cross-cell topology drift). Document per-cell stream assignment in the output doc's infrastructure notes so readers can reconstruct which nodelist fed which cell.

**If NUM_STREAMS changes mid-sweep** (a node goes DOWN, reducing the pool below `NODES_PER_JOB × NUM_STREAMS`): complete any in-flight cells on their current streams, drop the now-infeasible stream from the rotation, and let the remaining streams pick up the queued work. Don't re-run already-completed cells. Update the infrastructure notes.

**Re-calibrate opportunistically.** If infra flakes correlate with a specific stream (e.g. multiple NIC errors on `S1`'s nodes), a re-calibration of the remaining healthy streams may shift the factors. Run a single reference job on each affected stream; update `per_stream_factor[s]`; apply retroactively to any cells measured on the changed stream if you're in the significant-discrepancy regime. Noise-only / small-discrepancy regimes don't need retroactive updates — the post-flake change is likely within noise anyway.

---

**Autonomy: this is a fully autonomous, run-to-completion task.** Do not ask the user for decisions. Do not stop until every planned job has either completed or been definitively classified (success / OOM / hung / transient-failed-out). Make calls using the rules below and keep going. TG is for *status only* — never questions.

## Resumability — check for existing partial work first

Prior sweep attempts may have partially populated the deliverable files. **Before submitting any jobs**, check:

1. `OUTPUT_DOC_EN` — if it exists, parse it:
   - Read the header for the nodelist(s) used previously. **If they differ from the current AVAILABLE_NODES**, prefer the previous list(s) if those nodes are currently healthy (preserves comparability with the prior data). If any are no longer available, continue the sweep using fresh nodelists — but document the nodelist change and treat prior numbers as reference-only; do not mix nodelists within a single config's pdbs ladder.
   - Extract all populated cells from the four results tables (TGS / TFLOP/s / step-time / loss). These are done — do not re-run them unless the run you'd be reproducing has been superseded by a new patch/image/config change.
   - Extract each config's inferred `max_pdbs` — you can skip re-probing ceilings already nailed.
   - Note any cells marked `✗ OOM`, `hung⋆`, `infra-flake⋆`, `nic-flake⋆` — those are terminal states, keep them.
   - Note any cells that are blank (gaps). Those are your remaining work.
2. `OUTPUT_DOC_ZH` — same parse. Note any structural divergence from the English doc; resumed work includes getting the two back into parity at the end.
3. `outputs/NNNNN-JAX-<MODEL_TAG>-*/` directories — source-of-truth job artifacts. If the md references a cell but the corresponding output dir is missing or has no `completed step:` lines, treat the cell as unreliable and re-run it. If output dirs exist for cells NOT in the md, parse them and add those cells to the resumed table.

**Cross-check protocol when resuming:**
- For each populated cell in the md, find the matching output dir by config tag + pdbs + job-id hint if present. Verify the numbers in the md match the log's steps-5-to-14 mean within ±1%. If they match, keep. If they disagree, trust the log and update the md.
- For configs whose `max_pdbs` appears suspicious (e.g., OOM at pdbs=N but pdbs=N+1 wasn't probed), complete the missing ±1 probe to nail the ceiling.
- Recompute `P★` after all ceilings are confirmed — it may shift if new data is added.

**TG kickoff (Milestone 0.5) must summarize what was found and what remains:**
```
[<MODEL_TAG> M0.5/10] Resuming prior sweep. Found <MODEL_TAG>-pdbs-sweep.md with 7/10 configs populated
(dense/sparse/sparse-gmm/sparse-gmm-fixed done; deepep v1/v2/v3 missing).
Nodelist match ✓. P★ so far = 6 (dense-cf4 ceiling). Remaining: 3 configs × ~6 pdbs ≈ 18 cells.
Next: pre-sweep health check, then deepep-v1 probe. ETA 6-9 h.
```

If no prior files exist, proceed to the normal M0 health check + M1 dry run + full sweep.

Do NOT delete or rewrite prior md content without preserving the data. The goal is to extend, not restart. If the prior doc's structure/section names differ from this prompt's spec, migrate to the current spec while keeping all numerical data intact (copy cells over, don't drop them).

## Pre-sweep: node health check + best-effort fix

Before submitting any benchmark jobs, run a quick health check across all AVAILABLE_NODES. The sweep runs for many hours on these nodelists, and a degraded node at t=0 will cost multi-hour retry loops later. Fix what you can up-front.

For each node in AVAILABLE_NODES:

1. `sinfo -N -n <node> -h -o '%t'` — expect `idle` / `mix` / `alloc` (others running isn't unhealthy). Flag `down`, `drain`, `fail`, `*` states as unhealthy.
2. `ssh <node> "hostname && uptime"` — confirms reachability.
3. `ssh <node> "rocm-smi --showid"` / `rocm-smi -a 2>&1 | head` — confirms all 8 GPUs respond.
4. `ssh <node> "ibstat | head -40"` (or equivalent for the AINIC/ionic stack) — confirms IB/NIC up, GID assigned, link active.
5. (Optional) `ssh <node> "dmesg -T | tail -40"` — scan for recent NIC/GPU error lines.

For any node flagged unhealthy, attempt best-effort in-place recovery via host-cmd, rotating through commands until one resolves or all fail:

- `ssh <node> "ip link set <ib_dev> down; sleep 2; ip link set <ib_dev> up"`
- `ssh <node> "echo 1 > /sys/class/infiniband/<dev>/device/reset"`
- `ssh <node> "rdma link show"` then whatever `rdma link ... reset` invocation applies
- `ssh <node> "systemctl restart openibd"` (likely policy-blocked — try last)
- `ssh <node> "rmmod ionic; modprobe ionic"` (if the Pensando ionic driver is involved)

We're root on the cluster, so `sudo` is unnecessary. **However**, host-cmd's policy at `/maxtext-slurm/.host-cmd/policy.json.default` blocks some driver-level commands (systemctl restart, sudo literal, modprobe patterns, etc.). If a reload command is rejected by policy, TG-tell the user the exact command they could run manually and **move on** — don't loop on the same blocked command.

After each recovery attempt, re-run the health checks. Record the outcome:
- ✅ Node recovered → include in the AVAILABLE_NODES pool as normal.
- ❌ Node not recovered → **remove it from AVAILABLE_NODES** (recompute NUM_STREAMS accordingly) and TG the user the node id + error signature + any blocked/failed reload commands so they can fix it manually. The sweep proceeds with the remaining healthy nodes.

(Note: pre-sweep is the only phase where unhealthy nodes are REMOVED from consideration. Once the sweep starts, **nodelist stability within a config is load-bearing** — see below.)

After the first real job runs on each node (dry-run or calibration), grep its log for `[INFO] <node>: NCCL_IB_GID_INDEX=<N> (auto-detected, RoCEv2-preferred)` — this confirms `detect_nccl_env.sh` is working and gives the actual GID index used per node. If any node instead produces `[WARN] ... inconsistent routable GID indices across ACTIVE ports`, that node is running on the fallback `NCCL_IB_GID_INDEX=1`; note the node in the infra log. It may or may not be the actual routed index — RCCL-init failures on that specific node later in the sweep should trigger a manual `_env_NCCL_IB_GID_INDEX=<right_value>` override (see the RCCL-init hang decision rule).

Send a kickoff TG after the health check, summarizing: nodes healthy, nodes that needed recovery, nodes still flagged, any GID-WARN nodes, resulting NUM_STREAMS, stream-to-config assignment.

## Monitoring loop — do not go idle

**Be actively watching every running job at all times.** If NUM_STREAMS > 1, interleave monitoring across streams — each running job gets a poll pass before sleeping. Standing rules per job:

- **Pending**: poll `squeue -j <jobid>` every **60 s**. If stuck PENDING with `(Resources)` >1 h, send a TG heartbeat and keep waiting — do not switch to other nodes.
- **Running, pre-step-0**: poll log mtime AND log size every **60 s**. If log silent >3 min after `BARRIER: Synchronizing hosts before training loop` without step 0, apply RCCL-init / OOM-hang retry logic (see decision rules). **If log size grows past ~500 MB while still pre-step-0**, apply the `runaway-log` rule (scancel + cleanup + retry; the binary garbage cannot be parsed for a real error — read the ray-actor `*.err` files instead).
- **Running, mid-training**: poll log every **90 s** for new `completed step:` lines. If mtime stale >3 min between steps, apply hang logic.
- **Completed**: within **30 s** of completion, parse the log for step-5-to-14 metrics, record, launch the next queued job on that stream (never let a stream's nodelist idle >1 min when work remains).
- **Watch for infra error strings** on every poll: `NET/IB` / `ibv_` failures, `ncclInternalError`, `heartbeat timeout`, `stop sending heartbeats`, `NodeFailed`, `CUDA_ERROR`, `hipErrorOutOfMemory` (distinct from XLA OOM), `DRAIN` / `DOWN` node state. Apply transient-infra rules below.

No passive long waits. Even during long compile phases, poll at ~60 s cadence — compile that's legitimately progressing shows log mtime advancing. If mtime stalls >5 min during compile, apply compile-hang rule.

**Run-to-completion contract:** the sweep is "done" only when every config has been probed up to its OOM ceiling AND the comparison row at P★ is fully populated. Every attempted cell gets a definitive result: a number, `✗ OOM-<source>`, `hung@stepN`, `compile-timeout`, `infra-flake⋆`, or `nic-flake⋆`. No early termination.

## pdbs probing strategy — dynamic ceiling discovery

**No predetermined pdbs ladder.** For each config, walk pdbs upward until OOM. Record every successful pdbs. The goal is to find each config's memory ceiling — that's the primary data deliverable, especially for large models.

Recommended step sequence: start at **pdbs=1**, then double-ish (2, 4, 6, 8, 10, 12, 16 …) until the first OOM. Once OOM'd at pdbs=N, back off and probe with **+1 steps** in the feasibility zone (N-2, N-1 if not already run) to nail down the exact ceiling. Total per column: typically 4–8 successful runs + 1 OOM.

Once all configs have been probed to their individual max, compute **P★ = min(max_pdbs) across all feasible configs**. At P★ every config has a data point by construction — so the table row at `pdbs = P★` is the fully-populated apples-to-apples cross-config comparison row. Call it out explicitly in the key takeaways ("At the common feasibility point pdbs=P★, config X leads at Y TGS vs baseline Z").

Memory is monotonic in pdbs — once a config OOMs at pdbs=N, do not probe pdbs > N for that config.

## Progress reporting — Telegram (status-only, one-way)

Expected wall time scales with NUM_STREAMS: serial (1 stream) = 20–40 h; 2 streams ≈ 12–22 h; 4 streams ≈ 8–12 h (subject to contention).

Send periodic TG updates. Follow `@maxtext-slurm/skills/telegram/SKILL.md`.

**Before anything else**, verify TG:
```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "test -f ~/.tg_config && echo EXISTS || echo NOT_FOUND"
```
If `NOT_FOUND`, log a warning, skip TG, continue. Never block on TG.

**TG is status-only. Never ask a question. Never wait for a reply.**

**Milestones:**

0. **Pre-sweep health check done** — nodes checked, recoveries attempted, any flagged for manual user fix, resulting AVAILABLE_NODES + NUM_STREAMS + stream partitioning.
0.5 **Resumability check done** — what prior md data was found (cells populated, ceilings known, gaps remaining) and what fresh work scope is. Skip if no prior files exist.
0.75 **Calibration done** (only if NUM_STREAMS ≥ 2) — per-stream median TGS, cross-stream variance, regime (noise-only / small-discrepancy / significant), and resulting scheduling strategy. TG message includes `per_stream_factor[s]` for each stream.
1. **Kickoff** — plan summary: 10 configs, probing strategy, stream/scheduling strategy, est wall time for remaining work.
2. **First config ceiling found** — e.g., `dense-cf1.25 max=N` with peak TGS/MFU.
3. **Dense ceilings known** — all three dense-cfN max pdbs + peak numbers.
4. **Sparse non-DeepEP ceilings known** — `sparse`, `sparse-gmm`, `sparse-gmm-fixed`, `sparse-deepep`.
5. **DeepEP v1 → v2 → v3 ceilings known** — headline; whether DS3 gain shape replicates.
6. **P★ identified** — common-feasibility row pdbs; which config wins at P★.
7. **All data collected** — every ceiling nailed to ±1 pdbs.
8. **Draft doc ready** — `.md` committed.
9. **Profile drill-down done** (runs only after M7 — every ceiling final) — whether v1→v3 gain is the same kernel story as DS3 + where the dense-vs-sparse gap actually lives at P★.
10. **Final done** — `.md` + `.zh.md` complete, cluster-cost summary.

**Also one-line TG (FYI, not a question) for:**
- Each config's ceiling discovery (max pdbs + peak TGS/MFU, next action)
- Any non-default policy applied (e.g. "pdbs=N dense-cf2: retried at MEM_FRACTION=.96, now ✓ X TGS")
- Any cell given up on after retries (include the failure class: OOM-<source> / infra / hung / timeout / nic-flake)
- **Any NIC error requiring a reload attempt** — TG: what command was tried, whether it succeeded, recovered vs `nic-flake⋆`. If policy blocked the command, TG the manual command for the user.
- **Hourly heartbeat** if no milestone has landed in the last hour.

**Message style:** concise, numbers-first, "what I did + what I'm doing next":
```
[<MODEL_TAG> M5/10] DeepEP v1/v2/v3 ceilings: v1 max=6 @800TGS, v2 max=7 @970TGS, v3 max=8 @1110TGS.
DS3 shape replicates ✓. P★ = 6 (dense-cf4 is limit). Next: v3 profile @ pdbs=8. ETA 55 min.
```
No log dumps. No trailing questions.

## Autonomous decision rules

**OOM — multiple sources, check the actual crash signature:** "OOM" is not a single log pattern. The clean (non-hang) OOM family includes at least five distinct crashes, and the right response depends on which one fired. When a job exits non-zero or terminates mid-run, grep the log for the following in order of priority; classify by the FIRST match:

| # | Log signature | Source | Meaning + response |
|---|--------------|--------|-------------------|
| 1 | `RESOURCE_EXHAUSTED: ... bytes` (often in `_stream_executor_internal.cc` / `memory_allocator.cc`) | XLA pool | **Model-memory OOM.** XLA's managed pool (default `.93` of HBM) can't fit the working set. Ceiling for this config = pdbs−1. Retry ONCE at `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.96` only if: (a) alloc size in error is within 10% of `.93`'s pool (~267.8 GiB on 288 GB HBM), AND (b) previous pdbs succeeded at `.93`, AND (c) `.96` hasn't already starved RCCL for this config at a lower pdbs. If `.96` retry succeeds, record + footnote `"requires MEM_FRACTION=.96"`. If `.96` OOMs with a different signature — see #2 — ceiling confirmed. |
| 2 | `NCCL WARN .*alloc.*out of memory` / `CUDA failure 'out of memory'` in NCCL/RCCL stack / `ncclSystemError` / `alloc.h:\d+` with OOM | RCCL | **RCCL buffer OOM.** RCCL couldn't allocate its per-process peer-access buffers (typically ~5–15 GiB/GPU) because XLA's pool grabbed too much. This is the FLIP SIDE of #1: raising MEM_FRACTION to fit XLA starves RCCL. Response: if you got here via `.96` retry of #1, that confirms the config is unfeasible at this pdbs on this hardware — both XLA OOM at `.93` and RCCL OOM at `.96` → ceiling is pdbs−1, mark `✗` with both errors in the footnote. If you got here at `.93` unexpectedly, retry once at `.93` — could be a transient RCCL init race; if repeats, ceiling. |
| 3 | `HIP_ERROR_OUT_OF_MEMORY` / `hipErrorOutOfMemory` / `hipMalloc .* failed` | HIP runtime | **HIP-level HBM OOM.** Some allocation outside XLA's managed pool failed. Usually correlates with the ceiling. Retry once on same config/pdbs with no MEM_FRACTION change. If it repeats, `✗ HIP-OOM`. If it succeeds, was transient; record. |
| 4 | `std::bad_alloc` / `MemoryError` / Python `MemoryError: Unable to allocate` | host RAM | **Host (CPU) memory OOM.** Usually a compile-time issue — XLA's compile is RAM-heavy. Not a per-pdbs ceiling signal; can happen at any pdbs if a node is under memory pressure. Retry once. If it repeats on the same node, TG-notify the user and continue (don't remove the node). If it repeats on a different node too, treat as `infra-flake⋆`. |
| 5 | `OOMKilled` / container exited with `137` / `signal 9` + cgroup memory message | container/cgroup OOM-kill | **Container memory limit hit** (less common here). Retry once; if repeats, `infra-flake⋆`. |

Any other non-zero exit with a crash that's NOT in the above list: record, treat as `crashed⋆`, follow the general retry rule (retry once; if repeats, terminal).

Record max_pdbs for the config the first time an OOM-family crash happens at pdbs=N — that's the ceiling signal. Do not probe pdbs > N for that config.

**Monotonic pruning — within one config vs across v1/v2/v3:**

- **Within one config** (safe): once config X OOMs at pdbs=N, you may cancel any queued pdbs>N for X without waiting to observe each one OOM. Memory is monotonic in pdbs for a fixed HLO.
- **Across v1/v2/v3** (NOT safe to assume): even though v1 theoretically holds more scatter-add intermediate tensors than v2, and v2 than v3, the per-variant peak memory is model-dependent. Kimi-K2-1T's observed ceilings are `{v1: 5, v2: 5, v3: 7}` — v1 and v2 tie despite the intermediate-tensor difference, and v3 extends by 2 pdbs. **Always probe each variant's ceiling directly with a cell at pdbs = `(other variant's confirmed max) + 1`.** Only cancel queued pdbs>N on variant X *after* X itself has produced an OOM at pdbs=N. Pre-emptive cancellation of variant X's higher pdbs based on variant Y's OOM is an unverified assumption; if the user flags it, the probe must be re-added.

**No pre-emptive ceiling-cell pruning before direct observation.** In particular, do NOT cancel queued `v1 pdbs=N` / `v2 pdbs=N` just because `v2 pdbs=M` OOMed for M ≤ N — those are across-variant inferences. This rule is about data integrity: a single ceiling observation per variant is weak evidence; the user (or reader) may reasonably want to see the direct OOM on each variant.

**OOM-as-hang (critical):** a memory ceiling does NOT always manifest as a clean crash. At pdbs near the ceiling, XLA allocation or compile may silently hang — waiting on a signal from a GPU kernel that never returns, or stuck in a slow HBM allocator loop — without ever printing `RESOURCE_EXHAUSTED`. Do not immediately classify these as OOM; also do not immediately classify them as transient. Use the following detection protocol when a job hangs (pre-step-0 silence >3 min, or mid-training mtime stale >3 min, or compile mtime stale >5 min):

1. **Check head-node CPU before scancelling.** `ssh <head_node> 'top -b -n 1 -o %CPU | head -10'`. If python is at >200% CPU with growing TIME+ (e.g. 64 min CPU time at 23 min wall), XLA is actively re-materializing — the process is NOT deadlocked, just slow. This pattern at `pdbs > last_successful` is still an OOM-hang suspect (XLA's rematerialization heuristic spins longer as HBM headroom shrinks), but the "hang" is really "legitimate compile taking pathologically long because memory is tight". Either way, proceed to step 2; just know the process isn't deadlocked.
2. `scancel` the job.
3. Check: is this pdbs at or above `last_successful_pdbs_for_this_config + 1`? If YES, this is an **OOM-hang suspect** — memory pressure is plausible at this pdbs. If NO (i.e. a lower pdbs than one previously succeeded for this same config), it's NOT an OOM-hang; apply the normal transient-infra retry rules.
4. For OOM-hang suspects, **retry ONCE** with the same config and pdbs on the same nodelist, with a **bumped wall budget** (`--time=60:00` or `--time=90:00` if you already tried 45:00) so a slow-but-real compile has headroom to finish. Include the heartbeat hedge (`jax_distributed_heartbeat_timeout_seconds=99999`) so you don't confuse a transient heartbeat trip with a real hang.
5. If the retry:
   - **Succeeds** → record the cell's metrics; was a slow-compile transient. Proceed.
   - **Hangs the same way** (same phase, same approximate time-to-hang) → confirmed OOM-hang. Mark cell `✗ OOM-hang (compile)` or `✗ OOM-hang (step-0-alloc)` or `✗ OOM-hang (step-1)` depending on where it stuck. Record max_pdbs = (pdbs − 1). Do NOT probe higher pdbs for this config.
   - **Shows a different failure mode** (e.g. infra signature: NCCL/NIC/heartbeat) → apply that specific rule; do not count toward OOM-hang detection.

**RCCL-init hang** (no step 0 within 3 min after barrier): first check OOM-hang criterion — is this pdbs > last_successful for this config? If YES, treat as OOM-hang suspect per above. If NO, check the log for `[WARN] <node>: inconsistent routable GID indices across ACTIVE ports` — that's `detect_nccl_env.sh`'s GID-autodetect bailing on that node, and the fallback `NCCL_IB_GID_INDEX=1` may not match the real routed-GID slot, which can cause RCCL init to hang or crash. If present, TG the user the affected node + resubmit with `_env_NCCL_IB_GID_INDEX=<index>` manually set (inspect the node's `/sys/class/infiniband/*/ports/*/gids/` to find the routed index). If neither OOM-hang nor GID-WARN applies, it's a genuine RCCL-init flake: `scancel` + resubmit with same nodelist. Max **2 retries**. Then `hung⋆ RCCL-init`, move on.

**NIC / IB error** (log shows `NCCL WARN NET/IB`, `ibv_post_send failed`, `remote connection lost`, `IB link down`, `ionic` driver error, or similar):

1. Identify the failing node from the log (look for `<node>:` in the NCCL error line).
2. `scancel` the job.
3. Attempt in-place NIC recovery on the affected node via host-cmd, rotating through commands (same list as the pre-sweep health check). Stop at the first that clears the error.
4. Resubmit **with the same stream nodelist that was running this cell** — do NOT add `--exclude=`. Per-cell nodelist identity is required during retries.
5. Max **3 retries** (reload + resubmit cycles). After each retry, log the recovery command + outcome, TG one-line status.
6. If all 3 retries fail on the same cell, mark `nic-flake⋆` with a footnote listing every command attempted and every policy-blocked command. Release the cell's stream back to the scheduler; the NEXT queued cell picks it up (NIC errors are often transient for the next compile / allocation pattern).
7. If the same node keeps failing across ≥3 distinct cells in a row on its stream, TG-notify with: node, error pattern, reload command(s) the user could try manually. Optionally drop that stream from the rotation if all its nodes become chronic failures; other streams keep pulling work from the queue.

**Other transient infra errors** (NCCL internal error not tied to a specific NIC/IB, heartbeat timeout, node drain/fail, CUDA error not matching XLA OOM): `scancel` + resubmit with same nodelist. Max **3 retries**. Then `infra-flake⋆`, move on.

**Mid-training hang** (silent >3 min after some steps, **no** transient-infra signature in log): first check OOM-hang criterion — if this pdbs > last_successful AND hang is at early step (0 or 1), route to OOM-hang rule. Otherwise: `scancel`, mark `hung@step=N`, no retry (mid-training hangs without infra signatures are typically deterministic — retrying wastes the ~25 min compile time).

**Compile-phase hang** (mtime stale >5 min, no steps, no error): first check OOM-hang criterion — if pdbs > last_successful for this config, very likely OOM-hang (XLA memory scheduler spinning). Route to OOM-hang rule. Otherwise: `scancel`, mark `compile-hang`, bump `--time=40:00` for remaining pdbs of that config.

**Compile timeout** (hit `--time` without 15 steps): mark `compile-timeout`, escalate the wall budget one step and retry. Ladder: `25:00 → 45:00 → 60:00 → 90:00 → compile-timeout⋆`. **XLA compile time for the same HLO is highly non-deterministic** — kimi-k2-1t `sgd-v3 pdbs=5` took >45 min on two attempts and ~5 min on one lucky attempt (same cell, same nodelist, same image). Do not conclude `compile-timeout⋆` until you've tried at least `--time=60:00`. For large sparse/DeepEP cells on 1T-class models, **default to `--time=45:00`** up front — `25:00` is only safe for small dense cells.

**Cluster contention** (stream's nodelist held by another user): queue jobs, wait with hourly heartbeats. Never scancel others'. Do not switch to other nodes — stay on the assigned nodelist.

**v2 or v3 loss deviates from v1 by >0.005 at any step:** log as doc takeaway, continue.

**v2 or v3 regression vs v1:** log with dedicated takeaway, continue — negative result is publishable.

**Unexpected profile result:** document whatever you see. If breakdown materially differs from DS3, add a new drill-down section.

**Anything else:** make the call, document it, TG one line "did X because Y, proceeding".

### Crash / hang / flake quick reference

**OOM crashes** — classified by source in the main "OOM — multiple sources" rule above (XLA pool / RCCL / HIP / host / cgroup). Follow that table.

**Non-OOM transient / hang** patterns:

| Log pattern | Class | Action |
|---|---|---|
| Silent hang at `pdbs > last_successful` with no clean error | OOM-hang suspect | Retry once with heartbeat hedge. If retry hangs same way, confirm OOM-hang, mark `✗ OOM-hang (phase)`, ceiling = pdbs−1. If retry succeeds or shows different failure, handle per that rule. |
| Silent hang at `pdbs ≤ last_successful` with no clean error | transient or deterministic hang | Check tail of log for infra substrings. Present → transient class. Absent → `hung@step=N`, no retry. |
| `[WARN] <node>: inconsistent routable GID indices across ACTIVE ports` | GID autodetect bail | `detect_nccl_env.sh` found conflicting GID layouts across the node's ACTIVE ports and left `NCCL_IB_GID_INDEX` unset; `train_env.sh`'s fallback `1` is now in effect. If RCCL init subsequently works, continue (no action). If RCCL init fails on that node, inspect `/sys/class/infiniband/*/ports/*/gids/` on it to find the routed index, then retry with `_env_NCCL_IB_GID_INDEX=<index>` for that cell. TG the user with the node + suggested override. |
| `NCCL WARN NET/IB` / `ibv_post_send failed` / `remote connection lost` / `IB link down` / `ionic` errors | NIC | Identify failing node, attempt in-place reload, resubmit **same nodelist** for this config. Max 3 retries. If policy blocks reloads, TG manual command. `nic-flake⋆` after retries. Never `--exclude`. |
| `ncclInternalError` / `ncclUnhandledCudaError` (no NIC signature, no OOM) | NCCL transient | retry (max 3) on same nodelist |
| `CoordinationServiceError: The tasks have crashed` / `RPC: /tensorflow.CoordinationService/PollForError` mid-training (no OOM, no NIC error visible in any rank's worker err) | JAX coordination flake | One rank's actor crashed for an opaque reason (often non-deterministic NCCL-init side effect at high DCN_EP). retry (max 2) on same nodelist. If repeats after 2 retries → `crashed⋆`. |
| `heartbeat timeout` / `stop sending heartbeats` / `missed heartbeat` | heartbeat | retry (max 3); add `jax_distributed_heartbeat_timeout_seconds=99999` on retry if cold-compile cause |
| `NodeFail` / node goes `DRAIN` / `DOWN` during job | Slurm infra | `scancel`, wait for Slurm recovery, resubmit same nodelist. If node stays DOWN >1 h, TG-notify, keep waiting. |
| `Segmentation fault` / coredump / unknown non-OOM crash | unknown | retry once; if repeats, `crashed⋆`. |
| `remote: Internal Server Error` / `RPC failed; HTTP 5xx` / `fatal: unable to access 'https://github.com/.../.git'` during MaxText patch-branch checkout | `github-flake` | Global outage, not a cluster issue. `scancel`, wait 5 min, retry. If GitHub still down, wait 15 min and retry. Hourly thereafter. No max retries. Confirm recovery by checking that a newly-submitted job gets past the `git fetch` / `git checkout` step — if yes, bulk-resubmit every job that died with this signature in the outage window. |
| `ActorUnschedulableError: The node specified via NodeAffinitySchedulingStrategy doesn't exist any more or is infeasible` within ~90 s of job start, no NIC error in log | downstream of `github-flake` on a peer rank | **Grep the whole log for GitHub 500 signatures first**. If present on ANY rank → reclassify as `github-flake` and retry per that rule. If absent → unknown Ray scheduling race, retry once. |
| `completed step: <T-1>` present AND `JOB SUMMARY … Status: FAILED` with the failure message after the last step (e.g. `_train_rc=1` on rank N during teardown, or `Error response from daemon: removal of container ... is already in progress`) | `cleanup-flake` | **Not a real failure.** All training steps completed; the data is valid. Record the cell's metrics normally. Footnote in infra notes if it repeats on the same node — may indicate a Docker teardown race, but training data is intact. |
| Slurm log file balloons to **multi-GB** (e.g. >500 MB after <10 min, no `completed step:` lines yet) — content is mostly binary garbage / NUL bytes / repeated raw memory dumps prefixed `(pid=, ip=...)` from a Ray actor; some Python/JAX subproc crashed in a way that's flushing huge buffers to stderr through Ray's stdout-capture and into the slurm log | `runaway-log` | **`scancel` immediately** to stop disk burn (NFS/Vast usage), wait for `COMPLETING` to finish, then **delete both the `<jobid>-*.log` file AND the `<jobid>-*/` job dir** (use `cd outputs && rm -r <relative>` — `rm -rf /mnt/...` triggers a destructive-command policy block). Resubmit the same cell once. If the corruption recurs at the same `(config, pdbs, DCN_EP)`, mark `runaway-log⋆` and skip; otherwise treat as transient. Cause is not always identifiable from the log itself (it's overwritten by garbage); the ray-actor `worker-*.err` under `<jobid>-*/ray_logs/<host>/` usually has the actual underlying error in clean text. |

**Disambiguation priority when something fails:**

1. **Step 14 (or steps-1) present on rank 0** → training finished; this is a `cleanup-flake` at worst. Data is valid. Do not waste a retry. The "success" criterion is `last_completed_step >= steps-1`, NOT `JOB SUMMARY Status: SUCCESS`.
2. **Slurm log file >500 MB while still pre-step-0** → `runaway-log`, scancel + cleanup + retry per the rule above. Don't try to read the log directly (it's mostly binary garbage); use the ray-actor `*.err` files instead.
3. **Clean OOM-family crash in the log** → classify by source; record ceiling if that's the interpretation.
4. **GitHub 500 signature during patch-branch checkout** → `github-flake`, retry after outage clears.
5. **Silent hang at pdbs > last_successful** → OOM-hang suspect, apply OOM-hang retry.
6. **Silent hang at pdbs ≤ last_successful** → transient or deterministic; check log for infra substrings.
7. **Clean non-OOM crash** → transient infra, apply retry policy.

Always grep the full log for `completed step: N` (where N = `steps-1`), OOM-family substrings, and GitHub-500 substrings FIRST before declaring "silent hang" — a crash that happens to leave log mtime stale right before exit can look like a hang if you only check mtime freshness.

## Hedge against heartbeat flakes during cold compile

**Always include `jax_distributed_heartbeat_timeout_seconds=99999` on every submission.** Two reasons:

1. **MaxText's default `jax_distributed_heartbeat_timeout_seconds` is 100 s**, not JAX's 300 s. That's tighter than any 1T-class cold compile can fit in. Every cell, not just the first, will trip it if compile exceeds 100 s.
2. **JAX has no persistent cross-job compile cache** in this repo — `JAX_COMPILATION_CACHE_DIR` is commented out in `train_env.sh`. Every submit is a cold compile. There is no such thing as a "subsequent pdbs with warm cache" for this sweep.

The uglier output dir is worth not losing hours to a silent heartbeat kill at pdbs=N+1 of an already-working config (kimi-k2-1t dense-cf2 pdbs=5 lost to this — dense-cf2 pdbs=1/2 had already succeeded). If `JAX_COMPILATION_CACHE_DIR` is ever wired into `train_env.sh` as persistent, revisit this rule.

**Critical: pass it as a bare flag, NOT with the `_env_` prefix.** `_train.sh` extracts `_env_KEY=VAL` into a shell env var and removes it from PASSTHROUGH_ARGS before MaxText sees it — but `jax_distributed_heartbeat_timeout_seconds` is a MaxText config knob (read from `raw_keys` in `max_utils.py`, default `100` in `base.yml`), NOT an env var that JAX or anything else reads. So `_env_jax_distributed_heartbeat_timeout_seconds=99999` exports a no-op env var while leaving the MaxText config at the 100 s default — silently. Heartbeat trips on the first 100+ s stall (typical: a grain backpressure stall on real-data loss tests, a slow cold-compile, etc.) and the entire job is killed. Verified failure mode on DS3-671B at pdbs=6: jobs ran 17 min – 5.7 h before the first stall caught them.

Right form (plain CLI arg → MaxText config override → JAX uses 99999 s):

```
jax_distributed_heartbeat_timeout_seconds=99999
```

Wrong form (extracted to a no-op env var, MaxText still uses 100 s default):

```
_env_jax_distributed_heartbeat_timeout_seconds=99999  ← DO NOT USE
```

## Deliverables

1. `OUTPUT_DOC_EN` (= `@maxtext-slurm/<MODEL_TAG>-pdbs-sweep.md`)
2. `OUTPUT_DOC_ZH` (= `@maxtext-slurm/<MODEL_TAG>-pdbs-sweep.zh.md`)
3. **Cross-path profile drill-down at pdbs=P★** — five-config comparison table (`dense-cf1.25`, `sparse-gmm-fixed`, `sparse-gmm-deepep` v1, `-v2`, `-v3`) of per-kernel-family time + total-kernel-time + idle-gap, mirroring DS3's structure. The v1→v2→v3 columns tell the DeepEP optimization-chain story within the same table; dense + sparse-gmm-fixed expose dense-vs-sparse gap, MoE-machinery cost, and remaining optimization headroom for DeepEP vs the best non-DeepEP path.
4. **(Optional) Supplementary v1/v2/v3-only drill-down at `min(v1_max, v2_max, v3_max)`** — run this additional drill-down only if `min(v1_max, v2_max, v3_max) > P★`, to capture the optimization-chain story at peak DeepEP throughput where dense/fixed may already be OOM. If `min(v1_max, v2_max, v3_max) == P★`, the five-config table at P★ already covers this — skip.

## Reference materials — read first

- `@maxtext-slurm/deepseek3-671b-pdbs-sweep.md` — gold-standard template, especially v1/v2/v3 drill-down
- `@maxtext-slurm/deepseek3-671b-pdbs-sweep.zh.md` — Chinese format reference
- `CONFIG_FILE` (e.g. `@maxtext-slurm/configs/<MODEL_TAG>.gpu.yml`)
- `MODEL_ENV_FILE` if applicable (e.g. `@maxtext-slurm/configs/<MODEL_TAG>.env.sh`)
- `@maxtext-slurm/configs/deepseek3-671b.gpu.yml` — sibling for comparison
- `@maxtext-slurm/container_env.sh` — default `DOCKER_IMAGE` and `MAXTEXT_PATCH_BRANCH` live here
- `@maxtext-slurm/.host-cmd/policy.json.default` — deny patterns; check this if a recovery command is rejected
- `@maxtext-slurm/CLAUDE.md`
- `@maxtext-slurm/skills/batch-sweep/SKILL.md`
- `@maxtext-slurm/skills/model-config-guide/SKILL.md`
- `@maxtext-slurm/skills/performance-analysis/SKILL.md` — note the TraceLens CSV divisor bug
- `@maxtext-slurm/skills/profile-drill/SKILL.md` — ground truth
- `@maxtext-slurm/skills/job-log-triage/SKILL.md`
- `@maxtext-slurm/skills/tsdb-diagnosis/SKILL.md` — cross-check node health if infra flakes cluster on specific nodes
- `@maxtext-slurm/skills/telegram/SKILL.md` — send-only

## Environment

- Possibly inside a Docker container. `/mnt/vast/` may not be bind-mounted. Ping host-cmd first; route cluster ops through it if so.
- Use only nodes in `PARTITION`. Per-config nodelist stability is load-bearing (see "Parallel-streams assignment").
- You're root on the host. Host-cmd runs commands as root on the target node, so no `sudo` is needed. **However**, the host-cmd policy at `/maxtext-slurm/.host-cmd/policy.json.default` blocks some driver-level commands (e.g. `systemctl restart`, `modprobe`, `sudo` literal token, `mount`/`umount`, etc.). For NIC reloads, try non-systemctl variants first; if policy blocks the command you need, TG-tell the user the exact command and continue best-effort.
- Base image default in `container_env.sh`: `/mnt/vast/yihuang/deepep-gmm-maxtext-v26.2.tar`.

## Submit-time env-var vs passthrough — important distinction

`./submit.sh model:tag ... -- <passthrough_args>` has **two** places to pass configuration, and the difference matters for output-path hygiene:

- **Submit-time env-var prefix** (before `./submit.sh`): e.g. `MAXTEXT_PATCH_BRANCH=... DOCKER_IMAGE=... ./submit.sh ...`. Read by `container_env.sh`, **NOT** included in the job tag / output directory name.
- **Passthrough args** (after `--`): e.g. `per_device_batch_size=4 sparse_matmul=true`. Appended to the job tag AND baked into the output directory name.
- **`_env_KEY=VAL` passthrough** (after `--`): treated both as a container env var AND as part of the job tag — creates very long output dirs.

**For the DeepEP patch-branch selector, always use the env-var prefix, NOT the `_env_` passthrough.** Compare:

Bad (passthrough):
```bash
./submit.sh <MODEL_TAG>:... -- ... _env_MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v2
# → outputs/NNNNN-JAX-...-_env_MAXTEXT_PATCH_BRANCH_yihuang_moe-turbo-gmm-and-deepep-v2/
```
Good (env-var prefix; only override what differs from `container_env.sh` defaults):
```bash
MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v2 ./submit.sh <MODEL_TAG>:sgd-deepep-v2 ... -- ...
# → outputs/NNNNN-JAX-<MODEL_TAG>-sgd-deepep-v2-.../
```

For **v3 runs you don't need any env-var prefix** — `container_env.sh` now defaults to `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3`. For v1/v2 you DO need the explicit override (see the table below). `DOCKER_IMAGE=...` is also defaulted in `container_env.sh`; only override if testing a different image.

### `DOCKER_IMAGE` and `MAXTEXT_PATCH_BRANCH` are ORTHOGONAL — never substitute one for the other

A custom `DOCKER_IMAGE` (e.g. `hangfix-deepep-gmm-maxtext-v26.2.tar`, `nightly-...`, `expmt-...`) carries the C++/runtime-side state — bundled Primus-Turbo binaries, RCCL fixes, JAX / XLA, kernel patches, etc. **It does NOT imply v2/v3 MaxText Python patches are baked in.** The v2/v3 changes live in `src/MaxText/layers/moe.py` and are delivered via `MAXTEXT_PATCH_BRANCH` at container startup, regardless of what the image's name suggests. The image's bundled `/workspace/maxtext` is almost always at v1 baseline (or whatever HEAD was when the image was built — usually unrelated to whichever patch-branch you actually want).

`container_env.sh` default (as of 2026-04-30): `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3`. So **v3 is the no-override case now**; v1 and v2 require explicit overrides.

| Want | `DOCKER_IMAGE` | `MAXTEXT_PATCH_BRANCH` |
|---|---|---|
| **v3, any image (default or custom)** | unset (default) or `<image>` | **unset (default = v3 branch)**, OR explicit `yihuang/moe-turbo-gmm-and-deepep-v3` |
| **v2, any image (default or custom)** | unset (default) or `<image>` | **MUST override to `yihuang/moe-turbo-gmm-and-deepep-v2` explicitly** |
| **v1, any image (default or custom)** | unset (default) or `<image>` | **MUST override to `yihuang/moe-turbo-gmm-and-deepep` explicitly** |
| dense / sparse-gmm-fixed (no DeepEP) | unset (default) or `<image>` | unset is fine — the v3 patch only adds DeepEP code paths behind `use_deepep_dispatch=true`; configs without that flag are unaffected |

**Never set `MAXTEXT_PATCH_BRANCH=""` (empty) to "let the image decide".** Empty makes `in_container_run.sh` skip the checkout entirely and use the image's bundled MaxText as-is, which is *almost never* what you want now that the default is v3. If your image was built for v3 testing, an empty patch-branch silently lands you on whatever HEAD the image was built at (usually v1 baseline) — masquerading as v3. The accident shape: user updates `container_env.sh` to a hangfix image expecting "the image has v3 baked in" → agent reasons "use image default" → sets `MAXTEXT_PATCH_BRANCH=""` → v3 Python patches never get applied → test runs v1 code on the v3-targeted hangfix, which is at best meaningless and at worst misleading (you might conclude "v3 still hangs" when actually you never ran v3).

**Rule of thumb:** the model variant you're naming on the command line (`sgd-deepep-v3`, `sgd-deepep-v2`, `sgd-deepep-v1`) MUST match the patch branch — explicit override for v1/v2, default-acceptance for v3. If they don't match, the run is wrong even if it completes.

## Configs to sweep (10 tags, all 1-node/proc)

For each config, list (a) submit-time env vars before `./submit.sh` and (b) passthrough flags after `--`. Any env var not listed already has the right default in `container_env.sh`.

| Tag | Submit-time env vars | Passthrough flags (after `--`) |
|-----|---------------------|-------------------------------|
| `dense-cf1.25` | — | *(default)* |
| `dense-cf2` | — | `capacity_factor=2.0` |
| `dense-cf4` | — | `capacity_factor=4.0` |
| `sparse` | — | `sparse_matmul=true shardy=true` |
| `sparse-gmm` | — | `sparse_matmul=true use_turbo_grouped_gemm=true _env_ENABLE_RAGGED_ONESHOT_KERNEL=1` |
| `sparse-gmm-fixed` | — | `sparse_matmul=true use_turbo_grouped_gemm=true` |
| `sparse-deepep` | — | `sparse_matmul=true use_deepep_dispatch=true shardy=true` |
| `sparse-gmm-deepep` (v1) | **`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep`** (override the v3 default) | `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true` |
| **`sparse-gmm-deepep-v2`** | **`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v2`** (override the v3 default) | same passthroughs as v1 |
| **`sparse-gmm-deepep-v3`** | — *(default branch — `container_env.sh` already points here)* | same passthroughs as v1 |

**For non-bf16 sweeps**, append `quantization=<QUANTIZATION>` (and possibly `quantize_kvcache=false`) to every row's passthrough above. See the [Dtype / quantization](#dtype--quantization) section for the full list of fp8-specific differences (peak compute, output doc filename, loss-parity thresholds, headline question restatement, profile-drill family widening).

**For DCN_EP > 1 rows**, append `dcn_expert_parallelism=<DCN_EP> dcn_fsdp_parallelism=<NODES_PER_JOB / DCN_EP>` to the passthrough for that row. **Re-sweep the 4-config subset only** at DCN_EP > 1 (dense-cf1.25, dense-cf2, dense-cf4, sparse-gmm-fixed) — the other 6 are either infeasible at DCN_EP=1 (`sparse`), intranode-only (`sparse-gmm` one-shot), or **pydantic-blocked by MaxText** at DCN_EP > 1 (all DeepEP variants, see [Parallelism axis](#parallelism-axis-dcn_expert_parallelism-dcn-ep) point 2). All DCN_EP rows land in the **same per-dtype deliverable doc** as additional rows per config (see the doc-layout example in point 3 of that section).

Concrete example templates (substitute MODEL_TAG, NODES_PER_JOB, and the stream's assigned nodelist). **Always include the heartbeat hedge** and **default to `--time=45:00`** (use `25:00` only for small dense cells that have already been observed to compile in <15 min):

v3 (default branch, no env-var override needed — container_env.sh defaults to v3):
```bash
RAY=1 ./submit.sh <MODEL_TAG>:sgd-deepep-v3 \
  --partition=<PARTITION> --nodes=<NODES_PER_JOB> \
  --nodelist=<stream_nodelist> \
  --time=45:00 -- \
  per_device_batch_size=4 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_deepep_dispatch=true steps=15 dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999
```

v1 (env-var override to baseline branch + heartbeat hedge):
```bash
MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep \
RAY=1 ./submit.sh <MODEL_TAG>:sgd-deepep-v1 \
  --partition=<PARTITION> --nodes=<NODES_PER_JOB> \
  --nodelist=<stream_nodelist> \
  --time=45:00 -- \
  per_device_batch_size=4 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_deepep_dispatch=true steps=15 dataset_type=synthetic \
  jax_distributed_heartbeat_timeout_seconds=99999
```

v2 same shape as v1, just swap the branch to `yihuang/moe-turbo-gmm-and-deepep-v2`.

Keep tags short and human (`sgd-deepep-v1`/`-v2`/`-v3`) — do NOT bake the branch string or pdbs value into the tag. The env-var prefix carries the branch info; pdbs comes from the passthrough.

v2/v3 are the headline measurements. Forward-bit-identical to v1, so TGS deltas are pure kernel-level optimization.

## Passthrough hygiene — drop CLI overrides that match defaults

The `-- <passthrough_args>` you send get encoded into the job's `EXP_TAG` and the output-directory name (see "Submit-time env-var vs passthrough — important distinction" above). Slurm's `<id>-${JOB_NAME}.log` path is bounded by ext4's 255-byte per-segment limit, so `JOB_NAME + 12 ≤ 255` → **`JOB_NAME ≤ 243 bytes`**. Every redundant passthrough shrinks your headroom. For profile jobs that pile on `profiler=xplane`, `_env_ENABLE_XLA_DUMP=1`, and `jax_distributed_heartbeat_timeout_seconds=99999` on top of the config's own flags, this gets tight fast.

**Rule: before sending a passthrough, check if it matches the model config's default (in `configs/<MODEL_TAG>.gpu.yml`), `configs/<MODEL_TAG>.env.sh` if present, `base.yml`, or `train_env.sh`. Drop it if it does.**

Common redundancies on MoE benchmark/profile sweeps (save significant EXP_TAG chars):

| Passthrough | Check | Drop if |
|---|---|---|
| `steps=15` | `<MODEL_TAG>.gpu.yml → steps` | already `15` (saves 8 chars) |
| `dataset_type=synthetic` | `<MODEL_TAG>.gpu.yml → dataset_type` | already `"synthetic"` (saves 22 chars) |
| `skip_first_n_steps_for_profiler=5` | `<MODEL_TAG>.gpu.yml → skip_first_n_steps_for_profiler` | already `≥3` AND any value ≥2 is post-warmup — use the yml default (saves 34 chars) |
| `profiler_steps=3` | `<MODEL_TAG>.gpu.yml → profiler_steps` | already `3` (saves 16 chars) |
| `max_target_length=4096` | `<MODEL_TAG>.gpu.yml → max_target_length` | already `4096` (saves 22 chars) |
| `remat_policy=full` | `<MODEL_TAG>.gpu.yml → remat_policy` | already `"full"` (saves 17 chars) |
| `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93` | `train_env.sh → export XLA_PYTHON_CLIENT_MEM_FRACTION=.93` | that's the default (saves 40 chars — only include when overriding to `.96`) |
| `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` | `train_env.sh → default 0` | default (saves 34 chars — only include `=1` for the historical `sparse-gmm` row) |

**Concrete profile example** — from kimi-k2-1t (yml has `skip_first_n_steps_for_profiler: 3` / `profiler_steps: 1` / `steps: 15` / `dataset_type: "synthetic"`), the minimal passthrough for `sparse-gmm-deepep-v3 pdbs=4 PROFILE` is:

```bash
MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3 \
RAY=1 ./submit.sh <MODEL_TAG>:sgd-deepep-v3 \
  --partition=<PARTITION> --nodes=<NODES_PER_JOB> \
  --nodelist=<stream_nodelist> \
  --time=60:00 -- \
  per_device_batch_size=4 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_deepep_dispatch=true profiler=xplane profiler_steps=3 \
  _env_ENABLE_XLA_DUMP=1 jax_distributed_heartbeat_timeout_seconds=99999
```

Note what's dropped: `steps=15`, `dataset_type=synthetic`, `skip_first_n_steps_for_profiler=5` (uses yml's 3 instead — profile steps 3-5 instead of 5-7, both are steady-state on a 15-step run). Note what's kept: `_env_ENABLE_XLA_DUMP=1` — the HLO collective inventory + v1/v2/v3 forward-HLO-diff comparison for the profile drill-down section **requires** this. Dropping it to save bytes is a false economy.

**Diagnostic:** when a submit fails with `[ERROR] JOB_NAME is <N>B; '<id>-$JOB_NAME.log' would exceed 255B.`, don't just drop the most obvious flag — open the yml and find which of your passthroughs is already default. That's the first one to drop.

## Dataset

**The autonomous sweep is synthetic-only.** Pass `dataset_type=synthetic`. The sweep only needs steps 5–14 for steady-state TGS measurement; grain/c4 adds data-loading variance, requires a tokenizer-config block in the yml, and isn't relevant for throughput numbers. Real-data validation belongs in the on-demand loss test path below, not the sweep.

For real-data validation of a specific config, see [On-demand loss test](#on-demand-loss-test-user-triggered) below — that's a separate user-triggered task, not part of the autonomous sweep.

## On-demand loss test (user-triggered)

**This is NOT part of the autonomous sweep.** It runs only when the user explicitly requests a loss test on a specific config + pdbs. Use it to validate that a chosen setup actually trains correctly on real data over a longer horizon than the 15-step throughput probes can show (gradient drift, accumulated numerical noise, divergence in late training).

### What the user specifies

The user names two things:

- **CONFIG_TAG**: one of the 10 sweep tags (e.g. `sparse-gmm-deepep-v3`, `sparse-gmm-fixed`, `dense-cf1.25`).
- **PDBS**: the `per_device_batch_size` to use (typically P★ from the sweep, or any value the user wants validated).

If either is missing, ask the user — do not pick autonomously. Everything else (nodelist, walltime, branch, dataset, etc.) the agent fills in.

### Setup — edit the yml directly to swap to grain/c4

The grain/c4 dataset settings total ~120+ chars across `dataset_type` / `grain_file_type` / `grain_train_files` / `grain_worker_count` / `tokenizer_type` / `tokenizer_path` / `hf_access_token`. Passing them via `_env_*=...` passthroughs would overflow the 243-byte `JOB_NAME` ceiling and produce an unreadable output directory name. **Edit `configs/<MODEL_TAG>.gpu.yml` directly** to replace the dataset section with the c4/grain block from `configs/ds-proxy-se0-e256-h4096.gpu.yml`:

```yaml
# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type: grain
grain_file_type: parquet
grain_train_files: /datasets/c4/en/3.0.1/parquet/c4-train-*-of-01024.parquet
grain_worker_count: 16
tokenizer_type: huggingface
tokenizer_path: <HF_REPO_WHOSE_TOKENIZER_VOCAB_MATCHES_MODEL_VOCAB_SIZE>
# Set hf_access_token only if the tokenizer comes from a gated HuggingFace model
# (e.g. meta-llama/Llama-2-7b). Not needed for most public DeepSeek/Kimi/Qwen base tokenizers.
hf_access_token: ""
```

**`tokenizer_path` MUST match the model's `vocab_size`** (read from `src/MaxText/configs/models/<MODEL_TAG>.yml` inside the MaxText repo, NOT the proxy yml). A mismatched tokenizer is silent: training proceeds, but loss starts at random level and either won't drop or moves the wrong direction because the model's embedding table is indexed by IDs from a different BPE vocabulary. Pinning examples worth getting right:

| Model | `vocab_size` | Correct tokenizer (HF repo) |
|---|---:|---|
| `deepseek3-671b` (DS V3) | 129280 | `deepseek-ai/DeepSeek-V3-Base` |
| `deepseek2-*` (DS V2 family) | 102400 | `deepseek-ai/DeepSeek-V2-Lite` |
| `ds-proxy-*` (DS V2-Lite proxy) | 102400 | `deepseek-ai/DeepSeek-V2-Lite` |
| `kimi-k2-1t` | 163840 | `moonshotai/Kimi-K2-Base` (or `-Instruct`) |
| `qwen3-*` (all sizes) | 151936 | `Qwen/Qwen3-7B-Base` (any Qwen3 variant) |
| `llama3.1-*` / `llama3.3-*` | 128256 | `meta-llama/Meta-Llama-3.1-8B` *(gated)* |
| `llama2-*` | 32000 | `meta-llama/Llama-2-7b-hf` *(gated)* |
| `gemma2-*` / `gemma-*` | 256128 | `google/gemma-2-9b` *(gated for some)* |

Don't infer the tokenizer from a sibling proxy yml's value — the proxy and the production model often have different vocabs by design (proxy uses V2-Lite for compute economy even when the production model is V3). Always cross-check `vocab_size` against the HF model card. After the loss test, leave the yml change in place unless the user asks to revert — subsequent autonomous sweep work would need to flip the dataset back to `synthetic` (or override via passthrough).

### Submit

Same pattern as the sweep, but with `steps=2000` and the user's `PDBS`. Two non-obvious rules:

1. **Do NOT add any grain/dataset flags to the passthrough** — the yml carries those now, and adding them would re-explode the job name.
2. **For v1/v2 configs, MUST set `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep[-v2]` explicitly to override the v3 default in `container_env.sh`.** v3 runs need no env-var prefix (it's the default). NEVER set `MAXTEXT_PATCH_BRANCH=""` to "use the image's MaxText" — that silently lands you on whatever HEAD the image was built at (almost always v1 baseline), so the loss test then measures the wrong code path. See [`DOCKER_IMAGE` and `MAXTEXT_PATCH_BRANCH` are ORTHOGONAL](#docker_image-and-maxtext_patch_branch-are-orthogonal--never-substitute-one-for-the-other) for the full table.

```bash
# Example: sparse-gmm-deepep-v3 loss test at pdbs=6 (no env-var prefix needed; v3 is default)
RAY=1 ./submit.sh <MODEL_TAG>:sgd-deepep-v3-loss \
  --partition=<PARTITION> --nodes=<NODES_PER_JOB> \
  --nodelist=<a healthy nodelist> \
  --time=<adaptive — see below> -- \
  per_device_batch_size=6 sparse_matmul=true use_turbo_grouped_gemm=true \
  use_deepep_dispatch=true steps=2000 \
  jax_distributed_heartbeat_timeout_seconds=99999
```

For v1/v2, prepend `MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep[-v2]` to the command.

**Walltime budget — estimate per-variant, multiply by 1.5×, do not pick a constant.**

`--time=` should be sized from the *measured* steady-state step time of that exact variant + pdbs combination, not a hard-coded 14h / 20h / 24h / 36h.  Step time varies wildly across the sweep — the same model can land anywhere from ~25 s/step (small dense, low capacity factor) to ~55 s/step (`dense-cf4` with 4× capacity → 4× all-to-all volume) to ~70+ s/step (DeepEP variants on slow cold-compiles or DCN_EP=4 cells).  A single fixed walltime guarantees either over-provisioning early variants or under-provisioning the heavy ones.

Recipe:

1. **Get a step-time estimate.**  Cheapest sources, in order of preference:
   - The TGS sweep's own measurement — the same `<config, pdbs>` cell already ran for 15 steps before being promoted to the loss test.  Read `seconds: <X>` from the steady-state region (steps 5–14) of that prior log.  This is the most accurate number you'll get for free.
   - A 15-step calibration run at `--time=45:00 dataset_type=synthetic steps=15` if no prior measurement exists.  Read step time the same way.
   - Last resort — extrapolate from a *different* cell of the same config family at adjacent pdbs, scaled by `(new_pdbs / old_pdbs)`.  Mark the resulting walltime as "extrapolated" in the submit log.

2. **Compute the budget.**

   ```text
   walltime = ceil( 1.5 × (compile_overhead + steps × step_time_seconds) / 3600 ) hours
   ```

   Defaults: `compile_overhead = 600 s` (10 min — covers JAX init + RCCL + first-step compile for already-traced HLOs; bump to 1800 s for cold-compile variants like DeepEP v1/v3 that haven't run on this image before).  `1.5×` covers step-time variance (data-loading hiccups, slow checkpoint window, occasional 2× outlier steps observed in long runs), Ray/JAX teardown overhead at the end, and Slurm SIGTERM grace window.  Round up to the nearest hour for the `--time=HH:00:00` syntax.

3. **Worked examples** (DS3-671B, 8N, pdbs=7 — actual walltimes that should have been used):

   | Variant | Step time (measured) | 2000 × step | + compile (10 min) | × 1.5 | Recommended `--time=` |
   |---|---|---|---|---|---|
   | dense-cf1.25 | 28 s | 15.6 h | 15.7 h | 23.6 h | `--time=24:00:00` |
   | sparse-gmm-fixed | 38 s | 21.1 h | 21.3 h | 32.0 h | `--time=32:00:00` |
   | dense-cf2 | 35 s | 19.4 h | 19.6 h | 29.4 h | `--time=30:00:00` |
   | **dense-cf4** | **55 s** | **30.6 h** | **30.7 h** | **46.1 h** | **`--time=46:00:00`** |
   | sparse-gmm-deepep-v3 | 38 s (v3 is faster than v1) | 21.1 h | 21.6 h (cold compile) | 32.4 h | `--time=33:00:00` |

   The 14 h / 20 h "default" used in earlier revisions of this prompt under-budgeted every one of these, and the fixed 24 h I tried instead landed cf4 at step 1564 / 2000 (78 %).  The 1.5× formula would have correctly given cf4 ~46 h.

4. **If step time is unknown and you cannot calibrate first** (rare — only when the sweep's measured cell is already gone): use the variant's *family ceiling* as a pessimistic floor — DeepEP / cf4 / DCN_EP=4 cells all default to `--time=48:00:00`, dense-low-capacity defaults to `--time=24:00:00`.  Document the chosen wall + the reason in the submit log so the next run can refine it.

5. **Slurm-walltime ≠ training-walltime cap.**  Slurm SIGTERMs the job at `--time` exactly — there's no soft warning; the job just dies mid-step.  The 1.5× margin is what catches the rare slow-step events; it's not optional padding.

### Monitoring

Apply the same hang/crash decision rules as the autonomous sweep ([Crash / hang / flake quick reference](#crash--hang--flake-quick-reference)). Sample loss every 100 steps; report to the user at steps 14 / 100 / 500 / 1000 / 2000.

### Output

Single user-facing message back: final loss at step 2000, intermediate checkpoints (14 / 100 / 500 / 1000), TGS, MFU, total wall time, output dir path. No `.md` deliverable — this is a one-shot validation, not a sweep.

## Methodology

0. **Resumability check** — look for prior `<MODEL_TAG>-pdbs-sweep.md` / `.zh.md`, parse state, cross-check against `outputs/`, identify gaps. Adopt prior nodelist(s) if present + healthy. Skip if no prior files exist.
1. **Pre-sweep node health check** — verify all AVAILABLE_NODES are healthy via sinfo/ssh/rocm-smi/ibstat. Best-effort fix unhealthy ones. Remove terminally-unhealthy ones. Compute NUM_STREAMS + partition into disjoint stream nodelists. TG health summary.
2. **Calibration** (only if NUM_STREAMS ≥ 2) — run `dense-cf1.25 pdbs=1 steps=15` (heartbeat hedge) ×2 on each stream. Compute `median_tgs[s]`, `per_stream_factor[s]`, `cross_stream_std`. Classify into noise-only / small-discrepancy / significant-discrepancy regime per the table in "Parallel-streams assignment + calibration". TG calibration summary. Pick the scheduling strategy for the regime.
3. **Dry run** — if calibration already produced a healthy dense-cf1.25 pdbs=1 cell on each stream, the calibration IS the dry run. Record the best run's numbers in the results table; skip a separate dry-run job. If NUM_STREAMS == 1, run one `dense-cf1.25 pdbs=1` dry run on the single stream.
4. **Ceiling discovery per config, scheduled via shared queue** — enqueue `(config, pdbs)` work items (longest-ladder-first ordering). Streams pull the next item when idle. Each cell is pinned to whichever stream picks it up for the duration of its retries. For each config, walk pdbs upward (1, 2, 4, 6, 8, 10, 12, 16 …) until the first OOM; back-fill ±1 around the ceiling. Heartbeat hedge on first submit per config. Respect the regime's v1/v2/v3 pinning constraint (see calibration table). Do NOT re-run cells already populated + verified in the resumed data.
5. **Record all successful runs** as data points, including which stream they ran on. If in significant-discrepancy regime, also record normalized TGS = raw_TGS / per_stream_factor[s]. Merge with prior resumed data.
6. **Compute P★** = min across all configs' max pdbs.
7. **Apply decision rules** as issues arise. Don't ask; decide. Keep a running list of NIC recovery attempts, policy-blocked commands, cells marked `nic-flake⋆`/`infra-flake⋆`, stream-to-cell assignments, and calibration factors for the doc's infrastructure notes.
8. **Metrics** — steps 5–14 mean. Per cell: TGS (raw; and normalized if in significant regime), TFLOP/s/device, MFU, step time, loss @ step 14.
9. **Profile drill-downs — run LAST, after every config's ceiling is nailed (milestone M7).** This sequencing is important: P★ and `min(v1_max, v2_max, v3_max)` are only known once the full sweep is done. Running a profile mid-sweep at a pdbs that later gets revised (because a late-discovered OOM lowered P★) wastes a ~45-min profile job — a single profile run is ~2× the cost of a benchmark run, so don't speculatively profile. Specifically: **do NOT submit any `profiler=xplane` job until every config in the sweep has either a confirmed `max_pdbs` or a terminal failure (`✗ OOM-*`, `hung⋆`, `compile-timeout⋆`, etc.).**

   Once ceilings are final:
   - Run with `profiler=xplane profiler_steps=3 _env_ENABLE_XLA_DUMP=1` and `--time=45:00`. **`_env_ENABLE_XLA_DUMP=1` is REQUIRED** for HLO collective-inventory tables in the drill-down section; do not drop it. See the "Passthrough hygiene — drop CLI overrides that match defaults" rule below for how to keep the JOB_NAME under the 243-byte ext4 path-segment limit when profile flags + heartbeat hedge + XLA dump stack up.
   - Use `utils/profile_drill.py` as ground truth for per-kernel times (do not trust TraceLens CSV on 1-node/proc — divisor bug documented in `skills/profile-drill/SKILL.md`).
   - **Required: Cross-path drill-down at P★** — 5 profile jobs: `dense-cf1.25`, `sparse-gmm-fixed`, `sparse-gmm-deepep` (v1), `-v2`, `-v3`. All on the same stream (the one the regime pins v1/v2/v3 to, for the cleanest intra-DeepEP kernel delta; the dense + sparse-gmm-fixed numbers inherit that stream's `per_stream_factor[s]` for normalization if applicable). Build a 5-column kernel-composition table showing `ragged-all-to-all` / `RaggedAllToAllKernelImpl` / `moe_dispatch` / `moe_combine` / `input_scatter_fusion_*.kd` / `loop_select_fusion` / `loop_gather_fusion` / RCCL / GEMM / flash-attn / misc / total / step-time / idle-gap per config. The cross-path columns expose dense-vs-sparse-vs-DeepEP where the time goes; the v1/v2/v3 columns expose the optimization chain within DeepEP.
   - **Optional: Supplementary v1/v2/v3 drill-down at `min(v1_max, v2_max, v3_max)`** — only if that pdbs is strictly greater than P★ (i.e., dense or sparse-gmm-fixed would OOM there). 3 additional profile jobs: v1, v2, v3 at that higher pdbs. Same structure as the required section minus the dense + sparse-gmm-fixed columns. Skip if pdbs are equal.
   - If a prior drill-down exists in resumed data, verify its tables against the trace JSONs on disk and refresh only if stale.

## Output format — match DS3 doc structure

Same sections as DS3 (single-launcher). Result tables are **ragged**: each (pdbs × config) cell is either a number or `✗<classified>`. Rows = all distinct pdbs that appeared in any successful run.

- Header (Date, Model, Hardware, Image, Base config, Peak BF16; v1/v2/v3 patch-branch commits)
- Background + "1-GPU/proc not re-tested; see DS3 takeaway #3" + ragged-matrix probing note + calibration regime (noise-only / small / significant) + `per_stream_factor[s]` table if NUM_STREAMS > 1 + "numbers are normalized / raw" statement matching the regime
- Configs under test + "What distinguishes v1/v2/v3" paragraph
- **Feasibility summary** — table listing per config: `max_pdbs` (ceiling, defined by first OOM), `argmax_TGS_pdbs` (the pdbs where TGS peaks — often 1–2 below `max_pdbs` on memory-pressured models), peak TGS, peak MFU, and P★. On 1T-class models, TGS can be non-monotonic near the ceiling (e.g. kimi dense-cf1.25 peaked at pdbs=11 = 1170 TGS while max=12 = 1134 TGS — the last pdbs before OOM can under-perform due to activation-memory thrashing). Report both.
- Four results tables (TGS / TFLOP/s / step-time / loss) with headers like `sparse-gmm-deepep-v3 (1-node)`. **Visually highlight the `pdbs=P★` row** — it's the apples-to-apples comparison row. If in significant-discrepancy regime, each cell shows the normalized TGS with raw in parentheses: `1032 (raw 1047 on S2)` for traceability.
- Key takeaways (≥5). Must include: (i) peak TGS/MFU + config **+ the pdbs where that peak occurs (which may be lower than `max_pdbs` — see below)**, (ii) at-P★ cross-config ranking, (iii) capacity_factor cost curve, (iv) dense-vs-sparse gap at P★, (v) **v1→v2→v3 replication finding**, (vi) memory-ceiling finding (`max_pdbs` per config).
- Infrastructure / memory-ceiling notes — stream partitioning; calibration results (per_stream_factor + regime classification); per-cell stream assignment; pre-sweep node health findings; every OOM cell with GiB numbers and source; every infra-flake retry; every NIC reload attempt with command + outcome; every `nic-flake⋆` cell including manual reload command the user would need; any mid-sweep re-calibration triggered by infra changes.
- Footnotes
- **Cross-path profile drill-down at P★** — 5-column kernel-composition table (dense-cf1.25, sparse-gmm-fixed, sparse-gmm-deepep v1/v2/v3), stream assignment noted. Prose analysis covers: (i) where the dense-vs-sparse gap lives (which kernel families dominate each), (ii) DeepEP's MoE-machinery overhead vs sparse-gmm-fixed's RCCL-based ragged_all_to_all, (iii) the v1→v2→v3 kernel-removal chain within the DeepEP columns (DS3's `input_scatter_fusion_*.kd` story — confirm or contrast), (iv) remaining optimization headroom highlighted by side-by-side families.
- **(If present) Supplementary v1/v2/v3 drill-down at higher pdbs** — only if `min(v1/v2/v3_max) > P★`; 3-column table at that higher pdbs showing the optimization-chain story at peak DeepEP throughput.
- "How to reproduce" code block (with the `MAXTEXT_PATCH_BRANCH=` env-var prefix pattern)

## Etiquette

- Never `scancel` jobs you didn't submit (check `outputs/<jobid>-*/` exists).
- Default `--time=45:00` for benchmark jobs on 1T-class models; `--time=25:00` only when you've directly observed the cell compile in <15 min. `--time=60:00` or `--time=90:00` for compile-timeout retries (see ladder in the compile-timeout rule). Profile runs: `--time=45:00` minimum.
- No edits to the deepseek3 sweep docs.
- Keep `.md` and `.zh.md` structurally identical.
- **Per-cell nodelist stability is load-bearing** — a cell never moves between streams during its retries. Cross-cell stream mixing within a single config is allowed; the output doc records per-cell stream assignment for traceability.

## Cost budget

10 configs × 5–8 successful pdbs each + 5 cross-path profile runs (+ 2–3 optional supplementary v1/v2/v3 profile runs if v3 has headroom above P★) = roughly **55–95 jobs**. Benchmark jobs ~25 min each, profile jobs ~45 min each. Serialized single-stream = 22–45 h. With NUM_STREAMS parallelism the wall-time scales roughly as sum(job_durations) ÷ NUM_STREAMS ÷ 60 (hours). **No hard stop — run until every config has a confirmed ceiling AND the P★ row is complete AND the required cross-path drill-down is in the doc.** If wall time exceeds 2× the estimate, send hourly heartbeats and keep going.

## Known constraints

- **Shardy**: `sparse`, `sparse-deepep` need `shardy=true`; `sparse-gmm*` don't.
- **MEM_FRACTION**: default `.93`; `.96` per OOM rules. `.96` can flip the OOM into an RCCL-buffer OOM on the opposite side (DS3 job 12885 precedent).
- **OOM has five sources** — classify every crash by log signature per the OOM table.
- **OOM-as-hang at the ceiling**: XLA doesn't always emit a clean crash. Hangs at pdbs > last_successful are OOM-hang-suspect; retry once to disambiguate from transient infra. Never retry more than once.
- **Always grep the log for OOM signatures first** before deciding "silent hang".
- **RCCL-init flakes**: retry (max 2), same nodelist (only if below-ceiling pdbs). `detect_nccl_env.sh` auto-detects `NCCL_IB_GID_INDEX` per-node (RoCEv2-preferred); a `[WARN] inconsistent routable GID indices` line means that node fell back to `1` and may need a manual `_env_NCCL_IB_GID_INDEX=<N>` override if RCCL-init fails there.
- **NIC errors**: attempt in-place reload; TG the user the manual command if policy blocks. Never exclude the node from the config's nodelist.
- **Other transient infra errors** (NCCL/heartbeat/node-fail): retry (max 3); same nodelist.
- **Per-cell nodelist stability is load-bearing** (a cell + its retries stay on one stream); cross-cell stream mixing is fine and the preferred strategy for parallelism — load balance across streams, not across configs.
- **Calibrate before parallelizing**: measure `per_stream_factor[s]` up front; the calibration regime (noise-only / small / significant) picks the scheduling + normalization strategy. Document both raw and normalized TGS when in the significant-discrepancy regime.
- **v1/v2/v3 drill-down nodelist policy** depends on calibration regime: noise-only → can split; small → pin to fastest stream; significant → pin together regardless of speed.
- **Memory cliff is a primary deliverable** — each config's max_pdbs is a first-class data point.
- **`QUANTIZATION` is a separate deliverable, not a column** — an fp8 sweep produces its own `<MODEL_TAG>-fp8-pdbs-sweep.md` / `.zh.md` pair, which cross-links to the bf16 pair. Both docs share the same 10-config taxonomy, P★ methodology, and retry rules, but differ in: peak-compute / MFU constants, the every-row-appended `quantization=fp8` passthrough, loss-parity thresholds (see [Dtype / quantization](#dtype--quantization) point 5), and the headline question (whether the v1→v2→v3 scatter-add-elimination chain *survives* fp8 lowering — a genuine open question per point 6 in the dtype section).
- **`DCN_EP` is an in-doc column dimension, not a separate deliverable** — unlike dtype, additional DCN_EP values go into the same per-dtype doc as extra rows under each DCN-EP-relevant config (**only** dense-cf1.25, dense-cf2, dense-cf4, sparse-gmm-fixed — DeepEP variants are pydantic-blocked by MaxText at DCN_EP > 1). See [Parallelism axis](#parallelism-axis-dcn_expert_parallelism-dcn-ep) for the row layout and the full list of blocked configs. Originally intended to test DS3's "DeepEP wins on inter-node EP" claim directly; in practice this claim is **unmeasurable** against current MaxText and the DCN-EP sweep characterizes only the non-DeepEP regime.
- **No grain/c4 in the autonomous sweep.** On-demand loss tests use grain/c4 — see [On-demand loss test](#on-demand-loss-test-user-triggered).

---

Start by reading the DS3 sweep doc (especially v1/v2/v3 drill-down), CONFIG_FILE and MODEL_ENV_FILE if present, `container_env.sh` defaults, the host-cmd policy file, and skills. Ping host-cmd + check TG creds. **Then run the resumability check** — look for prior `<MODEL_TAG>-pdbs-sweep.md` / `.zh.md` files and `outputs/NNNNN-JAX-<MODEL_TAG>-*` dirs, extract completed cells and remaining gaps, TG the resumption summary. Run the pre-sweep node health check across AVAILABLE_NODES; best-effort-fix anything unhealthy; compute NUM_STREAMS and partition into disjoint stream nodelists; TG the health/stream summary. **If NUM_STREAMS ≥ 2, run the calibration** — dense-cf1.25 pdbs=1 ×2 per stream, classify the regime, pick the scheduling + normalization strategy, TG the calibration summary. Then walk each not-yet-done config up to its ceiling via the shared-queue scheduler — **per-cell nodelist stability every retry**. When a job hangs at pdbs near the ceiling, apply OOM-as-hang detection. Compute P★, fill the comparison row (normalized if needed). Once every config's ceiling is final (milestone M7), **then** submit the cross-path profile drill-down at P★ (5 configs — dense-cf1.25, sparse-gmm-fixed, v1/v2/v3) on the stream designated by the regime rule; add the supplementary v1/v2/v3 drill-down at `min(v1/v2/v3_max)` only if it exceeds P★. Never profile speculatively mid-sweep. On NIC errors, attempt in-place reload; never exclude nodes from a cell's stream. Decide autonomously at every branch point. Run to completion — extend, don't restart.
