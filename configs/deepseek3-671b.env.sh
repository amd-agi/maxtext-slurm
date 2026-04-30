# Per-model environment overrides for DeepSeek-V3 671B (`deepseek3-671b`).
# Sourced after train_env.sh, before CLI _env_ overrides.
#
# This recipe was tuned on FSDP=8 + sparse-gmm-deepep-v3 (`sgd-v3`) at pdbs=7
# via a 36-profile XLA-flag / NCCL-env / memory-fraction sweep — see
# `pp-vs-fsdp-deepseek3-671b.md` for the full methodology and results.
#
# Headline finding: the docker image's default
# `--xla_gpu_all_gather_combine_threshold_bytes=8589934592` (8 GiB) fuses every
# per-step all-gather into one serial mega-call that runs as a hard barrier
# before any layer's compute can start. Lowering only the all-gather threshold
# to 1 GiB splits it into ~4-5 chunks that XLA's latency-hiding scheduler
# (already on by default) interleaves with per-layer compute, recovering ~3 s
# of exposed comm per step. Result: +11.6 % TGS (1017.7 → 1135.7) on `sgd-v3`,
# beating the historical Apr-14 baseline (1097 TGS) by +3.5 %.
#
# Reduce-scatter is intentionally LEFT at the image's 8 GiB default — backward-
# pass reduce-scatters are inherently large (gradient buffers from cross-layer
# accumulation are 100s of MiB to several GiB per layer) and the backward pass
# is more compute-dense, so combining them into one large call is cheaper than
# splitting them. Empirically `xla_gpu_reduce_scatter_combine_threshold_bytes=1G`
# (which would otherwise stack with the all-gather change) regresses TGS from
# +11.60 % to +5.85 %.
#
# Nothing else from the sweep stacks meaningfully on top:
#   - NCCL_NCHANNELS_PER_NET_PEER=8        +0 to +0.01 % when stacked
#   - NCCL_IB_QPS_PER_CONNECTION=8         neutral (and sometimes triggers RCCL init hangs)
#   - XLA_PYTHON_CLIENT_MEM_FRACTION=.95   +0 to +0.01 % when stacked
#   - --xla_gpu_enable_latency_hiding_scheduler=true  no-op (image default already true)
#   - --xla_gpu_enable_while_loop_double_buffering=true  -2.7 % (memory pressure)
#   - --xla_gpu_experimental_parallel_collective_overlap_limit=2/4/8  -4 to -5 %
#   - --xla_gpu_enable_pipelined_*=true   OOMs (prefetch buffers exceed HBM)
#
# Safety: the image's other compiled-in defaults (`enable_latency_hiding_scheduler`,
# `reduce_scatter_combine_threshold_bytes=8 GiB`, etc.) are preserved because
# we only append a single flag rather than rebuilding XLA_FLAGS from scratch.

XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_all_gather_combine_threshold_bytes=1073741824"
export XLA_FLAGS
