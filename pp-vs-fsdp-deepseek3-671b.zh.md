# DeepSeek-V3 671B —— AMD MI355 上的 PP=8 vs FSDP=8 对比（8 节点 × 8 GPU）

- **日期：** 2026-04-30（初版 PP-vs-FSDP 对比 + FSDP=8 调优）；2026-05-01（PP=8 XLA / NCCL 调优扩展 + 跨 topology `remat_policy` 扫描）
- **模型：** `deepseek3-671b`（混合专家 / Mixture-of-Experts，58 层 decoder，词表大小 129280，256 个 routed experts，top-k=8）
- **硬件：** 8 节点 × 8× AMD MI355（每张卡 288 GB HBM，Pensando AINIC 互联）。共 64 张 GPU。BF16 峰值 ≈ 2500 TFLOP/s/device → MFU ≈ TFLOP/25。PP=8 调优阶段固定节点列表 `chi[2766,2800,2810,2832,2835,2865,2872,2883]`；4/30 PP-vs-FSDP 对比 + FSDP=8 sweep 用同一个 partition。
- **镜像：** [`/mnt/vast/yihuang/ppfix-hangfix-deepep-gmm-maxtext-v26.2.tar`](https://github.com/ROCm/Primus-Turbo)（包含针对 `nn.vmap("stage")`-of-`shard_map` 组合的 axis-aware Primus-Turbo batching 规则 + 上游 [`fix/deepep/combine_hang`](https://github.com/AMD-AGI/Primus-Turbo/tree/fix/deepep/combine_hang) C++ kernel 修复）
- **MaxText 分支**（仅 sgd 配置使用）：[`yihuang/moe-turbo-gmm-and-deepep-v3`](https://github.com/ROCm/maxtext/tree/yihuang/moe-turbo-gmm-and-deepep-v3) @ `f59be3c9` —— 自 2026-04-30 起也是 `container_env.sh` 默认值，因此 `sgd-v3` 运行不再需要显式设置 `MAXTEXT_PATCH_BRANCH=…`。v1/v2 基线仍需要显式覆盖。
- **基础配置：** [`configs/deepseek3-671b.gpu.yml`](configs/deepseek3-671b.gpu.yml)。PP=8 透传参数加上 `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1`（MaxText 自动从 `pipeline_parallel_layers=56` 推导出 `num_layers_per_pipeline_stage=1`、`num_pipeline_microbatches=8`）。
- **序列长度：** 4096。**步数：** 15。除非另作说明，所有数字都是 **steps 9-14 的稳态平均值**；标注「clean」表示 `profiler: ""`、无 XLA dump。
- **数据集：** `dataset_type=synthetic`（扫描时为 gpu.yml 默认值；该 yml 后续已切换为 `grain`/c4 —— 复现时需在 CLI 上覆盖 `dataset_type=synthetic`，详见 `skills/xla-tuning`）。

## TL;DR

我们测试的三种配置都是 MoE —— 它们的差别在于 MaxText `moe.py` 中选择了哪种 **MoE 实现**（`sparse_matmul=False` → `dense_matmul` 分支；`sparse_matmul=True` → `sparse_matmul` 分支）。这里的 "dense" 指的是 *MoE 实现的选择*，并不表示模型本身是非 MoE：

- **`dense-cf1.25`**：`dense_matmul` 分支 + capacity-factor 1.25 dropping（`sparse_matmul=False`、`capacity_factor=1.25`）。每个 expert 拿到一个固定大小的 `capacity_factor × tokens / num_experts` 张量；超出容量的 tokens 会被丢弃。"dense" 这个名字来源于 `moe.py` 在这个固定形状的 per-expert 张量上使用 dense matmul（没有 ragged 维度）。
- **`sparse-gmm-fixed`（sgf）**：`sparse_matmul` 分支，dropless 实现，使用 `ragged_all_to_all` + `ragged_dot`（`sparse_matmul=True`、`use_turbo_grouped_gemm=True`）。所有 tokens 严格路由到它们的 top-k experts；ragged GEMM 避免了 capacity-factor padding，但 `ragged_all_to_all` 会物化一个 `num_ranks × tokens × hidden` 的接收缓冲区，可扩展性差。
- **`sparse-gmm-deepep-v3`（sgd-v3）**：`sparse_matmul` 分支，dropless 实现，使用 DeepEP 节点内 IPC dispatch + Primus-Turbo grouped GEMM（`sparse_matmul=True`、`use_turbo_grouped_gemm=True`、`use_deepep_dispatch=True`，运行在 `yihuang/moe-turbo-gmm-and-deepep-v3` MaxText 分支上）。绕过 MaxText 的 `ragged_all_to_all`，改用 per-rank-prefix 路由的、大小为 `num_worst_tokens × hidden` 的接收缓冲区。

**最优配方依赖于 路径 × topology × 分支**。不存在能用于所有情形的全局默认；最终结果是：

| 路径 | FSDP=8 最佳 | PP=8 最佳 | 生产赢家 |
|---|---:|---:|---|
| **`sgd-v3`**（DeepEP dropless） | **1135.7**（14602）⭐ —— `ag=1 GiB` | 999.8 平均 n=2（14668/14674）—— `overlap2 + async_priority` | **FSDP=8（+12.0 %）** |
| **`dense-cf1.25`**（capacity-factor dropping） | 1207.9（14629）—— `ag=1 GiB` | **1224.1** 平均 n=2（14672/14673）⭐ —— 仅 `overlap2` | **PP=8（+1.34 %）** |
| **`sparse-gmm-fixed`**（ragged_dot dropless） | OOM @217 GiB temp | OOM @217 GiB temp | n/a |

**两个赢家 XLA flag 在两个 topology 之间符号相反** —— 把 FSDP 调优过的 `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`（目前已无条件部署在 `configs/deepseek3-671b.env.sh` 中）让所有 PP=8 dense 提交悄悄降速 **6.6 %**，让 PP=8 sgd-v3 略降 1.0 %（在 jitter 内）。该配方必须按 topology（PP=8 还要按 MoE 分支）守护 —— 详见 [§ 推荐部署方式](#推荐部署方式)。

其他操作性发现：

- **`remat_policy: 'full'` 在 pdbs=7 下对两个 topology、两个 MoE 分支都是最优**。更轻的 remat 要么 OOM（`nn.scan`-PP / `scan_layers=True`-FSDP layout 下激活内存放大约 56-58×），要么装得下但回退；在 FSDP=8 dense 下 `save_out_proj` 甚至会让 loss 偏离。详见 [§ `remat_policy` 灵敏度](#remat_policy-灵敏度覆盖-pp8-fsdp8--通用结论pdbs7-下-full-最优)。
- **`sgf`（sparse-gmm-fixed）在本机器栈生产规模下不可用**：所有可行 pdbs ≥ 7 都 OOM，因为 `ragged_all_to_all` 会物化一个完整的 `num_ranks × tokens × hidden` 接收缓冲区。

| 路径 | `dense-cf1.25`（dropping） | `sparse-gmm-fixed`（ragged_dot dropless） | `sparse-gmm-deepep-v3`（DeepEP dropless，镜像默认 XLA） | `sparse-gmm-deepep-v3`（DeepEP dropless，调优 XLA） |
|---|---:|---:|---:|---:|
| **PP=8 pdbs=7** TGS（镜像默认 XLA） | **1157.6**（14571） | OOM @217 GiB temp | **939.9**（14570） | — |
| **PP=8 pdbs=7** TGS（PP 调优：`overlap2`[+`async`]） | **1224.1**（14672/14673 平均）⭐ | （仍会 OOM） | — | **999.8**（14668/14674 平均） |
| **FSDP=8 pdbs=7** TGS（镜像默认 XLA） | **1002.4**（14573） | OOM @217 GiB temp | **1017.7**（14572） | — |
| **FSDP=8 pdbs=7** TGS（FSDP 调优：`ag=1 GiB`） | 1207.9（14629） | （仍会 OOM） | — | **1135.7**（14602）⭐ |
| 历史 sgd-fs8 基线（4 月 14 日，原 `deepep-gmm-maxtext-v26.2.tar`） | — | — | 1097 | — |

所有 clean 行（`profiler: ""`、无 XLA dump），steps 9-14 平均。PP=8 使用 `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1 pipeline_parallel_layers=56 num_layers_per_pipeline_stage=1 num_pipeline_microbatches=8`（V=7 个 virtual chunks）。FSDP=8 使用 YAML 默认 `dcn_fsdp_parallelism=8 dcn_pipeline_parallelism=1`。

## 配置（三个全部都是 MoE —— DS3-671B 有 256 个 routed experts、top-k=8）

| Tag | sparse_matmul | capacity_factor | use_turbo_grouped_gemm | use_deepep_dispatch | MoE matmul 分支 | MaxText 分支 |
|---|---|---:|---|---|---|---|
| **`dense-cf1.25`** | false | 1.25 | false | false | `dense_matmul`（固定形状 per-expert 张量，超出容量则丢弃） | base |
| **`sgf`** | true | n/a | true | false | `sparse_matmul`，使用 `ragged_all_to_all` + `ragged_dot`（MaxText 内置） | base |
| **`sgd-v3`** | true | n/a | true | true | `sparse_matmul`，使用 DeepEP IPC dispatch + Primus-Turbo grouped GEMM | `yihuang/moe-turbo-gmm-and-deepep-v3` |

`dense-cf1.25` 中的 "dense" 是 `moe.py` 的分支名（`dense_matmul`），不是说模型不是 MoE —— DS3-671B 始终是 MoE。`dense_matmul` 选择的是 MaxText 的 capacity-factor-bounded 固定形状 per-expert 路径（用 padding 到 capacity 的方式来绕过没有内置 ragged GEMM kernel 的限制）；`sparse_matmul` 选择的是 dropless 的 ragged 路径（`sgf` 用 MaxText 内置的 `ragged_dot`；`sgd-v3` 用 Primus-Turbo 的 grouped GEMM 配 DeepEP dispatch）。

每个 tag 跑两次：一次用 `dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1`（PP=8），一次用 YAML 默认 `dcn_fsdp_parallelism=8 dcn_pipeline_parallelism=1`（FSDP=8）。共 64 张 GPU，每个 DCN replica 8 张。

`pdbs=7` 是 apples-to-apples 的对比轴（与 FSDP 的内存预算匹配）；为了测内存可行性，PP-only 配置我们也测了 `pdbs=8`。

## 内存可行性矩阵

|  | pdbs=7 | pdbs=8 |
|---|---|---|
| sgd-pp8（DeepEP dropless） | 总 **253** / 临时 **178** GiB ✓ | 总 253 / 临时 178 ✓（启用 axis-aware Primus-Turbo 规则后） |
| sgd-fs8（DeepEP dropless） | 总 **236** / 临时 **178** GiB ✓ | （未跑；pdbs=7 是 FSDP 的天花板） |
| **sgf-pp8**（ragged_dot dropless） | 总 **276** / 临时 **217** GiB **✗ OOM** | 总 296 / 临时 222 ✗ OOM |
| **sgf-fs8**（ragged_dot dropless） | 总 **276** / 临时 **217** GiB **✗ OOM** | （跳过） |
| dense-cf1.25-pp8（capacity-factor dropping） | 总 ≈200 / 临时 ≈125 GiB ✓ | ✓ |
| dense-cf1.25-fs8（capacity-factor dropping） | 总 ≈200 / 临时 ≈125 GiB ✓ | ✓ |

每张卡 HBM 上限：扣除 BFCAllocator 开销后约 268 GiB。

基于 `ragged_dot` 的 dropless 路径（`sgf`）在 pdbs=7 一律 OOM，无论 PP 还是 FSDP。**DeepEP 是内存效率最高的 dropless 路由路径**；基于 `ragged_all_to_all` 的 `sgf` 物化一个完整的 `num_ranks × tokens × hidden` 接收缓冲区（因为它是普通的 all_to_all，不是拓扑感知的 dispatch），而 DeepEP 把每张卡的接收缓冲区按 `num_worst_tokens × hidden`（实际 fan-in 的紧致上界）来分配。`sgd-v3` 与 `sgf` 在同 pdbs 下的 +44 GiB temp 差异，正是接收缓冲区差异。capacity-factor dropping 路径（`dense-cf1.25`）更便宜，因为不论路由情况，每个 expert 看到的都是固定大小的 `capacity_factor × tokens / num_experts` 张量，所以 fan-in 张量被 `num_experts × capacity_factor × tokens / num_experts × hidden = capacity_factor × tokens × hidden` 上界 —— 比 `sgf` 的 `num_ranks × tokens × hidden` 节省约 5×。

## `dense_matmul` PP=8 的胜利为什么不能转移到 `sparse_matmul`-DeepEP PP=8

朴素的 roofline 分析认为 PP=8 应该击败 FSDP=8，因为它把每步的 DCN 通信量降低了 —— 流水线只在相邻 stage 间传送 hidden-state activations，而不需要每个 microbatch 都做一次完整的参数 all-gather。对 DS3-671B pdbs=7 来说，这相当于每步 DCN 字节数减少 ~3-4×。**对 `dense_matmul` 路径（`dense-cf1.25`），预测成立（镜像默认 XLA 下 +15.5 %、FSDP 调优 XLA 后仍 +1.34 %）。对 `sparse_matmul`-DeepEP（`sgd-v3`），不成立（镜像默认 XLA 下 -8.3 %、FSDP 调优后 -12.0 %）。**

`dense_matmul` 路径的胜利在 `sparse_matmul`-DeepEP 上消失，是因为 dropless `sparse_matmul` 路径独有的三种结构性成本叠加在一起，超出了 every-step-bytes-saved 的收益。关键变量是 **per-stage 计算的均匀性**：`dense_matmul` 中的 capacity-factor dropping 强制 per-expert 张量形状均匀（每个 expert 永远看到 `capacity_factor × tokens / num_experts` 个 token，不多不少）；`sparse_matmul`-DeepEP 路由的是真实的 top-k 分配，所以 per-stage 计算量随 256 个 expert 的路由 skew 而变化。

### 1. `nn.scan`-PP carry 把 pipeline 调度强制串行化（两个分支都受影响，但只有 `sparse_matmul` 真的付出代价）

MaxText 的 pipeline 实现 lower 到 `jax.lax.scan` over stage 轴，并带一个 `loop_state` carry。XLA 的 latency-hiding scheduler **不能跨迭代边界移动工作**，因为每次迭代的输入依赖于前一次迭代的输出。14563（sgd-pp8 pdbs=7 含 HLO dump，文件 `module_*.jit_train_step.gfx950_gpu_after_optimizations.txt`）的 HLO 显示，每个 `collective-permute-start` 紧跟着对应的 `collective-permute-done`，中间没有任何计算 —— scheduler 没有空闲来做 prefetch。同样的 scan 结构在 `dense-cf1.25`（14565）也存在，但每个 stage 的计算在所有 stages 间是均匀的（capacity-factor dropping 强制每个 expert 看到 `capacity_factor × tokens / num_experts` 个 token），所以 `collective-permute` 总是与"最快到达 rank"的计算边界对齐 —— 也就是说，暴露出来的等待最少。在 `sparse_matmul`-DeepEP 下，per-stage 计算随路由 skew 变化（一些 stage 派发的 token 更少），所以 `collective-permute` 总是与"最慢到达 rank"的等待对齐 —— 也就是说，暴露出来的等待最多。

经验上，`--xla_gpu_enable_pipelined_p2p=true`（即 `pipelined_all_gather` 在 P2P 上的对应物）在 `sparse_matmul` 路径上完全没有效果：没有空隙可填，自然没东西可流水化。PP=8 调优 sweep 进一步证实了这点 —— 14662 的 HLO（带 winning `overlap2` 配方 + HLO dump 开启）里每个 `collective-permute-start` 仍然标着 `is_pipelined=false`。已测试的 XLA flag 都不会改变这个 carry-dep。

### 2. `collective-permute` 是一个 per-call 的同步集合通信

每个 pipeline send/recv 都是 JAX 层的同步操作。Per-call rendezvous wait 累积为 `Σᵢ max_r(Tᵢ,r)`，而不是 `max_r(Σᵢ Tᵢ,r)` —— 也就是说，逐层的同步点会复合，而不会跨 rank 平均掉。从 14563 trace 中：每张 GPU 每步纯 rendezvous wait 4.4 秒；其中约 3.5 秒是暴露在每步耗时里的（即没有与任何东西重叠）。

这对 *`sparse_matmul`*（dropless）分支尤为致命：256 个 expert 上的 top-k 分配 skew 会在每一层产生 per-rank 的计算不均衡。在 FSDP 下，不均衡在 per-step 的 all-gather / reduce-scatter 边界上被摊销（每个 rank 在每层之前都做完整的 all-gather，所以 per-rank 不均衡在 step 粒度上抵消）。在 PP 下，不均衡通过 per-layer `collective-permute` sync 被放大（不均衡每层都体现一次，而不是每步只体现一次）。`dense_matmul` 分支的 capacity-factor 机制恰好把 per-stage 张量形状抹平了，所以这第三种成本在那里消失。

### 3. DeepEP 的 per-microbatch 固定开销

PP=8 配 `pipeline_parallel_layers=56`、`num_pipeline_microbatches=8`、V=7（`num_layers_per_pipeline_stage=1`）的组合，意味着 **8 microbatch × 7 chunks × 8 stages = 448 次 DeepEP dispatch+combine round-trip / 步**。DeepEP kernel 有固定的启动开销（~0.5 ms）加 per-rank IPC 握手（最理想 ~1 ms，skew 下更多）。这 448 次 round-trip 单独贡献 ~2 秒/步的固定开销，而 FSDP 不需要付（FSDP 每层每个 token group 只发一次 dispatch+combine，总共 ~58 × 7 = ~406 次调用，量级类似，但不在 bubble 关键路径上）。

### Bubble 计算

Pipeline bubble 占比 = (num_stages − 1) / (num_microbatches × V + num_stages − 1) = 7 / (8×7 + 7) = 7 / 63 ≈ **11.1 %**。

仅这一项就给 PP 收益设了上限 —— PP 必须在 FSDP comm 上节省 *超过* 11 % 的步耗时才能净赚。FSDP 在本机器上暴露的 comm 节省约 10 秒（27 秒/步中的 ~37 %），所以原则上 *如果* bubble 是唯一成本，PP 是有可能赢的。但 bubble + rendezvous-放大的 straggler wait + DeepEP 固定开销加在一起，超过了 FSDP 暴露的 comm：

```
PP 步耗时 ≈ FSDP 步耗时 − （FSDP 暴露 comm）
                       + （PP bubble）
                       + （PP rendezvous-放大 wait）
                       + （PP DeepEP 固定开销）
         ≈ 27.6  − 10.0  + 3.0  + 3.5  + 2.0
         ≈ 26.1 秒/步（预测）
经验 PP 步耗时 = 30.5 秒（clean） / 33.9 秒（含 profiler）
```

经验值比预测多出来的 ~4 秒/步，来自 `nn.scan`-PP layout 下的 XLA scheduler 开销（每次迭代有自己的 scheduling context，跨迭代优化被关闭）。

## `combine_hang` 修复带来什么影响？

4 月 30 日上游的 `fix/deepep/combine_hang` 合并给 `moe_dispatch`/`moe_cached_dispatch`/`moe_combine` 的 FFI lowering 加了 `has_side_effect=True`，并在 `moe_combine` 上多增加了第三个输出 `send_head_work`。14550（NEW image）vs 14539（OLD image）的 sgd-pp8 pdbs=8 对比显示：

| | TGS（steps 9-14） | step 14 loss |
|---|---:|---:|
| 14539 OLD（无 `combine_hang`） | 972.2 | 10.136 |
| 14550 NEW（有 `combine_hang`） | 966.9 | 10.136 |

**Δ = -5 TGS、-0.5 %** —— 在 step-to-step jitter 范围内（~7 TGS std dev）。Loss 逐位相同。这个修复在稳态下**实质上是性能中性的**：`has_side_effect=True` 阻止 XLA DCE 或重排 IPC ops，但本 MoE 工作负载里 XLA 本来就没有用过这个调度自由度（这些 call 已经有硬性的跨 rank 数据依赖）。纯正确性收益。

## FSDP=8 调优实验（仅 `sgd-v3`，36 次运行）

clean FSDP=8 基线落在 1017.7 TGS（`sgd-v3`、profiler 关闭、无 XLA dump、镜像默认 XLA 标志）。原 `deepep-gmm-maxtext-v26.2.tar` 镜像上的历史 sgd-fs8 基线是 1097 TGS —— 大约 5 % 的差距分别来自：(a) 4 月 14 日到 4 月 30 日间的集群抖动，(b) `combine_hang` 正确性修复（单独验证 ≤ 0.5 %）。

我通过在每次提交前编辑 `train_env.sh`，测试了 36 个不同的 XLA-flag / NCCL-env / memory-fraction 组合（每次 `submit.sh` 调用都会冻结自己的 artifact，所以 pending jobs 不受后续编辑影响 —— 详见 `submit.sh:53-69` 的 artifact 构建机制）。测试的假设：

1. **跨迭代重叠 flags** —— `xla_gpu_enable_while_loop_double_buffering`、`xla_gpu_enable_pipelined_all_gather/reduce_scatter/all_reduce`。假设：在当前迭代计算结束前预取下一次迭代的 all-gather。
2. **异步流优先级** —— `xla_gpu_enable_highest_priority_async_stream`。假设：让 async collectives 比 compute 享有更高优先级。
3. **per-call 并发** —— `xla_gpu_experimental_parallel_collective_overlap_limit ∈ {2, 4, 8}`。假设：允许更多 in-flight async collectives。
4. **强制 LHS** —— `xla_gpu_enable_latency_hiding_scheduler=true`。假设：镜像默认可能没开。
5. **Combiner threshold 扫描** —— `xla_gpu_all_gather_combine_threshold_bytes` 与 `xla_gpu_reduce_scatter_combine_threshold_bytes`，扫描 256 MiB / 384 MiB / 512 MiB / 768 MiB / 1 GiB / 2 GiB / 4 GiB；包括"两者一起改"和"只改 ag"两种。假设：镜像默认要么过小（collectives 碎片化）要么过大（collectives 串行化）。
6. **NCCL 调优** —— `NCCL_BUFFSIZE=16 MiB`、`NCCL_NCHANNELS_PER_NET_PEER=8`、`NCCL_IB_QPS_PER_CONNECTION=8`、`NCCL_PROTO=Simple`，以及它们的组合。
7. **Memory fraction** —— `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`（vs 默认 `.93`）。假设：给 prefetch buffer 更多 HBM headroom。

跑完 36 次后，最佳 recipe 都收敛到 **+11.6 % TGS over baseline**（1017.7 → 1135.7-1135.8）。最关键的杠杆是 all-gather combine threshold；reduce-scatter combiner、NCCL channel 数、memory fraction、强制 LHS 这些，叠加在 `ag1G_only` 上各自 <1 %。

### 排行榜（top 8 + bottom 9）

| 排名 | Profile | TGS | step | Δ% | 配方 |
|---:|---|---:|---:|---:|---|
| 1 | GLP_ag1G_chan8_mem95（14624） | 1135.8 | 25.24 秒 | +11.61 % | ag=1 GiB + NCCL_NCHANNELS=8 + mem_frac=.95 |
| **2** | **G_ag1G_only**（14602） | **1135.7** | **25.25 秒** | **+11.60 %** | 只设 `--xla_gpu_all_gather_combine_threshold_bytes=1073741824` |
| 3 | G_ag512M_only（14620） | 1135.2 | 25.26 | +11.55 % | 仅 ag=512 MiB |
| 4 | G_combine384M（14609） | 1135.0 | 25.26 | +11.53 % | ag 与 rs 都设为 384 MiB |
| 5 | GLP512_full_mem95（14613） | 1130.6 | 25.36 | +11.10 % | ag/rs=512 MiB + NCCL_NCHANNELS=8 + mem_frac=.95 |
| 6 | GLD_combine1G_chan8_LHS（14606） | 1130.1 | 25.37 | +11.05 % | ag/rs=1 GiB + NCCL_NCHANNELS=8 + LHS=true |
| 7 | G_ag2G_only（14621） | 1126.3 | 25.46 | +10.67 % | 仅 ag=2 GiB |
| 8 | G_combine512M（14598） | 1119.2 | 25.62 | +9.98 % | ag/rs 都设为 512 MiB |
| ... | （其余正向结果，详见附录表） | | | | |
| baseline | 14572 sgd-fs8c | 1017.7 | 28.21 | 0 | 镜像默认 XLA_FLAGS（ag、rs combine threshold 都 8 GiB） |
| | （负面结果，按影响排序） | | | | |
| | N_nccl_proto_simple（14594） | 1004.0 | 28.56 | -1.34 % | NCCL_PROTO=Simple |
| | O_nccl_combo（14595） | 1001.8 | 28.64 | -1.56 % | NCCL buffsize+channels+qps 一起 |
| | K_nccl_buffsize16M（14591） | 997.6 | 28.75 | -1.98 % | NCCL_BUFFSIZE=16 MiB |
| | A_doublebuffer（14579） | 990.3 | 28.97 | -2.69 % | `--xla_gpu_enable_while_loop_double_buffering=true` |
| | AB（14582） | 978.4 | 29.31 | -3.86 % | A + highest_priority_async_stream |
| | C_overlap_limit2（14581） | 975.6 | 29.40 | -4.14 % | `--xla_gpu_experimental_parallel_collective_overlap_limit=2` |
| | H_combine4G（14588） | 973.0 | 29.49 | -4.39 % | ag/rs 都 4 GiB（过粗 —— 接近 8 GiB 默认） |
| | J_async_unconstrained（14590） | 964.8 | 29.73 | -5.19 % | overlap_limit=8 |
| | ABC（14583） | 953.6 | 30.08 | -6.29 % | A+B+C 叠加 |

### 为什么 all-gather combine threshold 最关键

**Docker 镜像的默认设置是 `--xla_gpu_all_gather_combine_threshold_bytes=8589934592`（8 GiB），`reduce_scatter_combine_threshold_bytes` 同样。** baseline（14572 sgd-fs8c）日志中观察到的完整继承 XLA_FLAGS：

```
--xla_gpu_all_gather_combine_threshold_bytes=8589934592      ← 8 GiB（远远过粗）
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592  ← 8 GiB
--xla_gpu_enable_latency_hiding_scheduler=True
--xla_gpu_memory_limit_slop_factor=95
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_autotune_level=0
--xla_gpu_enable_all_gather_combine_by_dim=FALSE
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false
--xla_gpu_enable_command_buffer=''
```

这解释了为什么"强制 LHS"（D）只 +1.72 % —— LHS 本来就是开的。也解释了 combiner-threshold 扫描的形状：

| ag threshold | rs threshold | 每步 ag 估计数量 | TGS | Δ% |
|---:|---:|---:|---:|---:|
| 8 GiB（默认） | 8 GiB（默认） | 1（全部融合） | 1017.7 | 0 |
| 4 GiB | 4 GiB | ~1-2 | 973.0 | -4.4 % |
| 2 GiB | 2 GiB | ~2-3 | 1098.1 | +7.9 % |
| 1 GiB | 1 GiB | ~4-5 | 1077.2 | +5.85 % |
| 768 MiB | 768 MiB | ~5-6 | 1081.9 | +6.3 % |
| 512 MiB | 512 MiB | ~8-10 | 1119.2 | +9.98 % |
| 384 MiB | 384 MiB | ~10-13 | 1135.0 | +11.53 % |
| 1 GiB | **8 GiB（默认）** | ~4-5 | **1135.7** | **+11.60 %** |
| 512 MiB | **8 GiB（默认）** | ~8-10 | 1135.2 | +11.55 % |
| 2 GiB | **8 GiB（默认）** | ~2-3 | 1126.3 | +10.67 % |

**镜像 8 GiB 的阈值导致 XLA 把每一步的所有 all-gather 融合成一次串行的"巨型调用"，这一调用必须在任何层的计算开始之前完成。** 这与"prefetch / overlap"的方向恰好相反 —— 它是一道硬屏障。把 all-gather threshold 降到 384 MiB - 2 GiB 会把这次"巨型 all-gather"拆成 4-13 次小一些的 all-gather，XLA 的 latency-hiding scheduler 就能把它们与每层计算交错排程，回收每步约 3 秒的暴露 comm。

**Reduce-scatter 则相反 —— 让它保持在 8 GiB。** 反向 reduce-scatter 本来就很大（跨层累积的 gradient 缓冲区每层就有几百 MiB 到几 GiB）。把它们融合成一个 8 GiB 的 chunk 没问题，因为它们本来就够大；而且反向 pass 计算密度更高（gradient 计算每字节的算术运算更多），所以 RCCL launch 开销影响更小、流水化也没那么重要。把 reduce-scatter 也拆开（比如 `G_combine1G` 把 BOTH 都设成 1 GiB）实际上有害，因为它增加了 RCCL launch 开销但没有相应的重叠收益。这就是为什么 `G_ag1G_only`（+11.60 %，只拆 ag）和 `G_combine1G`（+5.85 %，两个都拆）有这么大差距。

`G_combine384M`（+11.53 %）能跟 `G_ag1G_only`（+11.60 %）打平，是因为在 384 MiB 下，那些原本被融合到 8 GiB 的 reduce-scatter 现在被拆成 ~3-4 个仍然较大的 chunk（每个 gradient 几百 MiB），所以反向开销有上限 —— 但拆开也不能带来额外收益。所以 384M-both 与 1G-ag-only 在 +11.5-11.6 % 这个 plateau 上功能等价。

### 负面结果（用于假设排查）

用户的初始假设是"all-gather/reduce-scatter 与 compute 没有重叠，理想情况下应该 prefetch"。实验矩阵表明 prefetch 类的 flags **在本工作负载上全部有害**：

- **`while_loop_double_buffering=true`（-2.69 %）**：跨迭代重叠 `train_step` 的 `while` body。代价是 HBM 要存下一次迭代的输入，这迫使 XLA 重计算或溢出，开销超过了（有限的）重叠窗口收益。
- **`pipelined_all_gather/all_reduce/reduce_scatter=true`（E，OOM）**：启用一个 XLA pass，把每个 collective 拆成两半跨迭代执行（一半运行在前一次迭代上）。同样的内存代价；本工作负载下 buffer 大到 OOM。
- **`experimental_parallel_collective_overlap_limit=2 / 4 / 8`（-4 % 到 -5 %）**：允许多个 in-flight async collectives。RCCL 争用（每个分到的 NIC channel 更少）超过了并发收益。
- **`highest_priority_async_stream=true` 单独使用（+0.32 %）**：镜像默认已经把 async stream 优先级排得足够高了。
- **`enable_latency_hiding_scheduler=true`（+1.72 %）**：小幅收益 —— 证实镜像默认 *没* 开 LHS。但一旦把它打开，就没什么能再叠加上去了。

教训：**在本 MoE 工作负载的 FSDP=8 上，瓶颈是 RCCL launch overhead，不是 comm-vs-compute 的暴露空隙。** 一旦通过提高 all-gather combine threshold 来折叠 launch count，额外的调度调整对 TGS 几乎没有影响 —— 因为剩余的 RCCL 已经接近峰值带宽运行了。

（本文后面的 PP=8 调优 sweep 显示，*这里所有的*符号在 PP=8 上都是反过来的。详见 [§ FSDP=8 与 PP=8 之间的符号反转](#为什么获胜的-flag-是-overlap_limit2pp8和-fsdp8-的符号反转)。）

### 为什么 dense-cf1.25 看到 +20.5 %、而 sgd-v3 只有 +11.6 %

同一个 flag 应用到两条路径，`dense-cf1.25` 的提升几乎是 `sgd-v3` 的两倍。可能的机制：

- **dense-cf1.25 每层的内在计算密度更高** —— capacity-factor padding 给每个 expert 一个固定大小的 GEMM，与路由无关，且没有 MoE dispatch/combine 开销。所以每层的计算窗口大且不间断，与流水化的 all-gather 重叠效果非常好。
- **sgd-v3 的每层计算更短、更碎片化** —— 每层要付 DeepEP `moe_dispatch`（~1-2 ms IPC）、然后真正的 GEMM、再 `moe_combine`（~1-2 ms）。all-gather 仍然能与 dispatch-GEMM-combine 序列重叠，但每段都更小，可达到的重叠窗口就更小。
- **MoE 路由 skew 天花板** —— sgd-v3 的每层计算时间随 top-k 分配 skew 而变化。最慢的 rank 的 per-layer wall 决定下一次 all-gather 何时启动（在拆分后的流水化版本里），所以重叠天花板被 `Σᵢ max_r(layer_iᵢ,r)` 上界限制。dense-cf1.25 跨 ranks 的 per-layer wall 是均匀的，所以它的天花板是 `max_r(Σᵢ layer_iᵢ,r) = Σᵢ layer_i`（没有 skew 放大）。

这是一个经验性的证明：FSDP 在 dropless-MoE 上的 comm-overlap 从根本上受路由 skew 方差的限制 —— 这是任何同步分布式训练下的 dropless MoE 的结构性属性，不限于 DS3-671B 或本机器栈。capacity-factor dropping（`dense-cf1.25`）按构造消除了这种方差，因此在调优后的配方下它赢得更多。

## PP=8 调优实验（sgd-v3 + dense-cf1.25，36 次运行，5/01）

把 FSDP 调优过的 `ag=1 GiB` flag 部署到 `configs/deepseek3-671b.env.sh` 之后，又做了一轮专门针对 PP=8 的调优。核心发现：**PP=8 dense-cf1.25 + 仅 `overlap2` 比 FSDP=8 dense 生产配方高 +1.34 %**，并且 **FSDP 调优过的 env flag 在 PP=8 上符号相反**（dense 上 -6.6 %）。PP=8 sgd-v3 比生产状态基线高 +4.66 %，但仍比 FSDP=8 sgd-v3 低约 12 %，这是结构性差距（DeepEP per-microbatch 开销 + scan-carry 串行化 + pipeline bubble），无法通过 XLA/NCCL 旋钮关闭。

### 已部署的 `configs/deepseek3-671b.env.sh`（FSDP 调优过的 `ag=1 GiB`）对 PP=8 的影响

4 月 30 日 FSDP=8 sweep 把 `--xla_gpu_all_gather_combine_threshold_bytes=1073741824` 提交到 per-model env 文件，因为它给 FSDP=8 带来了 +11.6 %（sgd-v3）/ +20.5 %（dense-cf1.25）。在 PP=8 下，这个 flag 的符号 **相反**：

| 路径 | ag=1 GiB（继承 env 文件） | ag=8 GiB（镜像默认） | Δ |
|---|---:|---:|---:|
| sgd-v3 PP=8 | 955.3（14638） | 965.2（14639） | -1.0 %（在 jitter 范围内） |
| dense-cf1.25 PP=8 | 1080.2（14640） | 1156.9（14641） | **-6.6 %（真实，远超 jitter）** |

机制是 path-dependent 的。对 FSDP=8，all-gather 是覆盖整个 DCN ring（8 节点 × 8 GPU）的 expert 权重，所以把它折叠成一次 8 GiB 调用就形成硬屏障；切到 1 GiB 让 XLA 的 latency-hiding scheduler 把若干块和 per-layer 计算交错调度（+11.6 至 +20.5 %）。对 PP=8，all-gather 在 ICI（节点内，replica_groups=[8,8]，沿 `ici_expert_parallelism=8` 轴）上覆盖一个小得多的 per-stage 权重张量；切小并不能帮上忙，反而徒增 RCCL launch 开销。

所以 `configs/deepseek3-671b.env.sh` 在被无条件应用到所有 `deepseek3-671b` 提交之前必须按 topology 守护 —— 详见 [§ 推荐部署方式](#推荐部署方式)。

### sgd-v3 PP=8 排行榜（固定 8 节点列表，pdbs=7，steps 9-14）

所有 Δ% 都是 vs `pp8-baseline-ag1G`（job 14638，955.3 TGS） —— 即生产状态主基线（继承 env 文件 ag=1 GiB）。Δ vs `pp8-restore_ag_default`（job 14639，965.2 TGS，镜像默认）作为对比也列出。

| 排名 | Profile | TGS | step | Δ vs ag=1G | Δ vs ag=8G | 配方 |
|---:|---|---:|---:|---:|---:|---|
| 1 ⭐ | `pp8-d_full_stack`（14669） | 1001.2 | 28.64 s | **+4.80 %** | +3.73 % | cp_decomp_1G + async_priority + overlap2 + NCCL_PROTO=Simple，ag=8 GiB 默认 |
| 2 | `pp8-d_overlap2_async`（14668，14674 重测 n=2 平均=999.8） | 1000.2 | 28.67 s | **+4.70 %** | +3.62 % | overlap2 + async_priority，ag=8 GiB 默认 |
| 3 | `pp8-d_overlap2_proto`（14670） | 999.3 | 28.69 s | +4.61 % | +3.53 % | overlap2 + NCCL_PROTO=Simple，ag=8 GiB 默认 |
| 4 | `pp8-cp1G_async_ov2`（14667） | 995.4 | 28.80 s | +4.20 % | +3.13 % | cp_decomp_1G + async_priority + overlap2，ag=1 GiB |
| 5 | `pp8-d_cp1G_async_ov2`（14666） | 995.0 | 28.82 s | +4.15 % | +3.08 % | cp_decomp_1G + async_priority + overlap2，ag=8 GiB 默认 |
| 6 | `pp8-overlap2`（14649，第 1 次测量） | 994.9 | 28.82 s | +4.14 % | +3.08 % | 仅 overlap2，ag=1 GiB |
| 7 | `pp8-evidence`（14662，overlap2 + 带 HLO+xplane 的 profiled run） | 989.2 | 28.99 s | +3.55 % | +2.49 % | overlap2 + ag=1 GiB + profiler+HLO |
| 8 | `pp8-d_cp1G_async`（14657） | 988.3 | 29.01 s | +3.45 % | +2.39 % | cp_decomp_1G + async_priority，ag=8 GiB 默认 |
| | （overlap2 + ag=1G，**n=3 平均** = 987.6，标准差 8.2） | | | **+3.38 %** | +2.32 % | 仅 overlap2，ag=1 GiB |
| | `pp8-cp_decomp_1G_async`（14665，重测） | 987.0 | 29.05 s | +3.32 % | +2.26 % | cp_decomp_1G + async_priority，ag=1 GiB |
| | `pp8-nccl_proto_simple`（14664） | 987.2 | 29.04 s | +3.34 % | +2.28 % | NCCL_PROTO=Simple，ag=1 GiB |
| | `pp8-overlap8`（14660） | 981.7 | 29.21 s | +2.76 % | +1.71 % | overlap_limit=8，ag=1 GiB |
| | `pp8-async_priority`（14647） | 972.1 | 29.50 s | +1.76 % | +0.71 % | 仅 async_priority，ag=1 GiB |
| | `pp8-d_overlap2`（14661） | 971.8 | 29.51 s | +1.72 % | +0.68 % | 仅 overlap2，ag=8 GiB |
| | `pp8-cp_decomp_1G`（14644） | 970.7 | 29.54 s | +1.62 % | +0.57 % | cp_decomp 1 GiB 阈值（对 392 MiB c-p 是 no-op），ag=1 GiB |
| | `pp8-mem95`（14663） | 968.8 | 29.60 s | +1.42 % | +0.38 % | XLA_PYTHON_CLIENT_MEM_FRACTION=.95，ag=1 GiB |
| 基线 | **`pp8-baseline-ag1G`**（14638） | **955.3** | 30.02 s | 0 | -1.04 % | env 文件继承（`ag=1 GiB`） |
| | （`pp8-restore_ag_default`，14639） | 965.2 | 29.71 s | +1.04 % | 0 | 镜像默认 `ag=8 GiB` |
| | `pp8-cp_decomp_256M`（14648） | 961.0 | 29.84 s | +0.60 % | -0.44 % | cp_decomp 256 MiB 阈值（会分解 392 MiB c-p） |
| | `pp8-double_buffer`（14646） | 962.8 | 29.78 s | +0.78 % | -0.25 % | `while_loop_double_buffering=true`；loss 偏离基线 0.02 |
| | （负面结果，按影响排序） | | | | | |
| | `pp8-d_chan8`（14653） | 958.9 | 29.90 s | +0.38 % | -0.65 % | NCCL_NCHANNELS_PER_NET_PEER=8，ag=8 GiB |
| | `pp8-nccl_chan8`（14652） | 938.6 | 30.57 s | -1.74 % | -2.75 % | NCCL_NCHANNELS_PER_NET_PEER=8，ag=1 GiB |
| | `pp8-overlap4`（14659） | 921.6 | 31.13 s | -3.53 % | -4.52 % | `parallel_collective_overlap_limit=4` |
| | `pp8-pp_p2p`（14643） | 904.6 | 31.73 s | **-5.31 %** | -6.28 % | `--xla_gpu_enable_pipelined_p2p=true` |
| | `pp8-pp_all_reduce`（14650） | OOM | — | — | — | `--xla_gpu_enable_pipelined_all_reduce=true`（217 GiB temp） |
| | `pp8-pp_all_gather`（14651） | OOM | — | — | — | `--xla_gpu_enable_pipelined_all_gather=true`（302 GiB temp） |

### dense-cf1.25 PP=8 排行榜

| 排名 | Profile | TGS | step | Δ vs ag=1G | Δ vs ag=8G | Δ vs FSDP=8 prod |
|---:|---|---:|---:|---:|---:|---:|
| 1 ⭐ | `pp8-dense_d_overlap2`（14673，第 2 次测量） | **1233.0** | 23.25 s | **+14.1 %** | **+6.58 %** | **+2.08 %（PP 击败 FSDP）** |
| 2 | `pp8-dense_d_overlap2`（14672，第 1 次测量） | 1215.2 | 23.60 s | +12.5 % | +5.04 % | +0.61 % |
| | （14672/14673 平均，n=2） | **1224.1** | 23.42 s | **+13.3 %** | **+5.81 %** | **+1.34 %** |
| 3 | `pp8-dense_d_full_stack`（14671） | 1206.3 | 23.78 s | +11.68 % | +4.27 % | -0.13 %（基本持平） |
| ref | （FSDP=8 dense + ag=1 GiB，14629） | 1207.9 | — | — | — | 0 |
| 4 | `pp8-dense_d_overlap2_async`（14676） | 1190.4 | 23.93 s | +10.20 % | +2.90 % | -1.45 %（async 损害 dense） |
| | （`pp8-restore_ag_default`，dense，14641） | 1156.9 | 24.79 s | +7.10 % | 0 | -4.22 % |
| 基线 | **`pp8-baseline-ag1G`** dense（14640） | **1080.2** | 26.60 s | 0 | -6.62 % | -10.57 % |

**重要的 path-dependent 发现**：在 dense-cf1.25 上加 `async_priority` 比仅 overlap2 退步 -2.7 %（1190.4 vs 1224.1 平均）。dense **仅 overlap2 是赢家配方**。sgd-v3 上加 async 在 overlap2 之上略有提升（+1.5 %，平均 999.8 vs 984.9）。配方的差异从机制上说得通：async_priority 在 schedule 紧张时帮忙（sgd-v3 的 MoE skew 造成 per-rank 落后者）；在 schedule 已经比较松的情形下反而损害（dense-cf1.25 计算均匀）。

**核心结论：** 最简单的配方 —— 仅 overlap2 —— 让 `dense-cf1.25` PP=8 **达到（甚至略胜）FSDP=8 dense 生产配方**。在 dense-cf1.25 上加上 full_stack 的额外 flag（cp_decomp、async_priority、NCCL_PROTO=Simple）反而带来微小回退 —— 仅 overlap2 是 dense-cf1.25 PP=8 的赢家配方。

### 为什么获胜的 flag 是 `overlap_limit=2`？PP=8 和 FSDP=8 的符号反转

`--xla_gpu_experimental_parallel_collective_overlap_limit=N` 控制 XLA latency-hiding scheduler 同时可以有多少个 async collectives 在飞。镜像默认是 1（同时只有一个 collective）。

从 14662（overlap2 + ag=1 GiB，作为证据采集的 profiled run）的 HLO，每一个 `collective-permute-start` 仍然标着 `is_pipelined=false` —— 也就是说 LHS scheduler 仍然不能跨迭代边界移动 c-p（与击败 14643 `pipelined_p2p` 同样的 `nn.scan`-PP carry-dep）。所以 overlap2 **不是** 通过 enable pipelining 来帮忙的。

overlap2 **真正** 能做的事：让 train_step HLO 中的 **5 个不同的 `collective-permute-start` 算子**（channels 41、42、105、200、201 —— 在前向和反向 stage rotation 之间分布）以及 FSDP 风格的 ICI collectives（MoE expert 权重 all-gather、`DeepSeekMoeBlock_0/shard_map/psum` all-reduce、梯度 reduce-scatter）在每次迭代内并发执行。在 overlap=1 下，即使数据依赖不强制串行化它们也会串行；在 overlap=2 下，两个可以在不同 RCCL stream 上并发跑。

FSDP=8 sweep 看到 `overlap_limit=2/4/8 = -4 至 -5 %`，是因为 FSDP 的 ICI all-gather 已经把 ICI fabric 吃满（再加一个并发 collective 就引起争用）。PP=8 的 ICI all-gather 小很多（per-stage 权重子集），DCN c-p 是 **per-stage 跳的点对点通信**，fabric 利用率低很多，所以两个并发 collective 在不同 fabric 上（一个 ICI，一个 DCN p2p）能真正重叠起来。这是 flag 在 PP→FSDP 间符号反转的结构性原因。

`overlap_limit=4` 已经引起争用（-3.53 %）；`overlap_limit=8` 又部分恢复（+2.76 %）但仍打不过 overlap=2（+3.38 % 平均）。**2 路 stream 这个甜点** 与 PP=8 schedule 中独立 fabric 的数量一致（ICI + DCN）。

其他 Wave 2 flag（`cp_decomp_1G`、`async_priority`、`NCCL_PROTO=Simple`）单独都在 **+3.3-3.5 %** 范围（与单 overlap2 在 jitter 内）—— 它们都是在解决同一个 DCN 调度瓶颈，互相之间不能很好地叠加。把四个全堆起来从 +3.4 % 单 flag 上限提升到 +4.7-4.8 %，3 个额外 flag 只换来 +1.4 pp 的边际提升。对 sgd-v3 推荐更简单的 2-flag（`overlap2 + async_priority`）；对 dense-cf1.25 单 `overlap2` 已足够。

不同 regime 间的系统性符号反转汇总如下 —— 每个在 FSDP=8 上有用的 flag 在 PP=8 上有害（或反之亦然），且机制相同（fabric 利用率差异）：

| Flag | FSDP=8 Δ | PP=8 Δ |
|---|---:|---:|
| `parallel_collective_overlap_limit=2` | **-4.14 %** | **+3.38 %** |
| `NCCL_NCHANNELS_PER_NET_PEER=8` | +4.26 % | -1.74 % |
| `pipelined_p2p=true` | no-op | -5.31 % |
| `NCCL_PROTO=Simple` | -1.34 % | +3.34 % |
| `all_gather_combine_threshold=1 GiB`（vs 8 GiB 镜像默认） | **+11.60 %** sgd / **+20.5 %** dense | -1.0 % sgd / **-6.6 %** dense |

**从一个 regime 调出来的结果不会转移到另一个。**

### 为什么 sgd-v3 PP=8 没能达到 FSDP=8 持平（结构性的 -12 % 差距）

虽然 dense-cf1.25 PP=8 + overlap2 达到 FSDP=8 持平，sgd-v3 PP=8 的最佳栈停在 1001 TGS —— 仍比 1135.7 的 FSDP=8 sgd-v3 生产配方低 -11.95 %。结构性成本分解依然成立：

1. **`nn.scan`-PP carry 把 schedule 串行化**，没有 XLA flag 能打开 carry-dep。14662（overlap2）的 HLO 证实即使在赢家配方下 `collective-permute-start` 还是 `is_pipelined=false`。
2. **`collective-permute` 是 per-call 同步**。5 个 `collective-permute-start` 算子 × 8 microbatches × 8 stages × 2（前向+反向）= 每步约 640 个 c-p 调用，每个的 rendezvous wait 在 MoE skew 下跨 rank 复合。
3. **DeepEP per-microbatch 固定开销**：8 microbatches × 7 V-chunks × 8 stages = 每步 448 次 DeepEP dispatch+combine 来回，每次约 3 ms = 约 1.3 s 固定开销（仅 sgd-v3）。
4. **Bubble fraction = 7/63 = 11.1 %** —— 与 XLA 调优无关的硬下界。

dense-cf1.25 完全不付（3）的代价（不用 DeepEP）。它也少付（2）的代价，因为 per-stage 计算均匀（capacity-factor dropping），所以 per-rank 落后者 skew 很小。只有（1）和（4）适用，而 overlap2 的 2-fabric 并发足以吸收（1）的大部分代价 —— 这就是持平结果的原因。

### PP=8 负面发现（含 Δ%）—— 给后续 agent 跳过用

| Profile | Δ vs sgd-v3 ag=1G | 备注 |
|---|---:|---|
| `pp_p2p` | -5.31 % | 启用 c-p 跨迭代 prefetch buffer，但 `nn.scan` carry-dep 让 LHS 实际上用不了那个 buffer；纯开销 |
| `overlap_limit=4` | -3.53 % | 越过 2-fabric 甜点，RCCL 争用超过并发收益 |
| `nccl_chan8`（与 ag=1G） | -1.74 % | 额外 NCCL channel 帮不上 per-stage 2-rank c-p；轻微回退 |
| `pp_all_reduce` | OOM 217 GiB temp | 与 FSDP=8 OOM 一致；pipelined_all_reduce buffer 在 PP=8 也超 HBM |
| `pp_all_gather` | OOM 302 GiB temp | 同样的 OOM 模式 |
| `cp_decomp_256M` | +0.60 % | 激进 c-p 分解（确实分解了 392 MiB 的 c-p）—— PP=8 上没有任何收益；LHS 仍然不能跨 carry-dep 流水 |
| `cp_decomp_1G` | +1.62 %（jitter 内） | PP=8 上是 no-op（392 MiB c-p < 1 GiB 阈值，所以不分解） |
| `double_buffer` | +0.78 % | jitter 内；loss14 偏离基线 0.02（数值效应，提示累加顺序变了） |
| `mem95` | +1.42 %（jitter 内） | 与 FSDP 发现一致（两条路径上都没有可测影响） |

## `remat_policy` 灵敏度（覆盖 PP=8 *和* FSDP=8 —— 通用结论：pdbs=7 下 `full` 最优）

YAML 默认 `remat_policy: 'full'`（最慢的重算、最低 HBM）原本被怀疑对 PP=8 来说过于保守，因为 4/30 的内存可行性矩阵显示有明显的 HBM 余量（sgd-pp8 总 253 / temp 178 GiB，dense-pp8 总 ≈200 / temp ≈125 GiB；BFC 之后 HBM 上限 268 GiB）。本次扫了 7 个其他 policy（`save_out_proj`、`save_qkv_proj`、`save_dot_except_mlp`、`save_dot_except_mlpwi`、`minimal_with_context`、`minimal`），都叠加在两个 topology 的赢家 XLA 配方之上。

| 路径 | Topology | remat_policy | TGS | step | Δ vs `full` | Total mem（编译期） | Loss14（Δ） | 状态 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| sgd-v3 | PP=8（overlap2+async） | **`full`** ⭐ | **999.8** | 28.68 s | 0 | 253 GB | 9.994（=） | ✓ | 赢家 |
| sgd-v3 | PP=8 | `save_out_proj`（14678） | 973.6 | 29.46 s | -2.62 % | 估 210-260 GB | 9.993（≈=） | ✓ | 装得下但更慢 |
| sgd-v3 | PP=8 | `save_qkv_proj`（14677） | OOM | — | — | 513.6 GB | — | ✗ | 单次申请 438.7 GiB |
| sgd-v3 | PP=8 | `save_dot_except_mlp`（14679） | OOM | — | — | 535.2 GB | — | ✗ | 申请 460.4 GiB |
| sgd-v3 | PP=8 | `save_dot_except_mlpwi`（14680） | OOM | — | — | 2121.8 GB | — | ✗ | 申请 2.00 TiB（HBM 的 8.4×） |
| sgd-v3 | PP=8 | `minimal_with_context`（14681） | OOM | — | — | 3098.3 GB | — | ✗ | 申请 2.95 TiB（HBM 的 12×） |
| sgd-v3 | FSDP=8（ag=1G 生产） | **`full`** ⭐ | **1135.7** | 24.78 s | 0 | 236 GB | 9.994（=） | ✓ | 赢家 |
| sgd-v3 | FSDP=8 | `save_out_proj`（14700） | 1093.0 | 26.24 s | **-3.76 %** | 255 GB | **10.031（+0.04）** | ✓ | 装得下，更大的 slowdown，**loss 偏离** |
| sgd-v3 | FSDP=8 | `save_qkv_proj`（14689） | OOM | — | — | 469.1 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `save_dot_except_mlp`（14690） | OOM | — | — | 488.0 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `save_dot_except_mlpwi`（14691） | OOM | — | — | 1909.0 GB | — | ✗ | |
| sgd-v3 | FSDP=8 | `minimal_with_context`（14692） | OOM | — | — | 2779.8 GB | — | ✗ | |
| dense-cf1.25 | PP=8（overlap2） | **`full`** ⭐ | **1224.1** | 23.42 s | 0 | ~200 GB | 9.998（=） | ✓ | 赢家 |
| dense-cf1.25 | PP=8 | `save_out_proj`（14686） | 1194.4 | 24.04 s | -2.42 % | 210.2 GB | 9.997（≈=） | ✓ | 装得下但更慢 |
| dense-cf1.25 | PP=8 | `save_qkv_proj`（14687） | OOM | — | — | 410.7 GB | — | ✗ | 申请 335.8 GiB |
| dense-cf1.25 | PP=8 | `save_dot_except_mlp`（14682） | OOM | — | — | 432.3 GB | — | ✗ | 申请 357.5 GiB |
| dense-cf1.25 | PP=8 | `save_dot_except_mlpwi`（14683） | OOM | — | — | 688.3 GB | — | ✗ | 申请 613.5 GiB |
| dense-cf1.25 | PP=8 | `minimal_with_context`（14684） | OOM | — | — | 922.8 GB | — | ✗ | 申请 848.0 GiB |
| dense-cf1.25 | PP=8 | `minimal`（14685） | OOM | — | — | 862.4 GB | — | ✗ | 申请 787.5 GiB |
| dense-cf1.25 | FSDP=8（ag=1G 生产） | **`full`** ⭐ | **1207.9** | — | 0 | ≈200 GB | 9.998（=） | ✓ | 赢家 |
| dense-cf1.25 | FSDP=8 | `save_out_proj`（14699） | 1049.8 | 27.32 s | **-13.09 %** | 190 GB | **10.032（+0.034）** | ✓ | 装得下，**大 slowdown**，loss 偏离 |
| dense-cf1.25 | FSDP=8 | `save_qkv_proj`（14694） | OOM | — | — | 368.4 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `save_dot_except_mlp`（14695） | OOM | — | — | 387.3 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `save_dot_except_mlpwi`（14696） | OOM | — | — | 608-688 GB | — | ✗ | |
| dense-cf1.25 | FSDP=8 | `minimal_with_context`（14697） | OOM | — | — | 800.9 GB | — | ✗ | |

**三个观察：**

1. **激活张量增长速度远超历史"余量"提示。** `dense-cf1.25` 用 `full` 时 temp 余量 ~70 GiB（200 vs 268 GiB HBM 上限），但切到 `save_dot_except_mlp`（4 张量保存的中等 policy）时 total 翻倍到 432 GB —— 多出来的 230 GB 远超那 70 GiB 余量。`pipeline_module/while/body/closed_call/.../scan(layers.func_to_vmap)` 这一 lowering 似乎为 **每个 microbatch × 每个 stage × 每个 V-chunk** 都保留了一份保存的激活，乘数变成 8×7 = 56 而不是预期的 per-stage V chunks 数 7。对 PP=8 + DeepSeek-V3 的 `bf16[1,7,4096,7168]` per-layer 激活（392 MiB），乘以 56 而非 7 正好解释了观察到的 8× 内存爆涨。FSDP=8 下同样的爆涨通过 `scan_layers=True` 在 58 层 decoder 上的复制实现 —— 因子相当（58 vs 56），所以两个 topology 的 OOM 阈值几乎一致。

2. **唯一装得下的 policy（`save_out_proj`）到处都比 `full` 慢，FSDP=8 dense（-13.1 %）的退步是 PP=8 dense（-2.4 %）的 5×。** 可能机制：FSDP=8 生产配方把 all-gather + reduce-scatter 的分块调度精细地匹配到了 `full` 的重算 pattern（+11.6 / +20.5 % 的 ag=1 GiB 胜利是基于 `full` 测得的）。引入额外保存的激活（每层 1 个）改变了调度的 allocator 指纹，足以打破 ag=1 GiB 的 prefetch overlap。PP=8 没有这种精细的 overlap（overlap2 才是整个优化），所以 `save_out_proj` 退步只是 BFC 压力的成本。

3. **loss 偏离 ~0.03-0.04 的现象只在 FSDP=8 + `save_out_proj` 下出现**（PP=8 不偏离）。这意味着 FSDP=8 reduce-scatter 调度在多保存激活时重新排序了累加 —— 同样的前向但梯度求和顺序略有不同 → 梯度值略有不同 → 几步之后 loss 略有不同。在 PP=8 下，per-stage 局部 reduce-scatter 不会重排到能扰动 loss 的程度。

**建议**：在 pdbs=7 下两个 topology 和两个 MoE 分支都保持 `remat_policy: 'full'`。更轻的 remat 仅在更小的 pdbs（本次未测试）或对 framework 做改动以避免 per-iteration 激活拷贝时才有可能可行。

## 推荐部署方式

配方依赖于 路径 × topology × 分支，所以 per-model env 文件需要三个分支：

```bash
# configs/deepseek3-671b.env.sh —— 根据 topology 轴 + MoE 分支拆分
if [[ "${MAXTEXT_DCN_PP:-1}" -le 1 ]]; then
    # FSDP=8 路径：保留 +11.6 % / +20.5 % 的 all-gather combiner 胜利。
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_all_gather_combine_threshold_bytes=1073741824"
else
    # PP=8 路径：镜像默认 ag=8 GiB + overlap2（sgd-v3 + dense 通用 +3 至 +5 %）。
    # async_priority 是 BRANCH-DEPENDENT —— sgd-v3 上 +1.5 %，dense 上 -2.7 %。
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    if [[ "${MAXTEXT_USE_DEEPEP_DISPATCH:-false}" == "true" || "${MAXTEXT_SPARSE_MATMUL:-false}" == "true" ]]; then
        # sgd-v3（DeepEP / sparse_matmul）才加 —— async_priority 解决 MoE skew 落后者等待。
        XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_highest_priority_async_stream=true"
    fi
    # dense-cf1.25（`sparse_matmul=False`）：不加 async_priority —— 均匀 per-stage 计算让
    # async-stream 优先级提升变得反作用（损失了 scheduler 的灵活性）。
fi
export XLA_FLAGS
```

**注意**：`submit.sh` 当前不会把 `MAXTEXT_DCN_PP` / `MAXTEXT_USE_DEEPEP_DISPATCH` / `MAXTEXT_SPARSE_MATMUL` 作为普通环境变量传透（它们是经 `--` 透传的 MaxText 配置 key）。Guard 变量名需要与 `submit.sh` / `_train.sh` 协调 —— 例如 `_env_PP_TOPOLOGY=pp` 可以是用户侧提示，由 `_job.sbatch` 或 wrapper 脚本在检测到透传参数中有 `dcn_pipeline_parallelism=8` 时自动设置。在那个机制落地之前，保守的做法是让 `configs/deepseek3-671b.env.sh` 保持 FSDP-only（即当前状态），并在跑 PP=8 时加 CLI 覆盖：

```bash
RAY=1 ./submit.sh deepseek3-671b:pp8-prod ... -- \
    per_device_batch_size=7 sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true \
    dcn_pipeline_parallelism=8 dcn_fsdp_parallelism=1 \
    _env_TUNE_PROFILE=pp8-d_overlap2_async   # for sgd-v3
    # OR _env_TUNE_PROFILE=pp8-dense_d_overlap2   # for dense-cf1.25 (overlap2 alone is enough)
```

这要求把 `pp8-*` 的 `TUNE_PROFILE` 块永久保留在 `train_env.sh` 里（目前是这个状态）。

## 推荐

1. **生产 sgd-v3（DeepEP dropless）：使用 FSDP=8 pdbs=7 + `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`。** 这个配方落在 1135.7 TGS（比 clean baseline 高 +11.6 %、比 4 月 14 日历史基线 1097 高 +3.5 %）。在可编辑范围内（Primus-Turbo + MaxText `moe.py`），PP=8 在 dropless 路径上无法匹敌；要弥合差距需要 framework-level 改动（重写 `pipeline.py` 改用显式 non-blocking `psend/precv`，或者把 `nn.vmap`-of-`shard_map` 组合替换为自定义 pipeline schedule）。
2. **生产 `dense-cf1.25`（dense_matmul + capacity-factor dropping）：使用 PP=8 + 仅 `--xla_gpu_experimental_parallel_collective_overlap_limit=2`，在镜像默认 XLA 之上。** 这个配方落在 1224.1 TGS 平均（n=2），比 FSDP=8 dense 生产配方（1207.9）高 +1.34 %。这条路径下 **不要继承** FSDP 调优过的 `ag=1 GiB` flag —— 它要 -6.6 %。在 dense 上加 `async_priority` 损害 -2.7 %，所以更简单的 1-flag 配方对 dense 才是正确的。
3. **PP=8 sgd-v3（仅当 FSDP-feasibility 强制 PP 时）：使用 `overlap_limit=2 + --xla_gpu_enable_highest_priority_async_stream=true`，在镜像默认之上。** 这个栈在生产状态基线之上达到 +4.66 %。4-flag 栈只多 +0.1 %，所以更简单的 2-flag 配方更佳。剩下的约 12 % 与 FSDP=8 sgd-v3 的差距是结构性的（DeepEP per-microbatch 开销 + scan-carry 串行化 + pipeline bubble），在这一栈层面不能通过 XLA / NCCL 旋钮解决。
4. **把 `configs/deepseek3-671b.env.sh`（FSDP 调优过的 `ag=1 GiB` flag）改为以 `dcn_pipeline_parallelism <= 1` 为条件。** 这个 flag 在 FSDP=8 上是 +11.6 % / +20.5 %，但 **在 PP=8 上是 -1.0 % / -6.6 %**（符号反转）—— 让它在所有路径上生效会悄悄把 PP=8 dense 生产降速 6.6 %。详见上面的部署模块。
5. **在 pdbs=7 下两个 topology 和两个 MoE 分支都保持 `remat_policy: 'full'`。** `nn.scan`-PP 与 `scan_layers=True`-FSDP 下激活内存的放大因子约 56-58×，所以 OOM 阈值在两个 topology 几乎一致。唯一不 OOM 的备选（`save_out_proj`）到处都退步 2.4-13.1 %，并且在 FSDP=8 下还会额外让 loss 偏离。
6. **PP=8 上不要用 `pp_p2p`、`overlap_limit≥4`、`nccl_chan8` 或任何 `pipelined_*=true`。** 全是负面或 OOM。FSDP=8 上这些 flag 的排序 **不会** 转移到 PP=8 —— 符号反转很常见（overlap2：FSDP -4.14 % → PP +3.38 %；nccl_chan8：FSDP +4.26 % → PP -1.74 %；pp_p2p：FSDP no-op → PP -5.31 %；NCCL_PROTO=Simple：FSDP -1.34 % → PP +3.34 %）。
7. **保留上游 `combine_hang` 修复。** 在长期训练中它对正确性是必需的；≤0.5 % 的稳态成本可以忽略。
8. **`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep-v3` 现在是 `container_env.sh` 默认值** —— sgd 配置不再需要显式 env-var prefix。v1/v2 基线仍需显式覆盖（`MAXTEXT_PATCH_BRANCH=yihuang/moe-turbo-gmm-and-deepep[-v2]`）。镜像携带的是 C++/runtime 修复；MaxText Python 集成是正交的，住在 patch branch 里（详见 `moe-pdbs-sweep-prompt.md` 中完整的 ORTHOGONAL 规则）。
9. **`sparse-gmm-fixed`（sparse_matmul + `ragged_dot`）在本机器栈这个规模下是错的路径。** 在所有可行 pdbs ≥ 7 都 OOM，因为 `ragged_all_to_all` 物化了一个完整的 `num_ranks × tokens × hidden` 接收缓冲区（在 pdbs=7 时比 DeepEP 的 per-rank `num_worst_tokens × hidden` 缓冲区多 44 GiB）。改用 `sgd-v3`（DeepEP），或者退一步用 `dense-cf1.25`（dense_matmul + capacity-factor dropping，如果不需要 dropless 路由）。

## 一句话总结

1. **最佳 `sgd-v3` 配方（任何 topology 选择下）：FSDP=8、pdbs=7、单一 XLA flag override `--xla_gpu_all_gather_combine_threshold_bytes=1073741824`。** 落在 **1135.7 TGS** vs 1017.7 baseline（+11.6 %），并且比 4 月 14 日历史基线（1097 TGS）高 +3.5 %。PP=8 sgd-v3 即使用最佳 PP 调优 XLA 栈仍比 FSDP=8 低约 12 %（结构性）。
2. **最佳 `dense-cf1.25` 配方：PP=8、pdbs=7、在镜像默认 XLA 之上仅 `--xla_gpu_experimental_parallel_collective_overlap_limit=2`。** 落在 **1224.1 TGS 平均（n=2），比 FSDP=8 dense 生产配方高 +1.34 %**（1207.9）。在 dense 上加 `async_priority` 损害（-2.7 %）；dense 想要更简单的配方。FSDP=8 dense 的 +20.5 % 来自 `ag=1 GiB`，但同一个 flag 在 *PP* 路径上要 -6.6 %。
3. **配方依赖于 路径 × topology × 分支。** FSDP=8 与 PP=8 之间的符号反转是系统性的（overlap2：FSDP -4.14 % → PP +3.38 %；ag=1 GiB：FSDP +11.6 % → PP -1 至 -6.6 %；NCCL_PROTO=Simple：FSDP -1.34 % → PP +3.34 %）。原因是 fabric 利用率不同（FSDP 的 ICI all-gather 把 fabric 吃满拒绝并发；PP=8 的 ICI ag 小 + DCN c-p 是低利用率点对点，所以 2-stream 并发能帮上）。从一个 regime 调出来的结果 **不会** 转移到另一个 —— 而 `configs/deepseek3-671b.env.sh` 在被无条件应用之前必须按 topology 守护。
4. **`overlap2` 是最优** —— PP=8 下 parallel-collective-overlap-limit 曲线上的甜点（overlap=1：0；overlap=2：+3.38 %；overlap=4：-3.53 %；overlap=8：+2.76 %）。2-stream 甜点匹配 PP=8 schedule 中独立 fabric 的数量（1 ICI + 1 DCN-p2p = 2）。
5. **`nn.scan`-PP carry-dep 是 `is_pipelined=false` 的结构性瓶颈** —— 在 14563 的 HLO（镜像默认，FSDP=8 sweep 中的 `xla_dump`）和 14662 的 HLO（PP=8 sweep 中用 `_env_ENABLE_XLA_DUMP=1` + 赢家配方 overlap2 采集）都得到证实。已测试的 XLA flag 都不会改变 carry-dep，所以跨迭代调度仍然关闭；PP 的胜利来自 *迭代内* 的并发（overlap2）。
6. **PP=8 c-p 算子全是 392 MiB（`bf16[1,7,4096,7168]`）** —— `cp_decomposer_threshold` 取 512 MiB 至 8 GiB 之间是 no-op（不分解），而 `=256 MiB` 会分解但对 PP=8 没有提速（LHS scheduler 还是不能跨 `nn.scan` carry-dep 流水）。c-p 大小不是瓶颈。
7. **FSDP=8 上 dense vs sgd-v3 的 +20.5 % vs +11.6 % 差距证明 dropless MoE 的 comm-overlap 天花板被路由 skew 限制。** `dense-cf1.25` 有均匀的 per-layer 计算（capacity-factor dropping） → 没有 skew 放大 → 重叠窗口更大 → 打破 all-gather fusion 屏障带来的加速更大。同样的路由 skew 惩罚也是 PP=8 sgd-v3 与 FSDP=8 sgd-v3 锁定 -12 % 差距的原因。
8. **`sgf`（sparse-gmm-fixed）在本机器栈规模下不可用。** 在所有可行 pdbs ≥ 7 都 OOM，原因是 `ragged_all_to_all` 的 `num_ranks × tokens × hidden` 接收缓冲区。改用 `sgd-v3`（DeepEP）。
9. **没有任何 NCCL flag、memory-fraction 调整、async-stream priority、或 while-loop double-buffering 在 FSDP=8 上的 noise 之上有效**，一旦 all-gather combine threshold 被降下来。+11.6 %（sgd-v3）/ +20.5 %（dense-cf1.25）就是 FSDP=8 在可编辑范围内的实际天花板。
10. **`pipelined_*=true` 系列 flag 在 PP=8 上也 OOM**（`pp_all_reduce` 217 GiB temp、`pp_all_gather` 302 GiB temp），与 FSDP=8 的 OOM 行为一致。两个 topology 的 per-stage HBM 余量都不足以吸收这些 collective 的 prefetch buffer。
11. **`combine_hang` 修复在稳态下性能中性。** Loss 与修复前的镜像逐位相同；TGS 差距 ≤ 0.5 %（在 step-to-step jitter 范围内）。纯正确性收益。
12. **`remat_policy: 'full'` 在 pdbs=7 下对两个 topology 和两个 MoE 分支都是最优**（共扫了 13 个其他 policy，jobs 14677-14700）。激活内存放大因子在 `nn.scan`-PP 与 `scan_layers=True`-FSDP 下都是约 56-58×，OOM 阈值几乎一致。唯一不 OOM 的备选（`save_out_proj`）装得下但到处都退步 —— FSDP=8 上还有 0.03-0.04 的 loss 偏离。

## 附录：数据来源

Job ID 按时间顺序排列。Tag 约定 `<config>-<topology>`（例如 `sgd-pp8` = `sgd-v3` on PP=8）加可选后缀（`c` = clean / 无 profiler，`+ag1G` = 带 FSDP 调优的 all-gather 阈值）。

| Job ID | Tag | Path | 镜像 / 配方 | Profiler | HLO dump | 状态 |
|---|---|---|---|---|---|---|
| 13711（历史） | sgd-fs8 pdbs=7 | sgd-v3 | 原 `deepep-gmm-maxtext-v26.2.tar`（4 月 14 日） | off | off | TGS=1097 baseline |
| 14539 | sgd-pp8 pdbs=8（OLD image） | sgd-v3 | OLD（无 `combine_hang`） | xplane | off | step 14 ✓ —— 用于 `combine_hang` Δ 检查 |
| 14550 | sgd-pp8 pdbs=8 | sgd-v3 | NEW（`combine_hang`） | xplane | yes | step 14 ✓ exit=1 cleanup |
| 14551 / 14564 | sgf-pp8 pdbs=8 / pdbs=7 | sgf | NEW | xplane / off | yes / off | OOM 217-222 GiB temp |
| 14552 | dense-cf1.25-pp8 pdbs=8 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14553 | sgd-fs8 pdbs=7 | sgd-v3 | NEW | xplane | yes | step 14 ✓ |
| 14554 | sgf-fs8 pdbs=7 | sgf | NEW | off | off | OOM 217 GiB temp |
| 14555 | dense-cf1.25-fs8 pdbs=7 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14563 | sgd-pp8 pdbs=7 | sgd-v3 | NEW | xplane | yes | step 14 ✓ exit=1 cleanup |
| 14565 | dense-cf1.25-pp8 pdbs=7 | dense-cf1.25 | NEW | xplane | yes | step 14 ✓ |
| 14570 | sgd-pp8c pdbs=7 | sgd-v3 | NEW（clean） | off | off | step 14 ✓ exit=1 cleanup |
| 14571 | dense-cf1.25-pp8c pdbs=7 | dense-cf1.25 | NEW（clean） | off | off | clean PP=8 dropping baseline |
| 14572 | sgd-fs8c pdbs=7 | sgd-v3 | NEW（clean） | off | off | clean FSDP=8 sgd-v3 baseline |
| 14573 | dense-cf1.25-fs8c pdbs=7 | dense-cf1.25 | NEW（clean） | off | off | clean FSDP=8 dropping baseline |
| 14579-14626 | fs8 XLA-tuning sweep | sgd-v3 | NEW + 各种 TUNE_PROFILE | off | off | 28 个 profile，仅 sgd-v3 |
| 14602 | FSDP=8 sgd-v3 + ag=1 GiB 调优 | sgd-v3 | `ag=1 GiB` | off | off | step 14 ✓ —— TGS=1135.7（FSDP=8 sgd-v3 生产） |
| 14629 | FSDP=8 dense-cf1.25 + ag=1 GiB 调优 | dense-cf1.25 | `ag=1 GiB` | off | off | step 14 ✓ —— TGS=1207.9（FSDP=8 dense 生产） |
| 14638 | sgd ag=1G 基线 | sgd-v3 PP=8 | `pp8-baseline-ag1G` | off | off | step 14 ✓ —— PP-sweep 主基线 |
| 14639 | sgd ag=8G 默认 | sgd-v3 PP=8 | `pp8-restore_ag_default` | off | off | step 14 ✓ |
| 14640 | dense ag=1G 基线 | dense-cf1.25 PP=8 | `pp8-baseline-ag1G` | off | off | step 14 ✓ |
| 14641 | dense ag=8G 默认 | dense-cf1.25 PP=8 | `pp8-restore_ag_default` | off | off | step 14 ✓ |
| 14642 | （Wave 1.5 第 1 次尝试） | sgd-v3 PP=8 | `pp8-evidence` | xplane | yes | 在 compile 阶段被 slurm 取消（2:20）—— 不可恢复 |
| 14643 | pp_p2p | sgd-v3 PP=8 | `pp8-pp_p2p` | off | off | step 14 ✓ |
| 14644 | cp_decomp_1G | sgd-v3 PP=8 | `pp8-cp_decomp_1G` | off | off | step 14 ✓ |
| 14645 | （async_priority 第 1 次尝试） | sgd-v3 PP=8 | `pp8-async_priority` | off | off | 15 min 卡死 RCCL flake，取消+重试 |
| 14646 | double_buffer | sgd-v3 PP=8 | `pp8-double_buffer` | off | off | step 14 ✓；loss 偏离 0.02 |
| 14647 | async_priority 重试 | sgd-v3 PP=8 | `pp8-async_priority` | off | off | step 14 ✓ |
| 14648 | cp_decomp_256M | sgd-v3 PP=8 | `pp8-cp_decomp_256M` | off | off | step 14 ✓ |
| 14649 | overlap2（第 1 次测量） | sgd-v3 PP=8 | `pp8-overlap2` | off | off | step 14 ✓ —— 第一次 +4.14 % 信号 |
| 14650 | pp_all_reduce | sgd-v3 PP=8 | `pp8-pp_all_reduce` | off | off | OOM 217 GiB temp |
| 14651 | pp_all_gather | sgd-v3 PP=8 | `pp8-pp_all_gather` | off | off | OOM 302 GiB temp |
| 14652 | nccl_chan8（ag=1G） | sgd-v3 PP=8 | `pp8-nccl_chan8` | off | off | step 14 ✓ |
| 14653 | nccl_chan8（ag=8G） | sgd-v3 PP=8 | `pp8-d_chan8` | off | off | step 14 ✓ |
| 14654 | （mem95 第 1 次尝试） | sgd-v3 PP=8 | `pp8-mem95` | off | off | RCCL flake，取消+重试 |
| 14655 | （proto_simple 第 1 次尝试） | sgd-v3 PP=8 | `pp8-nccl_proto_simple` | off | off | RCCL flake,取消+重试 |
| 14656 | （cp1G_async 栈第 1 次尝试） | sgd-v3 PP=8 | `pp8-cp1G_async` | off | off | RCCL flake，取消+重试 |
| 14657 | d_cp1G_async（ag=8G） | sgd-v3 PP=8 | `pp8-d_cp1G_async` | off | off | step 14 ✓ |
| 14658 | overlap2（第 2 次测量） | sgd-v3 PP=8 | `pp8-overlap2` | off | off | step 14 ✓ —— TGS=978.6 |
| 14659 | overlap4 | sgd-v3 PP=8 | `pp8-overlap4` | off | off | step 14 ✓ |
| 14660 | overlap8 | sgd-v3 PP=8 | `pp8-overlap8` | off | off | step 14 ✓ |
| 14661 | d_overlap2（overlap2 + ag=8G） | sgd-v3 PP=8 | `pp8-d_overlap2` | off | off | step 14 ✓ |
| 14662 | overlap2（第 3 次测量）+ profile + HLO | sgd-v3 PP=8 | `pp8-overlap2` | xplane | yes | step 14 ✓ —— TGS=989.2；HLO/xplane 在 outputs/14662-*/ |
| 14663 | mem95 重试 | sgd-v3 PP=8 | `pp8-mem95` | off | off | step 14 ✓ |
| 14664 | proto_simple 重试 | sgd-v3 PP=8 | `pp8-nccl_proto_simple` | off | off | step 14 ✓ |
| 14665 | cp1G_async 重试 | sgd-v3 PP=8 | `pp8-cp1G_async` | off | off | step 14 ✓ |
| 14666 | BS1 cp+as+ov2（ag=8G） | sgd-v3 PP=8 | `pp8-d_cp1G_async_ov2` | off | off | step 14 ✓ |
| 14667 | BS2 cp+as+ov2（ag=1G） | sgd-v3 PP=8 | `pp8-cp1G_async_ov2` | off | off | step 14 ✓ |
| 14668 | BS3 ov2+async（ag=8G） | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ —— 最佳 2-flag |
| 14669 | d_full_stack（cp+as+ov2+proto，ag=8G） | sgd-v3 PP=8 | `pp8-d_full_stack` | off | off | step 14 ✓ —— sgd-v3 排行第一 |
| 14670 | d_overlap2_proto（ag=8G） | sgd-v3 PP=8 | `pp8-d_overlap2_proto` | off | off | step 14 ✓ |
| 14671 | dense full_stack（ag=8G） | dense-cf1.25 PP=8 | `pp8-dense_d_full_stack` | off | off | step 14 ✓ |
| 14672 | dense overlap2（ag=8G） | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ —— **dense-cf1.25 排行第一** |
| 14673 | dense overlap2 重测（n=2 验证） | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ —— TGS=1233.0 |
| 14674 | sgd BS3 重测（overlap2+async ag=8G） | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ —— TGS=999.4 |
| 14675 | sgd d_overlap2 重测（仅 overlap2，ag=8G） | sgd-v3 PP=8 | `pp8-d_overlap2` | off | off | step 14 ✓ —— TGS=998.0 |
| 14676 | dense overlap2+async（通用配方测试） | dense-cf1.25 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ —— TGS=1190.4（async 损害 dense） |
| 14677 | sgd save_qkv_proj remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 513.6 GB total |
| 14678 | sgd save_out_proj remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | step 14 ✓ —— TGS=973.6 |
| 14679 | sgd save_dot_except_mlp remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 535.2 GB total |
| 14680 | sgd save_dot_except_mlpwi remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 2121.8 GB total |
| 14681 | sgd minimal_with_context remat | sgd-v3 PP=8 | `pp8-d_overlap2_async` | off | off | OOM 3098.3 GB total |
| 14682 | dense save_dot_except_mlp remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 432.3 GB total |
| 14683 | dense save_dot_except_mlpwi remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 688.3 GB total |
| 14684 | dense minimal_with_context remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 922.8 GB total |
| 14685 | dense minimal remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 862.4 GB total |
| 14686 | dense save_out_proj remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | step 14 ✓ —— TGS=1194.4 |
| 14687 | dense save_qkv_proj remat | dense-cf1.25 PP=8 | `pp8-dense_d_overlap2` | off | off | OOM 410.7 GB total |
| 14688 | FSDP=8 sgd save_out_proj 第 1 次尝试 | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | RCCL flake → 14698 |
| 14689 | FSDP=8 sgd save_qkv_proj | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 469.1 GB total |
| 14690 | FSDP=8 sgd save_dot_except_mlp | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 488.0 GB total |
| 14691 | FSDP=8 sgd save_dot_except_mlpwi | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 1909.0 GB total |
| 14692 | FSDP=8 sgd minimal_with_context | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 2779.8 GB total |
| 14693 | FSDP=8 dense save_out_proj 第 1 次尝试 | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | RCCL flake → 14699 |
| 14694 | FSDP=8 dense save_qkv_proj | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 368.4 GB total |
| 14695 | FSDP=8 dense save_dot_except_mlp | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 387.3 GB total |
| 14696 | FSDP=8 dense save_dot_except_mlpwi | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 608-688 GB total |
| 14697 | FSDP=8 dense minimal_with_context | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | OOM 800.9 GB total |
| 14698 | FSDP=8 sgd save_out_proj 第 2 次尝试 | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | failed（node8 中途 cuInit error 303）→ 14700 |
| 14699 | FSDP=8 dense save_out_proj | dense-cf1.25 FSDP=8 | （仅 env 文件 ag=1G） | off | off | step 14 ✓ —— TGS=1049.8（vs full -13.09 %） |
| 14700 | FSDP=8 sgd save_out_proj 第 3 次尝试 ✓ | sgd-v3 FSDP=8 | （仅 env 文件 ag=1G） | off | off | step 14 ✓ —— TGS=1093.0（vs full -3.76 %） |
