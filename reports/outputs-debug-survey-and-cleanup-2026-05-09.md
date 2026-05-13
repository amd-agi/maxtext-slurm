# Outputs Debug Survey & Storage Cleanup

Date: 2026-05-09

## 目的

复盘 `maxtext-slurm/outputs/` 目录里 **22250 → 23318 全部 60+ job** 的 debug 链条，并对 ~2 TB 的存储做一次按"重要程度 + 体积"的清理。

本报告与 `internode-deepep-debug-progress-2026-05-09.md` 互补：
- 后者聚焦最终结论 + 下一步 instrumentation 计划
- 本文档聚焦**整个时间线**与**每个 job 的角色定位**，便于后续回查"某个怀疑点之前是哪个 job 排除的"。

---

## 一、Debug 时间线（按阶段）

### 阶段 1 — 集群与镜像基础设施（4/30 上午）

目标：让 MaxText 的 Llama2-70B 在 2 节点上能跑起来。

| Job ID | 关键配置 | 结论 |
|---|---|---|
| 22250 / 22251 | 拉 `rocm/jax-training:maxtext-v26.2` | **Docker pull 失败** + `User missing groups: render, video` |
| 22253 | 默认 `per_device_batch_size=8, max_target_length=4096` | **HBM OOM**：`RESOURCE_EXHAUSTED ... 118.37GiB` |
| 22256 | 缩到 bs=1, seq=1024 | 编译期 SIGTERM (143) |

### 阶段 2 — RCCL/MSCCL channel 兼容（4/30 中午）

| Job ID | 关键 env | 现象 |
|---|---|---|
| 22257 | `NCCL_DEBUG=INFO`, `SUBSYS=INIT,NET,ENV` | verbose 观察，被 SIGTERM |
| 22260 | `NCCL_DMABUF_ENABLE=0` | **定位到** `'MSCCL: number of channels available (28) less than required (32)'` |
| 22262 | `RCCL_MSCCLPP_ENABLE=0` + `NCCL_ALGO=Ring,Tree` | 同样 28 vs 32 ❌ —— 仅关 MSCCLPP 不够 |
| 22263 | + `NCCL_MIN_NCHANNELS=32` | barrier 前卡住被 SIGTERM |
| **22264** | `RCCL_MSCCL_ENABLE=0` + `RCCL_MSCCLPP_ENABLE=0` | ✅ **第一次跑通 step 0** |
| 22268 | 同 22264 + 小 batch | ✅ 稳定 |
| 22972 | 同名 defaultfix 但回到默认大 batch | ❌ 重复 22253 OOM —— 修复必须配合小 batch |

> 阶段 2 结论：**关掉 MSCCL/MSCCLPP** 是稳定的 workaround。

### 阶段 3 — DeepSeek3 671B Proxy 2 节点 baseline（5/6 上午）

| Job ID | 配置 | 结论 |
|---|---|---|
| 22974 | proxy + `ONE_GPU_PER_PROCESS=true` | 启动期取消 (143) |
| **22975** | 同上重试 | ✅ **OGPP baseline 验证通过**，多 step |
| 22976 | + `internode-smoke` + `PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process` | ❌ Pydantic：`use_deepep_dispatch requires primus_turbo + DeepEP JAX bindings` |
| 22980 | 装上 primus_turbo | ❌ HIP OOM (178 GiB) + JAX coordinator fatal |
| 22982 | 换节点 (`idep2n-clean`) | ❌ **第一次出现 TE fused-attention 报错**：`fused_attn_rocm/utils_hip.cpp:252 ... PopulateRngStateAsync: CUDA Error: invalid configuration argument` |

### 阶段 4 — 拆解 DeepEP / TE / 注意力 / EP 拓扑（5/6 下午）

| Job ID | 改动 | 结论 |
|---|---|---|
| 23004 | `use_deepep_dispatch=False` | ❌ `Memory access fault by GPU node-6 ... address (nil)` |
| 23006 | + `attention=dot_product` | ❌ 同 nil GPU fault —— 不是 attention 单独的问题 |
| **23008** | + `dcn_expert_parallelism=1, dcn_fsdp_parallelism=2` | ✅ **跑通 0–2 步** —— 强烈提示**跨节点 EP 是触发点** |
| 23011 | `ONE_GPU_PER_PROCESS=false` | barrier 后挂死 SIGTERM |
| 23015 | `use_turbo_grouped_gemm=False` | ❌ XLA `Check failed: ... RaggedDot is only supported with Shardy.` |
| 23021 | + `shardy=True` | ❌ JAX OOM 382 GiB |
| 23024 | tgmm + shardy | 启动期 SIGTERM |
| 23025 | `sparse_matmul=False`（dense MoE） | ✅ 通过 —— 绕开 RaggedDot/sparse 的对照组 |
| 23027 | sparse + turbo GMM (bs=2) | ❌ OOM 129 GiB |

> 阶段 4 结论：**问题集中在 "internode + DeepEP + 跨节点 EP" 组合**；EP 限定在节点内 (23008) 或换 dense matmul (23025) 都能跑。

### 阶段 5 — 4 节点 "full" + Transformer Engine + DeepEP（5/7）

| Job ID | 说明 | 结论 |
|---|---|---|
| 23097 | 4n proxy + fsdp2 + ep2 + tgmm bs=2 | 容器 setns 失败 |
| 23098 / 23099 | proxy + fsdp2/ep2/tgmm bs=2 / bs=1 | ✅ proxy + 4n 是绿的 |
| 23116 | 切到 `4n-full` (TE + DeepEP) | ❌ JAX coord：`UNAVAILABLE: tasks are unhealthy` |
| 23118 | 重试 | Pydantic 重复 22976 |
| 23122 | `4n-full` + DeepEP + TE | ❌ 重现 22982 的 `PopulateRngStateAsync` + nil GPU fault |
| 23123 / 23124 | + `attention=dot_product` | ❌ 仍然 coord 死亡 + 81 GB coredump（**23124 已 gdb，bt 在 `23124-debug-bt/bt.txt`**） |
| **23125** | **诊断**：`use_deepep_dispatch=False` | ✅ **通过** —— 锁定问题在 DeepEP dispatch 路径 |
| 23137 | + `HIP_LAUNCH_BLOCKING=1, PYTHONFAULTHANDLER=1, NCCL_DEBUG=INFO` | SIGTERM，未额外定位 |

### 阶段 6 — ROCSHMEM 调参/Ablation（5/7 下午）

| Job ID | 关键 env | 结论 |
|---|---|---|
| 23143 / 23150 / 23153 / 23162 | `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`, `BOOTSTRAP_SOCKET_IFNAME=eth0`, `DEBUG_LEVEL=INFO` | `Shutdown barrier failed`、`9/32 reached barrier`、preflight FATAL |
| 23157 / 23163 | "abl"：去掉 `SOCKET_IFNAME=eth0` | ❌ 仍然 hip(9) + nil fault —— 网卡指定与否不影响 |
| 23166 | `HIP_LAUNCH_BLOCKING=1 + PYTHONFAULTHANDLER` | XLA 编译完后 SIGTERM |
| 23169 | `per_device_batch_size=1` | ❌ task 24 unhealthy —— 缩 batch 救不了 |

### 阶段 7 — 系统性 bA…bK Ablation（5/8 凌晨）

统一在 4n-full + dot attention + ROCSHMEM_BOOTSTRAP_TIMEOUT=60 + bs=1，每个 tag 只换一个变量：

| Tag | Job ID | 改动的那一个变量 | 结论 |
|---|---|---|---|
| nogmm | 23260 | `use_turbo_grouped_gemm=false` | SIGTERM |
| nogmm+shardy | 23262 | + `shardy=true` | OOM 682 GiB |
| ns | 23263 | 同 23262 | Docker pull 超时 (901s) |
| **bA** | 23266 | `HIP_PRINT_KERNEL_LAUNCH=1` | hip(9) + nil fault |
| **bB** | 23267 | 退到 2 节点 + HIP print | distributed unhealthy |
| **bC** | 23268 | HIP print + nogmm | SIGABRT + 大 coredump |
| **bD** | 23270 | `AMD_LOG_LEVEL=3, AMD_LOG_MASK=0x82` | SIGTERM |
| **bE** | 23275 | 仅 `shardy=true` | hip(9) + nil fault |
| **bF** | 23276 | 2 节点 + shardy | nil fault |
| **bG** | 23281 | 同 bD（AMD log 探针） | 1.7 GB log 全是 rocBLAS 噪声，无信号 |
| **bH** | 23282 | `XLA_FLAGS=--xla_dump_to=...` | env 引号被 launcher 吃掉，flag 没生效 |
| **bI** | 23283 (2n) / **23284** (4n) | `ROCSHMEM_HEAP_SIZE=4 GiB` | 都还是 hip(9) + nil fault —— **加大 ROCSHMEM 堆没用**。23284 已 gdb，bt 在 `23284-debug-bt/bt.txt` |
| **bJ** | 23285 | 2n + `HIP_LAUNCH_BLOCKING=1` | 取消 |

> 阶段 7 总体结论：所有 4 节点 / 跨节点变体 fingerprint 一致，需要换更小的复现路径。

### 阶段 8 — 最小复现 + K 系列定位（5/8 中午-下午）

走 `_env_REPRO_INTERNODE_DEEPEP=1` 的 `moe_dispatch + moe_combine` round-trip，绕开整个 MaxText 训练栈。

| Tag | Job ID | 目的 / 改动 | 结论 |
|---|---|---|---|
| bK | 23286 | 第一次启用 repro | ❌ `ModuleNotFoundError: No module named 'utils'`（import bug） |
| bK2 | 23287 | 修 import | ❌ `iter=0 CRASH inside moe_dispatch` —— **复现成功** |
| **K3-hash** | **23288** | 加 IPC handle SHA1 instrumentation | ❌ **核心定位**：`Primus-Turbo/csrc/jax/deep_ep/deep_ep.cpp:922 SyncFromIPCHandles Assertion failed: std::memcmp(ipc_handles_[i].reserved, handle_str.c_str(), HIP_IPC_HANDLE_SIZE) == 0`，gathered slot 是 `sha1(64 zero bytes)` |
| **K4-pretest** | **23290** | Raw JAX allgather pre-test | ❌ 各 rank 贡献 `(rank+1)*ones(64)`，gather 后 `[1,2,3,4,0,6,...]` —— **JAX `multihost_utils.process_allgather` 在掉一个 rank 的数据** |
| **K5-fixA** | **23302** | 用 JAX coordinator KV-store gather 替代 raw allgather | ✅ **bootstrap 在 16 ranks 全部成功**，`moe_dispatch` 全部 queued。fault 推到 GPU dispatch |
| **K6-fixA-noredo** | **23310** | bootstrap 仅一次 + dispatch 内放 `block_until_ready()` | ✅ 16/16 bootstrap OK；❌ 0/16 到达 `moe_dispatch GPU-DONE`，node-1 GPU 撞 (nil) |
| **K7-1node-sanity** | **23313** | **关键对照组**：单节点 8 rank | ✅ **PASS**：`internode=False, num_experts=256`，`warmup AFTER OK`，`moe_dispatch/combine GPU-DONE`，全 8 rank `PASS` |
| **K8-blocking-rsdebug** | **23314** | 16 rank + `HIP_LAUNCH_BLOCKING=1, ROCSHMEM_DEBUG=3, ROC_SHMEM_DEBUG=3` | walltime 撞墙，无新线索 |
| **K9-rsstate / K10-rsstate-v2** | **23315 / 23318** | ctypes 探 rocSHMEM state | ❌ 仍 nil fault；ctypes 解析不到 `rocshmem_my_pe` 符号 —— **报告明确"放弃外部 host-side 探针"** |

---

## 二、Cross-cutting 一句话总结

整个 debug 过程从最早的 RCCL/MSCCL channel mismatch (**22260-22264**)、HBM OOM (**22253/23021/23027**)、TE fused attention CUDA invalid configuration (**22982/23122**)、再到 4 节点 full 上 JAX coordinator dead-task 雪崩 (**23116/23124/23150/23169**)，一路收敛到**单一 root cause 候选**：

> **`Primus-Turbo/csrc/jax/deep_ep/deep_ep.cpp:917 / 922 SyncFromIPCHandles` 在 internode 16-rank 场景下，`hipIpcOpenMemHandle` 报 `HIP Error: invalid argument`，且 `ipc_handles_[i].reserved` 与各 rank gather 拿到的字节不一致；单节点 8-rank 同一段代码完全 PASS。**

最关键的几条证据（按重要性）：
- **23125** ✅ vs **23122/23124** ❌：关掉 `use_deepep_dispatch` 就好 → bug 在 DeepEP dispatch 路径
- **23008** ✅ vs **23004/23006** ❌：EP 限定在节点内就好 → **跨节点**触发
- **23288 (K3-hash)**：定位到 `deep_ep.cpp:917/922 SyncFromIPCHandles`
- **23290 (K4-pretest)**：定位到根因之一是 **JAX `process_allgather` 在掉数据**
- **23302 (K5-fixA)**：换 KV-store gather 后 **bootstrap 100% 通过**，剩余 fault 在 GPU dispatch kernel 内
- **23313 (K7-1node-sanity)** ✅：单节点完全 PASS → bug 仅在 internode RDMA / IPC handle 路径
- **23284 + `23284-debug-bt/bt.txt`**：SIGABRT 多线程 GDB 回溯
- **23283 (bI heap=4 GiB)**：排除"ROCSHMEM symmetric heap 不够"

**下一步**（详见 `internode-deepep-debug-progress-2026-05-09.md`）：
- 已经在 `runtime.cu` / `internode.cu` / `deep_ep.cpp` 加好 `[PT-RS]` / `PT_DEP_LOG` instrumentation
- ⚠️ 仍需把这些改动 hipify-sync 到 `runtime.hip` / `internode.hip`，否则 ROCm build 不会打印
- 重编后跑 2 节点 K-series repro，看 `rocshmem_ptr(p, peer)` 是否在某个 peer 返回 NULL

---

## 三、存储清理执行（2026-05-09）

清理前 `outputs/` 共占 **2.0 TB**，99% 是 Python coredump（每份 22-34 GB，多份/job）。

### 3.1 Coredump 统计

总计 16 个 job 产生 coredump，**仅 2 个被 gdb 分析过**：

| Job | core 数 | 单核大小 | 是否分析过 |
|---|---:|---:|---|
| 23015 | 14 | ~16 GB | ❌ |
| 23260 | 28 | ~22 GB | ❌ |
| 23268 | 28 | ~22 GB | ❌ |
| 23287 | 7 | ~14 GB | ❌ |
| 23288 | 7 | ~14 GB | ❌ |
| 23290 | 7 | ~14 GB | ❌ |
| 23124 | 3 | ~21 GB | ✅ → `23124-debug-bt/bt.txt` |
| 22980 | 2 | ~20 GB | ❌ |
| 23318 | 2 | ~16 GB | ❌ |
| 23284 | 1 | ~32 GB | ✅ → `23284-debug-bt/bt.txt` |
| 23267 | 1 | ~24 GB | ❌ |
| 23006 / 23004 / 22982 | 各 1 | ~23 GB | ❌ |
| 23315 | 1 | ~16 GB | ❌ |
| 23302 | 1 | ~16 GB | ❌ |

### 3.2 清理 Tier 1 — Coredump

**已分析过的 cores 已经把回溯抽到 `*-debug-bt/bt.txt`**；报告已明确"不再走 gdb 路线"，下一步是 inline instrumentation。

执行命令：

```bash
find /home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs \
    -mindepth 2 -maxdepth 2 -name 'core.*' -type f -delete
```

释放：**~1.9 TB**

### 3.3 清理 Tier 2 — 无信息量的旧 .log + 目录

按 survey 阶段分类，21 个 job 完全没有独立证据价值（被 SIGTERM、Docker 失败、Pydantic 守卫重复、Llama2 早期 infra 等），且**不在报告 cite 列表**也**不在 cores 列表**：

| 阶段 | Job IDs |
|---|---|
| Llama2 早期 infra | 22250, 22251, 22253, 22256, 22257 |
| RCCL/MSCCL（结论已写入报告） | 22260, 22262, 22263, 22264, 22268 |
| 重复 22253 的 OOM | 22972 |
| 22975 的取消前身 | 22974 |
| Pydantic 守卫重复 | 22976, 23118 |
| 启动期 SIGTERM 无数据 | 23011, 23024, 23097, 23123 |
| debug flag 但被 SIGTERM | 23137, 23166, 23270 |

执行命令：

```bash
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm/outputs
rm -rf {22250,22251,22253,22256,22257,22260,22262,22263,22264,22268,22972,22974,22976,23011,23024,23097,23118,23123,23137,23166,23270}-*
```

释放：约 100 MB（数量对清洁度有意义，但空间收益小）

### 3.4 保留下来的 job

清理后保留的 job 按"角色"归类，用于回查：

| 角色 | Job IDs |
|---|---|
| **报告 cite 的关键证据** | 23008, 23125, 23262, 23268, 23281, 23282, 23283, 23284, 23285, 23287, 23288, 23290, 23302, 23310, 23313, 23314, 23315, 23318 |
| **已分析的 GDB 回溯** | 23124-debug-bt, 23284-debug-bt |
| **绿色基线 / 对照组** | 22975 (proxy 2n baseline), 23098/23099 (4n proxy), 23025 (dense matmul) |
| **失败模式取证（仍保留 .log）** | 22980, 22982, 23004, 23006, 23015, 23021, 23027, 23116, 23122, 23143, 23150, 23153, 23157, 23162, 23163, 23169, 23260, 23263, 23266, 23267, 23275, 23276, 23286 |

---

## 四、下次再清理时的判断准则

1. **永远不删** `*-debug-bt/` 目录 —— gdb 抽过的 bt 是不可重现的（容器/磁盘清掉之后 cores 拿不回来）。
2. **永远不删**报告 cite 列表的 `.log` —— 文本证据是写入 markdown 的依据。
3. `core.*` 文件可以无脑删 —— 除非当下正打算 gdb 它。
4. 同一组 ablation 中**只跑了启动失败/SIGTERM/Docker 超时**的 job，全部可以连 `.log` + 目录删。
5. 重复发现的 OOM / Pydantic / preflight 失败也可删（一次发现就够）。
