# MI300 JAX Internode DeepEP 任务日志分类总结

日期：2026-05-13

目标：在 MI300 集群上跑通 JAX internode DeepEP。相关代码位置：

- `Primus-Turbo`：算子库 / DeepEP runtime
- `maxtext`：框架代码
- `maxtext-slurm`：Slurm launcher 与调试脚本
- `maxtext-slurm/outputs`：集群调试任务日志

分析范围：`maxtext-slurm/outputs/*.log` 中 71 个 Slurm 任务日志，以及现有调试报告中的取证结论。

## 总览

- 共分析任务日志：71 个
- 成功任务：9 个
- 明确失败任务：61 个
- 无最终摘要任务：1 个
- 当前 2N known-good：`25425` / `25426`

当前最重要结论：

1. 2N MaxText + JAX + Primus-Turbo internode DeepEP 已经在 `25425` 和 `25426` 连续跑通。
2. 当前已知好配置是：

   ```bash
   _env_ONE_GPU_PER_PROCESS=true
   _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60
   _env_ROCSHMEM_HEAP_SIZE=536870912
   _env_XLA_PYTHON_CLIENT_PREALLOCATE=false
   per_device_batch_size=1
   max_target_length=2048
   ```

3. `25423` 失败主因是内存：`per_device_batch_size=2`、`max_target_length=4096` 触发约 `110GiB` OOM。
4. `25434` 不是 rocSHMEM 网络失败，而是在 MaxText MoE shape 路径报：

   ```text
   TypeError: lt got incompatible shapes for broadcasting: (32768,), (2,).
   ```

5. 下一步最高信号动作是：用 `25425/25426` 同配置跑 4N，不要同时放大 batch/sequence 或恢复 XLA preallocation。

## 任务族分类

| 阶段 | 代表任务 | 调试目的 | 状态 | 获得的信息 | 对下一步的指导 |
|---|---|---|---|---|---|
| 1. 非 DeepEP / launcher 基线 | `22975`, `23008`, `23025`, `23098`, `23099`, `23125` | 确认 JAX 多机、MaxText 配置、`attention=dot_product`、TGMM / dense matmul、4N sharding 在关闭 DeepEP 时可跑通。 | 成功 | 问题不在基础 launcher、GSPMD/Shardy 主路径、`attention=dot_product` 或 turbo grouped GEMM 本身。 | 后续失败应优先归因到 `use_deepep_dispatch=True`、rocSHMEM、或内存形状，而不是反复怀疑基础训练路径。 |
| 2. 4N 全量 DeepEP 首轮 bring-up | `23116`, `23122`, `23124`, `23125` | 把 4N DeepSeek proxy 全路径拉起来，并逐个排除脏节点、错误镜像、TE fused attention。 | 失败但定位明确 | `23124` 的核心故障来自 rocSHMEM TCP bootstrap 默认 5s connect timeout；`23125` 关闭 DeepEP 成功，是强对照。 | 保留 preflight fail-fast、DeepEP 镜像默认值、`attention=dot_product`，并设置 `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`。 |
| 3. coredump / gdb 取证 | `23138`-`23140` 调试辅助，`23124` core | 用同镜像 roc-gdb 读取 root-owned core，确认 SIGABRT 来源。 | 成功取证 | backtrace 落在 `rocshmem::Socket::connect -> TcpBootstrap -> rocshmem_init_attr`，不是 GPU kernel SIGSEGV。 | 后续 core 继续用同镜像、`--privileged`、`rocgdb`、batch backtrace 的流程。 |
| 4. ablation 与误区排除 | `23260`, `23262`, `23268`, `23281`, `23283`, `23284`, `23285` | 验证 nogmm/shardy、heap size、HIP launch blocking、ROCm verbose logging 等假设。 | 大多失败 | `tgmm=false` + Shardy 路径 OOM 或不合法；4GiB rocSHMEM heap 不解决；HIP/ROCm verbose 日志低信号。 | 不要再把主要时间花在 nogmm/shardy 或 `HIP_LAUNCH_BLOCKING` 上，优先最小 repro 和 Primus-Turbo/rocSHMEM 插桩。 |
| 5. standalone DeepEP repro | `23286`-`23318`, `23457`, `23559` | 绕开 MaxText 编译/训练，直接测 Primus-Turbo `moe_dispatch` / `moe_combine`。 | 定位推进 | 1N repro 成功；2N raw JAX `process_allgather` 会丢/错 IPC handle；KV-store gather 修复 bootstrap 后，故障推进到 inter-node dispatch barrier。 | 保留 KV-store gather；当网络/内存变更后先跑 2N repro，再跑 MaxText。 |
| 6. pass3 rocSHMEM init 细化 | `25067`, `25163`, `25188`, `25191` | 确认 pass3 下是否仍是 dispatch kernel，还是 rocSHMEM init 本身卡住。 | 失败但根因更细 | `25191` 显示 dev=0 两端卡在 `rocshmem_init_attr`；其它 GPU 对能 init/barrier，剩余 rank 被 nvl barrier 拖死。 | 若再次出现类似现象，应做纯 C++ rocSHMEM repro、显式 GPU-NIC 映射、init_attr watchdog 和 mlx5 状态采集。 |
| 7. 镜像 v1.5 / v15 内存收敛 | `25396`-`25426`, `25434` | 在修过 side-effect / image 后回到完整 MaxText 2N DeepEP，调低内存占用。 | 最终 2N 连续成功 | `25423` 在 `batch=2, max_target_length=4096` OOM；`25425/25426` 用 `batch=1, length=2048, preallocate=false` 连续成功。 | 把 `25425/25426` 作为当前 known-good；下一步先 4N 同配置复现，再单独放大 batch/sequence 或恢复 prealloc。 |

## 关键分界点

### 已排除或低优先级方向

- DeepEP 关闭时 2N/4N 能跑通，说明基础 launcher、MaxText sharding 和 TGMM 主路径不是根因。
- `tgmm=false` / Shardy 路径会遇到不合法 lowering 或超大 OOM，不适合作为主线 workaround。
- ROCm verbose logging、HIP launch blocking、单纯加大 rocSHMEM heap 都没有带来高信号。
- `amd-arad` 不适合作为本 workload 的 GPU repro 队列：此前容器看到 0 AMD GPU，JAX 退到 CPU/Gloo 路径。

### 真正推动定位的信息

- `23124` core backtrace 指向 `rocshmem::Socket::connect` timeout，推动设置 `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`。
- standalone repro 证明 JAX `process_allgather` 会损坏 IPC handle，KV-store gather 是必要修复。
- `25191` 把一类失败定位到 dev=0 的 `rocshmem_init_attr` 双向卡死，应转向 NIC/rocSHMEM init 层排查。
- `25425/25426` 连续成功证明当前 2N DeepEP 主路径已可运行，剩余重点是 4N scale-up 和内存边界。

## 重要任务命令

### `22975`：2N launcher/JAX 基线，成功

```bash
./submit.sh deepseek3-671b-proxy:smoke2n-defaultfix -N 2 -p amd-rccl -t 01:00:00 -- _env_ONE_GPU_PER_PROCESS=true
```

意义：确认 2N JAX/launcher 基线可用。

### `23008`：2N DeepEP-off + dot_product + no DCN EP，成功

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:no-dcn-ep-dotattn -N 2 -p amd-rccl -t 01:00:00 -x useocpm2m-097-024 -- _env_ONE_GPU_PER_PROCESS=true _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process use_deepep_dispatch=False attention=dot_product dcn_expert_parallelism=1 dcn_fsdp_parallelism=2
```

意义：证明关闭 DeepEP 后 dot-product attention 和非 DCN expert 并行路径可跑。

### `23098`：4N TGMM baseline，成功

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n:fsdp2-ep2-tgmm-bs2-clean -N 4 -p amd-rccl -t 01:00:00 -x useocpm2m-097-024\,useocpm2m-097-028\,useocpm2m-097-030\,useocpm2m-097-038\,useocpm2m-097-039 -- _env_ONE_GPU_PER_PROCESS=true _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process
```

意义：证明 4N 下 TGMM / sharding baseline 可工作。

### `23124`：4N DeepEP reference crash，失败

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:full-deepep-te-bs2 -N 4 -p amd-rccl -t 01:00:00 -w useocpm2m-097-080\,useocpm2m-097-132\,useocpm2m-097-135\,useocpm2m-097-137 -- _env_ONE_GPU_PER_PROCESS=true _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process attention=dot_product
```

意义：4N DeepEP reference crash。后续 coredump 证明核心问题是 rocSHMEM TCP bootstrap connect timeout。

### `23125`：4N DeepEP-off diagnostic，成功

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:diag-no-deepep -N 4 -p amd-rccl -t 01:00:00 -w useocpm2m-097-080\,useocpm2m-097-132\,useocpm2m-097-135\,useocpm2m-097-137 -- _env_ONE_GPU_PER_PROCESS=true attention=dot_product use_deepep_dispatch=False
```

意义：关键强对照。相同 4N 环境关闭 DeepEP 成功，说明问题集中在 internode DeepEP / rocSHMEM。

### `23262`：nogmm + shardy OOM 路径，失败

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:nogmm-shardy -N 4 -p amd-rccl -t 01:00:00 -x useocpm2m-097-024\,useocpm2m-097-039\,useocpm2m-097-069\,useocpm2m-097-077 -- _env_ONE_GPU_PER_PROCESS=true attention=dot_product _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 per_device_batch_size=1 use_turbo_grouped_gemm=false shardy=true
```

意义：验证非 TGMM + Shardy 路径不是可用主线，容易进入超大内存需求。

### `23286`：2N standalone DeepEP repro，失败但有定位价值

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:bK -N 2 -p amd-rccl -t 20 -x useocpm2m-097-024\,useocpm2m-097-039\,useocpm2m-097-069\,useocpm2m-097-077\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-084\,useocpm2m-097-086\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_REPRO_INTERNODE_DEEPEP=1
```

意义：引入最小 DeepEP repro，避免每次完整 MaxText 编译/训练，显著缩短迭代。

### `23313`：1N standalone sanity，成功

```bash
submit.sh deepseek3-671b-proxy -N 1 -p amd-rccl -t 00:15:00 -x useocpm2m-097-024\,useocpm2m-097-039\,useocpm2m-097-069\,useocpm2m-097-077\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-084\,useocpm2m-097-086\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K7-1node-sanity
```

意义：证明 standalone repro 的 intranode DeepEP dispatch/combine 健康，问题只在 internode。

### `25191`：pass3 rocSHMEM repro，失败但定位到 dev=0 init

```bash
./submit.sh deepseek3-671b-proxy -N 2 -p amd-rccl -t 00:20:00 -x useocpm2m-097-039 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=pass3-rocshmem-repro
```

意义：定位到 dev=0 两端卡在 `rocshmem_init_attr`，不是 IPC handle 错乱，也不是 NVL buffer 分配失败。

### `25423`：v1.5 fullrun，OOM 失败

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke -N 2 -t 01:00:00 --exclude=useocpm2m-097-039\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false exp_tag=img-v1.5-sms32
```

意义：进入完整 MaxText DeepEP 路径，但 `per_device_batch_size=2`、`max_target_length=4096` 太大，出现约 `110GiB` OOM。

### `25425`：当前 2N known-good，成功

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:v15 -N 2 -t 01:00:00 --exclude=useocpm2m-097-039\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false per_device_batch_size=1 max_target_length=2048
```

意义：当前最重要 known-good。2N MaxText + internode DeepEP 成功完成，训练结束 `exit 0`。

### `25426`：当前 2N known-good repeat，成功

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:v15 -N 2 -t 01:00:00 --exclude=useocpm2m-097-039\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false per_device_batch_size=1 max_target_length=2048
```

意义：重复验证 `25425`，降低偶然成功概率。

### `25434`：v15 without `preallocate=false`，失败

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:v15-prealloc -N 2 -t 01:00:00 --exclude=useocpm2m-097-039\,useocpm2m-097-078\,useocpm2m-097-079\,useocpm2m-097-089\,useocpm2m-097-094\,useocpm2m-097-100\,useocpm2m-097-121 -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_ROCSHMEM_HEAP_SIZE=536870912 per_device_batch_size=1 max_target_length=2048
```

意义：试图恢复 XLA preallocation，但实际失败点是 MaxText MoE shape 错误：

```text
TypeError: lt got incompatible shapes for broadcasting: (32768,), (2,).
```

这条任务提示：在当前代码状态下，不应简单把失败归因为 rocSHMEM 或 preallocation 内存冲突；需要单独检查 MaxText MoE `actual_num_recv` / `recv_topk_idx` shape 逻辑。

## 下一步调试建议

### 1. 先跑 4N 同 known-good 配置

最高信号的下一步是把 `25425/25426` 配置原样扩到 4N，不要同时恢复 `max_target_length=4096`、`per_device_batch_size=2` 或 XLA preallocation。

建议命令形态：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:<new-tag> \
  -N 4 -t 01:00:00 \
  -- \
  _env_ONE_GPU_PER_PROCESS=true \
  _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
  _env_ROCSHMEM_HEAP_SIZE=536870912 \
  _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
  per_device_batch_size=1 \
  max_target_length=2048
```

若 4N 成功，再进入单变量放大。

### 2. 如果 4N 失败，按三类分流

#### OOM / allocator

特征：

- `RESOURCE_EXHAUSTED`
- `failed to allocate ... GiB`
- `BFCAllocator ran out of memory`

处理：

- 继续保持 `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- 暂不恢复 `max_target_length=4096`
- 不同时放大 batch 和 sequence

#### rocSHMEM init / network

特征：

- 卡在 `rocshmem_init_attr`
- 某个 `dev=N` 没有 `rocshmem_init_attr OK`
- JAX heartbeat 或 shutdown barrier 只是后续症状

处理：

- 跑纯 C++ rocSHMEM barrier/init repro
- 显式 pin GPU-NIC 映射
- 采集 `/sys/class/infiniband/mlx5_*/ports/1/state`
- 给 `init_attr` 加 watchdog，避免等到 JAX heartbeat 才发现

#### MaxText shape / compile-time

特征：

- Python traceback
- `TypeError: ... incompatible shapes`
- 发生在 `jax.eval_shape` / `model.init` / `moe.py`

处理：

- 不要先查 rocSHMEM
- 直接检查 MaxText MoE wrapper 中 `actual_num_recv`、`recv_topk_idx`、`valid_recv_rows` 的 shape
- 对比 `25425/25426` 与 `25434` 的 artifact / git diff

### 3. 成功后单变量放大

推荐顺序：

1. 4N + `per_device_batch_size=1` + `max_target_length=2048` + `preallocate=false`
2. 4N + `max_target_length=4096`
3. 4N + `per_device_batch_size=2`
4. 最后再考虑恢复 XLA preallocation

这样每一步失败都能明确归因，避免把内存、shape 和 rocSHMEM 网络问题混在一起。

