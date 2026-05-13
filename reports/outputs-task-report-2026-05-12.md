# JAX Internode DeepEP 2N 调通过程：outputs 任务记录汇总

- 报告生成日期：2026-05-12
- 数据来源：`maxtext-slurm/outputs/*.log` 与 `maxtext-slurm/outputs/.artifacts/artifact_*` 下的 `submit_cmd.txt` / `git_summary.txt`
- 涉及代码：`Primus-Turbo`（算子库 / DeepEP runtime），`maxtext`（框架），`maxtext-slurm`（Slurm launcher）
- 相关历史报告：`reports/internode-deepep-log-summary-2026-05-13.md`、`reports/25191-rocshmem-init-dev0-deadlock-2026-05-11.md`、`reports/internode-deepep-debug-progress-2026-05-09.md`、`reports/turbo-internode-deepep-debug-report.md`、`reports/one-gpu-per-process-deepep-debug-report.md`

每个任务/任务族按以下字段记录：
**目的**：要验证或定位什么。
**命令**：来自 `.artifacts/<id>/submit_cmd.txt` 或日志开头的代表性 submit 形态。
**状态**：日志末尾 `JOB SUMMARY` 给出的 `SUCCESS / FAILED / CANCELLED` 与时长。
**原因**：日志中能复述的根因或最近一次失败信号。
**获取的信息 / 解决的问题**：从这条任务上拿到的可复用信号。
**推断 / 下一步**：写下当时的判断，用于驱动后续 job。

---

## 0. 当前结论速览

1. **2N MaxText + Primus-Turbo internode DeepEP 已经在 `25425`、`25426`、`25613` 多次跑通**（`Status: SUCCESS (exit 0)`，wall ≈ 2–3 分钟，分别在 `useocpm2m-097-[024,026]` 与 `useocpm2m-097-[132,135]` 两组节点对上验证，分支 `llying/moe-turbo-gmm-and-deepep-v3-mp @ 9e1d255b`，镜像 `docker.io/tasimage/jax-deepep-1p1g:v1.5`）。其中 `25613` 是 prealloc=true 的对照（`_env_XLA_PYTHON_CLIENT_PREALLOCATE=false` 拿掉之后仍 SUCCESS），所以 prealloc=false 不是 known-good 的必要条件。
2. **当前 2N known-good 提交形态**（必须显式带 `DOCKER_IMAGE` 环境变量，否则会 fallback 到本地 v1.1 tarball）：

   ```bash
   DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:v1.5 \
   ./submit.sh deepseek3-671b-proxy-internode-smoke:v15 -N 2 -t 01:00:00 \
     --exclude=useocpm2m-097-039,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
     -- _env_ONE_GPU_PER_PROCESS=true \
        _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
        _env_ROCSHMEM_HEAP_SIZE=536870912 \
        _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
        per_device_batch_size=1 max_target_length=2048
   ```

   `_env_XLA_PYTHON_CLIENT_PREALLOCATE=false` 这一行去掉也能过（见 `25613`），但 `DOCKER_IMAGE` 不能省。

3. 过程中按时间分了 7 个阶段：基线 / 4N 全 DeepEP 首炸 / coredump 取证 / ablation 排错 / standalone repro / pass3 rocSHMEM init 细化 / 镜像 v1.x 内存收敛。
4. 已确认与排除的根因分布：

   - 基础 launcher、MaxText sharding、`attention=dot_product`、TGMM 主路径 → **不是根因**。
   - 一段时间内的 4N DeepEP 崩溃根因 → `rocSHMEM TCP bootstrap` 默认 5s connect 超时（`23124` core 给出强证据）。
   - `25191` 类型的 rocSHMEM 半 init → **dev=0 在 `rocshmem_init_attr` 双向卡死**。
   - 最近一批镜像 v1.3–v1.5 的 fullrun 失败 → **per-step 110GiB OOM**，与 DeepEP / 网络无关。
   - `25434` 失败 → 提交那次的 shell 没设外部环境变量 `DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:v1.5`，`container_env.sh` fallback 到默认本地 tarball `/home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar`（加载为 `jax-deepep-1p1g:v1.1`），与 v1.5 镜像里安装的 Primus-Turbo / JAX 版本不匹配，于是 MoE 路径报 `TypeError: lt got incompatible shapes for broadcasting: (32768,), (2,).`。**与 rocSHMEM、prealloc、recipe tag、MaxText 源码都无关**。

---

## 1. 阶段 A：launcher / 非 DeepEP 基线（2026-05-06 前后）

### 任务族 A1：2N JAX/launcher 主路径基线 — 代表 `22975`

**目的**：在加 DeepEP 前，先确认 2N MaxText + JAX coordinator + launcher 主路径无回归。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy:smoke2n-defaultfix -N 2 -p amd-rccl -t 01:00:00 \
  -- _env_ONE_GPU_PER_PROCESS=true
```

**状态**：成功，`completed step: 14...`，`Training END (624s)`。

**原因**：默认修复路径下 2N 主路径无障碍。

**获取的信息 / 解决的问题**：确认集群 → 容器 → MaxText 主路径在此时可用，后续 DeepEP 失败必须落在 DeepEP / rocSHMEM 子模块上，而不是再回去怀疑 launcher。

**推断**：可以在它之上叠加 internode smoke + DeepEP，逐步开关 `use_deepep_dispatch` / `PRIMUS_TURBO_JAX_DEEPEP_MODE` 做对照。

### 任务族 A2：2N internode smoke + 默认 DeepEP — 代表 `22980`、`22982`

**目的**：把 internode smoke 模型挂上 `_env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process`，看默认配置能否过。

**命令**（22980 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:idep2n -N 2 -p amd-rccl -t 01:00:00 \
  -- _env_ONE_GPU_PER_PROCESS=true \
     _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process
```

**状态**：失败。`22980` 在 `mfu_tracker.py` 行附近 `Aborted (core dumped)`；`22982` 已 `BARRIER PASSED: Starting training loop` 之后报 `JaxRuntimeError: ... fused_attn_rocm/utils_hip.cpp:252 PopulateRngStateAsync: CUDA Error: invalid configuration argument`，最终 `exit 143`。

**原因**：22982 失败点落在 **Transformer Engine ROCm fused attention RNG**，与 DeepEP 本身无关；22980 SIGABRT 早于 barrier，更可能与 DeepEP 初始化或 TE 加载相关。

**获取的信息 / 解决的问题**：把 internode smoke + per-process DeepEP 默认路径的失败拆成两条独立线 — TE RNG 一条，DeepEP / rocSHMEM 一条。

**推断**：后续若仅想验 DeepEP，应主动 `attention=dot_product` 绕开 TE fused attention，避免被 TE 路径污染信号。

### 任务族 A3：dot_product + 非 DCN expert 并行 — 代表 `23008`

**目的**：验证当前 2N 在「关 DeepEP dispatch + dot 注意力 + 非 DCN EP」组合下能否完成训练。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:no-dcn-ep-dotattn -N 2 -p amd-rccl -t 01:00:00 \
  -x useocpm2m-097-024 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process \
     use_deepep_dispatch=False attention=dot_product \
     dcn_expert_parallelism=1 dcn_fsdp_parallelism=2
```

**状态**：成功，`Training END (238s)`。

**原因**：关掉 DeepEP dispatch 后，dot 注意力 + 调整后的并行切片是健康的。

**获取的信息 / 解决的问题**：建立了一个 2N **关 DeepEP** 的可重复对照点，后续 4N 出现失败时可以立刻对比这条线。

**推断**：所有 internode smoke 后续问题，应优先怀疑 DeepEP / rocSHMEM，再回头查并行配置。

### 任务族 A4：dense / sparse TGMM 形态对照 — 代表 `23025`、`23027`

**目的**：把 OOM 来源拆成「稀疏 TGMM」与「dense matmul」两条独立线。

**命令**（23025 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:dense-matmul -N 2 -p amd-rccl -t 01:00:00 \
  -- _env_ONE_GPU_PER_PROCESS=true \
     sparse_matmul=False use_turbo_grouped_gemm=False
```

**状态**：`23025` dense 成功 (`Training END 232s`)；`23027` 稀疏 + bs=2 触发 `OOM ~129GiB`，`Training FAILED (exit=1)`。

**原因**：稀疏专家 + 通信形态下显存峰值远高于 dense。

**获取的信息 / 解决的问题**：定义了「稀疏 TGMM + DeepEP」是稳定的 OOM 高风险区，后续选 batch / seq 应该以此为参考。

**推断**：在 DeepEP 调通前，不应该把 dense / sparse / bs / max_target_length 同时拉满。

---

## 2. 阶段 B：4N 全量 DeepEP 首轮 bring-up（2026-05-07）

### 任务族 B1：4N full DeepEP 参考崩溃 — 代表 `23124`

**目的**：把 4N DeepSeek proxy + 全 DeepEP / TE 路径完整拉起来，作为参考崩溃做取证。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:full-deepep-te-bs2 -N 4 -p amd-rccl -t 01:00:00 \
  -w useocpm2m-097-080,useocpm2m-097-132,useocpm2m-097-135,useocpm2m-097-137 \
  -- _env_ONE_GPU_PER_PROCESS=true \
     _env_PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process \
     attention=dot_product
```

**状态**：失败，`Training FAILED (exit=143, 393s)`，进程产生数十 GB 量级 core。

**原因**：crash 早于训练步；与 `23286 / 23457` 同根：rocSHMEM 初始化阶段网络层崩溃。

**获取的信息 / 解决的问题**：

- 强对照 `23125` — 同 4N 同节点，仅 `use_deepep_dispatch=False`，**成功** `Training END (252s)`。这条对照证明：基础 4N 通路 + IB / 容器 / 调度都是健康的，故障被精准压缩到 DeepEP / rocSHMEM 路径。
- 23124 落下来的 core 后来被 `23138`-`23140` 用 `rocgdb` 解出，backtrace 落在 `rocshmem::Socket::connect → TcpBootstrap → rocshmem_init_attr`。

**推断**：

1. rocSHMEM bootstrap 默认 connect 超时太短，需要显式设置 `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`。
2. 后续 preflight 应做更严格的 fail-fast，避免脏节点把 DeepEP 测试时间浪费在排队。

### 任务族 B2：4N DeepEP-off 诊断 — 代表 `23125`

**目的**：在 `23124` 失败的相同 4N、相同节点上做强对照。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:diag-no-deepep -N 4 -p amd-rccl -t 01:00:00 \
  -w useocpm2m-097-080,useocpm2m-097-132,useocpm2m-097-135,useocpm2m-097-137 \
  -- _env_ONE_GPU_PER_PROCESS=true attention=dot_product use_deepep_dispatch=False
```

**状态**：成功，多 rank `completed step: 0,1,2,...`，`Training END (252s)`。

**原因**：关 DeepEP 后 4N 路径无障碍。

**获取的信息 / 解决的问题**：是后续所有「root-cause 是 internode DeepEP / rocSHMEM」结论的最强对照证据。

**推断**：之后所有 DeepEP 失败，先用 `use_deepep_dispatch=False` 跑一遍，再去拆 DeepEP 子系统。

---

## 3. 阶段 C：core / gdb 取证（2026-05-07）

### 任务族 C1：`23138`-`23140` 等取证脚本与重跑

**目的**：用 root-owned core 把 `23124` 的 crash 落到符号级。

**命令**：辅助脚本而非 `submit.sh` 命令，参见 `utils/debug_bt_23124.sh`、`utils/debug_bt_23284.sh`。流程：在同镜像里启 `--privileged` 容器，用 `rocgdb` 批量打 backtrace。

**状态**：成功取证。

**原因**：rocSHMEM TCP bootstrap connect 在 5s 默认超时下早于训练 abort，触发 SIGABRT。

**获取的信息 / 解决的问题**：

- 直接结论：`ROCSHMEM_BOOTSTRAP_TIMEOUT=60` 是必加默认。
- 流程沉淀：后续所有 GPU/通信 crash 都用「同镜像 + `--privileged` + `rocgdb` + batch backtrace」复用这套链路。

**推断**：不该在 `HIP_LAUNCH_BLOCKING` 或 verbose log 上耗时；要么取 core、要么做最小 repro，再回到 MaxText。

---

## 4. 阶段 D：ablation 与误区排除（2026-05-07 后期 ~ 2026-05-08）

### 任务族 D1：`tgmm=false` + Shardy / heap 调整等

**代表 job**：`23260`、`23262`、`23268`、`23281`、`23283`、`23284`、`23285`、`23021`（Shardy `RESOURCE_EXHAUSTED 382GiB`）。

**目的**：穷举可能配置 — `tgmm=false`、`shardy=true`、增大 rocSHMEM heap、`HIP_LAUNCH_BLOCKING`、`ROCSHMEM_DEBUG=3` / `ROC_SHMEM_DEBUG=3` 等。

**命令**（23262 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:nogmm-shardy -N 4 -p amd-rccl -t 01:00:00 \
  -x useocpm2m-097-024,useocpm2m-097-039,useocpm2m-097-069,useocpm2m-097-077 \
  -- _env_ONE_GPU_PER_PROCESS=true attention=dot_product \
     _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     per_device_batch_size=1 use_turbo_grouped_gemm=false shardy=true
```

**状态**：基本全部失败 — OOM 或不合法 lowering 或被外部 `SIGTERM` 取消。

**原因**：

- `nogmm + shardy` 路径不是受支持的稳定主线，编译期就走向不合法分片或巨大缓冲。
- 4 GiB rocSHMEM heap、`HIP_LAUNCH_BLOCKING`、ROCm verbose 日志都没有提供新的信号。

**获取的信息 / 解决的问题**：把若干「直觉上看似有意义、实际信号很低」的方向打掉，节约后续时间。

**推断**：不再花时间在 `nogmm` / shardy / debug verbosity 上；下一步必须做 standalone DeepEP repro，把 MaxText 编译耗时甩掉。

---

## 5. 阶段 E：standalone DeepEP repro（2026-05-08 ~ 2026-05-09）

### 任务族 E1：1N standalone sanity — 代表 `23313`（K7-1node-sanity）

**目的**：把 Primus-Turbo `moe_dispatch` / `moe_combine` 拆出 MaxText，在单节点先证明 intranode 路径健康。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy -N 1 -p amd-rccl -t 00:15:00 \
  -x useocpm2m-097-024,useocpm2m-097-039,useocpm2m-097-069,useocpm2m-097-077,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-084,useocpm2m-097-086,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=K7-1node-sanity
```

**状态**：成功，`Status: SUCCESS (exit 0)`。

**原因**：1N intranode 上 dispatch / combine 全程健康。

**获取的信息 / 解决的问题**：明确 repro 脚本本身是健康的，问题只在 internode；后续可放心用它做 30–60s 迭代代替 13min MaxText 编译。

**推断**：把所有 DeepEP 子系统断言放到这条线上做，比在 MaxText 上做快一个数量级。

### 任务族 E2：2N standalone DeepEP repro — 代表 `23286`、`23290`（K4-pretest）等

**目的**：在 2N 上跑 `REPRO_INTERNODE_DEEPEP=1`，找 IPC / bootstrap / dispatch / combine 哪一步会先炸。

**命令**（23286 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:bK -N 2 -p amd-rccl -t 20 \
  -x useocpm2m-097-024,...,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 _env_REPRO_INTERNODE_DEEPEP=1
```

**状态**：失败但定位推进。23290 的关键证据：`process_allgather OK=False`、`SyncFromIPCHandles: HIP Error: invalid argument`、`Assertion failed: std::memcmp(ipc_handles_...)==0`、Traceback 进 `repro_internode_deepep.py:_bootstrap_traced`。

**原因**：2N raw JAX `process_allgather` 会丢 / 损坏 IPC handle，导致 `SyncFromIPCHandles` 立即失败。

**获取的信息 / 解决的问题**：定位到必须用 KV-store gather 替换 `process_allgather`，作为 bootstrap 的强约束修复；这条修补本身后来被沉淀在 `_train.sh` / Primus-Turbo 的 host hook 里。

**推断**：之后任何「网络层 / 内存层」相关改动，先在 2N standalone repro 上跑一遍再回到 MaxText，避免误以为是 MaxText 问题。

### 任务族 E3：pass1 / pass2 仪器化 — 代表 `23457`（instrumented-pass1）、`23559`（pass2-repro）、`23562`（pass2-fullrun-no-repro）

**目的**：在 repro 之上加 PT-DBG / PT-INSTR 仪器化，把 rocSHMEM 符号解析 / barrier 流程录下来；并验证不接 repro 直接走 MaxText smoke 是否会因为同一根因失败。

**命令**（23562 形态，full smoke）：

```bash
submit.sh deepseek3-671b-proxy-internode-smoke -N 2 -p amd-rccl -t 00:30:00 \
  -x useocpm2m-097-039 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 exp_tag=pass2-fullrun-no-repro
```

**状态**：

- `23457`：失败，`rocshmem symbol resolution FAILED`，随后 `(nil) Memory access fault`，`scancel / SIGTERM`。
- `23559`：失败，能进 `[PT-DEP]` 深层，但最终 `Memory access fault (nil)`，`exit=143`。
- `23562`：失败，`Memory access fault (nil)`，`Training FAILED (exit=143, 1113s)`。

**原因**：同根 — rocSHMEM 在「全部 16 ranks」拓扑下，符号表 / 初始化未到达稳定共识，触发 nil 地址 memory access fault。

**获取的信息 / 解决的问题**：确认 MaxText full smoke 与 standalone repro 是同一类故障；只需在 standalone repro 上修就够了。

**推断**：把火力集中到 rocSHMEM init / barrier 上，下一步用更细粒度的「per-dev 钩子 + dev=0 init_attr 监控」复现。

---

## 6. 阶段 F：pass3 rocSHMEM init 细化（2026-05-11）

### 任务族 F1：pass3 / pass3-heap512 / pass4-no-prealloc — 代表 `25067`、`25163`、`25188`、`25191`、`25204`、`25205`、`25207`

**目的**：把失败再压缩 — 究竟是 dispatch kernel 行为，还是 rocSHMEM init 本身卡住。

**命令**（25191 形态）：

```bash
./submit.sh deepseek3-671b-proxy -N 2 -p amd-rccl -t 00:20:00 \
  -x useocpm2m-097-039 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_REPRO_INTERNODE_DEEPEP=1 exp_tag=pass3-rocshmem-repro
```

**状态**：

- `25191`：失败 (`~274s`)，`[rocSHMEM] backend=gda`、`rocshmem_init_attr OK`、`rocshmem_barrier_all OK (dev=2)` 等都见到了；随后部分 GPU `PT-DBG:* rocshmem symbol resolution FAILED`、`JAX coordination UNAVAILABLE / Failed to send RPC`、`Training FAILED (exit=143)`。
- `25204`：`XLA_PYTHON_CLIENT_MEM_FRACTION=.85`，失败 `~172s`，多进程 JAX coordination UNAVAILABLE。
- `25205` (`pass3-heap512`)、`25207` (`pass4-no-prealloc`)：失败，触发外部 `SIGTERM`，单独调 heap / preallocate 没解。

**原因**：rocSHMEM init 在不同 GPU 上不同步 — 一部分 dev 已完成 init+barrier，另一部分 dev=0 卡在 `rocshmem_init_attr`，进而拖死 JAX coordinator。详见 `reports/25191-rocshmem-init-dev0-deadlock-2026-05-11.md`。

**获取的信息 / 解决的问题**：

- 把失败精确缩到 **dev=0 的 `rocshmem_init_attr` 双向卡死**；不是 IPC handle 错乱，不是 NVL buffer 分配失败。
- 单独放大 heap、关 preallocate、抬 mem fraction 都无效。

**推断**：

- 写纯 C++ rocSHMEM barrier/init repro，把 JAX/MaxText 完全剥掉。
- 显式 pin GPU ↔ NIC 映射，采集 `/sys/class/infiniband/mlx5_*/ports/1/state`。
- 给 `init_attr` 加 watchdog，避免要等到 JAX heartbeat 才察觉。

### 任务族 F2：pass5* 仪器化 + scheduler — 代表 `25212`、`25217`、`25221`

**目的**：加上同步 timeout 仪器、resubmit 防偶发、drained/blacklist 调度策略。

**状态**：

- `25212` (`pass5-instr-sync-timeout`)：`*** SIGABRT received` + 多进程 `Aborted (core dumped)`，巨型 core。
- `25217` (`pass5b-instr-resubmit`)：同形态崩溃。
- `25221` (`pass5d-instr-drained-blacklist`)：被 `SIGTERM` 取消，更像调度 / drain。

**原因**：仪器化路径上的断言被触发，进入 `SIGABRT`；与 init deadlock 是同一片故障。

**获取的信息 / 解决的问题**：拿到适合 gdb 的 core，证明 resubmit 并不改变故障类。

**推断**：换思路 — 既然 rocSHMEM init / GDA QP 在某些节点对上始终不稳，把脏节点显式 `--exclude`，避免在调度层面继续耗时。

---

## 7. 阶段 G：镜像 v1.x + side-effect fix + 内存收敛（2026-05-12）

### 任务族 G1：`has-side-effect-fix` 系列 — 代表 `25396`、`25397`、`25398`、`25399`

**目的**：在 MaxText / JAX 侧加 side-effect 标注，防止 `use_deepep_dispatch=True` 路径被 XLA DCE 优化掉；同时换不同节点对验证。

**命令**（25398 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke -N 2 -t 01:00:00 \
  --nodelist=useocpm2m-097-032,useocpm2m-097-089 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
     exp_tag=has-side-effect-fix-032-089
```

**状态**：

- `25396`：失败，`GDABackend::modify_qps_init_to_rtr → modify_qp (RTR): Connection timed out (110)`，rocSHMEM init 阶段 SIGABRT，被 Slurm 终止。
- `25397`：失败，OOM `RESOURCE_EXHAUSTED: 110.06GiB @ p_train_step`，伴随 `RuntimeError: ... groups: render, video`。
- `25398`：失败，`exit=143`，多 rank `Aborted (core dumped)`，伴随 `render, video` 告警。
- `25399`：失败 `~470s`，与 25397 同根 — 110.06GiB OOM 后 JAX `Shutdown barrier has failed`（只有 1/16 rank 到 barrier）。

**原因**：

- 25396：节点 RDMA / GDA QP 建链不稳定，与 side-effect 修补正交。
- 25397/25399：side-effect-fix 把 dispatch 路径接通，但当前 yml 默认 batch/seq 下 per-step buffer 在某条 rank 上达到 110GiB。

**获取的信息 / 解决的问题**：

- side-effect 修复并不直接绕开 OOM；它只是让数据流不被 XLA 优化掉。
- 110GiB 是 `internode-smoke` 默认 batch=2 / seq=4096 的特征值，下一步必须减负。
- 容器内 `render` / `video` 组要修，否则会污染 ROCm 用户态告警。

**推断**：以「v15 + batch=1 + seq=2048」作为下一步 default；脏节点继续保留在 `--exclude` 列表里。

### 任务族 G2：镜像版本迭代 + `sms32` — 代表 `25402` (v1.2)、`25408` (v1.3)、`25410` (v1.4)、`25423` (v1.5)

**目的**：跟随 Primus-Turbo / rocSHMEM 修复滚动镜像，验证每个版本是否消除 DeepEP combine/dispatch 上的挂死。

**命令**（25410 形态）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke -N 2 -t 01:00:00 \
  --exclude=useocpm2m-097-039,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
     exp_tag=img-v1.4-sms32
```

**状态**：

- `25402` (v1.2 + side-effect-fix-sms32)：失败 `~2565s`，密集 `DeepEP combine forwarder (NVL check) timeout` / `[PT-INSTR] combine sync timeout`，随后 GPU core dump、`tasks are unhealthy (stopped sending heartbeats)`。
- `25408` (v1.3)：失败 (`Status: FAILED (exit 143) — CANCELLED`)，无 `completed step`，外部 `SIGTERM`。
- `25410` (v1.4)：失败 `~2246s`，`DeepEP notify_dispatch recv counter timeout` 风暴 + `tasks are unhealthy`，单节点 `exit 134` + 31GB 量级 core。
- `25423` (v1.5)：失败 `~469s`，OOM `110.06GiB` + JAX `Shutdown barrier` —— 与 25397 / 25399 同根，不是通信。

**原因**：

- v1.2/v1.3/v1.4 阶段，**DeepEP combine/dispatch** 上仍有跨机心跳 / 计数挂死，需要靠后续 rocSHMEM/IB 调优解决。
- v1.5 阶段，通信类挂死大幅缓解，瓶颈被推回 **per-step 110GiB OOM**。

**获取的信息 / 解决的问题**：

- 镜像不是唯一变量；同 `sms32` exclude 列表下 v1.5 + 默认 yml 仍 OOM，说明配置才是关键。
- DeepEP 通信类卡死和 OOM 在不同时间点是两条独立问题，不应一起讨论。

**推断**：先做减负（batch=1、seq=2048），再讨论 4N scale；OOM 通了之后再回头确认 v1.5 是否真把 combine 心跳挂死也修干净。

### 任务族 G3：当前 2N known-good — `25425` / `25426`

**目的**：把 `25423` 的 OOM 解掉：明确 `per_device_batch_size=1 max_target_length=2048`，配合 `_env_XLA_PYTHON_CLIENT_PREALLOCATE=false`。

**命令**：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:v15 -N 2 -t 01:00:00 \
  --exclude=useocpm2m-097-039,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_ROCSHMEM_HEAP_SIZE=536870912 _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
     per_device_batch_size=1 max_target_length=2048
```

**状态**：

- `25425`：成功，`Status: SUCCESS (exit 0)`，`Wall: 3m 1s`，节点 `useocpm2m-097-[024,026]`。
- `25426`：成功，`Status: SUCCESS (exit 0)`，`Wall: 3m 11s`，同节点对，artifact `artifact_20260512_130742_4c42`，MaxText `llying/moe-turbo-gmm-and-deepep-v3-mp @ 9e1d255b`，镜像 `tasimage/jax-deepep-1p1g:v1.5`。

**原因**：减负后 per-step buffer 不再触发 OOM；side-effect-fix 让 DeepEP dispatch 不被 XLA 优化掉；v1.5 镜像在通信层修过 combine 路径；配套 `ROCSHMEM_BOOTSTRAP_TIMEOUT=60` 让 rocSHMEM init 不会被默认 5s 卡死。

**获取的信息 / 解决的问题**：当前阶段「2N MaxText + JAX + Primus-Turbo internode DeepEP 全链路通」的第一份可重复证据。日志末尾 `Destroy: rocshmem teardown OK rank=0..15` 也证明 16 rank 都正常拆解。

**推断**：

1. 把它作为当前 known-good baseline 钉死，所有后续变更必须先在此 baseline 上保持绿。
2. 下一步顺序：先 4N + 完全相同参数 → 单独放大 `max_target_length=4096` → 单独放大 `per_device_batch_size=2` → 最后再考虑恢复 XLA preallocation。

### 任务族 G4：`25434` 反向验证 — 取消 `_env_XLA_PYTHON_CLIENT_PREALLOCATE=false`（实际跑成了错的镜像）

**目的**：验证在 known-good 上把 `XLA_PYTHON_CLIENT_PREALLOCATE` 复原为默认是否会回到失败。

**命令**（实际提交，包含一处变量错用）：

```bash
./submit.sh deepseek3-671b-proxy-internode-smoke:v15-prealloc -N 2 -t 01:00:00 \
  --exclude=useocpm2m-097-039,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_ROCSHMEM_HEAP_SIZE=536870912 \
     per_device_batch_size=1 max_target_length=2048
```

**状态**：失败，`Status: FAILED (exit 1) — Training FAILED (exit=143)`，`Wall: 7m 21s`。

**原因**：**提交 `25434` 那次 shell 里没有外部 `DOCKER_IMAGE` 环境变量**，`container_env.sh:26` 的 `DOCKER_IMAGE="${DOCKER_IMAGE:-/home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar}"` fallback 到默认本地 tarball，docker load 出 `jax-deepep-1p1g:v1.1`。日志证据：

- `25426`（成功）：`AINIC-enabled image detected: docker.io/tasimage/jax-deepep-1p1g:v1.5`，`Skipping pull of docker.io/tasimage/jax-deepep-1p1g:v1.5`。
- `25434`（失败）：`AINIC-enabled image detected: /home/liyingli/workspace/jax-deepep/jax-deepep-1p1g.tar` → `Loaded image: jax-deepep-1p1g:v1.1`。

注意 `submit.sh` 不解析 recipe 后面的 `:v15` / `:v15-prealloc`，这部分只参与 `MODEL_NAME` / `EXP_TAG` 拼接，并不会让 launcher 切换镜像；两边镜像差异完全来自调用前 shell 的 `DOCKER_IMAGE` 是否被 export。`submit_cmd.txt` 只保存 `./submit.sh ...` 本体，不会记录外部环境变量，所以仅看它会误以为两条命令一致。

两个 job 的 MaxText 源码都被 `_container.sh` 显式 `git checkout origin/llying/moe-turbo-gmm-and-deepep-v3-mp` 到同一 commit `9e1d255b fix internode deepep`，所以 MaxText 源码一致；但容器内自带的 Primus-Turbo / JAX wheel 在 v1.1 与 v1.5 上不同，与 MaxText 新代码不兼容，触发 MoE 路径 `TypeError: lt got incompatible shapes for broadcasting: (32768,), (2,).`。

**获取的信息 / 解决的问题**：

- 失败既不是 rocSHMEM，也不是 preallocation 内存冲突，更不是 MaxText 源码本身的 shape bug。
- 暴露了 launcher 的一个坑：`DOCKER_IMAGE` 未设置时静默 fallback 到本地 tarball，给上层调试制造迷惑信号；`submit_cmd.txt` 只记录 `./submit.sh` 之后的部分，无法看出外部 `DOCKER_IMAGE` 是否被设置。

**推断 / 下一步**：

- 想做「known-good 减去 `XLA_PYTHON_CLIENT_PREALLOCATE=false`」的对照实验，正确命令是显式带上 `DOCKER_IMAGE`，只在 passthrough 里删那一项（必要时用 `exp_tag=v15-prealloc-on` 区分 OUTPUT_PATH）。后续 `25613` 已用这条命令复跑成功，见任务族 G5。
- 建议在 launcher 里加 fail-fast 或把 `DOCKER_IMAGE` 实际取值（含 fallback 是否触发）也写进 `submit_cmd.txt` / artifact，避免下次又把镜像差异藏起来。
- 之后做 known-good 之外的实验，仍要把可能的失败维度（镜像 / OOM / rocSHMEM / shape）分开追。

### 任务族 G5：`25613` 正确对照 — prealloc=true 同样跑通

**目的**：用与 `25426` 严格一致、仅去掉 `_env_XLA_PYTHON_CLIENT_PREALLOCATE=false` 的命令复跑，并显式锁定 `DOCKER_IMAGE`，独立验证 prealloc 不是 known-good 的必要条件，同时把 `25434` 的失败归因彻底锁死在「镜像不一致」。

**命令**（这次显式带了 `DOCKER_IMAGE`，并用 `exp_tag` 区分 OUTPUT_PATH）：

```bash
DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:v1.5 \
./submit.sh deepseek3-671b-proxy-internode-smoke:v15 -N 2 -t 01:00:00 \
  --exclude=useocpm2m-097-039,useocpm2m-097-078,useocpm2m-097-079,useocpm2m-097-089,useocpm2m-097-094,useocpm2m-097-100,useocpm2m-097-121 \
  -- _env_ONE_GPU_PER_PROCESS=true \
     _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
     _env_ROCSHMEM_HEAP_SIZE=536870912 \
     per_device_batch_size=1 max_target_length=2048 \
     exp_tag=v15-prealloc-on
```

**状态**：成功，`Status: SUCCESS (exit 0)`，`Wall: 2m 16s`，节点 `useocpm2m-097-[132,135]`，artifact `artifact_20260513_072412_5b55`。`Training END (118s)`。

**关键日志**：

- 镜像：`AINIC-enabled image detected: docker.io/tasimage/jax-deepep-1p1g:v1.5` → 节点首次拉，`Status: Downloaded newer image for tasimage/jax-deepep-1p1g:v1.5`。
- MaxText：`HEAD is now at 9e1d255b fix internode deepep`（与 25425/25426 一致）。
- 训练：`step 0` loss `12.270` / `step 1` loss `10.816`，第二步 `TFLOP/s/device 44.825`、`MFU 3.43%`。
- rocSHMEM teardown 在 16 rank 上全部 `Destroy EXIT`，无 fault。

**获取的信息 / 解决的问题**：

- 把 `25434` 「FAIL 还是 PASS」这一刀的归因彻底坐实：**只要镜像锁到 v1.5，prealloc 开 / 关都能跑通**。`25434` 的 MoE `TypeError: lt got incompatible shapes for broadcasting: (32768,), (2,).` 是 v1.1 镜像里 Primus-Turbo / JAX wheel 与新 MaxText 不匹配的次生症状，不是 MaxText 源码 bug。
- 多出一组「不同节点对（132/135）也能 SUCCESS」的证据，known-good 不再依赖 024/026 这一对节点。

**推断 / 下一步**：

- 把 prealloc 从「known-good 必要条件」列表里拿掉；后续 4N 扩展也可以两条线分别跑（`PREALLOCATE=false` 与默认）。
- 把「`DOCKER_IMAGE` 必须显式 export」固化进文档或 wrapper 脚本；最小成本是改 `container_env.sh` 让默认值变成注册表 tag 而不是本地 tarball，或者在 `_container.sh` 启动时回显「DOCKER_IMAGE source = explicit env / fallback」一行，避免下次又把镜像差异藏起来。

---

## 8. 横切结论与下一步

### 已确认或排除的方向

- 关 DeepEP 时 2N / 4N 都能跑通，**基础 launcher、MaxText sharding、TGMM 主路径** 已经从嫌疑名单上排除。
- `nogmm + shardy`、`HIP_LAUNCH_BLOCKING`、单纯加大 rocSHMEM heap、ROCm verbose log 都不再做主线，信号低。
- `amd-arad` 队列不适合此 workload（容器看到 0 AMD GPU，JAX 退到 CPU/Gloo），后续只用 `amd-rccl`。
- 110GiB OOM 与 DeepEP 通信挂死是两类独立故障，不应一并讨论。

### 真正推动定位的信息

- `23124` core backtrace → `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`。
- standalone repro → 用 KV-store gather 替代 `process_allgather`。
- `25191` → dev=0 `rocshmem_init_attr` 双向卡死。
- `25425/25426` → 2N MaxText DeepEP 全链路第一份成功证据。
- `25613` → 锁定 `DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:v1.5` + prealloc=true 也能跑通，把 `25434` 的失败彻底归到「镜像 fallback 到 v1.1」，并把 prealloc 移出 known-good 必要条件。

### 推荐下一步

1. 4N 直接复用 `25425/25426/25613` 配置（**`DOCKER_IMAGE` 必须显式 export**），**不要同时**放大 batch / seq。prealloc 默认开 / 关都已被验证，可作为单独维度后置：

   ```bash
   DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:v1.5 \
   ./submit.sh deepseek3-671b-proxy-internode-smoke-4n-full:<new-tag> \
     -N 4 -t 01:00:00 \
     -- _env_ONE_GPU_PER_PROCESS=true \
        _env_ROCSHMEM_BOOTSTRAP_TIMEOUT=60 \
        _env_ROCSHMEM_HEAP_SIZE=536870912 \
        _env_XLA_PYTHON_CLIENT_PREALLOCATE=false \
        per_device_batch_size=1 max_target_length=2048
   ```

2. 如果 4N 失败，按四类分流：

   - 进 docker 阶段就看 `AINIC-enabled image detected:` 那一行确认是不是 `tasimage/jax-deepep-1p1g:v1.5`，不是的话先修 `DOCKER_IMAGE`。
   - 看到 `RESOURCE_EXHAUSTED` / `failed to allocate ... GiB` → 暂不恢复 `max_target_length=4096`，也不要同时放大 batch / seq。
   - 看到某个 `dev=N` 没有 `rocshmem_init_attr OK` / `Shutdown barrier has failed` → 跑纯 C++ rocSHMEM barrier/init repro，pin GPU-NIC 映射，采集 IB 状态。
   - 看到 Python `TypeError: ... incompatible shapes` → **先检查镜像**，再才看代码；`25434` 这一类几乎都是镜像版本不匹配。

3. 单变量放大顺序：seq=2048 → seq=4096 → batch=1→2 → preallocation 默认（已知与 v15 兼容）。

