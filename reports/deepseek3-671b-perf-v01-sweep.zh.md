# DeepSeek-V3 671B —— `perf-v0.1` 镜像 8 节点跑通验证

- **日期：** 2026-05-13（首次记录；随作业进度滚动更新）
- **模型：** `deepseek3-671b`（基于 [`configs/deepseek3-671b.gpu.yml`](../configs/deepseek3-671b.gpu.yml)）
- **硬件：** 8 节点 × 8× AMD MI355（每卡 288 GB HBM），Pensando AINIC 互联，Slurm 分区 `amd-rccl`
- **镜像：** `docker.io/tasimage/jax-deepep-1p1g:perf-v0.1`
- **MaxText 补丁分支：** `llying/moe-turbo-gmm-and-deepep-v3-mp`（`container_env.sh` 默认）
- **`train_env.sh` 默认 XLA / NCCL / rocSHMEM 设置**：未做修改，沿用仓库默认（包含 `_env_ENABLE_RAGGED_ONESHOT_KERNEL=0` 默认禁用 one-shot 内核、`XLA_PYTHON_CLIENT_MEM_FRACTION=.93`、`ROCSHMEM_GDA_PROVIDER=mlx5` 等）。
- **本文档目的：** 完整记录每个任务的（1）透传参数 / 环境变量、（2）`submit.sh` 启动命令、（3）作业 ID、节点、状态、（4）关键观测（TGS、step time、loss、错误日志摘要）。

---

## 受测组合一览

| # | 标签 | sparse_matmul | use_turbo_grouped_gemm | use_deepep_dispatch | shardy | dcn_fsdp | dcn_expert | ici_expert | 启动器 | 备注 |
|---|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| 1 | `task1-default`        | False *(默认)* | False *(默认)* | False *(默认)* | False *(默认)* | 8 | 1 | 8 | 1-node/proc | dense matmul，对应 sweep 中的 `dense-cf1.25` 路径 |
| 2 | `task2-sparse-shardy`  | True | False | False | True | 8 | 1 | 8 | 1-node/proc | 对应 sweep 中的 `sparse` 路径 |
| 3 | `task3-sparse-tgmm`    | True | True | False | False | 8 | 1 | 8 | 1-node/proc | 对应 sweep 中的 `sparse-gmm-fixed` 路径 |
| 4 | `task4-tgmm-deepep`    | True | True | True | False | 8 | 1 | 8 | 1-node/proc | 对应 sweep 中的 `sparse-gmm-deepep-v3` 路径（`v3-mp` 分支） |
| 4'| `task4p-tgmm-deepep-1g`| True | True | True | False | 8 | 1 | 8 | **1-GPU/proc** | sweep 中此组合在旧分支报 `AssertionError: EP ranks=1`；本次借 `v3-mp` 分支验证 |
| 5 | `task5-tgmm-deepep-1g-dcnep2` | True | True | True | False | **4** | **2** | 8 | **1-GPU/proc** | internode-DeepEP；旧 MaxText 上有 `dcn_expert_parallelism=1` 校验，本次借 `v3-mp` 分支测试 |

> 说明：
>
> - `dcn_data_parallelism=-1` 由 MaxText 自动 sharding 推断（8 节点 × 8 GPU = 64 ranks 下，task 5 中 `dcn_fsdp=4 × dcn_ep=2 × ici_ep=8 = 64` 已用完）。
> - 启动器 = "1-node/proc" 时每节点 1 个 JAX 进程、8 张本地设备；"1-GPU/proc" 时每节点 8 个进程、每进程 1 张本地设备（通过 `_env_ONE_GPU_PER_PROCESS=true` 激活）。
> - `per_device_batch_size` 默认沿用 yml 中的 `16`；如遇 OOM，会先尝试调小 pdbs，必要时再调 `_env_XLA_PYTHON_CLIENT_MEM_FRACTION`，并在对应任务节记录调整。

---

## 公共启动配置

```bash
# 镜像 / 补丁分支（在所有 submit.sh 调用前 export 一次即可，整个会话沿用）
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
# MAXTEXT_PATCH_BRANCH 沿用 container_env.sh 默认：llying/moe-turbo-gmm-and-deepep-v3-mp

# 公共 sbatch 参数（每个任务都会带）
SBATCH_ARGS_COMMON="--partition=amd-rccl --nodes=8 --time=90:00"

# 工作目录
cd /home/liyingli/workspace/jax-deepep/maxtext-slurm
```

> `--time=90:00`（90 分钟）沿用 sweep 文档"DCN_EP=4 时建议 90:00"的经验值；DS3 dense / sparse-gmm 在 8 节点上单作业普遍 < 30 分钟即可完成 15 步，留出余量。

---

## 任务 1：默认 dense 配置

> **目的：** 验证 `perf-v0.1` 镜像在 8 节点上能正常拉取、启动并完成 DS3-671B dense 训练；作为后续 5 个任务的基线。

**等价配置：** `sparse_matmul=False`（默认即 dense matmul，capacity_factor=1.25），1-node/proc，沿用 yml 中所有默认。

**启动命令（已执行）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task1-default \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    --
```

> *(`--` 后无透传参数；EXP_TAG 仅来自冒号后缀 `ds3-8n-sweep-task1-default`。)*

**作业信息：**

- **作业 ID：** `25631`
- **作业名：** `JAX-deepseek3-671b-ds3-8n-sweep-task1-default`
- **节点：** `useocpm2m-097-[041,047-048,078-079,085,115,131]`（8 个）
- **提交时间：** 2026-05-13 18:25:48 (UTC+8)
- **日志：** `outputs/25631-JAX-deepseek3-671b-ds3-8n-sweep-task1-default.log`
- **stage 进度：** preflight 28s ✓ → docker-pull ~7min ✓ → ECC 1s ✓ → train 启动后 ~6min 触发 RCCL/IB 错误（被人工 scancel）。
- **结束状态：** `FAILED 1:0`，墙钟 ~15min46s。

**首次运行结果：失败（非 OOM）。**

**关键错误（IB/RCCL 跨节点建链失败）：**

```text
jax.errors.JaxRuntimeError: INTERNAL: RCCL operation ncclCommInitRankConfig(...)
   failed: unhandled system error
Last RCCL warning: 'Call to ibv_modify_qp failed with 110 Connection timed out,
   on dev mlx5_0:1, curr state INIT, next state RTR,
   local GID index 3, local GID ::ffff:10.224.0.73,
   remote GID ::ffff:10.224.3.40'
```

8 张 mlx5 网卡（`mlx5_0/2/3/4/5/7/8/9`）在 `multihost_utils.process_allgather` 阶段全部 110s 超时，QP 状态从 `INIT → RTR` 切换失败。

**根因诊断（用户确认）：**

不是 IB 用户态不匹配，而是**这 8 个节点中存在网络异常的节点**——历史上其它任务也是用 `--exclude` 把坏节点排掉来跑通的。

发出 `ibv_modify_qp` 错误的节点（直接的"嫌疑节点"）：

- `useocpm2m-097-078`（IB 接口 IP 后缀 `.73`）—— 8 张 mlx5 网卡都报超时
- `useocpm2m-097-079`（IB 接口 IP 后缀 `.254`）—— 8 张 mlx5 网卡都报超时

它们建链失败的远端 IP 后缀是 `.40` 和 `.204`（这两个对端可能也有问题，但主机名映射在日志里看不出来；无法直接通过 IP 后缀逆推到 `useocpm2m-097-XXX`）。

**下一步：** 用 `--exclude=useocpm2m-097-[078-079]` 重提任务 1。如果 Slurm 重新挑出来的 8 个节点里还有"看不见的"坏节点（即上一轮 IP 后缀 `.40` / `.204` 对应的主机），届时再继续扩大排除集。

---

### 第二次提交（重试 1，已 RUNNING）

**启动命令：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task1-default-retry1 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    --exclude=useocpm2m-097-[078-079] \
    --
```

- **作业 ID：** `25646`
- **作业名：** `JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry1`
- **节点：** `useocpm2m-097-[041,047-048,070,085,115,131,136]`（排除 078, 079；新增 070, 136）
- **提交时间：** 2026-05-13 19:00:14 (UTC+8)
- **日志：** `outputs/25646-JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry1.log`
- **结果：** 通过 BARRIER（说明 RCCL 跨节点初始化无 ibv_modify_qp 错误），但随后在编译 / kernel-launch 阶段静默挂起 ~15 分钟（最后日志停在 `LL cutoff points not detected for a supported arch gfx942` 的 RCCL warning，无任何 step 输出）。30:50 时人工 scancel；CANCELLED+。
- **判断：** 这次 8 节点中 6 个（`041,047-048,085,115,131`）与第一次失败的 8 节点重叠，可能存在尚未识别出的"看不见的"坏节点（即上一轮 IP 后缀 `.40` / `.204` 对应的主机）。

---

### 第三次提交（重试 2，更激进的 exclude）

把前两次出现过的所有 10 个节点全部排除（041, 047, 048, 070, 078, 079, 085, 115, 131, 136），让 Slurm 从其它节点池里挑全新 8 个节点。

**启动命令：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task1-default-retry2 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    "--exclude=useocpm2m-097-[041,047-048,070,078-079,085,115,131,136]" \
    --
```

- **作业 ID：** `25647`
- **作业名：** `JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry2`
- **节点：** `useocpm2m-097-[024,026,032,077,089,094,100,137]`（与前两次完全不重叠的 8 个新节点）
- **提交时间：** 2026-05-13 19:35:54 (UTC+8)
- **日志：** `outputs/25647-JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry2.log`
- **结果：** **FAILED 1:0**，墙钟 6min29s。再次出现 `ibv_modify_qp failed` 110s 超时，错误来自 094 / 137 / 089 三个新节点；连不上的远端 IP 后缀是 `.182` / `.190` / `.235` / `.8.235`。所有 8 个 NODE_EXIT 全部 exit=143 / 1。

---

### 三次失败小结

| 试次 | 作业 | 节点 | 结果 |
|---|---|---|---|
| 1 | 25631 | 041, 047-048, 078-079, 085, 115, 131 | ibv_modify_qp 110s 超时（来自 078/079）|
| 2 | 25646 | 041, 047-048, **070**, 085, 115, 131, **136** | 通过 BARRIER，随后 RCCL kernel-launch 静默挂起 15min |
| 3 | 25647 | **024, 026, 032, 077, 089, 094, 100, 137**（8 个全新）| ibv_modify_qp 110s 超时（来自 094/137/089）|

每次随机重选节点后 IB 建链都失败，说明这不是少数坏节点的问题，更像是镜像 IB 用户态与集群普遍不匹配，或集群 IB fabric 当前整体状态异常。

**停下来等用户决策下一步。**

---

### 第四次提交（重试 3，永久 exclude 所有 `ibv_modify_qp` 源节点）

用户指示：保持原策略，**累积排除每次失败时发出 `ibv_modify_qp` 错误的节点**，重提。

| 失败作业 | `ibv_modify_qp` 源节点 |
|---|---|
| 25631 | 078, 079 |
| 25647 | 089, 094, 137 |

合并：`--exclude=useocpm2m-097-[078-079,089,094,137]`。

> 注：retry1 (25646) 是静默挂起，没出 `ibv_modify_qp`，所以按用户标准不计入排除列表（这次 retry3 选到的 8 节点中有 6 个跟 25646 重叠）。

**启动命令：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task1-default-retry3 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    "--exclude=useocpm2m-097-[078-079,089,094,137]" \
    --
```

- **作业 ID：** `25654`
- **作业名：** `JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry3`
- **节点：** `useocpm2m-097-[041,047-048,070,086-087,131,136]`（其中 086, 087 是首次出现；其余 6 个在 25646 出现过，已有 perf-v0.1 镜像缓存）
- **提交时间：** 2026-05-13 20:22:52 (UTC+8)
- **日志：** `outputs/25654-JAX-deepseek3-671b-ds3-8n-sweep-task1-default-retry3.log`
- **结果：** 跟 retry1 一样的进度模式：通过 BARRIER（无 `ibv_modify_qp` 错误），随后 RCCL kernel-launch 阶段静默挂起 16+ 分钟，无任何 step 0 输出；CANCELLED+，墙钟 31min47s。
- **关键观察：** retry1 (25646) 和 retry3 (25654) 都是 silent-hang，两次的 8 节点交集是 `useocpm2m-097-[041,047-048,070,131,136]`（6 个节点）——可能其中存在挂起元凶。

---

### 四次失败汇总

| 试次 | 作业 | 节点（**粗体**＝交集 / *斜体*＝独有） | 失败模式 |
|---|---|---|---|
| 原始 | 25631 | **041, 047-048**, *078-079*, *085*, **131**, *115* | `ibv_modify_qp`（源 078, 079）|
| retry1 | 25646 | **041, 047-048**, **070**, *085*, *115*, **131**, **136** | silent hang 15min |
| retry2 | 25647 | *024, 026, 032, 077*, *089*, *094*, *100*, *137* | `ibv_modify_qp`（源 089, 094, 137）|
| retry3 | 25654 | **041, 047-048**, **070**, *086-087*, **131**, **136** | silent hang 16min |

> retry1 ∩ retry3 共同节点 = `useocpm2m-097-[041,047-048,070,131,136]`（6 个）。两次 silent hang 都包含这组节点。

---

**当前累积排除清单：** `useocpm2m-097-[078-079,089,094,137]`（5 个 `ibv_modify_qp` 源节点）。

**`ibv_modify_qp` 路径已经按用户策略迭代了一轮**：retry2 报告了新的 3 个 `ibv_modify_qp` 源节点 (089, 094, 137)，加进排除后 retry3 不再出 `ibv_modify_qp` ——但落到了静默挂起。沉默挂起没有源节点指认，需要换一种处理思路。

**停下来等用户决策。**

---

## 任务 2：sparse_matmul=True、shardy=True（待运行）

**等价配置：** sweep 中的 `sparse` 列。

**启动命令（计划）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task2 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    -- sparse_matmul=true shardy=true
```

**作业信息：** *待提交后填入。*

**结果：** *待运行。*

---

## 任务 3：sparse_matmul=True、use_turbo_grouped_gemm=True（待运行）

**等价配置：** sweep 中的 `sparse-gmm-fixed (1-node)` 列。

**启动命令（计划）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task3 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    -- sparse_matmul=true use_turbo_grouped_gemm=true
```

**作业信息：** *待提交后填入。*

**结果：** *待运行。*

---

## 任务 4：sparse + turbo_gmm + deepep_dispatch（1-node/proc，待运行）

**等价配置：** sweep 中的 `sparse-gmm-deepep` 路径，但走 `v3-mp` 分支（对应 sweep 中的 v3 改动）。

**启动命令（计划）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task4 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    -- sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true \
       ici_expert_parallelism=8 dcn_fsdp_parallelism=8
```

> 显式写出 `ici_expert_parallelism=8 dcn_fsdp_parallelism=8`，与 yml 默认一致——便于复核。

**作业信息：** *待提交后填入。*

**结果：** *待运行。*

---

## 任务 4'：任务 4 + 1-GPU/proc 启动器（待运行）

**等价配置：** 与任务 4 完全相同，仅启动器换为 1-GPU/proc。旧 sweep 在该组合上报 `AssertionError: EP ranks=1`；本次借 `v3-mp` 分支验证是否已解锁。

**启动命令（计划）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task4p \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    -- sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true \
       ici_expert_parallelism=8 dcn_fsdp_parallelism=8 \
       _env_ONE_GPU_PER_PROCESS=true
```

**作业信息：** *待提交后填入。*

**结果：** *待运行。*

---

## 任务 5：internode-DeepEP（dcn_ep=2，1-GPU/proc，待运行；**重点关注**）

**等价配置：** 旧版 MaxText 在 `use_deepep_dispatch=true ⇒ dcn_expert_parallelism == 1` 校验下不允许该组合；本次借 `v3-mp` 分支尝试解锁 internode-DeepEP。

> **若运行时报非 OOM 错误（如 `pydantic ValidationError`、`AssertionError: EP ranks=...`、`rocSHMEM` 相关失败、其他 MaxText pyconfig 校验等），将立即停下来找用户确认，再决定是改分支、加环境变量还是改透传。**

**启动命令（计划）：**

```bash
export DOCKER_IMAGE=docker.io/tasimage/jax-deepep-1p1g:perf-v0.1
./submit.sh deepseek3-671b:ds3-8n-sweep-task5 \
    --partition=amd-rccl --nodes=8 --time=90:00 \
    -- sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true \
       ici_expert_parallelism=8 dcn_expert_parallelism=2 dcn_fsdp_parallelism=4 \
       _env_ONE_GPU_PER_PROCESS=true
```

**作业信息：** *待提交后填入。*

**结果：** *待运行。*

---

## OOM / 异常处理预案

按 sweep 文档的经验值，触发 OOM 时按以下顺序调整（每次只调一个变量、逐步重提）：

1. `per_device_batch_size`（默认 16）→ 8 → 4 → 2 → 1。dense 一般可保持 ≥ 8；sparse-gmm 路径预计降到 6–8；DeepEP 路径预计降到 1–2。
2. `_env_XLA_PYTHON_CLIENT_MEM_FRACTION`（默认 .93）→ .96。**注意：** sweep 文档结论 ⑥ 提示推到 .97+ 会饿死 RCCL，导致集合通信初始化 OOM；只在日志显示 XLA 自身分配上限时才提升。
3. `remat_policy`（默认 "full"）—— 若上述两项都已用尽仍 OOM，可考虑暂保留默认；超出本任务范围。

非 OOM 错误（如 pyconfig ValidationError、rocSHMEM 初始化失败、AssertionError 等）会立即停下来记录现场并找用户确认。

---

## 进度时间线

| 时间（UTC+8） | 事件 |
|---|---|
| 2026-05-13 18:08 | 文档初始化；准备提交任务 1。 |
| 2026-05-13 18:25 | 任务 1 提交，slurm 作业 ID `25631`，分配到 8 个节点 `useocpm2m-097-[041,047-048,078-079,085,115,131]`。 |
| 2026-05-13 18:26 | 任务 1 preflight 28s 通过，进入 docker pull 阶段（首次拉取 `docker.io/tasimage/jax-deepep-1p1g:perf-v0.1`，超时 900s）。 |
| 2026-05-13 18:33 | 任务 1 docker pull 完成（~7min），ECC 通过，进入训练阶段。 |
| 2026-05-13 18:36–18:38 | 8 张 mlx5 网卡集体在 `ibv_modify_qp INIT→RTR` 阶段超时 110s，RCCL `ncclCommInitRankConfig` 失败。 |
| 2026-05-13 18:41 | scancel 25631；FAILED 1:0；停下来与用户确认下一步。 |
| 2026-05-13 18:51 | 用户确认根因：是 8 节点中存在网络异常的节点，并非镜像问题。建议用 `--exclude` 排掉。 |
| 2026-05-13 19:00 | 任务 1 重试提交，slurm 作业 ID `25646`，新节点 `useocpm2m-097-[041,047-048,070,085,115,131,136]`，已排除 078-079。 |
| 2026-05-13 19:09 | 25646 通过 BARRIER（无 ibv_modify_qp 错误），随后 RCCL kernel-launch 阶段静默挂起，日志停在 `LL cutoff points not detected for gfx942` 的 RCCL warning。 |
| 2026-05-13 19:35 | scancel 25646；同期排除前两次所有 10 个节点（含 6 个重叠），重提任务 1 retry2，slurm 作业 ID `25647`，全新节点 `useocpm2m-097-[024,026,032,077,089,094,100,137]`。 |
| 2026-05-13 19:42 | 25647 在通过 BARRIER 后又出现同样的 `ibv_modify_qp` 110s 超时（来自 094 / 137 / 089，连不上 peer IP `.182`/`.190`/`.235`），FAILED 1:0。 |
| 2026-05-13 19:46 | 三次连续失败（每次随机的 8 节点中都有 IB 建链问题）：判断不是少数坏节点，停下来等用户决策。 |
| 2026-05-13 20:22 | 用户重申策略：继续累积排除 `ibv_modify_qp` 源节点。提交任务 1 retry3 (`25654`)，`--exclude=useocpm2m-097-[078-079,089,094,137]`，分配到 `useocpm2m-097-[041,047-048,070,086-087,131,136]`。 |
| 2026-05-13 20:31 | 25654 通过 BARRIER（无 `ibv_modify_qp`），随后 RCCL kernel-launch 阶段又一次 silent hang 16+ 分钟。 |
| 2026-05-13 20:55 | scancel 25654；retry1 ∩ retry3 共同节点 `041, 047-048, 070, 131, 136`（6 个）两次 silent-hang 都中招。停下来请示用户。 |

