# 25191 失败定位：dev=0 上 `rocshmem_init_attr` 双向死锁

- 任务：`JAX-deepseek3-671b-proxy-..._env_REPRO_INTERNODE_DEEPEP_1-exp_tag_pass3-rocshmem-repro`
- JOB_ID：25191
- 日期：2026-05-11
- 节点：useocpm2m-097-024 / useocpm2m-097-026，2 nodes × 8 GPU = 16 进程，`ONE_GPU_PER_PROCESS=true`
- 入口脚本：`maxtext-slurm/utils/repro_internode_deepep.py`（最小 DeepEP dispatch/combine repro，绕开 MaxText/优化器）
- 关键运行时环境：
  - `ROCSHMEM_BACKEND=gda`
  - `ROCSHMEM_GDA_PROVIDER=mlx5`
  - `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=eth0`
  - `ROCSHMEM_BOOTSTRAP_TIMEOUT=60`
  - `ROCSHMEM_HEAP_SIZE=2147483648`、`ROCSHMEM_MAX_NUM_CONTEXTS=64`

最终退出：`Training FAILED (exit=143, 274s)` —— srun task 1 因 SIGABRT（exit 134）被踢，task 0 随后被 srun 取消（exit 143）。

---

## 1. 失败链条总览

整个跑完全发生在 host-side rocSHMEM bootstrap 后、第一次 `moe_dispatch` 的 device-side barrier 上。日志关键尾巴：

```text
0: F0511 07:38:29 client.h:77] Terminating process because the JAX distributed service detected fatal errors ...
   UNAVAILABLE: Failed to send RPC to coordination service ...
1: /artifact_.../_train.sh: line 185: 315 Aborted (core dumped)
   python3 -u .../utils/mfu_tracker.py ...
```

退出码 **134 = 128 + SIGABRT**，触发了 ~17–38 GiB 的 coredump 写盘 30 s。

这两行只是症状：`JAX coordination service` 心跳超时（`heartbeat_timeout_s=100`）后强行撕掉所有 rank，并不是真正的故障。

---

## 2. 真正卡住的位置

`maxtext-slurm/utils/repro_internode_deepep.py` 在 `_install_primus_turbo_traces()` 里把 Primus-Turbo 的 host 调用全部插了 `PT-DBG`/`PT-DEP`/`PT-RS` 桩，对照源码逐行能精准定位阻塞点。

### 2.1 host-side rocSHMEM init —— dev=0 死锁

源码：

```74:93:Primus-Turbo/csrc/kernels/deep_ep/runtime.cu
int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks,
         bool low_latency_mode) {
    ...
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr));
    PT_RS_LOG("init: set_attr_uniqueid_args OK rank=%d num_ranks=%d", rank, num_ranks);
    PRIMUS_TURBO_CHECK_ROCSHMEM(
        rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));
    int post_my_pe = rocshmem::rocshmem_my_pe();
    int post_n_pes = rocshmem::rocshmem_n_pes();
    PT_RS_LOG("init: rocshmem_init_attr OK arg_rank=%d arg_num=%d -> my_pe=%d n_pes=%d",
              rank, num_ranks, post_my_pe, post_n_pes);
```

7 个 GPU 对（dev=1..7）两端都成功打出 `init: rocshmem_init_attr OK`、`pre/post rocshmem_barrier_all OK`，例如 dev=3：

```text
0: [PT-RS dev=3] init: rocshmem_init_attr OK arg_rank=0 arg_num=2 -> my_pe=0 n_pes=2
0: [PT-RS dev=3] init: pre rocshmem_barrier_all my_pe=0 n_pes=2
1: [PT-RS dev=3] init: rocshmem_init_attr OK arg_rank=1 arg_num=2 -> my_pe=1 n_pes=2
1: [PT-RS dev=3] init: pre rocshmem_barrier_all my_pe=1 n_pes=2
0: [PT-RS dev=3] init: post rocshmem_barrier_all OK my_pe=0
1: [PT-RS dev=3] init: post rocshmem_barrier_all OK my_pe=1
```

**唯独 dev=0**（即两端的 GPU 0，全局 rank=0 与 rank=8）只走到 `set_attr_uniqueid_args OK` 就再无下文。整份日志里 dev=0 出现次数统计：

| 调用桩位                                          | dev=0 出现次数 |
|--------------------------------------------------|---------------|
| `[PT-RS dev=0] init ENTER`                       | 2（node0+node1） |
| `[PT-RS dev=0] init: set_attr_uniqueid_args OK`  | 2 |
| `[PT-RS dev=0] init: rocshmem_init_attr OK`      | **0** |
| `[PT-RS dev=0] init: pre rocshmem_barrier_all`   | **0** |
| `[PT-RS dev=0] init: post rocshmem_barrier_all OK` | **0** |
| `[PT-RS dev=0] alloc:`                            | **0** |

即 **`rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr)` 在两端 dev=0 上对称死锁**（两端都需要返回，握手才完成，所以双向都看不到 OK）。

而且 dev=0 在这之前所有前置条件都已成立：

- KV-gather 出的 IPC handle 双端自校验通过：`match=True local=b5b7680591b96ed4 gathered[0]=b5b7680591b96ed4`
- NVL IPC 8 个 peer 全部 `cudaIpcOpenMemHandle` 成功并上传 `buffer_ptrs_gpu / barrier_signal_ptrs_gpu`
- `Buffer ctor: nvl alloc OK rank=0/8 ... buffer_ptrs[nvl_rank]=0x7f452f000000 ...`

唯一发生死锁的环节就是 dev=0 的 rocSHMEM 2-PE bootstrap。

### 2.2 其它 14 个 rank 因此在 device kernel 里被拖死

由于 rank=0、rank=8 卡在 host 端 init，它们的 dispatch kernel **从未被启动**。但另外 14 个 rank 的 host 都成功了，CPU 排队提交 `moe_dispatch`，kernel 进到 `internode.cu::notify_dispatch`：

```187:194:Primus-Turbo/csrc/kernels/deep_ep/internode.cu
PT_K_BLOCK0_LOG("notify_dispatch DEV stage=pre-barrier1(rdma)");
if (thread_id == kWarpSize)
    nvshmem_barrier_with_same_gpu_idx<kLowLatencyMode>(rdma_team);

PT_K_BLOCK0_LOG("notify_dispatch DEV stage=post-barrier1(rdma) pre-barrier1(nvl)");
barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
__syncthreads();
PT_K_BLOCK0_LOG("notify_dispatch DEV stage=post-barrier1(nvl)");
```

NVL barrier 实现为 8-PE 双向 atomicSub：

```315:370:Primus-Turbo/csrc/kernels/deep_ep/utils.cuh
template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int **barrier_signal_ptrs, int rank) {
    ...
    if (blockIdx.x == 0 && thread_id < kNumRanks) {
        printf("[PT-K-BB rank=%d sm=0 tid=%d] PRE  atomicAdd/Sub ... peer_slot=%p ...\n", ...);
    }
    if (thread_id < kNumRanks) {
        atomicAdd_system (barrier_signal_ptrs[rank]       + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank,        FINISHED_SUM_TAG);
    }
    ...
    if (blockIdx.x == 0 && thread_id < kNumRanks) {
        printf("[PT-K-BB rank=%d sm=0 tid=%d] POST atomicAdd/Sub OK; entering spin loop\n", ...);
    }
    // spin until own slot drops to 0
    ...
    if (blockIdx.x == 0 && thread_id < kNumRanks) {
        printf("[PT-K-BB rank=%d sm=0 tid=%d] EXIT spin loop done\n", rank, thread_id);
    }
}
```

每个 NVL peer 都必须对每个 peer 的 slot 做一次 `atomicSub_system`，谁的 8 个 slot 都被减完，谁才能跳出 spin。日志里 14 个 rank 全部打出 `PRE … POST atomicAdd/Sub OK; entering spin loop`，但：

- **没有任何一个 rank 打出 `EXIT spin loop done`**
- **也没有任何一行 `notify_dispatch DEV stage=post-barrier1(nvl)`**

原因就是 nvl_rank=0 的 GPU（dev=0）从未进入 kernel、永远不会把它那一份 atomicSub 写出来。

### 2.3 死锁如何演变成进程崩溃

- `barrier_block` 内部有 `clock64()` timeout 自检；HSA 也会在 N 秒后强杀挂住的 kernel，host 通常会拿到 `hipError_t(9) "Memory access fault on (nil)"`（即 `repro_internode_deepep.py` 文档里那个被试图复现的现象）。
- 但本次跑里**先压顶的是上层的 JAX 心跳**：`[jax-preinit] ... heartbeat_timeout_s=100`，~100 s 后协调服务先撕掉所有 rank（`UNAVAILABLE: Failed to send RPC to coordination service`），14 个 rank 同时 SIGABRT，python 进程 exit 134，最后变成 srun task 1 退 134 → `Training FAILED (exit=143)`。

---

## 3. 结论一句话

**根因不是 dispatch kernel 本身，也不是 IPC handle 错乱，而是 `rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, ...)` 在两个节点的 GPU 0 上对称死锁。** 由于这是 `Primus-Turbo` deep_ep `sync_per_process_buffer()` 必经的同步点，nvl_rank=0 永远不会启动 dispatch kernel；剩下 14 个 rank 在 `internode.cu` 的 `barrier_block<NUM_MAX_NVL_PEERS=8>` 里 spin 到 JAX 心跳超时被强杀。

观察到的非根因（已排除）：

- 不是 IPC handle 不一致 —— 每个 dev=0 上 `gathered[i]_sha1` 全 `match=True`
- 不是 NVL buffer 分配失败 —— `Buffer ctor: nvl alloc OK rank=0/8 ...` 都成功
- 不是 IPC peer 映射失败 —— `SyncFromIPCHandles: opened peer i=0..7 (gbl=0..7) buffer_ptrs[i]=...` 全部正常
- 不是 root_uid gather 错位 —— `bootstrap: kv-gather ipc handles match[rank]=True` 自校验通过

为什么偏偏卡在 dev=0 —— 日志里没有 rocSHMEM 内部更细的诊断（`PRIMUS_TURBO_CHECK_ROCSHMEM` 只在返回后断言，对挂死无能为力），但几个明显的怀疑点：

1. **NIC 拓扑/可用性**：`ROCSHMEM_GDA_PROVIDER=mlx5` 下，dev=0 可能配对到了一张未拉起或没可用 GID 的 mlx5 HCA。同一节点上 dev=1..7 都成功，说明 mlx5_1..7 是 OK 的，疑似 mlx5_0（或 dev=0 拿到的那张）有问题。
2. **bootstrap socket 竞争**：`ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=eth0` 上 8 个独立 2-PE 世界并发握手，dev=0 这一对可能踩上了 port 冲突或 listen backlog 满。
3. **timeout 没起作用**：`ROCSHMEM_BOOTSTRAP_TIMEOUT=60` 并没看到 rocSHMEM 抛超时，最后是 JAX 100 s 心跳先到。要么这版 rocSHMEM 没 honor 这个变量，要么超时窗口被吃在某段不可中断的同步原语里。

---

## 4. 建议的下一步排查

1. 在 `Primus-Turbo/csrc/kernels/deep_ep/runtime.cu` 的 `init()` 里 `rocshmem_init_attr` 两侧再加一行 `PT_RS_LOG("about to rocshmem_init_attr ...")` + SIGALRM watchdog，确认是 init_attr 本身阻塞还是其内部 socket recv，把日志做实证。
2. 跑 `maxtext-slurm/utils/run_rocshmem_barrier_all_repro.sh`（`_train.sh` 已经接好 `REPRO_ROCSHMEM_BARRIER_ALL=1` 路径），用纯 C++ 在 dev=0 单独验一遍 8 对并发 2-PE init 是否能成功。
3. 显式 pin GPU → NIC 映射（例如 `ROCSHMEM_NIC_DEVICE` / `MLX5_DEVICE_*` 类的变量），看一旦 dev=0 用上某张可用 mlx5 卡，挂死是否消失。
4. 把 `_env_ROCSHMEM_BOOTSTRAP_TIMEOUT` 拉成 5–10 s 验证 timeout 路径是否能走通；若仍然不抛超时，就说明这个变量在当前 rocSHMEM 版本里没接好，需要改用外部 watchdog（例如在 `_bootstrap_per_process` 里用 `signal.setitimer` 强 abort）。
5. 在 `dump_rocshmem_state_via_ctypes()` 之前再加一段，对 `/sys/class/infiniband/mlx5_*/ports/1/state` 做 cross-rank KV 比较，确认 dev=0 那张 NIC 在 init 前的链路状态。

---

## 5. 相关行号速查（25191 log）

- 1189–1192：`[repro:0] primus_turbo.jax.initialize() OK`
- 1304–1305：`[PT-DBG:0] _bootstrap_per_process ENTER ... internode=True`
- 1412：`[PT-DBG:0] bootstrap: rocSHMEM root_uid acquired (rdma_rank=0 nvl_rank=0 root_global_rank=0)`
- 1419–1420：`[PT-DBG:0] create_per_process_buffer ... local_handle_sha1=b5b7680591b96ed4`
- 1495–1511：`[PT-DBG:0] sync_per_process_buffer ENTER ... match=True` + 全表 handle hash
- 1535：`[PT-DEP dev=0] SyncFromIPCHandles: NVL ptr tables uploaded ...`
- 1536：`[PT-DEP dev=0] SyncFromIPCHandles: pre rocshmem init rdma_rank=0 num_rdma_ranks=2 root_uid_bytes=128`
- 1537：`[PT-RS dev=0] init ENTER: rank=0 num_ranks=2 ...`
- **1538：`[PT-RS dev=0] init: set_attr_uniqueid_args OK rank=0 num_ranks=2`**  ← 最后一条 dev=0 的 host 日志（node 0）
- 1712–1713：dev=0 (node 1) 同上，到 `set_attr_uniqueid_args OK rank=1` 后断流
- 1871–1876：dev=3 完整走完 `rocshmem_init_attr OK` + `barrier_all OK` 作为对照
- 2317–2332：rank=3 在 device 端进入 `notify_dispatch DEV ENTER ... post-barrier1(rdma) pre-barrier1(nvl)`
- 2333–2348：rank=3 的 8 个线程 `[PT-K-BB] PRE/POST atomicAdd/Sub OK; entering spin loop`，再无 `EXIT spin loop done`
- 2653：`1: connection closed by remote peer useocpm2m-097-024.amd.com<50068>`（首次连接断）
- 2700–2710：JAX coordination service 集体 fatal
- 2711：`/artifact_.../_train.sh: line 185: 315 Aborted (core dumped) python3 -u ... mfu_tracker.py`
- 2959：`== Training FAILED (exit=143, 274s) ==`
