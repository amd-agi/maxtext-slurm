# InfiniBand P_Key 修复报告

**集群**: hungry-hippo-fin-03（NVIDIA B200 × 8 / node，ConnectX-7 × 8 400 Gbps native IB）
**软件**: NCCL 2.29.7 + CUDA 13.0，JAX + MaxText
**日期**: 2026-04-10

---

## 1. 问题现象

多节点训练（`-N 2`）在 NCCL 初始化阶段挂起，日志反复输出：

```
NCCL WARN NET/IB: Got completion from peer ... with status=IBV_WC_RETRY_EXC_ERR(12) vendor_err=129
```

所有 8 条 IB rail（`mlx5_0` ~ `mlx5_7`）均受影响。单节点训练正常。

## 2. 根因分析

### 2.1 排除干扰项

| 排查项 | 结论 |
|--------|------|
| `NCCL_IB_SL=1`（原配置） | 集群 SM 未配置 QoS，SL=1 映射到未配置的 VL → 数据包被丢弃。注释掉后消除此问题 |
| `NCCL_IB_HCA` 自动枚举 | 枚举所有 HCA 可能包含不可达设备。改为让 NCCL 自动探测 |

### 2.2 核心问题：P_Key limited membership

通过 sysfs 检查 P_Key 表：

```
/sys/class/infiniband/mlx5_0/ports/1/pkeys/0 → 0x7fff  (limited member)
/sys/class/infiniband/mlx5_0/ports/1/pkeys/1 → 0xf0dc  (full member)
```

IB 规范要求 RDMA 通信双方**至少一个是 full member**。两个 limited member 之间的通信会被硬件拒绝。该集群的 SM 将默认 P_Key（index 0）配置为 `0x7fff`（limited member），属于非标准配置（标准应为 `0xffff`）。

使用 `ib_write_bw` 验证：
- `--pkey_index=0`：连接失败
- `--pkey_index=1`：成功，带宽 392 Gbps

### 2.3 NCCL 层面的复杂性

| 尝试方案 | 结果 |
|----------|------|
| `NCCL_IB_PKEY=0xf0dc` | NCCL 2.29.7 的 DOCA 层在 native IB 上匹配 GID 失败，返回 GID index = -1，`ibv_modify_qp` 报错 EINVAL |
| `NCCL_IB_PKEY` + `NCCL_IB_GID_INDEX=0` | 无效，NCCL 内部覆盖了用户设置的 GID index |
| HPC-X RDMA SHARP 外部插件 + `NCCL_IB_PKEY` | 插件加载成功（`IBext_v11`），但 NCCL 的 `ibvwrap.c` 是共享代码层，GID = -1 的 bug 仍然存在 |
| LD_PRELOAD shim（v1，使用 `RTLD_NEXT`） | 进程 segfault。原因：NCCL 通过 `dlopen("libibverbs.so.1")` + `dlsym(handle, ...)` 获取 verbs 函数，`RTLD_NEXT` 在 libibverbs 被 RTLD_LOCAL 加载时返回 NULL |

## 3. 最终方案

### 3.1 思路

不设置 `NCCL_IB_PKEY`，让 NCCL 走默认代码路径（GID 选择正常）。通过 `LD_PRELOAD` 一个 C shim，在 `ibv_modify_qp` 层面将 `pkey_index=0` 透明替换为 `pkey_index=1`。

### 3.2 关键实现

**shim 的函数解析**（解决 v1 的 segfault）：

```c
static int (*resolve_real_fn(void))(...) {
    // 1. 先尝试 RTLD_NEXT（libibverbs 在全局作用域时有效）
    fn_t f = dlsym(RTLD_NEXT, "ibv_modify_qp");
    if (f && f != ibv_modify_qp) return f;

    // 2. 用 dlopen 显式打开 libibverbs（处理 RTLD_LOCAL 场景）
    void *h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libibverbs.so.1", RTLD_NOW);
    if (h) {
        f = dlsym(h, "ibv_modify_qp");
        if (f && f != ibv_modify_qp) return f;
    }
    return NULL;
}
```

**pkey_index 替换逻辑**：

```c
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
    if (target_pkey_index >= 0 &&
        (attr_mask & IBV_QP_PKEY_INDEX) &&
        attr->pkey_index == 0) {
        attr->pkey_index = target_pkey_index;
    }
    return real_fn(qp, attr, attr_mask);
}
```

**自动检测与加载**（`utils/ib_pkey_fix.sh`）：
1. 读取 `/sys/class/infiniband/<hca>/ports/1/pkeys/0`，检查是否为 limited member
2. 如果是，扫描 pkey 表找到第一个 full-member index
3. 在容器内编译 shim（`.so`），通过 `MAXTEXT_LD_PRELOAD` 加载
4. 如果默认 P_Key 已经是 full member，跳过（no-op）

### 3.3 文件变更

| 文件 | 改动 |
|------|------|
| `utils/ib_pkey_fix.sh` | 新增。自动检测 + 编译 + 加载 shim |
| `train_env.nvidia.sh` | 添加 `source utils/ib_pkey_fix.sh`；注释掉 `NCCL_IB_SL=1` |
| `utils/detect_nccl_env.sh` | 注释掉 `NCCL_IB_HCA` 盲目枚举 |

## 4. 验证

| Job ID | 配置 | 结果 |
|--------|------|------|
| 1114 | 原配置（`NCCL_IB_SL=1`） | `IBV_WC_RETRY_EXC_ERR(12)` |
| 1119 | `NCCL_IB_PKEY=0xf0dc` | `ibv_modify_qp` failed, GID index -1 |
| 1124 | LD_PRELOAD shim v1（`RTLD_NEXT`） | segfault（NULL 函数指针） |
| 1129 | HPC-X plugin + `NCCL_IB_PKEY` | 插件加载成功，但 ibvwrap GID 仍为 -1 |
| **1133** | **LD_PRELOAD shim v2（`dlopen`）** | **SUCCESS，exit 0，无 IB 错误** |

## 5. 适用范围与限制

- **适用**：默认 P_Key 为 limited member 的 native IB 集群 + NCCL 2.29.x（DOCA）
- **不影响**：默认 P_Key 为 full member（`0xffff`）的集群（shim 自动跳过）
- **不影响**：AMD / RoCE 集群（仅从 `train_env.nvidia.sh` 加载）
- **限制**：需要容器内有 `gcc` 和 `infiniband/verbs.h`（NVIDIA NGC 容器均包含）
