# `submit.sh` 完整调用链分析 — B200 集群适配参考

## 总体架构

```
submit.sh (登录节点)
  └─ sbatch → _job.sbatch (每节点)
       ├─ srun → preflight.sh    ← Stage 1: 清理
       ├─ srun → _container.sh pull-only ← Stage 2: 拉镜像
       ├─ srun → check_ecc.sh   ← Stage 3: ECC 检查
       └─ srun → _container.sh  ← Stage 4: 启动容器
            └─ docker run → _train.sh (容器内)
                 ├─ source train_env.sh  ← 环境变量
                 └─ python3 mfu_tracker.py ← 训练
```

---

## 阶段 0：`submit.sh`（登录节点，提交前）

### 0.1 参数解析

**调用链**: `submit.sh` → `utils/split_script_args.sh` → `utils/parse_job_args.sh` → `utils/parse_model_spec.sh` + `utils/resolve_model_name.sh`

| 步骤 | 文件 | 做了什么 |
|------|------|----------|
| 分割参数 | `split_script_args.sh` | 以 `--` 为界，左侧为 `SCRIPT_ARGS`（model + sbatch 参数），右侧为 `PASSTHROUGH_ARGS`（传给 MaxText 的参数） |
| 解析模型 | `parse_model_spec.sh` | 从 `SCRIPT_ARGS[0]` 解析 `model_name[:alias]:exp_tag` 格式，设置 `MODEL_NAME`, `MODEL_NAME_ALIAS`, `EXP_TAG`，剩余的为 `SBATCH_ARGS` |
| 解析模型名 | `resolve_model_name.sh` | 在 `configs/` 目录中精确匹配或模糊匹配 `*.gpu.yml` 文件，支持缩写（如 `llama` 匹配 `llama2-70b`） |
| 构建 JOB_NAME | `parse_job_args.sh` | 拼接为 `JAX-${MODEL_NAME}${EXP_TAG:+-$EXP_TAG}` |

**GPU 相关**: 无。纯字符串解析。

### 0.2 Slurm 预留检测

**调用链**: `submit.sh` → `utils/reservation.sh`

| 步骤 | 做了什么 |
|------|----------|
| `resolve_reservation` | 调 `scontrol show reservation`，找到当前 user 名下所有活跃预留，选最新的一个，加 `--reservation=` 选项 |

**GPU 相关**: 无。

### 0.3 构建 artifact（代码快照）

**调用链**: `submit.sh` → `utils/artifact.sh` → `utils/git_summary.sh`

| 步骤 | 做了什么 |
|------|----------|
| `build_artifact` | `rsync -a --exclude='.git/' --filter=':- .gitignore'` 把代码复制到 `$JOB_WORKSPACE/.artifacts/artifact_YYYYMMDD_HHMMSS_XXXX/`，隔离提交后的代码修改 |
| `git_summary.sh` | 记录 branch、last commit、diff、untracked files 到 `git_summary.txt` |

**GPU 相关**: 无。

### 0.4 提交 sbatch

```bash
sbatch "${RESERVATION_OPT[@]}" -J "$JOB_NAME" \
    --output="$JOB_WORKSPACE/%j-$JOB_NAME.log" \
    "${RAY_EXPORT[@]}" "${SBATCH_ARGS[@]}" \
    "$SCRIPT_DIR/_job.sbatch" \
    "$MODEL_NAME" "$EXP_TAG" "$MODEL_NAME_ALIAS" -- "${PASSTHROUGH_ARGS[@]}"
```

### 0.5 创建符号链接

提交成功后，创建 `$JOB_WORKSPACE/<JOB_ID>-<JOB_NAME>/` 目录，建立 `artifact` 和 `log` 符号链接。

---

## 阶段 1：`_job.sbatch`（Slurm 调度器分配节点后执行）

### 1.1 Slurm 资源声明

```bash
#SBATCH --exclusive             # exclusive node access
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # all mem avail
#SBATCH --ntasks-per-node=1     # n tasks per machine (one task per gpu)
#SBATCH --overcommit
```

**GPU 相关**: `--gpus-per-task=8` 假设每节点 8 GPU。B200 DGX 也是 8 GPU，**不用改**。

### 1.2 变量映射

将 Slurm 专用变量映射为通用名称：

| Slurm 变量 | 通用变量 | 用途 |
|-------------|----------|------|
| `SLURM_JOB_ID` | `JOB_ID` | 任务 ID |
| `SLURM_JOB_NUM_NODES` | `NNODES` | 节点数 |
| `SLURM_NODEID` | `NODE_RANK` | 节点编号（在 srun 中映射） |
| `SLURM_LAUNCH_NODE_IPADDR` | `JAX_COORDINATOR_IP` | 协调器 IP（在 srun 中映射） |

**GPU 相关**: 无。

### 1.3 端口分配

**调用**: `utils/pick_port.sh`

随机选一个 20000-52767 的空闲端口给 JAX coordinator（如启用 Ray 还会选第二个端口）。

**GPU 相关**: 无。

### 1.4 Stage Timeout 初始化

**调用**: `utils/stage_timeout.sh`

```bash
stage_timeout_init preflight:900 pull:900 ecc:300 train:none
```

注册 4 个阶段的超时：preflight 900s、pull 900s、ecc 300s、train 无限。

### 1.5 代码溯源

**调用**: `utils/code_provenance.sh`

打印 git summary（从 artifact 中的 `git_summary.txt`）到日志。

### 1.6 四个 Stage 的执行

`_job.sbatch` 按顺序执行以下 4 个 stage，任一 stage 失败或超时都会终止整个 job：

```bash
run_stage preflight "Preflight"  srun ... bash utils/preflight.sh
run_stage pull      "Docker pull" srun ... bash _container.sh pull-only
run_stage ecc       "ECC check"   srun ... bash utils/check_ecc.sh
run_stage train     "Training"    srun ... bash _container.sh <args>
```

每个 `srun` 使用 `NODE_REPORT` wrapper 在 per-task 级别映射 `NODE_RANK` 和 `JAX_COORDINATOR_IP`，并在失败时报告 `NODE_EXIT host=... exit=...`。

---

## 阶段 2：Stage `preflight`（每节点 srun）

**调用链**: `_job.sbatch` → `srun` → `utils/preflight.sh` → `utils/release_gpu.sh` + `utils/docker_utils.sh`

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| **GPU 清理** | `release_gpu.sh`：停止所有旧容器，用 `rocm-smi --showpids` 或 `nvidia-smi` 找到残留 GPU 进程，SIGTERM → SIGKILL → 等待释放 | **是**。AMD/NVIDIA 双路径已实现，**B200 兼容** |
| **磁盘清理** | 清理 `/tmp`、`/var/tmp` 旧文件；`docker container prune`、`docker image prune` | 否 |
| **NUMA 平衡** | 读取 `/proc/sys/kernel/numa_balancing`（默认不修改，注释了） | 否 |
| **信号量清理** | 删除 `/dev/shm/sem.*` 泄漏的信号量 | 否 |
| **THP 禁用** | 将 Transparent Huge Pages 设为 `never`（避免 latency spikes） | 否 |
| **IPv6 路由检查** | 多节点时检查 `ip -6 rule` 中 fd-prefix 规则数量（针对 Pensando AINIC） | **是**。`EXPECTED_IPV6_FD_RULES=8` 是 Pensando 特有的。B200+Mellanox 集群此检查不适用 |
| **DCQCN 检查** | `nicctl show dcqcn` 检查拥塞控制（Pensando 工具） | **是**。`nicctl` 是 Pensando 专有。B200+Mellanox 不需要 |

---

## 阶段 3：Stage `pull`（每节点 srun）

**调用链**: `_job.sbatch` → `srun` → `_container.sh pull-only`

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| Source `container_env.sh` | 读取 `DOCKER_IMAGE`（默认 `rocm/jax-training:maxtext-v26.2`）、`DOCKER_REGISTRY`、路径配置 | **是**。镜像名是 ROCm 的，B200 需换 CUDA 镜像 |
| 镜像检查 | `docker image inspect` 检查本地是否已有 | 否 |
| Smart prune | `docker_smart_prune` 确保至少 120GB 空闲磁盘 | 否 |
| Pull 镜像 | 先尝试匿名 pull，失败则用 `container_env.local.sh` 中的凭证登录 pull | 否 |
| `pull-only` 模式 | Pull 完成后直接 `exit 0`，不启动容器 | 否 |

---

## 阶段 4：Stage `ecc`（每节点 srun）

**调用链**: `_job.sbatch` → `srun` → `utils/check_ecc.sh`

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| 展开节点列表 | `scontrol show hostnames` 或手动解析 bracket 表示法 | 否 |
| ECC 检查 | `rocm-smi --showrasinfo` 解析每个 GPU 的 UMC CE/UE 错误计数 | **是**。完全依赖 `rocm-smi`，B200 上不可用，需要改为 `nvidia-smi` |

---

## 阶段 5：Stage `train`（每节点 srun → Docker 容器）

这是最复杂的阶段，包含多层嵌套。

### 5.1 `_container.sh`（宿主机，启动 Docker）

**调用链**: `_job.sbatch` → `srun` → `_container.sh`

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| Source `container_env.sh` | 获取 `DOCKER_IMAGE`、`MAXTEXT_REPO_DIR`、`DATASET_DIR`、`COREDUMP_EXTRA_DIRS` | **是**。镜像名 |
| 挂载 datasets | 如存在 `$DATASET_DIR`，挂载为 `-v $DATASET_DIR:/datasets:ro` | 否 |
| Coredump 目录 | 找一个有 >500GB 空间的目录挂载为 `/coredump` | 否 |
| **GPU 设备检测** | `/dev/kfd` → AMD (`--device=/dev/kfd --device=/dev/dri`)；`nvidia-smi` → NVIDIA (`--gpus all` 或 CDI) | **双路径已实现，B200 兼容** |
| **IB 挂载** | AINIC 镜像：无额外挂载；非 AINIC + `bnxt_re`：挂载 `/etc/libibverbs.d` | **是**。B200+Mellanox 不需要 bnxt_re。AINIC 默认值需调整 |
| InfiniBand 设备 | 如存在 `/dev/infiniband`，传入 `--device` | 通用，不用改 |
| **SETUP_CMDS** (容器内首先执行) | `ulimit` → `setup_coredump` → `pip install py-spy, google_cloud_mldiagnostics` → `cd MAXTEXT_REPO_DIR` → 可选 checkout `MAXTEXT_PATCH_BRANCH` | 否（pip 可能装不同包） |
| Docker run 参数 | `--privileged --network=host --ipc=host`，挂载 `/boot:ro`、脚本目录、outputs 目录 | 否 |
| 清理 trap | `_cleanup_container` → `release_gpu.sh --container NAME` | 否 |

### 5.2 `_train.sh`（容器内，训练入口）

**调用链**: `_container.sh` → `docker run` → `_train.sh`

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| 分割参数 | 同上，解析 `_env_KEY=VALUE` passthrough 参数 | 否 |
| 验证模型配置 | 检查 `configs/$MODEL_NAME.gpu.yml` 存在 | 否 |
| 解析 OUTPUT_PATH | `job_dir.sh` → `resolve_output_path`：checkpointing 用 model name，否则用 job dir | 否 |
| **Source `train_env.sh`** | **加载所有运行时环境变量** | **改动最大的文件，见下文** |
| 可选 source model.env.sh | 如 `configs/$MODEL_NAME.env.sh` 存在，加载模型专用环境覆盖 | 视模型而定 |
| 导出 extracted envs | `_env_KEY=VALUE` 参数作为环境变量导出 | 否 |
| DMABUF 安全检查 | 二次校验 `NCCL_DMABUF_ENABLE=1` 时 `/boot` 内核元数据可用性 | **是**。B200 上行为需验证 |
| LD_PRELOAD 处理 | 可选设置 `LD_PRELOAD`，否则清除 | 否 |
| **启动训练** | 非 Ray: `python3 -u mfu_tracker.py <config.yml> [args]`；Ray: `python3 -u _ray_actor.py [args]` | 否 |

### 5.3 `train_env.sh`（容器内，环境变量配置）— 改动核心

**调用链**: `_train.sh` → `source train_env.sh` → `source utils/detect_nccl_env.sh` → `source utils/detect_ainic_nccl_ib_tc.sh` + `source utils/choose_nccl_socket_ifname.sh`

按功能分组：

| 类别 | 变量 | B200 需要? |
|------|------|------------|
| **XLA Flags** | 大段注释掉（使用镜像默认），仅启用 XLA dump 和 `--xla_gpu_enable_command_buffer=''` | **需要验证** CUDA XLA 是否需要不同的 flags |
| **NCCL 通用** | `NCCL_CHECKS_DISABLE=1`, `NCCL_DEBUG=WARN` | **保留** |
| **内存** | `XLA_PYTHON_CLIENT_MEM_FRACTION=.93`, `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=512` | **保留**，但数值可能需调 |
| **多网卡优化** | `NCCL_NCHANNELS_PER_NET_PEER=4`, `NCCL_NSOCKS_PERTHREAD=4`, `NCCL_SOCKET_NTHREADS=8` | **保留**，但值需针对 Mellanox 调优 |
| **IB 调优** | `NCCL_IB_QPS_PER_CONNECTION=4` | **保留**，数值可能不同 |
| **NCCL 网络自动检测** | `detect_nccl_env.sh` → `NCCL_IB_HCA`（列举 IB 设备）+ Pensando AINIC TC 检测 + `NCCL_SOCKET_IFNAME` | **部分保留**。IB_HCA 和 SOCKET_IFNAME 通用，Pensando 检测不触发 |
| **GPU 计算** | `CUDA_DEVICE_MAX_CONNECTIONS=1`, `GPU_MAX_HW_QUEUES=2` | **保留** CUDA 的，删除 GPU_MAX_HW_QUEUES（AMD） |
| **AMD HIP/HSA** | `HIP_FORCE_DEV_KERNARG=1`, `HSA_*` (4 个) | **删除** |
| **Transformer Engine (ROCm)** | `NVTE_USE_ROCM=1`, `NVTE_FUSED_ATTN_CK=1`, `NVTE_USE_HIPBLASLT=1`, `NVTE_CK_*` | **删除或替换**为 CUDA TE 配置 |
| **Composable Kernel** | `CK_TILE_*`, `NVTE_CK_*` | **删除**（AMD 专有） |
| **Ionic/Pensando** | `IONIC_LOCKFREE=all` | **删除** |
| **DMABUF** | `NCCL_DMABUF_ENABLE=1` + `/boot` kernel metadata 检查 | **保留**但需验证 |
| **GDR/IB 高级** | `NCCL_GDRCOPY_ENABLE=1`, `NCCL_GDR_FLUSH_DISABLE=1`, `NCCL_IB_ECE_ENABLE=0`, `NCCL_IB_GID_INDEX=1`, `NCCL_IB_PCI_RELAXED_ORDERING=1` 等 | **保留**但数值需按 Mellanox 调优 |
| **RCCL** | `RCCL_GDR_FLUSH_*`, `RCCL_LL128_*`, `RCCL_MSCCLPP_*` | **删除** |

**`detect_nccl_env.sh` 子调用链：**

| 步骤 | 做了什么 | B200 适配 |
|------|----------|-----------|
| `NCCL_IB_HCA` | 列举 `/sys/class/infiniband` 设备 | **通用**，Mellanox HCA 也在这里 |
| `detect_ainic_nccl_ib_tc.sh` | 检查是否 Pensando AINIC，读取 `nicctl show qos` 获取 DSCP→优先级映射 | **不触发**（`is_pensando` 返回 false） |
| `choose_nccl_socket_ifname.sh` | 按优先级选择 10.x > 172.16.x > 192.168.x 的网卡 | **通用**，不用改 |

### 5.4 `mfu_tracker.py`（容器内，训练入口包装器）

| 步骤 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| GPU 检测 | `rocminfo` → `amd-smi` → `nvidia-smi` 三级 fallback | **已兼容**，B200 走 nvidia-smi 路径 |
| Peak TFLOPS 查表 | B200: bf16=2250, fp8=4500 | **已包含 B200** |
| 包装 stdout | 在每行 `TFLOP/s/device:` 后追加 `MFU: X.XX%` | 通用 |
| 启动训练 | `from MaxText import train; train.main(...)` | 通用 |

### 5.5 可观测性（Ray 模式下的附加组件）

如果启用了 `RAY=1`（通过 `_train_with_ray.sh` 路径）：

**调用链**: `ray_cluster.sh` → `metrics_exporter.sh` → `gpu_metrics_plugin.sh` + `host_metrics_plugin.sh` + `tb_metrics_plugin.sh`

| 组件 | 做了什么 | GPU 相关? |
|------|----------|-----------|
| **Ray cluster** | Head 节点启动 Ray head + Prometheus + TensorBoard；Worker 节点加入 Ray 集群 | 否 |
| **Metrics exporter** | 每 10s 采集所有 `*_metrics_plugin.sh`，通过 HTTP 暴露 Prometheus metrics | 否 |
| **gpu_metrics_plugin.sh** | 通过 AMD sysfs (`/sys/class/hwmon/amdgpu`) 采集温度、功耗、时钟、VRAM、RAS 错误、PCIe AER | **完全 AMD 专有**，B200 需全部重写 |
| **host_metrics_plugin.sh** | 网络/TCP/RDMA/调度/OOM/存储/dmesg 指标 | **大部分通用**，dmesg 关键词 (`amdgpu/xgmi`) 和 GPU 进程计数 (`/sys/class/kfd`) 需适配 |
| **tb_metrics_plugin.sh** | 从 TensorBoard events 提取 loss/TGS 等 | 通用 |

---

## 完整文件依赖图

```
submit.sh
├── utils/split_script_args.sh
├── utils/parse_job_args.sh
│   ├── utils/split_script_args.sh
│   ├── utils/parse_model_spec.sh
│   └── utils/resolve_model_name.sh
├── utils/reservation.sh
├── utils/artifact.sh
│   └── utils/git_summary.sh
├── utils/ray_cluster.sh (if RAY=1)
│   ├── utils/job_dir.sh
│   ├── utils/prometheus.sh
│   ├── utils/detect_ip.sh
│   └── utils/metrics_exporter.sh
│       ├── utils/gpu_metrics_plugin.sh      ★ AMD only
│       ├── utils/host_metrics_plugin.sh     ★ mostly generic
│       └── utils/tb_metrics_plugin.sh
├── utils/job_dir.sh
│
└── _job.sbatch
    ├── utils/split_script_args.sh
    ├── utils/pick_port.sh
    ├── utils/stage_timeout.sh
    ├── utils/code_provenance.sh
    │   └── utils/git_summary.sh
    │
    ├── [Stage: preflight] utils/preflight.sh
    │   ├── utils/release_gpu.sh             ✓ dual AMD/NVIDIA
    │   └── utils/docker_utils.sh
    │
    ├── [Stage: pull] _container.sh (pull-only)
    │   └── container_env.sh                 ★ ROCm image default
    │
    ├── [Stage: ecc] utils/check_ecc.sh      ★ AMD only (rocm-smi)
    │
    └── [Stage: train] _container.sh
        ├── container_env.sh                 ★ ROCm image default
        ├── utils/detect_ip.sh
        ├── utils/job_dir.sh
        ├── utils/docker_utils.sh
        ├── utils/split_script_args.sh
        ├── utils/code_provenance.sh
        ├── utils/coredump.sh
        │
        └── _train.sh (inside Docker)
            ├── utils/split_script_args.sh
            ├── utils/job_dir.sh
            ├── train_env.sh                 ★★★ 改动最大
            │   └── utils/detect_nccl_env.sh
            │       ├── utils/detect_ainic_nccl_ib_tc.sh  (safe: no-op on non-Pensando)
            │       └── utils/choose_nccl_socket_ifname.sh ✓ generic
            ├── configs/$MODEL.env.sh (optional)
            └── utils/mfu_tracker.py         ✓ B200 already in table
```

图例：`★` = 需要改动，`✓` = 已兼容，无标记 = 通用不涉及 GPU

---

## B200 适配改动清单（按调用链顺序）

| # | 文件 | 改动类型 | 具体内容 |
|---|------|----------|----------|
| 1 | `container_env.sh` | 改默认值 | `DOCKER_IMAGE` 换成 CUDA JAX 镜像；`DOCKER_IMAGE_HAS_AINIC` 改为 `false` |
| 2 | `utils/preflight.sh` | 条件化 | IPv6 rules / DCQCN 检查加守卫（仅 Pensando 环境执行） |
| 3 | `utils/check_ecc.sh` | 新增 NVIDIA 路径 | 检测到 `nvidia-smi` 时用 `--query-gpu=ecc.*` 查 ECC 错误 |
| 4 | `_container.sh` | 小改 | IB 挂载逻辑中的 `bnxt_re` 分支在 Mellanox 环境下跳过即可（目前逻辑已可以） |
| 5 | **`train_env.sh`** | **大改** | 删除所有 AMD 专有变量（HIP/HSA/CK/RCCL/IONIC），NVTE 从 ROCm 模式改为 CUDA 模式，NCCL 参数按 NVLink/NVSwitch/Mellanox 重新调优 |
| 6 | `utils/gpu_metrics_plugin.sh` | **重写** | 从 AMD sysfs 改为 `nvidia-smi`/NVML 采集 |
| 7 | `utils/host_metrics_plugin.sh` | 小改 | dmesg 关键词加 `nvidia/nvrm/nvlink/xid`；GPU 进程计数从 `/sys/class/kfd` 改为 NVIDIA 方式 |
| 8 | `configs/*.gpu.yml` | 调优 | `per_device_batch_size`、并行策略等按 B200 显存/带宽重新 benchmark |

### 建议实现方案

与其在每个文件里加 if/else 分支，更好的做法是让 `train_env.sh` 做 GPU vendor 检测（通过 `/dev/kfd` 或 `nvidia-smi`），然后分别 source 一个 `train_env.amd.sh` 或 `train_env.nvidia.sh`。类似的模式可以用在 `check_ecc.sh` 和 `gpu_metrics_plugin.sh` 上。
