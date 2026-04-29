# ONE_GPU_PER_PROCESS 模式下 DeepEP JAX 集成问题排查报告

**日期**: 2026-04-27 ~ 2026-04-28  
**涉及仓库**: Primus-Turbo (JAX DeepEP bindings), MaxText (训练框架)  
**硬件**: AMD MI355X, 单节点 8 GPU, Slurm 调度

---

## 1. 背景

MaxText-Slurm 支持 `ONE_GPU_PER_PROCESS` 模式：每个 JAX 进程只拥有 1 个 GPU（共 8 个进程），
而非默认的 1 个进程拥有 8 个 GPU。在此模式下启用 DeepEP（Primus-Turbo 提供的高性能 MoE
dispatch/combine 内核）时，DeepEP 需要使用 **进程间 IPC 缓冲区** 替代进程内共享内存来完成
expert parallelism 通信。

本次调试的目标是让 `ds-proxy-se0-e256-h4096` 模型在 `ONE_GPU_PER_PROCESS` + DeepEP 模式下
成功训练。

---

## 2. 问题时间线与修复

### 2.1 问题 1: JAX 插件循环导入 (Circular Import)

**错误信息**:
```
Value error, use_deepep_dispatch requires the primus_turbo package with DeepEP JAX bindings.
```

**根因**: `primus_turbo/jax/__init__.py` 在模块顶层直接 `import jax` 和 `from jax.interpreters import mlir` 等。
当 JAX 的 `jax_plugins` 入口点机制尝试加载 `primus_turbo.jax` 时，JAX 本身尚未完成初始化，
导致循环导入。MaxText 的 `types.py` 中用 `import primus_turbo.jax.lax.moe` 做可用性检测，
该 import 也因此失败，被误报为 "primus_turbo 未安装"。

**验证**: 在容器内交互式 `python3 -c "import primus_turbo.jax.lax.moe"` 可成功，
说明仅在 JAX 插件发现的特定加载顺序下才触发。

**修复 (Primus-Turbo)**:
将 `primus_turbo/jax/__init__.py` 中所有重量级导入（`jax`, `mlir`, `_C.registrations`, `primitive`）
推迟到 `initialize()` 函数内部，并用 `_initialized` 标志确保幂等：

```python
_initialized = False

def initialize():
    global _initialized
    if _initialized:
        return
    _initialized = True

    import jax
    from jax.interpreters import mlir
    from primus_turbo.jax._C import registrations
    from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE
    # ... 注册 FFI targets、primitives、lowerings ...
```

**文件**: `primus_turbo/jax/__init__.py`

---

### 2.2 问题 2: DCN 并行度配置不匹配

**错误信息**:
```
Value error, DCN parallelism requested but only one slice available.
```

**根因**: 配置文件 `ds-proxy-se0-e256-h4096.gpu.yml` 原本设置 `dcn_fsdp_parallelism: 8`（对应 8 节点），
但测试只使用了 1 个节点。MaxText 的 `get_num_slices()` 返回 1，与 DCN 并行度不匹配。

**修复**: 提交命令中覆盖 DCN 参数：
```bash
./submit.sh ds-proxy-se0-e256-h4096 -N 1 ... -- \
    dcn_fsdp_parallelism=1 dcn_data_parallelism=1 ...
```

同时将配置文件中 `dcn_fsdp_parallelism` 改为 `1` 以匹配单节点场景。

**说明**: 参考 `deepseek3-671b` 的配置，其 `dcn_fsdp_parallelism: 8` 是因为实际使用了 8 个物理节点。

---

### 2.3 问题 3: JAX Tracing 中的 TracerArrayConversionError（核心问题）

**错误信息**:
```
jax.errors.TracerArrayConversionError: The numpy.ndarray conversion method
__array__() was called on traced array with shape int32[8,1]
```

**调用栈关键路径**:
```
maxtext_utils.get_abstract_state()
  → jax.eval_shape(init_state_partial)
    → layers/moe.py: moe_dispatch()
      → _moe_dispatch_impl()
        → deep_ep_runtime.ensure_deepep_runtime()
          → _bootstrap_per_process()
            → multihost_utils.process_allgather(jnp.array(handle_np))
              → np.asarray(all_handles_jax)  # ← Tracer 转 NumPy 失败
```

**根因**: MaxText 在训练开始前调用 `jax.eval_shape()` 来推导模型状态的形状（不执行实际计算）。
此时所有数组都是 JAX Tracer（抽象值）。然而 `_moe_dispatch_impl` 内部调用了
`ensure_deepep_runtime()`，后者在 `per_process` 模式下会执行 `_bootstrap_per_process()`——
这是一个需要真实数据的 IPC 操作（创建共享内存缓冲区、`process_allgather` 交换 handle），
Tracer 无法被转换为 `numpy.ndarray`，因此崩溃。

**修复（两层防御）**:

**(a) Tracer 守卫 (Primus-Turbo)**:

在 `_moe_dispatch_impl` 和 `_moe_combine_impl` 中添加 Tracer 类型检查，
tracing 阶段跳过 runtime bootstrap：

```python
# primus_turbo/jax/lax/moe/moe_dispatch_combine.py
def _moe_dispatch_impl(x, ...):
    config = get_dispatch_config() if config is None else config
    if not isinstance(x, jax.core.Tracer):
        deep_ep_runtime.ensure_deepep_runtime(hidden_bytes=_get_hidden_bytes(x), config=config)
    # ... 继续正常逻辑 ...
```

同样在 `_moe_combine_impl` 中添加了相同守卫。

**文件**: `primus_turbo/jax/lax/moe/moe_dispatch_combine.py` (L214, L502)

**(b) 提前 Bootstrap (MaxText + Primus-Turbo warmup API)**:

仅有 Tracer 守卫是不够的——如果 `jax.eval_shape` 时跳过了 bootstrap，那 IPC 缓冲区就不会被创建，
后续实际执行时 DeepEP 也无法工作。解决方案是在 `jax.eval_shape` **之前** 显式完成 bootstrap。

在 Primus-Turbo 侧新增 `warmup()` 公共 API：

```python
# primus_turbo/jax/lax/moe/moe_dispatch_combine.py
def warmup(hidden_bytes: int, *, config: Optional[Config] = None) -> None:
    """Eagerly bootstrap DeepEP runtime outside any JAX tracing context."""
    deep_ep_runtime.auto_detect_mode()
    config = get_dispatch_config() if config is None else config
    deep_ep_runtime.ensure_deepep_runtime(hidden_bytes=hidden_bytes, config=config)
```

在 MaxText 侧，`get_abstract_state()` 调用 `jax.eval_shape()` 之前先调用 warmup：

```python
# maxtext/src/MaxText/maxtext_utils.py
def _eagerly_bootstrap_deepep(config):
    try:
        from primus_turbo.jax.lax.moe import warmup as deepep_warmup
    except ImportError:
        return
    hidden_bytes = config.emb_dim * max(jnp.dtype(config.dtype).itemsize, 2)
    deepep_warmup(hidden_bytes)

def get_abstract_state(model, tx, config, rng, mesh, is_training=True):
    if config.use_deepep_dispatch:
        _eagerly_bootstrap_deepep(config)
    # ... jax.eval_shape() ...
```

**文件**:
- `primus_turbo/jax/lax/moe/moe_dispatch_combine.py` (新增 `warmup`)
- `primus_turbo/jax/lax/moe/__init__.py` (导出 `warmup`)
- `maxtext/src/MaxText/maxtext_utils.py` (新增 `_eagerly_bootstrap_deepep`)

---

### 2.4 问题 4: dtype AttributeError

**错误信息**:
```
AttributeError: 'numpy.dtype[bfloat16]' object has no attribute 'value'
```

**根因**: `_eagerly_bootstrap_deepep` 中最初使用了 `jnp.dtype(str(config.dtype.value))`，
但 MaxText 的 Pydantic 验证层已将 `config.dtype` 从枚举字符串转换为 `numpy.dtype` 对象，
`numpy.dtype` 没有 `.value` 属性。

**修复 (MaxText)**:
```python
# 修改前
hidden_bytes = config.emb_dim * max(jnp.dtype(str(config.dtype.value)).itemsize, 2)

# 修改后
hidden_bytes = config.emb_dim * max(jnp.dtype(config.dtype).itemsize, 2)
```

**文件**: `maxtext/src/MaxText/maxtext_utils.py`

---

### 2.5 问题 5: EP Ranks 断言失败

**错误信息**:
```
AssertionError: Unsupported number of EP ranks: 1
```

**根因**: `warmup()` 调用 `get_dispatch_config()` 时，后者通过 `deep_ep_runtime.get_ep_size()`
获取 EP 大小。但此时 `auto_detect_mode()` 尚未被调用，`PRIMUS_TURBO_JAX_DEEPEP_MODE` 环境变量
未设置，模式默认为 `INPROC`，于是 `ep_size = jax.local_device_count() = 1`（ONE_GPU_PER_PROCESS
模式下每个进程只有 1 个本地设备）。`config_map[1]` 不存在，触发断言。

**修复 (Primus-Turbo)**:

在 `warmup()` 函数最前面调用 `auto_detect_mode()`，确保在查询 EP size 之前就完成模式自动检测：

```python
def warmup(hidden_bytes, *, config=None):
    deep_ep_runtime.auto_detect_mode()      # ← 先自动检测模式
    config = get_dispatch_config() if ...    # ← 现在 ep_size 正确
    deep_ep_runtime.ensure_deepep_runtime(...)
```

同时在 `ensure_deepep_runtime()` 入口处也调用 `auto_detect_mode()`，作为兜底：

```python
# primus_turbo/jax/deep_ep/runtime.py
def auto_detect_mode() -> None:
    if os.environ.get(_MODE_ENV_VAR) is not None:
        return
    if jax.local_device_count() == 1 and jax.process_count() > 1:
        os.environ[_MODE_ENV_VAR] = "per_process"

def ensure_deepep_runtime(*, hidden_bytes=None, config=None):
    auto_detect_mode()     # ← 确保模式已设置
    mode = get_mode(lock=True)
    ...
```

**文件**:
- `primus_turbo/jax/deep_ep/runtime.py` (新增 `auto_detect_mode`)
- `primus_turbo/jax/lax/moe/moe_dispatch_combine.py` (`warmup` 中调用)

---

### 2.6 问题 6: RaggedDot 需要 Shardy 分区器

**错误信息**:
```
Check failed: hlo->parent()->parent()->config().use_shardy_partitioner()
RaggedDot is only supported with Shardy.
```

**根因**: DeepEP 稀疏 MoE 使用 `RaggedDot` HLO 操作，该操作仅在 JAX 的 Shardy 分区器下受支持。
配置文件中 `shardy: False` 导致使用了旧的 GSPMD 分区器。

**修复**: 提交命令中添加 `shardy=True`：
```bash
./submit.sh ... -- ... sparse_matmul=True use_deepep_dispatch=True shardy=True
```

---

## 3. 变更汇总

### 3.1 Primus-Turbo 侧 (未提交, 工作区变更)

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `primus_turbo/jax/__init__.py` | 重构 | 延迟导入到 `initialize()`，避免循环导入 |
| `primus_turbo/jax/deep_ep/runtime.py` | 新增 | `auto_detect_mode()`：自动检测 `per_process` 模式；`ensure_deepep_runtime()` 入口调用 |
| `primus_turbo/jax/lax/moe/moe_dispatch_combine.py` | 新增+修改 | `warmup()` 公共 API；`_moe_dispatch_impl` / `_moe_combine_impl` 中添加 Tracer 守卫 |
| `primus_turbo/jax/lax/moe/__init__.py` | 修改 | 导出 `warmup` |
| `tests/jax/lax/test_mp_dispatch_combine.py` | 新增 | `test_eval_shape_with_warmup` 和 `test_eval_shape_no_warmup` 两个测试用例 |

### 3.2 MaxText 侧 (已提交)

| Commit | 文件 | 说明 |
|--------|------|------|
| `5b6bbbf9` "add one gpu one process mode for intranode deepep" | `configs/types.py`, `layers/moe.py` | 放宽 EP 校验逻辑；MoE 层中调用 `auto_detect_mode()` 并记录日志 |
| `a263478d` "move deepep buffer alloc before abstract shape eval" | `maxtext_utils.py`, `layers/moe.py` | 新增 `_eagerly_bootstrap_deepep()`，在 `jax.eval_shape` 前调用 `warmup()`；简化 `moe.py` 中的冗余逻辑 |
| `c3b2b4c6` "fix dtype error" | `maxtext_utils.py` | 修复 `config.dtype.value` → `config.dtype` |

---

## 4. 架构总结

```
训练启动流程 (ONE_GPU_PER_PROCESS + DeepEP)
─────────────────────────────────────────────

 ┌─────────────────────────────────────────────────┐
 │ MaxText: get_abstract_state()                   │
 │                                                 │
 │  if use_deepep_dispatch:                        │
 │    _eagerly_bootstrap_deepep(config)            │
 │      └─→ warmup(hidden_bytes)                   │
 │            ├─ auto_detect_mode()                 │  ← 检测 1-GPU-per-proc → 设置 env var
 │            ├─ get_dispatch_config()              │  ← ep_size 现在正确 (8)
 │            └─ ensure_deepep_runtime()            │  ← IPC 缓冲区 + allgather 交换 handles
 │                                                 │
 │  jax.eval_shape(init_state_partial)             │  ← tracing: 只推导形状
 │    └─→ moe_dispatch(Tracer, ...)                │
 │          └─ _moe_dispatch_impl(x=Tracer)        │
 │               if not isinstance(x, Tracer):     │  ← 守卫: tracing 时跳过
 │                 ensure_deepep_runtime(...)       │
 │               # 继续执行 primitive.bind(...)     │
 │                                                 │
 │  jax.jit(train_step)(real_data)                 │  ← 实际执行: IPC 缓冲区已就绪
 │    └─→ moe_dispatch(real_array, ...)            │
 │          └─ _moe_dispatch_impl(x=real)          │
 │               isinstance check: False → 跳过     │  ← 缓冲区已由 warmup() 创建
 │               primitive.bind(...)               │  ← 直接使用 IPC 通信
 └─────────────────────────────────────────────────┘
```

---

## 5. 测试验证

### 5.1 单元测试 (Primus-Turbo)

新增两个多进程测试用例：

- **`test_eval_shape_with_warmup`**: 先调用 `warmup()`，再执行 `jax.eval_shape(model_fn, ...)`，
  验证不会触发 `TracerArrayConversionError`。
- **`test_eval_shape_no_warmup`**: 不调用 `warmup()`，直接 `jax.eval_shape()`，
  验证 Tracer 守卫能独立防止崩溃（shape 推导仍正确，只是 IPC 未初始化）。

### 5.2 端到端验证 (MaxText-Slurm)

- **Job 14176** (`ONE_GPU_PER_PROCESS=true, sparse_matmul=True, use_deepep_dispatch=True, per_device_batch_size=2`):
  成功训练 1970 步，TGS ≈ 17,033 tokens/s/device, MFU ≈ 6.96%（手动 scancel 终止）。

---

## 6. 关键经验

1. **JAX 插件加载顺序敏感**: `jax_plugins` 入口点在 JAX 完全初始化前触发，
   插件 `__init__.py` 中不能有顶层 `import jax`。

2. **`jax.eval_shape` 是纯 tracing**: 任何需要真实数据的操作（IPC、allgather、NumPy 转换）
   都不能在 `eval_shape` 路径中执行。自定义 primitive 的 impl 函数需考虑 Tracer 输入。

3. **模式自动检测需尽早**: `auto_detect_mode()` 必须在任何查询 `ep_size` 之前调用，
   否则 `jax.local_device_count() == 1` 会给出错误的 EP 大小。

4. **Pydantic 类型转换**: MaxText 的配置层会将 dtype 字符串预转换为 `numpy.dtype` 对象，
   不能假设 `config.dtype` 仍保持枚举类型。

5. **XLA 分区器兼容性**: `RaggedDot`（稀疏 MoE 核心 HLO）只支持 Shardy 分区器，
   使用 DeepEP + sparse_matmul 时必须 `shardy=True`。
