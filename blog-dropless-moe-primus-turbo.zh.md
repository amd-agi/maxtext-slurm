<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# 在 JAX 中使用 Primus-Turbo 进行无丢弃 MoE 训练

混合专家(MoE)模型已成为在不付出全部算力代价的前提下扩展 Transformer 参数量的一种
标准做法 —— 但要在 GPU 上高效训练它们,却逼人做一个别扭的取舍。JAX/MaxText 的默认
路径让每个专家的张量保持固定形状,直接*丢弃(drop)*那些超出各专家容量的 token,以
模型质量换速度。而完全*无丢弃(dropless)*的替代方案保留每一个 token,但在纯 JAX 中
会撞上一堵内存墙,使其在生产规模下不切实际。

AMD 的 [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) 填上了这道缝。它把两个
基于 Composable Kernel(CK)的 primitive 带到 JAX —— 用 **grouped GEMM** 处理不规则、
变长的专家矩阵乘,用 **DeepEP dispatch/combine** all-to-all 处理感知 token 的专家并行
路由 —— 并通过 XLA FFI 把它们暴露为一等的 JAX op,同时不牺牲自动微分、分片(sharding)
与数值保真度。

在本文中,你将了解我们如何用两个简单的开关,把这些 kernel 接入
[MaxText](https://github.com/AI-Hypercomputer/maxtext) 中的大型 MoE 训练图。你会看到
grouped GEMM 与 DeepEP 如何工作、如何通过 JAX 的 FFI 集成一个自定义 kernel(含
custom VJP、分片契约,以及每进程一次的引导),以及无丢弃路径在吞吐与收敛两方面相对
容量因子默认值的表现如何。读完之后,你将看到 Primus-Turbo 如何把 AMD Instinct GPU 上
的无丢弃 MoE 训练从"不可行"变成一个可用、更快、更省显存的默认选择。

---

## 混合专家(MoE)速览

[混合专家(MoE)](https://arxiv.org/abs/1701.06538) 层把 Transformer block 中单个
前馈网络(FFN)替换为**多个** FFN("专家",experts),并配一个小的**路由器
(router)**,只把每个 token 发给其中少数几个。它在不付出全部 FLOP 代价的前提下换来
参数量 —— 进而换来容量,因为每个 token 仍然只经过它的 top-`k` 个专家。

一个有代表性的现代大型 MoE 大致是这样:

- 几十个 decoder 层,hidden size 数千,词表很大;
- 每个 MoE 层有**数百个被路由的专家**(例如 256),**top-k = 8**(外加共享专家);
- 采用带可学习 per-expert bias 的 sigmoid 路由器做负载均衡。

下图追踪了单个 token 如何流经这样一层 —— 被路由到其 top-k 个(共 N 个)专家、计算,
再加权求和合并:

![MoE 层数据流:token 经 gating 网络路由到其 top-k 个(共 N 个)专家,计算后再加权求和合并。](images/diagram-moe-flow.png)

MoE 难的不是专家本身 —— 它们就是普通的 GEMM —— 而是**路由**。每个 token 的 top-k
选择依赖数据、每步都在变,因此每个专家拿到的 token 数是*不规则(ragged)*且*动态*的。
由此引出两点:

1. 每个专家的矩阵乘是一个**分组(grouped)GEMM**(一批 `M` 维不同的 GEMM),而不是单个
   稠密 GEMM。
2. 在**专家并行(EP)**下(专家分布在不同 GPU 上),token 必须被物理地搬运到拥有其专家
   的 GPU、算完再搬回 —— 这是一个消息大小本身就依赖数据的 **all-to-all**。

框架如何处理这两点,正是下文各实现的分水岭。

---

## `dense_matmul`:容量因子丢弃

让 MoE 形状保持静态最简单的办法,就是**拒绝不规则**。`dense_matmul` 路径(MaxText 的
常见默认)选定一个*容量(capacity)* —— 这就是来自 [GShard](https://arxiv.org/abs/2006.16668)
与 [Switch Transformers](https://arxiv.org/abs/2101.03961) 的容量因子思路:

```python
expert_capacity = capacity_factor * (num_tokens * top_k / num_experts)
```

每个专家分到一个恰好 `expert_capacity` 个槽位的固定形状缓冲区。token 被 scatter 进一个
`[num_experts, expert_capacity, hidden]` 张量;若某专家在该步被超额订阅,溢出的 token 被
**丢弃**(贡献置零),不足则做**填充(padding)**。专家 FFN 于是就是在这个规整张量上的
单个**批量 GEMM(batched GEMM)**。

权衡如你所料:

- **优点:** 完全静态、*规整*的形状。专家矩阵乘是普通 batched GEMM —— **比 grouped GEMM
  开销更低**(各组等长,tile 干净,没有 per-group 偏移记账或不规则边界)—— 因此它本身是
  所有路径里最高效的矩阵乘,不需要自定义 kernel,内存也最省(dispatch 张量固定为
  `num_experts * expert_capacity * hidden`,与路由倾斜无关)。
- **缺点:** 它是**有损的**。`capacity_factor=1.25` 每步丢弃一小部分被路由的 token;调到
  2.0 或 4.0 能找回保真度,但填充缓冲区会把 batched GEMM 的 FLOP 花在 padding 上、并抬高
  内存。你是在用数值正确性换形状上的方便 —— 而不是矩阵乘速度,后者恰恰是稠密路径的强项。

`dense_matmul` 是合适的基线,但"丢 token 来保持形状方正"并不是忠实 MoE 训练所要的。

---

## `sparse_matmul`:无丢弃路由

`sparse_matmul` 路径是**无丢弃(dropless)**的:每个被路由的 token 都到达其专家,没有容量、
没有 padding-to-capacity、没有被丢弃的 token。做法是按专家对 token 排序,再在变长的分组上
做*不规则(ragged)*GEMM。

在 MaxText 的内置实现里有三步:

- **permute(置换):** 对展平的 `[tokens * top_k]` 专家分配做 `argsort`,把激活 gather 成
  按专家连续的顺序,并计算 `group_sizes[e]` = 专家 `e` 的 token 数;
- **专家矩阵乘:** 在这些分组上做 ragged GEMM,每个专家一个逻辑矩阵乘。MaxText 内置的
  `jax.lax.ragged_dot` 是自然之选,但其内存占用在规模上高得离谱:纯 `ragged_dot` 即便在
  per-device batch size(pdbs)为 1 时也会 **OOM 到约 444 GiB** —— 根本训不起来,这是要换用
  专门 grouped-GEMM kernel 的第一个理由(见 [Grouped GEMM](#grouped-gemm-gmm));
- **unpermute + combine:** 把专家输出 scatter 回 token 顺序,并按权重对 top-k 贡献求和。

在专家并行下,跨 GPU 的搬运用 **`ragged_all_to_all`** 完成。这正是无丢弃的痛点。由于路由是
动态的、没有容量上限,每个 rank 收到多少 token 是一个运行时才知道的值 —— 但 **`jax.jit`
追踪的是固定形状的图,所以接收缓冲区必须是静态形状**,在任何 token 被路由之前的 trace 阶段
就要定下来。唯一能保证成立的静态形状是**最坏情况**(`num_ranks * tokens * hidden`,因为任何
rank *都可能* 把全部 token 发给同一个 peer),于是 JAX 实际上被**逼着做悲观分配**。(若想按
真实运行时数量来分配,就得把这个数读回 host、跳出 `jit` —— 也就是我们在 [*无同步(sync-free)*一节](#悲观分配的隐藏收益无同步sync-free)再谈的
device-to-host 同步。)这笔固定的内存税随 per-device batch size 增长,因此**它逼你用比稠密
路径更小的 batch**:在生产规模下(数千亿参数的大型 MoE),仅 `ragged_all_to_all` 缓冲区就能
把无丢弃路径在 pdbs=8 推到 **OOM 约 242 GiB**,而稠密路径能跑到 pdbs=16。即便配上高效的
grouped-GEMM 专家矩阵乘,这个 all-to-all 仍是那堵墙 —— 无丢弃的正确性是有了,但要以更小的
batch 为代价。

下图把两种布局并排对比 —— 左边是容量因子丢弃,右边是无丢弃的不规则路径:

![丢弃 vs 无丢弃:容量因子丢弃(左)把每个专家填到固定形状、丢掉溢出 token,以便做单个 batched GEMM;无丢弃路径(右)把每个 token 都保留在变长的 per-expert 分组中,做 grouped GEMM。](images/diagram-dropping-vs-dropless.png)

于是无丢弃训练撞上**两堵内存墙** —— `ragged_dot` 专家矩阵乘(上面的约 444 GiB)和
`ragged_all_to_all` 搬运。接下来两节分别击破它们:用一个 **grouped-GEMM kernel** 拆掉矩阵乘
那堵墙([Grouped GEMM](#grouped-gemm-gmm)),用 **DeepEP** —— 一个更省的路由 all-to-all ——
把搬运那堵墙削薄到能挪回一档 batch([DeepEP](#deepep))。

---

## Grouped GEMM (GMM)

**grouped GEMM(分组 GEMM)** 计算一批彼此独立、共享相同 `K` 与 `N` 但每组 `M` **不同** 的
矩阵乘:

```python
for e in range(num_experts):
    out[off[e] : off[e+1]] = a[off[e] : off[e+1]] @ b[e]   # [m_e, k] @ [k, n]
```

其中 `off = cumsum(group_lens)`。整件事是在一个连续的 `[sum(m_e), k]` 激活张量与一个
`[num_experts, k, n]` 权重张量上的单次 kernel 启动 —— 正是 token 按专家排序后,无丢弃专家
FFN 所需的操作。没有 padding 到容量、没有被丢弃的 token;kernel 沿着组偏移走,为每个专家做
恰当大小的矩阵乘。

反向需要两个 grouped GEMM:

- `grad_a = grad_c @ bᵀ` —— 又一个 grouped GEMM(同样的分组布局);
- `grad_b = aᵀ @ grad_c` —— 一个**变长 `K`(variable-`K`)** 的 grouped GEMM(这里不规则的
  是收缩维)。

grouped GEMM 比稠密路径的规整 batched GEMM([容量因子丢弃](#dense_matmul容量因子丢弃))开销更大 —— 每组 `M` 不同意味着 per-group
偏移与不规则的 tile 边界 —— 因此相对稠密它并非"免费"。但它正是让你能做到无丢弃的那个矩阵
乘;一个调优良好的 grouped-GEMM kernel 是快速无丢弃 MoE 训练里最重要的单个 primitive(这一
思路由 [MegaBlocks](https://arxiv.org/abs/2211.15841) 在 GPU 上开创)。

---

## DeepEP

矩阵乘那堵墙已由 grouped GEMM 处理,剩下的就是路由 all-to-all。
**[DeepEP](https://github.com/deepseek-ai/DeepEP)** 是一个专家并行通信库:一对 **dispatch**
与 **combine** kernel,以**感知 token(token-aware)**的方式实现 MoE all-to-all。

- **dispatch** 把每个 rank 的本地 token 发给拥有其所选专家(`topk_idx`)的 rank,intranode 走
  NVLink/xGMI、internode 走 RDMA。它的接收缓冲区仍按最坏情况(`num_tokens * ep_size`)分配
  —— DeepEP 并没有逃过[无丢弃路由](#sparse_matmul无丢弃路由)的悲观分配 —— 但它对该缓冲区的管理更精简(分块 send/recv,比通用
  `ragged_all_to_all` 少很多中间拷贝),所以**瞬时显存占用要小一些**。
- **combine** 是其精确逆操作:把每个专家的输出送回贡献过这些 token 的各 rank,并在目的地做
  规约(求和)。

下图展示了这条跨 GPU 的 dispatch → 专家计算 → combine 往返路径:

![DeepEP dispatch/combine:每个 GPU 把本地 token 发给拥有其所选专家的 rank(dispatch all-to-all),专家在本地计算,再用一次反向 all-to-all 把输出规约回各源 rank(combine)。](images/diagram-deepep.png)

dispatch 返回一个不透明的**句柄(handle)**,描述通信布局(rank/channel 前缀矩阵、源索引、
send head)。该句柄会回传给 combine 以精确还原 dispatch,也让反向复用布局而不必重算。概念上:
**dispatch 的反向是一次 combine,combine 的反向是一次 dispatch。**

DeepEP 给了你一个无丢弃的路由 all-to-all,把[无丢弃路由](#sparse_matmul无丢弃路由)悲观分配的瞬时占用削薄 —— 幅度不大,但足以
挪回一档 batch。

### 悲观分配的隐藏收益:无同步(sync-free)

最坏情况接收缓冲区还有第二个、更微妙的好处。另一种做法 —— *恰好* 按每个 rank 实际收到的
token 数分配 —— 意味着缓冲区形状取决于一个只有在路由之后才存在于 GPU 上的值,因此你必须把
这些 per-expert 计数**读回 host** 才能确定形状。这种 **device-to-host(D2H)同步** 会拖空 CPU
的启动队列、阻断计算/通信重叠。AMD 的 PyTorch/Megatron 栈把消除这些同步当作头等优化 ——
即 [*Feature 3: Sync-Free MoE*](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html#feature-3-sync-free-moe),
靠 `num_worst_token` 开关并把 token 计数保持为 GPU 张量来实现 —— 并明确指出其代价:完全
sync-free 的等级"会显著增加 GPU 显存占用"。

最坏情况形状是**静态**的,所以没有任何计数需要往返 host —— 这正是悲观分配之所以 sync-free
的原因。在 JAX 里这甚至不是一个开关:`jax.jit` 本就要求静态形状,因此 DeepEP primitive 直接
用 `num_worst_tokens = num_tokens * ep_size`,整个 MoE 前向**默认就是 sync-free** 的。(数据
依赖的形状会逼出一次 host callback —— 也正是 PyTorch 栈费力消除的那种 D2H 停顿。)所以[无丢弃路由](#sparse_matmul无丢弃路由)的
内存税与无停顿的启动流,是同一个决定的一体两面。

---

## Primus-Turbo

我们现在有了无丢弃 MoE 所需的两个 primitive;在 AMD GPU 上它们就装在同一个库里。

[Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) 是 AMD 面向 ROCm 栈(MI300/MI350 级
GPU,gfx94x/gfx95x)的高性能训练 primitive 库。它打包了基于
[Composable Kernel(CK)](https://github.com/ROCm/composable_kernel) 的 GEMM、normalization、
量化、FP8,以及 —— 我们这里关心的两个 —— 一个 **grouped GEMM** 和一个 **DeepEP
dispatch/combine** 实现,同时提供 **PyTorch** 与 **JAX** 两套前端。

对 MaxText 而言有意思的是 JAX 前端,因为把一个手写的 HIP/CK kernel 暴露给一个被 `jax.jit`
追踪、被 XLA 编译、要做自动微分、还被 `shard_map` 分片的程序,绝不只是"调用 kernel"那么
简单 —— 它必须成为一等的 JAX primitive。这正是 Primus-Turbo 的 `primus_turbo.jax` 包所做的。

---

## Primus-Turbo 如何为 JAX 实现 GMM 与 DeepEP

### 基于 FFI 的自定义 primitive

每个 kernel 都被暴露为一个 JAX **primitive**,其 lowering 通过
**[XLA FFI](https://docs.jax.dev/en/latest/ffi.html)**(foreign-function interface,外部函数
接口)调用 C++/CK kernel。在裸 primitive 之上,Primus-Turbo 注册了让 JAX 把它们当作原生 op
所需的一切:

- **abstract evaluation**(形状/dtype 规则),使 trace 能进行;
- **`custom_vjp`**,使自动微分可行,而不必让 XLA 去(徒劳地)对一个不透明调用求导;
- **sharding 规则**,使该 op 在 `shard_map`(FSDP)下正确分片。

**不用 FFI 还有别的办法吗?** 有三种,但没有一种适合一个要调用既有 CK/HIP kernel 和基于
[rocSHMEM](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/) 的 all-to-all 的逐层训练热路径:

1. **停留在纯 JAX/XLA primitive** —— 用 `jax.lax.ragged_dot` 与 `jax.lax.ragged_all_to_all`
   表达。这*不需要*自定义 kernel,但纯 `ragged_dot` 专家矩阵乘在规模上内存不可行(训不起来),
   而通用 `ragged_all_to_all` 会撞上[无丢弃路由](#sparse_matmul无丢弃路由)那堵内存墙 —— 这正是这里的无丢弃配置改用 grouped-GEMM
   kernel 与 DeepEP 的原因。DeepEP 的 IPC/RDMA dispatch 没有纯 XLA 的等价物。
2. **用 [Pallas/Mosaic](https://docs.jax.dev/en/latest/pallas/index.html) 写 kernel** —— JAX 自带的 kernel DSL,由 XLA 编译、无需外部 C++。对
   grouped GEMM 可行,但你得*重新实现* CK kernel(并放弃它的调优),而且 Pallas 没有通往
   DeepEP 所需的跨 rank rocSHMEM/IPC primitive 的路径 —— 通信这一半根本走不通。
3. **Host callback**(`jax.pure_callback` / `io_callback`)—— 把张量丢回 Python/C++ 回调。对一个
   每个 MoE 层都要调用的 op 来说,host 往返慢了好几个数量级,而且与 `jit`、donation、自动微分、
   `shard_map` 都难以良好组合。

FFI 恰恰是对的工具:它让**既有的**、调优过的 CK grouped GEMM 与 DeepEP 通信作为 XLA
custom-call **在图内**运行 —— 没有 host 往返、不用重写 kernel —— 而 `custom_vjp` 给了 JAX
围绕这个不透明调用做求导与分片所需的一切。

### Grouped GEMM

公开 API 很小:

```python
from primus_turbo.jax.lax.grouped_gemm import grouped_gemm, compute_group_offs

out = grouped_gemm(a, b, group_lens, transA=False, transB=False, num_cu=-1)
```

底层 `grouped_gemm` 是一个 `jax.custom_vjp`。前向 bind CK grouped-GEMM primitive
(`ck_grouped_gemm_p`);反向发起 [Grouped GEMM](#grouped-gemm-gmm) 里的两个 grouped GEMM —— `grad_a` 用标准版,`grad_b` 用
**变长 K(variable-K)**变体(`ck_grouped_gemm_variable_k_p`)。`compute_group_offs` 把
`group_lens` 转成 kernel 要遍历的 `[num_experts+1]` 偏移数组。CK kernel 要求 **int64** 的
group 长度,这一点后面会成为一个小但重要的集成细节(见[保证正确性的关键细节](#保证正确性的关键细节))。

### DeepEP Dispatch/Combine

公开 API 与 [DeepEP](#deepep) 的概念对应:

```python
from primus_turbo.jax.lax.moe import setup, moe_dispatch, moe_combine

setup(mesh=mesh, ep_axis_name="expert", hidden_bytes=emb_dim * 2)   # 每进程一次
recv_x, recv_idx, recv_w, handle = moe_dispatch(x, topk_idx, topk_w, num_experts)
out = moe_combine(expert_out, handle)
```

JAX 实现里有两个设计点值得一提:

**一次性的 `setup()`,把运行时冻结。** DeepEP 有三种运行时模式 —— 单进程多 GPU(INPROC)、
每进程一 GPU 的 intranode(PER_PROCESS,≤8 rank,走 NVLink/xGMI)、以及每进程一 GPU 的
internode(PER_PROCESS over RDMA,>8 rank)。`setup()` 从 `jax.process_count()` /
`jax.local_device_count()` 自动判别模式,从 JAX `Mesh` 的 `"expert"` 轴固定专家并行通信组,
按 `hidden_bytes` 给 per-process NVL/RDMA 缓冲区定大小,发出 internode rocSHMEM 握手所需的跨
host barrier,然后把这一切(模式、EP 大小、SM 数)**冻结**成一个不可变快照。冻结之后,每次
`moe_dispatch`/`moe_combine` 都读快照,而不必在每次 trace 重新查询可变全局量 —— HLO 更紧凑,
而且如果你忘了调 `setup()`,会得到清晰的 `RuntimeError`,而不是 C++ 深处一个不透明的崩溃。

**dispatch/combine 各是一个 `custom_vjp`,彼此互为反向。** combine 的 VJP 用保存的 handle 跑
一次 dispatch;dispatch 的 VJP 跑一次 combine。handle 把通信布局从前向贯穿到反向,无需重算。

---

## 在 MaxText 中的 FFI 集成

MaxText 这一侧的工作
([`ROCm/maxtext @ feature/primus-turbo-gmm-deepep-integration`](https://github.com/ROCm/maxtext/tree/feature/primus-turbo-gmm-deepep-integration))
是**两个自包含的 commit** ——
[grouped GEMM](https://github.com/ROCm/maxtext/commit/9aeaa97b22676842a2d73e18146ff1b2f37b3a7f)
与
[DeepEP dispatch/combine](https://github.com/ROCm/maxtext/commit/1c5729088c0c2463eae108ffaeaa3735b7395f51)
—— 把 Primus-Turbo 接入 `MaxText/layers/moe.py`,由两个配置开关控制,且**开关关闭时零开销**。

| 开关 | 作用 | 使用的 Primus-Turbo primitive |
|---|---|---|
| `use_turbo_grouped_gemm=true` | 替换无丢弃专家矩阵乘 | `grouped_gemm`、`compute_group_offs` |
| `use_turbo_deepep_dispatch=true` | 替换 EP all-to-all | `moe_dispatch`、`moe_combine`(+ grouped GEMM) |

### Grouped GEMM 替换

在 `sparse_matmul` 的 ragged-GEMM 处,`use_turbo_grouped_gemm` 用 Primus-Turbo 的
`grouped_gemm` 替换 `ragged_dot`/Megablox。此时激活已按专家排序、`group_sizes` 也已算好,
因此这只是矩阵乘本身的就地替换。

### DeepEP Dispatch/Combine 替换

当 `use_turbo_deepep_dispatch` 打开且专家并行分片多于一个时,路由路径变为:`moe_dispatch`
→ fan-out 到按专家排序的布局 → grouped GEMM → fan-in → `moe_combine`。dispatch 句柄通过一个
小的 `_DeepEPCombineState` 具名元组从 dispatch 处带到 combine 处。

下图展示了完整的前向数据流 —— 绿色为 Primus-Turbo FFI custom call,灰色为无 scatter
的 MaxText 胶水代码:

![无丢弃 MoE 前向数据流:x → moe_dispatch → fan-out(custom-VJP gather)→ grouped GEMM → fan-in(reduce-sum)→ moe_combine → 输出;绿色为 Primus-Turbo FFI custom call,灰色为无 scatter 的 MaxText 胶水代码。](images/diagram-integration-flow.png)

### 每进程一次的惰性 `setup()` 引导

DeepEP 的 `setup()` 必须在**真正执行 MoE 前向的那个进程**里运行,因为 Primus-Turbo 是按进程
固定 EP 通信组的。单控制器(single-controller)启动时那是主进程;Ray 启动时则是每个 worker
actor —— 而 driver 侧的引导会完全漏掉这些 actor。因此集成把 `setup()` 折叠进一个每进程一次、
带 guard 的辅助函数,在 MoE 前向的 DeepEP 分支顶部调用:

```python
def _ensure_deepep_setup(mesh, config):
    global _deepep_setup_done
    if _deepep_setup_done:
        return
    try:
        from primus_turbo.jax.lax.moe import setup as _deepep_setup
    except ImportError:
        _deepep_setup = None
    if _deepep_setup is not None:
        _deepep_setup(mesh=mesh, ep_axis_name="expert", hidden_bytes=config.emb_dim * 2)
    _deepep_setup_done = True
```

这个带 guard 的 import 让集成对新旧 Primus-Turbo 都能工作 —— 新版要求显式 `setup()`,旧版
自动引导且不暴露 `setup()`。

### 保证正确性的关键细节

忠实的集成不止是调用 kernel。需要小心的地方:

- **无原子 scatter 的 fan-out / fan-in。** dispatch 之后,token 要 fan-out 到
  `[N*K, hidden]` 的按专家排序布局,专家算完再折回每个收到 token 一行。朴素做法是一个重复
  索引的 gather,其自动微分会在反向发出**原子 scatter-add** —— 又慢又非确定。集成把 fan-out
  实现为 `custom_vjp`:前向是单次复合 gather `recv_x[sort_idx // K]`;反向用 `argsort` 求逆
  置换,并用纯 **reduce-sum** 把 top-k 重复项折叠,绝不 scatter。fan-in 是在 `top_k` 轴上
  reshape + 加权 `sum`(float32 累加),同样无 scatter。
- **屏蔽越界(out-of-group)行。** 超过 `sum(group_sizes)` 的行在 dispatch fan-out 与 combine
  之前都被置零;否则 padding 行的垃圾梯度会经 reduce-sum 折回,在初始化阶段就让训练停滞。
- **线程局部的 int64 group 长度。** CK grouped GEMM 要求 int64 的 `group_lens`,但全局打开
  x64 会破坏 `argsort`(XLA-ROCm 的 s32/s64 scatter 不匹配)。集成只在 kernel 调用处包一层
  `jax.experimental.enable_x64()` —— 线程局部,对并发的 `shard_map` 线程安全。
- **仅 bf16。** DeepEP 路径断言 `dtype == bfloat16`;FP8 dispatch 是另一套 kernel 接口,明确
  不在本文范围。

最终效果:打开这两个开关,你得到一个**无丢弃**的 MoE —— 每个 token 都经 DeepEP 路由、专家
走 CK grouped GEMM、求导正确、并在 FSDP 下干净地 trace —— 而默认(`sparse_matmul=false`)的
图逐字节不变。

---

## 实验

Primus-Turbo 的设计目标,是让无丢弃 MoE 在 AMD Instinct GPU 上变得可用;本节的实验评估它
兑现这一承诺的程度,并回答这些经验问题:无丢弃是否可行、是否快、是否真的训得更好?

**设置(Setup)。** 所有运行都是 [DeepSeek-V3 671B](https://arxiv.org/abs/2412.19437),跑在
**8 节点 × 8 张 AMD MI355X**(64 GPU,每卡 288 GB HBM)上,序列长度 4096,FSDP=8,bf16,
`remat_policy: full`。基础作业配置是
[`configs/deepseek3-671b.gpu.yml`](https://github.com/AMD-AGI/maxtext-slurm/blob/ad3c5245ae5cb82df79d49b213014c1fc6669391/configs/deepseek3-671b.gpu.yml);
下面五个配置只在叠加其上的 MoE 开关上不同。软件栈为 **`rocm/jax-training:maxtext-v26.2`**
容器,并在其上**手动安装 Primus-Turbo(JAX 版,gfx950)**。五个配置:

| 标签 | 路径 |
|---|---|
| `dense-cf{1.25,2,4}` | 容量因子丢弃(`dense_matmul`) |
| `sparse-gmm` | 无丢弃:Primus-Turbo grouped GEMM + `ragged_all_to_all` |
| `sparse-gmm-deepep` | **无丢弃:Primus-Turbo grouped GEMM + DeepEP** |

注意**原生 `sparse_matmul`**(纯 `jax.lax.ragged_dot`,无 grouped-GEMM kernel)不在此列:它
即便 pdbs=1 也会 OOM 到约 444 GiB,在此规模下训不起来。因此上面两个无丢弃配置都用
Primus-Turbo grouped GEMM,只在 all-to-all 上不同(`ragged_all_to_all` vs DeepEP)—— 这正是
让它们的正面对比成为对路由通信的干净隔离。

> **范围说明(Scope)。** 这是一个受控的 A/B 对比,不是峰值性能基准。每个配置都跑同一套固定
> 配方,没有 per-config 的 kernel 或 XLA 调优,因此配置之间的**相对**差距才是结论 —— 看差值,
> 不要看绝对数值。绝对吞吐是地板(还有调优空间),不是天花板。

### 吞吐(合成数据)

稳态 tokens/s/device(TGS),`dataset_type=synthetic`,步 5–14 的均值(FSDP=8):

| pdbs | dense-cf1.25 | dense-cf2 | dense-cf4 | sparse-gmm | sparse-gmm-deepep |
|-----:|-------------:|----------:|----------:|-----------------:|---------------------:|
|    4 |        961.1 |     813.1 |     532.6 |            757.1 |                873.4 |
|    7 |       1209.8 |     934.3 |     575.5 |          974.5 † |               1121.6 |
| **8**|   **1292.7** |  **960.1**|  **580.7**|    **OOM** 242 GiB |           **1179.7** |
|    9 |       1399.0 |     994.8 | OOM 212 GiB |    OOM 242 GiB |               1189.4 |
|   16 |       1438.1 |    1015.1 | OOM 278 GiB |              —   |            OOM 316 GiB |

† `sparse-gmm` 在 pdbs=7 只有降低显存占比(memory fraction)才放得下。

### 数据解读

- **DeepEP 挪回一档 batch。** 两个无丢弃路径都付[无丢弃路由](#sparse_matmul无丢弃路由)的最坏情况悲观分配,但 DeepEP 的缓冲区
  管理更精简 —— 同 pdbs 下瞬时占用低约 40 GiB(~15%)。这点不大的削减足以跨过可行性阈值:
  内置 `sparse-gmm` 在 pdbs=8 **OOM**(~242 GiB),而 `sparse-gmm-deepep` 在 pdbs=8 乃至
  pdbs=9 都能跑。它没让无丢弃变便宜 —— 而是让它多放下一档。
- **DeepEP 是最快的无丢弃方案。** 在每个可行的 pdbs 上,`sparse-gmm-deepep` 都胜过
  `sparse-gmm`(如 pdbs=7 的 1121.6 vs 974.5),并在保持**无丢弃**的同时逼近有损的
  `dense-cf1.25` 基线。无丢弃峰值吞吐:`sparse-gmm-deepep` 在 pdbs=8 → **1179.7 TGS**。
- **容量因子代价是真实的。** `dense-cf2`/`dense-cf4` 靠提高容量来找回保真度,但吞吐崩塌
  (cf4 ≈ 575 TGS,且 pdbs>8 OOM)。无丢弃 DeepEP 路径以约 **2×** 于 cf4 的吞吐提供完整
  保真度。

### 收敛与 loss 质量(C4 数据)

上面的合成扫描只衡量**吞吐与可行性** —— 固定输入、短跑,说明不了模型是否学得对。为此需要
一次真实训练 —— DeepSeek-V3 671B 在 **[C4](https://arxiv.org/abs/1910.10683)**(`dataset_type=grain`,parquet
`c4-train-*-of-01024.parquet`,HF `deepseek-ai/DeepSeek-V3-Base` tokenizer,seq 4096)上跑
**2000 步**,FSDP=8,pdbs=7,相同初始化,覆盖全部五个配置。逐步训练 loss(TensorBoard
`learning/loss`)绘制如下图:

![C4 上的训练 loss 随步数变化:`sparse-gmm-deepep`(无丢弃)收敛到低于全部三个容量因子稠密配置 —— 第 2000 步终值 `cf1.25` 5.163、`cf2` 5.119、`cf4` 5.081、`sparse-gmm-deepep` 5.003(`sparse-gmm` ≈ 4.999,未画出,与 `sparse-gmm-deepep` 重合)。](images/loss-per-step.png)

(这些 C4 运行用的是同一集成的较早构建 —— kernel 与路由算法与当前一致。)

两条干净的结论:

1. **DeepEP 无丢弃与 `ragged_all_to_all` 无丢弃路径吻合。** 两条路径共享*同一个* grouped-GEMM
   专家矩阵乘,只在 all-to-all 上不同,理想情况下曲线应当一模一样。实测中 `sparse-gmm-deepep`
   (5.003)与 `ragged_all_to_all` 的 `sparse-gmm`(4.999)在第 2000 步相差 **0.004**,且全程
   贴合。这点残差是 2000 步累积的**非确定性**(非确定的规约/kernel 调度),不是算法差异 ——
   DeepEP 通信 + grouped GEMM + custom-VJP fan-out/fan-in 端到端地复现了无丢弃训练,而不仅是
   单步。
2. **无丢弃相对容量因子丢弃是 Pareto 占优的**(同 batch size)—— `sparse-gmm-deepep` 在*两个*
   维度上都胜过 `dense-cf{1.25, 2, 4}`:

   - **固定步数下 loss 更低**(收敛质量 —— 每步学到更多)。两个无丢弃路径都收敛到低于任一
     稠密配置之下,而稠密家族内 loss 随容量单调下降(cf1.25 = 5.163 → cf2 = 5.119 →
     cf4 = 5.081):丢得越少,loss 越低,而**无丢弃就是极限**。DeepEP 达到
     **5.003 —— 比 `dense-cf1.25` 默认低 0.16 nat** —— 因为它什么都不丢。
   - **固定墙钟时间下 loss 更低**(time-to-loss)。在 C4 上,无丢弃路径每小时步数比
     `dense-cf1.25` 少约 19%(818 vs 1009 TGS;吞吐代价见[下一节](#真实数据上的吞吐代价)),但它在相同墙钟下仍达到
     *更低*的 loss —— 每步质量收益足以抵回损失的步速。

下图把这个 time-to-loss 对比具体化,画出丢弃默认路径与无丢弃路径的 loss 随墙钟时间的
变化:

![C4 上的训练 loss 随墙钟时间变化,`dense-cf1.25` vs `sparse-gmm-deepep`:尽管步速更低,无丢弃路径在相同墙钟下达到更低的 loss。](images/loss-per-walltime.png)

### 真实数据上的吞吐代价

有一个代价要老实说。从合成切到真实 C4 数据会拉低每个配置的 TGS —— 但其中大部分只是合成
loader 不必付的数据加载开销,而且对稠密与无丢弃一视同仁(`dense-cf1.25` 自己也掉了 ~17%)。
无丢弃特有的信号是 **`dense-cf1.25` 与无丢弃路径之间的差距**,它在真实数据上明显拉大。
pdbs=7(FSDP=8)的 TGS:

| 配置 | 合成 | C4 |
|---|---:|---:|
| dense-cf1.25 | 1210 | 1009 |
| dense-cf2 | 934 | 810 |
| dense-cf4 | 576 | 522 |
| sparse-gmm | 975 | 753 |
| **sparse-gmm-deepep** | 1122 | **818** |

`sparse-gmm-deepep` 在合成上只落后 `dense-cf1.25` **~7%**(1122 vs 1210),但在 C4 上落后
**~19%**(818 vs 1009)。因为 `dense-cf1.25` 在结构上**对路由倾斜免疫** —— 容量因子丢弃把
每个专家都填到固定大小,与 token 怎么路由无关 —— 所以这*额外*拉大的部分正是无丢弃路径上
**路由不均衡**的代价:真实文本把 token 集中到少数热门专家,于是 ragged grouped GEMM 与
dispatch/combine all-to-all 都被最忙的那个专家拖住。合成路由接近均匀,把这一点掩盖了。

要点在于,这种不均衡是**可调的、而非固定的**:更强的 MoE 负载均衡(辅助)loss 或
[router z-loss](https://arxiv.org/abs/2202.08906) 调参会把每个专家的 token 分布拉平,从而缩小这个差距。本文的所有运行都没有动这些旋钮
—— 每个配置都用模型默认的路由与 loss 设置 —— 因此实测的代价是一个**上界**,而非无丢弃路径
固有的性质。

即便如此,真实数据上仍有两点成立:

- **`sparse-gmm-deepep` 明显快于 `sparse-gmm`**(818 vs 753,+9%):DeepEP 的 dispatch 比
  `ragged_all_to_all` 路径更能容忍这种不均衡。
- **`sparse-gmm-deepep` 不低于 `dense-cf2`**(818 vs 810)—— 无丢弃 DeepEP 路径在吞吐上追平
  一个中等容量的稠密配置,同时 loss 严格更优。

C4 上的 per-device 吞吐汇总如下图:

![C4 上的 per-device TGS(第 2000 步附近):`sparse-gmm-deepep`(~818)胜过 `sparse-gmm`(~753)、略高于 `dense-cf2`(~810);`dense-cf1.25` 领先(~1009),`dense-cf4` 垫底(~522)。](images/tgs-c4.png)

而这笔吞吐税已被[收敛与 loss 质量](#收敛与-loss-质量c4-数据)的 time-to-loss 结果**付清**:即便相对 `dense-cf1.25` 让出约 19% 的
步速,`sparse-gmm-deepep` 在相同墙钟下仍达到更低的 loss。不均衡的代价是真实的,但收敛质量的
收益绰绰有余地把它盖过。

---

## 总结

在 ROCm/JAX 栈上,无丢弃 MoE 历来逼人做一个两难选择:要么保持形状方正而**丢 token**
(`dense_matmul`),要么坚持**无丢弃**却撞上不规则路径的内存墙 —— 一个贵得离谱的 `ragged_dot`
专家矩阵乘,和一个最坏情况的 `ragged_all_to_all` 搬运(`sparse_matmul`)。Primus-Turbo 把两个
基于 CK 的 primitive 带到 JAX —— 用 **grouped GEMM** 做不规则专家矩阵乘,用 **DeepEP
dispatch/combine** 做感知 token 的路由 all-to-all —— 并把它们暴露为带自动微分、`shard_map`
分片与干净 `setup()` 契约的一等 JAX op,从而化解了这一两难。

通过两个 `use_turbo_*` 开关接入 MaxText —— 配上精心设计的 custom-VJP fan-out/fan-in、越界行
屏蔽、线程局部 int64,以及一个覆盖单控制器与 Ray 两种启动方式的每进程一次引导 —— 得到的 MoE
训练路径具备:

- **无丢弃**(每个 token 都到达其专家),
- **更少受显存约束**(更精简的 all-to-all 缓冲挪回一档 batch —— 在 `ragged_all_to_all` 于
  pdbs=8 就 OOM 的地方仍能跑 pdbs=8–9),
- **最快的无丢弃方案**(pdbs=8 约 1180 TGS,约为 cf4 容量路径的 2×),
- **数值忠实**(2000 步 C4 训练中,loss 与 `ragged_all_to_all` 无丢弃路径相差 0.004),
- 以及在同 batch size 下**相对容量因子丢弃 Pareto 占优** —— 无论按步(5.003 vs
  `dense-cf1.25` 的 5.163)还是按墙钟,C4 loss 都更低,

且开关关闭时**零开销**。kernel、显存、吞吐、梯度与端到端 C4 收敛均已验证(FSDP=8)。

让这一切变得可行,功劳归于 Primus-Turbo:它把基于 CK 的 grouped GEMM 与 DeepEP kernel
作为干净、且感知自动微分与分片的 JAX op 提供出来,从而把 AMD Instinct GPU 上的无丢弃
MoE 从一条不可行的研究路径,变成一个用一个开关就能打开的默认选择。如果你在
JAX/MaxText 中训练 MoE 模型,我们鼓励你在自己的运行中打开这些开关,亲眼看看无丢弃能
带来什么。

---

## 附加资源

### 混合专家(MoE)

- Shazeer 等,*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
  Layer*(2017)— [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Lepikhin 等,*GShard*(2020)— [arXiv:2006.16668](https://arxiv.org/abs/2006.16668);
  Fedus 等,*Switch Transformers*(2021)— [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
  ([`dense_matmul`](#dense_matmul容量因子丢弃) 的容量因子 / token 丢弃脉络)
- Gale 等,*MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*(2022)—
  [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)(无丢弃 / grouped-GEMM;见 [`sparse_matmul`](#sparse_matmul无丢弃路由) 与 [Grouped GEMM](#grouped-gemm-gmm))
- Zoph 等,*ST-MoE: Designing Stable and Transferable Sparse Expert Models*(2022)—
  [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)(router z-loss;见[真实数据上的吞吐代价](#真实数据上的吞吐代价))
- DeepSeek-AI,*DeepSeekMoE*(2024)— [arXiv:2401.06066](https://arxiv.org/abs/2401.06066);
  *DeepSeek-V3 Technical Report*(2024)— [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 系统与库

- DeepEP(DeepSeek 专家并行 dispatch/combine)—
  [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- Composable Kernel(CK),GEMM 后端 —
  [github.com/ROCm/composable_kernel](https://github.com/ROCm/composable_kernel)
- rocSHMEM(以 GPU 为中心的 OpenSHMEM,DeepEP 的 internode 通信底座)—
  [rocm.docs.amd.com/projects/rocSHMEM](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/)
- JAX 外部函数接口(FFI)—
  [docs.jax.dev/en/latest/ffi.html](https://docs.jax.dev/en/latest/ffi.html)
- JAX Pallas(自带 kernel DSL)—
  [docs.jax.dev/en/latest/pallas](https://docs.jax.dev/en/latest/pallas/index.html)
- C4(Colossal Clean Crawled Corpus),训练数据 —— Raffel 等,*T5*(2019)—
  [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- MaxText — [github.com/AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext);
  集成位于 AMD fork 分支
  [`ROCm/maxtext @ feature/primus-turbo-gmm-deepep-integration`](https://github.com/ROCm/maxtext/tree/feature/primus-turbo-gmm-deepep-integration)
- Primus-Turbo — [github.com/AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
- maxtext-slurm(本文运行所用的启动器 —— 用于在 Slurm 管理的 GPU 集群上启动与观测 MaxText
  训练)— [github.com/AMD-AGI/maxtext-slurm](https://github.com/AMD-AGI/maxtext-slurm)
- PyTorch/Megatron 对应实现(Primus-Turbo MoE,含 *Sync-Free MoE*):
  [MoE Training Best Practices on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.

However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, AMD Instinct, AMD ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. PyTorch is a registered trademark of Meta Platforms, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved
