# B200 性能测试报告 (DeepSeek3-671B & Kimi-K2-1T)

## 1. 摘要与核心结论

本报告汇总了在 NVIDIA B200 (180GB HBM3e) 8 节点集群上，针对 **DeepSeek3-671B** 和 **Kimi-K2-1T** 模型的性能测试结果。

### 核心突破 (2026-05-10)
1.  **Kimi-K2-1T 显存解锁**: 通过将 `XLA_PYTHON_CLIENT_MEM_FRACTION` 提高至 `.97`，成功将 Kimi 的最大 Batch Size 从 `bs=1` 提升至 **`bs=2`** (BF16 & FP8)。
2.  **DeepSeek3-671B 高 CF 解锁**:
    *   `cf=2.0`: 成功解锁 **`bs=6`** (之前上限为 `bs=5`)。
    *   `cf=4.0`: 成功解锁 **`bs=3`** (之前上限为 `bs=2`)。
3.  **DeepSeek3-671B FP8 性能**: 在 `cf=1.25` 下达到 **350.4 TFLOP/s/dev** (`bs=9`)。
4.  **最佳 XLA 优化**: 确认 `xla_gpu_experimental_parallel_collective_overlap_limit=4` 是提升 TFLOP/s 的最有效单项优化（约 +9%），但在极高 Batch Size 或高 CF 下可能触发网络不稳定 (`IBV_WC_RETRY_EXC_ERR`)。

---

## 2. 汇总表格

### 2.1 DeepSeek3-671B (8 Nodes)

#### BF16 精度
| Job ID | 配置 (bs, cf) | 状态 | Step (s) | MFU (%) | TFLOP/s/dev | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4405 | bs=9, cf=1.0 | SUCCESS | 38.30 | 10.70 | 240.8 | **BF16 最大 BS** |
| 4404 | bs=8, cf=1.25 | SUCCESS | 38.10 | 10.75 | 242.0 | `mem97` 解锁 |
| 4417 | bs=6, cf=2.0 | SUCCESS | 28.43 | 9.62 | 216.5 | `mem97` 解锁 |
| 4419 | bs=3, cf=4.0 | SUCCESS | 26.39 | 5.18 | 116.6 | `mem97` 解锁 |
| 4423 | bs=7, cf=2.0 | SUCCESS | 31.40 | 10.16 | 228.7 | **解锁 bs=7!** |
| 4424 | bs=4, cf=4.0 | FAILED | -- | -- | -- | 网络错误 (IBV_WC_RETRY) |
| 4426 | bs=7, cf=1.25 | SUCCESS | -- | -- | -- | **Profile/Dump (Best) - Verified** |
| 4427 | bs=7, cf=1.25 | SUCCESS | -- | -- | -- | **Profile/Dump (NV) - Verified** |
| 4428 | bs=7, cf=1.25 | SUCCESS | -- | -- | -- | **Profile/Dump (AMD) - Verified** |
| 4429 | bs=9, cf=1.25 | CANCELLED | -- | -- | -- | 预约结束 (Reservation end) |
| 4430 | Kimi BF16 bs=2 | CANCELLED | -- | -- | -- | 预约结束 (Reservation end) |
| 4431 | Kimi FP8 bs=2 | CANCELLED | -- | -- | -- | 预约结束 (Reservation end) |
| 4242 | bs=7, cf=1.25 | SUCCESS | 25.43 | 12.06 | 271.4 | `overlap4` 最佳性能 |

#### FP8 精度
| Job ID | 配置 (bs, cf) | 状态 | Step (s) | MFU (%) | TFLOP/s/dev | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4413 | bs=9, cf=1.25 | SUCCESS | 31.85 | 15.58 | 350.4 | **FP8 最大吞吐** |
| 4170 | bs=8, cf=1.0 | SUCCESS | 25.96 | 16.96 | 381.5 | |

### 2.2 Kimi-K2-1T (8 Nodes)

| Job ID | 精度 | 配置 (bs, cf) | 状态 | Step (s) | MFU (%) | TFLOP/s/dev | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4410 | BF16 | bs=2, cf=1.0 | SUCCESS | 22.18 | 4.01 | 90.2 | `mem97` 解锁 |
| 4414 | BF16 | bs=2, cf=1.25 | SUCCESS | 22.95 | 3.88 | 87.2 | |
| 4411 | FP8 | bs=2, cf=1.0 | SUCCESS | 23.34 | 3.81 | 85.7 | |
| 4415 | FP8 | bs=2, cf=1.25 | SUCCESS | 24.13 | 3.69 | 82.9 | |

---

## 3. 详细测试记录 (最近任务)

#### Job 4417: DS3 BF16 bs=6, cf=2.0, overlap4, mem97
*   **结果**: SUCCESS, 216.5 TFLOP/s.
*   **意义**: 成功突破了之前 `cf=2.0` 下 `bs=5` 的限制。

#### Job 4419: DS3 BF16 bs=3, cf=4.0, overlap4, mem97
*   **结果**: SUCCESS, 116.6 TFLOP/s.
*   **意义**: 成功突破了之前 `cf=4.0` 下 `bs=2` 的限制。

#### Job 4413: DS3 FP8 bs=9, cf=1.25, mem97
*   **结果**: SUCCESS, 350.4 TFLOP/s.
*   **意义**: FP8 在 `cf=1.25` 下的最高吞吐配置。

#### Job 4410/4411/4414/4415: Kimi bs=2 Breakthrough
*   **结果**: 全部 SUCCESS。
*   **意义**: 确认 `XLA_PYTHON_CLIENT_MEM_FRACTION=.97` 是解锁 Kimi `bs=2` 的关键。

---

## 4. 待办与建议
1.  **网络稳定性**: 观察到 `overlap_limit=4` 在极高负载下（如 DS3 FP8 bs=9 cf=1.0 或 BF16 bs=5 cf=2.0）偶尔触发 `IBV_WC_RETRY_EXC_ERR`。如果追求极致稳定性，建议在生产环境回退到 NV Defaults。
2.  **进一步探测**: 目前 `bs=6 (cf=2.0)` 和 `bs=3 (cf=4.0)` 已接近 LHS 显存预算上限，进一步提升 BS 的空间较小。
