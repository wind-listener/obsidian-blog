

### 什么是 Einops？

`einops`（**Ein**stein **Op**eration **S**ummary）是一个旨在让张量（多维数组）操作变得直观、清晰和可靠的库。它的名字灵感来源于爱因斯坦求和约定，因为它同样强调通过标注维度名称来定义操作。

它的核心思想是：**你只需要告诉库你“想要什么样子”，而不必关心“如何通过一系列基础操作（如`reshape`, `transpose`, `squeeze`等）实现它”**。

**主要优势：**
1.  **可读性极强**：代码即文档，一看 `rearrange(x, 'b c h w -> b h w c')` 就知道这是在做什么（例如，将通道维从第二位移到最后一位）。
2.  **可靠性高**：`einops` 会隐式地检查操作的一致性（例如，确保重塑前后的元素总数相等），大大减少了因维度不匹配而导致的错误。
3.  **强大且统一**：它用一个统一的接口替代了多个 NumPy/PyTorch/TensorFlow/JAX 中的函数（如 `reshape`, `transpose`, `squeeze`, `stack`, `concatenate` 等）。
4.  **支持多框架**：完美支持 NumPy、PyTorch、TensorFlow、JAX 等主流张量库。

---

### 安装

使用 pip 安装即可：

```bash
pip install einops
```

---

### 三个最核心、最常用的方法

`einops` 提供了三个核心函数，几乎涵盖了所有常见的张量操作场景。

#### 1. `rearrange` - 重新排列/重塑维度

这是最常用、最核心的函数。它可以实现维度重排、展平、分解、挤压（移除长度为1的维度）以及通过重复维度实现简单的张量组合。

**基本语法：** `rearrange(tensor, 'input_pattern -> output_pattern')`

**常用操作示例：**

假设我们有一个图像批量张量 `x`，其形状为 `(batch, height, width, channel)`，即 `(2, 32, 32, 3)`。

*   **置换维度 (Transpose)**：
    ```python
    from einops import rearrange
    import numpy as np

    x = np.random.randn(2, 32, 32, 3)

    # 将通道维从最后移到第二维 (NHWC -> NCHW，PyTorch常用格式)
    y = rearrange(x, 'b h w c -> b c h w')
    print(y.shape) # (2, 3, 32, 32)
    ```

*   **展平/分解维度 (Flatten/Decompose)**：
    ```python
    # 将高度和宽度展平为一个“空间”维度
    y = rearrange(x, 'b h w c -> b (h w) c')
    print(y.shape) # (2, 1024, 3)

    # 将高度分解为 2 个 16 的块
    y = rearrange(x, 'b (h1 h2) w c -> b h1 h2 w c', h1=2)
    print(y.shape) # (2, 2, 16, 32, 3)
    ```

*   **空间维度的重新排序（例如，将图像分割为补丁）**：
    ```python
    # 一个非常强大的操作：将图像划分为 8x8 的补丁 (patches)
    # 原理：将高度和宽度分别分解成 4 个 8 的块 (32/8=4)
    y = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=8, p2=8)
    print(y.shape) # (2, 16, 192)
    # 结果解释：2个样本，16个补丁(4x4)，每个补丁是8x8x3=192维
    ```

#### 2. `reduce` - 减少维度（聚合操作）

它不仅能够重新排列维度，还能在指定的维度上进行聚合操作（如求和、求平均、取最大/最小值）。

**基本语法：** `reduce(tensor, 'input_pattern -> output_pattern', reduction)`

**常用操作示例：**

*   **全局平均池化 (Global Average Pooling)**：
    ```python
    from einops import reduce

    # 在高度和宽度维度上求平均，实现全局平均池化
    y = reduce(x, 'b h w c -> b c', 'mean')
    print(y.shape) # (2, 3)
    ```

*   **空间维度的最大池化（类似2x2 MaxPool）**：
    ```python
    # 将高和宽分别分解为 2x2 的块，并在每个块内取最大值
    y = reduce(x, 'b (h h2) (w w2) c -> b h w c', 'max', h2=2, w2=2)
    print(y.shape) # (2, 16, 16, 3) 形状缩小了一半
    ```

*   **求和与减少多个维度**：
    ```python
    # 对所有批次的数据求和，得到一个代表所有数据的张量
    y = reduce(x, 'b h w c -> h w c', 'sum')
    print(y.shape) # (32, 32, 3)
    ```

#### 3. `repeat` - 重复张量

与 `reduce` 相反，`repeat` 用于沿着指定的新维度或现有维度重复张量。

**基本语法：** `repeat(tensor, 'input_pattern -> output_pattern', repetitions)`

**常用操作示例：**

*   **沿着新维度重复**：
    ```python
    from einops import repeat

    # 创建一个新的“组”维度，并沿其重复3次
    y = repeat(x, 'b h w c -> g b h w c', g=3)
    print(y.shape) # (3, 2, 32, 32, 3)
    ```

*   **沿着现有维度重复（广播）**：
    ```python
    # 将通道维度上的数据重复2次
    # 注意：输出模式中 c 变成了 (c*2)，需要匹配
    y = repeat(x, 'b h w c -> b h w (c repeat)', repeat=2)
    print(y.shape) # (2, 32, 32, 6)
    ```

*   **模仿 `np.tile` 的功能**：
    ```python
    # 在高度和宽度上分别重复2次和3次
    y = repeat(x, 'b h w c -> b (h repeat_h) (w repeat_w) c', repeat_h=2, repeat_w=3)
    print(y.shape) # (2, 64, 96, 3)
    ```

---

### 模式字符串的语法规则

这是 `einops` 的灵魂，理解它就能理解所有操作：

*   **空格分隔**：输入和输出模式中的维度用空格分隔，例如 `b c h w`。
*   **括号 `()`**：用于将多个轴组合在一起，例如 `(h w)` 表示将 `h` 和 `w` 两个维度展平为一个。
*   **分解命名**：可以通过添加新名称来分解维度，例如 `(h h1 h2)` 表示将 `h` 分解为 `h1` 和 `h2` 两个维度。你必须提供 `h1=` 或 `h2=` 参数来指定分解的大小。
*   **省略号 `...`**：用于处理任意数量的维度，非常方便。
    ```python
    # 无论张量有多少个维度，只对最后两个维度进行转置
    y = rearrange(x, '... a b -> ... b a')
    ```

---

### 一个综合例子：Vision Transformer (ViT) 的补丁嵌入

`einops` 在 Transformer 模型中极其常用。以下是如何用 `einops` 实现 ViT 的第一步：将图像划分为补丁并展平。

```python
import torch
from einops import rearrange

# 假设输入图像张量：batch_size=4, channels=3, height=224, width=224
x = torch.randn(4, 3, 224, 224)

# 补丁大小
patch_size = 16

# 一步到位：划分补丁并展平
patches = rearrange(x,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size, p2=patch_size)

print(patches.shape) # (4, 196, 768)
# 解释：
# h = 224 / 16 = 14, w = 14
# 所以有 14*14 = 196 个补丁
# 每个补丁是 16*16*3 = 768 维
```
如果没有 `einops`，你需要组合使用 `.unfold`, `.permute`, `.reshape` 等，代码会冗长且难以理解。

### 总结

| 方法 | 功能 | 相当于原生操作的组合 |
| :--- | :--- | :--- |
| **`rearrange`** | 重塑、转置、展平、分割、挤压维度 | `reshape`, `transpose`, `flatten`, `squeeze`/`unsqueeze` |
| **`reduce`** | 在指定维度上进行聚合（如求和、求平均） | `mean(axis=...)`, `sum(axis=...)`, `max(axis=...)` + `reshape` |
| **`repeat`** | 沿新维度或现有维度重复张量 | `repeat`, `tile`, `expand` |

**强烈建议**：在任何涉及多维数组操作的项目中尝试使用 `einops`。它会显著提升你代码的可读性和可维护性，让你更专注于实现想法而不是调试维度错误。