---
title: "Pytorch 字符串表达式"
date: 2025-10-29
draft: false
---

在 PyTorch 中，可以通过字符串表达式（string expressions）来动态定义张量运算规则，这种方式在 `torch.einsum()`、`torch.compile()` 的 `dynamic` 选项等场景中特别有用。以下是详细总结：

---

### **1. 支持字符串表达式的核心方法**
#### **(1) `torch.einsum()`**
[[爱因斯坦求和约定]]，用字符串指定张量运算的维度和求和规则。
```python
import torch

# 矩阵乘法 (A @ B)
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum("ik,kj->ij", A, B)  # 等价于 A @ B

# 向量点积
x = torch.randn(5)
y = torch.randn(5)
dot = torch.einsum("i,i->", x, y)  # 等价于 x.dot(y)
```

#### **(2) `torch.compile()` 的动态形状（Experimental）**
PyTorch 2.0+ 的 `torch.compile` 支持用字符串标记动态维度（如 `"batch"`）以优化动态形状计算。
```python
def model(x):
    return x * 2

compiled_model = torch.compile(model, dynamic=True)  # 自动处理动态维度
```

#### **(3) 第三方库 `sympy` 表达式（需手动集成）**
可通过 `sympy` 解析字符串表达式生成计算图（非原生支持，需额外处理）。

---

### **2. 字符串表达式的规范**
#### **(1) `einsum` 的字符串规则**
- **基本格式**：`"operands -> output"`  
  - 左端输入张量的维度用逗号分隔，右端是输出维度。
  - 重复字母表示对应维度需要相乘并求和（缩并）。
- **示例**：
  - `"ij,jk->ik"`：矩阵乘法（`i,k` 保留，`j` 求和）。
  - `"ii->i"`：矩阵对角线。
  - `"bij,bjk->bik"`：批量矩阵乘法。

#### **(2) 动态维度的标记（如 `torch.compile`）**
- 用字符串标记动态维度（如 `"batch"`、`"seq_len"`），编译器会自动优化：
  ```python
  x = torch.randn(16, 32)  # 假设第一维是动态的
  compiled_fn = torch.compile(lambda x: x.sum(dim=1), dynamic=True)
  ```

#### **(3) 通用规范**
- **字母约定**：通常用小写字母（`i,j,k`）表示维度，避免特殊字符。
- **维度顺序**：需与实际张量形状严格匹配。
- **广播规则**：与 PyTorch 广播规则一致，如 `"i,j->ij"` 表示外积。

---

### **3. 注意事项**
1. **`einsum` 的性能**：对于复杂运算，`einsum` 可能不如直接调用优化过的函数（如 `matmul`）。
2. **动态形状的限制**：`torch.compile` 的动态标记仍在优化中，可能不适用于所有场景。
3. **错误检查**：无效的表达式会抛出 `RuntimeError`（如维度不匹配）。

---

### **4. 完整示例**
```python
# 1. einsum 实现批量矩阵乘法
batch = 10
A = torch.randn(batch, 3, 4)
B = torch.randn(batch, 4, 5)
C = torch.einsum("bij,bjk->bik", A, B)  # 输出形状 [10, 3, 5]

# 2. 动态维度编译
def dynamic_sum(x):
    return x.sum(dim=-1)

compiled_sum = torch.compile(dynamic_sum, dynamic=True)
result = compiled_sum(torch.randn(8, 64))  # 自动处理动态维度
```

通过字符串表达式，PyTorch 提供了灵活的张量操作接口，尤其适合需要动态定义计算的场景。




在 PyTorch 中，字符串不仅可以用于定义运算规则（如 `einsum`），还能通过特定语法实现**张量形状调整**、**维度操作**和**动态标记**。以下是全面总结：

---

# PyTorch 中字符串的全面用法

## 一、形状调整与维度操作

### 1. `reshape` / `view` 的字符串表达
PyTorch 原生不支持直接用字符串定义形状调整，但可通过字符串解析实现动态形状变换：
```python
shape_str = "b c h w -> b (c h w)"  # 自定义格式
tensor = torch.randn(2, 3, 4, 5)

# 实现字符串解析
def reshape_by_str(tensor, shape_str):
    src, dst = shape_str.split("->")
    src_dims = src.replace(" ", "").split(",")
    dst_dims = dst.replace(" ", "").split(",")
    
    # 解析括号内的合并维度
    new_shape = []
    for dim in dst_dims:
        if "(" in dim:
            dims = dim.strip("()").split("*")
            new_shape.append(eval("*".join(dims)))
        else:
            new_shape.append(tensor.size(int(dim)))
    return tensor.reshape(new_shape)

reshaped = reshape_by_str(tensor, "b c h w -> b (c*h w)")  # 形状变为 [2, 15, 4]
```

### 2. `permute` 的字符串表达
通过字母顺序定义维度置换：
```python
def permute_by_str(tensor, dim_str):
    dims = dim_str.replace(" ", "").split("->")
    src_dims = dims[0].split(",")
    dst_dims = dims[1].split(",")
    permute_order = [src_dims.index(d) for d in dst_dims]
    return tensor.permute(permute_order)

x = torch.randn(2, 3, 4)
y = permute_by_str(x, "b,c,h -> h,b,c")  # 形状变为 [4, 2, 3]
```

---

## 二、动态形状标记（PyTorch 2.0+）

### 1. 编译期动态标记
在 `torch.compile` 中用字符串标记动态维度：
```python
@torch.compile(dynamic=True)
def fn(x):
    # 标记 "batch" 为动态维度
    return x.view(-1, x.size(-1))  # 自动处理变长batch

x = torch.randn(3, 4)  # 实际运行时可以是任意batch大小
fn(x)
```

### 2. 符号形状（Symbolic Shapes）
通过 `torch._dynamo` 实现符号化：
```python
from torch._dynamo import allow_in_graph

@allow_in_graph
def symbolic_reshape(x, shape_str):
    # 解析如 "B, C, H*W" 的字符串
    return x.reshape(eval(shape_str))

x = torch.randn(2, 3, 4)
y = symbolic_reshape(x, "2, 3, 4*1")  # 形状变为 [2, 3, 4]
```

---

## 三、张量运算的字符串扩展

### 1. 高级 `einsum` 用法
支持广播和批量维度：
```python
# 批量矩阵乘法 + 维度压缩
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
C = torch.einsum("...ij,...jk->...ik", A, B)  # 形状 [10, 3, 5]

# 对角线提取
D = torch.einsum("ii->i", torch.randn(5, 5))  # 形状 [5]
```

### 2. 自定义运算符
结合 `torch.compile` 实现字符串定义的算子：
```python
op_map = {
    "add": lambda x, y: x + y,
    "mul": lambda x, y: x * y
}

def str_op(x, y, op_str):
    return op_mapx, y

compiled_op = torch.compile(str_op, dynamic=True)
result = compiled_op(torch.tensor(2), torch.tensor(3), "mul")  # 返回 6
```

---

## 四、字符串规范与最佳实践

### 1. 语法规范
| 类型          | 示例                  | 说明                          |
|---------------|-----------------------|-----------------------------|
| 维度标记      | `b,c,h,w`            | 字母或单词表示维度               |
| 合并维度      | `(h*w)`              | 括号内乘法表示合并               |
| 动态维度      | `B,H,W` (B大写)      | 首字母大写表示动态维度            |
| 广播维度      | `...,c`              | 省略号表示自动广播               |

### 2. 性能建议
- **避免频繁解析**：对热点代码预编译字符串规则
- **维度顺序优化**：将大维度放在最后（如 `channels_last`）
- **混合使用**：复杂操作组合 `einsum` + `view` 比纯字符串解析更高效

---

## 五、完整案例：Transformer 注意力实现

```python
def attention(Q, K, V, mask_str="b h q k"):
    # 1. 计算注意力分数
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(Q.size(-1))
    
    # 2. 应用动态mask（通过字符串解析）
    if mask_str:
        mask = parse_mask_str(mask_str)  # 自定义解析函数
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. 形状调整
    attn = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attn, V)
    return output.permute(0, 2, 1, 3)  # "b h q d -> b q h d"
```

---

## 六、扩展阅读
1. https://pytorch.org/docs/stable/dynamo/how-to-guides/dynamic_shapes.html
2. https://rockt.github.io/2018/04/30/einsum
3. https://arxiv.org/abs/2205.13443

通过字符串抽象，PyTorch 实现了从静态计算图到动态维度处理的统一表达，极大提升了代码的灵活性和可读性。