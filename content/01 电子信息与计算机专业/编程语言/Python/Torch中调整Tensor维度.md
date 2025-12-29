---
title: "Torch中调整Tensor维度"
date: 2025-08-07
draft: false
---

在 PyTorch 中，调整张量维度的操作非常常见。这里总结了几种常用的调整维度的方法，涵盖了不同的操作和用途。

  

**1. torch.squeeze()**

• **作用**：去除维度为1的维度。

• **示例**：

```
tensor = torch.randn(1, 3, 1, 5)
tensor_squeezed = tensor.squeeze()
print(tensor_squeezed.shape)  # 输出: torch.Size([3, 5])
```

  

• **注意**：如果你只想去除某个特定维度为 1 的维度，可以传入维度索引作为参数。

```
tensor_squeezed = tensor.squeeze(0)  # 只去除第0维
```

  

  

**2. torch.unsqueeze()**

• **作用**：在指定位置插入维度为1的维度。

• **示例**：

```
tensor = torch.randn(3, 5)
tensor_unsqueezed = tensor.unsqueeze(0)  # 在第0维插入
print(tensor_unsqueezed.shape)  # 输出: torch.Size([1, 3, 5])
```

  

  

**3. torch.view()**

• **作用**：重塑张量的形状。需要注意的是，这个操作不会改变数据本身。

• **示例**：

```
tensor = torch.randn(2, 3, 4)
reshaped_tensor = tensor.view(3, 8)
print(reshaped_tensor.shape)  # 输出: torch.Size([3, 8])
```

  

• **注意**：你也可以使用 -1 来自动推断某个维度的大小。

```
reshaped_tensor = tensor.view(-1, 4)  # 自动计算第一个维度大小
```

  

  

**4. torch.permute()**

• **作用**：重新排列张量的维度。

• **示例**：

```
tensor = torch.randn(2, 3, 5)
permuted_tensor = tensor.permute(2, 0, 1)  # 改变维度顺序
print(permuted_tensor.shape)  # 输出: torch.Size([5, 2, 3])
```

  

  

**5. torch.transpose()**

• **作用**：交换两个维度。

• **示例**：

```
tensor = torch.randn(2, 3)
transposed_tensor = tensor.transpose(0, 1)  # 交换第0维和第1维
print(transposed_tensor.shape)  # 输出: torch.Size([3, 2])
```

  

• **注意**：这个操作仅适用于二维张量（矩阵）。

  

**6. torch.flatten()**

• **作用**：将多维张量展平为一维张量。

• **示例**：

```
tensor = torch.randn(2, 3, 4)
flattened_tensor = tensor.flatten()
print(flattened_tensor.shape)  # 输出: torch.Size([24])
```

  

• **注意**：也可以指定从哪个维度开始展平：

```
flattened_tensor = tensor.flatten(start_dim=1)
print(flattened_tensor.shape)  # 输出: torch.Size([2, 12])
```

  

  

**7. torch.reshape()**

• **作用**：与 view() 类似，重新调整张量的形状，但会返回一个新的张量。

• **示例**：

```
tensor = torch.randn(2, 3, 4)
reshaped_tensor = tensor.reshape(3, 8)
print(reshaped_tensor.shape)  # 输出: torch.Size([3, 8])
```

  

  

**8. torch.cat()**

• **作用**：沿着指定维度拼接两个或多个张量。

• **示例**：

```
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)
concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)  # 沿着第0维拼接
print(concatenated_tensor.shape)  # 输出: torch.Size([4, 3])
```

  

  

**9. torch.stack()**

• **作用**：沿着新的维度将多个张量堆叠起来。

• **示例**：

```
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(3, 4)
stacked_tensor = torch.stack((tensor1, tensor2), dim=0)  # 新维度放在第0维
print(stacked_tensor.shape)  # 输出: torch.Size([2, 3, 4])
```

  

  

**10. torch.index_select()**

• **作用**：选择指定维度上的元素。

• **示例**：

```
tensor = torch.randn(3, 5)
selected_tensor = torch.index_select(tensor, dim=1, index=torch.tensor([0, 2, 4]))
print(selected_tensor.shape)  # 输出: torch.Size([3, 3])
```

  

  

**11. torch.split()**

• **作用**：将张量沿指定维度分割成多个子张量。

• **示例**：

```
tensor = torch.randn(6, 4)
split_tensors = torch.split(tensor, 2, dim=0)  # 按第0维分割，每块大小为2
for split in split_tensors:
    print(split.shape)  # 输出: torch.Size([2, 4]) 对应的3次输出
```

  

  

**12. torch.chunk()**

• **作用**：将张量沿指定维度分成多个块。

• **示例**：

```
tensor = torch.randn(6, 4)
chunks = torch.chunk(tensor, 3, dim=0)  # 将张量沿第0维分为3块
for chunk in chunks:
    print(chunk.shape)  # 输出: torch.Size([2, 4]) 对应的3次输出
```

  

  

**13. torch.broadcast_tensors()**

• **作用**：使多个张量广播到相同的形状。

• **示例**：

```
tensor1 = torch.randn(3, 1)
tensor2 = torch.randn(3, 4)
broadcasted_tensors = torch.broadcast_tensors(tensor1, tensor2)
print(broadcasted_tensors[0].shape)  # 输出: torch.Size([3, 4])
```

  

  

**总结**

• **维度变化方法**：squeeze(), unsqueeze(), view(), reshape(), flatten(), permute(), transpose()

• **拼接与堆叠方法**：cat(), stack()

• **选择与分割方法**：index_select(), split(), chunk()

• **广播与调整**：broadcast_tensors()

  

通过这些方法，你可以灵活地处理 PyTorch 中张量的维度调整，满足不同的计算需求。