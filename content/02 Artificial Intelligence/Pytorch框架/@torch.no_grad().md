---
title: "@torch.no_grad()"
date: 2025-09-12
draft: false
---

`@torch.no_grad()` 是 PyTorch 中的一个装饰器，用于**禁用梯度计算**，主要作用包括：

---

### 1. **禁用自动求导（节省计算资源）**
- 在该装饰器修饰的代码块中，PyTorch 不会跟踪张量的操作（不构建计算图），从而：
  - **减少内存消耗**：不保存中间结果的梯度。
  - **加速计算**：避免不必要的梯度计算开销。

---

### 2. **适用场景**
- **模型推理（Inference）**：在测试或部署时，不需要计算梯度。
- **参数评估**：例如计算模型在验证集上的指标（如准确率）。
- **显式避免梯度更新**：例如在更新模型参数时冻结部分网络。

---

### 3. **与 `torch.inference_mode()` 的区别**
- `@torch.no_grad()` 仅禁用梯度计算，但仍允许修改张量的 `requires_grad` 属性。
- `@torch.inference_mode()`（PyTorch 1.9+）更严格，还会禁用部分 PyTorch 的调试机制，进一步优化推理速度。

---

### 4. **示例代码**
```python
import torch

@torch.no_grad()
def evaluate_model(model, data):
    outputs = model(data)  # 不会构建计算图
    return outputs.mean()  # 无梯度跟踪

# 等价的手动写法
with torch.no_grad():
    y = model(x)  # 此代码块内同样禁用梯度
```

---

### 5. **注意事项**
- 被装饰的函数内部所有操作均不受梯度影响。
- 若需临时恢复梯度计算，需结合 `torch.enable_grad()` 上下文管理器。

总结：**在不需要反向传播的场景下（如推理、评估），使用此装饰器可以显著提升性能**。