---
title: "train 和 eval"
date: 2025-08-07
draft: false
---

在 PyTorch 中，model.train() 和 model.eval() 主要用于设置模型的 **训练模式** 和 **推理模式**，它们的作用如下：

  

**1. model.train()**

• 作用：将模型设置为训练模式，使得某些特定的层（如 **BatchNorm** 和 **Dropout**）在训练过程中表现出不同的行为。

• 影响：

• **BatchNorm**（批归一化）：在训练时会计算当前 mini-batch 的均值和方差，并用于标准化，同时更新全局均值和方差。

• **Dropout**（随机失活）：在训练时会随机丢弃一部分神经元，以防止过拟合。

  

**2. model.eval()**

• 作用：将模型设置为推理模式，冻结某些层的行为，使其在测试或推理时表现一致。

• 影响：

• **BatchNorm**：使用训练时累计的全局均值和方差，而不是 mini-batch 统计信息。

• **Dropout**：不会随机丢弃神经元，而是以全部神经元的加权输出参与计算。

  

**示例代码：**

```
import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(3)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        return x

# 创建模型
model = SimpleModel()
x = torch.randn(5, 3)  # 假设输入是 5x3 的张量

# 训练模式
model.train()
output_train = model(x)

# 推理模式
model.eval()
output_eval = model(x)

print("Train Mode Output:\n", output_train)
print("Eval Mode Output:\n", output_eval)
```

**关键区别：**

1. 在 train() 模式下：

• Dropout 会随机丢弃部分神经元，因此相同输入每次可能得到不同的结果。

• BatchNorm 会更新均值和方差。

2. 在 eval() 模式下：

• Dropout 失效，所有神经元都参与计算，输出稳定。

• BatchNorm 使用全局均值和方差，不再更新。

  

**总结：**

• **model.train()** 用于训练阶段，会启用 BatchNorm 的更新和 Dropout 的随机性。

• **model.eval()** 用于推理阶段，保证 BatchNorm 和 Dropout 具有确定性，提高模型在测试时的稳定性。

  

**⚠️ 注意：**

在推理时如果忘记 model.eval()，可能会导致 BatchNorm 仍然更新统计信息，Dropout 仍然随机丢失神经元，影响推理结果的稳定性。