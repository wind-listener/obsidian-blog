---
title: "在一个batch内网络参数是如何优化的？"
date: 2025-08-07
draft: false
---

在 **一个 batch 内，网络参数的优化过程** 主要涉及 **前向传播（Forward Pass）、损失计算（Loss Computation）、反向传播（Backward Pass）和参数更新（Parameter Update）**，具体流程如下：

---

**1. 计算图构建 & 前向传播**

  

在 PyTorch 或 TensorFlow 等深度学习框架中，**计算图（Computational Graph）** 被自动构建。

  

对于一个小批量（batch）的输入数据：

1. 输入 batch 数据 X 通过神经网络 f_{\theta} 进行**前向传播**：

$$

Y_{\text{pred}} = f_{\theta}(X)

$$

其中：

• X 是输入数据（形状：(batch_size, input_dim)）。

• Y_{\text{pred}} 是模型的预测值。

2. 计算 **损失函数（Loss Function）**，衡量预测值 Y_{\text{pred}} 与真实标签 Y_{\text{true}} 之间的差异：

$$

\mathcal{L} = \text{Loss}(Y_{\text{pred}}, Y_{\text{true}})

$$

例如：

• **分类任务**：交叉熵损失（Cross Entropy Loss）

• **回归任务**：均方误差（MSE Loss）

---

**2. 计算梯度（反向传播 - Backward Pass）**

  

在 **前向传播过程中**，PyTorch 记录所有操作，形成计算图。然后，**损失函数对参数的梯度** 通过 **反向传播（Backpropagation）** 计算：

1. **对损失函数求梯度**（使用 **自动微分 Autograd** 计算损失对参数的偏导数）：

$$

\frac{\partial \mathcal{L}}{\partial \theta} = \nabla_{\theta} \mathcal{L}

$$

这里：

• \theta 代表所有网络参数（如权重和偏置）。

• 计算每个参数对损失的影响。

2. **梯度存储**：

• 在 PyTorch 中，每个参数 \theta 都有 tensor.grad 属性，用于存储梯度。

```
loss.backward()  # 计算所有参数的梯度
```

  

---

**3. 参数更新（Optimizer Step）**

  

一旦获得了梯度，优化器（如 **SGD、Adam**）使用这些梯度 **更新参数**。

  

**梯度下降（Gradient Descent）更新公式**

  

对于每个参数 \theta，更新规则如下：

$$

\theta = \theta - \eta \cdot \nabla_{\theta} \mathcal{L}

$$

其中：

• \eta 是学习率（learning rate）。

• \nabla_{\theta} \mathcal{L} 是当前 batch 计算出的梯度。

  

在 PyTorch 中：

```
optimizer.step()  # 更新参数
```

**优化器示例**

```
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降（SGD）

optimizer.zero_grad()  # 清除上一轮梯度，防止梯度累积
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数
```

  

---

**4. 处理梯度累积**

  

在 PyTorch 默认行为下，梯度是 **累积** 的，因此在每个 batch 计算完梯度后，需要手动清零：

```
optimizer.zero_grad()
```

否则，梯度会在多个 batch 之间累积，影响训练效果。

---

**5. 总结**

  

在 **一个 batch 内**，参数优化的完整流程如下：

1. **前向传播**：

• 计算模型输出 Y_{\text{pred}} = f_{\theta}(X)。

• 计算损失 \mathcal{L}。

2. **反向传播**：

• 计算梯度 \nabla_{\theta} \mathcal{L}（loss.backward()）。

3. **参数更新**：

• 使用优化器（如 Adam、SGD）更新权重（optimizer.step()）。

4. **梯度清零**：

• 运行 optimizer.zero_grad() 以防止梯度累积。

  

这个流程在 **每个 batch 内执行一次**，整个训练过程中不断迭代优化参数，使得模型逐步逼近最优解 🎯。