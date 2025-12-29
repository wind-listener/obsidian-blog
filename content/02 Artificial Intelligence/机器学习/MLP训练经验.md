---
title: "MLP训练经验"
date: 2025-08-07
draft: false
---

MLP 训练时，除了层数和大小，**优化器、学习率、批量大小、正则化**等超参数对训练效果也有很大影响。以下是一些推荐的超参数设置及调整建议：

  

**1. 学习率（lr）**

  

学习率决定了模型的更新步长，选择合适的学习率至关重要：

• **太大 (**1e-2 **以上)** → 训练不稳定，损失震荡或不收敛。

• **太小 (**1e-6 **以下)** → 训练速度慢，可能停滞在局部最优。

  

**推荐范围**

  

**任务类型** **推荐学习率**

小规模数据集（<10w 样本） 1e-3 ~ 5e-3

大规模数据集（>10w 样本） 1e-4 ~ 1e-3

预训练模型微调 1e-5 ~ 5e-5

  

**建议**

• **首选** 1e-3**，根据训练情况调整**：

  

lr: 1e-3

  

  

• **使用学习率调度器**，如 ReduceLROnPlateau（验证损失不下降时降低学习率）：

  

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

  

**2. 批量大小（batch_size）**

• **太小（≤16）** → 训练不稳定，梯度噪声大。

• **太大（≥256）** → 收敛慢，需要更高学习率。

  

**推荐范围**

  

**任务** **推荐** batch_size

小数据集 32 ~ 64

大数据集 64 ~ 256

受 GPU 显存限制 16 ~ 128

  

**建议**

• **默认** 32 **或** 64**，GPU 资源允许时尽量加大**：

  

batch_size: 64

  

**3. 优化器（optimizer）**

  

优化器影响梯度更新方式：

  

**优化器** **适用场景** **推荐**

SGD 需要手动调节学习率，适用于简单任务 ❌

SGD + momentum 适用于小批量 SGD（如 batch_size < 32） ✅

Adam 适用于大多数任务，训练稳定 ✅✅

AdamW Adam 的改进版本，适用于 Transformer 结构 ✅✅

RMSprop 适用于递归神经网络（RNN） ❌

  

**建议**

• 一般任务，优先选择 Adam **或** AdamW：

  

optimizer: "Adam"

  

  

• 如果 Adam 过拟合严重，可以尝试 SGD + momentum：

  

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

  

**4. 损失函数（loss）**

  

损失函数决定了训练的目标：

  

**任务** **推荐损失函数**

二分类 BCELoss（sigmoid）

多分类 CrossEntropyLoss（softmax）

回归 MSELoss

置信度估计 HuberLoss

  

**建议**

• 二分类问题（如 0/1 标签），使用 BCELoss：

  

loss: "BCELoss"

  

  

• 如果数据存在异常点，可以用 SmoothL1Loss：

  

loss: "SmoothL1Loss"

  

**5. 正则化（防止过拟合）**

  

**(1) Dropout**

• **默认** 0.2 ~ 0.5，防止过拟合：

  

dropout: 0.3

  

  

• 层数较少（<3）时，dropout=0.1~0.2，层数较多时 0.3~0.5。

  

**(2) 权重衰减（L2 正则化）**

• **防止权重过大，增加泛化能力**：

  

weight_decay: 1e-4

  

  

• 推荐 1e-5 ~ 1e-4，搭配 AdamW 效果更佳。

  

**(3) 批归一化（BatchNorm）**

• **防止梯度消失，加快训练收敛**：

  

nn.BatchNorm1d(hidden_size)

  

  

• **通常在** ReLU **之前使用**：

  

nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3))

  

**6. 训练周期（epochs）**

• **太短（<20 轮）** → 可能欠拟合

• **太长（>500 轮）** → 可能过拟合

• **推荐** 50~200 轮，结合 Early Stopping ：

  

epochs: 100

  

**Early Stopping**

• 如果 validation loss 多轮不下降，提前停止训练：

  

from torch.optim.lr_scheduler import ReduceLROnPlateau

  

early_stopping = EarlyStopping(patience=10, delta=0.001)

  

**7. 训练日志记录（W&B 监控）**

  

为了方便分析训练曲线，建议记录：

  

wandb.log({

    "Train Loss": train_loss / len(train_loader),

    "Train Accuracy": train_accuracy,

    "Test Loss": test_loss / len(test_loader),

    "Test Accuracy": test_accuracy,

    "Learning Rate": optimizer.param_groups[0]["lr"]

})

  

可以在 W&B 界面观察：

• **学习率曲线**（确定 lr 是否合适）

• **Train/Test Loss 对比**（检测过拟合）

• **Test Accuracy 变化**（确定 epochs 是否合适）

  

**8. 参考 YAML 配置**

  

batch_size: 64

epochs: 100

lr: 1e-3

hidden_sizes: [512, 256, 128, 32]

dropout: 0.3

optimizer: "AdamW"

loss: "BCELoss"

weight_decay: 1e-4

save_interval: 20

  

**9. 关键超参数调整建议**

  

**现象** **可能原因** **解决方案**

训练损失一直不降 学习率太大/太小 lr 调整为 1e-3 ~ 1e-4

训练损失下降但测试损失上升 过拟合 增加 Dropout=0.3，加大 weight_decay

训练慢，效果一般 模型太小 增加隐藏层，如 [1024, 512, 256, 128, 32]

训练快但效果不好 模型太大，欠拟合 降低层数，如 [512, 128, 32]

训练到后期效果变差 过训练 加入 Early Stopping，减少 epochs

  

**总结**

• lr=1e-3 ~ 1e-4，训练稳定，避免过大/过小。

• batch_size=32~128，小数据用 32，大数据 64~256。

• AdamW **+** weight_decay=1e-4，防止过拟合。

• Dropout=0.2~0.5，层数多时增加。

• BatchNorm **防止梯度消失**。

• Early Stopping + ReduceLROnPlateau，自动调节学习率。

  

按照这些调整，可以优化你的 MLP 训练效果，让模型更稳定、高效！🚀