#cbam #attention #cnn 

  

> CBAM（Convolutional Block Attention Module）是一种轻量级但有效的注意力机制，由 **通道注意力（Channel Attention）** 和 **空间注意力（Spatial Attention）** 两部分组成。它可以嵌入到卷积神经网络（CNN）中，以增强特征提取能力，提高图像分类、目标检测和语义分割等任务的性能。
> 
> [论文](https://arxiv.org/abs/1807.06521) [Non-official代码仓库](https://github.com/luuuyi/CBAM.PyTorch)


# CBAM 的整体结构
CBAM 由两个串联的子模块构成：

1. **通道注意力（Channel Attention Module, CAM）**

2. **空间注意力（Spatial Attention Module, SAM）**
![[CBAM结构.png]]

在 CBAM 中，输入特征图 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ 先经过 **通道注意力模块**，然后通过 **空间注意力模块**，最终得到增强的特征表示。
**计算流程：**

$$

\mathbf{F’} = \text{CAM}(\mathbf{F}) \cdot \mathbf{F}

$$

$$

\mathbf{F’’} = \text{SAM}(\mathbf{F’}) \cdot \mathbf{F’}

$$

  

其中：

• $\mathbf{F}$ 是输入特征图

• $\mathbf{F’}$ 是经过通道注意力模块后的特征图

• $\mathbf{F’’}$ 是最终输出的特征图

• CAM(·) 和 SAM(·) 分别表示通道注意力和空间注意力

---

**2. 通道注意力机制（Channel Attention）**

  

通道注意力模块用于**捕捉全局通道关系**，通过自适应地调整各通道的重要性，来提高网络对关键通道的关注。

  

**计算步骤**

1. **全局平均池化（GAP, Global Average Pooling）** 和 **全局最大池化（GMP, Global Max Pooling）** 计算通道维度上的全局特征：

$$

\mathbf{F}_{\text{avg}}^c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}(i,j)

$$

$$

\mathbf{F}_{\text{max}}^c = \max_{i \in H, j \in W} \mathbf{F}(i,j)

$$

其中，$\mathbf{F}_{\text{avg}}^c, \mathbf{F}_{\text{max}}^c \in \mathbb{R}^{C}$。

2. **共享 MLP（Multi-Layer Perceptron）** 进行特征融合：

• MLP 由两层全连接层组成：

$$

MLP(x) = W_2 \sigma(W_1 x)

$$

• 计算两个池化特征的注意力权重：

$$

\mathbf{M}_c = \sigma(\text{MLP}(\mathbf{F}_{\text{avg}}) + \text{MLP}(\mathbf{F}_{\text{max}}))

$$

3. **Sigmoid 归一化** 并与原特征相乘：

$$

\mathbf{F’} = \mathbf{M}_c \cdot \mathbf{F}

$$

  

**通道注意力结构**

• 输入：$\mathbf{F} \in \mathbb{R}^{C \times H \times W}$

• 经过 GAP 和 GMP，得到两个通道描述

• 经过共享 MLP 并相加

• 通过 Sigmoid 激活

• 乘回原特征 $\mathbf{F}$

---

**3. 空间注意力机制（Spatial Attention）**

  

通道注意力增强了通道之间的相关性，而**空间注意力关注的是每个位置的重要性**，即哪些区域更重要。

  

**计算步骤**

1. **通道维度的最大池化和平均池化**，在空间维度上进行：

• 对特征图的通道维度做最大池化和平均池化：

$$

\mathbf{F}_{\text{avg}}^s = \frac{1}{C} \sum_{k=1}^{C} \mathbf{F}_k_

_$$_

_$$_

_\mathbf{F}_{\text{max}}^s = \max_{k \in C} \mathbf{F}_k

$$

• 其中，$\mathbf{F}_{\text{avg}}^s, \mathbf{F}_{\text{max}}^s \in \mathbb{R}^{1 \times H \times W}$。

2. **将两个特征拼接（Concat），再通过一个 $7 \times 7$ 的卷积层进行特征提取**：

$$

\mathbf{M}_s = \sigma(\text{Conv}_{7 \times 7}([\mathbf{F}_{\text{avg}}^s; \mathbf{F}_{\text{max}}^s]))

$$

3. **归一化并乘回原特征**：

$$

\mathbf{F’’} = \mathbf{M}_s \cdot \mathbf{F’}

$$

  

**空间注意力结构**

• 输入：$\mathbf{F’} \in \mathbb{R}^{C \times H \times W}$

• 经过通道方向的 GAP 和 GMP

• 经过 $7 \times 7$ 卷积

• 通过 Sigmoid 激活

• 乘回输入特征 $\mathbf{F’}$

---

**4. CBAM 的优点**

1. **轻量级**：相比于 SE（Squeeze-and-Excitation）模块，CBAM 计算量更小。

2. **更强的特征表达能力**：同时利用了通道注意力和空间注意力，更全面地增强关键特征。

3. **容易嵌入 CNN 结构**：可以无缝集成到主流 CNN，如 ResNet、VGG 等，提高分类、检测等任务的性能。

---

**5. PyTorch 实现 CBAM**

```python
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).view(x.shape[0], -1))
        max_out = self.mlp(self.max_pool(x).view(x.shape[0], -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.shape[0], x.shape[1], 1, 1) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```

  

---

**6. 总结**

• **CBAM = 通道注意力 + 空间注意力**

• 先执行通道注意力，再执行空间注意力

• 轻量、高效，适用于各种 CNN 任务

---

以上就是 **CBAM（卷积块注意力模块）** 的详细介绍及 PyTorch 实现 🚀。