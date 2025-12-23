**TResNet（Tightly-coupled Residual Network）[TResNet](https://github.com/Alibaba-MIIL/TResNet)**

  

TResNet 是 **Alibaba DAMO Academy** 提出的一个高效的图像分类网络，旨在 **提高ResNet的计算效率，同时保持高精度**。相比于标准 ResNet，TResNet 采用了一系列优化策略，使其在相同 FLOPs（计算量）下取得更好的性能。

**1. 主要创新点**

  

TResNet 主要通过以下技术优化标准 ResNet：

  

**1.1 Anti-Aliasing Downsampling（反混叠降采样）**

• 传统 CNN 网络的池化层容易丢失信息，导致特征图变得粗糙，影响分类性能。

• TResNet 采用了 **Anti-Aliasing Filters** 进行平滑降采样，保留更多低频信息，使得特征图更加平滑，提高模型的泛化能力。

  

**1.2 SpaceToDepth 模块**

• SpaceToDepth 通过 **在模型的输入端增加通道数，并减少空间分辨率**，以此加速模型计算，同时减少内存占用。

• 这个操作相当于在输入时进行等效的降采样，提高计算效率。

  

**1.3 In-Place Activated BatchNorm (Inplace-ABN)**

• Inplace-ABN 结合了 **Batch Normalization（BN）和 Activation Function（ReLU or Swish）**，减少了额外的内存消耗，提升计算效率。

• 在训练过程中 **减少了近 50% 的显存占用**，特别适用于大规模训练任务。

  

**1.4 Optimized Stem Block**

• ResNet 的 Stem 部分（最开始的卷积层）包含 **7x7 大卷积核 + MaxPooling**，计算开销较大。

• TResNet 用 **多个小卷积核（3×3）堆叠** 来代替大卷积核，并使用更高效的降采样策略，提高计算效率。

  

**1.5 SE-ResNet 结构**

• TResNet 采用 **SE（Squeeze-and-Excitation）模块** 来增强重要特征的表达能力。

• 通过全局池化的方式学习通道注意力权重，使得模型在计算开销较低的情况下提高准确率。

**2. 网络结构**

  

TResNet 主要基于 ResNet-50 及其变种进行优化，形成 **TResNet-M / L / XL** 这几个不同规模的版本，结构如下：

|**Model**|**Params (M)**|**FLOPs (B)**|**Top-1 Accuracy (%)**|
|---|---|---|---|
|ResNet-50|25.6|4.1|76.0|
|TResNet-M|30.9|4.3|80.7|
|TResNet-L|65.1|8.4|82.1|
|TResNet-XL|88.0|12.4|83.2|

• **TResNet-M**: 轻量级版本，在 FLOPs 只增加 5% 的情况下，提升 4.7% 的 Top-1 精度。

• **TResNet-L / XL**: 适用于更高精度任务，在 ImageNet 上的 Top-1 精度超越 EfficientNet 和 ResNeXt。

**3. 代码实现**

  

TResNet 的 PyTorch 代码如下：

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SpaceToDepth, InplaceAbn

class TResNetBlock(nn.Module):
    """基本残差块，带有SE模块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(TResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = InplaceAbn(out_channels)  # 采用 Inplace-ABN
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = InplaceAbn(out_channels)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                InplaceAbn(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        se_weight = self.se(out)
        out = out * se_weight
        out += self.shortcut(x)
        return F.relu(out)

class TResNet(nn.Module):
    """TResNet 结构"""
    def __init__(self, num_classes=1000):
        super(TResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            InplaceAbn(64),
            SpaceToDepth(2)  # SpaceToDepth 处理
        )
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        self.layer4 = self._make_layer(512, 1024, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            TResNetBlock(in_channels, out_channels, stride),
            TResNetBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# 测试模型
model = TResNet(num_classes=1000)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # 计算参数量
```

**4. 性能比较**

  

TResNet 在 ImageNet 上的表现优于 ResNet，并且在高效计算下仍保持良好的精度：

|**Model**|**Params (M)**|**FLOPs (B)**|**Top-1 Acc (%)**|**Top-5 Acc (%)**|
|---|---|---|---|---|
|ResNet-50|25.6|4.1|76.0|92.9|
|ResNet-101|44.5|7.8|77.4|93.5|
|TResNet-M|30.9|4.3|80.7|95.2|
|TResNet-L|65.1|8.4|82.1|95.9|
|TResNet-XL|88.0|12.4|83.2|96.4|

• **TResNet-M 在参数量比 ResNet-50 多 20% 的情况下，Top-1 提升 4.7%**。

• **TResNet-XL 进一步优化，精度可达 83.2%**，接近 EfficientNet 但计算量更小。

**5. 适用场景**

• **高效推理**：适用于 GPU/TPU 计算优化，提升吞吐量。

• **大规模图像分类**：适用于 ImageNet、Fine-grained 分类任务。

• **实时应用**：适合移动端或云端部署，计算量小但性能强。

**6. 结论**

  

TResNet 通过一系列优化（反混叠降采样、SpaceToDepth、Inplace-ABN等），在保持高效计算的同时提升了 ResNet 的精度，是一个极具应用价值的 CNN 结构。