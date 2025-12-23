---
aliases:
  - 3D Convolution
  - 3D 卷积
---
#卷积 #cnn #conv

3D 卷积（3D Convolution）是一种 **针对三维数据（如视频、医学影像、3D 点云等）** 的卷积操作，它扩展了 2D 卷积的概念，在 **空间（Height, Width）和时间（Depth）三个维度上同时进行特征提取**。

# **3D 卷积的基本概念**
在 **标准的 2D 卷积** 中，卷积核在 **高度 (H) 和 宽度 (W)** 维度上滑动，而在 **3D 卷积** 中，卷积核会在 **高度 (H)、宽度 (W) 和深度 (D)** 这三个维度上滑动。其数学定义如下：
  

$$

\text{Output}(x, y, z) = \sum_{i=0}^{K_H-1} \sum_{j=0}^{K_W-1} \sum_{k=0}^{K_D-1} W(i, j, k) \cdot X(x+i, y+j, z+k)

$$
其中：

• $X$ 是输入数据，形状为 $(C_{\text{in}}, H, W, D)$。

• $W$ 是 3D 卷积核，形状为 $(C_{\text{out}}, C_{\text{in}}, K_H, K_W, K_D)$。

• $H, W, D$ 分别是输入数据的高度、宽度和深度。

• $K_H, K_W, K_D$ 是卷积核的大小。

• $C_{\text{in}}, C_{\text{out}}$ 分别是输入和输出的通道数。

3D 卷积会生成一个新的三维特征图，既保留了空间信息，也能在深度方向（通常是时间维度）上提取特征。

# **3D 卷积的应用场景**

## **视频分析**

• **动作识别（Action Recognition）**：如 Kinetics, UCF101 等数据集中的视频动作分类任务。

• **视频分割（Video Segmentation）**：分析视频中的场景变化。

• **运动目标检测（Motion Detection）**：在视频帧之间检测对象的运动。

  

## **医学影像处理**

• **MRI / CT 影像分析**：3D 卷积能在 3D 体积数据上提取特征，例如 **肺部病变检测、脑部肿瘤识别**。

• **3D 分割（3D Medical Segmentation）**：如 UNet3D 在医学图像上的应用。

  

## **3D 物体识别**

• **点云处理（Point Cloud Processing）**：3D-CNN 可用于 3D 物体检测，如 **自动驾驶中的 LiDAR 点云数据处理**。

• **3D 建模和重建**：如 **3D GAN（Voxel-based）** 生成 3D 物体。

# **3D 卷积 vs. 2D 卷积**

|**对比项**|**2D 卷积（2D-CNN）**|**3D 卷积（3D-CNN）**|
|---|---|---|
|适用数据|单张图片、单帧数据|视频、医学影像、点云|
|输入形状|$(C, H, W)$|$(C, H, W, D)$|
|计算复杂度|低|高|
|特征提取|只能从 2D 空间获取信息|同时在空间和时间维度提取信息|
|典型应用|图像分类、目标检测|视频分析、医学影像、3D 物体识别|

**4. 3D 卷积的计算过程**

  

假设我们有一个 **输入体积 (Depth × Height × Width)**，使用一个 3D 卷积核计算输出：

  

**示例**

• **输入：** $(3, 32, 32, 16)$（3 通道，32×32 空间大小，16 帧）

• **3D 卷积核：** $(64, 3, 3, 3, 3)$（64 个输出通道，每个通道 3×3×3）

• **步幅 (stride)：** $(1,1,1)$

• **填充 (padding)：** $(1,1,1)$（保持大小）

  

**输出计算：**

• 每个 3D 滤波器会在 **时间、空间方向** 进行滑动计算。

• 输出大小可以用公式计算：

  

$$

\text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1

$$

  

例如：

• 在 **空间方向 (H, W)**：

$$

\frac{32 - 3 + 2 \times 1}{1} + 1 = 32

$$

• 在 **时间方向 (D)**：

$$

\frac{16 - 3 + 2 \times 1}{1} + 1 = 16

$$

  

最终输出尺寸为 **$(64, 32, 32, 16)$**。

# 3D 卷积的代码实现

## PyTorch 实现

```python
import torch
import torch.nn as nn

class Simple3DConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Simple3DConvNet, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3d(x)  # 3D 卷积
        x = self.relu(x)     # 激活函数
        x = self.pool3d(x)   # 3D 池化
        return x

# 创建输入张量 [batch_size, channels, depth, height, width]
input_tensor = torch.randn(1, 3, 16, 32, 32)
model = Simple3DConvNet()
output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")  # (1, 3, 16, 32, 32)
print(f"Output shape: {output.shape}")  # (1, 64, 8, 16, 16)
```

# 3D-CNN 应用于视频分类

```python
class VideoClassifier3D(nn.Module):
    def __init__(self, num_classes=10):
        super(VideoClassifier3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 测试
input_video = torch.randn(1, 3, 16, 32, 32)  # 1 个视频，16 帧
model = VideoClassifier3D(num_classes=10)
output = model(input_video)
print(output.shape)  # (1, 10)
```

# 3D 卷积的改进版本

• **(2+1)D 卷积**：将 3D 卷积分解为 **2D 空间卷积 + 1D 时间卷积**，减少计算量（如 R(2+1)D 网络）。

• **3D 深度可分离卷积**：减少参数，提高效率（如 MobileNet3D）。

• **注意力机制结合**：如 SE3D、CBAM3D，提高通道和空间的特征提取能力。

# 总结

  

3D 卷积适用于 **时序信息（视频）和 3D 体积数据（医学影像、点云）**，可以高效提取空间和时间特征，但计算成本高。改进方案如 **(2+1)D 卷积、可分离 3D-CNN** 可降低计算量，提高实用性。