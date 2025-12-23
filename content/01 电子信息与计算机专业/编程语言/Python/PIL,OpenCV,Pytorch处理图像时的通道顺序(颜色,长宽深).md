---
aliases:
  - 颜色通道
---

|   项目    |      颜色通道顺序       | 长宽通道顺序 |  数据类型   |                 取值范围                 |
| :-----: | :---------------: | :----: | :-----: | :----------------------------------: |
|   PIL   |        RGB        |  HWC   | ndarray |             0-255 (byte)             |
| OpenCV  |        BGR        |  HWC   | ndarray |             0-255 (byte)             |
| PyTorch | RGB/BGR (取决于如何读取) | (N)CHW | tensor  | 0-1 (float, 标准化后); 0-255 (int, 未标准化) |

注意以下几点：

1. **颜色通道顺序**：PIL默认使用[RGB](https://so.csdn.net/so/search?q=RGB&spm=1001.2101.3001.7020)顺序，而OpenCV使用[BGR](https://so.csdn.net/so/search?q=BGR&spm=1001.2101.3001.7020)顺序。PyTorch不直接指定颜色通道顺序，它取决于你如何将图像数据加载到tensor中。如果你直接从PIL或OpenCV加载图像到PyTorch tensor，那么颜色通道顺序将保持不变（除非你进行了额外的转换）。
    
2. **长宽通道顺序**：PIL和OpenCV都使用HWC（高度、宽度、通道）顺序。PyTorch在处理图像数据时，通常期望的输入是CHW（通道、高度、宽度）顺序，特别是当使用卷积神经网络等模型时。但是，PyTorch的`torchvision.transforms`模块提供了`ToTensor()`等转换函数，可以自动将HWC顺序的PIL图像或NumPy数组转换为CHW顺序的tensor。此外，PyTorch还允许使用额外的维度N（批量大小）来扩展CHW到NCHW，这在处理批量图像时很常见。
    
3. **数据类型**：PIL和OpenCV都使用NumPy数组来存储图像数据，而PyTorch使用tensor。
    
4. **取值范围**：PIL和OpenCV中的图像数据通常以字节（byte）形式存储，取值范围为0-255。PyTorch中的tensor可以存储浮点数或整数，具体取决于你的设置。在大多数情况下，PyTorch期望输入图像的像素值被标准化到0-1的浮点数范围内，这是通过除以255来实现的。但是，如果你在处理分类任务等场景时，可能需要将图像数据保持为0-255的整数范围，并在模型中进行相应的调整。
    
5. **(N)HWC vs. (N)CHW**：我在PyTorch的“长宽通道顺序”列中添加了(N)来表示可能存在的批量大小维度。在PyTorch中，处理单个图像时通常使用CHW顺序，但在处理批量图像时，则使用NCHW顺序。然而，需要注意的是，这种约定主要适用于CUDA操作和某些特定的PyTorch层/函数（如`torch.nn.Conv2d`）。在大多数情况下，当你使用`torchvision.transforms`将PIL图像或NumPy数组转换为tensor时，你得到的是一个CHW顺序的tensor（除非你使用了特定的转换函数来改变这个顺序）。然后，如果你需要将tensor输入到支持批量处理的模型中，你可能需要手动添加一个额外的维度（即批量大小N）来形成NCHW顺序的tensor。但是，这通常是由PyTorch的数据加载器（如`torch.utils.data.DataLoader`）自动完成的。