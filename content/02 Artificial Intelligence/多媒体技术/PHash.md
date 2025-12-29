---
title: "**pHash 的数学原理和步骤：**"
date: 2025-09-15
draft: false
---

# **pHash 的数学原理和步骤：**

pHash 的原理基于图像的特征提取，它通过离散余弦变换（DCT）将图像转换到频域，并利用低频信息生成哈希值。低频部分代表图像的整体特征，而高频部分则是噪声和细节。pHash 通过舍弃高频信息，仅保留图像的整体特征，来生成感知哈希。

  

**主要步骤：**

1. **图像缩放**：将图像缩放到一个固定的较小尺寸（如 32x32），从而简化计算并保持图像的整体结构信息。这个步骤可以忽略图像的细节，保留大体轮廓。用小尺寸图像进行哈希计算，有助于去除噪声和图像的细微变化。
    
2. **转换为灰度图像**：将缩放后的图像转换为灰度图，这样就可以只处理亮度信息，而忽略颜色信息。灰度化减少了数据维度，同时保留了图像的结构和明暗关系。
    
3. **离散余弦变换（DCT）**：对灰度图像进行离散余弦变换（DCT）。DCT 是一种将图像从空间域转换为频域的方法，类似于傅里叶变换，但适用于离散数据。通过 DCT，可以将图像转换为一个频域矩阵，矩阵中包含了图像的低频和高频信息。
    
4. **保留低频部分**：由于低频信息代表了图像的主要特征，而高频信息通常代表噪声或细节，pHash 只保留 DCT 矩阵左上角的低频部分。通常会取前 8x8 的系数进行后续处理。
    
5. **计算 DCT 均值**：计算 DCT 矩阵的均值，忽略矩阵的第一个系数（直流分量，代表图像的整体亮度）。
    
6. **生成哈希值**：将每个 DCT 系数与均值进行比较。如果系数大于均值，则记为 1，否则记为 0。这样就可以将图像转换为一串 0 和 1，形成一个 64 位的二进制哈希值。
    

**pHash 的优点：**

- **鲁棒性**：pHash 对于一些常见的图像变换，如缩放、旋转、色彩变化等非常鲁棒，即这些变换不会大幅改变图像的哈希值。
    
- **高效性**：计算量相对较小，适合大规模图像相似性检测任务。
    
- **易于比较**：通过计算两个 pHash 哈希值的汉明距离，可以简单快速地比较图像的相似度。
    

# 用法

在 Python 中，可以使用 imagehash 库中的 phash 方法来实现 pHash。下面是示例代码：

```Python
from PIL import Image
import imagehash

# 加载图像
image1 = Image.open('image1.jpg')
image2 = Image.open('image2.jpg')

# 计算图像的感知哈希值
hash1 = imagehash.phash(image1)
hash2 = imagehash.phash(image2)

# 打印哈希值
print(f"Image 1 pHash: {hash1}")
print(f"Image 2 pHash: {hash2}")

# 计算两张图片的汉明距离
distance = hash1 - hash2
print(f"Hamming distance between images: {distance}")

# 判断图像相似性
if distance <= 10:
    print("Images are similar.")
else:
    print("Images are not similar.")
```

> 1. imagehash.phash：这是 imagehash 库提供的感知哈希（pHash）函数。它会对输入图像进行上述一系列步骤，生成图像的 pHash 值。
>     
> 2. **汉明距离（Hamming Distance）**：在哈希比较中，汉明距离用于衡量两个哈希值之间的相似度。两个哈希值的每个位进行对比，差异的位数就是汉明距离。距离越小，图像越相似。
>     
> 3. **相似性判断**：通常，pHash 的汉明距离小于某个阈值（如 10）时，可以认为图像是相似的。
>     

数据处理框架中，计算一个视频的phash，具体来说是**相邻视频帧之间感知哈希差异的平均值：**

```Python
phash_frames = video.clip(0,255).cpu().numpy().astype(np.uint8)
                    def calculate_phash(image):
                        return imagehash.phash(Image.fromarray(image), hash_size=16)
                    phashs = []
                    prev_phash = None
                    for frame in phash_frames:
                        self.logger.debug(f"frame shape:{frame.shape}")
                        current_phash = calculate_phash(frame)
                        phash_diff = current_phash - prev_phash if prev_phash is not None else 0
                        prev_phash = current_phash
                        phashs.append(phash_diff)
                    phash = sum(phashs) / len(phashs) #  视频帧之间感知哈希差异的平均值
```

# **总结**

- pHash 是一种基于图像感知的哈希算法，通过离散余弦变换提取图像的低频信息，并将其转换为哈希值。
    
- pHash 对图像的缩放、旋转、色彩变化等具有鲁棒性。
    
- Python 中可以使用 imagehash 库轻松实现 pHash，并通过汉明距离来衡量图像的相似性。