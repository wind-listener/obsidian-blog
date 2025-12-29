---
title: "EfficientSAM"
date: 2025-08-07
draft: false
---

### EfficientSAM: 利用掩码图像预训练实现高效的Segment Anything模型

#### 1. 引言
EfficientSAM 是一个轻量级的 Segment Anything Model (SAM)，旨在大幅降低计算复杂度，同时保持良好的性能。该模型利用了掩码图像预训练技术（SAMI），从而有效地学习视觉表示。
> 个人理解：具体方法是通过MAE，来预训练出一个encoder，再加上SAM的decoder，最后实现分割任务同时减少参数量。

#### 2. 方法

**SAMI 预训练**
- SAMI 是一种基于掩码自编码器 (MAE) 的预训练方法，通过重建 SAM 图像编码器 (ViT-H) 的特征来进行视觉表示学习。
- SAMI 使用轻量级的 ViT 编码器（如 ViT-Tiny 和 ViT-Small），通过掩码图像模型训练这些编码器，使其能够重建来自 ViT-H 的特征，而不是重建图像像素。

**EfficientSAM**
- EfficientSAM 使用经过 SAMI 预训练的轻量级编码器和 SAM 的掩码解码器，通过在 SA-1B 数据集上微调，构建高效的 SAM 模型。
- 该模型大幅减少了参数量和推理时间，但在多个视觉任务中的表现依然优异。

#### 3. 实验

**预训练设置**
- 使用 ImageNet-1K 数据集进行预训练，不使用标签信息。采用 MAE 的高掩码比例 (75%) 进行训练。

**下游任务评估**
- 图像分类：在 ImageNet-1K 数据集上，使用 SAMI 预训练的 ViT 模型进行微调，并与其他预训练方法进行比较。结果显示，SAMI 在 ViT-Tiny、ViT-Small 和 ViT-Base 上都取得了显著的性能提升。
- 目标检测和实例分割：在 COCO 数据集上，使用 ViTDet 框架，将预训练的 ViT 编码器应用于目标检测和实例分割任务，表现优于其他基线方法。
- 语义分割：在 ADE20K 数据集上，使用 Mask2former 框架，SAMI 预训练的 ViT 编码器在 mIoU 上有显著提升。
- Segment Anything 任务：在 COCO 和 LVIS 数据集上进行零样本实例分割评估，EfficientSAM 在性能上超越了 MobileSAM 和 FastSAM。

**消融研究**
- **重建损失**：使用均方误差 (MSE) 重建损失比余弦相似性损失效果更好。
- **掩码比例**：75% 的掩码比例在性能上表现最佳，与 MAE 的观察结果一致。
- **微调步数**：微调步数的增加显著提升了模型的性能，特别是在1个 epoch 后性能提升较大。

#### 4. 结论

EfficientSAM 提出了一个基于掩码图像预训练的高效模型，能够在降低复杂度的同时，保持优异的性能。通过广泛的实验验证，SAMI 预训练方法在多个下游任务中表现出色，显示了其广泛的应用潜力。

#### 参考文献
- Xiong 等，2023，《EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything》【28†source】