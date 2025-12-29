---
title: "SAM"
date: 2025-08-07
draft: false
---

LLM的成功显示了基座大模型的强大泛用能力，只需要少量的prompt engineering即可实现媲美于fine-tune得到的模型，在zero-shot的任务上具有良好甚至SOTA的表现。

在CV领域能否复刻大模型的成功？
The success of this plan hinges on three components: **task, model, and data**. To develop them, we address the following questions about image segmentation: 
1. What task will enable zero-shot generalization? 
2. What is the corresponding model architecture? 
3. What data can power this task and model?
# 

### Segment Anything项目总结

#### 1. 项目简介

Segment Anything（SA）项目旨在创建一个基础模型，用于图像分割。该项目包括三个主要组件：一个可提示的分割任务、一个分割模型（SAM）和一个数据引擎/一种数据集的构建和扩展（SA-1B）。这些组件协同工作，旨在实现图像分割任务的广泛泛化能力。

#### 2. 任务和模型

**任务：**
- 可提示的分割任务：给定任何分割提示，生成一个有效的分割掩码。提示可以是前景/背景点、粗略的框或掩码、自由形式的文本等。

**模型：**
- Segment Anything Model (SAM) 由图像编码器、提示编码器和快速掩码解码器组成。图像编码器生成图像嵌入，提示编码器将提示嵌入，掩码解码器结合这两者生成分割掩码。
![[Pasted image 20240618144917.png]]
	1. Image encoder：an MAE [47] pre-trained Vision Transformer (ViT) [33] minimally adapted to process high resolution input
	2. 


- SAM支持灵活的提示，并能实时生成掩码，允许交互式使用。它设计为能够处理多义性，一个提示可以生成多个有效掩码。
- 

#### 3. 数据引擎和数据集

**数据引擎：**
- 数据引擎有三个阶段：辅助手动标注、半自动标注和全自动标注。在这些阶段中，SAM逐步改进，通过模型在环数据标注循环中收集和生成数据。
	- 1. **Assisted-manual stage**: 先用公开的数据集训练出一个SAM，给出mask然后人工可以brush/eraser修改，建议每张图片处理不超过30s，主要是显眼的物体。据统计，每个mask需要14s，4.3M masks from 120k images in this stage，retrained 6 times。
	- 2. **Semi-automatic stage**：目的在于提升diversity，使用FastRCNN作为a bounding box detector on all first stage masks using a generic “object” category. 据统计，additional 5.9M masks in 180k images (for a total of 10.2M masks). retrained 5 times
	- 3. **Fully automatic stage**： 使用一个32✖️32的点状网格，自动生成mask，然后select the *confident and stable*(有具体标准) masks, we applied non-maximal suppression (NMS) to filter duplicates. 还要去除重叠。最后，11M images in our dataset, 1.1B high-quality masks.
- 

**数据集：**
- SA-1B数据集包含11M张隐私保护的高分辨率图像和1.1B高质量分割掩码，是迄今为止最大的分割数据集。SA-1B的掩码由SAM全自动生成，并经过人类标注者验证，质量和多样性较高。

#### 4. 评估与实验

**评估：**
- SAM在23个不同的数据集上进行了评估，结果显示其零样本性能非常出色，往往与或优于先前的完全监督模型。SAM在多种下游任务中表现优异，包括边缘检测、对象提议生成、实例分割等。

**实验结果：**
- 在单点有效掩码评估中，SAM在许多数据集上表现优于最强的基准方法。
- 在零样本边缘检测、对象提议生成和实例分割任务中，SAM的性能也非常出色，尽管有时不及专门训练的模型，但其多样性和广泛的泛化能力使其在多个领域具有应用潜力。

#### 5. 消融实验的意义
1. **组件重要性评估**：通过逐一移除或替换模型的组件，研究者可以确定哪些部分对模型性能最为关键。这有助于优化模型结构，使其更加高效。
    
2. **数据集依赖分析**：了解不同数据量和数据质量对模型训练的影响，可以帮助研究者设计更高效的数据收集和标注策略，减少不必要的数据标注成本。
    
3. **模型优化**：消融实验的结果可以为模型的进一步优化提供方向，例如在计算资源有限的情况下，如何调整模型以获得最佳性能。

#### 6. 结论与发布

Segment Anything项目通过引入新的任务、模型和数据集，将图像分割提升到基础模型时代。该项目的贡献包括一个新的可提示分割任务、Segment Anything Model (SAM) 和 SA-1B 数据集。这些资源的公开发布将促进计算机视觉基础模型的研究和发展。SAM和SA-1B数据集在 [segment-anything.com](https://segment-anything.com) 上开放获取。

通过这些组件和数据，Segment Anything项目为未来的研究和应用奠定了坚实基础，展示了基础模型在计算机视觉中的广泛应用潜力。