---
aliases:
  - Diffusion Transformers
---
DiT即Diffusion Transformer，是一种基于Transformer架构的扩散模型，用于图像和视频等视觉生成任务。以下是具体介绍：
- **核心思想**：DiT将Transformer作为扩散模型的骨干网络，替代传统的卷积神经网络（如U-Net），结合去噪扩散概率模型（DDPM），通过模拟扩散过程逐步添加噪声到数据中，再学习反转该过程，从噪声中构建出所需的数据样本。
- **架构与组件**：DiT架构基于Latent Diffusion Model（LDM）框架，采用Vision Transformer（ViT）作为主干网络。其核心组件包括上下文条件（In-context conditioning）、交叉注意力块（Cross-Attention）和自适应层归一化块（AdaLN）。模型有三种变种形式，分别与In-Context Conditioning、Cross-Attention、adaLN-Zero相组合。
- **工作流程**：首先使用预训练的变分自编码器（VAE）将输入图像编码成潜在空间的表示，并分块化为Transformer模型的输入标记。然后标记序列通过一系列Transformer块进行处理。在训练过程中，DiT模型学习逆向扩散过程，即从噪声数据中恢复出清晰的图像。训练完成后，从标准正态分布中采样一个潜在表示，通过DiT模型逆向扩散过程，逐步去除噪声，最终解码回像素空间，得到生成的图像。
- **与传统扩散模型区别**：传统扩散模型常以U-Net为骨干，DiT则完全替换为Transformer架构，能更高效地捕获数据中的依赖关系。此外，DiT扩散过程采用简单的Linear scheduler，而传统U-Net扩散模型通常采用Scaled Linear scheduler。
- **优势与性能**：DiT验证了Transformer架构在扩散模型上具备较强的Scaling能力，随着模型参数量增大和数据质量增强，其生成性能稳步提升。例如最大的DiT-XL/2模型在ImageNet 256x256的类别条件生成上达到了当时的SOTA性能，FID为2.27。
- **应用领域**：DiT可用于各种图像（如SD3、FLUX等）和视频（如Sora等）视觉生成任务，是AIGC时代图像和视频生成领域的重要模型。

> 融合扩散模型与Transformer的架构创新，重塑图像与视频生成范式

## 概述与核心概念
**Diffusion Transformers（DiT）** 是一种将Transformer架构与扩散模型相结合的生成式模型架构。它通过**替换传统扩散模型中的U-Net主干**，利用Transformer的全局建模能力和卓越扩展性，显著提升了图像与视频生成的质量和效率。这一架构创新由Peebles与Xie在论文《Scalable Diffusion Models with Transformers》中系统提出，并迅速成为OpenAI的Sora等前沿生成模型的核心基础。

传统扩散模型的核心是**去噪扩散概率模型（DDPM）**，其数学本质是通过马尔可夫链实现数据加噪与去噪：
- **正向过程**：逐步添加高斯噪声
$$ q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1},\mathbf{0},\beta_t \mathbf{I}) $$
- **逆向过程**：学习噪声预测以重建数据
$$ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t,t),\Sigma_\theta(\mathbf{x}_t,t)) $$

DiT的创新在于**将Transformer作为噪声预测网络**，克服了传统U-Net的三大局限：
1. **扩展瓶颈**：U-Net在增大模型时性能回报递减
2. **架构割裂**：与NLP等领域的主流架构不统一
3. **全局依赖建模不足**：卷积操作的感受野有限

## 架构设计解析

### 整体框架
DiT建立在**Latent Diffusion Model（LDM）** 框架上，在低维潜在空间进行操作：
1. **编码阶段**：VAE编码器将图像压缩至潜在空间（如256×256×3 → 32×32×4）
2. **扩散阶段**：在潜在空间执行扩散过程
3. **解码阶段**：VAE解码器将去噪后的潜在表示恢复为像素空间

!https://example.com/dit_workflow.png

### 核心组件
#### 1. Patchify模块
将空间表示转换为Transformer可处理的序列：
```python
class Patchify(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B,C,H,W) -> (B,d,H/p,W/p)
        B,C,H,W = x.shape
        return x.reshape(B, C, H*W).permute(0,2,1)  # -> (B,N,d)
```
- **关键参数**：`patch_size`（p）决定token数量 $T=(H/p)×(W/p)$
- 典型配置：p=2/4/8，d=1152（XL模型）

#### 2. 条件注入机制
DiT探索了三种条件融合方式：
1. **In-Context Conditioning**：将时间步t和类别c的嵌入作为额外token拼接
2. **Cross-Attention**：在自注意力后添加条件信息的交叉注意力层
3. **Adaptive Layer Norm (adaLN)**：动态生成LayerNorm参数

**adaLN-Zero**被证明为最优方案：
```python
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)  # 生成γ,β,α
        )
    
    def forward(self, x, c):
        # c: 条件向量 (时间步t + 类别c的融合嵌入)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).chunk(6,1)
        
        # 自注意力分支
        x = x + alpha1 * self.attn(gamma1 * self.norm1(x) + beta1)[0]
        # 前馈分支
        x = x + alpha2 * self.mlp(gamma2 * self.norm2(x) + beta2)
        return x
```
此设计**初始化为恒等函数**（α初始为0），确保训练稳定性。

#### 3. Transformer解码器
将处理后的token序列映射回空间表示：
```python
class DiTDecoder(nn.Module):
    def __init__(self, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(embed_dim, patch_size**2 * out_channels * 2)
    
    def forward(self, x, spatial_shape):
        # x: (B, T, d) -> (B, T, p²*2C)
        x = self.linear(x)
        B, T, _ = x.shape
        H, W = spatial_shape
        x = x.reshape(B, H//p, W//p, p, p, -1).permute(0,5,1,3,2,4)
        return x.reshape(B, -1, H, W)  # 分割噪声预测和协方差
```
输出包含**噪声预测$\epsilon_\theta$和协方差$\Sigma_\theta$**两部分。

## 训练与优化策略

### 可扩展性设计
DiT通过三维度实现模型扩展：
1. **深度**：DiT-S（12层）→ DiT-XL（28层）
2. **宽度**：隐藏维度从384（S）到1152（XL）
3. **Token数量**：减少patch尺寸（p=8→2）增加序列长度

!https://example.com/dit_scaling.png

实验表明，**计算量（Gflops）与生成质量（FID）呈强相关性**。DiT-XL/2（p=2）在ImageNet 256×256生成任务上达到FID 2.27，超越所有U-Net扩散模型。

### 条件机制比较
四种条件注入方式性能对比：

| **机制**          | **FID↓** | **训练速度** | **参数量** |
|-------------------|----------|--------------|------------|
| In-Context        | 5.21     | 1.0x         | 最小       |
| Cross-Attention   | 4.58     | 0.85x        | 增加20%    |
| adaLN             | 3.75     | 0.95x        | 不变       |
| **adaLN-Zero**    | **2.27** | **0.98x**    | 微增       |

adaLN-Zero的**优势源于其初始化为恒等函数**的特性，避免早期训练不稳定。

## 扩展与变体

### 1. U-ViT
融合U-Net的跳跃连接思想，将**所有中间层特征通过残差连接聚合**到解码端：
```
Input → Patchify → [Transformer Block]₁ → ... → [Transformer Block]ₙ
      ↘_______________________________________ ↗
```

### 2. MDT（Masked Diffusion Transformer）
引入**掩码潜在建模**增强语义学习：
- 训练时随机mask 30-50%的patch token
- 通过Side-Interpolater预测mask区域
- 推理时移除mask机制，保持效率

### 3. DiffiT（by NVIDIA）
结合**U-Net层级结构**与Transformer：
- 编码器：下采样阶段+Transformer块
- 解码器：上采样阶段+跳跃连接
- 采用Time-dependent Self-Attention注入时间步信息

## 实战应用

### Sora中的DiT实现
作为OpenAI的视频生成模型，**Sora的核心架构包含三个组件**：
1. **VAE编码器**：压缩视频帧至潜在空间
2. **ViT分词器**：将时空块转换为token序列
3. **DiT主干**：在扩散过程中处理噪声预测

关键创新在于**时空块划分**：
```python
def extract_spatiotemporal_patches(video, patch_size=(2,4,4)):
    B, T, C, H, W = video.shape
    return video.unfold(1, patch_size[0], patch_size[0])
              .unfold(2, patch_size[1], patch_size[1])
              .unfold(3, patch_size[2], patch_size[2])
```

### 图像生成示例
使用Hugging Face `diffusers` 库调用DiT-XL/2：
```python
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 类条件生成（ImageNet类别）
input_ids = pipe.get_label_ids(["strawberry", "cat"])
image = pipe(input_ids=input_ids).images[0]
image.save("dit_generated.png")
```

## 总结与未来方向
Diffusion Transformers通过**融合Transformer的扩展性与扩散模型的稳定训练特性**，在生成质量上实现了突破性进展。其核心优势体现在：
1. **架构统一性**：与NLP、CV领域主流架构对齐
2. **计算可扩展**：模型增大持续提升生成质量
3. **全局一致性**：自注意力机制优化长距离依赖

未来发展方向包括：
- **多模态对齐**：文本-图像-视频的统一DiT框架（如Sora）
- **3D生成**：扩展时空块处理能力
- **自监督学习**：结合MAE等预训练策略
- **硬件协同设计**：针对Transformer特性优化芯片架构

随着模型效率的进一步提升和开源生态的完善，DiT有望成为**通用生成式AI的基础架构**，赋能从创意设计到科学模拟的广泛应用场景。

> 正如Peebles所言：“Transformer的扩展定律尚未看到尽头，DiT只是揭开了生成式模型新范式的序幕。”