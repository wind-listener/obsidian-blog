---
aliases:
  - diffusion
type: blog
status: writing
---

## 1 定义与发展历程
Diffusion Model（扩散模型）是一类**基于[[马尔可夫链]]** 的生成式模型，其核心思想是通过**逐步添加噪声**破坏数据分布，再学习**逆向去噪过程**以重建数据。受非平衡统计物理学的启发，该模型通过**前向扩散**系统性地破坏数据结构，再通过**反向扩散**恢复结构，形成高度灵活的生成框架。

### 1.1 发展里程碑
- **奠基工作**：DDPM（Denoising Diffusion Probabilistic Models）首次给出严谨的数学推导与可复现代码，建立**加噪-去噪**的基本范式
- **效率突破**：DDIM（Denoising Diffusion Implicit Model）改进逆向过程，实现**确定性采样加速**，生成速度提升10×
- **跨模态演进**：Stable Diffusion引入**潜在空间操作**，在低维空间执行扩散，显著降低计算开销
- **工业级应用**：OpenAI的GLIDE实现**文本引导图像生成**，推动多模态融合

## 2 核心原理剖析

### 2.1 前向扩散过程
将原始数据 $X_0$ 逐步转化为高斯噪声，每步添加可控噪声：
$$X_t = \sqrt{\alpha_t}X_{t-1} + \sqrt{1-\alpha_t}Z_t, \quad Z_t \sim \mathcal{N}(0,I)$$
其中 $\alpha_t = 1 - \beta_t$，$\beta_t$ 为**噪声调度系数**。经推导可得闭式解：[[diffusion前向加噪过程公式推导]]
$$q(X_t|X_0) = \mathcal{N}(X_t; \sqrt{\bar{\alpha}_t}X_0, (1-\bar{\alpha}_t)I)$$
其中 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。该过程将数据分布转化为**各向同性高斯分布**。

### 2.2 逆向去噪过程
关键目标：学习映射 $p_\theta(X_{t-1}|X_t)$ 以重建数据。通过变分推断优化**变分下界（ELBO）**：
$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[ \log \frac{q(X_{1:T}|X_0)}{p_\theta(X_{0:T})} \right]$$
分解为逐时间步的KL散度项：
$$\mathcal{L}_t = D_{\text{KL}}\left( q(X_t|X_{t+1},X_0) \parallel p_\theta(X_t|X_{t+1}) \right)$$
其中 $q(X_t|X_{t+1},X_0)$ 可作为**去噪训练的目标分布**。

### 2.3 训练目标简化
通过参数重整化，目标简化为**噪声预测任务**：
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,X_0,\epsilon} \left[ \| \epsilon - \epsilon_\theta(X_t,t) \|^2 \right]$$
其中 $X_t = \sqrt{\bar{\alpha}_t}X_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。U-Net模型 $\epsilon_\theta$ 学习预测添加的噪声。

## 3 关键技术与架构创新

### 3.1 噪声调度策略
| **策略类型** | **公式** | **优势** | **局限** |
|------------|---------|---------|---------|
| **线性调度** | $\beta_t = \beta_{\text{min}} + (\beta_{\text{max}} - \beta_{\text{min}})\frac{t}{T}$ | 实现简单，小规模任务稳定 | 噪声增减不均衡 |
| **余弦调度** | $\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \ \bar{\alpha}_t = \frac{\cos(t/T \cdot \pi/2)}{\cos(\pi/2)}$ | 平滑过渡，保留更多细节 | 计算复杂度较高 |

### 3.2 U-Net架构改进
原始U-Net针对扩散任务深度优化：
```python
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        # 下采样路径
        self.down1 = DownsampleBlock(64)  # 64×64→32×32
        self.attn1 = SelfAttentionBlock(128) 
        # 上采样路径
        self.up1 = UpsampleBlock(256)  # 跳跃连接融合
        # 输出层
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)
```
核心改进：
1. **残差块替换**：基础ResNet块替换为**自注意力块**（Multi-Head Attention）
2. **时间步嵌入**：将时间步 $t$ 编码为特征向量，通过**加法/乘法**融入各层
3. **跳跃连接**：编码器特征直连解码器，保留**空间信息完整性**

## 4 应用场景与实战

### 4.1 典型应用领域
| **领域** | **案例** | **技术亮点** |
|----------|----------|-------------|
| 图像生成 | Stable Diffusion | 潜在空间扩散，512×512图像生成仅需2秒 |
| 图像编辑 | GLIDE | 文本引导局部编辑，实现语义操控 |
| 视频生成 | Make-A-Video | 时间维度扩散，帧间一致性保持 |
| 科学计算 | AlphaFold3 | 蛋白质结构扩散生成 |

### 4.2 训练调优经验
- **权重归一化**：约束层权重 $\|W\|=1$，避免**激活值漂移**导致的训练不稳定
- **EMA优化**：采用**指数移动平均**保存权重，需精细调节EMA长度（最佳值约0.1-0.2倍训练步长）
  ```python
  # 指数移动平均实现
  def update_ema(model, ema_model, decay=0.9999):
      with torch.no_grad():
          for param, ema_param in zip(model.parameters(), ema_model.parameters()):
              ema_param.copy_(decay * ema_param + (1 - decay) * param)
  ```
- **后重建技术**：训练后组合不同EMA长度的快照，快速获得最优模型

## 5 最新研究进展

### 5.1 EDM2架构突破
!https://developer.nvidia.com/blog/wp-content/uploads/2024/04/edm2_perf.png  
*EDM2在ImageNet-512上FID=1.81，模型缩小5倍仍保持SOTA性能*

关键创新：
1. **激活值保持**：强制每层输入/输出激活值范数不变
2. **组归一化移除**：简化网络结构，避免特征失真
3. **偏置项消除**：实验证明不影响性能，提升训练稳定性

### 5.2 一致性模型（Consistency Models）
- **单步生成**：将扩散轨迹映射为ODE，通过蒸馏实现一步采样
- **零样本编辑**：仅需预训练模型即可实现图像修复、插值

### 5.3 多模态融合
- **CLIP引导**：文本编码器与扩散模型联合训练，实现细粒度跨模态生成
- **3D扩散**：NeRF+Diffusion实现三维场景生成（如NVIDIA GET3D）

## 6 完整代码实现
参考Stable Diffusion官方代码库：https://github.com/CompVis/stable-diffusion

```python
# 简化的DDPM采样代码
def ddpm_sampling(model, noise, T, alpha_bars):
    x = noise
    for t in range(T, 0, -1):
        z = torch.randn_like(x) if t > 1 else 0
        eps = model(x, t)  # U-Net预测噪声
        x = (1 / torch.sqrt(alpha_bars[t])) * 
             (x - (1 - alpha_bars[t]) / torch.sqrt(1 - alpha_bars[t]) * eps) + 
             torch.sqrt(1 - alpha_bars[t]) * z
    return x
```

## 7 总结与展望
Diffusion Model通过**物理启发的生成范式**，在生成质量、模式覆盖等方面超越GAN/VAE等传统模型。未来发展方向包括：
1. **生成速度优化**：通过蒸馏/隐式采样实现实时生成
2. **3D生成统一架构**：融合NeRF、点云等三维表示
3. **生物计算应用**：蛋白质设计、分子生成等科学计算场景
4. **训练稳定性提升**：EDM2等架构持续降低训练成本

> 技术革命总在否定之否定中演进：Diffusion Model用简单的噪声扰动代替复杂的对抗训练，却实现了更稳定的生成性能。其本质在于**将生成过程转化为可学习的物理方程**，这正是AI与科学交叉的迷人之处。  

**扩展阅读**：  
- https://arxiv.org/abs/2006.11239  
- https://huggingface.co/docs/diffusers/index  
- https://arxiv.org/abs/2312.02696