
高质量博客/综述
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 
> 英文版介绍
- [[Diffusion Models A Comprehensive Survey of Methods and Applications]] 
> 非常全面的综述，包含基础推导和各种diffusion base的方法

我的博客
- [[Diffusion Condition]]
- [[diffusion loss]]
- [[Diffusion Model：生成式AI的核心引擎]]
- [[Diffusion和Transformer结合]]
- [[Diffusion Models学习入门]]  
- 

[[DDPM的数学原理推导]]

## Questions

### 采样时间步是什么意思？

在Diffusion模型中，采样时间步（Sampling Timestep）是指扩散过程或去噪过程中的时间阶段。

扩散模型将数据生成视为一个逐步去噪的过程，这个过程被分为多个时间步骤，从初始时刻到最终时刻，每个时刻对应一个采样时间步。例如，若总共有T个时间步，那么采样时间步t可以是从1到T中的任意一个整数，表示当前处于扩散或去噪过程的哪一个阶段。

在训练时，通常会从均匀分布Uniform({1, 2, 3, ..., T})中采样一个时间步t。模型会根据这个采样时间步t，对输入数据（通常是加噪后的数据）进行处理，预测该时间步添加的噪声或前一个时间步的样本。在推理阶段，采样时间步则决定了去噪的步数，采样时间步越多，去噪过程越精细，生成的样本质量可能越高，但所需的计算时间和资源也越多。

在Diffusion模型中，T的设置没有固定标准，通常会根据具体任务和模型需求在几十到几千之间取值，常见取值有100、1000、2000等。例如，在MNIST实验中，T设置为100以上通常就能保证效果。原始的DDPM论文中使用T = 1000。一些研究为了追求更精细的过程或更好的生成效果，可能会将T设置得更大，如2000左右。

关于t是否必须为整数，这取决于扩散模型的具体形式：
- **离散时间扩散模型**：在常见的离散时间Diffusion模型中，t通常被定义为整数。因为离散时间模型将扩散过程分为有限个离散的步骤，每个步骤对应一个整数时间步，从0开始到T结束，如DDPM中t就是在0到T这个范围内的整数。
- **连续时间扩散模型**：在连续时间扩散模型中，t可以是实数。连续时间扩散模型将扩散过程视为一个连续的随机过程，由随机微分方程描述，t可以在0到T这个区间内取任意值，无需局限于整数。

### 怎么和Transformer结合的？
[[Diffusion和Transformer结合]]


如何实现Condition的？怎么接入到模型中？
[[Diffusion Condition]]



训练的loss有哪些？
[[diffusion loss]]


# 论文
## 入门介绍
## 重要改进
- [[TASD（Tiny Autoencoder for Stable Diffusion）]]

## 应用
- [[MarDini Masked Autoregressive Diffusion for Video Generation at Scale]]
- [[LTX-Video Realtime Video Latent Diffusion]]
