---
aliases:
  - 变分
type: blog
done: "true"
---

>在人工智能的浪潮中，我们习惯于模型给出一个确定的答案：这张图是“猫”，这段文本的情感是“积极”。然而，在现实世界中，“不确定性”无处不在。模型能否不仅给出预测，还能告诉我们它对这个预测有多大的把握？这正是贝叶斯深度学习（Bayesian Deep Learning）的核心议题，而**变分推断（Variational Inference, VI）** 则是实现这一目标最主流、最强大的技术武器之一。
>
>本文将带你深入探索“变分”的世界，理解它如何从一个经典的数学概念演变为现代AI，尤其是生成模型和不确定性量化领域的基石。

# 什么是“变分”？从思想本源说起

“变分”一词最早源于数学中的**变分法（Calculus of Variations）**。普通函数求极值，是找到一个“点”$x$使得$f(x)$最小；而变分法处理的是“泛函（Functional）”的极值问题，即找到一个“函数”$f$使得$J[f]$最小。它的核心思想是在一个函数空间中，通过微小的“变动”（Variation）来寻找最优的那个函数。

这个思想被巧妙地迁移到了概率推断领域。在概率模型中，我们常常关心一个核心问题：给定观测数据$X$，模型中潜在变量（Latent Variable）$Z$的**后验概率**$p(Z|X)$是什么？这个后验概率包含了在看到数据后，我们对模型内部状态的所有认知，是进行预测、生成新数据和理解模型的关键。

然而，根据贝叶斯公式：
$$p(Z|X) = \frac{p(X|Z)p(Z)}{p(X)} = \frac{p(X|Z)p(Z)}{\int p(X|Z)p(Z) dZ}$$
问题出在分母$p(X)$，它被称为**证据（Evidence）** 或 **边际似然（Marginal Likelihood）**。计算它需要对所有可能的隐变量$Z$进行积分（或求和），这在$Z$的维度很高或者模型很复杂（例如深度神经网络）时，几乎是不可能完成的，我们称之为 ***“ intractable”***。

> [!note]
>  **X被称为 “证据”**：因为它是贝叶斯推断中唯一的 “观测事实”，是更新对隐变量Z的信念的核心依据 —— 没有X，就没有 “从先验到后验” 的推断基础。


既然直接计算行不通，变分推断（VI）提供了一种绝妙的近似思路：我们不再直接计算真实的后验$p(Z|X)$，而是**寻找一个我们能够轻松处理的、更简单的分布$q(Z)$，使其尽可能地逼近$p(Z|X)$**。

这里的$q(Z)$就是我们在函数空间中要寻找的那个“最优函数”。我们通过“变分”的方法，在某个预设的分布族（例如高斯分布族）中，调整$q(Z)$的参数，来最小化它与真实后验$p(Z|X)$之间的“距离”。

# 核心原理：证据下界（ELBO）的推导与解读

如何衡量$q(Z)$和$p(Z|X)$的“距离”？最常用的度量是[[03 Mathematics/Kullback-Leibler散度|KL散度]]，它衡量了两个概率分布的差异性。我们的目标是：
$$\min_{q} D_{KL}(q(Z) || p(Z|X))$$

接下来是变分推断中最为核心的数学推导。
$$
\begin{align*}
D_{KL}(q(Z) || p(Z|X)) &= \int q(Z) \log \frac{q(Z)}{p(Z|X)} dZ \\
&= \mathbb{E}_{q(Z)}[\log q(Z) - \log p(Z|X)] \\
&= \mathbb{E}_{q(Z)}[\log q(Z) - \log \frac{p(X, Z)}{p(X)}] \\
&= \mathbb{E}_{q(Z)}[\log q(Z) - \log p(X, Z) + \log p(X)] \\
&= \mathbb{E}_{q(Z)}[\log q(Z) - \log p(X, Z)] + \log p(X) \quad (\text{因为} \log p(X) \text{与} Z \text{无关})
\end{align*}
$$
整理一下可得：$$
\log p(X) = D_{KL}(q(Z) || p(Z|X)) + \mathbb{E}_{q(Z)}[\log p(X, Z) - \log q(Z)]
$$由于KL散度$D_{KL}(q(Z) || p(Z|X)) \ge 0$，我们得到了一个非常重要的不等式：$$
\log p(X) \ge \mathbb{E}_{q(Z)}[\log p(X, Z) - \log q(Z)]
$$不等式右边的部分，就是大名鼎鼎的**证据下界（Evidence Lower BOund, ELBO）**，我们通常记作$\mathcal{L}(q)$。

$$\mathcal{L}(q) = \mathbb{E}_{q(Z)}[\log p(X, Z) - \log q(Z)]$$

最大化ELBO $\mathcal{L}(q)$ 等价于 最小化$q(Z)$和$p(Z|X)$之间的KL散度。由于ELBO是我们能够计算的（因为它只涉及$p(X,Z)$和$q(Z)$，避开了$p(X)$），我们将一个无法解决的推断问题（最小化KL散度）转化为了一个可以解决的**优化问题**（最大化ELBO）。

### ELBO的两种形式与直观理解

为了更深刻地理解，ELBO通常被写成以下两种等价形式：

1.  **形式一：似然与先验**


$$\mathcal{L}(q) = \underbrace{\mathbb{E}_{q(Z)}[\log p(X|Z)]}_{\text{重构项}} - \underbrace{D_{KL}(q(Z) || p(Z))}_{\text{正则项}}
$$

**重构项 (Reconstruction Term)**：$\mathbb{E}_{q(Z)}[\log p(X|Z)]$。它鼓励模型在给定从$q(Z)$采样的隐变量$Z$后，能够很好地“重构”出原始数据$X$。这部分对应了模型的**似然度**，希望模型学到的隐变量包含足够的信息来解释数据。


 **正则项 (Regularization Term)**：$D_{KL}(q(Z) || p(Z))$。它要求我们找到的近似后验$q(Z)$不能离先验分布$p(Z)$太远。$p(Z)$通常选择简单的标准正态分布$\mathcal{N}(0, I)$。这一项起到了正则化的作用，防止$q(Z)$为了完美重构数据而变得过于复杂和奇特。

2.  **形式二：证据与熵**

$$\mathcal{L}(q) = \mathbb{E}\_{q(Z)}[\log p(X, Z)] + H(q(Z))$$

**联合概率期望**：鼓励模型找到那些能够让数据和隐变量联合概率$p(X,Z)$最大的$Z$。

**熵 (Entropy)**：$H(q(Z))$。熵衡量了分布的不确定性。最大化熵使得$q(Z)$分布尽可能地“宽”，防止它坍缩到一个点上，从而保留了对不确定性的建模。

这两种形式揭示了变分推断的内在权衡：**模型既要努力解释数据（最大化重构项），又要保持自身的简约和泛化能力（最小化KL散度或最大化熵）。**

# 变分自编码器（VAE）：变分思想的明星应用

如果说变分推断是理论基石，那么**变分自编码器（Variational Autoencoder, VAE）** 就是其在深度学习领域最璀璨的明星。VAE巧妙地将变分推断与神经网络结合，构建了一个强大的深度生成模型。
``

一个VAE包含两个核心部分：

1.  **编码器（Encoder）**，也叫推断网络（Inference Network）：它是一个神经网络，输入数据$X$，输出近似后验分布$q(Z|X)$的参数。例如，如果假设$q$是高斯分布，编码器就会输出均值$\mu(X)$和方差$\sigma^2(X)$。这就是所谓的 **“摊销推断”（Amortized Inference）**，即用一个网络一次性地为所有数据点预测其后验参数，而不是像传统VI那样为每个数据点单独优化。
2.  **解码器（Decoder）**，也叫生成网络（Generative Network）：它也是一个神经网络，输入从$q(Z|X)$采样的隐变量$Z$，尝试重构原始数据$X$。它定义的正是$p(X|Z)$。

### 关键技巧：重参数化（Reparameterization Trick）

在训练VAE时，我们需要从$q(Z|X)$中采样$Z$来计算ELBO的重构项。然而，采样这个操作是随机的，不可微分的，这会导致梯度无法从解码器反向传播到编码器。

**重参数化技巧** [2] 优雅地解决了这个问题。以高斯分布为例，从$\mathcal{N}(\mu, \sigma^2)$中采样一个$Z$，等价于先从标准正态分布$\mathcal{N}(0, 1)$中采样一个随机噪声$\epsilon$，然后通过确定性变换得到$Z$：
$$Z = \mu + \sigma \odot \epsilon, \quad \text{其中} \epsilon \sim \mathcal{N}(0, 1)$$
通过这种方式，随机性被转移到了与网络参数无关的$\epsilon$上，而$Z$与编码器输出的$\mu$和$\sigma$之间的计算路径是确定且可微的。这样，梯度就可以顺利地反向传播了。

### VAE的训练与应用

VAE的损失函数就是负的ELBO：
$$\text{Loss} = -\mathcal{L}(q) = \mathbb{E}_{q(Z|X)}[-\log p(X|Z)] + D_{KL}(q(Z|X) || p(Z))$$

  * **重构损失**：对于图像，通常是二元交叉熵或均方误差；对于文本，是交叉熵。
  * **KL散度损失**：可以被解析地计算出来（如果$q$和$p$都是高斯分布）。

VAE的应用非常广泛：

  * **图像/文本生成**：在隐空间中采样一个$Z$，然后通过解码器可以生成新的、与训练数据相似的数据。
  * **数据降维与表示学习**：编码器将高维数据映射到一个有意义的低维隐空间，这个空间具有良好的结构。
  * **异常检测**：正常数据会有较小的重构损失，而异常数据通常无法被很好地重构。
  * **半监督学习**：利用VAE学习到的数据结构来提升在少量标注数据上的学习效果。

# 变分方法的更多应用场景

除了VAE，变分思想还渗透在AI的许多其他角落。

  * **贝叶斯神经网络（Bayesian Neural Networks, BNN）**：传统神经网络的权重是确定的点估计。BNN则为每个权重都学习一个概率分布。变分推断被用来近似这些权重极其复杂的后验分布，从而让模型在预测时能给出置信区间，实现不确定性量化。这在医疗诊断、自动驾驶等安全攸关领域至关重要。
  * **主题模型（Topic Models）**：像潜在狄利克雷分配（LDA）这样的模型，其原始推断方法是吉布斯采样，而变分推断为其提供了更快速、确定性的推断方案（SVI，Stochastic Variational Inference [3]）。
  * **强化学习（Reinforcement Learning）**：在一些基于模型的强化学习或探索策略中，变分推断可以用来建模环境动态或策略的不确定性。

# 最新进展与前沿方向

变分推断领域至今仍然非常活跃，不断涌现出新的研究成果。

1.  **更灵活的后验近似**：经典VI使用简单的均场（Mean-field）假设（即$q(Z) = \prod_i q(Z_i)$），这限制了其捕捉变量间相关性的能力。**归一化流（Normalizing Flows）** 等技术被用来构建更复杂、更具表达能力的$q$分布，通过一系列可逆变换将简单分布扭曲成复杂分布，极大地提升了VI的精度。
2.  **与扩散模型的联系**：近年来大火的**扩散模型（Diffusion Models）** 被证明与一类特殊的层级变分自编码器（Hierarchical VAE）在数学上存在深刻联系。理解这一点有助于我们统一两大生成模型范式，并相互借鉴思路。
3.  **摊销推断的改进**：如何解决VAE中的“后验坍塌”（Posterior Collapse）问题（即KL散度项过早变为0，导致隐变量失去意义）是持续的研究热点。各种对ELBO的修改、新的网络结构和训练策略被不断提出。
4.  **大规模应用**：如何将变分方法高效地应用到拥有数十亿甚至上万亿参数的超大模型（如LLMs）中，以进行不确定性量化或模型压缩，是一个充满挑战和机遇的前沿方向。

# 实践经验与代码示例

下面我们用PyTorch实现一个简单的VAE来处理MNIST手写数字数据集。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义超参数
latent_dim = 20
hidden_dim = 400
image_size = 784 # 28*28
batch_size = 128
epochs = 20

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE模型定义
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(image_size, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim) # 均值
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim) # log(方差)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 从标准正态分布采样
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重构损失 (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, image_size), reduction='sum')
    # KL散度损失 (解析解)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 数据加载
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# 模型和优化器
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# 开始训练
for epoch in range(1, epochs + 1):
    train(epoch)

```

**实践要点**：

  * **KL退火（KL Annealing）**：在训练初期，KL散度项的梯度可能会压倒重构项，导致模型“偷懒”（后验坍塌）。一种常见的技巧是给KL项增加一个权重，从0开始随训练逐步增加到1，让模型先学会重构，再学习正则化。
  * **监控KL散度**：在训练过程中，监控KL散度的值。如果它迅速降为0，说明可能发生了后验坍塌。
  * **选择隐空间维度**：`latent_dim`是一个重要的超参数。太小，模型无法捕捉数据复杂度；太大，容易过拟合且可能导致后验坍塌。

# 总结与推荐学习资源

变分推断是一套优雅而强大的思想框架，它将复杂的概率推断问题转化为深度学习擅长的优化问题，为在模型中引入和量化不确定性打开了大门。从作为生成模型基石的VAE，到赋予模型“自知之明”的贝叶斯神经网络，再到与扩散模型等前沿技术的交融，变分方法已经成为现代AI不可或缺的一部分。

掌握变分思想，不仅仅是学会一个工具，更是理解AI如何处理“未知”和“随机”的关键一步。

**推荐学习资源**：

1.  **开创性论文**：
      * [1] **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013): VAE的奠基之作。 [链接](https://arxiv.org/abs/1312.6114)
      * [2] **Stochastic Backpropagation and Approximate Inference in Deep Generative Models** (Rezende, Mohamed & Wierstra, 2014): 与VAE同期独立提出的类似工作，也对重参数化有重要贡献。 [链接](https://arxiv.org/abs/1401.4082)
      * [3] **Stochastic Variational Inference** (Hoffman et al., 2013): 提出SVI，使得变分推断能应用于大规模数据集。 [链接](https://arxiv.org/abs/1206.7051)
2.  **深度教程与博客**：
      * **An Introduction to Variational Autoencoders** (by Jaan Altosaar): 一篇非常详尽且直观的VAE入门文章。 [链接](https://www.google.com/search?q=https://jaan.io/what-is-variational-autoencoder/)
      * **Tutorial on Variational Autoencoders** (by Carl Doersch): 内容深入，包含许多理论细节。 [链接](https://arxiv.org/abs/1606.05908)
3.  **前沿进展相关论文**：
      * [4] **Variational Inference with Normalizing Flows** (Rezende & Mohamed, 2015): 将Normalizing Flows引入VI的开创性工作。 [链接](https://arxiv.org/abs/1505.05770)
      * [5] **Denoising Diffusion Probabilistic Models** (Ho, Jain & Abbeel, 2020): 现代扩散模型的代表作。 [链接](https://arxiv.org/abs/2006.11239)

希望这篇博客能为你打开理解“变分”世界的大门，并激发你进一步探索的兴趣。