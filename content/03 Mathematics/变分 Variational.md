---
title: "基本概念"
date: 2025-08-07
draft: false
---

> “**变分**”是一个在数学和计算机科学中非常重要的概念，特别是在优化、概率论和机器学习等领域。它通常涉及对某种函数或函数空间进行“变动”或“优化”的过程。变分的应用广泛，尤其是在推导和近似计算中。

  

# 基本概念
“变分”（Variational）这个术语源自于变分法（Calculus of Variations），它是研究在给定约束条件下，如何极小化或极大化函数的理论方法。在数学中，变分法通常用于寻找**泛函**的极值，而泛函是作用于函数的映射。

  

**泛函的定义**

  

一个泛函是将函数映射到实数的规则。形式上，可以表示为：

  

$$

J[f] = \int_a^b F(x, f(x), f’(x)) , dx

$$

  

其中 是一个函数， 是其导数， 是泛函中的目标函数。通过变分法，我们要寻找一个函数 ，使得 达到极值。

  

**2. 变分推理（Variational Inference）**

  

在统计学和机器学习中，**变分推理（Variational Inference，VI）**是一种用来近似计算后验分布的技术，特别是在贝叶斯推理中。它基于变分原理，通过引入一个可调的分布族，并通过优化找到一个最优分布，以此来近似目标后验分布。

  

**变分推理的核心思想**

  

在贝叶斯推理中，我们通常关心的是后验分布：

  

$$

p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}

$$

  

这里 是潜变量， 是观察到的数据， 是似然函数， 是先验， 是边际似然。

  

然而，直接计算后验分布 是困难的，因为边际似然 需要对所有潜变量进行积分，计算量非常大。因此，我们使用变分推理来近似计算后验分布。

  

变分推理的核心是使用一个简单的分布 来逼近真实的后验分布 。我们通过最小化变分分布 和真实后验分布 之间的**Kullback-Leibler（KL）散度**来优化这个近似分布：

  

$$

\text{KL}(q(\mathbf{z}) | p(\mathbf{z} | \mathbf{x})) = \mathbb{E}_{q(\mathbf{z})} \left[ \log \frac{q(\mathbf{z})}{p(\mathbf{z} | \mathbf{x})} \right]

$$

  

通过最小化KL散度，我们得到最优的变分分布 ，从而有效地近似后验分布。

  

**变分下界（ELBO）**

  

为了进行优化，我们通常通过对数边际似然的变分下界（Evidence Lower Bound, ELBO）进行优化。由于直接优化后验似然困难，我们引入变分分布 ，并推导出下界：

  

$$

\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \text{KL}(q(\mathbf{z}) | p(\mathbf{z}))

$$

  

这个下界的优化可以通过最大化ELBO来实现，即通过优化以下目标：

  

$$

\mathcal{L}_{\text{VI}} = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]

$$

  

这种方法能够在计算复杂度较低的情况下，进行高效的后验分布近似。

  

**3. 变分自动编码器（VAE）中的变分**

  

**变分自动编码器（Variational Autoencoder, VAE）**是变分推理的一种应用，它是一种生成模型，通过引入变分推理来对潜在变量进行建模。VAE的核心思想是通过变分推理来逼近潜在变量的后验分布，训练过程中优化变分下界。

  

VAE的变分下界（ELBO）如下：

  

$$

\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(\mathbf{z} | \mathbf{x})} [\log p(\mathbf{x} | \mathbf{z})] - \text{KL}(q(\mathbf{z} | \mathbf{x}) | p(\mathbf{z}))

$$

  

这里，第一项是重构误差（通过对 的近似分布进行采样），第二项是KL散度，用来测量近似分布 与先验分布 的差距。

  

**4. 变分的应用总结**

• **变分法**：主要用于数学优化，寻找泛函的极值。

• **变分推理（Variational Inference）**：在贝叶斯推理中，使用一个简单的分布来近似目标后验分布，优化变分下界来得到最优解。

• **变分自动编码器（VAE）**：利用变分推理在生成模型中对潜在空间进行建模，并通过优化变分下界来学习生成数据。

  

**5. Python代码示例：VAE的变分推理**

  

以下是一个简单的VAE模型的PyTorch实现：

  

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision

import torchvision.transforms as transforms

  

class VAE(nn.Module):

    def __init__(self, latent_dim=20):

        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)

        self.fc21 = nn.Linear(400, latent_dim)  # Mean of z

        self.fc22 = nn.Linear(400, latent_dim)  # Log variance of z

        self.fc3 = nn.Linear(latent_dim, 400)

        self.fc4 = nn.Linear(400, 784)

  

    def encode(self, x):

        h1 = torch.relu(self.fc1(x.view(-1, 784)))

        return self.fc21(h1), self.fc22(h1)

  

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)

        eps = torch.randn_like(std)

        return mu + eps*std

  

    def decode(self, z):

        h3 = torch.relu(self.fc3(z))

        return torch.sigmoid(self.fc4(h3))

  

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

  

def loss_function(recon_x, x, mu, logvar):

    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence between q(z|x) and p(z)

    # p(z) is a standard normal distribution, so the KL term is simple

    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # where mu and sigma^2 are the parameters of q(z|x)

    # logvar is the log of sigma^2

    # Here logvar is the log-variance, mu is the mean

    # so we use the formula directly for KL divergence

    MSE = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + MSE

  

# Example usage

model = VAE()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

  

# Train loop

for epoch in range(10):

    for data in DataLoader(...):

        optimizer.zero_grad()

        x, _ = data

        recon_batch, mu, logvar = model(x)

        loss = loss_function(recon_batch, x, mu, logvar)

        loss.backward()

        optimizer.step()

  

在VAE中，变分推理通过**编码器**来计算潜变量的均值 和对数方差 ，然后通过**重参数化技巧**将这些参数转换为潜变量 ，最终通过**解码器**生成数据。

  

**结论**

  

变分推理是现代机器学习中强大的工具，特别是在生成模型中，如VAE、DDPM等。它通过引入可优化的分布族和KL散度来逼近目标分布，从而高效地计算和近似复杂的后验分布。