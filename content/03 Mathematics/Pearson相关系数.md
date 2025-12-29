---
title: "Pearson相关系数"
date: 2025-08-07
draft: false
---

#相关性


## 定义
Pearson相关系数（Pearson Correlation Coefficient），又称**皮尔逊积矩相关系数**，是衡量两个连续变量$X$和$Y$之间线性关系强度和方向的统计量，取值范围为$[-1, 1]$。其定义为：
$$
r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$
其中：
- $\text{cov}(X, Y)$为协方差
- $\sigma_X, \sigma_Y$为标准差
- $\bar{X}, \bar{Y}$为均值

## 发展历史
由英国统计学家**卡尔·皮尔逊**（Karl Pearson）在1895年提出，基于弗朗西斯·高尔顿（Francis Galton）的回归分析理论发展而来。Pearson相关系数成为现代统计学中相关性分析的基础工具之一。

## 数学原理
### 协方差与标准化
Pearson系数的核心思想是通过**协方差**衡量变量变化方向的一致性，再通过标准差标准化消除量纲影响。其数学性质包括：
1. **对称性**：$r(X, Y) = r(Y, X)$
2. **无量纲性**：对线性变换$X'=aX+b$不改变$r$值
3. **零相关性**：若$X$与$Y$独立，则$r=0$（逆命题不成立）

### 假设检验
常用$t$-检验判断相关系数的显著性：
$$
t = r \sqrt{\frac{n-2}{1-r^2}}
$$
服从自由度为$n-2$的$t$分布。

## 适用场景
### 适用条件
1. 变量为**连续型数据**
2. 数据服从**二元正态分布**
3. 关系为**线性**（对非线性关系不敏感）

### 典型应用
- 金融分析（股票收益率相关性）
- 医学研究（生理指标关联性）
- 社会科学（问卷评分一致性检验）

## 局限性
1. **对异常值敏感**：极端值可能导致$r$值失真
2. **仅检测线性关系**：如$X \sim U(-1,1), Y=X^2$时$r \approx 0$
3. **不适用于序数数据**：需改用Spearman相关系数

## 实践建议
1. **可视化优先**：先绘制散点图观察趋势
2. **结合统计检验**：报告$p$-value和置信区间
3. **样本量要求**：建议$n \geq 30$以保证稳定性

## Python实现
```python
import numpy as np
from scipy import stats

# 生成示例数据
np.random.seed(42)
X = np.random.normal(0, 1, 100)
Y = 0.5 * X + np.random.normal(0, 0.5, 100)

# 计算Pearson相关系数
r, p_value = stats.pearsonr(X, Y)
print(f"r = {r:.3f}, p = {p_value:.4f}")

# 可视化
import matplotlib.pyplot as plt
plt.scatter(X, Y, alpha=0.6)
plt.title(f"Pearson r = {r:.2f}")
plt.xlabel("X"); plt.ylabel("Y")
plt.show()
```

## 扩展阅读
- [Pearson的原始论文](https://royalsocietypublishing.org/doi/10.1098/rspl.1895.0076)
- [SciPy文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
- [可视化相关系数矩阵](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
- https://blog.csdn.net/chenxy_bwave/article/details/121576303