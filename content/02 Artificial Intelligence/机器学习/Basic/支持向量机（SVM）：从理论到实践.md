## 概述

支持向量机（Support Vector Machine, SVM）是一种经典的**监督学习算法**，由Vapnik等人于1992年正式提出[[1]](#references)。它在**小样本、非线性及高维模式识别**中表现出色，广泛应用于分类和回归任务。

## 发展历程

- ​**1963年**​：Vapnik提出**最大间隔分类器**的概念
- ​**1992年**​：引入**核技巧（Kernel Trick）​**，解决非线性问题
- ​**1995年**​：Cortes和Vapnik发表经典论文，奠定现代SVM基础
- ​**2000年后**​：随着核方法研究深入，SVM成为机器学习主流算法之一

## 数学原理

### 线性可分情况

对于线性可分数据集，SVM寻找**最优分离超平面**​：

$$
w^Tx + b = 0
$$

其中`w`是法向量，`b`是偏置项。优化目标是最大化间隔：

$$
\max \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1
$$

### 软间隔与松弛变量

对于线性不可分数据，引入松弛变量$\xi_i$：

$$
\min \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
$$

其中`C`是惩罚参数，控制分类错误的容忍度。

### 核技巧

通过核函数$\phi(x)$将数据映射到高维空间：

$$
K(x_i, x_j) = \phi(x_i)^T\phi(x_j)
$$

常用核函数包括：

- 线性核：`K(x_i, x_j) = x_i^Tx_j`
- 多项式核：`K(x_i, x_j) = (x_i^Tx_j + c)^d`
- RBF核：`K(x_i, x_j) = \exp(-\gamma\|x_i - x_j\|^2)`

## 适用场景

|场景|适用性|备注|
|---|---|---|
|小样本数据|★★★★★|SVM在小样本表现优异|
|高维数据|★★★★☆|如文本分类、基因分析|
|非线性问题|★★★★☆|需配合核函数使用|
|大规模数据|★★☆☆☆|训练复杂度O(n²)~O(n³)|

## 实践指南

### 数据预处理

1. 标准化：SVM对特征尺度敏感
2. 特征选择：高维数据可考虑PCA降维

### 参数调优

```
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 评估指标

- 准确率（Accuracy）
- 精确率-召回率（Precision-Recall）
- ROC曲线

## 最新进展

1. ​**SVM+深度学习**​：结合CNN特征的混合模型
2. ​**增量学习**​：适用于流式数据
3. ​**多核学习**​：自动学习最优核组合

## 代码示例

```python
# 使用scikit-learn实现SVM
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 加载数据
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# 评估
print("Accuracy:", clf.score(X_test, y_test))
```

## 常见问题

​**Q：如何选择核函数？​**​  
A：线性核适合高维数据，RBF核适合低维非线性数据

​**Q：SVM对缺失值敏感吗？​**​  
A：非常敏感，需要提前处理缺失值

## 参考文献

1. Cortes, C., & Vapnik, V. (1995). _Support-vector networks_. Machine Learning.
2. [scikit-learn SVM文档](https://scikit-learn.org/stable/modules/svm.html)
3. [[机器学习实战]] - 第6章SVM内容

## 延伸阅读

- [[核方法导论]]
- [[统计学习理论]]
- [[模式识别与机器学习]]