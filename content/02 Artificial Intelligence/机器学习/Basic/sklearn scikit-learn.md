---
title: "sklearn scikit-learn"
date: 2025-08-07
draft: false
---

### 1. **Sklearn 简介**
`scikit-learn`（简称 `sklearn`）是一个基于 Python 的机器学习库，提供了简单易用的工具，用于数据挖掘和数据分析。Sklearn 建立在 NumPy、SciPy 和 Matplotlib 之上，广泛用于构建机器学习模型，特别是分类、回归、聚类、降维等任务。
https://scikit-learn.org/stable/api/sklearn.html
https://scikit-learn.org/stable/index.html

### Sklearn 的主要功能：
- **分类**：比如 k-近邻、决策树、随机森林等算法。
- **回归**：比如线性回归、岭回归、Lasso 等。
- **聚类**：比如 k-means、层次聚类等。
- **降维**：比如 PCA、t-SNE 等。
- **模型选择**：交叉验证、网格搜索等。
- **数据预处理**：标准化、归一化、缺失值填充等。

---

### 2. **Sklearn 的基本使用**

这里以一个简单的机器学习任务为例，演示如何使用 `sklearn` 进行分类任务（以鸢尾花数据集为例）。

#### **步骤 1：安装 Sklearn**
首先，确保已经安装了 `scikit-learn`，可以使用以下命令安装：
```bash
pip install scikit-learn
```

#### **步骤 2：加载数据集**
Sklearn 提供了一些内置的数据集，比如经典的鸢尾花（Iris）数据集。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### **步骤 3：选择模型**
我们可以选择一个分类模型，比如 `KNeighborsClassifier`（K 近邻算法）。

```python
from sklearn.neighbors import KNeighborsClassifier

# 初始化 KNN 分类器，设置 K 值为 3
model = KNeighborsClassifier(n_neighbors=3)
```

#### **步骤 4：训练模型**
使用训练数据训练模型。

```python
# 训练模型
model.fit(X_train, y_train)
```

#### **步骤 5：评估模型**
用测试集评估模型的准确率。

```python
# 使用测试数据进行预测
y_pred = model.predict(X_test)

# 计算模型的准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
```

#### **步骤 6：保存和加载模型**
你可以使用 `joblib` 库来保存和加载训练好的模型：

```python
from joblib import dump, load

# 保存模型到文件
dump(model, 'knn_model.joblib')

# 加载模型
model_loaded = load('knn_model.joblib')
```

---

### 3. **常用功能概述**

#### **模型评估**
除了 `accuracy_score` 之外，`sklearn` 还提供了其他评估指标，如：
- **混淆矩阵**：`confusion_matrix`
- **精确率和召回率**：`precision_score`、`recall_score`
- **F1 值**：`f1_score`

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

# 计算精确率、召回率和 F1 值
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"精确率: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 值: {f1:.2f}")
```

#### **交叉验证**
交叉验证可以帮助更好地评估模型的性能，`cross_val_score` 是常用的交叉验证方法：

```python
from sklearn.model_selection import cross_val_score

# 使用交叉验证评估模型
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"交叉验证准确率: {cv_scores.mean():.2f}")
```

#### **超参数调优**
可以使用 `GridSearchCV` 来优化模型的超参数：

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.2f}")
```

---

### 总结
`Sklearn` 是一个功能强大且易于使用的机器学习库，适合各种机器学习任务。其提供了丰富的算法、评估工具和数据预处理功能，方便用户快速构建和评估机器学习模型。在日常项目中，它可以极大地简化模型开发和评估的流程。