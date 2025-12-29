---
title: "查看所有基本颜色名称"
date: 2025-09-26
draft: false
---

Python 中有多个预设的颜色库可以方便调用。以下是几个最常用的颜色库及其使用方法：

## 1. Matplotlib 内置颜色

```python
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# 查看所有基本颜色名称
print("基本颜色名称:")
print(list(mcolors.BASE_COLORS.keys()))
print("\nCSS4颜色名称:")
print(list(mcolors.CSS4_COLORS.keys())[:20])  # 显示前20个

# 使用示例
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
hex_colors = [mcolors.CSS4_COLORS[name] for name in colors]

print("十六进制颜色值:", hex_colors)
```

## 2. Seaborn 调色板

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 查看所有可用调色板
print("Seaborn调色板:")
print(sns.color_palettes())

# 使用预设调色板
# 分类调色板（适用于离散数据）
palette1 = sns.color_palette("deep")      # 深色调
palette2 = sns.color_palette("pastel")    # 柔和色调
palette3 = sns.color_palette("Set2")      # 集合2
palette4 = sns.color_palette("husl", 8)   # 均匀分布的8种颜色

# 连续调色板（适用于连续数据）
sequential_palette = sns.color_palette("Blues", 10)  # 蓝色渐变
diverging_palette = sns.color_palette("RdBu_r", 10)  # 红蓝发散

# 可视化调色板
sns.palplot(palette1)
plt.title("Deep Palette")
plt.show()
```

## 3. Plotly 颜色库

```python
import plotly.express as px
import plotly.graph_objects as go

# Plotly 内置颜色序列
print("Plotly定性颜色序列:")
print(px.colors.qualitative.__dict__.keys())

print("Plotly顺序颜色序列:")
print(px.colors.sequential.__dict__.keys())

# 使用示例
qualitative_colors = px.colors.qualitative.Set1  # 分类颜色
sequential_colors = px.colors.sequential.Viridis  # 顺序颜色

print("Set1颜色:", qualitative_colors[:5])
print("Viridis颜色:", sequential_colors[:3])
```

## 4. 完整的颜色使用示例

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def demonstrate_colors():
    # 创建示例数据
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    values = np.random.rand(8, 5)
    
    # 1. 使用Matplotlib基本颜色
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    basic_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (cat, color) in enumerate(zip(categories, basic_colors)):
        plt.bar(i, values[i, 0], color=color, label=cat)
    plt.title('Matplotlib 基本颜色')
    plt.legend()
    
    # 2. 使用Seaborn调色板
    plt.subplot(2, 3, 2)
    seaborn_colors = sns.color_palette("husl", 8)
    for i, (cat, color) in enumerate(zip(categories, seaborn_colors)):
        plt.bar(i, values[i, 1], color=color, label=cat)
    plt.title('Seaborn HUSL调色板')
    plt.legend()
    
    # 3. 使用Tableau调色板
    plt.subplot(2, 3, 3)
    tableau_colors = sns.color_palette("tab10", 8)
    for i, (cat, color) in enumerate(zip(categories, tableau_colors)):
        plt.bar(i, values[i, 2], color=color, label=cat)
    plt.title('Tableau 调色板')
    plt.legend()
    
    # 4. 使用Set调色板
    plt.subplot(2, 3, 4)
    set_colors = sns.color_palette("Set2", 8)
    for i, (cat, color) in enumerate(zip(categories, set_colors)):
        plt.bar(i, values[i, 3], color=color, label=cat)
    plt.title('Set2 调色板')
    plt.legend()
    
    # 5. 使用深色调色板
    plt.subplot(2, 3, 5)
    deep_colors = sns.color_palette("deep", 8)
    for i, (cat, color) in enumerate(zip(categories, deep_colors)):
        plt.bar(i, values[i, 4], color=color, label=cat)
    plt.title('Deep 调色板')
    plt.legend()
    
    # 6. 使用Pastel调色板
    plt.subplot(2, 3, 6)
    pastel_colors = sns.color_palette("pastel", 8)
    for i, (cat, color) in enumerate(zip(categories, pastel_colors)):
        plt.bar(i, values[i, 0], color=color, label=cat)
    plt.title('Pastel 调色板')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

demonstrate_colors()
```

## 5. 高级颜色管理工具

```python
import matplotlib.colors as mcolors
import seaborn as sns

class ColorManager:
    """颜色管理器，方便调用不同调色板"""
    
    def __init__(self):
        self.palettes = {
            # Seaborn 调色板
            'deep': sns.color_palette("deep"),
            'husl': lambda n: sns.color_palette("husl", n),
            'Set1': lambda n: sns.color_palette("Set1", n),
            'Set2': lambda n: sns.color_palette("Set2", n),
            'Set3': lambda n: sns.color_palette("Set3", n),
            'pastel': sns.color_palette("pastel"),
            'dark': sns.color_palette("dark"),
            
            # Tableau 调色板
            'tab10': sns.color_palette("tab10"),
            'tab20': sns.color_palette("tab20"),
            
            # 顺序调色板
            'blues': lambda n: sns.color_palette("Blues", n),
            'viridis': lambda n: sns.color_palette("viridis", n),
            'plasma': lambda n: sns.color_palette("plasma", n),
        }
    
    def get_palette(self, name, n_colors=10):
        """获取指定名称的调色板"""
        palette_func = self.palettes.get(name)
        if palette_func:
            if callable(palette_func):
                return palette_func(n_colors)
            return palette_func[:n_colors]
        else:
            # 默认返回husl调色板
            return sns.color_palette("husl", n_colors)
    
    def show_all_palettes(self):
        """显示所有可用的调色板"""
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, palette_func) in enumerate(list(self.palettes.items())[:12]):
            if callable(palette_func):
                colors = palette_func(8)
            else:
                colors = palette_func[:8]
            
            sns.palplot(colors, ax=axes[i])
            axes[i].set_title(name)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.show()

# 使用示例
cm = ColorManager()
colors = cm.get_palette('Set1', 5)
print("Set1调色板(5色):", colors)

# 显示所有调色板
cm.show_all_palettes()
```

## 6. 推荐的最佳实践

```python
def get_optimal_colors(n, palette_type='qualitative'):
    """
    根据数据类别数量获取最优颜色方案
    """
    if palette_type == 'qualitative':
        if n <= 10:
            return sns.color_palette("tab10", n)
        elif n <= 20:
            return sns.color_palette("tab20", n)
        else:
            return sns.color_palette("husl", n)
    
    elif palette_type == 'sequential':
        return sns.color_palette("viridis", n)
    
    elif palette_type == 'diverging':
        return sns.color_palette("RdBu_r", n)

# 使用示例
colors_5 = get_optimal_colors(5)  # 5个分类的最佳颜色
colors_15 = get_optimal_colors(15)  # 15个分类的最佳颜色
```

## 主要颜色库总结：

1. **Matplotlib 基本颜色**：简单直接，适合基础需求
2. **Seaborn 调色板**：专业美观，颜色协调性好
3. **Tableau 颜色**：商业可视化常用，对比度好
4. **Plotly 颜色**：交互式图表专用
5. **HUSL 颜色空间**：颜色分布均匀，相邻色差异大

**推荐**：对于科研和数据分析，建议使用 Seaborn 的调色板，特别是 `husl`、`Set1-3` 和 `tab10/20`，它们在视觉区分度和美观度方面都有很好的平衡。