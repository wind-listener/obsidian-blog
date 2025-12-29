---
title: "Conda 常用命令总结"
date: 2025-08-07
draft: false
---

#虚拟环境 #包管理 

## 环境管理

### 创建环境
```bash
conda create --name myenv python=3.8   # 创建指定Python版本的环境
conda create --prefix /path/to/env     # 创建到自定义路径的环境
conda create --clone base --name myenv # 克隆基础环境
```

### 激活/停用环境
```bash
conda activate myenv      # 激活环境
conda deactivate          # 停用当前环境
```

### 删除环境
```bash
conda env remove --name myenv          # 删除指定环境
conda env remove --prefix /path/to/env # 删除自定义路径环境
```

### 环境列表
```bash
conda info --envs       # 查看所有环境
conda list              # 查看当前环境的包列表
conda list --name myenv # 查看指定环境的包列表
```

## 包管理

### 安装/卸载包
```bash
conda install numpy=1.19.2      # 安装指定版本包
conda install -c conda-forge pkg # 从特定channel安装
conda remove numpy               # 卸载包
```

### 更新包
```bash
conda update numpy       # 更新单个包
conda update --all       # 更新所有包
conda update conda       # 更新conda本身
```

### 搜索包
```bash
conda search numpy       # 搜索可用版本
conda search --full-name pkg # 精确搜索
```

## 配置与维护

### 配置管理
```bash
conda config --show              # 查看所有配置
conda config --add channels conda-forge  # 添加channel
conda config --remove channels conda-forge # 删除channel
```

### 清理缓存
```bash
conda clean --all       # 删除所有缓存文件
conda clean --packages  # 删除未使用的包
conda clean --tarballs  # 删除下载的tarballs
```

### 环境导出/导入
```bash
conda env export > environment.yml  # 导出环境配置
conda env create -f environment.yml # 从文件创建环境
```

## 实用技巧

### 跨平台共享环境
```bash
conda env export --no-builds > environment.yml # 忽略平台特定依赖
```

### 快速复制环境
```bash
conda create --name newenv --clone oldenv
```

### 查看conda信息
```bash
conda info     # 查看conda系统信息
conda --version # 查看conda版本
```

> **提示**：使用`-n`指定环境名，`-p`指定环境路径。对于需要管理员权限的操作，可能需要添加`sudo`。
