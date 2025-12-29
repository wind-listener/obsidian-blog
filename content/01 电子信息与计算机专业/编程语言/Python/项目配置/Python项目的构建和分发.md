---
title: "Python项目的构建和分发"
date: 2025-08-07
draft: false
---

运行 python -m build 的效果是**为 Python 项目生成分发包**，包括**源码分发包（Source Distribution,** .tar.gz**）和构建分发包（Wheel,** .whl**）**。这些分发包可用于将项目上传到 PyPI（Python Package Index）或分发给其他用户。

  

**python -m build 的作用**

1. **生成分发包**：

• **Source Distribution (SDist)**：包含源码和必要的构建信息。

• **Wheel (BDist)**：一种二进制分发包，用户可以快速安装，不需要重新编译。

2. **构建结果存储位置**：

• 默认会将生成的分发包放在 dist/ 目录下。

  

**使用前提**

  

**1. 安装 build 工具**

  

python -m build 是通过第三方工具包 build 实现的。

  

安装方式：

  

pip install build

  

**2. 项目需要有构建配置文件**

  

必须有 pyproject.toml 文件，作为构建配置的入口，符合 [PEP 517](https://peps.python.org/pep-0517/) 和 [PEP 518](https://peps.python.org/pep-0518/) 标准。

  

一个典型的 pyproject.toml 文件示例如下：

  

[build-system]

requires = ["setuptools>=42", "wheel"]

build-backend = "setuptools.build_meta"

  

[project]

name = "my_project"

version = "0.1.0"

description = "A sample Python project"

authors = [

    {name = "Your Name", email = "your.email@example.com"}

]

dependencies = ["requests>=2.25.0"]

  

**基本用法**

  

**1. 构建项目**

  

在项目根目录下运行：

  

python -m build

  

构建完成后，会在项目的 dist/ 目录中生成两个文件：

• my_project-0.1.0.tar.gz（源码分发包）

• my_project-0.1.0-py3-none-any.whl（Wheel 分发包）

  

**2. 指定只生成某种分发包**

• 只生成 Wheel 分发包：

  

python -m build --wheel

  

  

• 只生成源码分发包：

  

python -m build --sdist

  

**生成后的文件用途**

1. **源码分发包 (**.tar.gz**)**

• 包含源码文件和构建脚本。

• 下载后需要编译安装。

• 常见于需要自定义构建流程的项目。

2. **Wheel 分发包 (**.whl**)**

• 预编译的二进制包，安装速度快。

• 直接通过 pip install 安装：

  

pip install my_project-0.1.0-py3-none-any.whl

  

**适用场景**

1. **发布到 PyPI**

配合 twine 工具将生成的分发包上传到 PyPI：

  

pip install twine

twine upload dist/*

  

  

2. **离线安装**

将生成的 .whl 或 .tar.gz 文件分发给其他用户，在目标机器上安装。

3. **测试分发包**

构建后可以测试是否正确包含所有文件：

  

pip install dist/my_project-0.1.0-py3-none-any.whl

  

**构建失败的常见原因**

1. **缺少** pyproject.toml **文件**

确保项目根目录下有 pyproject.toml 文件，并正确配置 build-system 部分。

2. **未安装依赖的构建工具**

如果项目使用 setuptools 或 wheel，确保它们已安装：

  

pip install setuptools wheel

  

  

3. **源文件缺失**

确保 MANIFEST.in 文件或 setuptools 自动包含规则正确，避免漏掉必要的文件。

  

**总结**

  

python -m build 是现代 Python 项目的标准构建工具，简化了生成分发包的过程。通过生成 .tar.gz 和 .whl 文件，它为项目发布、分发和安装提供了便捷的途径。