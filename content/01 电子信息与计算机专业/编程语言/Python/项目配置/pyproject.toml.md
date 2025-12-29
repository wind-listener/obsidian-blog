---
title: "**基本结构**"
date: 2025-08-07
draft: false
---

[[toml]]

pyproject.toml 是 Python 项目的配置文件，定义了项目的元数据、依赖项、构建工具等。它是 PEP 518 和 PEP 621 的一部分，用于统一配置现代 Python 项目。

# **基本结构**
一个典型的 pyproject.toml 文件包含以下部分：

## 1. **[build-system]**
定义项目的构建工具和要求，支持 PEP 518。
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]  # 构建所需的工具及版本， 构建项目所需的依赖。
build-backend = "setuptools.build_meta"  # 构建后端， 指定用于构建项目的工具。
```
常见的 build-backend 值：
• setuptools.build_meta（默认使用 setuptools）
• poetry.core.masonry.api
• flit_core.buildapi

## 2. **[tool]**  
为特定工具提供配置。例如：
• poetry
• black
• isort
• 自定义工具
```toml
[tool.poetry]
name = "my_project"  # 项目名称
version = "0.1.0"  # 项目版本
description = "A sample Python project"  # 项目描述
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
packages = ["my_project"]  # 项目包含的包
dependencies = { # 运行时依赖。
    python = "^3.8",
    requests = "^2.25.1"
}
dev-dependencies = { # 开发时依赖
    pytest = "^6.2"
    black = "^22.1"
}
```

# **手动或者工具创建 pyproject.toml**
## **1. 手动创建**
  
直接创建 pyproject.toml 文件，并按照上面格式填写。

## **2. 使用工具生成**

• **Poetry**

初始化项目并生成 pyproject.toml：
```bash
poetry init

# 安装依赖并更新到 pyproject.toml：
poetry add <package_name>
poetry add --dev <dev_package_name>
```
  
• **其他工具**
• 使用 cookiecutter 模板生成。
• 使用 IDE（如 PyCharm）的模板。

  

# **用法/ 应用场景**
## **1. 作为构建配置文件**
如果项目需要打包和分发，pyproject.toml 是现代的替代方案，可以取代传统的 setup.py 和 setup.cfg。
```bash
python -m build
```

## **2. 配置开发工具**

常见工具的配置：
```toml
[tool.black] # 配置代码格式化
line-length = 88
target-version = ["py38"]

[tool.isort] # 配置导入排序工具
profile = "black"
line_length = 88

[tool.pytest.ini_options] # 配置测试工具
minversion = "6.0"
addopts = "--strict-markers"
```

## **3. 配置依赖管理**
通过 poetry 或 pip 读取依赖。
```bash
# 安装项目依赖
pip install .
# 安装开发依赖：
pip install .[dev]

# 开源项目中常见使用editable模式
pip install -e .
pip install -e .[dev]
```
  


# **优点**
1. **统一性**：将构建、依赖和工具配置集中在一个文件中。
2. **兼容性**：支持主流的构建工具和依赖管理器。
3. **简洁性**：TOML 格式易读易写，适合手动编辑。

  

# **示例文件**
完整的 pyproject.toml 示例：
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.poetry]
name = "my_project"
version = "0.1.0"
description = "A sample Python project"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
packages = ["my_project"]
dependencies = {
    python = "^3.8", # ^的含义是至少是，保证版本号从左往右第一个不为0的数字一样，后面的可以不同
    requests = "^2.25.1" 
}

dev-dependencies = {
    pytest = "^6.2",
    black = "^22.1"
}

[tool.black]
line-length = 88
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 88
```

# **总结**
pyproject.toml 是现代 Python 项目配置的核心文件。通过它，项目的构建、依赖管理和开发工具配置变得更加统一和规范。在实际使用中，可根据需求选择支持的工具（如 Poetry 或 [[setuptools]]），灵活配置项目结构。