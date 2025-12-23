你遇到的这个问题在管理多个独立Python项目时非常常见。当工作目录设置为 `models` 的父目录时，各个子项目内部的相对导入会因为 Python 的模块搜索路径 (`sys.path`) 不包含这些子项目目录而失败。

别担心，有几种优雅的方式可以解决这个问题。让我用一个表格帮你快速了解各方案的特点：

| 解决方案               | 核心思路                                     | 优点                                           | 缺点                                           | 适用场景                                   |
| :--------------------- | :------------------------------------------- | :--------------------------------------------- | :--------------------------------------------- | :----------------------------------------- |
| **修改 `sys.path`**    | 运行时动态添加子项目路径到Python模块搜索路径   | 简单直接，无需改变项目结构                     | 需在每个入口文件或管理脚本中添加代码，可能重复 | 快速调试、脚本开发、临时性需求             |
| **设置 `PYTHONPATH`**  | 通过环境变量指定额外的模块搜索路径             | 一次设置，对所有项目生效，无需修改代码         | 依赖系统配置，不同操作系统设置方式不同 | 长期开发、需要多个项目共享模块         |
| **优化项目结构**       | 将每个子项目打造成真正的可安装包（Package）    | 结构清晰，依赖管理明确，最规范                 | 需要为每个子项目创建 `setup.py`，改动稍大      | 长期维护的大型项目、计划分发或部署的项目 |
| **使用 `importlib`**   | 动态加载模块，避免直接修改全局 `sys.path`    | 灵活性强，避免路径冲突和命名冲突                   | 语法相对复杂，可读性稍差                       | 需要动态加载特定模块的场景                 |

下面是每种方案的详细说明和示例。

### 📝 1. 修改 `sys.path`（运行时动态添加路径）

这是最直接的方法，在你的**主启动脚本**或**每个子项目的入口文件**中，在导入任何子项目模块之前，将子项目的路径添加到 `sys.path`。

```python
import os
import sys

# 获取当前脚本所在目录（models的父目录）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 假设你要运行 Practical-RIFE 下的某个脚本
project_path = os.path.join(current_dir, "models", "Practical-RIFE")

# 将子项目路径添加到 sys.path
sys.path.append(project_path)

# 现在可以尝试导入子项目中的模块了
try:
    from video_interpolation_operator import VideoInterpolationOperator
    print("导入成功!")
except ImportError as e:
    print(f"导入失败: {e}")
```

**优点**：简单快速，无需改变项目结构。
**缺点**：需要在每个入口文件或管理脚本中添加代码；若路径过多，管理起来可能稍显混乱。

### 🔧 2. 设置 `PYTHONPATH` 环境变量

通过设置 `PYTHONPATH` 环境变量，可以告诉 Python 解释器在哪些额外目录中搜索模块。

*   **Linux/MacOS** (在终端中执行):
    ```bash
    export PYTHONPATH=/path/to/your/models/Practical-RIFE:/path/to/your/models/AMT:$PYTHONPATH
    ```
*   **Windows** (在CMD或PowerShell中执行):
    ```cmd
    set PYTHONPATH=C:\path\to\your\models\Practical-RIFE;C:\path\to\your\models\AMT;%PYTHONPATH%
    ```

**优点**：一次设置，对所有项目生效；无需修改Python代码。
**缺点**：依赖系统环境变量，不同操作系统设置方式不同；在共享环境或部署时可能需要额外配置脚本。

### 📦 3. 优化项目结构（推荐长期方案）

最规范的做法是将每个子项目（如 `Practical-RIFE`）打造成一个**真正的Python包**。这意味着每个子项目都需要有自己的 `setup.py` 和 `pyproject.toml` 文件，并且通过 `pip install -e .` 以“可编辑”模式安装到你的Python环境中。

1.  **在每个子项目根目录创建 `setup.py`**:
    ```python
    # models/Practical-RIFE/setup.py
    from setuptools import setup, find_packages

    setup(
        name="practical-rife",
        version="0.1.0",
        packages=find_packages(),
        # ... 其他元信息
    )
    ```
2.  **安装子项目**：
    在你的虚拟环境中，进入到 `Practical-RIFE` 目录下，运行：
    ```bash
    pip install -e .
    ```
    这会将 `Practical-RIFE` 链接到你的Python环境里，之后你就可以在任何地方通过 `from video_interpolation_operator import VideoInterpolationOperator` 来导入，就像导入标准库一样。

**优点**：依赖关系明确；项目结构清晰规范；移植和部署方便。
**缺点**：需要为每个子项目进行初始配置。

### 🔄 4. 使用 `importlib` 动态导入

对于需要非常灵活地动态加载模块的场景，可以使用 `importlib` 模块。

```python
import importlib.util
import os

# 指定模块的绝对路径
module_path = os.path.join(os.path.dirname(__file__), "models", "Practical-RIFE", "video_interpolation_operator.py")

# 创建模块规格
spec = importlib.util.spec_from_file_location("video_interpolation_operator", module_path)
if spec is None:
    raise ImportError(f"无法从文件创建规格: {module_path}")

# 根据规格创建模块
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module) # 执行模块代码
except FileNotFoundError as e:
    raise ImportError(f"无法加载模块: {e}") from e

# 现在可以使用模块中的类了
VideoInterpolationOperator = module.VideoInterpolationOperator
```

**优点**：非常灵活，无需修改 `sys.path`，避免了全局路径污染和潜在的命名冲突。
**缺点**：语法相对复杂，可读性不如标准导入方式。

### 💡 通用建议

*   **使用虚拟环境**：**强烈建议**为你的整个项目（`models` 的父目录）创建一个独立的虚拟环境（如 `python -m venv myenv`），并在该环境下安装所有子项目和管理依赖。这可以完美隔离不同项目间的包版本冲突。
    ```bash
    # 创建虚拟环境
    python -m venv myenv
    # 激活虚拟环境 (Linux/MacOS)
    source myenv/bin/activate
    # 激活虚拟环境 (Windows)
    myenv\Scripts\activate
    ```
*   **处理相对导入**：确保每个子项目目录都是一个**有效的Python包**，即包含 `__init__.py` 文件（可以是空文件）。这对于其内部的相对导入（如 `from . import module`）是必需的。
*   **避免循环导入**：在设计代码时，注意避免模块间的循环导入（A导入B，B又导入A），这会导致导入错误。

### 💎 如何选择？

*   若**快速调试**或写**独立脚本**，尝试 **`修改 sys.path`**。
*   若项目**长期发展**或需要**良好维护**，强烈推荐花些时间 **`优化项目结构`**，这是最根本的解决方案。
*   若需要在**不同机器**或**不同 shell 会话**中稳定工作，**`设置 PYTHONPATH`** 是个不错的选择。
*   若需要**精细控制模块加载过程**或开发插件化系统，可以考虑 **`importlib`**。

希望这些方案能帮助你优雅地解决问题！