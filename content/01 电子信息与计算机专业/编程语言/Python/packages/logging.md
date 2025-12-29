---
title: "logging"
date: 2025-08-07
draft: false
---

`logging` 是 Python 标准库中的一个模块，用于记录和管理应用程序的日志信息。相较于直接使用 `print`，`logging` 模块提供了更灵活、更强大的日志记录功能，能够控制日志的输出格式、级别、目标位置（如文件、控制台），以及自动处理日志文件轮换等操作。
相关链接：
[官方文档](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)

### 一、`logging` 模块的基本概念

1. **Logger**：
   - `Logger` 是日志系统的核心对象，负责记录日志消息。每个日志记录都与一个 `Logger` 实例关联，且可以有不同的日志级别和输出目标。

2. **Log Level（日志级别）**：
   - 日志级别用于指定日志的重要性，`logging` 提供了五种常用的日志级别：
     - `DEBUG`：详细信息，通常只在诊断问题时使用。
     - `INFO`：确认一切按预期运行的消息。
     - `WARNING`：表示可能会出现的问题，提醒你要注意。
     - `ERROR`：表示程序在执行某些操作时遇到了问题，但程序没有停止运行。
     - `CRITICAL`：表示严重的错误，可能导致程序无法继续运行。

3. **Handler（处理器）**：
   - `Handler` 决定了日志消息的输出目标，比如输出到文件、控制台或远程服务器。`logging` 支持多种处理器，如 `StreamHandler`（输出到控制台）、`FileHandler`（输出到文件）等。

4. **Formatter（格式化器）**：
   - `Formatter` 用于指定日志消息的显示格式，比如是否包括时间戳、日志级别等信息。

5. **Filter（过滤器）**：
   - `Filter` 可以进一步控制哪些日志消息会被处理器记录，通常用于根据自定义规则过滤日志。

### 二、`logging` 的基本使用步骤

1. **基本配置**：
   可以使用 `logging.basicConfig` 配置日志系统，指定日志的级别、格式以及输出目标。

```python
import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，DEBUG级别的日志不会显示
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志的输出格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
    handlers=[logging.StreamHandler()]  # 指定输出到控制台
)

# 开始记录日志
logging.debug("这是一条调试信息")
logging.info("这是一条普通信息")
logging.warning("这是一个警告信息")
logging.error("这是一个错误信息")
logging.critical("这是一个严重错误信息")
```

### 输出示例：
```plaintext
2024-09-24 15:35:12 - INFO - 这是一条普通信息
2024-09-24 15:35:12 - WARNING - 这是一个警告信息
2024-09-24 15:35:12 - ERROR - 这是一个错误信息
2024-09-24 15:35:12 - CRITICAL - 这是一个严重错误信息
```

但是注意⚠️，什么时候不适合使用 basicConfig：
• basicConfig **是全局性的**：一旦调用 basicConfig，它会应用于全局日志配置。如果你想为每个 Worker 实例创建单独的 logger，且可能有不同的日志级别或输出目标，basicConfig 不能满足这个需求。
• **自定义多个** Handler：当你需要为不同的 logger 配置不同的 Handler（如文件处理器、控制台处理器等）时，basicConfig 过于简单，无法为不同的实例分别配置。



2. **Logger 的基本用法**：
   你可以创建多个 `Logger` 实例来记录不同模块的日志。

```python
import logging

# 创建一个自定义的Logger
logger = logging.getLogger('my_logger')

# 设置Logger的日志级别
logger.setLevel(logging.DEBUG)

# 创建Handler，设置输出位置和日志格式
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 将Handler添加到Logger中
logger.addHandler(handler)

# 记录日志
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误信息")
```

### 输出示例：
```plaintext
2024-09-24 15:35:12 - my_logger - DEBUG - 调试信息
2024-09-24 15:35:12 - my_logger - INFO - 普通信息
2024-09-24 15:35:12 - my_logger - WARNING - 警告信息
2024-09-24 15:35:12 - my_logger - ERROR - 错误信息
2024-09-24 15:35:12 - my_logger - CRITICAL - 严重错误信息
```

### 三、`logging` 配置详解

1. **日志级别控制**：
   日志系统会过滤比设置级别低的日志。例如设置为 `INFO` 级别，则 `DEBUG` 级别的日志不会显示。

   ```python
   logging.basicConfig(level=logging.WARNING)
   ```

2. **日志格式化**：
   格式化器定义了日志的输出格式。常用的格式占位符有：
   - `%(asctime)s`：日志的时间
   - `%(name)s`：Logger 的名称
   - `%(levelname)s`：日志的级别
   - `%(message)s`：日志消息内容

   例如：

   ```python
   formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
   ```

3. **Handler（处理器）**：
   `Handler` 定义了日志的输出目标。常用的处理器包括：
   - `StreamHandler`：将日志输出到控制台。
   - `FileHandler`：将日志输出到文件中。

   例如，将日志记录到文件中：

   ```python
   file_handler = logging.FileHandler('logfile.log')
   file_handler.setLevel(logging.INFO)
   ```

4. **日志轮换**：
   `logging` 还支持日志文件轮换，通过 `RotatingFileHandler` 来限制日志文件的大小，并可以自动备份日志。

   ```python
   from logging.handlers import RotatingFileHandler

   # 创建一个RotatingFileHandler, 10MB为最大文件大小，最多保留5个旧文件
   handler = RotatingFileHandler('logfile.log', maxBytes=10*1024*1024, backupCount=5)
   logger.addHandler(handler)
   ```

### 四、综合示例：多处理器输出日志并使用日志轮换

```python
import logging
from logging.handlers import RotatingFileHandler

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建一个控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个日志文件处理器，并设置轮换机制
file_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=3)
file_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器中
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 测试日志
logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
logger.critical("这是严重错误信息")
```

### 五、未捕获异常自动记录

通过全局捕获异常的 `sys.excepthook`，可以将未捕获的异常记录到日志中：

```python
import logging
import sys

# 配置日志记录器
logging.basicConfig(level=logging.ERROR, filename='error.log')

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# 全局未捕获异常的处理
sys.excepthook = handle_exception

# 测试未捕获异常
raise ValueError("测试未捕获的异常")
```

### 总结

- `logging` 提供了强大且灵活的日志管理功能，适用于简单到复杂的应用。
- 你可以控制日志的输出位置、格式、轮换机制，以及日志级别过滤。
- 通过捕获未处理的异常，确保应用程序在遇到异常时能够记录下详细信息，便于调试和维护。

根据项目需求，你可以简单地使用 `basicConfig` 来快速记录日志，或者配置复杂的 `Logger` 来满足更高级的日志需求。