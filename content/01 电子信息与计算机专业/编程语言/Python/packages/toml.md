---
title: "toml"
date: 2025-08-07
draft: false
---

[[深入对比TOML，JSON和YAML]]

TOML（Tom’s Obvious, Minimal Language）是一种简洁易读的配置文件格式，常用于项目配置文件（如 pyproject.toml）。它使用键值对、分层结构以及易读的语法，支持多种数据类型。

以下是 TOML 文件语法的详细介绍：

  

**基本结构**

  

TOML 文件由以下几个部分组成：

1. **键值对**
2. **数据类型**

3. **表（Sections）**

4. **数组表（Array of Tables）**

5. **注释**

  

**1. 键值对**

  

键和值以 key = value 的形式表示，等号两侧用空格分隔。

```toml
key = "value"  # 字符串值
pi = 3.14       # 数字值
enabled = true  # 布尔值
```

**注意：**

• 键名只能包含字母、数字、下划线（_）、短横线（-）。

• 键名可以用双引号括起来，例如 "key-name" = "value"。

  

**2. 数据类型**

  

**字符串**

  

支持两种字符串类型：

1. **基本字符串**：用双引号 " 包裹。

  

name = "Tom"

  

  

2. **多行字符串**：用三个双引号 """ 包裹，支持换行。

  

text = """

This is a multi-line

string example.

"""

  

  

  

**数字**

  

支持整数和浮点数。

  

integer = 42

float = 3.14159

  

**布尔值**

  

true 或 false。

  

is_enabled = true

  

**日期和时间**

  

支持 ISO 8601 格式：

  

date = 2025-01-09T15:30:00Z  # UTC 时间

local_date = 2025-01-09  # 本地日期

  

**数组**

  

用方括号表示，数组内元素类型必须一致。

  

numbers = [1, 2, 3, 4]

names = ["Alice", "Bob", "Charlie"]

  

**3. 表（Sections）**

  

表用方括号 [table_name] 定义，用于分组键值对。

  

[server]

host = "localhost"

port = 8080

  

**嵌套表**：

使用 . 表示嵌套关系。

  

[database]

[database.connection]

host = "127.0.0.1"

port = 5432

  

等价于：

  

database = { connection = { host = "127.0.0.1", port = 5432 } }

  

**4. 数组表（Array of Tables）**

  

数组表用于定义具有相同结构的多个表，使用 [[table_name]]。

  

[[products]]

name = "Product 1"

price = 10.99

  

[[products]]

name = "Product 2"

price = 20.99

  

生成的结构：

  

{

  "products": [

    {"name": "Product 1", "price": 10.99},

    {"name": "Product 2", "price": 20.99}

  ]

}

  

**5. 注释**

  

注释以 # 开头，支持行尾注释。

  

# 这是一个注释

name = "Tom"  # 键值对后的注释

  

**完整示例**

  

# 基本信息

title = "TOML Example"

version = 1.0

  

# 数值

[settings]

pi = 3.14159

enabled = true

dates = ["2025-01-01", "2025-01-02"]

  

# 嵌套表

[database]

type = "PostgreSQL"

[database.connection]

host = "localhost"

port = 5432

username = "admin"

password = "secret"

  

# 数组表

[[users]]

name = "Alice"

age = 30

  

[[users]]

name = "Bob"

age = 25

  

**语法特点**

1. **易读性**：语法简单，适合人类手动编辑。

2. **数据一致性**：强制数组元素类型一致，避免歧义。

3. **结构清晰**：支持分组和嵌套，适合配置复杂系统。

  

**常见应用**

• **配置文件**：如 pyproject.toml、Cargo.toml（Rust）。

• **数据序列化**：替代 JSON 或 YAML，用于存储结构化数据。

  

TOML 简洁明了，适合开发者和工具构建的项目配置文件场景。