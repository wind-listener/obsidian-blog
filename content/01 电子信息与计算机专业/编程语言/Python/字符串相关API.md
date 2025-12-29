---
title: "字符串相关API"
date: 2025-08-07
draft: false
---

Python 字符串提供了许多实用的 API（方法）来处理和操作字符串。以下是一些常用的字符串方法及其简单说明：

### 1. **字符串操作方法**

- **`len(s)`**：返回字符串的长度（字符数）。
  ```python
  s = "Hello"
  print(len(s))  # 输出: 5
  ```

- **`s.lower()`**：将字符串中的所有字符转换为小写。
  ```python
  s = "Hello"
  print(s.lower())  # 输出: "hello"
  ```

- **`s.upper()`**：将字符串中的所有字符转换为大写。
  ```python
  s = "Hello"
  print(s.upper())  # 输出: "HELLO"
  ```

- **`s.capitalize()`**：将字符串的首字母大写，其他字符小写。
  ```python
  s = "hello"
  print(s.capitalize())  # 输出: "Hello"
  ```

- **`s.title()`**：将字符串中的每个单词的首字母大写，其他字母小写。
  ```python
  s = "hello world"
  print(s.title())  # 输出: "Hello World"
  ```

- **`s.strip()`**：移除字符串两端的空白字符（或其他指定字符）。
  ```python
  s = "  hello  "
  print(s.strip())  # 输出: "hello"
  ```

- **`s.lstrip()`**：移除字符串左侧的空白字符。
  ```python
  s = "  hello"
  print(s.lstrip())  # 输出: "hello"
  ```

- **`s.rstrip()`**：移除字符串右侧的空白字符。
  ```python
  s = "hello  "
  print(s.rstrip())  # 输出: "hello"
  ```

### 2. **查找与替换**

- **`s.find(sub)`**：返回子字符串 `sub` 在字符串 `s` 中第一次出现的索引，如果没有找到则返回 `-1`。
  ```python
  s = "hello world"
  print(s.find("world"))  # 输出: 6
  ```

- **`s.rfind(sub)`**：返回子字符串 `sub` 在字符串 `s` 中最后一次出现的索引，如果没有找到则返回 `-1`。
  ```python
  s = "hello world, world"
  print(s.rfind("world"))  # 输出: 13
  ```

- **`s.index(sub)`**：与 `find()` 类似，但如果 `sub` 不存在会引发 `ValueError` 异常。
  ```python
  s = "hello world"
  print(s.index("world"))  # 输出: 6
  ```

- **`s.replace(old, new)`**：将字符串中的所有 `old` 子字符串替换为 `new` 子字符串。
  ```python
  s = "hello world"
  print(s.replace("world", "Python"))  # 输出: "hello Python"
  ```

### 3. **判断类型**

- **`s.isdigit()`**：判断字符串是否只包含数字字符。
  ```python
  s = "12345"
  print(s.isdigit())  # 输出: True
  ```

- **`s.isalpha()`**：判断字符串是否只包含字母字符。
  ```python
  s = "hello"
  print(s.isalpha())  # 输出: True
  ```

- **`s.isalnum()`**：判断字符串是否只包含字母和数字字符。
  ```python
  s = "hello123"
  print(s.isalnum())  # 输出: True
  ```

- **`s.isspace()`**：判断字符串是否只包含空白字符（空格、换行等）。
  ```python
  s = "   "
  print(s.isspace())  # 输出: True
  ```

- **`s.startswith(prefix)`**：判断字符串是否以指定前缀 `prefix` 开头。
  ```python
  s = "hello world"
  print(s.startswith("hello"))  # 输出: True
  ```

- **`s.endswith(suffix)`**：判断字符串是否以指定后缀 `suffix` 结尾。
  ```python
  s = "hello world"
  print(s.endswith("world"))  # 输出: True
  ```

### 4. **字符串拆分与连接**

- **`s.split(sep)`**：按照指定的分隔符 `sep` 拆分字符串，返回一个列表。如果不指定 `sep`，默认按照空白字符拆分。
  ```python
  s = "hello world"
  print(s.split())  # 输出: ['hello', 'world']
  ```

- **`s.splitlines()`**：按行分割字符串，返回一个列表。
  ```python
  s = "hello\nworld"
  print(s.splitlines())  # 输出: ['hello', 'world']
  ```

- **`sep.join(iterable)`**：用 `sep` 连接一个可迭代对象中的所有字符串，返回一个拼接后的字符串。
  ```python
  words = ['hello', 'world']
  print(" ".join(words))  # 输出: "hello world"
  ```

### 5. **字符串对齐**

- **`s.center(width)`**：将字符串 `s` 居中，并使用空格填充至指定宽度 `width`。
  ```python
  s = "hello"
  print(s.center(10))  # 输出: "  hello   "
  ```

- **`s.ljust(width)`**：将字符串 `s` 左对齐，并使用空格填充至指定宽度 `width`。
  ```python
  s = "hello"
  print(s.ljust(10))  # 输出: "hello     "
  ```

- **`s.rjust(width)`**：将字符串 `s` 右对齐，并使用空格填充至指定宽度 `width`。
  ```python
  s = "hello"
  print(s.rjust(10))  # 输出: "     hello"
  ```

### 6. **其他常用方法**

- **`s.count(sub)`**：返回子字符串 `sub` 在字符串 `s` 中出现的次数。
  ```python
  s = "hello world"
  print(s.count("o"))  # 输出: 2
  ```

- **`s.zfill(width)`**：返回一个字符串，长度为 `width`，原字符串右对齐，左侧用零填充。
  ```python
  s = "42"
  print(s.zfill(5))  # 输出: "00042"
  ```

- **`s.partition(sep)`**：将字符串 `s` 按 `sep` 分为三部分，返回一个三元组 `(head, sep, tail)`。
  ```python
  s = "hello world"
  print(s.partition(" "))  # 输出: ('hello', ' ', 'world')
  ```

- **`s.rpartition(sep)`**：与 `partition` 类似，但从字符串末尾开始搜索 `sep`。
  ```python
  s = "hello world world"
  print(s.rpartition(" "))  # 输出: ('hello world', ' ', 'world')
  ```

这些方法涵盖了常见的字符串操作，可以帮助你处理和操作字符串数据。根据具体需求选择合适的方法即可。