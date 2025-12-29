---
title: "pathlib.Path"
date: 2025-08-07
draft: false
---


pathlib.Path 是 Python 中用于处理文件路径的强大工具，提供了很多方便的方法来操作文件和目录。以下是一些常用的 Path 方法：

  

**1. Path(name)**

• 用于创建一个 Path 对象。

```python
from pathlib import Path
path = Path('path/to/file.txt')
```

**2. path.name**

• 返回路径的最后一部分，即文件名或最后一个目录名。

```
file_name = path.name  # 'file.txt'
```

**3. path.stem**

• 返回文件名，不包含扩展名。

```python
file_stem = path.stem  # 'file' (去掉了扩展名)
```

**4. path.suffix**

• 返回文件的扩展名（包括点 .）。

```python
file_suffix = path.suffix  # '.txt'
```

**5. path.suffixes**

• 返回文件扩展名的列表（如果文件有多个扩展名）。

```python
file_suffixes = path.suffixes  # ['.tar', '.gz']
```

**6. path.parent**

• 返回路径的父目录（即去掉最后一部分后的路径）。

```
parent_dir = path.parent  # 'path/to'
```

**7. path.parents[n]**

• 返回路径的第 n 个父目录。n 从 0 开始。

```
parent_dir_1 = path.parents[0]  # 'path/to'
parent_dir_2 = path.parents[1]  # 'path'
```

**8. path.exists()**

• 检查路径是否存在（文件或目录）。

```
path_exists = path.exists()  # True 或 False
```

**9. path.is_file()**

• 检查路径是否是一个文件。

```
is_file = path.is_file()  # True 或 False
```

**10. path.is_dir()**

• 检查路径是否是一个目录。

```
is_dir = path.is_dir()  # True 或 False
```

**11. path.is_symlink()**

• 检查路径是否是一个符号链接。

```
is_symlink = path.is_symlink()  # True 或 False
```

**12. path.mkdir(parents=False, exist_ok=False)**

• 创建一个目录。parents=True 表示如果父目录不存在，则创建父目录；exist_ok=True 表示如果目录已经存在，不会抛出异常。

```
path.mkdir(parents=True, exist_ok=True)
```

**13. path.rmdir()**

• 删除一个空目录。

```
path.rmdir()  # 删除目录
```

**14. path.rename(target)**

• 重命名文件或目录。

```
path.rename('new_name.txt')  # 将 'file.txt' 重命名为 'new_name.txt'
```

**15. path.unlink()**

• 删除文件。如果路径是一个符号链接，它会删除链接。

```
path.unlink()  # 删除文件
```

**16. path.touch(exist_ok=False)**

• 创建一个空文件，如果文件已存在，且 exist_ok=False，则会抛出异常。

```
path.touch(exist_ok=True)  # 如果文件不存在则创建它
```

**17. path.open(mode='r', encoding=None)**

• 打开文件并返回文件对象（类似于 open() 函数）。

```
with path.open('r') as f:
    content = f.read()
```

**18. path.iterdir()**

• 返回当前目录下的所有条目（文件和目录）的生成器。

```
for child in path.iterdir():
    print(child)  # 输出当前目录下的所有文件和子目录
```

**19. path.glob(pattern)**

• 返回当前目录下匹配 pattern 的文件路径生成器。可以使用通配符（如 *, ?）。

```
for txt_file in path.glob('*.txt'):
    print(txt_file)  # 输出所有 .txt 文件
```

**20. path.rglob(pattern)**

• 类似于 glob()，但是它递归地查找当前目录及其所有子目录中的匹配文件。

```
for txt_file in path.rglob('*.txt'):
    print(txt_file)  # 输出所有子目录中的 .txt 文件
```

**21. path.resolve()**

• 返回路径的绝对路径，并解析符号链接。

```
absolute_path = path.resolve()  # 获取绝对路径
```

**22. path.relative_to(*other)**

• 返回当前路径相对于其他路径的相对路径。

```
relative_path = path.relative_to('/home/user')  # 获取相对路径
```

**23. path.with_name(name)**

• 返回路径的副本，将文件名部分替换为给定的 name。

```
new_path = path.with_name('new_name.txt')  # 替换文件名
```

**24. path.with_suffix(suffix)**

• 返回路径的副本，将文件的扩展名部分替换为给定的 suffix。

```
new_path = path.with_suffix('.md')  # 将扩展名修改为 .md
```

**示例：**

```python
from pathlib import Path

# 创建路径对象
path = Path("/home/user/docs/example.txt")

# 获取文件名、父目录、扩展名
print(path.name)      # 输出 'example.txt'
print(path.stem)      # 输出 'example'
print(path.suffix)    # 输出 '.txt'
print(path.parent)    # 输出 '/home/user/docs'

# 检查文件是否存在
if path.exists():
    print(f"{path} exists.")
else:
    print(f"{path} does not exist.")
```

这些方法非常适合用来进行文件路径的操作，尤其是在处理文件系统时。