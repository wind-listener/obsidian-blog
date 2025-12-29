---
title: "argparse"
date: 2025-08-08
draft: false
---

argparse 是 Python 标准库中的一个模块，主要用于解析命令行参数和选项。它让我们可以轻松地处理用户通过命令行传递的参数，并自动生成帮助文档。使用 argparse 可以大大简化命令行工具的开发过程。

  

**1. 基本用法**

  

下面是一个简单的例子：

```python
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="A simple argument parser example")

# 添加命令行参数
parser.add_argument('name', type=str, help='Your name')
parser.add_argument('age', type=int, help='Your age')

# 解析命令行参数
args = parser.parse_args()

# 使用传入的参数
print(f"Hello {args.name}, you are {args.age} years old.")
```

**解释：**

• ArgumentParser：创建一个解析器对象，用来处理命令行参数。

• add_argument：定义命令行参数。第一个参数是参数的名称或位置（例如 'name' 和 'age'）。type 定义参数的数据类型，help 提供帮助信息。

• parse_args：解析命令行传入的参数并返回一个包含解析结果的对象 args，我们可以通过 args.name 和 args.age 访问相应的参数值。

  

**2. 可选参数**

  

有些参数是可选的，可以使用 -- 或 - 来传递，通常这些参数有默认值。

```python
import argparse

parser = argparse.ArgumentParser(description="A program with optional arguments")
parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
parser.add_argument('--output', type=str, default='output.txt', help="Output file name")

args = parser.parse_args()

if args.verbose:
    print("Verbose mode enabled")

print(f"Output will be saved to {args.output}")
```

**解释：**

• --verbose 是一个可选参数，action='store_true' 表示如果用户提供了 --verbose，则 args.verbose 会被设置为 True，否则默认为 False。

• --output 是一个可选参数，用户可以通过它指定输出文件名，如果没有提供，则默认值为 'output.txt'。

  

**3. 类型和默认值**

  

argparse 支持多种类型的参数，并且可以为参数设置默认值。

```python
parser.add_argument('--size', type=int, default=10, help="Size of the output")
```

• type=int 会将传入的值转换为整数。

• default=10 表示如果没有提供该参数，args.size 将默认是 10。

  

**4. 位置参数和可选参数**

  

位置参数是必需的，必须按照顺序提供。而可选参数是可选的，通常以 -- 或 - 开头。

```python
parser.add_argument('input', type=str, help="Input file")
parser.add_argument('--output', type=str, help="Output file", default="result.txt")
```

• input 是位置参数，必须在命令行中提供。

• --output 是可选参数，如果不提供将使用默认值 result.txt。

  

**5. 多个参数和列表**

  

可以定义多个值的参数，并将其解析为列表。

```python
parser.add_argument('--files', type=str, nargs='+', help="List of files to process")
```

• nargs='+' 表示接受一个或多个值，args.files 将是一个列表，包含所有传入的文件名。

  

**6. 互斥参数**
`group = parser.add_mutually_exclusive_group()`

互斥参数是一组参数中的一个只能出现一个。当你定义了互斥组后，传入多个互斥参数将会引发错误。

```python
parser = argparse.ArgumentParser(description="Mutually exclusive arguments example")
group = parser.add_mutually_exclusive_group()
group.add_argument('--compress', action='store_true', help="Compress the output")
group.add_argument('--no-compress', action='store_true', help="Do not compress the output")
```

• add_mutually_exclusive_group() 用来定义一个互斥组，组内的参数只能选择一个。

• 传入 --compress 或 --no-compress 中的一个，但不能同时传入。

  

**7. 显示帮助信息**

  

argparse 自动为你生成帮助文档。当用户在命令行中输入 -h 或 --help 时，程序会自动显示如何使用该脚本及其参数：

```bash
$ python script.py --help
usage: script.py [-h] [--verbose] [--output OUTPUT] input

A program with optional arguments

positional arguments:
  input                 Input file

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Enable verbose output
  --output OUTPUT       Output file name
```

**8. 完整示例**

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Command-line tool example")
    
    # 位置参数
    parser.add_argument('input_file', type=str, help="Path to the input file")
    
    # 可选参数
    parser.add_argument('--output_file', type=str, default='output.txt', help="Path to the output file")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose mode")
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行操作
    if args.verbose:
        print(f"Processing input file: {args.input_file}")
    
    with open(args.input_file, 'r') as f:
        content = f.read()
    
    # 输出结果
    with open(args.output_file, 'w') as f:
        f.write(content)
    
    print(f"Content saved to {args.output_file}")

if __name__ == "__main__":
    main()
```

**9. 总结**

• argparse 让你轻松地解析命令行参数。

• 支持位置参数和可选参数。

• 支持类型转换、默认值、多个值的参数、互斥参数等。

• 自动生成帮助文档，提升用户体验。

  

使用 argparse 可以让你的脚本更具灵活性，并且更易于理解和使用。


# `parse_known_args` 用法介绍

`parse_known_args` 是 Python 标准库 `argparse` 模块中的一个方法，用于解析命令行参数，但允许保留无法识别的参数。

## 基本用法

与 `parse_args()` 不同，`parse_known_args()` 会返回一个包含两个元素的元组：
1. 包含已解析参数的命名空间对象
2. 无法识别的参数字符串列表

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args, unknown = parser.parse_known_args(['--foo', '1', '--bar', '2'])

print(args)      # 输出: Namespace(foo='1')
print(unknown)   # 输出: ['--bar', '2']
```

## 使用场景

1. **部分参数解析**：当你只想解析部分参数，而将其他参数传递给其他程序或模块时
2. **子命令处理**：在实现复杂命令行工具时，可能需要将未识别的参数传递给子命令
3. **参数转发**：需要将某些参数转发给其他程序或脚本

## 示例

### 示例1：基本使用

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='increase output verbosity')
args, unknown = parser.parse_known_args(['--verbose', '--unknown', 'value'])

print(f"Known args: {args}")
print(f"Unknown args: {unknown}")
```

输出：
```
Known args: Namespace(verbose=True)
Unknown args: ['--unknown', 'value']
```

### 示例2：与子命令结合

```python
import argparse

# 主解析器
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='debug mode')
subparsers = parser.add_subparsers(dest='command')

# 子命令解析器
parser_foo = subparsers.add_parser('foo')
parser_foo.add_argument('bar', help='bar help')

# 解析
args, unknown = parser.parse_known_args(['--debug', 'foo', 'baz', '--extra'])

print(f"Main args: {args}")
print(f"Unknown args: {unknown}")
```

输出：
```
Main args: Namespace(command='foo', debug=True, bar='baz')
Unknown args: ['--extra']
```

## 注意事项

1. 未识别的参数会被收集到返回的列表中，而不是引发错误
2. 如果需要严格参数检查，应该使用 `parse_args()` 而不是 `parse_known_args()`
3. 在编写脚本时，确保正确处理未知参数，避免意外行为

`parse_known_args` 提供了更大的灵活性，特别适合构建复杂的命令行工具或需要参数转发的情况。