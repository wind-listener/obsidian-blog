---
title: "zip"
date: 2025-10-17
draft: false
---

`zip` 是 Linux/Unix 系统中常用的文件压缩与归档工具，用于创建 ZIP 格式的压缩包（.zip 文件），支持跨平台（Windows、macOS 等均兼容）。以下是其详细用法：


### **基本语法**
```bash
zip [选项] 压缩包名.zip 源文件/目录...
```


### **常用选项**
| 选项 | 说明 |
|------|------|
| `-r` | 递归压缩目录（必须用于目录压缩，否则只压缩目录名本身） |
| `-q` | 安静模式，不显示压缩过程信息 |
| `-v` | 详细模式，显示压缩过程的详细信息（文件大小、压缩率等） |
| `-m` | 压缩后删除原文件（move，仅保留压缩包） |
| `-u` | 更新压缩包：仅添加新文件或修改过的文件（不会重复压缩旧文件） |
| `-d` | 从压缩包中删除指定文件（用法：`zip -d 压缩包.zip 文件名`） |
| `-e` | 创建加密压缩包（需输入密码，解压时也需密码） |
| `-P 密码` | 直接指定加密密码（避免交互式输入，适合脚本，但密码会明文显示） |
| `-j` | 仅压缩文件内容，不保留目录结构（所有文件直接放在压缩包根目录） |
| `-0` | 仅归档不压缩（速度最快，文件大小不变） |
| `-9` | 最高压缩比（速度最慢，压缩率最高，默认是 `-6`） |


### **典型示例**

#### 1. 压缩单个文件
```bash
zip test.zip file1.txt  # 将 file1.txt 压缩为 test.zip
```

#### 2. 压缩多个文件
```bash
zip docs.zip report.pdf notes.txt  # 压缩多个文件到 docs.zip
```

#### 3. 压缩目录（必须加 `-r`）
```bash
zip -r backup.zip ./data  # 递归压缩 data 目录及其所有内容
```

#### 4. 加密压缩（交互式输入密码）
```bash
zip -e secret.zip private.txt  # 生成加密压缩包，需输入密码
```

#### 5. 加密压缩（直接指定密码，适合脚本）
```bash
zip -P mypassword secure.zip sensitive.doc  # 密码为 mypassword（明文风险）
```

#### 6. 更新压缩包（添加新文件或修改过的文件）
```bash
zip -u backup.zip new_file.txt  # 向 backup.zip 中添加/更新 new_file.txt
```

#### 7. 从压缩包中删除文件
```bash
zip -d backup.zip old_file.txt  # 从 backup.zip 中删除 old_file.txt
```

#### 8. 不保留目录结构压缩
```bash
zip -j images.zip ./pics/*.jpg  # 压缩 pics 目录下的所有 jpg，但压缩包内无 pics 目录
```

#### 9. 最高压缩比（牺牲速度换大小）
```bash
zip -9 -r big_compress.zip ./large_dir  # 最高压缩级别处理 large_dir
```

#### 10. 压缩后删除原文件
```bash
zip -m temp.zip temp.log  # 压缩 temp.log 为 temp.zip 后，删除原 temp.log
```


### **注意事项**
- ZIP 格式支持单个文件最大 4GB，压缩包总大小无限制（需文件系统支持）。
- 解压 ZIP 包需用 `unzip` 命令（需预先安装，如 `sudo apt install unzip`）。
- 加密压缩的密码在传输或存储时需注意安全，`-P` 选项会在命令历史中留下明文密码，慎用。

通过组合上述选项，可满足大多数文件归档与压缩需求。

### 解压指令 [[unzip]]