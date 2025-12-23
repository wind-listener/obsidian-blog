在 Linux/Unix 系统中，解压 ZIP 格式的压缩包通常使用 `unzip` 命令。它支持解压 `.zip` 文件，还能查看压缩包内容、测试完整性等。以下是其详细用法：


### **基本语法**
```bash
unzip [选项] 压缩包名.zip [文件/目录...]  # 可指定解压部分内容，默认解压全部
```


### **常用选项**
| 选项 | 说明 |
|------|------|
| `-l` | 列出压缩包内的文件/目录（不解压） |
| `-v` | 详细模式，列出文件信息（大小、压缩率等） |
| `-t` | 测试压缩包完整性（检查是否损坏） |
| `-d 目录` | 指定解压到目标目录（目录不存在会自动创建） |
| `-q` | 安静模式，不显示解压过程信息 |
| `-o` | 覆盖已存在的文件（无需确认） |
| `-n` | 不覆盖已存在的文件（遇到重复文件则跳过） |
| `-j` | 仅解压文件内容，忽略压缩包内的目录结构（所有文件放当前目录） |
| `-x 文件` | 排除指定文件（不解压该文件） |


### **典型示例**

#### 1. 解压整个压缩包到当前目录
```bash
unzip test.zip  # 解压 test.zip 中所有内容到当前目录
```

#### 2. 解压到指定目录
```bash
unzip docs.zip -d ./output  # 将 docs.zip 解压到 ./output 目录（自动创建 output）
```

#### 3. 查看压缩包内容（不解压）
```bash
unzip -l backup.zip  # 列出 backup.zip 中的所有文件和目录
```

#### 4. 测试压缩包是否损坏
```bash
unzip -t secret.zip  # 检查 secret.zip 的完整性，输出是否正常
```

#### 5. 解压单个文件（从压缩包中提取指定文件）
```bash
unzip images.zip pics/photo.jpg  # 仅从 images.zip 中解压 pics/photo.jpg
```

#### 6. 覆盖已存在的文件（无需确认）
```bash
unzip -o update.zip  # 解压 update.zip，覆盖当前目录中已存在的同名文件
```

#### 7. 排除指定文件（不解压某文件）
```bash
unzip data.zip -x temp.log  # 解压 data.zip 中所有内容，但排除 temp.log
```

#### 8. 忽略目录结构解压
```bash
unzip -j archive.zip  # 解压 archive.zip 中所有文件到当前目录，不保留原目录结构
```

#### 9. 解压加密压缩包（需输入密码）
```bash
unzip secure.zip  # 若压缩包加密，会提示输入密码后解压
```


### **注意事项**
- 若系统未安装 `unzip`，需先安装（如 Ubuntu/Debian：`sudo apt install unzip`；CentOS：`sudo yum install unzip`）。
- 解压包含中文文件名的 ZIP 包时，若出现乱码，可尝试添加 `-O GBK` 选项（针对 Windows 生成的 ZIP 包）：`unzip -O GBK 中文压缩包.zip`。
- 对于大压缩包，`-q` 选项可减少输出干扰，`-v` 可查看详细进度。

通过 `unzip` 的选项组合，可灵活处理 ZIP 压缩包的解压、查看和校验需求。