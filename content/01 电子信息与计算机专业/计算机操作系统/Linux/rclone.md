---
title: "rclone"
date: 2025-11-10
draft: false
---

`rclone` 是一款功能强大的命令行工具，用于在本地存储、云存储（如 Google Drive、AWS S3、OneDrive 等）之间同步、复制、迁移数据，支持超过 40 种存储服务。以下是其核心指令用法的详细解释：


### **一、基础概念**
1. **远程存储（Remote）**：指配置好的云存储或网络存储（如 `gdrive:`、`s3:`），需先通过 `rclone config` 配置。
2. **本地路径**：直接使用本地文件路径（如 `/home/user/file` 或 `C:\data`）。
3. **语法格式**：`rclone [选项] 命令 源路径 目标路径`


### **二、核心命令**

#### 1. **配置远程存储：`rclone config`**
用于添加、删除、修改远程存储配置（必做第一步）。  
```bash
rclone config
```
- 交互流程：输入 `n` 新建远程 → 输入名称（如 `mygdrive`）→ 选择存储类型（如 Google Drive 对应编号）→ 按提示完成授权（如输入令牌、密钥等）→ 输入 `q` 退出。
##### 配置文件
- **Linux / macOS**：配置文件默认位于用户主目录下的隐藏目录 `.config/rclone/` 中，路径为：`~/.config/rclone/rclone.conf`
- **Windows**：配置文件默认位于用户目录下的 `AppData/Roaming/rclone/` 中，路径为：`C:\Users\用户名\AppData\Roaming\rclone\rclone.conf`

可以通过以下命令查看 `rclone` 当前使用的配置文件路径：  
```bash
rclone config file
```

`rclone.conf` 是文本文件，内容为 ini 格式，每个远程存储的配置以 `[远程名称]` 为段头，例如：  
```ini
[mygdrive]
type = drive
scope = drive
token = {"access_token":"xxx","refresh_token":"xxx",...}

[mys3]
type = s3
env_auth = false
access_key_id = YOUR_KEY
secret_access_key = YOUR_SECRET
region = us-east-1
```

如果需要使用非默认路径的配置文件，可以通过 `-f` 或 `--config` 选项***临时指定***，例如：  
```bash
# 使用自定义路径的配置文件执行命令
rclone --config /path/to/my_config.conf copy /local remote:path
```

`--config` 选项是“单次临时生效”，而通过环境变量或替换默认文件可以实现“永久生效”。
 在系统中定义环境变量 `RCLONE_CONFIG`，指定你的自定义配置文件路径，这样所有 `rclone` 命令会默认读取该路径。  替换默认文件直接用cp即可。
 
```bash
export RCLONE_CONFIG=/path/to/my_config.conf
echo 'export RCLONE_CONFIG=/path/to/my_config.conf' >> ~/.bashrc && source ~/.bashrc
```

#### 2. **复制文件：`rclone copy`**
将源路径的文件复制到目标路径（仅复制新文件或修改过的文件，不删除目标多余文件）。  
```bash
# 本地 → 远程
rclone copy /home/documents mygdrive:backup/docs

# 远程 → 本地
rclone copy mygdrive:photos ./local_photos

# 远程 → 远程
rclone copy mygdrive:music mys3:media/music
```
- 常用选项：
  - `-P`：显示实时进度条。
  - `-v`：显示详细日志（`-vv` 更详细，用于调试）。
  - `--ignore-existing`：跳过目标已存在的文件。
  - `--max-size 100M`：仅复制小于 100M 的文件。


#### 3. **同步文件：`rclone sync`**
将源路径同步到目标路径（使目标与源完全一致，会删除目标中源没有的文件，**慎用**）。  
```bash
# 本地同步到远程（目标会删除多余文件）
rclone sync /data mygdrive:data_backup -P
```
- 注意：同步是单向的（源 → 目标），建议先使用 `rclone check` 确认差异。


#### 4. **移动文件：`rclone move`**
将源路径的文件移动到目标路径（类似剪切+粘贴）。  
```bash
# 远程文件移动到本地
rclone move mygdrive:temp ./local_temp -P
```
- 选项 `--delete-empty-src-dirs`：移动后删除源的空目录。


#### 5. **列出文件：`rclone ls` / `rclone lsd`**
- `rclone ls [路径]`：列出文件（含大小和名称）。  
  ```bash
  rclone ls mygdrive:docs  # 列出远程目录文件
  ```
- `rclone lsd [路径]`：仅列出目录。  
  ```bash
  rclone lsd /home/user  # 列出本地目录
  ```


#### 6. **检查文件差异：`rclone check`**
对比源和目标的文件是否一致（大小、哈希值）。  
```bash
rclone check /local mygdrive:backup  # 检查本地与远程的差异
```
- 选项 `--download`：对远程文件强制重新下载校验（确保准确性，较慢）。


#### 7. **删除文件：`rclone delete` / `rclone purge`**
- `rclone delete [路径]`：删除路径下的所有文件（保留目录结构）。  
  ```bash
  rclone delete mygdrive:old_files  # 删除远程目录下的文件
  ```
- `rclone purge [路径]`：删除路径下的所有文件和目录（彻底删除）。  


#### 8. **创建目录：`rclone mkdir`**
在远程或本地创建目录。  
```bash
rclone mkdir mygdrive:new_folder  # 在远程创建目录
```


#### 9. **显示存储信息：`rclone about`**
查看远程存储的总空间、已用空间等。  
```bash
rclone about mygdrive:  # 查看 Google Drive 空间信息
```


### **三、常用全局选项**
- `-P` / `--progress`：显示实时传输进度（速度、剩余时间）。
- `-n` / `--dry-run`：模拟操作（不实际执行，用于测试同步/删除等危险操作）。
- `--bwlimit 1M`：限制传输速度（如 1M 表示 1MB/s）。
- `--transfers 8`：并发传输文件数（默认 4，可根据网络调整）。
- `--exclude "*.log"`：排除特定文件（支持通配符）。
- `--include "*.txt"`：仅包含特定文件。









### **四、示例场景**
1. **每天自动备份本地照片到 OneDrive**：  
   ```bash
   rclone sync -P /home/photos myonedrive:backup/photos
   ```
2. **从 S3 下载大文件并限制速度**：  
   ```bash
   rclone copy -P --bwlimit 5M mys3:large_files ./downloads
   ```
3. **模拟删除远程冗余文件（确认后再执行）**：  
   ```bash
   rclone sync -n /local mygdrive:data  # 先模拟，无问题后去掉 -n
   ```


### **五、帮助与文档**
- 查看所有命令：`rclone help`
- 查看特定命令详情：`rclone copy --help`
- 官方文档：[rclone.org](https://rclone.org/)（含支持的存储类型及配置细节）

通过上述命令，可灵活实现本地与云存储、云存储之间的数据管理，适合批量操作和脚本自动化。