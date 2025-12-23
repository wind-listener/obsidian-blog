要在 **VSCode 连接服务器** 并读取 `infer_result/20250407_113748/VideoClassifier_pious-sweep-1_epoch23_acc0.7581--classifiers_test_all_mp4--_infer_result--bad.csv` 文件，提取 **第一个文件路径**（如 `/workspace/ckpt_downstream/zzm/mydataset/classifiers_test/bili_vdata_20250122_horizontal_0353_00002/2230ca17b0caabecc05884be35d51077.mp4`），然后 **下载到本地**，可以使用 **Python + Paramiko（SSH）** 或 **VSCode Remote SSH** 扩展来实现。

---

## **方法 1：Python + Paramiko（SSH 下载）**
适用于 **Python 脚本自动化** 下载文件。

### **1. 安装 `paramiko`**
```bash
pip install paramiko
```

### **2. Python 代码**
```python
import paramiko
import os

# 服务器 SSH 配置
host = "your_server_ip"
port = 22
username = "your_username"
password = "your_password"  # 或使用 SSH Key

# CSV 文件路径（服务器上的路径）
remote_csv_path = "/path/to/infer_result/20250407_113748/VideoClassifier_pious-sweep-1_epoch23_acc0.7581--classifiers_test_all_mp4--_infer_result--bad.csv"

# 本地保存路径
local_save_dir = "./downloaded_videos"
os.makedirs(local_save_dir, exist_ok=True)

# 1. 连接服务器
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

# 2. 读取 CSV 文件
stdin, stdout, stderr = ssh.exec_command(f"cat {remote_csv_path}")
csv_lines = stdout.read().decode("utf-8").splitlines()

# 3. 提取第一个文件路径
first_line = csv_lines[0]  # 假设第一行是数据行
file_path = first_line.split("\t")[0]  # 提取第一个字段（文件路径）
print("Extracted file path:", file_path)

# 4. 下载文件到本地
remote_file_path = file_path
local_file_path = os.path.join(local_save_dir, os.path.basename(file_path))

# 使用 SFTP 下载
sftp = ssh.open_sftp()
sftp.get(remote_file_path, local_file_path)
sftp.close()

print(f"Downloaded: {local_file_path}")

# 5. 关闭 SSH
ssh.close()
```

### **3. 运行**
```bash
python download_video.py
```
✅ **文件会被下载到 `./downloaded_videos/` 目录。**

---

## **方法 2：VSCode Remote SSH（手动下载）**
适用于 **手动操作**，适合少量文件。

### **1. 安装 VSCode Remote SSH 扩展**
- 在 VSCode 安装 **Remote - SSH** 扩展。
- 配置 SSH 连接（`~/.ssh/config` 或 VSCode 的 Remote Explorer）。

### **2. 连接服务器**
- 在 VSCode 按 `F1` → `Remote-SSH: Connect to Host` → 选择你的服务器。

### **3. 打开 CSV 文件**
- 在服务器上找到：
  ```
  /path/to/infer_result/20250407_113748/VideoClassifier_pious-sweep-1_epoch23_acc0.7581--classifiers_test_all_mp4--_infer_result--bad.csv
  ```
- 右键 → **Open** 查看内容。

### **4. 提取文件路径**
- 复制第一行的第一个字段（如 `/workspace/.../2230ca17b0caabecc05884be35d51077.mp4`）。

### **5. 下载文件**
- 在 VSCode 左侧 **Remote Explorer** → 找到该文件 → **右键 Download**。
- 文件会自动下载到本地。

---

## **总结**
| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **Python + Paramiko** | 批量下载、自动化 | 可编程，适合大量文件 | 需要 Python 环境 |
| **VSCode Remote SSH** | 手动下载少量文件 | 可视化操作，简单 | 不适合批量 |

**推荐**：
- 如果 **只需要下载 1 个文件** → **VSCode Remote SSH**（手动操作）。
- 如果 **需要批量下载** → **Python + Paramiko**（自动化）。

希望这能帮到你！🚀