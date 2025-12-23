# Source 指令与 Linux $PATH 环境变量 完整总结
## 一、Source 指令（`.` 指令）
### 1. 核心定义
`source` 是 Linux/macOS 系统中 **Shell 内置命令**（Windows 终端不支持，需 WSL/Git Bash 等类 Unix 环境），作用是 **在当前 Shell 进程中执行指定脚本文件的内容**，而非启动新的子进程。

其简写形式为 `.`（英文句点），两条命令完全等价：
```bash
source /path/to/script.sh
. /path/to/script.sh  # 简写，效果一致
```

### 2. 核心特性：环境变量“全局生效”
Shell 脚本（如激活脚本、配置文件）常包含环境变量设置、路径配置等操作，`source` 与直接运行脚本（`./script.sh`）的核心区别在于 **执行进程不同**：

| 执行方式       | 执行进程               | 环境配置生效范围       | 典型场景                     |
|----------------|------------------------|------------------------|------------------------------|
| `source 脚本`  | 当前 Shell 进程        | 全局生效（终端会话内） | 加载 Conda 激活脚本、修改 `.bashrc` 后生效 |
| `./script.sh`  | 新建子 Shell 进程      | 仅子进程内生效（退出失效） | 运行独立任务脚本（不影响当前环境） |

### 3. 常用场景（含此前 Conda 相关用法）
#### （1）激活 Conda 环境（核心场景）
Conda 安装后，需通过 `source` 执行激活脚本，让当前终端识别 `conda` 命令：
```bash
# 执行自定义路径的 Conda 激活脚本
source /workspace/ckpt_downstream/zzm/miniconda3/bin/activate
```
执行后，终端才能识别 `conda activate 环境名` 等命令（配置临时生效，关闭终端失效）。

#### （2）让配置文件修改即时生效
修改 `.bashrc`（Bash 配置）、`.zshrc`（Zsh 配置）后，无需重启终端，用 `source` 即时加载：
```bash
source ~/.bashrc  # 让 Bash 配置修改生效
source ~/.zshrc   # 让 Zsh 配置修改生效
```

#### （3）加载自定义环境变量/别名
若脚本中定义了环境变量或命令别名，用 `source` 执行可在当前终端直接使用：
```bash
# 脚本 env_config.sh 内容：
export TEST_PATH="/home/user/test"
alias ll="ls -l"

# 加载脚本，变量和别名即时生效
source env_config.sh
echo $TEST_PATH  # 输出 /home/user/test
ll               # 等价于 ls -l
```

### 4. 注意事项
- 仅能执行文本格式的 Shell 脚本（`.sh`、配置文件等），无法执行二进制文件；
- 脚本路径需正确（相对路径/绝对路径），否则报错 `No such file or directory`；
- 执行时需脚本有“读取权限”（无需执行权限，因为是当前 Shell 解析执行）。

