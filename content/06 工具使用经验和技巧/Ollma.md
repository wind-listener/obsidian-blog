Ollama 是一款轻量级的本地大模型运行工具，支持快速部署、管理和交互开源大模型，以下是其**核心常用指令**总结，按使用场景分类，方便查阅和使用：

### 一、基础启动/状态指令
#### 1. 启动 Ollama 服务
（不同系统启动方式略有差异，核心指令）
```bash
# Linux/macOS 前台启动（调试用）
ollama serve

# Linux/macOS 后台启动（默认推荐，一般安装后自动后台运行）
# 若需手动重启服务（以 systemd 为例，Linux）
sudo systemctl restart ollama

# Windows（命令行/终端）
ollama serve
```

#### 2. 查看 Ollama 版本
```bash
ollama -v
# 或
ollama version
```

#### 3. 查看 Ollama 帮助（所有指令说明）
```bash
ollama help
# 查看特定指令的帮助（如 run 指令）
ollama help run
```

### 二、模型管理指令（核心）
#### 1. 拉取（下载）模型
Ollama 模型仓库：https://ollama.com/library，支持指定版本（默认最新）。
```bash
# 拉取官方模型（如 Llama 3 8B）
ollama pull llama3
# 拉取指定版本（如 llama3:70b）
ollama pull llama3:70b
# 拉取自定义/第三方模型（需提供模型文件或远程地址）
ollama pull localhost:11434/my-model:v1
```

#### 2. 列出本地已下载的模型
```bash
ollama list
# 输出示例：
# NAME            ID              SIZE    MODIFIED
# llama3:latest   78e26419b446    4.7GB   2 hours ago
```

#### 3. 运行（启动）模型交互
```bash
# 基础交互（启动模型并进入对话终端）
ollama run llama3

# 运行模型时指定参数（如调整温度、上下文长度）
ollama run llama3 --temperature 0.1 --ctx-size 4096

# 单次提问（非交互式，适合脚本调用）
ollama run llama3 "请解释什么是大语言模型"
```

#### 4. 删除本地模型
```bash
# 删除指定模型（如 llama3:70b）
ollama rm llama3:70b
# 强制删除（若有依赖）
ollama rm -f llama3:70b
```

#### 5. 复制/重命名模型
```bash
# 将模型 llama3:latest 复制为 my-llama3:v1
ollama cp llama3:latest my-llama3:v1
```

#### 6. 推送模型到远程仓库
```bash
# 推送自定义模型到 Ollama 仓库（需先登录）
ollama push my-model:v1
```

#### 7. 从远程仓库拉取私有模型
```bash
ollama pull <username>/my-model:v1
```

### 三、模型自定义指令
#### 1. 创建自定义模型（基于 Modelfile）
Modelfile 是 Ollama 定义模型配置的文件，支持指定基础模型、系统提示、参数等。
```bash
# 基于 Modelfile 构建自定义模型（当前目录下的 Modelfile）
ollama create my-custom-model -f Modelfile

# 示例 Modelfile 内容（基础模板）：
FROM llama3:latest          # 基础模型
SYSTEM "你是一个专业的编程助手，回答简洁准确"  # 系统提示
PARAMETER temperature 0.2   # 温度参数
PARAMETER ctx-size 8192     # 上下文长度
```

#### 2. 编辑已有模型的配置
```bash
# 编辑模型的 Modelfile（编辑后需重新构建）
ollama edit my-custom-model
```

### 四、进阶指令
#### 1. 查看模型详细信息
```bash
ollama show llama3
# 仅查看模型的系统提示
ollama show llama3 --system
# 仅查看模型的参数配置
ollama show llama3 --parameters
```

#### 2. 停止正在运行的模型
```bash
# 停止指定模型进程
ollama stop llama3
# 停止所有运行中的模型
ollama stop all
```

#### 3. 清理 Ollama 缓存/无用数据
```bash
# 清理未使用的模型层（释放磁盘空间）
ollama prune
# 强制清理（无需确认）
ollama prune -f
```

### 五、常用参数说明（run/create 时）
| 参数 | 说明 | 示例 |
|------|------|------|
| `--temperature` | 随机性（0-1，越低越严谨） | `--temperature 0.1` |
| `--ctx-size` | 上下文窗口大小（最大可处理的文本长度） | `--ctx-size 8192` |
| `--num-predict` | 最大生成token数 | `--num-predict 1024` |
| `--top-k` | 采样候选词数量（越小越聚焦） | `--top-k 40` |
| `--top-p` | 核采样阈值（0-1，越小越聚焦） | `--top-p 0.9` |

### 六、核心使用流程示例
1. 拉取模型 → 2. 运行交互 → 3. 自定义模型 → 4. 清理缓存
```bash
# 1. 拉取 Llama 3 8B
ollama pull llama3
# 2. 交互式对话
ollama run llama3
# 3. 编写 Modelfile 后构建自定义模型
ollama create my-llama -f Modelfile
# 4. 清理无用数据
ollama prune
```

# 启动服务

参考文档： https://github.com/logancyang/obsidian-copilot/blob/master/local_copilot.md
```bash
tmux
OLLAMA_ORIGINS="app://obsidian.md*" ollama serve
```



