#linux 

> Tmux 是一款强大的终端复用器，它不仅能让你在一个终端窗口中运行多个会话，还能在断开连接后保持会话持续运行。无论是日常开发还是远程工作，Tmux 都能显著提升你的工作效率。本文将详细介绍 Tmux 的安装、基本用法、配置技巧以及高级功能。

## 1. 安装 Tmux

在不同的 Linux 发行版中，Tmux 的安装命令略有差异：

```bash
# 对于基于 Debian 的系统（如 Ubuntu）：
sudo apt update
sudo apt install tmux

# 对于基于 RHEL 的系统（如 Fedora）：
sudo dnf install tmux
```

## 2. 启动 Tmux

启动一个新的 Tmux 会话非常简单：

```bash
tmux
```

如果你想为会话指定一个名称，可以使用以下命令：

```bash
tmux new -s my_session
```

## 3. 基本用法

### 3.1 分离和重新连接会话

- **分离当前会话**（使会话在后台运行）：  
  按下 `Ctrl-b` 后按 `d`。

- **查看所有 Tmux 会话**：  
  ```bash
  tmux ls
  ```

- **重新连接到一个会话**：  
  ```bash
  tmux attach -t my_session
  ```

### 3.2 窗口和窗格管理

- **创建新窗口**：  
  按下 `Ctrl-b` 后按 `c`。

- **切换到下一个窗口**：  
  按下 `Ctrl-b` 后按 `n`。

- **切换到上一个窗口**：  
  按下 `Ctrl-b` 后按 `p`。

- **垂直分割窗格**：  
  按下 `Ctrl-b` 后按 `%`。

- **水平分割窗格**：  
  按下 `Ctrl-b` 后按 `"`。

- **在窗格之间切换**：  
  按下 `Ctrl-b` 后按箭头键（上下左右）。

## 4. 常用命令

- **重命名窗口**：  
  按下 `Ctrl-b` 后按 `,`，然后输入新的窗口名称。

- **关闭当前窗格**：  
  输入 `exit` 或按 `Ctrl-d`。

- **关闭当前窗口**：  
  输入 `exit` 或按 `Ctrl-d`，当窗口中的所有窗格都关闭时，窗口也会关闭。

- **列出所有窗口**：  
  按下 `Ctrl-b` 后按 `w`。

- **重新加载 Tmux 配置文件**：  
  ```bash
  tmux source-file ~/.tmux.conf
  ```

## 5. 配置 Tmux

通过编辑 `~/.tmux.conf` 文件，你可以自定义 Tmux 的行为。以下是一些常用的配置示例：

```bash
# 设置前缀键为 Ctrl-a
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# 启用鼠标支持
set -g mouse on

# 重新加载配置文件
bind r source-file ~/.tmux.conf \; display "Reloaded!"
```

## 6. 在 Tmux 中上下滚动

初次使用 Tmux 时，可能会对滚动终端感到困惑。以下是滚动操作的详细说明：

1. 按下 `Ctrl-b` 后按 `[` 进入复制模式。
2. 使用箭头键（上下）逐行滚动，或使用 `Page Up` 和 `Page Down` 进行页面滚动。
3. 要跳转到特定行号，按下 `g` 并输入行号。

## 7. 高级用法

### 7.1 保存和恢复会话

你可以使用 `tmux-resurrect` 插件来保存和恢复 Tmux 会话。首先安装 `tmux-plugin-manager`（TPM）：

```bash
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
```

然后在 `~/.tmux.conf` 中添加以下内容：

```bash
# TPM 配置
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-resurrect'

# 初始化 TPM
run '~/.tmux/plugins/tpm/tpm'
```

重新加载配置文件并使用 `prefix + I` 安装插件。你可以使用 `prefix + Ctrl-s` 保存会话，使用 `prefix + Ctrl-r` 恢复会话。

## 8. 删除 Tmux 会话

### 方法 1：在会话内退出

如果你已经在 Tmux 会话中，可以通过终止所有运行的进程来关闭会话。例如，按下 `Ctrl + D` 或直接输入 `exit` 命令来关闭当前会话。当会话内的所有终端窗口关闭后，该会话就会被删除。

### 方法 2：从外部关闭会话

1. 先列出所有会话：
   ```bash
   tmux ls
   ```

2. 使用 `tmux kill-session` 删除指定的会话：
   ```bash
   tmux kill-session -t <session_name>
   ```

例如，删除名为 `my_session` 的会话：
```bash
tmux kill-session -t my_session
```

### 方法 3：删除所有会话

如果你想删除所有的 Tmux 会话，可以使用以下命令：
```bash
tmux kill-server
```

---

通过以上步骤和命令，你可以灵活地使用 Tmux 来管理多个终端会话，显著提升工作效率。无论是日常开发还是远程工作，Tmux 都是你不可或缺的利器。