
#linux 

> `tmux` 是一个终端复用器，它可以让你在一个终端窗口内运行多个终端会话，并在断开连接后保持会话运行。以下是对 `tmux` 的详细介绍，包括安装、基本用法和常用命令。

# 个人常用
```bash
tmux
tmux attach -t xx
tmux ls


```

### 安装 tmux

```bash
# 对于基于 Debian 的系统（如 Ubuntu）：
sudo apt update
sudo apt install tmux

# 对于基于 RHEL 的系统（如 Fedora）：
sudo dnf install tmux
```

### 启动 tmux

启动一个新的 `tmux` 会话：

```bash
tmux
```

你也可以为会话指定一个名称：

```bash
tmux new -s my_session
```

### 基本用法

#### 分离和重新连接会话

- 分离当前会话（使会话在后台运行）：

  按 `Ctrl-b` 后按 `d`

- 查看当前所有的 tmux 会话：

  ```bash
  tmux ls
  ```

- 重新连接到一个会话：

  ```bash
  tmux attach -t my_session
  ```

#### 窗口和窗格

- 创建一个新窗口：

  按 `Ctrl-b` 后按 `c`

- 切换到下一个窗口：

  按 `Ctrl-b` 后按 `n`

- 切换到上一个窗口：

  按 `Ctrl-b` 后按 `p`

- 创建一个新窗格（垂直分割）：

  按 `Ctrl-b` 后按 `%`

- 创建一个新窗格（水平分割）：

  按 `Ctrl-b` 后按 `"`

- 在窗格之间切换：

  按 `Ctrl-b` 后按箭头键（上下左右）

### 常用命令

- 重命名窗口：

  按 `Ctrl-b` 后按 `,`，然后输入新的窗口名称

- 关闭当前窗格：

  输入 `exit` 或按 `Ctrl-d`

- 关闭当前窗口：

  输入 `exit` 或按 `Ctrl-d`，当窗口中的所有窗格都关闭时，窗口也会关闭

- 列出所有窗口：

  按 `Ctrl-b` 后按 `w`

- 重新加载 tmux 配置文件：

  ```bash
  tmux source-file ~/.tmux.conf
  ```

### 配置 tmux

你可以通过编辑 `~/.tmux.conf` 文件来配置 `tmux`。例如，设置更方便的按键绑定：

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

### 在 Tmux 中上下滚动

当您第一次开始使用 Tmux 时，您可能会发现很难进行基本操作，例如滚动终端、切换窗格和窗口、分割窗口以及调整窗格大小。让我们看看如何使用 Tmux 中的滚动功能。

**“Ctrl+b”** 是控制 Tmux 操作的最重要的键绑定。如果您想滚动 Tmux 终端，请按**“Ctrl+b ” **组合键并输入**“[ ” **  进入复制模式。现在，您可以使用箭头（向上和向下）等导航键逐行移动。左右箭头可用于逐个字符移动。使用“向上翻页”和“向下翻页”按钮进行页面滚动。

还可以使用键绑定“Ctrl+b”和“Page Up”。这样，您将进入复制模式。要转到特定行号，请使用“g”并从底部开始输入行号。￥

### 高级用法

- 保存和恢复会话

你可以使用 `tmux-resurrect` 插件来保存和恢复 `tmux` 会话。首先安装 `tmux-plugin-manager`（TPM）：

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

通过这些步骤和命令，你可以灵活地使用 `tmux` 来管理多个终端会话，提高工作效率。

# 在 `tmux` 中删除一个会话可以通过以下几种方法实现：

### 方法 1：在会话内退出
如果你已经在 `tmux` 会话中，可以通过终止所有运行的进程来关闭会话。例如，按下 `Ctrl + D` 或直接输入 `exit` 命令来关闭当前会话。当会话内的所有终端窗口关闭后，该会话就会被删除。

### 方法 2：从外部关闭会话
1. 先列出所有会话：
   ```bash
   tmux ls
   ```
   这会列出所有活动的 `tmux` 会话及其名称。

2. 使用 `tmux kill-session` 删除指定的会话：
   ```bash
   tmux kill-session -t <session_name>
   ```
   其中 `<session_name>` 是你要删除的会话的名称。

例如，如果会话名称是 `my_session`，则执行：
```bash
tmux kill-session -t my_session
```

### 方法 3：删除所有会话
如果你想删除所有的 `tmux` 会话，可以使用以下命令：
```bash
tmux kill-server
```
这会关闭 `tmux` 服务器并删除所有会话。

根据需求选择合适的方式来删除会话。