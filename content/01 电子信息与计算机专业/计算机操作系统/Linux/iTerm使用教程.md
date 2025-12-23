# iTerm2：Mac上的终极终端神器

## 一、基础入门

### 1. 安装与启动

**安装方式**：
- **官网下载**：访问[iterm2.com](https://iterm2.com)下载.dmg文件，拖动到Applications文件夹 
- **Homebrew安装**：执行`brew install --cask iterm2` 

**首次启动**：
- 授权辅助功能（系统设置→安全性与隐私→辅助功能）以启用鼠标报告等高级功能 
- 建议勾选"Restore window arrangement across restarts"以保存会话状态 

### 2. 界面与基本操作

**核心组件**：
- **标签页(Tabs)**：Cmd+T新建，Cmd+数字切换，Cmd+W关闭 
- **分屏(Panes)**：iTerm2标志性功能，支持垂直/水平分割 

**基础快捷键**：

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 垂直分屏 | Cmd+D | 左右布局  |
| 水平分屏 | Cmd+Shift+D | 上下布局  |
| 切换分屏 | Cmd+Option+箭头键<br>Cmd+[ / Cmd+] | 在相邻分屏间移动焦点  |
| 最大化分屏 | Cmd+Shift+Enter | 隐藏其他分屏，再次按下恢复  |
| 清屏 | Cmd+R / Ctrl+L | 清除当前屏幕内容  |
| 查看历史命令 | Cmd+; | 显示历史命令自动补全  |
| 粘贴历史 | Cmd+Shift+H | 显示复制/粘贴历史记录，支持模糊搜索  |
| 全屏模式 | Cmd+Enter | 切换全屏，背景透明会自动关闭  |

**文本操作**：
- 双击：选中单词
- 三击：选中整行
- 四击：智能选择（识别URL、引号字符串、邮箱等） 
- 选中文本即自动复制到剪贴板，无需额外操作 

## 二、进阶功能详解

### 1. 分屏魔法：提高工作效率的核心

**分屏技巧**：
- 垂直分屏(Cmd+D)后，新面板会自动继承当前工作目录 
- 使用鼠标拖动分屏边界调整大小
- 拖放标签页到其他窗口创建新分屏布局 
- **广播输入**：Shell→Broadcast Input→All Sessions in Tab，在所有分屏执行相同命令 

**实战示例**：
```
# 左侧编辑代码，右侧运行/调试
1. 打开项目文件夹，执行vim main.py
2. Cmd+D垂直分屏
3. 在右侧面板执行python main.py或make test
```

### 2. 热键窗口：一键呼出的秘密武器

- **设置路径**：Preferences→Keys→Hotkey，勾选"Show/hide iTerm2 with a system-wide hotkey" 
- **推荐设置**：选择F12或Ctrl+Space作为全局热键
- **应用场景**：临时执行系统命令，无需切换应用 

### 3. 即时回放：时间旅行般的终端体验

- 按下Cmd+Option+B进入"即时回放"模式 
- 使用左右箭头键浏览历史输出，精确到秒
- Esc退出回放模式
- **适用场景**：追踪程序执行过程、分析错误发生时间点 

### 4. 智能搜索与触发器：让终端更聪明

**搜索增强**：
- Cmd+F打开搜索框，支持正则表达式(点击放大镜旁下拉箭头启用) 
- Cmd+G/Shift+G在匹配项间正向/反向跳转 

**触发器(Triggers)**：
- **功能**：当终端输出匹配特定正则表达式时，自动执行预设操作 
- **设置路径**：Preferences→Profiles→Advanced→Triggers→Add 
- **应用示例**：
  ```
  # 高亮显示ERROR/WARNING级别日志
  正则: ^(ERROR|WARNING):
  动作: Highlight text in red/yellow
  ```

## 三、深度配置：打造个性化工作环境

### 1. 配置文件(Profiles)：多环境切换的钥匙

**创建配置**：
- Preferences→Profiles→点击"+"添加新配置 
- **常用配置项**：
  - **General**：设置启动命令(如SSH连接) 
  - **Colors**：选择/自定义配色方案 
  - **Text**：设置字体(推荐等宽字体如Menlo、Fira Code) 
  - **Window**：调整透明度(0.6-0.8)和背景模糊效果 

**自动切换**：
- 配置"Automatic Profile Switching"，根据用户名/主机名自动切换配置 
- 示例：连接生产环境时自动使用红色警告主题 

### 2. 性能优化：让iTerm2飞起来

**关键设置**：

| 设置项 | 推荐值 | 说明 |
|--------|--------|------|
| 滚动缓冲区限制 | 5,000-10,000行 | 减少内存占用  |
| 字体选择 | Menlo/Monaco等简单等宽字体 | 避免复杂字体渲染  |
| 动画效果 | 关闭 | Preferences→Appearance→Animations  |
| 刷新率 | Maximize throughput | 降低至30FPS，提高数据处理速度  |
| 内存优化 | Prefer integrated GPU | 减少功耗，适合笔记本  |

**高级优化**（通过defaults命令）：
```bash
# 减少窗口更新延迟
defaults write com.googlecode.iterm2 UpdateScreenParamsDelay -float 0.1

# 调整粘贴历史记录数量
defaults write com.googlecode.iterm2 MaxPasteHistoryEntries -int 50
```

## 四、插件生态：扩展无限可能

### 1. 与Zsh完美集成：生产力倍增

**安装Oh My Zsh**：
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**必装插件**：

| 插件名称 | 功能 | 安装方式 |
|----------|------|----------|
| zsh-syntax-highlighting | 命令语法高亮 | `git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting`  |
| zsh-autosuggestions | 命令自动补全 | `git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions` |
| iterm2-zmodem | 文件上传下载 | 官方插件，iTerm2菜单中安装  |

**主题配置**：
- 安装Powerline字体（如MesloLGS NF）支持特殊字符显示 
- 选择agnoster等流行主题 

### 2. 精选第三方插件

**实用插件**：
- **iTerm2-Color-Schemes**：数百种精美配色方案，一键导入 
- **Shell Integration**：iTerm2内置插件，增强命令历史、自动补全、文件传输等 
  ```bash
  # 安装命令
  iTerm2菜单 → Shell → Install Shell Integration
  ```
- **imgcat**：在终端显示图片，支持PNG/JPG/GIF 
- **AI插件**：iTerm2 v3.6.1+新增，可直接在终端内查询资料、生成代码 

## 五、高级技巧：效率提升10倍的秘密

### 1. 分屏与窗口管理秘籍

**窗口排列**：
- 保存当前布局：Window→Save Window Arrangement 
- 自动恢复：设置General→Open saved window arrangement on startup 

**会话管理**：
- **埋入会话**：将当前会话隐藏(Cmd+Option+H)，随时通过Open Quickly(Cmd+Shift+O)找回 
- **会话恢复**：重启iTerm2后自动恢复上次会话，无需手动重连 

### 2. 文本处理与交互技巧

**无鼠标操作**：
- 使用`Cmd+F`进入"无鼠标复制"模式：输入文本开头，按Tab扩展选择 
- 标记与跳转：`Cmd+Shift+M`标记位置，`Cmd+Shift+J`跳转回标记点 

**智能粘贴**：
- 粘贴时自动处理缩进，适合代码片段 

### 3. Shell集成高级应用

**文件传输**：
- 使用Shell Integration替代传统rz/sz命令，支持直接拖放上传、点击下载 
- 示例：在远程服务器输出中点击文件链接直接下载到本地 

**自动补全增强**：
- 补全历史命令(Cmd+;)和剪贴板内容(Cmd+Shift+H) 
- 支持模糊搜索，输入部分内容后按Cmd+;显示匹配项 

## 六、总结与下一步

**iTerm2核心优势**：
- 强大分屏与窗口管理，大幅提升多任务效率
- 高度可定制的外观与行为，打造个人专属终端
- 深度Shell集成，无缝衔接开发工作流
- 丰富插件生态，功能扩展无限可能

**行动清单**：
1. 安装iTerm2并设置全局热键
2. 配置1-2个常用Profile（如开发/生产环境）
3. 安装Oh My Zsh及核心插件
4. 选择一款喜欢的主题（推荐Dracula或Solarized）
5. 学习3-5个高频快捷键，替代鼠标操作

> 下一步探索：深入研究Shell Integration和触发器功能，它们将彻底改变你与终端的交互方式，让命令行工作变得前所未有的高效与愉悦。