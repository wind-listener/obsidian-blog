---
title: "Hugo 学习指南"
date: 2025-12-23
draft: false
---


### 一、前置准备（先搞定环境）
#### 1. 安装 Hugo
Hugo 是基于 Go 开发的单二进制文件，安装极简单，推荐安装 **Hugo Extended 版本**（支持 SCSS 样式编译，适配更多主题）：
- **Windows**：
  ```bash
  # 方式1：Chocolatey 安装（推荐）
  choco install hugo-extended
  # 方式2：手动下载（官网：https://github.com/gohugoio/hugo/releases）
  # 解压后将可执行文件路径加入系统环境变量
  ```
- **Linux（CentOS/Ubuntu）**：
  ```bash
  # Ubuntu/Debian
  sudo apt install hugo-extended
  # CentOS
  sudo dnf install hugo-extended
  ```
- **验证安装**：
  ```bash
  hugo version  # 输出 Hugo 版本号（如 hugo v0.120.0-extended）即成功
  ```

#### 2. 辅助工具
- 代码编辑器：VS Code（安装 Hugo 插件 `Hugo Language and Syntax Support`）；
- Git：用于管理代码和部署；
- 基础知识：了解 Markdown 语法、HTML/CSS 基础（无需精通，够用即可）。

### 二、入门阶段（0-3 天：搭建第一个 Hugo 站点）
核心目标：理解 Hugo 核心概念，生成第一个可预览的静态站点。

#### 1. 核心概念认知（先记关键术语）
- **站点（Site）**：整个 Hugo 项目的总称，包含配置、内容、模板、静态资源；
- **内容（Content）**：存放 Markdown 笔记/文章的目录（`content/`）；
- **主题（Theme）**：决定站点样式和布局的模板集合（`themes/`）；
- **模板（Template）**：控制页面渲染逻辑的文件（基于 Go Template 语法）；
- **静态资源（Static）**：图片、JS/CSS 等无需编译的文件（`static/`）；
- **配置文件**：`hugo.toml`（或 `yaml/json`），全局配置入口。

#### 2. 快速创建第一个站点
```bash
# 1. 创建站点（命名为 my-hugo-blog）
hugo new site my-hugo-blog
cd my-hugo-blog

# 2. 安装主题（以经典的 PaperMod 为例，适配 Obsidian 笔记）
git init
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod

# 3. 配置主题（修改 hugo.toml）
echo 'theme = "PaperMod"' >> hugo.toml

# 4. 创建第一篇文章（Obsidian 笔记可直接复制到 content/posts/）
hugo new posts/first-note.md

# 5. 本地预览（启动开发服务器）
hugo server -D  # -D 表示显示草稿文章
```
访问 `http://localhost:1313`，即可看到你的第一个 Hugo 站点。

#### 3. 核心目录结构解析（新手必看）
```
my-hugo-blog/
├── content/          # 核心：存放所有 Markdown 内容（对应 Obsidian 笔记）
│   └── posts/        # 博客文章目录（可新建其他目录如 notes/ 分类）
├── themes/           # 主题目录
├── static/           # 静态资源：图片、视频等（对应 Obsidian 的 attachments）
├── hugo.toml         # 全局配置文件
├── public/           # 构建后生成的静态文件（部署时上传这个目录）
└── layouts/          # 自定义模板（覆盖主题默认模板）
```

### 三、进阶阶段（3-7 天：适配 Obsidian + 自定义站点）
核心目标：让 Hugo 兼容 Obsidian 笔记语法，定制博客样式和功能。

#### 1. 适配 Obsidian 笔记（关键步骤）
Hugo 默认不支持 Obsidian 特有语法，需通过插件/配置适配：
- **步骤 1：同步 Obsidian 笔记到 Hugo**
  ```bash
  # 复制 Obsidian 笔记到 Hugo 的 content/notes/ 目录（可写脚本自动化）
  cp -r /你的 Obsidian 笔记路径/* content/notes/
  ```
- **步骤 2：转换 Obsidian 双链**
  安装 `hugo-obsidian` 插件（自动将 `[[笔记名]]` 转为 Hugo 链接）：
  ```bash
  # 安装插件
  go install github.com/jackyzha0/hugo-obsidian@latest

  # 在站点根目录创建 config.yaml（插件配置）
  echo 'links:
    inDir: content
    outDir: content
    urlPrefix: /' > config.yaml

  # 运行插件转换双链
  hugo-obsidian convert
  ```
- **步骤 3：处理 Obsidian 多媒体**
  将 Obsidian 的 `attachments` 文件夹复制到 Hugo 的 `static/` 目录，笔记中图片路径改为：
  ```markdown
  # Obsidian 中原路径
  ![图片](./attachments/photo.jpg)
  # Hugo 中修改为（static/ 是根目录）
  ![图片](/attachments/photo.jpg)
  ```
- **步骤 4：适配 Obsidian Callout**
  PaperMod 等主题原生支持 Callout 语法（`> [!note]`），无需额外配置；若主题不支持，可添加自定义 CSS（放在 `assets/css/custom.css`）。

#### 2. 自定义站点配置
修改 `hugo.toml` 配置文件，调整站点基础信息：
```toml
baseURL = "https://你的域名.com/"  # 站点域名
languageCode = "zh-CN"             # 语言
title = "我的 Hugo 博客"            # 站点标题
theme = "PaperMod"                 # 使用的主题

# 启用中文支持
[params]
  defaultTheme = "auto"            # 自动切换浅色/深色模式
  ShowReadingTime = true           # 显示阅读时长
  ShowPostNavLinks = true          # 显示上一篇/下一篇
  ShowToc = true                   # 显示目录
```

#### 3. 自定义主题样式
无需修改主题源码（避免更新主题丢失配置），通过「覆盖模板」实现：
- **修改 CSS**：在 `assets/css/` 新建 `custom.css`，添加自定义样式：
  ```css
  /* 调整笔记内容字体大小 */
  .post-content {
    font-size: 16px;
    line-height: 1.8;
  }
  /* 适配 Obsidian Callout 样式 */
  .callout {
    margin: 1rem 0;
    padding: 1rem;
    border-radius: 8px;
  }
  ```
- **修改导航栏**：在 `layouts/partials/` 新建 `navbar.html`，覆盖主题默认导航栏（参考主题源码修改）。

### 四、实战阶段（7-10 天：部署 + 自动化）
核心目标：将 Hugo 站点部署到服务器/免费平台，实现 Obsidian 笔记自动同步发布。

#### 1. 构建静态文件
```bash
hugo --minify  # 生成优化后的静态文件到 public/ 目录
```

#### 2. 部署方案（二选一）
- **方案 1：免费平台（Vercel/GitHub Pages）**
  - 将 Hugo 项目推送到 GitHub 仓库；
  - Vercel：导入仓库，自动识别 Hugo，一键部署（无需配置）；
  - GitHub Pages：配置 GitHub Actions 自动构建（参考官方文档）。
- **方案 2：自己的服务器（如阿里云）**
  ```bash
  # 用 scp 将 public/ 目录上传到服务器的 /var/www/hugo-blog/
  scp -r public/* root@你的服务器IP:/var/www/hugo-blog/
  # 配置 Nginx 指向该目录（参考 Nginx 静态站点配置）
  ```

#### 3. 自动化同步 Obsidian 笔记
编写简单脚本（`sync-obsidian-to-hugo.sh`），实现笔记自动同步+构建：
```bash
#!/bin/bash
# 1. 复制 Obsidian 笔记到 Hugo 目录
cp -r /你的 Obsidian 路径/* /my-hugo-blog/content/notes/
# 2. 转换 Obsidian 双链
cd /my-hugo-blog
hugo-obsidian convert
# 3. 构建静态文件
hugo --minify
# 4. 上传到服务器（或推送到 GitHub 触发自动部署）
scp -r public/* root@服务器IP:/var/www/hugo-blog/
```
添加到定时任务（`crontab -e`），实现定时同步：
```bash
# 每小时同步一次
0 * * * * /bin/bash /你的脚本路径/sync-obsidian-to-hugo.sh
```

### 四、进阶学习资源（持续提升）
1. **官方文档**：[Hugo 官方文档](https://gohugo.io/documentation/)（最权威，建议优先看）；
2. **主题文档**：所选主题的 README（如 [PaperMod 文档](https://adityatelange.github.io/hugo-PaperMod/)）；
3. **中文教程**：
   - Hugo 中文网：https://www.gohugo.org/
   - 掘金/知乎：搜索「Hugo 搭建博客」「Hugo 适配 Obsidian」；
4. **实战案例**：GitHub 搜索「hugo obsidian blog」，参考他人的配置。

### 五、避坑指南（新手常见问题）
1. **路径错误**：Hugo 中静态资源路径以 `static/` 为根目录，避免用相对路径；
2. **中文乱码**：确保 `hugo.toml` 中 `languageCode = "zh-CN"`，Markdown 文件编码为 UTF-8；
3. **主题不生效**：检查 `theme` 配置是否正确，主题是否已克隆到 `themes/` 目录；
4. **双链转换失败**：确保 `hugo-obsidian` 插件配置的 `inDir`/`outDir` 指向 `content/` 目录。

### 总结
1. Hugo 学习核心路径：「环境安装→基础建站→适配 Obsidian→自定义配置→部署自动化」，新手先聚焦“能用”，再逐步定制；
2. 适配 Obsidian 的关键是：同步笔记+转换双链+处理多媒体路径；
3. 优先用成熟主题（如 PaperMod）减少定制成本，后续再学习 Go Template 深入定制。

