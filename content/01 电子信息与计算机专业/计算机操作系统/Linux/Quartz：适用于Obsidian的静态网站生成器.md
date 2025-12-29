---
title: "Quartz：适用于Obsidian的静态网站生成器"
date: 2025-11-18
draft: false
---

你提到的“Quartz”需要先明确：在与 Obsidian 结合生成网页的场景中，这里的 **Quartz** 指的是由开发者 jackyzha0 开发的**静态网站生成器（Static Site Generator）**，专为 Obsidian 笔记设计，而非 Java 领域的任务调度框架 Quartz Scheduler。以下详细介绍这个静态网站生成器的核心特性、用法，以及如何用它将 Obsidian 笔记转化为网页。

官方wiki： https://quartz.jzhao.xyz/
github： https://github.com/jackyzha0/quartz.git

### 一、Quartz 静态网站生成器核心特性
Quartz 的核心定位是“把 Obsidian 笔记变成可访问的网页”，它的设计完全贴合 Obsidian 的使用习惯，主要特点包括：  
- **原生支持 Obsidian 语法**：自动解析双链（`[[笔记名]]`）、标签（`#标签`）、文件夹结构，无需修改笔记内容。  
- **轻量高效**：基于 Node.js 构建，生成速度快，部署后网页加载迅速。  
- **内置知识图谱**：自动生成笔记间的关联图谱（类似 Obsidian 的“图谱视图”），可视化知识连接。  
- **响应式设计**：生成的网页适配手机、电脑等多设备。  
- **可定制性**：支持修改主题、布局、导航栏等，满足个性化需求。  


### 二、使用 Quartz 创建网页的完整步骤
#### 前置准备
- 安装 **Node.js**（v16+，推荐 v18，用于运行 Quartz 构建工具）：[Node.js 官网](https://nodejs.org/)  
- 安装 **Git**（用于版本控制和部署）：[Git 官网](https://git-scm.com/)  
- 一个 Obsidian 笔记库（存放你的 Markdown 笔记和多媒体文件）。  


#### 步骤 1：初始化 Quartz 项目
1. **克隆 Quartz 仓库**  
   打开终端（命令行），执行以下命令下载 Quartz 源码：  
   ```bash
   git clone https://github.com/jackyzha0/quartz.git
   cd quartz  # 进入项目目录
   ```

2. **安装依赖**  
   执行命令安装项目所需的 Node 依赖：  
   ```bash
   npm install
   ```


#### 步骤 2：关联 Obsidian 笔记库
Quartz 需要知道你的 Obsidian 笔记存放在哪里，以便读取内容生成网页。  

1. **配置笔记路径**  
   打开 Quartz 项目根目录下的 `quartz.config.ts` 文件（核心配置文件），找到 `contentPath` 字段，修改为你的 Obsidian 笔记库路径：  
   ```typescript
   // quartz.config.ts
   export default {
     contentPath: "./my-obsidian-notes",  // 替换为你的 Obsidian 笔记文件夹路径（相对路径或绝对路径）
     // 其他配置保持默认即可
   }
   ```  
   例如：如果你的 Obsidian 笔记库在 `D:/Notes/Personal`，则改为 `contentPath: "D:/Notes/Personal"`。

2. **验证笔记结构**  
   确保你的 Obsidian 笔记库符合以下规范（避免网页生成错误）：  
   - 笔记以 `.md` 格式保存；  
   - 多媒体文件（图片、视频等）放在固定文件夹（如 `attachments`，后续会讲如何处理）；  
   - 避免使用特殊字符（如 `#`、`?`）作为笔记文件名（可能导致网页链接错误）。  


#### 步骤 3：本地预览网页
配置完成后，可在本地生成并预览网页，确认效果：  

1. **启动本地服务器**  
   执行以下命令，Quartz 会读取 Obsidian 笔记并生成临时网页，同时启动本地预览服务：  
   ```bash
   npm run dev
   ```  

2. **访问预览页面**  
   终端会显示预览地址（通常是 `http://localhost:8080`），打开浏览器访问该地址，即可看到你的 Obsidian 笔记被转化为网页的效果：  
   - 左侧是笔记文件夹结构（类似 Obsidian 的文件浏览器）；  
   - 中间是笔记内容（支持 Markdown 渲染、双链跳转）；  
   - 右侧是标签列表和知识图谱（可点击切换）。  


#### 步骤 4：自定义网页（可选）
如果想修改网页样式、导航栏、主题等，可通过以下方式定制：  

1. **修改主题**  
   Quartz 内置了多个主题（如浅色、深色、学术风格），在 `quartz.config.ts` 中修改 `theme` 字段：  
   ```typescript
   export default {
     theme: "dark",  // 可选："light"（默认）、"dark"、"academic" 等
   }
   ```  

2. **修改导航栏**  
   打开 `quartz/layouts/partials/navbar.njk` 文件（Nunjucks 模板），可添加自定义链接（如“关于我”“分类”）：  
   ```html
   <!-- 在导航栏添加一个"关于我"链接 -->
   <a href="/about" class="nav-link">关于我</a>
   ```  
   注意：`/about` 对应 Obsidian 中 `about.md` 笔记的网页地址（需提前创建该笔记）。

3. **调整样式**  
   若需深度定制 CSS，可修改 `quartz/assets/styles/custom.scss` 文件（SCSS 语法），例如修改字体大小：  
   ```scss
   // 自定义笔记内容字体大小
   .page-content {
     font-size: 18px;
   }
   ```  


#### 步骤 5：生成最终网页并部署
本地预览满意后，生成可部署的静态文件，并发布到网络上（让他人访问）。  

1. **生成静态文件**  
   执行以下命令，Quartz 会在项目根目录生成 `public` 文件夹，里面是最终的网页文件（HTML、CSS、JS 等）：  
   ```bash
   npm run build
   ```  

2. **部署到免费平台**  
   推荐将 `public` 文件夹部署到以下免费平台，支持自动更新：  

   - **GitHub Pages**：  
     1. 在 GitHub 新建仓库（如 `my-quartz-blog`）；  
     2. 将 Quartz 项目推送到该仓库（包含 `public` 文件夹）；  
     3. 进入仓库设置 → Pages → 选择“部署来源”为 `public` 文件夹，保存后即可通过 `https://用户名.github.io/仓库名` 访问。  

   - **Vercel**（更简单，推荐）：  
     1. 注册 Vercel 账号并关联 GitHub 仓库；  
     2. 导入 `my-quartz-blog` 仓库，Vercel 会自动识别 Quartz 项目，默认使用 `npm run build` 构建，部署后生成临时域名（可绑定自定义域名）。  


#### 步骤 6：自动化更新（关键！）
为了实现“Obsidian 写完笔记，网页自动更新”，需要配置自动化流程：  

1. **用 Git 管理 Obsidian 笔记**  
   在你的 Obsidian 笔记库中初始化 Git 仓库，并关联到 GitHub（与 Quartz 项目可共用一个仓库，或分开）：  
   ```bash
   cd my-obsidian-notes  # 进入 Obsidian 笔记库
   git init
   git add .
   git commit -m "初始笔记"
   git remote add origin https://github.com/你的用户名/你的仓库名.git
   git push -u origin main
   ```  

2. **配置 GitHub Actions 自动构建**  
   在 Quartz 项目的 `.github/workflows` 文件夹中，创建 `deploy.yml` 文件，内容如下（自动检测笔记更新并部署）：  
   ```yaml
   name: Deploy Quartz
   on:
     push:
       branches: [main]  # 当 main 分支有更新时触发
   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Install Node.js
           uses: actions/setup-node@v3
           with:
             node-version: 18
         - name: Install dependencies
           run: npm install
         - name: Build Quartz
           run: npm run build
         - name: Deploy to GitHub Pages
           uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./public
   ```  
   保存后推送到 GitHub，后续当你在 Obsidian 中更新笔记并推送到 Git 时，GitHub Actions 会自动重新构建并部署网页。

3. **Obsidian 自动提交（可选）**  
   安装 Obsidian 插件 **Git**（在 Obsidian 社区插件中搜索“Git”），配置“自动提交”和“自动推送”（如每 5 分钟），无需手动执行 Git 命令，彻底自动化。  


### 三、处理 Obsidian 中的多媒体内容
在 Obsidian 中插入的图片、视频等，需要确保在网页中正常显示，步骤如下：  

1. **统一存放多媒体文件**  
   在 Obsidian 笔记库中新建 `attachments` 文件夹（或 `assets`），将所有图片、视频放入其中。  
   在 Obsidian 设置中配置：`设置 → 文件与链接 → 附件默认保存位置` 选择 `./attachments`，确保新插入的多媒体自动存到该文件夹。  

2. **正确插入多媒体**  
   在 Markdown 中用**相对路径**插入（Obsidian 和 Quartz 都支持）：  
   ```markdown
   <!-- 插入图片 -->
   ![我的截图](./attachments/screenshot.png)

   <!-- 插入视频 -->
   <video src="./attachments/demo.mp4" controls width="800"></video>
   ```  
   避免用 Obsidian 的内部链接 `![[screenshot.png]]`（Quartz 虽能解析，但相对路径更稳妥）。  

3. **验证路径**  
   本地预览时（`npm run dev`），检查多媒体是否正常显示。若不显示，通常是路径错误，确认 `attachments` 文件夹是否在 Obsidian 库根目录，且路径拼写正确。  


### 四、总结
Quartz 是 Obsidian 用户将笔记转为网页的“零门槛”工具，核心流程可概括为：  
`Obsidian 写笔记 → Git 同步 → Quartz 构建 → 自动部署到网页`。  

它的优势在于**原生支持 Obsidian 特性**（双链、图谱、标签），无需修改笔记内容，且部署和自动化流程简单，适合非技术背景用户快速搭建个人知识博客。如果需要更复杂的自定义（如评论区、统计分析），可通过添加 Quartz 插件或修改源码实现。