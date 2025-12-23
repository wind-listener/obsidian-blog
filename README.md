# 我的 Obsidian 博客

基于 Hugo 和 Obsidian 构建的个人博客网站，支持双链、知识图谱等 Obsidian 原生特性。

## 特性

- 📝 **Obsidian 原生支持**: 支持 `[[双链]]` 语法和知识图谱
- 🎨 **深色模式**: 内置深色/亮色主题切换
- 🔍 **全文搜索**: 使用 Fuse.js 实现的快速搜索
- 🕸️ **知识图谱**: D3.js 可视化文章关系
- 🏷️ **标签和分类**: 灵活的内容组织方式
- ⚡ **极速构建**: Hugo 静态网站生成器
- 📱 **响应式设计**: 支持各种设备

## 项目结构

```
obsidian-blog/
├── content/          # Obsidian vault（在此编写文章）
│   ├── posts/       # 博客文章
│   └── attachments/ # 图片等附件
├── layouts/         # HTML 模板
├── static/          # 静态资源
├── scripts/         # 构建和部署脚本
├── public/          # 生成的网站（Git 忽略）
└── hugo.toml        # Hugo 配置文件
```

## 快速开始

### 1. 编写文章

在 `content/posts/` 目录下创建 Markdown 文件：

```markdown
---
title: "文章标题"
date: 2025-12-23
tags: ["标签1", "标签2"]
categories: ["分类"]
draft: false
---

文章内容...

使用 [[双链]] 连接其他文章。
```

### 2. 构建网站

```bash
# 开发模式（实时预览）
hugo server --bind 0.0.0.0 --port 1313

# 生产构建
./scripts/build.sh
```

### 3. 快速发布

```bash
# 一键提交、构建和部署
./scripts/publish.sh
```

## Obsidian 集成

### 设置 Obsidian

1. 打开 Obsidian
2. 选择 "打开文件夹作为仓库"
3. 选择 `/home/obsidian-blog/content` 目录
4. 开始编写！

### Obsidian 插件推荐

- **Templater**: 文章模板
- **Calendar**: 日历视图
- **Tag Wrangler**: 标签管理
- **Shell commands**: 一键发布（配置运行 `publish.sh`）

### 快速发布设置

在 Obsidian 中安装 **Shell commands** 插件，添加命令：

```bash
cd /home/obsidian-blog && ./scripts/publish.sh
```

绑定快捷键即可一键发布博客！

## 可用脚本

### `scripts/build.sh`
构建静态网站到 `public/` 目录。

```bash
./scripts/build.sh
```

### `scripts/deploy.sh`
构建并部署到生产环境。

```bash
./scripts/deploy.sh
```

### `scripts/publish.sh`
快速发布：自动提交 Git、构建网站。

```bash
./scripts/publish.sh
```

## 部署到生产环境

### 使用 Nginx

1. 安装 Nginx：
```bash
sudo apt install nginx  # Debian/Ubuntu
sudo yum install nginx  # CentOS/RHEL
```

2. 配置 Nginx（编辑 `/etc/nginx/conf.d/blog.conf`）：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /home/obsidian-blog/public;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

3. 重启 Nginx：
```bash
sudo systemctl restart nginx
```

### 推送到 GitHub

```bash
# 添加远程仓库
git remote add origin https://github.com/your-username/your-blog.git

# 推送代码
git add .
git commit -m "Initial commit"
git push -u origin master
```

## 自定义

### 修改网站信息

编辑 `hugo.toml`：

```toml
baseURL = 'https://your-domain.com/'
title = '你的博客标题'

[params.author]
  name = "你的名字"
  email = "your.email@example.com"
```

### 添加评论系统

在 `layouts/_default/single.html` 底部添加 [Giscus](https://giscus.app/) 代码。

### 自定义样式

修改 `layouts/_default/baseof.html` 中的 CSS 变量：

```css
:root {
    --bg-primary: #ffffff;
    --text-primary: #333333;
    --link-color: #0066cc;
    /* ... */
}
```

## 技术栈

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [Obsidian](https://obsidian.md/) - Markdown 编辑器
- [D3.js](https://d3js.org/) - 知识图谱可视化
- [Fuse.js](https://fusejs.io/) - 客户端搜索
- [Nginx](https://nginx.org/) - Web 服务器

## 许可证

MIT License

## 支持

如有问题，请查看：
- [Hugo 文档](https://gohugo.io/documentation/)
- [Obsidian 帮助](https://help.obsidian.md/)

---

Made with ❤️ using Hugo & Obsidian
