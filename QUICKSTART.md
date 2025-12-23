# 快速开始指南

欢迎使用你的 Obsidian + Hugo 博客！这个指南将帮助你快速开始使用。

## 📁 项目位置

你的博客位于：`/home/obsidian-blog/`

## 🎯 现在可以做什么？

### 1. 查看博客（开发模式）

Hugo 开发服务器正在运行中：

```
访问地址：http://你的服务器IP:1313
```

你可以实时看到文章的效果！

### 2. 在 Obsidian 中编写文章

#### 设置 Obsidian

1. 打开 Obsidian 应用
2. 点击"打开文件夹作为仓库"
3. 选择路径：`/home/obsidian-blog/content`
4. 开始写作！

#### 创建新文章

在 `content/posts/` 目录下创建新的 `.md` 文件：

```markdown
---
title: "我的第一篇博客"
date: 2025-12-23
tags: ["生活", "思考"]
categories: ["随笔"]
draft: false
---

这是文章内容...

可以使用 [[Hugo和Obsidian集成指南]] 创建双向链接。
```

### 3. 构建和发布

#### 方法 A：手动构建

```bash
cd /home/obsidian-blog
./scripts/build.sh
```

构建后的静态文件在 `public/` 目录。

#### 方法 B：快速发布（推荐）

```bash
cd /home/obsidian-blog
./scripts/publish.sh
```

这会自动：
- 提交 Git 变更
- 构建网站
- 准备发布

#### 方法 C：从 Obsidian 一键发布

安装 **Shell commands** 插件，添加命令：

```bash
cd /home/obsidian-blog && ./scripts/publish.sh
```

绑定快捷键（如 `Ctrl+P`），写完文章直接按快捷键发布！

## 🌐 部署到生产环境

### 安装 Nginx

如果还没有安装 Nginx：

```bash
# 临时取消代理（如果需要）
unset http_proxy https_proxy

# 安装 Nginx
sudo yum install -y nginx

# 启动 Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 配置 Nginx

1. 复制配置示例：

```bash
sudo cp /home/obsidian-blog/nginx.conf.example /etc/nginx/conf.d/blog.conf
```

2. 编辑配置文件：

```bash
sudo vi /etc/nginx/conf.d/blog.conf
```

修改 `server_name`为你的域名或 IP 地址。

3. 测试和重启：

```bash
# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

4. 访问你的博客：

```
http://你的服务器IP/
```

## 📤 推送到 GitHub

### 1. 在 GitHub 创建仓库

访问 https://github.com/new 创建新仓库。

### 2. 添加远程仓库

```bash
cd /home/obsidian-blog
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin master
```

## 🔧 常用命令

### Hugo 命令

```bash
# 开发服务器
hugo server --bind 0.0.0.0 --port 1313

# 构建（生产环境）
hugo --minify

# 创建新文章
hugo new posts/my-post.md
```

### Git 命令

```bash
# 查看状态
git status

# 提交变更
git add .
git commit -m "更新文章"

# 推送到 GitHub
git push
```

## 🎨 自定义网站

### 修改网站标题和描述

编辑 `hugo.toml`：

```toml
title = '你的博客名称'

[params]
  description = "你的博客描述"

  [params.author]
    name = "你的名字"
    email = "your.email@example.com"
```

### 修改颜色主题

编辑 `layouts/_default/baseof.html`，找到 CSS 变量：

```css
:root {
    --bg-primary: #ffffff;
    --text-primary: #333333;
    --link-color: #0066cc;  /* 修改这个颜色 */
}
```

## 🆘 常见问题

### Q: 如何停止开发服务器？

```bash
# 查看运行的任务
/tasks

# 停止任务
pkill hugo
```

### Q: 构建失败怎么办？

检查文章的 Front Matter 格式是否正确：

```yaml
---
title: "标题"  # 必需
date: 2025-12-23  # 必需
tags: ["标签"]  # 可选
draft: false  # 必需
---
```

### Q: 如何添加图片？

1. 将图片放到 `static/images/` 目录
2. 在文章中引用：

```markdown
![图片描述](/images/your-image.png)
```

或使用 Obsidian 语法：

```markdown
![[your-image.png]]
```

## 📚 进一步学习

- [Hugo 文档](https://gohugo.io/documentation/)
- [Obsidian 帮助](https://help.obsidian.md/)
- [Markdown 语法](https://www.markdownguide.org/)

## 🎉 开始创作吧！

现在一切已经就绪，开始你的写作之旅吧！

---

需要帮助？查看 README.md 或访问项目文档。
