# 配置 Giscus 评论系统

Giscus 是一个基于 GitHub Discussions 的评论系统，无需数据库，完全免费。

## 🚀 快速配置步骤

### 1. 准备 GitHub 仓库

1. 在 GitHub 创建或使用现有的**公开仓库**
2. 进入仓库 Settings → General → Features
3. 勾选 **Discussions** 启用讨论功能

### 2. 安装 Giscus App

1. 访问：https://github.com/apps/giscus
2. 点击 **Install**
3. 选择要安装的仓库
4. 授权访问

### 3. 获取配置代码

1. 访问：https://giscus.app/zh-CN
2. 填写你的仓库信息：
   - **仓库**：`你的用户名/仓库名`
   - **页面 ↔️ discussion 映射关系**：选择 `pathname`
   - **Discussion 分类**：选择 `General` 或创建新分类
   - **特性**：勾选你想要的功能（推荐全选）
3. 向下滚动，复制生成的配置参数

### 4. 更新博客配置

编辑 `/home/obsidian-blog/layouts/_default/single.html`，找到 Giscus 脚本部分（约第93行），替换以下参数：

```html
<script src="https://giscus.app/client.js"
        data-repo="YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"  <!-- 改成你的仓库 -->
        data-repo-id="YOUR_REPO_ID"                      <!-- 从 giscus.app 复制 -->
        data-category="General"                           <!-- 你的分类名称 -->
        data-category-id="YOUR_CATEGORY_ID"              <!-- 从 giscus.app 复制 -->
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"               <!-- 自动跟随主题 -->
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>
```

### 5. 重新构建网站

```bash
cd /home/obsidian-blog
./scripts/build.sh
```

## 📝 配置示例

假设你的 GitHub 用户名是 `johndoe`，仓库名是 `my-blog`：

```html
data-repo="johndoe/my-blog"
data-repo-id="R_kgDOAbcdef"  <!-- 这个从 giscus.app 获取 -->
data-category="General"
data-category-id="DIC_kwDOAbcdef4AABcd"  <!-- 这个也从 giscus.app 获取 -->
```

## 🎨 主题配置说明

当前配置使用 `preferred_color_scheme`，会自动跟随博客的深色/亮色模式。

其他可选主题：
- `light` - 亮色主题
- `dark` - 暗色主题
- `dark_dimmed` - 暗淡暗色
- `transparent_dark` - 透明暗色
- `preferred_color_scheme` - 跟随系统（推荐）

## ✅ 验证配置

1. 重新构建网站
2. 打开任意文章页面
3. 滚动到底部
4. 你应该看到 Giscus 评论框
5. 使用 GitHub 账号登录即可评论

## 💡 使用建议

### 分类设置
在 GitHub 仓库的 Discussions 中可以创建不同分类：
- **General** - 通用评论
- **Blog Comments** - 博客评论（推荐单独创建）
- **Q&A** - 问答
- **Ideas** - 想法建议

### 权限管理
在仓库 Settings → Discussions 可以：
- 设置谁可以创建讨论
- 设置谁可以评论
- 管理评论规则

### 通知设置
评论后你会收到 GitHub 邮件通知，可以在：
Settings → Notifications 中管理通知偏好

## 🔧 故障排查

### 评论框不显示
1. 检查仓库是否公开
2. 确认 Discussions 已启用
3. 确认 Giscus App 已安装
4. 检查配置参数是否正确
5. 查看浏览器控制台是否有错误

### 无法登录
1. 清除浏览器缓存
2. 确认 GitHub 账号正常
3. 检查是否被仓库屏蔽

### 样式问题
在 `single.html` 的样式部分可以自定义 `.giscus-container` 的样式。

## 📚 更多资源

- Giscus 官网：https://giscus.app/zh-CN
- GitHub Discussions 文档：https://docs.github.com/en/discussions
- Giscus GitHub 仓库：https://github.com/giscus/giscus

---

配置完成后，你的博客就有了一个完全免费、无广告、支持 Markdown 的评论系统！
