---
title: "Hugo和Obsidian集成指南"
date: 2025-12-23
tags: ["Hugo", "Obsidian", "教程"]
categories: ["技术"]
draft: false
---

本文介绍如何将 Obsidian 与 Hugo 集成，打造一个强大的个人知识管理系统。

## 为什么选择 Hugo + Obsidian?

### Hugo 的优势
- ⚡ **极速构建**: 秒级生成静态网站
- 📦 **单一二进制**: 无需复杂依赖
- 🎨 **灵活主题**: 高度可定制

### Obsidian 的优势
- 📝 **本地优先**: 数据完全属于你
- 🔗 **双向链接**: 构建知识网络
- 🔌 **插件丰富**: 扩展性强

## 核心功能

### 1. Wikilinks 支持

在 Obsidian 中，你可以使用双方括号创建链接：

```markdown
[[欢迎来到我的数字花园]]
[[另一篇文章|显示文本]]
```

这些链接会自动转换为 Hugo 的内部链接。

### 2. 图片引用

Obsidian 风格的图片引用：

```markdown
![[image.png]]
```

也可以使用标准 Markdown：

```markdown
![描述](/images/image.png)
```

### 3. 反向链接

每篇文章底部会自动显示哪些文章引用了它，这有助于发现知识之间的关联。参见[[欢迎来到我的数字花园]]。

## 工作流程

1. 在 Obsidian 中编写 Markdown 文章
2. 添加 Front Matter（标题、日期、标签等）
3. 使用双链连接相关内容
4. 运行构建脚本生成网站
5. 部署到服务器

## 代码示例

这是一段 Python 代码示例：

```python
def hello_world():
    print("Hello from Hugo + Obsidian!")
    return "Success"

if __name__ == "__main__":
    hello_world()
```

## 总结

Hugo + Obsidian 的组合让我们能够：
- 用熟悉的工具编写内容
- 享受双链和知识图谱的便利
- 生成快速、美观的静态网站

继续阅读：[[如何使用标签和分类]]
