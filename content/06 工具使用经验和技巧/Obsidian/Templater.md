# 学习资源
1. 视频 [B站教学](https://www.bilibili.com/video/BV1c64y1W7c2/?spm_id_from=333.337.search-card.all.click&vd_source=7ef7dff4b509c161e6b86a796dbad2c5)
2. Documents [官方文档](https://silentvoid13.github.io/Templater/internal-functions/internal-modules/file-module.html#tpfilecursororder-number)
3. [Github repo](https://github.com/SilentVoid13/Templater)
4. 中文教程： [Obsidian 插件：Templater 可以替代核心模板插件的效率神器](https://pkmer.cn/Pkmer-Docs/10-obsidian/obsidian%E7%A4%BE%E5%8C%BA%E6%8F%92%E4%BB%B6/templater/templater-obsidian/)

# 


# Templater插件使用指南

## 基础概念

Templater是Obsidian中一个强大的模板插件，它允许你创建动态模板，比Obsidian内置模板功能更强大。

### 核心概念

1. **模板文件**：普通的Markdown文件，包含占位符和脚本
2. **占位符**：形如`<% %>`的语法，用于插入动态内容
3. **命令**：以`tp.`开头的JavaScript函数，用于执行各种操作
4. **触发器**：自动执行模板的条件（如文件创建、命令面板等）

## 常用场景与用法

### 1. 基本变量插入

```markdown
<%*
// 当前日期
const today = tp.date.now("YYYY-MM-DD")
%>
# 每日笔记 - <%= today %>
```

### 2. 动态文件名

```javascript
<%*
const title = await tp.system.prompt("输入标题")
const filename = `${tp.date.now("YYYY-MM-DD")}-${title}`
await tp.file.rename(filename)
%>
```

### 3. 内容生成

```markdown
<%*
// 生成随机ID
const randomId = Math.random().toString(36).substring(2,8)
%>
唯一标识符: <%= randomId %>
```

### 4. 文件操作

```javascript
<%*
// 创建新文件并插入链接
const newNote = await tp.file.create_new(tp.file.find_tfile("模板"), "新笔记")
%>
相关笔记: [[<% newNote.basename %>]]
```

### 5. 用户输入

```javascript
<%*
const project = await tp.system.suggester(
    ["项目A", "项目B", "项目C"], 
    ["projA", "projB", "projC"]
)
%>
当前项目: **<%= project %>**
```

### 6. 条件逻辑

```javascript
<%*
const time = tp.date.now("H")
let greeting
if (time < 12) greeting = "早上好"
else if (time < 18) greeting = "下午好"
else greeting = "晚上好"
%>
<%= greeting %>！现在是<% tp.date.now("HH:mm") %>
```

### 7. 循环结构

```markdown
<%*
const days = ["周一", "周二", "周三", "周四", "周五"]
%>
本周计划:
<% for (let day of days) { %>
- [ ] <%= day %>: 
<% } %>
```

### 8. 调用系统命令

```javascript
<%*
// 获取剪贴板内容
const clipboard = await tp.system.clipboard()
%>
剪贴板内容: `<%= clipboard %>`
```

### 9. 模板嵌套

```javascript
<%*
// 插入另一个模板
await tp.file.include("[[模板-头部]]")
%>
```

### 10. 自动任务列表

```markdown
<%*
const tasks = [
    "检查邮件",
    "团队会议",
    "编写报告"
]
%>
今日任务:
<% for (let task of tasks) { %>
- [ ] <%= task %>
<% } %>
```

## 高级技巧

1. **使用JavaScript模块**：通过`tp.user`访问用户定义的JS函数
2. **自定义触发器**：设置模板在特定条件下自动执行
3. **结合Dataview**：在模板中嵌入Dataview查询
4. **错误处理**：使用try-catch处理可能的错误

## 最佳实践

1. 将常用模板存放在专用文件夹中
2. 为复杂模板添加注释
3. 使用版本控制备份模板
4. 定期整理和优化模板库

Templater的强大之处在于它将Markdown的静态性与JavaScript的动态性完美结合，可以极大提升笔记效率。