---
title: "学习使用dataview"
date: 2025-08-13
draft: false
---

中文文档：[Dataview 基本语法学习指南](https://pkmer.cn/Pkmer-Docs/10-obsidian/obsidian%E7%A4%BE%E5%8C%BA%E6%8F%92%E4%BB%B6/dataview/dataview%E5%9F%BA%E6%9C%AC%E8%AF%AD%E6%B3%95/dataview%E5%9F%BA%E6%9C%AC%E8%AF%AD%E6%B3%95/)
Github：[obsidian-dataview](https://github.com/blacksmithgu/obsidian-dataview)


以下是 Obsidian Dataview 插件的核心使用方法总结，结合配置、语法和应用场景：

---

### 🔧 **1. 安装与配置**
- **安装**：在 Obsidian 社区插件市场搜索 "Dataview" 并安装。
- **关键配置**（插件设置中启用）：
  - `Enable JavaScript Queries`：允许执行 JS 查询。
  - `Enable Inline JavaScript Queries`：支持行内 JS 语法。
  - 日期格式建议设为 `yyyy-MM-dd`，时间格式设为 `yyyy-MM-dd HH:MM:ss`。

---

### 📊 **2. 元数据定义（查询基础）**
Dataview 通过笔记的元数据（属性）进行查询，支持三种定义方式：
- **Frontmatter（YAML区块）**：在笔记开头用 `---` 包裹，例如：
  ```yaml
  ---
  tags: [Book, AI]
  rating: 4.5
  due: 2025-08-20
  ---
  ```
- **内联字段**：在内容任意位置用 `字段:: 值` 格式，例如：
  `进度:: 75%` 或 `[优先级:: 紧急]`。
- **文件自带属性**：如 `file.name`（文件名）、`file.mtime`（修改时间）等。

> 💡 **字段命名规范**：含空格或特殊字符需转小写连字符（如 `Review Status` → `review-status`）。

---

### ⚙️ **3. 查询语法（DQL）**
在代码块中写入 ` ```dataview` 后接查询语句：
- **基础结构**：
  ```demo
  TABLE|LIST|TASK [字段]
  FROM "文件夹" OR #标签
  WHERE 条件
  SORT 字段 [ASC/DESC]
  ```
- **常用场景**：
  - **列出读书笔记**：
    ```dataview
    TABLE author, rating
    FROM #Book
    SORT rating DESC
    ```
  - **追踪待办任务**：
    ```dataview
    TASK FROM "Projects"
    WHERE !completed AND due <= date(today) + dur(7 days)
    GROUP BY priority
    ```
  - **最近修改文件**：
    ```dataview
    TABLE file.mtime
    FROM ""
    SORT file.mtime DESC
    LIMIT 5
    ```
- **条件过滤技巧**：
  - 文本：`icontains(file.name, "报告")`（不区分大小写）。
  - 日期：`file.cday >= date("2025-01-01")`。
  - 类型检查：`WHERE typeof(rating) = "number"` 避免类型错误。

---

### 💻 **4. JavaScript API（高阶扩展）**
在 ` ```dataviewjs` 代码块中用 JS 实现复杂逻辑：
- **示例：动态输出页面属性**：
  ```javascript
  dv.list(dv.pages("#Book").map(p => p.file.link));
  ```
- **支持操作**：
  - `dv.table()`：生成表格。
  - `dv.taskList()`：渲染任务列表。
  - `dv.el("span", "文本")`：自定义 HTML 元素。

---

### ⚠️ **5. 常见问题与优化**
- **错误处理**：内联字段需严格按 `字段:: 值` 格式，文本值加引号（如 `状态:: "进行中"`），避免 `Parsing Failed`。
- **性能优化**：
  - 用标签/文件夹限定查询范围（避免 `FROM ""` 全库扫描）。
  - 避免复杂正则或全文搜索。
- **数据联动**：结合 Charts 等插件可视化结果（如生成阅读进度图表）。

---

### 💎 **总结**
Dataview 将 Obsidian 转化为动态知识库，核心步骤：  
1. **定义属性** → 2. **编写查询**（DQL/JS）→ 3. **自动化输出表格/任务/列表**。  
适合需结构化管理笔记（如项目追踪、阅读记录）的用户。





## 可以复用的优秀代码片段

```dataviewjs
// 仅显示以数字开头的文件夹及其标签
for (let group of dv.pages("")
    .filter(p => p.file.folder != "")
    .groupBy(p => p.file.folder.split("/")[0])
    .filter(g => /^\d/.test(g.key))) {  // 正则匹配以数字开头的文件夹名
  dv.paragraph(`## ${group.key}`);
  dv.paragraph(
    dv.pages(`"${group.key}"`)
      .file.tags
      .distinct()
      .map(t => `[${t}](${t})`)
      .sort()
      .join(" | ")
  );
}
```


## 学习资源
[入门教学视频](https://www.bilibili.com/video/BV1Hu4y1w7tL/?spm_id_from=333.337.search-card.all.click&vd_source=7ef7dff4b509c161e6b86a796dbad2c5)
[官网](https://blacksmithgu.github.io/obsidian-dataview/)
[Github仓库](https://github.com/blacksmithgu/obsidian-dataview)
[Obsidian论坛经验分享](https://forum-zh.obsidian.md/t/topic/5954)
