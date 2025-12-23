---
aliases:
  - Obsidian
  - 如何做好笔记
  - Obsidian入门指南
creation date: "{{data}}"
obsidianUIMode: preview
---
我很早就下载了Obsidian，但是刚开始真的很难上手。很长一段时间内，Obsidian只是我一个markdown编辑器而已，还不怎么能想起来用它。所谓的双链功能、打造第二大脑如何强大，我一直难以理解。

但是随着时间累计，Obsidian的简洁优雅确实呈现出了一些优势，才发现原来这里藏有如此巨大的宝藏。

# 一些很有启发的优秀经验
做好笔记系统总重要的一直是人，而不是工具。如果使用工具的思路不清晰，再好的设计也是暴殄天物。下面收集了一些我在探索过程中发现的非常有借鉴意义的经验，相信这些内容会帮助你更好的理解Obsidian的设计思路，更重要的是，修炼出属于自己的笔记心法。

[3年用户用心整理 从底层逻辑出发的Obsidian教程 | 用得越朴实，就越有价值](https://www.bilibili.com/video/BV1y5D6YpEB7/?share_source=copy_web&vd_source=bc768ece925f350d8d8d46d7f5ccddd1)

[『晒晒我的工作区』原生主题+Snippets爱好者 | 学生党](https://forum-zh.obsidian.md/t/topic/178)

[工作流分享：我的文献阅读笔记流程](https://forum-zh.obsidian.md/t/topic/292)

[笔记系统？方法论？](https://functoreality.github.io/blog-pkm/)

[MOC的组织](https://functoreality.github.io/blog-pkm/contents/MOC%E7%9A%84%E7%BB%84%E7%BB%87/)

[Obsidian中文教程](https://publish.obsidian.md/chinesehelp/01+2021%E6%96%B0%E6%95%99%E7%A8%8B/2021%E5%B9%B4%E6%96%B0%E6%95%99%E7%A8%8B)一个收集了很多优秀Obsidian实践的库

[LYT笔记法](https://publish.obsidian.md/chinesehelp/01+2021%E6%96%B0%E6%95%99%E7%A8%8B/LYT%E7%AC%94%E8%AE%B0%E6%B3%95)



# 基础
掌握以下基础功能，就已经可以实现非常丝滑的体验了：
- [[Markdown]]语法
- 双链功能
- 笔记属性，metadata
- 标签系统  
- 检索方式  [搜索](https://publish.obsidian.md/help-zh/%E6%A0%B8%E5%BF%83%E6%8F%92%E4%BB%B6/%E6%90%9C%E7%B4%A2)
- 自定义快捷键方式 [[软件使用/Obsidian/Obsidian快捷键|Obsidian快捷键]]
这些基础功能在 [Obsidian中文官方帮助网站可以找到很好的介绍](https://publish.obsidian.md/help-zh/%E7%94%B1%E6%AD%A4%E5%BC%80%E5%A7%8B)

# 进阶
Obsidian的软件本体确实十分基础——但是一般来说对于实现一个笔记系统是足够的——如果的确需要一些进阶功能，需要借助于开放的插件生态。

## 模板

## 日记

## 同步
同步这个事情苦恼了我很久。最简单的是直接把仓库文件夹放在一个同步文件夹内，可以提供这种的服务的有iCloud、OneDrive、以及其他各种网盘工具等。

WPS云盘的同步文件夹功能很慢。

### Remotely Save


## Dataview

# 实用小功能
一些小的特性或者插件，不是必须的，内容也比较杂乱，如果发现有适合的功能就再好不过了。

## 设置单条笔记的默认视图
发现 [Force note view mode](https://github.com/bwydoogh/obsidian-force-view-mode-of-note) 插件支持在 frontmatter 中添加属性来设置单条笔记的视图模式，很好用。

- `obsidianUIMode` 属性设置视图模式，属性值支持 `source` 和 `preview`，分别对应「编辑视图」和「阅读视图」
- `obsidianEditingMode` 属性设置编辑模式，属性值支持 `live` 和 `source`，分别对应「实时预览」和「源码模式」

# TODO
- [ ] 如何快速键入当天的时间，在编辑状态；或者直接新建笔记时添加时间
- [ ] 厨艺MOC中列出已会的菜肴
- [ ] 如何让dataview显示的文件可以在关系图谱中显示？
- [ ] 有必要使用Linter插件吗？
