---
aliases:
  - latex
  - Latex
type: blog
---
# 介绍

# 快速入门
[编程三分钟|一个非常快速的 Latex 入门教程](https://www.bilibili.com/video/BV11h41127FD/?spm_id_from=333.337.search-card.all.click&vd_source=7ef7dff4b509c161e6b86a796dbad2c5)

在线LaTeX编辑器：https://www.overleaf.com
TeX Live下载：https://www.tug.org/texlive/acquire-iso.html 
MikTeX下载：https://miktex.org/download 
LaTeX 公式编辑器：https://latex.codecogs.com/eqneditor/editor.php 
[一份不太简短的LaTeX介绍](https://github.com/CTeX-org/lshort-zh-cn)

https://oi-wiki.org/tools/latex/


# MacOS本地配置latex环境
1. 安装，下载链接 [**MacTeX**](https://tug.org/mactex/mactex-download.html)
2. vscode安装 latex workshop ，[这篇博客](http://zhuanlan.zhihu.com/p/166523064)关于这个插件的使用配置讲的很好。复制配置参数如下：
```json
{
    "latex-workshop.latex.autoBuild.run": "onSave", 
    "latex-workshop.showContextMenu": true,
    "latex-workshop.intellisense.package.enabled": true,
    "latex-workshop.message.error.show": false,
    "latex-workshop.message.warning.show": false,
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOCFILE%"
            ]
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "-outdir=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DOCFILE%"
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": [
                "xelatex"
            ]
        },
        {
            "name": "PDFLaTeX",
            "tools": [
                "pdflatex"
            ]
        },
        {
            "name": "BibTeX",
            "tools": [
                "bibtex"
            ]
        },
        {
            "name": "LaTeXmk",
            "tools": [
                "latexmk"
            ]
        },
        {
            "name": "xelatex -> bibtex -> xelatex*2",
            "tools": [
                "xelatex",
                "bibtex",
                "xelatex",
                "xelatex"
            ]
        },
        {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        },
    ],
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk"
    ],
    "latex-workshop.latex.autoClean.run": "onFailed",
    "latex-workshop.latex.recipe.default": "lastUsed",
    "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click"
}
```









# MacOS VSCode MaTex+Skim pdf相互跳转
skim是macos上的一个免费pdf阅读编辑器，搜索官网下载即可。
参考博客 [Mac VS Code+TexLive+Skim实现正反跳转](https://zhuanlan.zhihu.com/p/570559163)设置快捷键实现相互跳转：
## vscode配置：
### 配置文件
```json
"latex-workshop.view.pdf.viewer": "external",

"latex-workshop.view.pdf.external.synctex.command": "/Applications/Skim.app/Contents/SharedSupport/displayline",

"latex-workshop.view.pdf.external.synctex.args": [

"-r",

"%LINE%",

"%PDF%",

"%TEX%"

],

"latex-workshop.view.pdf.external.viewer.command": "displayfile",

"latex-workshop.view.pdf.external.viewer.args": [

"-r",

"%PDF%"

],
```

### 修改快捷键
默认的快捷键是command+option+j , 可以修改keybindings.json文件实现覆盖：
```json
{
// 前向搜索
"key": "cmd+shift+a", // 按照个人习惯修改
"command": "latex-workshop.synctex",
"when": "editorTextFocus"
},
```

## skim配置
如下设置即可，默认的跳转到latex源码处的快捷键是command+shift+click（鼠标左键单击），似乎无法修改快捷键：
![[Pasted image 20250519110359.png]]

# 个人经验

[[制造空白、间距]]