#!/bin/bash

# Obsidian + Hugo 快速发布脚本
# 用途：从 Obsidian 一键提交、构建和部署

set -e

echo "📝 开始快速发布..."

# 进入项目目录
cd /home/obsidian-blog

# 检查 Git 状态
if ! git diff-index --quiet HEAD --; then
    echo "📂 检测到文件变更"

    # 显示变更的文件
    echo "变更的文件："
    git status --short

    # 添加所有变更
    echo "➕ 添加文件到 Git..."
    git add .

    # 提交变更（使用时间戳作为提交信息）
    COMMIT_MSG="📝 Update blog content - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "💾 提交变更：$COMMIT_MSG"
    git commit -m "$COMMIT_MSG"

    echo "✅ Git 提交完成"
else
    echo "ℹ️  没有检测到文件变更"
fi

# 构建网站
echo "📦 构建网站..."
./scripts/build.sh

# 可选：推送到 GitHub
# echo "⬆️  推送到 GitHub..."
# git push origin main

echo "🎉 发布完成！"
echo ""
echo "💡 下一步："
echo "  1. 在浏览器中查看博客"
echo "  2. （可选）运行 'git push' 推送到 GitHub"
