#!/bin/bash

# Obsidian + Hugo 博客构建脚本
# 用途：构建静态网站

set -e  # 遇到错误立即退出

echo "🚀 开始构建博客..."

# 进入项目目录
cd /home/obsidian-blog

# 清理旧的构建
echo "🧹 清理旧的构建文件..."
rm -rf public

# 运行 Hugo 构建
echo "📦 运行 Hugo 构建..."
hugo --minify

# 检查构建结果
if [ -d "public" ]; then
    echo "✅ 构建成功！"
    echo "📊 构建统计："
    echo "  - 文件数量: $(find public -type f | wc -l)"
    echo "  - 目录大小: $(du -sh public | cut -f1)"
else
    echo "❌ 构建失败！"
    exit 1
fi

echo "🎉 构建完成！"
