#!/bin/bash

# Obsidian + Hugo 博客部署脚本
# 用途：构建并部署到生产环境（Nginx）

set -e  # 遇到错误立即退出

echo "🚀 开始部署博客..."

# 进入项目目录
cd /home/obsidian-blog

# 运行构建
echo "📦 构建网站..."
./scripts/build.sh

# 可选：重启 Nginx（如果需要）
# echo "🔄 重启 Nginx..."
# sudo systemctl restart nginx

echo "✅ 部署完成！"
echo "🌐 博客地址：http://您的服务器IP/"
echo ""
echo "💡 提示："
echo "  - 静态文件位于: /home/obsidian-blog/public"
echo "  - 请确保 Nginx 配置指向 public 目录"
