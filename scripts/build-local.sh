#!/bin/bash

# Local build script for Obsidian + Hugo blog
# Usage: Run from project root or anywhere (auto-detects directory)

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting local build..."
echo "📁 Project directory: $PROJECT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Clean old build
echo "🧹 Cleaning old build files..."
rm -rf public

# Run Hugo build
echo "📦 Running Hugo build..."
hugo --minify

# Check build result
if [ -d "public" ]; then
    echo "✅ Build successful!"
    echo "📊 Build statistics:"
    echo "  - File count: $(find public -type f | wc -l | tr -d ' ')"
    echo "  - Directory size: $(du -sh public | cut -f1)"
    echo ""
    echo "💡 Next step: Run ./scripts/deploy-local.sh to sync to server"
else
    echo "❌ Build failed!"
    exit 1
fi

echo "🎉 Build complete!"
