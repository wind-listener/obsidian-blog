#!/bin/bash

# Deploy script to sync built site to remote server via rsync
# Usage: ./scripts/deploy-local.sh

set -e  # Exit on error

# Configuration
REMOTE_HOST="Aliyun"
REMOTE_PATH="/home/zzm/obsidian-blog/public"
LOCAL_PUBLIC="./public"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Starting deployment to remote server..."
echo "📡 Remote host: $REMOTE_HOST"
echo "📁 Remote path: $REMOTE_PATH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Check if public directory exists
if [ ! -d "$LOCAL_PUBLIC" ]; then
    echo -e "${RED}❌ Error: public directory not found!${NC}"
    echo "💡 Please run ./scripts/build-local.sh first"
    exit 1
fi

# Show what will be synced
echo ""
echo -e "${YELLOW}📊 Files to sync:${NC}"
find public -type f | wc -l | xargs echo "  - Total files:"
du -sh public | cut -f1 | xargs echo "  - Total size:"
echo ""

# Ask for confirmation
read -p "Continue with deployment? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}❌ Deployment cancelled${NC}"
    exit 0
fi

# Sync with rsync
echo ""
echo "📦 Syncing files to remote server..."
rsync -avz --delete \
    --progress \
    --exclude='.git*' \
    --exclude='*.log' \
    "$LOCAL_PUBLIC/" \
    "$REMOTE_HOST:$REMOTE_PATH/"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo "💡 Tip: SSH into server to restart Nginx if needed:"
    echo "   ssh Aliyun"
    echo "   sudo systemctl reload nginx"
else
    echo ""
    echo -e "${RED}❌ Deployment failed!${NC}"
    exit 1
fi
