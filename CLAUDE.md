# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Obsidian + Hugo static blog** that transforms Obsidian vault content into a static website. The key architectural principle is that content lives in `content/` as standard Markdown files that can be edited in Obsidian, and Hugo transforms them into a static site in `public/`.

**Critical Path**: `content/*.md` → Hugo build → `public/` → Nginx serves static files

## Build and Deployment Commands

### Local Development
```bash
# Start Hugo development server (live reload)
hugo server --bind 0.0.0.0 --port 1313 --buildDrafts

# View at http://localhost:1313 or http://server-ip:1313
```

### Production Build
```bash
# Standard build (minified)
./scripts/build.sh
# OR directly:
hugo --minify

# Output goes to: /home/obsidian-blog/public/
```

### Publishing Workflow
```bash
# Quick publish: auto-commit + build (does NOT push to GitHub)
./scripts/publish.sh

# Deploy: build + optionally restart Nginx
./scripts/deploy.sh
```

### Nginx Management
```bash
# Test configuration
sudo nginx -t

# Restart Nginx to apply changes
sudo systemctl restart nginx

# Check status
sudo systemctl status nginx

# View logs
sudo tail -f /var/log/nginx/blog_error.log
sudo tail -f /var/log/nginx/blog_access.log
```

### Git Operations
```bash
# The repository is at: /home/obsidian-blog/
# Remote: https://github.com/wind-listener/obsidian-blog.git

git add .
git commit -m "message"
git push
```

## Architecture

### Content Organization
- **`content/`**: Main Obsidian vault - edit content here
  - Organized by numbered categories: `01 电子信息与计算机专业/`, `02 Artificial Intelligence/`, etc.
  - `posts/`: Blog posts
  - `attachments/`: Media files
  - Front matter required: `title`, `date`, `draft: false`

- **`public/`**: Generated static site (gitignored) - **never edit directly**

### Template System (Hugo Layouts)
- **`layouts/_default/baseof.html`**: Base template with CSS variables for theming
- **`layouts/_default/single.html`**: Article template with:
  - Sidebar navigation tree (categorized by section)
  - Giscus comment system integration (lines 86-101)
  - Breadcrumb navigation
- **`layouts/_default/list.html`**: Category/section listing pages
- **`layouts/index.html`**: Homepage with category cards
- **`layouts/_default/search.html`**: Search page using Fuse.js
- **`layouts/_default/graph.html`**: Knowledge graph visualization using D3.js

### Configuration Files
- **`hugo.toml`**: Hugo configuration
  - `baseURL`: Currently `http://zzmblog.top/`
  - `params.author`: name="Benjamin", email="zzm_ai@bupt.edu.cn"
  - `security.exec.allow`: Whitelist for AsciiDoc (`asciidoctor`) and reStructuredText (`rst2html`)

- **`/etc/nginx/conf.d/blog.conf`**: Nginx server configuration
  - Serves from: `/home/obsidian-blog/public`
  - Server names: `zzmblog.top`, `www.zzmblog.top`

### Special Features
1. **Obsidian Wikilinks**: `[[page-name]]` syntax supported
2. **Knowledge Graph**: D3.js visualization of page relationships
3. **Giscus Comments**: GitHub Discussions-based comments on single pages
4. **Dark/Light Mode**: CSS variable-based theming with localStorage persistence
5. **Full-text Search**: Client-side search using Fuse.js over JSON index

## Key Technical Details

### Hugo Security Policy
The `hugo.toml` includes a security whitelist for external processors:
```toml
[security.exec]
  allow = ['^asciidoctor$', '^rst2html(\.py)?$', ...]
```
This is required because content includes Python example code with AsciiDoc/reStructuredText files.

### Build Warnings
- `rst2html / rst2html.py not found` warnings during build are non-critical (those files are left unrendered)
- Build succeeds as long as you see `Total in Xms` and `✅ 构建成功！`

### Nginx Configuration Notes
- The main `/etc/nginx/nginx.conf` has the default server block **commented out** to avoid conflicts
- Active config is in `/etc/nginx/conf.d/blog.conf`
- If you see `conflicting server name "_"` warnings, they're harmless

### Proxy Considerations
This server has system-level proxy settings (`http_proxy=http://127.0.0.1:7890`). When testing locally:
```bash
# Bypass proxy for local testing
curl --noproxy "*" http://localhost/
```

## Content Creation

### New Article Template
```markdown
---
title: "Article Title"
date: 2025-12-23
tags: ["tag1", "tag2"]
categories: ["category"]
draft: false
---

Content here...

Use [[Wikilink]] for internal links.
```

### Creating Articles
```bash
# Using Hugo
hugo new posts/my-post.md

# Or manually create in content/posts/ or any category folder
```

### Image Handling
1. Place images in `static/images/` or `content/attachments/`
2. Reference as: `![alt](/images/name.png)` or `![[name.png]]` (Obsidian syntax)

## Customization

### Theme Colors
Edit CSS variables in `layouts/_default/baseof.html`:
```css
:root {
    --bg-primary: #ffffff;
    --text-primary: #1a1a1a;
    --link-color: #5568fe;
    /* ... */
}
```

### Giscus Configuration
Current setup:
- Repo: `wind-listener/obsidian-blog`
- Repo ID: `R_kgDOQtryCg`
- Category: `General`
- Edit in: `layouts/_default/single.html` (lines 86-101)

See `GISCUS_CONFIG.md` for reconfiguration instructions.

## Troubleshooting

### Build Fails
1. Check front matter format (YAML must be valid)
2. Ensure `draft: false` is set
3. Run `hugo --verbose` for detailed errors

### Changes Not Visible
1. Clear `public/` and rebuild: `rm -rf public && ./scripts/build.sh`
2. Hard refresh browser (Ctrl+F5)
3. Check Nginx is serving from correct path: `/home/obsidian-blog/public`

### Development Server Issues
```bash
# Kill existing Hugo processes
pkill hugo

# Restart
hugo server --bind 0.0.0.0 --port 1313 --buildDrafts
```

## Important Notes

- **Never edit files in `public/`** - they are regenerated on each build
- **Always rebuild after config changes**: `./scripts/build.sh`
- **Nginx must be restarted** after changing `/etc/nginx/conf.d/blog.conf`
- Content is version-controlled, `public/` is gitignored
- This blog uses Chinese category names - preserve them when modifying structure
