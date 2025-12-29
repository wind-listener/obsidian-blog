# Implementation Plan: Obsidian Blog Improvements

## Overview

Implement 5 major improvements to the Obsidian + Hugo blog:
1. Enable HTTPS with SSL certificates
2. Delete content/posts folder
3. Fix article readability (batch add front matter + fix URL mismatch)
4. Add collapsible sidebar navigation with JavaScript
5. Add right sidebar knowledge graph widget

## Critical Issues Identified

**Root Cause of Unreadable Articles:**
- 570+ MD files missing Hugo front matter (title, date, draft: false)
- URL mismatch: Templates use "03 Mathematics" but Hugo generates "03-mathematics"
- Both issues prevent proper article display and navigation

## Implementation Phases

### PHASE 1: Add Front Matter to All Articles

**Create Script:** `/home/obsidian-blog/scripts/add_frontmatter.py`

Python script that:
- Scans all `.md` files in content/ (excluding posts, attachments, hidden files)
- Checks if file already has front matter (starts with `---`)
- Extracts title from first `# Heading` or uses filename
- Uses file modification time as date
- Adds YAML front matter with title, date, draft: false

**Execution:**
```bash
chmod +x /home/obsidian-blog/scripts/add_frontmatter.py
python3 scripts/add_frontmatter.py --dry-run  # Preview changes
python3 scripts/add_frontmatter.py            # Execute
hugo --minify                                 # Test build
```

**Rollback:** Git commit before execution, use `git reset --hard HEAD` if needed

### PHASE 2: Delete Posts Folder

**Actions:**
1. Backup: `cp -r content/posts/ /tmp/posts_backup_$(date +%Y%m%d)`
2. Update `hugo.toml`: Remove "文章" menu item (lines 42-44)
3. Update `layouts/_default/graph.html` line 15: Change from `{{ range where .Site.RegularPages "Section" "posts" }}` to `{{ range .Site.RegularPages }}`
4. Delete: `rm -rf /home/obsidian-blog/content/posts/`
5. Test rebuild and verify menu updated

### PHASE 3: Fix URL Mismatch in Templates

**File:** `/home/obsidian-blog/layouts/_default/single.html`

**Changes:**
- Line 10-13: Add `{{ $currentSectionUrlized := .Section | urlize }}`
- Line 32: Fix comparison to use `{{ if eq $currentSection $section }}`
- Lines 33, 58: Already use `urlize` correctly, verify only

This fixes the active section highlighting and ensures all links use hyphenated URLs.

### PHASE 4: Enhanced Sidebar Navigation

**Create:** `/home/obsidian-blog/static/js/sidebar.js`

JavaScript features:
- Add collapse/expand toggle buttons (▼/▶) to each section
- Click handler to toggle section visibility
- Save collapsed state to localStorage
- Restore state on page load
- Auto-expand active section

**Update:** `/home/obsidian-blog/layouts/_default/single.html`

Add CSS:
- `.section-toggle` styling for arrow icons
- `.tree-section.collapsed` for hidden state
- Smooth transitions for expand/collapse

**Update:** `/home/obsidian-blog/layouts/_default/baseof.html`

Add before `</body>`: `<script src="/js/sidebar.js"></script>`

### PHASE 5: Right Sidebar Knowledge Graph Widget

**Create:** `/home/obsidian-blog/static/js/article-graph.js`

JavaScript logic:
- Fetch `/index.json` (site-wide content index)
- Parse wikilinks from current article: `[[Page Title]]` format
- Find outgoing links (pages this article links to)
- Find incoming links (pages that link to this article)
- Display up to 10 connections per direction
- Show message if no connections found

**Update:** `/home/obsidian-blog/layouts/_default/single.html`

Layout changes:
- Line 110: Change grid from 2-column to 3-column: `grid-template-columns: 280px 1fr 260px`
- Add responsive breakpoints:
  - 1400px: Hide right sidebar (keep left)
  - 1024px: Hide both sidebars
- Add right sidebar HTML after article content:
  ```html
  <aside class="right-sidebar">
      <div class="graph-widget">
          <h3>🔗 Connections</h3>
          <div id="article-graph-widget"
               data-page-title="{{ .Title }}"
               data-page-url="{{ .Permalink }}">
          </div>
      </div>
  </aside>
  ```
- Add CSS for `.right-sidebar`, `.graph-widget`, `.graph-links`, etc.

**Update:** `/home/obsidian-blog/layouts/_default/baseof.html`

Add before `</body>`: `<script src="/js/article-graph.js"></script>`

### PHASE 6: HTTPS Configuration

**Step 1: Install Certbot**
```bash
sudo yum install -y epel-release
sudo yum install -y certbot python3-certbot-nginx
```

**Step 2: Obtain SSL Certificate**
```bash
sudo systemctl stop nginx
sudo certbot certonly --standalone \
    -d zzmblog.top \
    -d www.zzmblog.top \
    --email zzm_ai@bupt.edu.cn \
    --agree-tos \
    --no-eff-email
```

Certificates will be at:
- `/etc/letsencrypt/live/zzmblog.top/fullchain.pem`
- `/etc/letsencrypt/live/zzmblog.top/privkey.pem`

**Step 3: Update Nginx Configuration**

**File:** `/etc/nginx/conf.d/blog.conf`

Replace entire content with:
- HTTP server block (port 80): Redirect to HTTPS
- HTTPS server block (port 443):
  - SSL certificate paths
  - TLSv1.2, TLSv1.3 protocols
  - HSTS header
  - All existing settings (gzip, caching, security headers)

**Step 4: Update Hugo Configuration**

**File:** `/home/obsidian-blog/hugo.toml`

Line 1: Change `baseURL = 'http://zzmblog.top/'` to `baseURL = 'https://zzmblog.top/'`

**Step 5: Deploy**
```bash
sudo nginx -t                  # Test config
sudo systemctl start nginx     # Start Nginx
cd /home/obsidian-blog
./scripts/build.sh             # Rebuild with HTTPS URLs
curl -I https://zzmblog.top/   # Verify HTTPS works
```

**Step 6: Auto-Renewal**
```bash
sudo certbot renew --dry-run   # Test renewal
sudo systemctl list-timers | grep certbot  # Verify timer
```

### PHASE 7: Final Testing

**Build & Deploy:**
```bash
cd /home/obsidian-blog
rm -rf public/
./scripts/build.sh
```

**Test Checklist:**
- [ ] All 570+ articles accessible with correct titles
- [ ] Section navigation links work (no 404s)
- [ ] Sidebar sections collapse/expand on click
- [ ] Collapsed state persists across page reloads
- [ ] Right sidebar shows article connections
- [ ] Graph widget links are clickable
- [ ] HTTPS loads successfully
- [ ] HTTP redirects to HTTPS
- [ ] No SSL warnings in browser
- [ ] Giscus comments work with HTTPS
- [ ] Responsive design works on mobile
- [ ] Dark mode toggle still functions

## Critical Files

**New Files:**
- `/home/obsidian-blog/scripts/add_frontmatter.py` - Front matter batch script
- `/home/obsidian-blog/static/js/sidebar.js` - Sidebar collapse/expand
- `/home/obsidian-blog/static/js/article-graph.js` - Graph widget logic

**Modified Files:**
- `/home/obsidian-blog/hugo.toml` - Remove posts menu, change baseURL to HTTPS
- `/home/obsidian-blog/layouts/_default/single.html` - URL fixes, 3-column layout, CSS
- `/home/obsidian-blog/layouts/_default/graph.html` - Iterate all pages not just posts
- `/home/obsidian-blog/layouts/_default/baseof.html` - Include JavaScript files
- `/etc/nginx/conf.d/blog.conf` - Complete HTTPS configuration

**Deleted:**
- `/home/obsidian-blog/content/posts/` - Directory and all contents

## Rollback Procedures

**If front matter fails:** `git reset --hard HEAD`

**If HTTPS breaks:**
```bash
sudo cp /etc/nginx/conf.d/blog.conf.backup /etc/nginx/conf.d/blog.conf
sudo systemctl reload nginx
# Revert hugo.toml baseURL to http://
```

**If JavaScript breaks:** Comment out script includes in baseof.html

## Estimated Timeline

- Phase 1 (Front Matter): 30-45 min
- Phase 2 (Delete Posts): 10 min
- Phase 3 (URL Fixes): 20 min
- Phase 4 (Sidebar JS): 45 min
- Phase 5 (Graph Widget): 60 min
- Phase 6 (HTTPS): 30-45 min
- Phase 7 (Testing): 30 min

**Total: 3.5-4.5 hours**

## Success Criteria

1. All articles readable with proper titles and dates
2. All navigation links work correctly
3. Sidebar interactive with collapse/expand
4. Graph widget shows article connections
5. HTTPS enabled with valid certificate
6. Site loads at https://zzmblog.top/
7. No console errors or broken links
