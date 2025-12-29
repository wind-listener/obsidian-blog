#!/usr/bin/env python3
"""
Add Hugo front matter to markdown files that are missing it.
This script adds title, date, and draft status to MD files.
"""

import os
import re
from datetime import datetime
from pathlib import Path


def has_frontmatter(content):
    """Check if file already has YAML front matter"""
    return content.strip().startswith('---')


def get_file_date(filepath):
    """Get file modification time as Hugo date format"""
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')


def extract_title_from_filename(filename):
    """Convert filename to title, removing .md extension"""
    title = filename.replace('.md', '')
    # Clean up common patterns
    title = title.strip()
    return title


def extract_title_from_content(content):
    """Try to extract title from first H1 heading"""
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        if line.startswith('# '):
            return line[2:].strip()
    return None


def add_frontmatter(filepath, dry_run=False):
    """Add front matter to a markdown file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has front matter
        if has_frontmatter(content):
            return 'skipped', 'Already has front matter'

        # Extract title (prefer content H1, fallback to filename)
        title = extract_title_from_content(content)
        if not title:
            title = extract_title_from_filename(os.path.basename(filepath))

        # Get date from file modification time
        date = get_file_date(filepath)

        # Build front matter
        frontmatter = f"""---
title: "{title}"
date: {date}
draft: false
---

"""

        # Combine front matter with content
        new_content = frontmatter + content

        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return 'success', f'Added: title="{title}", date={date}'
        else:
            return 'dry_run', f'Would add: title="{title}", date={date}'

    except Exception as e:
        return 'error', str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Add front matter to Hugo markdown files')
    parser.add_argument('--path', default='/home/obsidian-blog/content',
                        help='Content directory path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--exclude-posts', action='store_true', default=True,
                        help='Exclude posts directory')

    args = parser.parse_args()

    # Find all markdown files
    content_path = Path(args.path)
    md_files = []

    for md_file in content_path.rglob('*.md'):
        # Skip posts directory if requested
        if args.exclude_posts and 'posts' in md_file.parts:
            continue
        # Skip attachments directories
        if 'attachments' in md_file.parts:
            continue
        # Skip hidden files and .obsidian folders
        if any(part.startswith('.') for part in md_file.parts):
            continue
        # Skip special files
        if md_file.name in ['graph.md', 'search.md']:
            continue
        md_files.append(md_file)

    print(f"Found {len(md_files)} markdown files")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    # Process files
    results = {'success': 0, 'skipped': 0, 'error': 0, 'dry_run': 0}

    for filepath in sorted(md_files):
        status, message = add_frontmatter(str(filepath), args.dry_run)
        results[status] += 1

        if status in ['success', 'error', 'dry_run']:
            print(f"[{status.upper()}] {filepath}")
            print(f"  {message}\n")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']} (already have front matter)")
    print(f"  Errors:  {results['error']}")
    if args.dry_run:
        print(f"  Dry Run: {results['dry_run']}")
    print("="*60)


if __name__ == '__main__':
    main()
