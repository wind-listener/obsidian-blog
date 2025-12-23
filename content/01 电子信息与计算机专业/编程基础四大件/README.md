## Win系统同步
先启动 git bash
```bsah
# first
cd /c/WPSSync/Blogs/编程基础四大件
git add .
git commit -m "first"
git push -u origin main

# 日常
git add .
git commit -m "normal sync"
git push

# 版本发布
git push --tags
```