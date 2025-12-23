gitignore文件可以定义git不要跟踪的文件或者路径。



注意事项：

要在 Git 中忽略当前相对路径下的 `config/` 目录，同时保留工程中其他位置的 `config/` 目录的跟踪，您可以使用以下 `.gitignore` 配置：

#### 1. **项目根目录下的 `.gitignore` 文件**
```gitignore
# 忽略当前目录下的 config 文件夹
/config/
```

#### 2. **如果要忽略特定子目录下的 config**
```gitignore
# 忽略特定路径下的 config 文件夹
/path/to/specific/directory/config/
```

#### 3. **例外情况处理**
如果需要保留某些特定的 `config/` 目录，可以使用 `!` 来取消忽略：

```gitignore
# 忽略所有 config 目录
*/config/

# 但保留某些特定的 config 目录
!/path/to/important/config/
!/another/path/config/
```

#### 4. **验证配置**
```bash
# 检查哪些文件会被忽略
git status --ignored
```

### 注意事项
1. **路径特异性**：确保路径前缀足够具体以避免误匹配
2. **斜杠用法**：
   - `/config/` 只匹配根目录下的 config
   - `config/` 匹配所有层级的 config 目录
3. **已跟踪文件**：如果 `config/` 已被跟踪，需要先删除缓存：
   ```bash
   git rm -r --cached config/
   git add .
   git commit -m "停止跟踪 config 目录"
   ```

### 推荐方案
在您的情况下，如果只想忽略当前相对路径下的 `config/`，最佳做法是：
1. 在需要忽略的目录同级创建 `.gitignore`
2. 内容为：
   ```gitignore
   config/
   ```
3. 这样只会忽略该目录下的 `config/`，不会影响其他位置

这种配置方式既满足了您的需求，又保持了 Git 管理的清晰性。