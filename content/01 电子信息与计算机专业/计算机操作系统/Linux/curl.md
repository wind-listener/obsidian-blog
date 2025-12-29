---
title: "curl 指令详解与总结"
date: 2025-12-09
draft: false
---

# curl 指令详解与总结
curl（全称 Client URL）是一款**跨平台、无界面的命令行工具**，核心用于在客户端与服务器之间传输数据，支持 HTTP/HTTPS/FTP/SCP/SFTP 等数十种协议，是开发者调试接口、自动化脚本、传输文件、排查网络问题的必备工具。


## 一、核心特性
1. **跨平台**：支持 Linux、macOS、Windows（PowerShell/CMD）、BSD 等主流系统；
2. **多协议**：覆盖 HTTP/HTTPS（核心）、FTP/SFTP、SMTP/POP3、LDAP 等；
3. **高度灵活**：可定制请求头、Cookie、认证方式、代理、超时等几乎所有网络请求参数；
4. **可编程**：可嵌入 Shell/Python/ShellScript 脚本，实现自动化数据传输；
5. **调试能力**：支持详细日志输出，便于排查接口/网络问题。


## 二、常用参数分类详解
curl 参数丰富，以下按「使用场景」分类，列出最常用的核心参数：

### 1. 请求方法相关（控制 HTTP 请求类型）
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-X/--request <METHOD>` | 指定 HTTP 请求方法（GET/POST/PUT/DELETE/PATCH/HEAD 等），默认 GET； |
| `-d/--data <DATA>`      | 发送 POST 数据（表单/JSON 格式），自动将请求方法设为 POST；          |
| `--data-urlencode <DATA>` | 对 POST 数据做 URL 编码（如空格转 %20、中文转 UTF-8 编码）；         |
| `-F/--form <KEY=VALUE>`  | 模拟 HTML 表单提交（`multipart/form-data` 格式），支持文件上传；    |

### 2. 请求头相关（定制 HTTP 请求头）
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-H/--header <HEADER>` | 自定义请求头（可多次使用添加多个头），如 `-H "Content-Type: application/json"`； |
| `-A/--user-agent <UA>` | 设置请求的 User-Agent（模拟浏览器/客户端），如 `-A "Mozilla/5.0 (MacOS) Chrome/120.0"`； |
| `-e/--referer <URL>`   | 设置 Referer 头（模拟从某个页面跳转而来）；                          |

### 3. Cookie 相关（管理 Cookie）
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-b/--cookie <COOKIES>` | 携带 Cookie（格式：`key1=value1;key2=value2` 或 Cookie 文件路径）；  |
| `-c/--cookie-jar <FILE>` | 将响应中的 Cookie 保存到指定文件（持久化 Cookie）；                  |

### 4. 输出控制（控制响应内容的展示/保存）
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-o/--output <FILE>`   | 将响应体保存到指定文件（自定义文件名），如 `-o response.txt`；       |
| `-O/--remote-name`     | 按服务器端的文件名保存响应（下载文件时常用）；                       |
| `-s/--silent`          | 静默模式（隐藏进度条、警告等非必要输出）；                           |
| `-S/--show-error`      | 静默模式下仅显示错误信息（搭配 `-s` 使用）；                         |
| `-i/--include`         | 显示响应头 + 响应体；                                                |
| `-I/--head`            | 仅发送 HEAD 请求，显示响应头（不返回响应体）；                       |
| `-v/--verbose`         | 详细调试模式（显示请求头、响应头、TCP 握手、SSL 协商等所有细节）；   |
| `-L/--location`        | 跟随服务器的 3xx 重定向（默认不跟随）；                              |

### 5. 文件传输相关
| 参数                        | 说明                                                                   |
| ------------------------- | -------------------------------------------------------------------- |
| `-T/--upload-file <FILE>` | 上传文件到服务器（如 FTP/HTTP PUT），如 `-T test.txt https://example.com/upload`； |

### 6. 代理与网络配置
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-x/--proxy <PROXY>`   | 设置代理（格式：`[protocol://]host:port`），如 `-x http://127.0.0.1:8080`； |
| `-U/--proxy-user <USER:PASS>` | 代理认证（用户名:密码）；                                            |
| `--connect-timeout <SEC>` | 设置连接超时时间（秒）；                                             |
| `--max-time <SEC>`     | 设置整个请求的最大耗时（秒）；                                       |

### 7. SSL/TLS 相关（HTTPS 场景）
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-k/--insecure`        | 忽略 HTTPS 证书验证（测试环境常用，生产禁用）；                      |
| `--cert <CERT_FILE>`   | 指定客户端 SSL 证书文件；                                            |
| `--key <KEY_FILE>`     | 指定客户端 SSL 私钥文件；                                            |

### 8. 认证相关
| 参数                | 说明                                                                 |
|---------------------|----------------------------------------------------------------------|
| `-u/--user <USER:PASS>` | HTTP 基本认证（用户名:密码），如 `-u admin:123456`；                 |


## 三、典型使用示例
以下示例覆盖 90% 的日常使用场景，结合参数讲解实际用法：

### 1. 基础 GET 请求（默认）
```bash
# 最简单的 GET 请求，直接返回响应体
curl https://api.example.com/user/1

# 显示响应头 + 响应体
curl -i https://api.example.com/user/1

# 仅显示响应头（HEAD 请求）
curl -I https://api.example.com/user/1
```

### 2. 带参数的 GET 请求
```bash
# 方式1：直接拼接在 URL 后（推荐）
curl https://api.example.com/search?keyword=curl&page=1

# 方式2：用 -G 配合 -d 生成 GET 参数（自动 URL 编码）
curl -G -d "keyword=curl 教程" -d "page=1" https://api.example.com/search
```

### 3. POST 表单数据（application/x-www-form-urlencoded）
```bash
curl -X POST -d "username=test&password=123456" https://api.example.com/login
# 等价于（省略 -X POST，-d 自动触发 POST）
curl -d "username=test&password=123456" https://api.example.com/login
```

### 4. POST JSON 数据（application/json）
```bash
# 必须指定 Content-Type 头，否则服务端无法解析
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"123456"}' \
  https://api.example.com/login
```

### 5. 模拟表单文件上传（multipart/form-data）
```bash
# 上传本地文件 avatar.png，表单字段名是 file
curl -F "file=@/Users/test/avatar.png" https://api.example.com/upload

# 同时上传文件 + 其他表单字段
curl -F "file=@/Users/test/avatar.png" -F "username=test" https://api.example.com/upload
```

### 6. 下载文件
```bash
# 自定义保存文件名
curl -o local_file.zip https://example.com/remote_file.zip

# 按服务器文件名保存（-O 大写）
curl -O https://example.com/remote_file.zip

# 断点续传（中断后继续下载）
curl -C - -O https://example.com/large_file.zip
```

### 7. 上传文件（PUT 方式）
```bash
# 上传本地文件到服务器
curl -T local_file.txt https://example.com/upload
```

### 8. 携带/保存 Cookie
```bash
# 携带 Cookie 请求
curl -b "session_id=123456;user_id=789" https://api.example.com/user

# 从文件读取 Cookie（文件格式为 Netscape Cookie 格式）
curl -b cookie.txt https://api.example.com/user

# 将响应的 Cookie 保存到文件
curl -c cookie.txt https://api.example.com/login
```

### 9. 调试模式（排查接口问题）
```bash
# 详细输出所有请求/响应细节（推荐）
curl -v https://api.example.com/user/1

# 更精简的调试（仅显示请求/响应头）
curl -i https://api.example.com/user/1
```

### 10. 忽略 HTTPS 证书（测试环境）
```bash
curl -k https://test-api.example.com/user/1
```

### 11. 使用代理访问
```bash
# HTTP 代理
curl -x http://127.0.0.1:8080 https://api.example.com

# SOCKS5 代理
curl -x socks5://127.0.0.1:1080 https://api.example.com

# 带认证的代理
curl -x http://127.0.0.1:8080 -U proxy_user:proxy_pass https://api.example.com
```

### 12. HTTP 基本认证
```bash
curl -u admin:123456 https://api.example.com/admin
```

### 13. RESTful API 操作（PUT/DELETE）
```bash
# PUT 更新资源
curl -X PUT -H "Content-Type: application/json" -d '{"name":"new_name"}' https://api.example.com/user/1

# DELETE 删除资源
curl -X DELETE https://api.example.com/user/1
```

### 14. 跟随重定向
```bash
# 自动跟随 3xx 重定向（如 HTTP 跳 HTTPS）
curl -L https://example.com
```


## 四、总结
### 1. 核心价值
curl 是「轻量、灵活、跨平台」的网络数据传输工具，无需图形界面，可快速完成接口调试、文件传输、网络排查，是命令行/脚本场景下的首选。

### 2. 核心适用场景
- 接口调试：快速验证 RESTful API/HTTP 接口的请求/响应是否符合预期；
- 自动化脚本：嵌入 Shell/Python 脚本，实现定时数据同步、文件上传下载；
- 网络排查：通过 `-v`/`-i` 调试证书、Cookie、请求头等问题；
- 文件传输：替代 FTP 客户端，命令行完成文件上传/下载。

### 3. 关键注意点
- POST JSON 时必须指定 `Content-Type: application/json`，否则服务端可能解析失败；
- 生产环境禁用 `-k`（忽略 HTTPS 证书），避免安全风险；
- 敏感信息（如密码）尽量避免直接写在命令行（可通过文件/环境变量传递）；
- 不同系统的 curl 版本略有差异，但核心参数（`-X/-d/-H/-v` 等）完全兼容。

### 4. 拓展技巧
- 结合管道 `|`：将 curl 响应传递给 jq（JSON 解析工具）格式化，如 `curl -s https://api.example.com/user/1 | jq`；
- 批量请求：通过 Shell 循环实现多接口批量测试，如 `for id in {1..10}; do curl https://api.example.com/user/$id; done`。

通过灵活组合参数，curl 可满足从简单 GET 请求到复杂文件上传、代理认证、SSL 双向认证等几乎所有网络数据传输需求。