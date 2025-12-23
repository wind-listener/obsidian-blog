#Linux #常用指令 
## 解决“远程主机密钥已更改，端口转发已禁用”
### 使用命令行清除特定主机的已知密钥

你也可以使用以下命令从 `known_hosts` 文件中删除特定主机的条目：
```bash
ssh-keygen -R <远程主机IP或主机名>
ssh-keygen -R 172.19.128.119
```

# 精简版SSH公钥添加指令
```bash
# 1. 确保目录存在并设置权限
mkdir -p ~/.ssh && chmod 700 ~/.ssh

# 2. 追加公钥到授权文件（替换YOUR_PUBLIC_KEY）
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys

# 3. 设置严格权限
chmod 600 ~/.ssh/authorized_keys
```

