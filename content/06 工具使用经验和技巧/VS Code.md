# 清理ssh host 缓存


```bash
# MAC电脑上，一般存放路径如下：
ls -la ~/Library/Application\ Support/Code/User/globalStorage/ms-vscode-remote.remote-ssh/

# 删除 VS Code 的 SSH 缓存目录（这会强制重建所有缓存）
rm -rf ~/Library/Application\ Support/Code/User/globalStorage/ms-vscode-remote.remote-ssh/

# 同时也可以清理 VS Code 的通用缓存
rm -rf ~/Library/Application\ Support/Code/Cache/
rm -rf ~/Library/Application\ Support/Code/CachedData/
```


```json

'num'
{

// Use IntelliSense to learn about possible attributes.

// Hover to view descriptions of existing attributes.

// For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387

"version": "0.2.0",

"configurations": [

{

"name": "Python Debugger: Current File",

"type": "debugpy",

"request": "launch",

"program": "${file}",

"console": "integratedTerminal",

"args": [

"--model", "glm-4-private-FC",

"--test-category", "rest"

]

},

{

"name": "Python Debugger:executable_parallel_multiple_function",

"type": "debugpy",

"request": "launch",

"program": "${file}",

"console": "integratedTerminal",

"args": [

"--model", "ipo:glm-4-air-biz",

"--test-category", "executable_parallel_multiple_function"

]

}

]

}
```

# 快捷键
代码折叠相关的快捷键：

- 折叠当前代码块：`Cmd + Shift + [`
- 展开当前代码块：`Cmd + Shift + ]`
- 折叠所有子代码块：`Cmd + K Cmd + [`
- 展开所有子代码块：`Cmd + K Cmd + ]`
- 折叠所有代码块：`Cmd + K Cmd + 0`
- 展开所有代码块：`Cmd + K Cmd + J`