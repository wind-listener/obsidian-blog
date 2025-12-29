---
title: "pkill"
date: 2025-12-14
draft: false
---

`pkill` 是 Linux/Unix 系统中用于 **根据进程属性批量查找并终止进程** 的命令，是 `pgrep` 的配套工具（`pgrep` 查找进程，`pkill` 终止进程），核心优势是无需手动输入 PID，直接通过进程名、用户、端口等条件精准终止进程，效率远高于 `kill` 命令。


### 一、核心功能与工作原理
`pkill` 的核心逻辑：先按指定条件（如进程名、命令行、用户）搜索匹配的进程，再自动对这些进程执行 `kill` 操作（默认发送 `SIGTERM` 信号，即终止信号）。

- 区别于 `kill`：`kill` 需手动指定 PID（如 `kill 1234`），而 `pkill` 直接按“属性条件”匹配，适合批量终止进程。
- 区别于 `pstree`/`ps`：后两者仅用于查看进程，`pkill` 专注于“查找+终止”一体化。


### 二、基本语法
```bash
pkill [选项] [匹配条件]
```
- 选项：控制匹配规则、发送的信号等（关键部分）。
- 匹配条件：通常是进程名、命令行关键词、用户等（如 `run_mpi.sh`、`python`）。


### 三、常用选项与实战案例
#### 1. 基础用法：按进程名终止（最常用）
默认按“进程名前缀”匹配（无需完整进程名，匹配到前缀即终止）。
```bash
# 终止所有进程名包含 "run_mpi" 的进程（如 run_mpi.sh、run_mpi_main）
pkill run_mpi

# 终止所有 Python 进程（进程名通常是 python 或 python3）
pkill python3
```
- 注意：默认匹配“前缀”，若需精准匹配完整进程名，需加 `-x` 选项：
  ```bash
  # 仅终止进程名完全等于 "run_mpi.sh" 的进程（不匹配 run_mpi.sh.bak 等）
  pkill -x run_mpi.sh
  ```

#### 2. 按命令行关键词匹配（核心！解决“进程名相同但命令不同”场景）
用 `-f` 选项，匹配“完整命令行”（包括启动参数、脚本路径等），而非仅匹配进程名。  
**这是你之前杀死 `run_mpi.sh` 进程的关键用法**！
```bash
# 终止所有命令行中包含 "run_mpi.sh" 的进程（不管进程名是什么）
pkill -f run_mpi.sh

# 精准终止：命令行包含 "flux_fill_scene_tags" 和 "test3_dino_filtered_2" 的进程
pkill -f "flux_fill_scene_tags.*test3_dino_filtered_2"
```
- 场景举例：若有两个 `python` 进程（一个是 `python train.py`，一个是 `python test.py`），用 `-f` 可只终止 `train.py`：
  ```bash
  pkill -f "python train.py"
  ```

#### 3. 按用户终止进程（多用户环境常用）
用 `-u` 选项，仅终止指定用户的进程，避免误杀其他用户的进程。
```bash
# 终止用户 root 所有的 Python 进程
pkill -u root python3

# 终止用户 zhangsan 所有包含 "mpi" 的进程
pkill -u zhangsan -f mpi
```

#### 4. 强制终止进程（避免“假死”）
默认 `pkill` 发送 `SIGTERM` 信号（信号编号 15），进程可能会执行清理操作后退出；若进程卡住、无法正常退出，需发送 `SIGKILL` 信号（信号编号 9，强制终止，无清理过程），用 `-9` 或 `-SIGKILL` 指定。
```bash
# 强制终止所有 run_mpi.sh 进程（等价于 kill -9 PID）
pkill -9 -f run_mpi.sh

# 等价写法（用信号名）
pkill -SIGKILL -f run_mpi.sh
```
- 注意：`-9` 是“终极手段”，尽量先尝试默认信号（无 `-9`），避免数据丢失或资源泄露。

#### 5. 预览匹配结果（不实际终止，安全验证）
用 `-l` 选项（list），仅列出匹配的进程名和 PID，不执行终止操作，适合先验证匹配是否准确。
```bash
# 预览：哪些进程会被 pkill -f run_mpi.sh 终止（仅显示，不杀死）
pkill -l -f run_mpi.sh
```
- 输出示例（PID + 进程名）：
  ```
  106 run_mpi.sh
  119 run_mpi.sh
  ```

#### 6. 忽略大小写匹配
用 `-i` 选项，匹配时不区分大小写。
```bash
# 终止进程名包含 "Run_MPI"、"run_mpi"、"RUN_MPI" 的所有进程
pkill -i run_mpi
```

#### 7. 按进程组终止
用 `-g` 选项，终止指定进程组的所有进程（进程组 ID 通常等于父进程 PID）。
```bash
# 终止进程组 ID 为 106 的所有进程（父进程 106 及其子进程）
pkill -g 106
```


### 四、常见信号说明（补充）
`pkill` 可通过 `-s` 或直接加信号编号指定发送的信号，常用信号：

| 信号编号 | 信号名 | 作用 |
|----------|--------|------|
| 15       | SIGTERM | 默认信号，进程正常退出（可执行清理） |
| 9        | SIGKILL | 强制终止，进程无法拒绝（无清理） |
| 2        | SIGINT  | 等价于 Ctrl+C，中断进程 |
| 1        | SIGHUP  | 重启进程（部分服务可用，如 Nginx） |

示例：发送 `SIGHUP` 信号重启 Nginx 进程
```bash
pkill -SIGHUP nginx
```


### 五、注意事项（避坑关键）
1. **避免误杀系统进程**：  
   匹配条件要精准，尤其是用 `pkill python3` 这类宽泛条件时，可能会终止系统依赖的 Python 进程（如运维脚本）。建议先加 `-l` 预览匹配结果，再执行终止。
   
2. **`-f` 选项的优先级**：  
   当进程名和命令行关键词冲突时，`-f` 优先匹配命令行，例如：`pkill -f "python test.py"` 会忽略进程名，只看命令行是否包含该字符串。

3. **无输出≠执行成功**：  
   `pkill` 执行后默认无输出，若要确认是否终止成功，可结合 `pgrep` 验证：
   ```bash
   # 终止后检查是否还有残留
   pgrep -f run_mpi.sh  # 无输出则说明已终止
   ```

4. **权限问题**：  
   普通用户只能终止自己的进程，终止其他用户（如 root）的进程需加 `sudo`：
   ```bash
   sudo pkill -u root -f run_mpi.sh
   ```


### 六、总结：`pkill` 核心优势与适用场景
| 优势 | 适用场景 |
|------|----------|
| 无需手动查 PID | 批量终止同类进程（如所有 Python 进程、所有 MPI 进程） |
| 按命令行精准匹配 | 区分同名进程（如不同参数的 `run_mpi.sh`） |
| 支持多条件组合 | 复杂场景（如“root 用户的、命令行含 mpi 的、强制终止”） |

结合你之前的需求，`pkill -9 -f run_mpi.sh` 是最精准、高效的终止命令，既匹配了脚本名，又强制终止了卡住的进程，避免资源冲突。