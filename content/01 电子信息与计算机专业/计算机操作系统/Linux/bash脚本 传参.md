Bash 脚本接受参数的写法主要分 **3 种场景**（从简单到灵活），结合实际需求（如传递普通参数、带选项的参数、可选参数）逐步讲解，每个场景都附完整示例，方便直接套用。


## 一、基础场景：位置参数（最简单，无选项）
适合参数数量固定、顺序明确的场景（如直接传递 1-2 个路径/数值），通过 `$1`、`$2`... 直接获取参数（`$0` 是脚本本身名称）。

### 语法规则
| 变量   | 含义                     |
|--------|--------------------------|
| `$0`   | 脚本文件名（如 `test.sh`）|
| `$1`   | 第 1 个参数              |
| `$2`   | 第 2 个参数              |
| `$#`   | 参数总个数               |
| `$*`   | 所有参数（作为单个字符串）|
| `$@`   | 所有参数（作为独立字符串）|
| `$?`   | 上一条命令的执行结果（0=成功）|

### 示例脚本（`pos_param.sh`）
```bash
#!/bin/bash
# 位置参数示例：传递 输入路径、进程数

# 检查参数数量（可选，避免参数缺失）
if [ $# -ne 2 ]; then
    echo "用法：bash $0 <输入路径> <进程数>"
    echo "示例：bash $0 ./test_data 2"
    exit 1  # 退出脚本，状态码 1 表示错误
fi

# 读取参数
input_path=$1  # 第 1 个参数：输入路径
mlp_gpu=$2     # 第 2 个参数：进程数

# 执行逻辑
echo "开始处理任务..."
echo "输入路径：$input_path"
echo "进程数：$mlp_gpu"
# 后续可执行命令（如 python main.py --input $input_path --mlp-gpu $mlp_gpu）
```

### 运行方式
```bash
# 正确用法（2 个参数，顺序固定）
bash pos_param.sh ./test_data 2

# 错误用法（参数数量不对）
bash pos_param.sh ./test_data  # 会提示用法并退出
```


## 二、常用场景：带选项的参数（最灵活，推荐）
适合参数数量不固定、需要指定选项的场景（如 `--input`、`--gpu`），通过 `while` 循环解析参数，支持任意顺序、可选参数，和你之前改造的脚本逻辑一致。

### 核心语法
用 `while [[ $# -gt 0 ]]` 循环遍历所有参数，通过 `case` 匹配选项：
- 带值的选项（如 `--input ./data`）：匹配选项后，用 `$2` 获取值，再 `shift 2` 跳过这两个参数；
- 开关选项（如 `--clear-queue`）：无需值，匹配后 `shift 1` 跳过当前参数；
- 其他参数（如额外的 `--batch-size 32`）：直接收集，透传给后续命令。

### 示例脚本（`opt_param.sh`）
```bash
#!/bin/bash
# 带选项的参数示例：支持 --input、--mlp-gpu、--redis-host 等，可任意顺序

# 1. 设置默认值（可选，避免参数缺失时无默认值）
input_path="./default_data"  # 默认输入路径
mlp_gpu=1                    # 默认进程数
redis_host="10.178.148.233"  # 默认 Redis 地址
clear_queue=0                # 开关选项：0=不清除队列（默认），1=清除

# 2. 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        # 带值的选项：--input 后接路径
        --input)
            input_path="$2"  # 获取选项后的值（$2）
            shift 2          # 跳过 $1（--input）和 $2（路径）
            ;;
        # 带值的选项：--mlp-gpu 后接数字
        --mlp-gpu)
            mlp_gpu="$2"
            shift 2
            ;;
        # 带值的选项：--redis-host 后接地址
        --redis-host)
            redis_host="$2"
            shift 2
            ;;
        # 开关选项：--clear-queue 无需值，触发时设为 1
        --clear-queue)
            clear_queue=1
            shift 1  # 仅跳过 $1（--clear-queue）
            ;;
        # 帮助选项：--help 显示用法
        --help)
            echo "用法：bash $0 [选项]"
            echo "选项："
            echo "  --input <路径>        输入待处理路径（默认：$input_path）"
            echo "  --mlp-gpu <数字>      进程数（默认：$mlp_gpu）"
            echo "  --redis-host <地址>   Redis 地址（默认：$redis_host）"
            echo "  --clear-queue         清除队列（默认：不清除）"
            echo "  --help                显示帮助"
            exit 0
            ;;
        # 未知选项：提示错误并退出
        *)
            echo "错误：未知选项 $1"
            echo "使用 --help 查看支持的选项"
            exit 1
            ;;
    esac
done

# 3. 执行逻辑
echo "========================================"
echo "任务配置："
echo "输入路径：$input_path"
echo "进程数：$mlp_gpu"
echo "Redis 地址：$redis_host"
echo "是否清除队列：$(if [ $clear_queue -eq 1 ]; then echo "是"; else echo "否"; fi)"
echo "========================================"

# 后续命令（如执行 Python 脚本）
python main.py --input "$input_path" --redis-host "$redis_host" \
               $(if [ $clear_queue -eq 1 ]; then echo "--clear-queue"; fi)
```

### 运行方式（灵活组合）
```bash
# 1. 使用默认配置（所有参数用默认值）
bash opt_param.sh

# 2. 指定输入路径和进程数（任意顺序）
bash opt_param.sh --input ./test_data --mlp-gpu 3
bash opt_param.sh --mlp-gpu 3 --input ./test_data  # 顺序不影响

# 3. 同时指定 Redis 地址和清除队列
bash opt_param.sh --redis-host 10.178.148.234 --clear-queue

# 4. 查看帮助
bash opt_param.sh --help
```


## 三、进阶场景：用 `getopts` 解析短选项（简洁）
适合需要短选项（如 `-i` 代替 `--input`、`-g` 代替 `--mlp-gpu`）的场景，语法更简洁，但仅支持单字符选项（不支持长选项如 `--input`）。

### 核心语法
- `getopts "选项字符串" 变量名`：循环解析短选项；
- 选项字符串中：带 `:` 表示该选项需要值（如 `i:g:r:` 表示 `-i`、`-g`、`-r` 需带值）；
- `OPTARG`：存储选项对应的值；
- `OPTIND`：下一个要解析的参数索引。

### 示例脚本（`short_opt.sh`）
```bash
#!/bin/bash
# 短选项参数示例：-i（输入路径）、-g（进程数）、-r（Redis地址）、-c（清除队列）

# 1. 设置默认值
input_path="./default_data"
mlp_gpu=1
redis_host="10.178.148.233"
clear_queue=0

# 2. 解析短选项（i:g:r:c 表示 -i/-g/-r 带值，-c 是开关）
while getopts "i:g:r:ch" opt; do
    case "$opt" in
        i) input_path="$OPTARG" ;;  # -i 后接输入路径
        g) mlp_gpu="$OPTARG" ;;     # -g 后接进程数
        r) redis_host="$OPTARG" ;;  # -r 后接 Redis 地址
        c) clear_queue=1 ;;         # -c 触发清除队列
        h)  # -h 显示帮助
            echo "用法：bash $0 [-i 输入路径] [-g 进程数] [-r Redis地址] [-c] [-h]"
            echo "选项："
            echo "  -i <路径>   输入待处理路径（默认：$input_path）"
            echo "  -g <数字>   进程数（默认：$mlp_gpu）"
            echo "  -r <地址>   Redis 地址（默认：$redis_host）"
            echo "  -c          清除队列（默认：不清除）"
            echo "  -h          显示帮助"
            exit 0
            ;;
        \?)  # 未知选项
            echo "错误：未知选项 -$OPTARG"
            echo "使用 -h 查看支持的选项"
            exit 1
            ;;
        :)  # 选项缺少值
            echo "错误：选项 -$OPTARG 需要指定值"
            exit 1
            ;;
    esac
done

# 3. 执行逻辑（和之前一致）
echo "输入路径：$input_path"
echo "进程数：$mlp_gpu"
echo "Redis 地址：$redis_host"
echo "是否清除队列：$( [ $clear_queue -eq 1 ] && echo "是" || echo "否" )"
```

### 运行方式
```bash
# 1. 短选项组合使用
bash short_opt.sh -i ./test_data -g 2 -r 10.178.148.234 -c

# 2. 单个选项
bash short_opt.sh -i ./new_data

# 3. 查看帮助
bash short_opt.sh -h
```


## 四、3 种写法的对比与选择建议
| 写法类型       | 优点                  | 缺点                  | 适用场景                          |
|----------------|-----------------------|-----------------------|-----------------------------------|
| 位置参数       | 语法最简单，无需解析  | 顺序固定、无选项标识  | 参数少（1-2个）、顺序明确的场景    |
| 带选项的参数（while+case） | 支持长选项、任意顺序、可选参数 | 代码稍长              | 大多数场景（推荐，如你的 Python 脚本参数传递） |
| getopts 短选项 | 语法简洁、支持短选项  | 不支持长选项          | 需要短选项（如 `-i` `-g`）的场景  |


## 五、实用技巧（避坑指南）
1. **参数带空格的处理**：若参数包含空格（如路径 `./my data`），传递时用引号包裹（`"../my data"`），脚本中变量也用引号（`"$input_path"`），避免被拆分成多个参数；
2. **默认值兜底**：关键参数设置默认值，避免用户未传递时脚本报错；
3. **参数校验**：对数字参数（如进程数）校验是否为正整数，路径参数校验是否存在：
   ```bash
   # 校验进程数是否为正整数
   if ! [[ $mlp_gpu =~ ^[1-9][0-9]*$ ]]; then
       echo "错误：进程数必须是正整数"
       exit 1
   fi
   # 校验输入路径是否存在
   if [ ! -d "$input_path" ]; then
       echo "错误：输入路径 $input_path 不存在"
       exit 1
   fi
   ```
4. **透传额外参数**：若需要将脚本接收的未定义参数透传给其他命令（如 Python 脚本），用 `$@` 收集：
   ```bash
   # 脚本中收集额外参数
   extra_args=()
   while [[ $# -gt 0 ]]; do
       case "$1" in
           --input) input_path="$2"; shift 2 ;;
           --mlp-gpu) mlp_gpu="$2"; shift 2 ;;
           *) extra_args+=("$1"); shift 1 ;;  # 收集额外参数
       esac
   done
   # 透传给 Python 脚本
   python main.py --input "$input_path" --mlp-gpu "$mlp_gpu" "${extra_args[@]}"
   ```


## 总结
- 简单场景用「位置参数」，复杂场景用「带选项的参数（while+case）」；
- 你的需求（传递 Python 脚本的多个参数+进程数）最适合用「带选项的参数」写法，支持长选项、灵活组合，和你之前改造的脚本逻辑一致；
- 记住核心原则：参数解析后先校验，再执行逻辑，避免因无效参数导致脚本异常。