---
title: "torchrun"
date: 2025-08-07
draft: false
---

`torchrun` 是 PyTorch 提供的一个用于分布式训练的命令行工具，它支持多种分布式训练方式，包括单机多卡和多机多卡训练。以下是其主要用法和参数介绍：


### **基本用法**
```bash
torchrun [options] your_script.py [args]
```

### **常用参数**
1. **节点配置**
   - `--nnodes=N`：指定总节点数（默认为 1）。
   - `--nproc_per_node=G`：每个节点的进程数（通常等于 GPU 数量）。
   - `--node_rank=R`：当前节点的排名（从 0 开始）。

2. **主节点设置**
   - `--master_addr=HOST`：主节点地址（默认为 `127.0.0.1`）。
   - `--master_port=PORT`：主节点端口（默认为 `29500`）。

3. **其他参数**
   - `--rdzv_backend`：启动方法（如 `c10d`、`etcd` 等）。
   - `--max_restarts`：自动重启次数（用于容错）。
   - `--standalone`：单机模式，简化配置。


### **示例**
#### 1. **单机多卡训练**
在 4 个 GPU 上运行：
```bash
torchrun --standalone --nproc_per_node=4 your_script.py --args
```

#### 2. **多机多卡训练**
假设有 2 个节点，每个节点 4 个 GPU：

**节点 0**：
```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="node0_ip" --master_port=12345 your_script.py --args
```

**节点 1**：
```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr="node0_ip" --master_port=12345 your_script.py --args
```


### **脚本内配置**
在 `your_script.py` 中，需要初始化分布式环境：
```python
import torch
import torch.distributed as dist

def main():
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 创建分布式数据加载器和模型
    # ...

if __name__ == "__main__":
    main()
```


### **注意事项**
- 使用 `--standalone` 时无需设置 `master_addr` 和 `node_rank`。
- 确保所有节点间网络互通，主节点端口可访问。
- 脚本需兼容分布式训练（如使用 `DistributedDataLoader`）。

更多详细参数可通过 `torchrun --help` 查看。