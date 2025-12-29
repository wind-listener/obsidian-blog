---
title: "PyTorch TensorFlow内置计时​"
date: 2025-09-12
draft: false
---


深度学习框架的轻量级计时方法：
关键是需要使用 torch.cuda.synchronize()  # 同步等待


```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 执行模型或操作
end.record()
torch.cuda.synchronize()  # 同步等待
print(f"耗时: {start.elapsed_time(end)} ms")  # 输出时间
```
​注意事项​：
- 避免在计时中包含数据生成（如 torch.rand），否则会统计内存分配时间。
- 预热GPU缓存（运行空操作多次）以减少首次执行误差。