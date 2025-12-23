

这样写装饰器无法使用？
```python
@track_cuda_memory_on_device(self.device.split(':')[-1])
    def calculate(self, video_tensor, video_binary: bytes, *args, **kwargs):
        """主计算方法"""
        start_time = time.time()
        T, H, W, C = video_tensor.shape
        video_segments = {}
        
```