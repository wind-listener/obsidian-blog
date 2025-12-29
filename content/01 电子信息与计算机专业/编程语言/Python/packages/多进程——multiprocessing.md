---
title: "多进程——multiprocessing"
date: 2025-08-07
draft: false
---

相关链接：
https://docs.python.org/zh-cn/3/library/multiprocessing.html
https://docs.python.org/3/library/multiprocessing.html

`multiprocessing`库是Python标准库的一部分，用于支持多进程并行执行。它提供了一种创建和管理进程的方式，与`threading`库类似，但它使用进程而不是线程，从而避免了GIL（全局解释器锁）限制。

### 基本概念

1. **进程（Process）**：独立的运行环境，每个进程都有自己的内存空间。
2. **全局解释器锁（GIL）**：Python解释器的一个机制，同一时刻只允许一个线程执行Python字节码。使用`multiprocessing`可以绕过GIL，从而实现真正的并行计算。

### 基本用法

1. **创建进程**：使用`multiprocessing.Process`类来创建一个新进程。

```python
import multiprocessing
import time

def worker(name):
    print(f"Worker {name} is starting")
    time.sleep(2)
    print(f"Worker {name} is done")

if __name__ == "__main__":
    process1 = multiprocessing.Process(target=worker, args=("A",))
    process2 = multiprocessing.Process(target=worker, args=("B",))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    print("Both processes are done")
```

2. **进程池（Pool）**：使用`multiprocessing.Pool`来管理多个进程池，以便轻松管理并行任务。

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as p:
        results = p.map(square, [1, 2, 3, 4, 5])
    print(results)
```

3. **队列（Queue）**：用于在进程之间传递消息。

```python
import multiprocessing

def producer(queue):
    for item in range(5):
        queue.put(item)
        print(f"Produced {item}")

def consumer(queue):
    while not queue.empty():
        item = queue.get()
        print(f"Consumed {item}")

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    producer_process = multiprocessing.Process(target=producer, args=(queue,))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue,))

    producer_process.start()
    producer_process.join()

    consumer_process.start()
    consumer_process.join()
```

4. **共享内存（Value 和 Array）**：用于在进程间共享数据。

```python
import multiprocessing

def increment(counter):
    for _ in range(1000):
        counter.value += 1

if __name__ == "__main__":
    counter = multiprocessing.Value('i', 0)
    processes = [multiprocessing.Process(target=increment, args=(counter,)) for _ in range(4)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print(f"Counter value: {counter.value}")
```

### 学习资源

1. **官方文档**：Python的官方`multiprocessing`文档是最权威的学习资源。 [multiprocessing — Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html)

2. **书籍**：推荐《Python Cookbook》第三版，作者David Beazley和Brian K. Jones，其中有关于`multiprocessing`的详细章节。

3. **视频教程**：
   - [Corey Schafer的Python Multiprocessing教程](https://www.youtube.com/watch?v=fKl2JW_qrso)
   - [Python Engineer的Python Multiprocessing基础教程](https://www.youtube.com/watch?v=3FrnJdXnLAE)

通过这些资源，你可以进一步加深对`multiprocessing`库的理解，并能在实际项目中应用它。