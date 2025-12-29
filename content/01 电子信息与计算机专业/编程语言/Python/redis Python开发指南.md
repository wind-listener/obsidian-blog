---
title: "redis Python开发指南"
date: 2025-12-14
draft: false
---

Redis 凭借其高性能和丰富的数据类型，在缓存、会话存储和消息队列等场景中应用广泛。通过 Python 的 `redis-py` 库，我们可以方便地操作 Redis。下面为你梳理关键要点和用法。

### 📦 安装与连接

首先需要安装 `redis-py` 库，并确保 Redis 服务器已运行。
```bash
pip install redis
```

在 Python 中建立连接时，推荐使用**连接池**以提升性能。
```python
import redis

# 创建连接池（设置最大连接数，例如20）
pool = redis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=20, decode_responses=True)
client = redis.Redis(connection_pool=pool)

# 测试连接
try:
    if client.ping():
        print("✅ Redis连接成功")
except redis.ConnectionError:
    print("❌ Redis连接失败")
```
**参数说明**：
- `decode_responses=True`：自动将返回的字节数据解码为字符串。
- `max_connections=20`：设置连接池的最大连接数。

### 🔧 核心数据类型操作

Redis 支持多种数据结构，下表汇总了它们的常见操作和典型应用场景，方便你快速查阅。

| 数据结构 | 典型应用场景 | 关键写入命令示例 | 关键读取命令示例 |
| :--- | :--- | :--- | :--- |
| **String（字符串）** | 缓存会话、计数器、分布式锁 | `set(key, value)`<br>`setex(key, 秒数, value)`<br>`mset({k1:v1, k2:v2})`<br>`incr(key)` | `get(key)`<br>`mget([key1, key2])` |
| **Hash（哈希）** | 存储用户信息、商品详情等对象 | `hset(name, key, value)`<br>`hmset(name, {k1:v1, k2:v2})` | `hget(name, key)`<br>`hgetall(name)`<br>`hkeys(name)` |
| **List（列表）** | 消息队列、最新列表 | `lpush(key, *values)`<br>`rpush(key, *values)` | `lrange(key, 0, -1)`<br>`lpop(key)`<br>`rpop(key)` |
| **Set（集合）** | 标签系统、共同好友 | `sadd(key, *members)` | `smembers(key)`<br>`sismember(key, member)`<br>`sinter(key1, key2)` |
| **Sorted Set（有序集合）** | 排行榜、优先级队列 | `zadd(key, {member1:score1, member2:score2})` | `zrange(key, 0, -1, withscores=True)`<br>`zrevrange(...)` |

以下是针对上述数据结构的常用 Python 代码示例。

**1. String（字符串）**
```python
# 设置和获取值
client.set('username', '张三')
client.setex('verify_code', 300, '654321')  # 5分钟后过期
value = client.get('username')  # 返回 '张三' (因设置了decode_responses=True)

# 数值操作
client.set('page_views', 100)
client.incr('page_views')  # 自增1 → 101
```

**2. Hash（哈希）**
```python
# 设置和获取哈希字段
client.hset('user:1001', 'name', '李四')
client.hset('user:1001', mapping={'age': 30, 'city': '深圳'})  # 批量设置
user_info = client.hgetall('user:1001')  # 获取所有字段和值，返回字典
```

**3. List（列表）**
```python
# 列表操作
client.lpush('tasks', 'task1', 'task2')  # 左侧插入
client.rpush('tasks', 'task3')  # 右侧插入
tasks = client.lrange('tasks', 0, -1)  # 获取所有元素
```

**4. Set（集合）**
```python
# 集合操作
client.sadd('tags', 'Python', 'Redis')
tags = client.smembers('tags')  # 获取所有元素
common_tags = client.sinter('tags1', 'tags2')  # 求交集
```

**5. Sorted Set（有序集合）**
```python
# 有序集合操作 (排行榜示例)
client.zadd('leaderboard', {'张三': 95, '李四': 88, '王五': 92})
# 获取分数最高的前两名
top2 = client.zrevrange('leaderboard', 0, 1, withscores=True)
```

### ⚙️ 高级功能

1.  **事务处理**
    使用 `pipeline` 确保多个命令的原子性执行。
    ```python
    try:
        with client.pipeline() as pipe:
            pipe.multi()  # 开启事务
            pipe.set('key1', 'val1')
            pipe.hset('user:1002', 'name', 'wangwu')
            results = pipe.execute()  # 执行事务
            print(f"事务执行成功: {results}")
    except redis.RedisError as e:
        print(f"事务执行失败: {e}")
    ```

2.  **发布订阅 (Pub/Sub)**
    可用于实现简单的消息队列。
    ```python
    # 发布者
    client.publish('news_channel', 'Hello, World!')

    # 订阅者 (需要在另一个线程或进程中运行)
    pubsub = client.pubsub()
    pubsub.subscribe('news_channel')
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"收到消息: {message['data']}")
    ```

3.  **过期时间**
    可以设置键的生存时间，到期后自动删除。
    ```python
    client.setex('temp_data', 60, '临时数据')  # 60秒后过期
    ttl = client.ttl('temp_data')  # 查看剩余生存时间
    ```

### 💡 性能优化与最佳实践

- **使用连接池**：避免频繁建立和关闭连接的开销。
- **批量操作**：优先使用 `mset`、`mget` 等批量命令，或使用 `pipeline` 打包多个命令，减少网络往返次数。
- **选择合适的数据类型**：根据场景选择最有效的数据结构。例如，存储对象使用 Hash 通常比多个 String 更节省内存。
- **异常处理**：重要的 Redis 操作应放在 `try-except` 块中。
    ```python
    from redis.exceptions import ConnectionError, TimeoutError

    try:
        client.set('key', 'value')
    except ConnectionError as e:
        print(f"Redis连接错误: {e}")
    except TimeoutError as e:
        print(f"Redis请求超时: {e}")
    except Exception as e:
        print(f"其他错误: {e}")
    ```

### 💎 简单应用示例

1.  **缓存函数结果**
    ```python
    def get_expensive_data(key):
        # 先尝试从Redis缓存获取
        cached_data = client.get(key)
        if cached_data:
            return cached_data
        else:
            # 如果缓存中没有，则从数据库或其他耗时操作中获取
            data = ...  # 这里是耗时的数据获取逻辑
            client.setex(key, 3600, data)  # 存入缓存，设置1小时过期
            return data
    ```

2.  **简单分布式锁**
    ```python
    # 获取锁，设置过期时间防止死锁
    lock_acquired = client.set('my_lock', 'locked', nx=True, ex=10)
    if lock_acquired:
        try:
            # 执行需要加锁的临界区代码
            print("锁已获取，执行关键操作...")
        finally:
            # 释放锁
            client.delete('my_lock')
    else:
        print("获取锁失败，其他进程正在操作。")
    ```

希望这份详细的总结能帮助你在 Python 项目中高效地使用 Redis。如果你有特定的应用场景想深入了解，我们可以继续探讨。