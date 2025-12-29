---
title: "redis常用指南"
date: 2025-12-14
draft: false
---

Redis 因其丰富的数据类型和极高的性能，成为了非常流行的内存数据存储系统。下面我为你梳理了其核心数据结构和常用命令，以及一些典型应用场景，方便你快速上手和查阅。

### 📂 Redis 核心数据结构与常用命令速览

首先，通过下表你可以快速了解 Redis 的五种核心数据类型及其特性和典型应用场景。

| 数据结构 | 主要特性 | 典型应用场景 | 关键命令示例 |
| :--- | :--- | :--- | :--- |
| **String（字符串）** | 可存储文本、数字或二进制数据 | **缓存**：存储热点数据，如会话信息。<br>**计数器**：文章阅读量，点赞数。<br>**分布式锁**：通过 `SETNX` 实现。 | `SET`, `GET`, `INCR`, `DECR`, `SETNX` |
| **Hash（哈希）** | 适合存储对象，可表示多个字段的映射 | **用户信息**：将用户的所有属性存储在一个哈希中。<br>**商品详情**：存储商品的多个详细信息。 | `HSET`, `HGET`, `HGETALL`, `HMSET` |
| **List（列表）** | 按插入顺序排序的字符串元素集合，可重复 | **消息队列**：使用 `LPUSH` 和 `BRPOP` 实现简单的队列。<br>**最新列表**：存储最新的用户动态、新闻等。 | `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `LRANGE` |
| **Set（集合）** | 无序的字符串集合，元素唯一 | **标签系统**：为内容打标签。<br>**共同好友**：计算用户间的共同关注。<br>**抽奖**：利用 `SRANDMEMBER` 随机取元素。 | `SADD`, `SREM`, `SINTER`, `SUNION`, `SISMEMBER` |
| **Sorted Set（有序集合）** | 成员关联分数（score），按分数排序 | **排行榜**：实现游戏积分榜、商品销量榜等。<br>**带优先级的队列**：任务调度。 | `ZADD`, `ZRANGE`, `ZREVRANGE`, `ZRANK` |

### ⌨️ 各类型常用命令详解

以下是每种数据类型更详细的命令说明和示例。

#### 1. String（字符串）
字符串是 Redis 最基本的数据类型。
-   **设置与获取**
    ```bash
    SET username "john_doe"  # 设置键 username 的值为 "john_doe"
    GET username             # 获取键 username 的值
    ```
-   **数值操作**：当字符串值是数字时，可以进行增减。
    ```bash
    SET page_views 100
    INCR page_views         # 将 page_views 的值增加 1，变为 101
    DECRBY page_views 10    # 将 page_views 的值减少 10，结果为 91
    ```
-   **批量操作与条件设置**
    ```bash
    MSET key1 "value1" key2 "value2"  # 批量设置多个键值对
    SETNX lock:resource 1             # 仅当键不存在时才设置，常用于分布式锁
    SETEX user_session 3600 "data"    # 设置键值对并指定过期时间（秒）
    ```

#### 2. Hash（哈希）
哈希适合存储对象。
-   **设置与获取字段**
    ```bash
    HSET user:1000 name "Alice" age 30  # 为键 user:1000 设置字段 name 和 age
    HGET user:1000 name                 # 获取 user:1000 的 name 字段值
    HMSET product:500 name "手机" price 2999 stock 50  # 批量设置字段
    HGETALL user:1000                   # 获取键的所有字段和值
    ```
-   **其他操作**
    ```bash
    HINCRBY user:1000 age 1  # 将 age 字段的值增加 1
    HDEL user:1000 age       # 删除 age 字段
    HEXISTS user:1000 email  # 检查 email 字段是否存在
    ```

#### 3. List（列表）
列表是双向链表结构，便于在两端操作。
-   **推送与弹出元素**
    ```bash
    LPUSH tasks "task1"     # 从左侧插入元素
    RPUSH tasks "task2"     # 从右侧插入元素
    LPOP tasks              # 从左侧弹出一个元素
    RPOP tasks              # 从右侧弹出一个元素
    ```
-   **获取元素**
    ```bash
    LRANGE tasks 0 -1       # 获取列表所有元素
    LRANGE tasks 0 2        # 获取前3个元素
    LLEN tasks              # 获取列表长度
    ```

#### 4. Set（集合）
集合内的元素无序且唯一。
-   **基本操作**
    ```bash
    SADD tags "Redis" "Database" "Cache"  # 向集合 tags 添加元素
    SREM tags "Cache"         # 移除元素
    SMEMBERS tags             # 获取集合内所有元素
    SISMEMBER tags "Redis"   # 判断元素是否在集合中
    ```
-   **集合运算**
    ```bash
    SINTER set1 set2      # 返回 set1 和 set2 的交集
    SUNION set1 set2      # 返回 set1 和 set2 的并集
    SDIFF set1 set2       # 返回 set1 与 set2 的差集（即在set1中但不在set2中的元素）
    ```

#### 5. Sorted Set（有序集合）
有序集合为每个元素关联一个分数（score），便于排序。
-   **添加与范围查询**
    ```bash
    ZADD leaderboard 95 "Alice" 87 "Bob"  # 向有序集合 leaderboard 添加元素及其分数
    ZRANGE leaderboard 0 -1 WITHSCORES    # 按分数从低到高返回所有元素及其分数
    ZREVRANGE leaderboard 0 2 WITHSCORES  # 按分数从高到低返回前三名（常用于排行榜）
    ```
-   **按分数查询与其他操作**
    ```bash
    ZRANGEBYSCORE leaderboard 90 100       # 获取分数在90到100之间的元素
    ZRANK leaderboard "Alice"              # 获取元素正序排名（从0开始）
    ZINCRBY leaderboard 5 "Bob"            # 为 Bob 的分数增加 5
    ```

### ⚙️ 关键管理与通用命令
除了数据类型特定命令，以下命令对于日常管理和操作也至关重要。
-   **键管理**
    ```bash
    KEYS user:*         # 查找所有以 'user:' 开头的键（生产环境慎用，可能影响性能）
    DEL key1            # 删除一个键
    EXISTS key1         # 检查键是否存在
    EXPIRE key 60       # 设置键的过期时间为60秒
    TTL key             # 查看键的剩余生存时间
    TYPE key            # 查看键的数据类型
    ```
-   **服务器管理**
    ```bash
    INFO                # 查看Redis服务器的各种统计信息和指标
    SELECT 1            # 切换数据库（默认使用0号数据库）
    DBSIZE              # 获取当前数据库的键总数
    ```

### 💡 进阶特性与应用
Redis 的强大还体现在一些高级功能上：
-   **事务支持**：通过 `MULTI` 和 `EXEC` 命令可以将多个命令打包执行，确保原子性。
-   **发布订阅 (Pub/Sub)**：可以实现简单的消息系统。
-   **管道 (Pipeline)**：可以一次性发送多个命令，减少网络往返时间，提高性能。

希望这份总结能帮助你更好地理解和使用 Redis。如果你对某个特定功能或命令想了解更多细节，或者有具体的应用场景想讨论，随时可以告诉我。