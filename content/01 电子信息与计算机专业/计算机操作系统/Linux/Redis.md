Redis是一款高性能的键值存储系统，支持多种数据结构，广泛应用于缓存、队列、排行榜等场景。下面我将详细介绍Redis的核心数据结构、常用命令、高级功能及典型应用场景。

# 🔑 Redis核心知识与实战指南

## 📊 Redis数据结构与命令概览

| 数据结构                 | 主要特性                   | 典型应用场景           | 关键命令示例                         |
| -------------------- | ---------------------- | ---------------- | ------------------------------ |
| **String（字符串）**      | 可存储文本、数字或二进制数据，最大512MB | 缓存、计数器、分布式锁      | `SET`, `GET`, `INCR`, `DECR`   |
| **Hash（哈希）**         | 键值对集合，适合存储对象           | 用户信息、商品详情        | `HSET`, `HGET`, `HGETALL`      |
| **List（列表）**         | 按插入顺序排序的字符串元素集合，可重复    | 消息队列、最新消息列表      | `LPUSH`, `RPOP`, `LRANGE`      |
| **Set（集合）**          | 无序的字符串集合，元素唯一          | 标签系统、好友关系、唯一ID存储 | `SADD`, `SMEMBERS`, `SINTER`   |
| **Sorted Set（有序集合）** | 成员关联分数，按分数排序           | 排行榜、优先级队列        | `ZADD`, `ZRANGE`, `ZREVRANGE`  |
| **Bitmap（位图）**       | 字符串的位操作，极大节省空间         | 用户签到、活跃度统计       | `SETBIT`, `GETBIT`, `BITCOUNT` |
| **HyperLogLog**      | 基数估算，占用内存固定            | 大规模独立访客统计        | `PFADD`, `PFCOUNT`             |
| **Stream（流）**        | 日志式数据结构，支持消费者组         | 消息队列、事件溯源        | `XADD`, `XREAD`, `XGROUP`      |

## 🌟 核心数据结构详解

### 字符串（String）
字符串是Redis最基本的数据类型，一个键最多能存储512MB数据。

**常用命令：**
```bash
# 设置和获取值
SET user:1000:name "张三"
GET user:1000:name

# 数值操作
SET counter 100
INCR counter  # 结果：101
INCRBY counter 5  # 结果：106

# 带过期时间的设置
SETEX session:1234 3600 "session_data"

# 条件设置（仅当键不存在时）
SETNX lock:resource 1
```

**应用场景：**
- **缓存系统**：存储热点数据，如HTML片段、数据库查询结果
- **计数器**：实现文章的阅读量、点赞数统计
- **分布式锁**：通过`SETNX`命令实现简单的分布式锁

### 哈希（Hash）
哈希适合存储对象，可以将一个对象的多个属性存储在同一个键中。

**常用命令：**
```bash
# 设置和获取对象属性
HSET user:1000 name "李四" age 28 email "lisi@example.com"
HGET user:1000 name
HGETALL user:1000

# 批量操作
HMSET product:500 name "手机" price 2999 stock 50
HMGET product:500 name price

# 数值操作
HINCRBY product:500 stock -1  # 库存减1
```

**应用场景：**
- **用户信息存储**：将用户的所有属性存储在一个哈希中
- **商品信息缓存**：存储商品的多个详细信息

### 列表（List）
Redis列表是简单的字符串列表，按插入顺序排序，支持从两端插入和弹出元素。

**常用命令：**
```bash
# 从两端推入和弹出元素
LPUSH messages "msg1"
RPUSH messages "msg2"
LPOP messages
RPOP messages

# 获取列表片段
LRANGE messages 0 10  # 获取前11个元素
LTRIM messages 0 100  # 只保留前101个元素
```

**应用场景：**
- **消息队列**：使用`LPUSH`和`RPOP`实现简单的消息队列
- **最新列表**：存储最新的用户动态、新闻等

### 集合（Set）
集合是无序的字符串集合，且元素不可重复，支持交集、并集、差集等集合运算。

**常用命令：**
```bash
# 添加和获取元素
SADD tags:article:1000 "Redis" "数据库" "缓存"
SMEMBERS tags:article:1000

# 集合运算
SADD user:1000:follows "用户A" "用户B"
SADD user:1001:follows "用户B" "用户C"
SINTER user:1000:follows user:1001:follows  # 共同关注
```

**应用场景：**
- **标签系统**：为内容打标签并计算相似内容
- **唯一性保证**：确保数据的唯一性，如用户ID

### 有序集合（Sorted Set）
有序集合每个元素都关联一个分数（score），元素按分数从小到大排序，分数可以重复但成员不能重复。

**常用命令：**
```bash
# 添加元素和分数
ZADD leaderboard 95 "玩家A" 87 "玩家B" 92 "玩家C"

# 范围查询
ZREVRANGE leaderboard 0 9 WITHSCORES  # 获取前十名
ZRANGEBYSCORE leaderboard 90 100  # 获取90-100分的玩家

# 排名操作
ZRANK leaderboard "玩家A"  # 获取正序排名
ZREVRANK leaderboard "玩家A"  # 获取逆序排名
```

**应用场景：**
- **排行榜系统**：实现游戏积分榜、商品销量榜等
- **带权重的队列**：优先级任务调度

## ⚡ 高级功能与特性

### 事务处理
Redis支持简单的事务功能，可以一次执行多个命令。

```bash
# 开启和执行事务
MULTI
SET user:1000:name "王五"
SET user:1000:age 30
EXEC

# 监视键（乐观锁）
WATCH user:1000:balance
MULTI
DECRBY user:1000:balance 50
EXEC  # 如果balance在WATCH后发生变化，事务将失败
```

### 过期时间与数据持久化
Redis支持为键设置过期时间，并提供了两种持久化方式。

```bash
# 过期时间设置
SETEX session:1234 1800 "session_data"  # 30分钟后过期
EXPIRE user:1000:token 3600  # 设置1小时过期
TTL user:1000:token  # 查看剩余时间

# 持久化命令
SAVE  # 同步保存（生产环境慎用）
BGSAVE  # 后台异步保存
```

### 发布订阅模式
Redis提供了发布订阅功能，可以实现简单的消息系统。

```bash
# 订阅频道（客户端1）
SUBSCRIBE news.sports

# 发布消息（客户端2）
PUBLISH news.sports "比赛结果：主队3-0客队"
```

### 管道技术
管道技术可以一次性发送多个命令，减少网络往返时间，提高性能。

```bash
# 使用管道批量操作
(echo -en "PING\r\n SET key1 value1\r\n GET key1\r\n INCR counter\r\n"; sleep 1) | nc localhost 6379
```

## 🏗️ 实际应用场景

### 1. 电商平台应用
在电商平台中，Redis可以应用于多种场景：

- **商品缓存**：使用字符串或哈希缓存商品详情页
- **购物车**：使用哈希存储用户购物车信息
- **秒杀系统**：通过Redis原子操作保证库存扣减的准确性
- **用户会话**：存储用户登录状态和会话数据

### 2. 社交网络应用
社交网络可以利用Redis实现以下功能：

- **好友关系**：使用集合存储用户的好友和粉丝列表
- **时间线**：使用列表或有序集合实现用户动态时间线
- **实时聊天**：结合列表和发布订阅实现聊天功能

### 3. 实时数据处理
Redis适合实时数据统计和分析：

- **用户行为追踪**：使用位图记录用户活跃情况
- **实时计数器**：统计页面浏览量、点击量等
- **地理位置服务**：使用GEO数据类型存储和查询附近位置

## 📈 性能优化与最佳实践

1. **键名设计**：使用简洁但有意义的键名，避免过长的键名浪费内存

2. **内存优化**：根据数据特性选择合适的数据结构，如小对象使用哈希而非多个字符串键

3. **连接管理**：使用连接池减少连接建立开销，合理设置超时时间

4. **批量操作**：使用管道或批量命令减少网络往返次数

5. **监控与警报**：定期监控内存使用情况、命中率和响应时间

Redis的强大之处在于其丰富的数据结构和原子操作能力，合理利用这些特性可以构建出高性能的应用程序。根据实际需求选择最适合的数据结构和命令，是充分发挥Redis优势的关键。

希望这份详细的Redis用法指南对你有所帮助！如果你有特定场景下的具体问题，我可以提供更有针对性的建议。