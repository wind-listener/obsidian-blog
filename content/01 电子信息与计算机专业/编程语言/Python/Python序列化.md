在 Python 中，序列化（Serialization）是将对象转换为可存储或传输的格式（如字节流、字符串），反序列化（Deserialization）则是将序列化后的数据还原为原始对象的过程。以下是常见的序列化和反序列化方法：

---

### **1. 使用 `pickle` 模块（Python 原生）**
**特点**：支持几乎所有 Python 对象（包括自定义类），但仅限 Python 使用，不安全（反序列化可能执行任意代码）。
```python
import pickle

# 序列化：对象 → 字节流
data = {"name": "Alice", "age": 30}
serialized = pickle.dumps(data)  # 返回 bytes

# 保存到文件
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# 反序列化：字节流 → 对象
deserialized = pickle.loads(serialized)  # 返回原始对象

# 从文件加载
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
```

---

### **2. 使用 `json` 模块（跨语言）**
**特点**：跨语言兼容（JSON 格式），但仅支持基础数据类型（字典、列表、字符串、数字等）。自定义对象需额外处理。
```python
import json

# 序列化：对象 → JSON 字符串
data = {"name": "Bob", "scores": [90, 85]}
json_str = json.dumps(data)  # 返回字符串

# 保存到文件
with open("data.json", "w") as f:
    json.dump(data, f)

# 反序列化：JSON 字符串 → 对象
obj = json.loads(json_str)  # 返回字典

# 从文件加载
with open("data.json", "r") as f:
    loaded_data = json.load(f)
```

**处理自定义对象**：
```python
# 序列化自定义对象（需定义转换函数）
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

def encode_user(obj):
    if isinstance(obj, User):
        return {"name": obj.name, "age": obj.age}  # 转为字典
    raise TypeError("Object not serializable")

user = User("Charlie", 25)
user_json = json.dumps(user, default=encode_user)

# 反序列化自定义对象
def decode_user(dct):
    if "name" in dct and "age" in dct:
        return User(dct["name"], dct["age"])
    return dct

user_obj = json.loads(user_json, object_hook=decode_user)
```

---

### **3. 使用 `marshal` 模块（不推荐）**
**特点**：Python 内部格式，用于 `.pyc` 文件。**不推荐**用于通用序列化（格式不稳定，文档明确警告）。

---

### **4. 第三方库**
#### **(1) `yaml`（YAML 格式）**
```python
import yaml  # 需安装 pyyaml

# 序列化
data = {"key": "value"}
yaml_str = yaml.dump(data)

# 反序列化
obj = yaml.safe_load(yaml_str)  # 安全加载避免代码执行
```

#### **(2) `msgpack`（高效二进制）**
```python
import msgpack  # 需安装 msgpack

# 序列化
data = {"id": 1, "tags": ["a", "b"]}
packed = msgpack.packb(data)

# 反序列化
unpacked = msgpack.unpackb(packed)
```

#### **(3) `pydantic` + `json`（带验证的序列化）**
适用于数据模型验证和序列化：
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

# 对象 → JSON
item = Item(name="Book", price=9.99)
json_str = item.json()  # '{"name": "Book", "price": 9.99}'

# JSON → 对象
new_item = Item.parse_raw(json_str)
```

---

### **关键注意事项**
1. **安全性**：
   - `pickle` 反序列化可能执行恶意代码，**切勿加载不受信任的数据**。
   - 优先使用 `json` 或 `yaml.safe_load` 处理外部数据。
2. **自定义对象**：
   - `pickle` 自动处理自定义类（需类定义在作用域内）。
   - `json` 需手动实现转换逻辑（如 `default` 和 `object_hook`）。
3. **性能**：
   - 二进制格式（如 `pickle`、`msgpack`）通常比文本格式（如 `json`）更快更紧凑。

---

### **总结**
| **场景**                     | **推荐工具**          |
|-----------------------------|----------------------|
| Python 内部对象持久化        | `pickle`             |
| 跨语言/Web API 数据交换      | `json`               |
| 配置文件读写                | `yaml`              |
| 高性能二进制传输            | `msgpack`            |
| 数据模型验证 + 序列化       | `pydantic` + `json` |

根据需求选择合适的方法，优先考虑安全性和兼容性！