在 Python 中，`vars()` 是内置函数，核心作用是**返回对象的属性和属性值的字典表示**，或当前作用域的局部变量字典（无参数时）。以下从「语法、使用场景、核心特性、注意事项」四方面详细解释：

### 一、基本语法
```python
vars([object])
```
- **参数**：`object`（可选）—— 可以是类、实例、模块、函数等具有 `__dict__` 属性的对象；
- **返回值**：
  - 传参时：返回对象的 `__dict__` 属性（存储对象属性的字典）；
  - 不传参时：等价于 `locals()`，返回当前作用域的局部变量字典。

### 二、核心使用场景
#### 1. 无参数：获取当前作用域的局部变量
```python
# 示例：获取函数内的局部变量
def test_vars():
    a = 10
    b = "hello"
    print(vars())  # 等价于 print(locals())

test_vars()
# 输出：{'a': 10, 'b': 'hello'}
```
- 注意：在全局作用域中，`vars()` 等价于 `globals()`（返回全局变量字典）。

#### 2. 传实例对象：获取实例的属性字典
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 25)
print(vars(p))  # 返回实例p的属性字典
# 输出：{'name': 'Alice', 'age': 25}

# 可直接修改字典，同步修改实例属性
vars(p)["age"] = 26
print(p.age)  # 输出：26
```

#### 3. 传类对象：获取类的属性字典
```python
class Person:
    species = "human"  # 类属性
    def __init__(self, name):
        self.name = name

print(vars(Person))
# 输出：{'__module__': '__main__', 'species': 'human', '__init__': <function Person.__init__ at 0x...>, ...}
```

#### 4. 传模块：获取模块的属性/变量字典
```python
import math
print(vars(math))  # 返回math模块的所有属性（如pi、sqrt等）
# 输出：{'__name__': 'math', 'pi': 3.141592653589793, 'sqrt': <built-in function sqrt>, ...}
```

### 三、核心特性与注意事项
1. **并非所有对象都有 `__dict__`**：
   - 内置类型（如 `int`、`str`、`list`）无 `__dict__` 属性，调用 `vars(123)` 会报 `TypeError`；
   - 示例：
     ```python
     print(vars(123))  # 报错：TypeError: vars() argument must have __dict__ attribute
     ```

2. **`vars()` vs `dir()` 的区别**：
   | 函数   | 核心作用                     | 返回值类型       |
   |--------|------------------------------|------------------|
   | `vars()` | 返回对象的属性-值映射        | 字典（key=属性名，value=属性值） |
   | `dir()`  | 返回对象的所有属性名（含方法） | 列表（仅属性名，无值）|
   - 示例对比：
     ```python
     p = Person("Alice", 25)
     print(vars(p))  # {'name': 'Alice', 'age': 25}
     print(dir(p))   # ['__class__', '__delattr__', 'age', 'name', ...]
     ```

3. **只读对象的限制**：
   若对象的 `__dict__` 是只读的（如部分内置类实例），修改 `vars()` 返回的字典会报错。

4. **作用域特性**：
   - 在函数内调用 `vars()`（无参），返回的局部变量字典是「实时快照」，修改字典不会影响实际局部变量（与 `locals()` 一致）；
   - 示例：
     ```python
     def test():
         x = 1
         local_dict = vars()
         local_dict["x"] = 10  # 修改字典
         print(x)  # 输出：1（局部变量未变）
     test()
     ```

### 四、实用场景举例
#### 场景1：快速打印对象属性（调试用）
```python
class Product:
    def __init__(self, id, name, price):
        self.id = id
        self.name = name
        self.price = price

p = Product(1, "Phone", 2999)
# 替代手动 print(p.id, p.name, p.price)
print(f"Product info: {vars(p)}")
# 输出：Product info: {'id': 1, 'name': 'Phone', 'price': 2999}
```

#### 场景2：动态设置对象属性
```python
attrs = {"name": "Bob", "age": 30}
p = Person("", 0)
vars(p).update(attrs)  # 批量设置属性
print(p.name, p.age)  # 输出：Bob 30
```

### 总结
`vars()` 是调试和动态操作对象属性的便捷工具：
- 传参时：核心用于获取/修改对象（实例/类/模块）的属性字典；
- 无参时：等价于 `locals()`，用于查看当前作用域的局部变量；
- 注意区分 `vars()` 和 `dir()`，且仅对有 `__dict__` 属性的对象有效。