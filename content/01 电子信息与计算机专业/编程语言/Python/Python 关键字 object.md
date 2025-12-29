---
title: "Python 关键字 object"
date: 2025-10-29
draft: false
---


在 Python 中，`object` 是一个**基础类**（base class），它是所有类的终极父类。以下是关于 `object` 关键字的详细介绍：

---

### **1. 基本概念**
- `object` 是 Python 中所有类的**根类**（root class）。
- 如果一个类没有显式继承其他类，它会默认继承 `object`。
- 在 Python 3 中，`class Foo:` 和 `class Foo(object):` 是等价的（Python 2 中二者不同）。

---

### **2. 核心作用**
#### （1）提供基础方法
`object` 类定义了所有对象共有的**默认方法**，例如：
- `__init__`: 构造函数
- `__str__`: 返回对象的字符串表示（`print(obj)` 时调用）
- `__eq__`: 实现 `==` 操作符
- `__hash__`: 支持对象作为字典键

#### （2）类型系统基础
```python
isinstance(42, object)  # 返回 True（所有实例都是 object 的子类）
issubclass(int, object) # 返回 True（所有类都继承 object）
```

---

### **3. 实际用途**
#### （1）显式继承（Python 2 兼容性）
```python
class MyClass(object):  # Python 2 中必须显式继承 object 才能用新式类
    pass
```

#### （2）通用类型提示
```python
def func(arg: object) -> None:  # 接受任意类型的参数
    pass
```

#### （3）默认方法重载
```python
class Person:
    def __str__(self):
        return "Person object"  # 覆盖 object 的默认 __str__

print(Person())  # 输出 "Person object"
```

---

### **4. 与 `type` 的关系**
- `object` 是**所有类的父类**。
- `type` 是**所有类的元类**（包括 `object` 本身）。
- 特殊关系验证：
  ```python
  isinstance(object, type)   # True（object 是 type 的实例）
  isinstance(type, object)   # True（type 是 object 的子类）
  ```

---

### **5. 注意事项**
- **Python 2 vs 3**：  
  - Python 2 中需要显式继承 `object` 才能使用新式类（否则是旧式类）。
  - Python 3 中所有类默认继承 `object`（旧式类已移除）。

- **性能**：  
  直接继承 `object` 的类比多重继承的类更轻量。

---

### **代码示例**
```python
# 检查对象的基类
class Foo: pass
print(Foo.__bases__)  # 输出 (<class 'object'>,)

# 使用 object 的方法
obj = object()
print(dir(obj))  # 显示所有默认方法（__class__, __repr__ 等）
```

`object` 是 Python 面向对象编程的基石，理解它有助于深入掌握 Python 的类型系统和继承机制。