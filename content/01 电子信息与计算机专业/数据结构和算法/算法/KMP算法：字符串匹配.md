---
title: "KMP算法：字符串匹配"
date: 2025-08-07
draft: false
---

[最浅显易懂的 KMP 算法讲解_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AY4y157yL/?spm_id_from=333.337.search-card.all.click&vd_source=7ef7dff4b509c161e6b86a796dbad2c5)

KMP（Knuth-Morris-Pratt）算法是一种用于字符串搜索的高效算法，其核心思想是当在文本字符串中搜索一个词时，能够利用已经部分匹配的信息避免从头开始搜索，从而提高搜索效率。这是通过预处理模式字符串来实现的，预处理阶段生成一个部分匹配表（也称为前缀表或失败函数），该表被用来决定下一步的搜索位置。

### KMP算法的工作原理

假设有文本字符串`T`和模式字符串`P`，我们的目标是在`T`中找到`P`的出现位置。

1. **预处理阶段**：生成模式字符串`P`的**部分匹配表 next数组** 。这个表会告诉我们在不匹配发生时，模式字符串`P`应该如何移动。对于`P`中的每一个位置`i`，部分匹配表记录了`P[0..i]`的前缀与后缀的最长公共元素的长度。这个值也表示当发生不匹配时，`P`应该向右移动的距离。

2. **搜索阶段**：使用部分匹配表来执行搜索。当`T`和`P`在某个位置不匹配时，我们可以利用部分匹配表来决定`P`下一步应该移动多远，而不是简单地将`P`向右移动一位。这样，我们就可以跳过一些不必要的比较。

### 优点

- KMP算法的主要优势在于其高效性。在最坏情况下，它的时间复杂度是线性的，即`O(n+m)`，其中`n`是文本字符串`T`的长度，`m`是模式字符串`P`的长度。这比简单的暴力搜索算法要好得多，后者在最坏情况下的时间复杂度为`O(n*m)`。

### 示例

考虑模式字符串`P = "ABCDABD"`，它的部分匹配表如下：

```
P:  A  B  C  D  A  B  D
   0  0  0  0  1  2  0
```

如果在文本字符串`T`中搜索`P`，当我们发现第六个字符（假设为`B`）与`T`中的某个字符不匹配时，根据部分匹配表，我们可以将`P`向右移动`2`位，而不是重新开始比较。

### 实现

```python
def KMP_search(T, P):
    # 部分匹配表的构建
    def build_partial_match_table(P):
        table = [0] * len(P)
        length = 0
        i = 1
        while i < len(P):
            if P[i] == P[length]:
                length += 1
                table[i] = length
                i += 1
            else:
                if length != 0:
                    length = table[length - 1]
                else:
                    table[i] = 0
                    i += 1
        return table
    
    table = build_partial_match_table(P)
    i = j = 0
    while i < len(T):
        if P[j] == T[i]:
            i += 1
            j += 1
        elif j > 0: # 字符失配，根据table跳过一些字符
            j = table[j - 1]
        else:
            i += 1 # j==0 的情况，第一个字符就失配
            
        if j == len(P):
            print("Found pattern at index " + str(i - j))
            j = table[j - 1] # 为了找到所有的匹配的index

# 使用示例
T = "ABABDABACDABABCABAB"
P = "ABABCABAB"
KMP_search(T, P)
```

# 相关的面试题
- 计算一个模式字符串的next数组
	思路：找到当前index之前的字符串的最长前后缀长度
