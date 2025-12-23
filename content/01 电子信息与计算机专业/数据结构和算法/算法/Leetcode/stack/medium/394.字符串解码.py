"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-25 23:56:24
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-26 00:03:08
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\stack\medium\394.字符串解码.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=394 lang=python3
#
# [394] 字符串解码
#
# https://leetcode.cn/problems/decode-string/description/
#
# algorithms
# Medium (57.95%)
# Likes:    1818
# Dislikes: 0
# Total Accepted:    348.7K
# Total Submissions: 599.6K
# Testcase Example:  '"3[a]2[bc]"'
#
# 给定一个经过编码的字符串，返回它解码后的字符串。
# 
# 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
# 
# 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
# 
# 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：s = "3[a]2[bc]"
# 输出："aaabcbc"
# 
# 
# 示例 2：
# 
# 
# 输入：s = "3[a2[c]]"
# 输出："accaccacc"
# 
# 
# 示例 3：
# 
# 
# 输入：s = "2[abc]3[cd]ef"
# 输出："abcabccdcdcdef"
# 
# 
# 示例 4：
# 
# 
# 输入：s = "abc3[cd]xyz"
# 输出："abccdcdcdxyz"
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= s.length <= 30
# s 由小写英文字母、数字和方括号 '[]' 组成
# s 保证是一个 有效 的输入。
# s 中所有整数的取值范围为 [1, 300] 
# 
# 
#

# @lc code=start
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        current_num = 0
        current_string = ''
        
        for char in s:
            if char.isdigit():
                current_num = current_num*10 + int(char)
            elif char == '[':
                stack.append((current_string, current_num))
                current_string = ''
                current_num = 0
            elif char == ']':
                last_string, num = stack.pop()
                current_string = last_string + num*current_string
            else:
                current_string += char
        
        return current_string
# @lc code=end

