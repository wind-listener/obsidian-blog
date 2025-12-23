"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-11 17:48:22
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-11 17:52:26
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\stack\medium\402.移掉-k-位数字.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=402 lang=python3
#
# [402] 移掉 K 位数字
#
# https://leetcode.cn/problems/remove-k-digits/description/
#
# algorithms
# Medium (32.01%)
# Likes:    1067
# Dislikes: 0
# Total Accepted:    164.4K
# Total Submissions: 513.5K
# Testcase Example:  '"1432219"\n3'
#
# 给你一个以字符串表示的非负整数 num 和一个整数 k ，移除这个数中的 k 位数字，使得剩下的数字最小。请你以字符串形式返回这个最小的数字。
# 
# 
# 示例 1 ：
# 
# 
# 输入：num = "1432219", k = 3
# 输出："1219"
# 解释：移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219 。
# 
# 
# 示例 2 ：
# 
# 
# 输入：num = "10200", k = 1
# 输出："200"
# 解释：移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
# 
# 
# 示例 3 ：
# 
# 
# 输入：num = "10", k = 2
# 输出："0"
# 解释：从原数字移除所有的数字，剩余为空就是 0 。
# 
# 
# 
# 
# 提示：
# 
# 
# 1 
# num 仅由若干位数字（0 - 9）组成
# 除了 0 本身之外，num 不含任何前导零
# 
# 
#

# @lc code=start
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for char in num:
            while k>0 and stack and stack[-1]>char:
                stack.pop()
                k-=1
            stack.append(char)
            
        while k>0:
            stack.pop()
            k-=1
            
        result = ''.join(stack).lstrip('0')
        
        return result if result else '0'
# @lc code=end

