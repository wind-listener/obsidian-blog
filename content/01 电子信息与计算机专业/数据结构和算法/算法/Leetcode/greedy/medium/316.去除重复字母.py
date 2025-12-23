"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-11 18:05:22
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-11 18:10:23
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\greedy\medium\316.去除重复字母.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=316 lang=python3
#
# [316] 去除重复字母
#
# https://leetcode.cn/problems/remove-duplicate-letters/description/
#
# algorithms
# Medium (49.16%)
# Likes:    1096
# Dislikes: 0
# Total Accepted:    143.7K
# Total Submissions: 291.9K
# Testcase Example:  '"bcabc"'
#
# 给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：s = "bcabc"
# 输出："abc"
# 
# 
# 示例 2：
# 
# 
# 输入：s = "cbacdcbc"
# 输出："acdb"
# 
# 
# 
# 提示：
# 
# 
# 1 <= s.length <= 10^4
# s 由小写英文字母组成
# 
# 
# 
# 
# 注意：该题与 1081
# https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters
# 相同
# 
#

# @lc code=start
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        seen = set()
        lastindex = {c:i for i,c in enumerate(s)}
        
        for i,c in enumerate(s):
            if c in seen:
                continue
            while stack and c<stack[-1] and i<lastindex[stack[-1]]:
                seen.remove(stack.pop())
                
            stack.append(c)
            seen.add(c)
            
        return ''.join(stack)
# @lc code=end

