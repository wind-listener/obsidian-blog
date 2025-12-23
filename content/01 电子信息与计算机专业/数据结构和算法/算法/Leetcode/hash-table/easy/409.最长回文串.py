"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-26 00:04:48
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-26 00:08:57
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\hash-table\easy\409.最长回文串.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=409 lang=python3
#
# [409] 最长回文串
#
# https://leetcode.cn/problems/longest-palindrome/description/
#
# algorithms
# Easy (55.64%)
# Likes:    611
# Dislikes: 0
# Total Accepted:    207.6K
# Total Submissions: 373.2K
# Testcase Example:  '"abccccdd"'
#
# 给定一个包含大写字母和小写字母的字符串 s ，返回 通过这些字母构造成的 最长的 回文串 的长度。
# 
# 在构造过程中，请注意 区分大小写 。比如 "Aa" 不能当做一个回文字符串。
# 
# 
# 
# 示例 1: 
# 
# 
# 输入:s = "abccccdd"
# 输出:7
# 解释:
# 我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
# 
# 
# 示例 2:
# 
# 
# 输入:s = "a"
# 输出:1
# 解释：可以构造的最长回文串是"a"，它的长度是 1。
# 
# 
# 
# 
# 提示:
# 
# 
# 1 <= s.length <= 2000
# s 只由小写 和/或 大写英文字母组成
# 
# 
#

# @lc code=start
class Solution:
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter
        char_count = Counter(s)
        length = 0
        odd_found = False
        
        for count in char_count.values():
            if count%2 == 0:
                length+=count
            else:
                length += count - 1
                odd_found = True
                
        if odd_found:
             length += 1
             
        return length
# @lc code=end

