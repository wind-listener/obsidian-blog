"""
Author: Benjamin zzm_88orz@163.com
Date: 2024-08-20 22:57:18
LastEditors: Benjamin zzm_88orz@163.com
LastEditTime: 2024-08-20 23:15:27
FilePath: \编程基础四大件\数据结构和算法\算法\Leetcode\tree\hard\124.二叉树中的最大路径和.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""
#
# @lc app=leetcode.cn id=124 lang=python3
#
# [124] 二叉树中的最大路径和
#
# https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/
#
# algorithms
# Hard (45.70%)
# Likes:    2262
# Dislikes: 0
# Total Accepted:    442.5K
# Total Submissions: 966.4K
# Testcase Example:  '[1,2,3]'
#
# 二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个
# 节点，且不一定经过根节点。
# 
# 路径和 是路径中各节点值的总和。
# 
# 给你一个二叉树的根节点 root ，返回其 最大路径和 。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：root = [1,2,3]
# 输出：6
# 解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
# 
# 示例 2：
# 
# 
# 输入：root = [-10,9,20,null,null,15,7]
# 输出：42
# 解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
# 
# 
# 
# 
# 提示：
# 
# 
# 树中节点数目范围是 [1, 3 * 10^4]
# -1000 <= Node.val <= 1000
# 
# 
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def helper(node):
            nonlocal maxSum
            if not node:
                return 0
            leftMax = max(helper(node.left), 0)
            rightMax = max(helper(node.right), 0)
            currentSum = node.val+leftMax+rightMax
            maxSum = max(maxSum, currentSum)
            return node.val+max(leftMax, rightMax)
        maxSum = float('-inf')
        helper(root)
        return maxSum
            
# @lc code=end

