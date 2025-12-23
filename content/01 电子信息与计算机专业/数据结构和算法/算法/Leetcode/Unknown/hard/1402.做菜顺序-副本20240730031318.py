#
# @lc app=leetcode.cn id=1402 lang=python3
#
# [1402] 做菜顺序
#
# https://leetcode.cn/problems/reducing-dishes/description/
#
# algorithms
# Hard (79.53%)
# Likes:    207
# Dislikes: 0
# Total Accepted:    33.8K
# Total Submissions: 42.5K
# Testcase Example:  '[-1,-8,0,5,-7]'
#
# 一个厨师收集了他 n 道菜的满意程度 satisfaction ，这个厨师做出每道菜的时间都是 1 单位时间。
# 
# 一道菜的 「 like-time 系数 」定义为烹饪这道菜结束的时间（包含之前每道菜所花费的时间）乘以这道菜的满意程度，也就是
# time[i]*satisfaction[i] 。
# 
# 返回厨师在准备了一定数量的菜肴后可以获得的最大 like-time 系数 总和。
# 
# 你可以按 任意 顺序安排做菜的顺序，你也可以选择放弃做某些菜来获得更大的总和。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：satisfaction = [-1,-8,0,5,-9]
# 输出：14
# 解释：去掉第二道和最后一道菜，最大的 like-time 系数和为 (-1*1 + 0*2 + 5*3 = 14) 。每道菜都需要花费 1
# 单位时间完成。
# 
# 示例 2：
# 
# 
# 输入：satisfaction = [4,3,2]
# 输出：20
# 解释：可以按照任意顺序做菜 (2*1 + 3*2 + 4*3 = 20)
# 
# 
# 示例 3：
# 
# 
# 输入：satisfaction = [-1,-4,-5]
# 输出：0
# 解释：大家都不喜欢这些菜，所以不做任何菜就可以获得最大的 like-time 系数。
# 
# 
# 
# 
# 提示：
# 
# 
# n == satisfaction.length
# 1 <= n <= 500
# -1000 <= satisfaction[i] <= 1000
# 
# 
#

# @lc code=start
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        satisfaction.sort()  # 排序
        total = 0  # 总的 like-time 系数和
        current_sum = 0  # 当前的满意度总和
        for s in reversed(satisfaction):  # 从满意度最高的菜开始考虑
            current_sum += s  # 当前的满意度总和
            if current_sum > 0:  # 只要加上这道菜的总和是增加的就继续加
                total += current_sum
            else:  # 如果当前的满意度总和小于等于0，则不再考虑前面的菜
                break
        return total
# @lc code=end

