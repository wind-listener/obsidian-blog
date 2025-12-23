#
# @lc app=leetcode.cn id=936 lang=python3
#
# [936] 戳印序列
#
# https://leetcode.cn/problems/stamping-the-sequence/description/
#
# algorithms
# Hard (46.83%)
# Likes:    73
# Dislikes: 0
# Total Accepted:    4.8K
# Total Submissions: 10.1K
# Testcase Example:  '"abc"\n"ababc"'
#
# 你想要用小写字母组成一个目标字符串 target。 
# 
# 开始的时候，序列由 target.length 个 '?' 记号组成。而你有一个小写字母印章 stamp。
# 
# 在每个回合，你可以将印章放在序列上，并将序列中的每个字母替换为印章上的相应字母。你最多可以进行 10 * target.length  个回合。
# 
# 举个例子，如果初始序列为 "?????"，而你的印章 stamp 是 "abc"，那么在第一回合，你可以得到
# "abc??"、"?abc?"、"??abc"。（请注意，印章必须完全包含在序列的边界内才能盖下去。）
# 
# 如果可以印出序列，那么返回一个数组，该数组由每个回合中被印下的最左边字母的索引组成。如果不能印出序列，就返回一个空数组。
# 
# 例如，如果序列是 "ababc"，印章是 "abc"，那么我们就可以返回与操作 "?????" -> "abc??" -> "ababc" 相对应的答案
# [0, 2]；
# 
# 另外，如果可以印出序列，那么需要保证可以在 10 * target.length 个回合内完成。任何超过此数字的答案将不被接受。
# 
# 
# 
# 示例 1：
# 
# 输入：stamp = "abc", target = "ababc"
# 输出：[0,2]
# （[1,0,2] 以及其他一些可能的结果也将作为答案被接受）
# 
# 
# 示例 2：
# 
# 输入：stamp = "abca", target = "aabcaca"
# 输出：[3,0,1]
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= stamp.length <= target.length <= 1000
# stamp 和 target 只包含小写字母。
# 
# 
#

# @lc code=start
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        M, N = len(stamp), len(target)
        target = list(target)
        stamp = list(stamp)
        res = []
        done = [False]*N
        total_stars = 0

        def can_replace(start):
            changed=False
            for i in range(M):
                if target[start+i]=='?':
                    continue
                if target[start+i]!=stamp[i]:
                    return False
                changed = True
            return changed
        
        def do_replace(start):
            nonlocal total_stars
            for i in range(M):
                if target[start+i]!='?':
                    target[start+i]='?'
                    total_stars+=1
            res.append(start)
        
        while total_stars<N: # 没有全部替换回？就一直继续
            replaced=False
            for i in range(N-M+1):
                if not done[i] and can_replace(i):
                    do_replace(i)
                    done[i] = True
                    replaced = True
            if not replaced:
                return [] # 一次都没有发生替换，但是还有？没有被替换，说明没办法实现
            
        return res[::-1]


        

# @lc code=end

