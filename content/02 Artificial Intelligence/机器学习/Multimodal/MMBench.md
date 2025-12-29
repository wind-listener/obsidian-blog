---
title: "特点"
date: 2025-08-07
draft: false
---

https://mmbench.opencompass.org.cn/home

https://github.com/open-compass/mmbench/

# 特点
1. 高质量、多样性
2. CircularEval + LLMs  to extract answer
3. 双语

# 数据集构建过程
## 收集
MMBench中80%以上的问题来自互联网。对于剩下的20%的样本，图像是从公共数据集的验证集(如果存在的话)中收集的，而问题是自构建的，不应该用于训练。
## 质量控制
剔除两种：
1. 答案可以只依据文本推理出来
	1. 使用多个LLMs，超过一半得到正确答案——> 人工审核
2. 存在错误：图片、问题、答案
	1. 使用多个VLMs，没一个能得到正确答案——> 人工审核
![[Pasted image 20240529145904.png]] 

## MMBench-CN 如何获得
GPT-4翻译+人工审核

# 数据集组成
任务分为三级

感知
	粗粒度感知
	细粒度感知-单例
	细粒度感知-多例
推理
	属性推理
	逻辑推理
	关系推理

- 共收集了3217个数据样本
- 涵盖了20种不同的L-3能力
- 每个L-3类别至少有125个样本
- MMBench DEV:TEST = 4:6
![[Pasted image 20240529145645.png]]

# 评估方法
## CircularEval 
![[Pasted image 20240529151318.png]]
![[Pasted image 20240529151433.png]]
## LLMs  to extract answer 
提升效果
![[Pasted image 20240529151213.png]]

# 结论
## 闭源模型的长处：
1. 结构化的文本理解
2. 需要外部知识的任务

## 现有VLMs的通病：
1. 理解低级图像特征
2. 结构化的图表
3. 空间关系
![[Pasted image 20240529151732.png]]