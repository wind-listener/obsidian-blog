---
title: "InternVL"
date: 2025-08-07
draft: false
---

为了链接视觉模型和大语言模型，已有的Vision large language models (VLLMs)通常使用轻量化的”glue“ layers：
- QFormer
- linear projection
这种 glue layers 的缺陷：
	1. 参数体量相差悬殊
	2. 表述不一致
	3. glue连接效率低下


**InternVL** is designed with 
- a vision encoder **InternViT-6B**
- a language middleware **QLLaMA**
- 
![[Pasted image 20240513165448.png]]