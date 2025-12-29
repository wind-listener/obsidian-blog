---
title: "MAE"
date: 2025-08-07
draft: false
---

https://zhuanlan.zhihu.com/p/446761025

MAE(Masked Autoencoders)是用于CV的自监督学习方法，优点是扩展性强的（scalable），方法简单。在MAE方法中会随机mask输入图片的部分patches，然后重构这些缺失的像素。MAE基于两个核心设计：（1）不对称的（asymmetric）[编码解码结构](https://zhida.zhihu.com/search?content_id=187409738&content_type=Article&match_order=1&q=%E7%BC%96%E7%A0%81%E8%A7%A3%E7%A0%81%E7%BB%93%E6%9E%84&zhida_source=entity)，编码器仅仅对可见的patches进行编码，不对mask tokens进行任何处理，解码器将编码器的输出（latent representation）和mask tokens作为输入，重构image；（2）使用较高的mask比例（如75%）。MAE展现了很强的[迁移性能](https://zhida.zhihu.com/search?content_id=187409738&content_type=Article&match_order=1&q=%E8%BF%81%E7%A7%BB%E6%80%A7%E8%83%BD&zhida_source=entity)，在ImageNet-1K上取得了best accuracy（87.8%），且因为方法简单，可扩展性极强（scalable）