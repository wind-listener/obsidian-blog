---
title: "AMD CPU详细介绍"
date: 2025-10-07
draft: false
---

在AMD CPU中，CCD是Core Complex Die的缩写，即核心模块芯片，是AMD Ryzen处理器中的核心计算单元，负责处理器的主要计算任务。

每个CCD通常包含多个处理器核心，比如在Zen 2架构中，每个CCD包含两个CCX（CPU Complex），也就是8核16线程。而在Zen 5架构中，每个CCD包含8个Zen 5 CPU核心，这些核心共享32MB的三级缓存。在即将推出的Zen 6架构中，每个CCD的核心数量将增至12个，且每个CCD将配备48MB三级缓存。

AMD处理器通过将多个CCD与一个负责数据输入输出的IOD（Input/Output Die）集成在一个封装内，形成完整的CPU。这种模块化设计允许AMD根据需求扩展核心、线程和缓存数量，针对消费客户、服务器和HPC市场推出不同的产品。同时，CCD架构也有助于提高生产效率、控制成本，并使所有核心在运行时表现更为统一，提升程序运行的稳定性。