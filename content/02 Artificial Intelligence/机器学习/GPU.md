---
title: "GPU"
date: 2025-12-14
draft: false
---


作为刚接触 GPU 计算的新手，理解 GPU 在 AI 场景下的计算原理确实需要从基础架构到实际应用逐步深入。GPU 已经成为现代 AI 计算的核心驱动力，从 ChatGPT 到各种深度学习模型，GPU 的大规模并行计算能力为 AI 的爆发式发展提供了硬件基础。让我从 GPU 的计算原理出发，系统地讲解它在 AI 场景下的独特优势以及在 PyTorch 等框架中的具体实现。

## 一、GPU 与 CPU 架构的根本差异

### 1.1 架构设计理念的本质区别

GPU 和 CPU 在设计理念上存在根本性差异，这种差异决定了它们在不同计算场景中的适用性。CPU 的设计目标是**低延迟响应和复杂逻辑控制**，它拥有少量但强大的核心（通常 2-64 个），每个核心都配备了复杂的控制逻辑单元、多级缓存（L1、L2、L3）和分支预测机制。这种设计使得 CPU 擅长处理需要频繁分支判断、逻辑推理和快速上下文切换的任务，如操作系统调度、数据库查询、复杂算法执行等。

相比之下，GPU 的设计理念是**大规模并行处理和高吞吐量计算**。现代 GPU 包含数千个相对简单的计算核心，这些核心被组织成多个流式多处理器（SM）[(5)](https://cloud.tencent.com/developer/article/2556778)。GPU 将 80% 以上的芯片面积用于算术逻辑单元（ALU），而 CPU 的 ALU 占比通常不到 30%[(5)](https://cloud.tencent.com/developer/article/2556778)。这种设计使得 GPU 能够同时执行数万个线程，每个线程处理一个数据元素，特别适合处理可以并行化的计算密集型任务[(8)](https://www.cnblogs.com/dingxingdi/p/18767525)。

以 NVIDIA 的 A100 GPU 为例，它包含 108 个流式多处理器（SM）、40MB 的 L2 缓存，配备 80GB HBM2 内存，内存带宽高达 2039 GB/s[(21)](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)。这种架构设计让 GPU 在处理大规模数据并行任务时展现出压倒性优势，而 CPU 在处理串行逻辑任务时仍然不可替代。

### 1.2 并行计算模式的差异

GPU 的并行计算能力源于其独特的架构设计和执行模型。GPU 采用 **SIMT（单指令多线程）** 架构，这是一种将 SIMD（单指令多数据）与多线程技术相结合的并行计算模式。在 SIMT 架构中，多个线程可以执行相同的指令，但操作于不同的数据元素。

GPU 的线程组织采用**三级层次结构**：网格（Grid）- 线程块（Block）- 线程（Thread）[(5)](https://cloud.tencent.com/developer/article/2556778)。这种层次结构允许开发者将大规模计算任务分解为多个可并行执行的子任务。在实际执行时，GPU 以 **warp（束）** 为单位调度线程，每个 warp 包含 32 个线程，这些线程在同一个流式多处理器上执行。每个 warp 内的线程执行相同的指令，但可以访问不同的寄存器和内存地址，支持发散的控制流路径。

相比之下，CPU 的并行处理主要依赖于多核心和超线程技术，每个核心的设计更加复杂，包含了分支预测、乱序执行等高级特性。CPU 的并行处理通常是**任务级并行**，即不同的核心处理不同的任务或程序。而 GPU 的并行处理是**数据级并行**，即同一个任务被分解为多个子任务，在多个核心上同时执行。

### 1.3 内存架构的优化设计

GPU 的内存架构经过专门优化，以支持高带宽的数据访问模式。现代 GPU 配备了高带宽内存（HBM）或 GDDR6 显存，这些内存具有以下特点：

**HBM 内存**的优势在于其超高的带宽。HBM 采用 3D 堆叠封装技术，每堆 HBM 的总线宽度可达 1024 位甚至 2048 位，传输速率可达 6.4GT/s 以上。HBM2 内存的带宽计算公式为：时钟频率 × 2 × 1024 位 / 堆 × 堆数。相比之下，GDDR6 的带宽约为 0.064 TB/s，而 HBM2 可以达到 256 GB/s 的带宽[(16)](https://ru.wikipedia.org/wiki/High_Bandwidth_Memory)。

这种高带宽内存设计对 AI 任务至关重要，因为深度学习模型的训练和推理涉及大量的矩阵运算和数据访问。以 NVIDIA H200 为例，其更高的内存带宽确保了数据能够被高效访问和处理，在内存密集型的 HPC 应用和 AI 任务中，相比 CPU 可实现高达 110 倍的速度提升。

GPU 还采用了独特的内存层次结构，包括片上 L2 缓存、共享内存（Shared Memory）和本地内存（Local Memory）。这种多层次的内存结构使得 GPU 能够高效地处理大规模数据集，减少了对外部内存的访问次数，从而提高了整体的计算效率。

## 二、GPU 在 AI 场景下的独特优势

### 2.1 大规模并行处理能力的优势

GPU 在 AI 场景下的核心优势源于其**大规模并行处理能力**。AI 模型的训练和推理过程本质上是大量矩阵运算的集合，这些运算具有天然的并行性。GPU 拥有数千个计算核心，能够同时处理大量的数据元素，这种能力完美匹配了 AI 任务的计算需求。

在图像分类任务中，GPU 可以同时处理数千张图像的特征提取，相比 CPU 逐帧处理的 "串行模式"，效率提升可达 100 倍以上[(36)](https://juejin.cn/post/7494124948855062566)。这种并行处理能力不仅体现在图像任务上，在自然语言处理、推荐系统、科学计算等领域同样发挥着重要作用。

GPU 的并行架构专门针对**高吞吐量**进行优化，这使得它在执行神经网络训练和推理所需的大量运算时具有极高的效率[(37)](https://www.intel.cn/content/www/us/en/learn/gpu-for-ai.html)。每个 GPU 核心虽然运行速度比 CPU 核心慢，但数千个核心协同工作创造了惊人的计算能力。这种设计理念使得 GPU 特别适合处理需要大量重复计算的任务，如矩阵乘法、卷积运算、激活函数计算等。

以 NVIDIA 的 RTX 4090 为例，它搭载 16384 个 CUDA 核心，提供高达 83 TFLOPS 的张量浮点性能，原生支持 FP8、FP16、BF16 等低精度计算模式，显著提升了训练与推理效率[(40)](https://blog.csdn.net/weixin_42168902/article/details/152081358)。这种强大的计算能力使得原本需要数天甚至数周的模型训练任务可以在数小时内完成。

### 2.2 专用硬件加速单元的作用

GPU 在 AI 计算中的另一个重要优势是其**专用的硬件加速单元**，特别是张量核心（Tensor Core）。张量核心是 NVIDIA 为深度学习专门设计的处理单元，能够在单个时钟周期内执行复杂的矩阵运算。

Tensor Core 的设计理念是直接对小矩阵进行运算，而不是一次处理一个标量或向量元素。在一个时钟周期内，Tensor Core 可以执行 4x4x4 的 GEMM（通用矩阵乘法）运算，相当于同时进行 64 个浮点乘法累加（FMA）运算[(53)](https://juejin.cn/post/7447776289434812425)。这种设计使得 Tensor Core 在处理神经网络的矩阵运算时具有极高的效率。

最新一代的 Tensor Core 相比第一代性能提升了**60 倍**，在训练万亿参数生成 AI 模型时速度提升 4 倍，推理性能提升 30 倍[(43)](https://www.nvidia.com/zh-tw/data-center/tensor-cores/?ncid=so-twit-877903%3Fncid)。在 NVIDIA 的 Blackwell 架构中，每个流式多处理器包含 4 个第五代 Tensor Core，总计 640 个 Tensor Core，支持新的 NVFP4 精度格式，提供 15 PetaFLOPS 的密集 NVFP4 计算能力[(20)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。

除了张量核心，GPU 还包含了其他专用硬件单元：



* **CUDA 核心**：用于执行标准的浮点和整数运算

* **特殊函数单元（SFU）**：用于执行超越函数运算，如指数、对数、三角函数等

* **加载 / 存储单元**：用于高效的内存访问

* **warp 调度器**：负责调度和分发指令给各个 warp[(22)](https://aman.ai/primers/ai/gpu-architecture/)

这些硬件单元的协同工作使得 GPU 能够高效地执行各种 AI 相关的计算任务。

### 2.3 内存带宽与计算能力的匹配

GPU 在 AI 场景下的优势还体现在其**内存带宽与计算能力的完美匹配**上。AI 任务通常具有高算术强度的特点，即每个数据元素需要进行大量的计算。这种特性使得 GPU 的高内存带宽能够被充分利用。

以线性层（全连接层）为例，一个具有 4096 个输出、1024 个输入、批量大小为 512 的线性层，其算术强度为 315 FLOPS/B，属于算术密集型操作。在这种情况下，GPU 的计算能力成为性能瓶颈，而不是内存带宽。相比之下，ReLU 激活函数的算术强度仅为 0.25 FLOPS/B，属于内存密集型操作，这时候内存带宽就成为了性能瓶颈。

GPU 的高带宽内存设计，特别是 HBM2 和 GDDR6 等技术，确保了数据能够被快速传输到计算单元。NVIDIA 的 Ampere 架构 GPU 拥有领先的 2 TB/s 内存带宽，是上一代的两倍多。这种高带宽内存系统与强大的计算核心相结合，使得 GPU 在处理各种 AI 工作负载时都能保持高效的性能。

此外，GPU 还采用了多种技术来优化内存访问效率：



* **合并内存访问**：将多个线程的内存访问合并为一个大的内存事务

* **缓存优化**：利用片上缓存减少对外部内存的访问

* **内存对齐**：确保数据在内存中的对齐方式有利于高效访问

这些技术的综合应用使得 GPU 能够充分发挥其计算能力，避免了因内存瓶颈而导致的性能损失。

## 三、GPU 在 AI 任务中的计算机制

### 3.1 模型训练过程中的 GPU 计算

在 AI 模型训练过程中，GPU 主要负责执行**前向传播**和**反向传播**两个核心计算阶段。这两个阶段都涉及大量的矩阵运算，非常适合 GPU 的并行处理能力。

**前向传播的 GPU 计算机制**：

前向传播过程中，输入数据通过一系列矩阵乘法和激活函数变换，最终生成预测结果。在 GPU 上，这些计算被分解为多个并行的子任务。每个 GPU 核心负责计算输出张量中的一个或多个元素，通过大规模并行处理显著提高计算速度。

具体而言，前向传播中的每个线性层可以表示为矩阵乘法操作：Y = X × W + b，其中 X 是输入矩阵，W 是权重矩阵，b 是偏置向量。在 GPU 上，这个操作被映射为多个并行的线程块，每个线程块负责计算输出矩阵 Y 的一部分。现代 GPU 通过高度优化的 GEMM（通用矩阵乘法）库来执行这些计算，这些库充分利用了 GPU 的并行架构和张量核心。

**反向传播的 GPU 计算机制**：

反向传播是训练过程中计算梯度的关键步骤，同样非常适合 GPU 的并行计算特性。反向传播依赖于与前向传播类似的矩阵运算来计算梯度，GPU 能够并行执行这些乘法运算，大幅减少计算时间。

在分布式训练场景下，多个 GPU 通过**All-Reduce 通信方式**协同工作：



1. 每个 GPU 独立进行前向传播和反向传播，计算得到各自的梯度

2. 各 GPU 通过卡间通信，将梯度推送给一个类似管理者的 GPU（Server）

3. Server GPU 计算全局梯度并广播回每个 Worker GPU

4. 每个 GPU 使用全局梯度更新本地模型权重[(52)](https://it.ithome.com/archiver/0/901/261.htm)

这种分布式计算模式充分发挥了 GPU 的并行计算能力，使得大规模模型的训练成为可能。

### 3.2 数据预处理的 GPU 加速

数据预处理是 AI 工作流程中的重要环节，GPU 在这方面也发挥着重要作用。传统的数据预处理通常在 CPU 上进行，这往往成为整个训练流程的瓶颈。通过将数据预处理迁移到 GPU 上，可以显著提高整体的训练效率。

**GPU 加速数据预处理的优势**：

GPU 加速工具如 RAPIDS 的核心作用是通过将传统 CPU 主导的数据处理、分析和建模流程迁移到 GPU 上，利用 GPU 的并行计算能力（数千个核心对比 CPU 的数十个核心）实现大幅提速[(55)](https://blog.51cto.com/u_16213585/14198419)。

NVIDIA 数据加载库（DALI）是一个专门用于 GPU 加速数据加载和预处理的库，它提供了高度优化的数据处理构建块，能够加速深度学习应用的训练和推理过程。使用 DALI 实现的数据处理管道具有良好的可移植性，可以轻松地重定向到 TensorFlow、PyTorch 和 MXNet 等主流框架。

**流式数据处理的创新**：

Deep Lake 通过革命性的**惰性加载（Lazy Loading）技术**解决了传统数据加载的瓶颈问题。其核心原理是仅在 GPU 需要时才从云端流式传输并处理数据块，将传统的 "下载 - 预处理 - 训练" 三阶段流程压缩为单一流水线。实验表明，在 100 epoch 的 ResNet 训练中，该机制可使重复数据访问速度提升 40 倍，将数据加载时间从总训练时间的 35% 降至 5% 以下[(57)](https://blog.csdn.net/gitblog_00184/article/details/152063278)。

这种流式处理方式不仅减少了内存占用，还提高了 GPU 的利用率。通过将数据预处理与模型训练重叠执行，可以确保 GPU 始终处于忙碌状态，避免了因等待数据而造成的空闲时间。

### 3.3 不同 AI 模型架构的 GPU 优化策略

不同类型的 AI 模型在 GPU 上的计算机制和优化策略存在差异，GPU 厂商针对这些差异开发了相应的优化技术。

**CNN（卷积神经网络）的 GPU 计算机制**：

卷积运算是 CNN 的核心操作，GPU 通过多种方式优化卷积计算：



* **im2col 技术**：将输入数据和滤波器展开为大矩阵，将传统卷积转换为矩阵乘法，然后使用高度优化的 GEMM 库进行计算[(63)](http://howie.seas.gwu.edu/publications/GPU-CNN-ICPP16.pdf)

* **逐像素 / 线程并行**：每个像素或输出元素独立计算，允许每个输出像素或窗口对应一个线程，对空间不变和空间变化的卷积都特别有效[(61)](https://www.emergentmind.com/topics/gpu-accelerated-convolution)

* **FFT 卷积算法**：通过快速傅里叶变换将卷积转换为频域的点乘运算，适用于大规模卷积运算[(65)](https://arxiv.org/pdf/2103.16234)

在实际应用中，VGG-16 在 GPU 上训练 CIFAR-100 需要 13,000 秒，而在 CPU 上需要 130,000 秒；ResNet-18 作为最省时的模型，在 GPU 上仅需 150 秒，而在 CPU 上需要 1,740 秒[(48)](https://benthamscience.com/article/148842)。

**Transformer 模型的 GPU 计算机制**：

Transformer 模型的核心是**自注意力机制**，它通过计算序列中各 token 之间的关系权重实现全局交互。FlashAttention 是一种针对 Transformer 模型中注意力机制的计算优化技术，它在数学上与标准注意力等价，但实现方式完全不同，能够显著减少内存访问和计算复杂度[(66)](https://juejin.cn/post/7551214653553721379)。

在大规模部署中，GPU 通过多种并行策略优化 Transformer 的计算：



* **KV 并行（KVP）**：将百万 token 的 KV 缓存沿序列维度分片到多个 GPU 上

* **张量并行（TPA）**：将 QKV 投影在注意力头之间分片，避免重复计算

最新的 FlashAttention-3 算法通过沿两个轴协调异步性 —— 生产者 - 消费者并行和交错计算，将传统的基于 tile 的注意力计算分解为异步阶段，进一步提高了计算效率[(67)](https://www.emergentmind.com/topics/flashattention-3)。

**RNN/LSTM 的 GPU 计算机制**：

RNN 和 LSTM 模型的计算具有顺序依赖的特点，这在一定程度上限制了 GPU 的并行能力。然而，GPU 仍然通过以下方式优化 RNN 的计算：



* **批量处理**：同时处理多个序列，利用序列间的并行性

* **向量化操作**：将循环操作转换为向量化的矩阵运算

* **时间步并行**：在可能的情况下，并行处理不同时间步的计算

尽管 RNN 的并行性不如 CNN 和 Transformer，但 GPU 的优化仍然能够带来显著的性能提升。在处理较短序列时，CNN 表现更好；而在处理需要长期依赖建模的任务时，Transformer 具有优势[(45)](https://mljourney.com/cnn-vs-transformer-for-sequence-data-2/)。

## 四、GPU 计算在 PyTorch 中的实现与应用

### 4.1 PyTorch 的 GPU 张量与内存管理

PyTorch 作为目前最流行的深度学习框架之一，对 GPU 计算提供了全面而高效的支持。理解 PyTorch 如何在 GPU 上管理张量和内存，是掌握 GPU 计算的关键。

**CUDA 张量的基础架构**：

PyTorch 通过`torch.cuda`包提供 CUDA 张量类型支持，这些张量实现了与 CPU 张量相同的功能，但利用 GPU 进行计算。PyTorch 为其所有原生函数提供了 CUDA 实现，以确保在 NVIDIA GPU 硬件上的操作速度更快。

PyTorch 支持多种 CUDA 张量类型：



* `torch.cuda.FloatTensor`：32 位浮点型

* `torch.cuda.DoubleTensor`：64 位浮点型

* `torch.cuda.HalfTensor`：16 位浮点型

* `torch.cuda.BFloat16Tensor`：BF16 类型[(69)](https://glaringlee.github.io/tensors.html)

这些张量类型与 CPU 张量具有相同的接口，但存储在 GPU 显存中，并在 GPU 上执行计算。

**GPU 内存管理机制**：

PyTorch 的 GPU 内存管理采用了**缓存内存分配器**来加速内存分配。这种设计使得`nvidia-smi`显示的值通常不能反映真实的内存使用情况。PyTorch 为 GPU 编写了自定义内存分配器，确保深度学习模型具有最大的内存效率，使开发者能够训练比以前更大的深度学习模型。

在实际的内存分配中，PyTorch 会在目标设备（如 GPU）上分配连续的显存，实际分配的大小受最小粒度影响，通常大于或等于理论占用。当多个张量迁移到 GPU 后，总显存占用为各张量实际分配显存之和[(71)](https://blog.csdn.net/Bruce_Liu001/article/details/151142776)。

**异步内存传输机制**：

PyTorch 支持异步内存传输，通过设置`pin_memory=True`可以将张量保存到 CPU 的锁页内存中，并在解包时异步复制到 GPU。这种机制减少了数据传输的延迟，提高了整体的训练效率。

### 4.2 自动微分与 GPU 加速

PyTorch 的自动微分机制在 GPU 上的实现是其高效性的关键之一。自动微分通过计算图来跟踪所有的计算操作，并在反向传播时自动计算梯度。

**GPU 上的自动微分实现**：

PyTorch 的自动微分机制在 GPU 上的工作原理与 CPU 类似，但利用了 GPU 的并行计算能力来加速梯度计算。在 GPU 上，梯度计算通过向量化操作实现，每个线程负责计算梯度张量中的一个元素。

在混合精度训练中，PyTorch 通过`torch.cuda.amp`包提供了便利的方法。混合精度训练中，一些操作使用`torch.float32`数据类型，另一些操作使用`torch.float16`数据类型[(72)](https://glaringlee.github.io/amp.html)。自动混合精度训练通常使用`torch.cuda.amp.autocast`和`torch.cuda.amp.GradScaler`配合使用。

**计算图的 GPU 优化**：

PyTorch 的自动微分实现会在存储计算图时忽略参数可微分性的信息。虽然这些信息在许多现代微调任务中（只需要部分参数的梯度）有助于减少内存使用，但目前的实现并未充分利用这一点。

为了减少内存使用，开发者可以使用梯度检查点技术。通过在计算图中设置检查点，并在反向传播期间重新计算检查点之间的图部分，可以在降低内存成本的同时计算梯度。

### 4.3 混合精度训练的 GPU 实现

混合精度训练是 GPU 计算在 AI 场景中的一个重要应用，它通过使用不同精度的数据类型来平衡计算速度和模型精度。

**自动混合精度的工作原理**：

自动混合精度训练的典型实现模式如下：



```
with autocast():

&#x20;   output = model(input)

&#x20;   loss = loss\_fn(output, target)

scaler.scale(loss).backward()

scaler.step(optimizer)

scaler.update()
```

在这个过程中，`autocast`上下文管理器自动为选定的 GPU 操作选择适当的精度以提高性能，同时保持准确性。`GradScaler`则负责处理混合精度训练中的梯度缩放，防止梯度下溢。

**混合精度训练的优势**：

混合精度训练能够带来多方面的优势：



1. **计算速度提升**：使用 16 位浮点运算比 32 位浮点运算更快

2. **内存占用减少**：16 位浮点占用的内存是 32 位的一半

3. **带宽需求降低**：减少了内存访问，提高了内存利用率

实验结果表明，通过 Intel Extension for PyTorch 实现的自动混合精度优化能够在保持模型精度的前提下显著提升训练和推理性能[(73)](https://blog.51cto.com/deephub/14204323)。

### 4.4 CUDA 流与事件的高级应用

CUDA 流和事件是 PyTorch 中用于管理 GPU 并行和同步的高级机制，它们对于优化复杂的 AI 工作流程至关重要。

**CUDA 流的工作机制**：

CUDA 流是一个任务队列，所有提交到同一个流中的操作会按照顺序执行，但不同流中的操作可以并行执行[(74)](https://blog.csdn.net/2303_77224751/article/details/142426250)。在 PyTorch 中，当多个算子和内核被并行执行时，PyTorch 通过 CUDA 的流和事件机制来管理并发和同步。

CUDA 事件是同步标记，用于监控设备进度、精确测量时间和同步 CUDA 流。事件在第一次记录或导出到另一个进程时惰性初始化。

**流和事件的实际应用**：

开发者可以通过以下方式使用流和事件：



1. **记录事件**：使用`stream.record_event(event)`在流中记录事件

2. **等待事件**：使用`stream.wait_event(event)`让流等待特定事件完成

3. **流间同步**：使用`s0.wait_stream(s1)`让流 s0 等待流 s1 完成

这些机制在需要精确控制计算和数据传输顺序的场景中特别有用，例如：



* 数据预处理与模型推理的重叠执行

* 多 GPU 间的数据同步

* 性能分析和基准测试

### 4.5 分布式训练中的 GPU 协同

分布式训练是训练大规模 AI 模型的必要技术，PyTorch 通过多种方式支持 GPU 的分布式协同工作。

**PyTorch Lightning 的简化接口**：

PyTorch Lightning 通过封装 DP/DDP 的底层逻辑，提供了简化的多卡训练接口。开发者只需通过 Trainer 的参数配置多卡策略，无需手动管理进程组或数据分片[(82)](https://blog.csdn.net/sjtu_wyy/article/details/149631582)。典型的配置如下：



```
trainer = Trainer(accelerator="gpu", devices=8, strategy="ddp")
```

在 GPU 上运行时，Lightning 默认会选择 NCCL 后端而非 gloo，以获得更好的性能[(76)](https://lightning.ai/docs/pytorch/2.5.1/accelerators/gpu_intermediate.html)。

**DDP 的底层机制**：

PyTorch 的分布式数据并行（DDP）通过以下核心机制实现：



1. **进程组初始化**：所有进程创建一个进程组，使它们能够参与集合通信操作如 All-Reduce

2. **梯度归约**：各节点的梯度被汇聚（求和或平均），然后同步回所有节点[(83)](https://cloud.tencent.cn/developer/article/2593964)

3. **桶化优化**：梯度被分桶以增加通信和计算之间的重叠

DDP 需要手动初始化进程组，通常通过`torchrun`或`torch.distributed.launch`启动多进程[(82)](https://blog.csdn.net/sjtu_wyy/article/details/149631582)。

**全分片训练策略**：

全分片训练（FSDP）是一种更高效的分布式训练策略，它将整个模型分片到所有可用的 GPU 上，允许扩展模型规模，同时使用高效通信减少开销[(79)](https://lightning.ai/docs/pytorch/1.8.4/api/pytorch_lightning.strategies.DDPFullyShardedNativeStrategy.html)。这种策略特别适合训练超大规模的模型，如具有数百亿参数的语言模型。

### 4.6 PyTorch 中的其他 GPU 优化技术

除了上述核心机制，PyTorch 还提供了多种其他 GPU 优化技术。

**内存映射技术**：

在 NVIDIA Grace Blackwell 和 Grace Hopper 架构中，CPU 和 GPU 通过 NVLink-C2C 连接，提供 900 GB/s 带宽的内存一致性互连，比 PCIe Gen 5 高 7 倍。NVLink-C2C 的内存一致性功能允许 CPU 和 GPU 共享统一的内存地址空间，无需显式的数据传输或重复复制即可共同访问和处理相同的数据。

这种架构使得超过传统 GPU 内存容量的大规模数据集和模型能够更容易地被访问和处理。例如，在 NVIDIA GH200 Grace Hopper 超级芯片这样的统一内存架构平台上加载模型时，可以利用 96GB 的高带宽 GPU 内存和 480GB 的 CPU 连接 LPDDR 内存，而无需显式的数据传输。

**模型并行与流水线并行**：

对于无法在单个 GPU 内存中容纳的超大规模模型，PyTorch 支持模型并行和流水线并行：



* **模型并行**：将模型的不同层分布在不同的 GPU 上

* **流水线并行**：将模型的层划分为多个阶段，在不同的 GPU 上并行处理[(50)](https://www.deepspeed.ai/tutorials/pipeline/)

这些技术使得训练具有数千亿参数的模型成为可能。

**性能分析工具**：

PyTorch 提供了多种性能分析工具来帮助开发者优化 GPU 代码：



* `torch.autograd.profiler`：用于分析计算图的执行时间

* `torch.cuda.profiler`：用于分析 GPU 内核的执行时间

* `torch.cuda.memory_profiler`：用于分析 GPU 内存使用情况

通过这些工具，开发者可以识别性能瓶颈并进行针对性的优化。

## 五、GPU 计算的发展趋势与挑战

### 5.1 最新 GPU 架构的技术突破

2025 年，GPU 技术在 AI 领域取得了重大突破，各大厂商都推出了新一代的架构和产品。

**NVIDIA 的技术进展**：

NVIDIA 在 2025 年推出了基于 Blackwell 架构的新一代产品，包括：



* **Blackwell Ultra**：采用双芯片设计，通过 NV-HBI 互连提供 10 TB/s 带宽，包含 2080 亿晶体管，是 H100 的 2.6 倍[(20)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

* **RTX 50 系列**：针对神经网络渲染进行了优化，Blackwell 多单元流处理器具有更高的处理吞吐量，并与 Tensor Core 更紧密结合[(25)](https://www.nvidia.cn/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)

* **DGX Spark 和 DGX Station**：扩展了基于 Grace Blackwell 技术的 AI 计算能力[(86)](https://blogs.idc.com/2025/05/19/nvidia-dgx-and-the-future-of-ai-desktop-computing/)

NVIDIA 还推出了 Cosmos 平台，该平台推进物理 AI 发展，为机器人、自动驾驶车辆和视觉 AI 提供新模型和视频数据处理管道。

**AMD 的竞争策略**：

AMD 在 2025 年推出了 RDNA 4 架构，发布了 Radeon RX 9000 系列显卡，包括 RX 9070 XT 和 RX 9070，配备 16GB 显存[(26)](https://www.amd.com/zh-cn/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html)。AMD 还推出了基于 CDNA 4 架构的 Instinct MI350 系列，采用 3nm 工艺，单个设备集成 1850 亿晶体管，专为下一代 AI 加速而设计。

**架构创新的共同趋势**：

2025 年 GPU 架构的发展呈现以下共同趋势：



1. **集成度提升**：采用更先进的制程工艺，集成更多晶体管

2. **专用化增强**：为 AI 工作负载设计专门的硬件单元

3. **互连技术进步**：提供更高带宽的 GPU 间互连

4. **内存容量扩展**：支持更大容量的高带宽内存

### 5.2 新兴 AI 芯片的竞争格局

随着 AI 应用的快速发展，GPU 面临着来自多种新兴 AI 芯片的竞争，形成了多元化的技术格局。

**TPU 的技术特点与优势**：

Google 的 TPU（张量处理单元）是 GPU 的主要竞争对手之一。TPU 的设计理念是用硬件直接映射神经网络中的张量运算，不浪费任何逻辑判断时间。在同等功耗下，TPU 在 AI 推理任务上能比 GPU 提高数倍性能[(89)](https://blog.csdn.net/aifs2025/article/details/153814907)。

TPU 的发展呈现以下特点：



* **专业化程度高**：专门为 AI/ML 任务设计，在特定场景下效率极高

* **成本效益好**：批量生产降低了单位成本

* **性能持续提升**：TPU v5 承诺性能提升 2 倍，增强了稀疏性支持和动态形状支持[(92)](https://www.ofzenandcomputing.com/tpus-vs-npus-complete-ai-hardware-comparison-guide-cy/)

Google 的 TPU 生产预计到 2028 年将达到 700 万台[(91)](https://www.alphamatch.ai/blog/google-tpu-nvidia-ai-chip-competition-2025)，这显示了 TPU 在 AI 芯片市场的重要地位。

**NPU 的技术突破**：

韩国科学技术院（KAIST）的最新突破标志着一个强有力的新挑战者：一种专门为生成式 AI 模型设计的高能效神经处理单元（NPU）核心技术[(90)](https://tgdaily.com/technology/nvidias-huge-risk-kaist-npus-are-better-at-ai-than-gpus/)。

NPU 的发展重点是将桌面级 AI 性能带到移动设备，在某些推理任务上，NPU 的功耗可能比 GPU 低 10 倍。这种能效优势使得 NPU 在边缘 AI 应用中具有独特的竞争力。

**未来的多元化格局**：

专家预测，未来的 AI 计算将依靠**GPU + TPU + NPU + DPU + FPGA + ASIC**的生态协作[(89)](https://blog.csdn.net/aifs2025/article/details/153814907)。这种多元化格局的形成基于以下原因：



1. **应用场景多样化**：不同的 AI 应用对性能、功耗、成本有不同需求

2. **技术路线差异化**：各种芯片在架构设计上各有特色

3. **产业生态发展**：不同厂商基于自身优势选择不同的技术路线

### 5.3 大模型时代的 GPU 挑战

随着 AI 模型规模的快速增长，GPU 在支撑大模型训练和推理方面面临着前所未有的挑战。

**内存容量的瓶颈**：

大模型时代的 GPU 面临严重的内存容量限制。训练一个万亿参数的大模型需要 180 张 80GB 显存的 A100 GPU，这样的 "豪华配置" 普通团队根本无法承受[(95)](https://m.kepuchina.cn/tuwendetail?id=613978)。模型参数呈指数级增长，而 GPU 显存的增长速度远远跟不上。

以最新的大语言模型为例：



* **LLaMA 3 70B**：需要约 140GB 内存（FP16 精度）

* **LLaMA 4 Scout 109B**：需要约 218GB 内存（FP16 精度）

* **KV 缓存需求**：处理 128k token 上下文窗口需要约 40GB 内存，且随用户数线性增长

常用的经验法则是，全精调需要约每十亿参数 16GB 显存，这意味着一个 7B 参数的模型如果没有优化可能需要超过 100GB 的 GPU 内存。

**功耗与成本的急剧上升**：

GPU 功耗的急剧上升是另一个重大挑战。NVIDIA GPU 的功耗增长呈现指数级趋势：



* **A100**：约 250W（重负载）

* **H100**：高达 700W（几乎是 A100 的 3 倍）[(99)](https://www.nextdc.com/blog/whats-next-for-neo-clouds-0)

* **Blackwell**：约 2000W，每月电费可达 300 美元（假设满负荷运行，电价 150 美元 / 千瓦・月）[(100)](https://www.cloudsyntrix.com/blogs/how-nvidia-blackwell-gpus-are-transforming-cloud-costs-and-revenue/)

* **Kyber 世代**：预计功耗从 Ampere 基准水平放大 100 倍，单机架热设计功耗预计激增 100 倍[(98)](http://m.toutiao.com/group/7561603907743466035/?upstream_biz=doubao)

这种功耗增长带来了多重影响：



1. **运营成本高企**：2025 年，在生产环境中运行单个 AI 模型每月可能花费数百万美元的 GPU 计算成本[(88)](https://www.linkedin.com/pulse/ai-cloud-nexus-why-gpus-new-gold-2025-edition-lab7ai-fsvaf)

2. **基础设施压力**：数据中心需要大幅升级电力和冷却系统

3. **环境影响**：巨大的能源消耗对环境造成压力

**技术发展的新方向**：

为应对这些挑战，GPU 技术发展呈现以下新方向：



1. **统一内存架构**：通过 CPU-GPU 内存共享扩展可用内存

2. **量化技术**：使用更低精度的数据类型减少内存需求

3. **模型并行优化**：将模型分片到多个 GPU 上

4. **推理优化**：针对推理场景优化功耗和性能

### 5.4 未来发展展望

展望未来，GPU 在 AI 领域的发展将呈现以下趋势：

**技术创新的重点方向**：



1. **更高的集成度**：采用更先进的制程工艺，集成更多的计算单元

2. **专用化程度提升**：为特定的 AI 工作负载设计专门的硬件

3. **互连技术突破**：提供更高带宽、更低延迟的 GPU 间通信

4. **内存技术革新**：开发更高容量、更高带宽的显存技术

**应用场景的拓展**：

GPU 在 AI 领域的应用将不断拓展：



1. **边缘 AI**：将 AI 计算能力带到终端设备

2. **实时 AI**：支持实时的语音识别、图像理解等应用

3. **多模态 AI**：同时处理文本、图像、语音等多种数据类型

4. **科学计算**：在气候模拟、药物研发等领域发挥更大作用

**产业生态的演进**：

GPU 产业生态将继续演进：



1. **软件栈完善**：AI 框架和工具链将更加成熟和易用

2. **标准化推进**：行业标准和接口将更加统一

3. **生态合作加强**：硬件厂商、软件开发商、云服务商将形成更紧密的合作

**面临的长期挑战**：

尽管 GPU 在 AI 领域取得了巨大成功，但仍面临长期挑战：



1. **能效比提升**：需要在提高性能的同时降低功耗

2. **成本控制**：需要降低 AI 计算的门槛，让更多人能够使用

3. **可持续发展**：需要考虑环境影响，发展绿色 AI 计算

4. **技术替代风险**：需要应对来自其他技术路线的竞争

## 结语

通过对 GPU 在 AI 场景下计算原理的深入探讨，我们可以看到 GPU 已经成为现代 AI 计算的基石。从基础的硬件架构到复杂的并行计算机制，从底层的 CUDA 编程到高层的 AI 框架集成，GPU 通过其独特的大规模并行处理能力和专用硬件加速单元，为 AI 的快速发展提供了强大的算力支撑。

在 PyTorch 等主流 AI 框架中，GPU 计算的优势得到了充分的发挥和体现。通过 CUDA 张量、自动微分、混合精度训练、分布式计算等技术，开发者能够轻松地利用 GPU 的强大算力来训练和部署各种 AI 模型。特别是在处理大规模、计算密集型的 AI 任务时，GPU 相比 CPU 展现出了压倒性的优势。

然而，我们也必须清醒地认识到，GPU 计算在 AI 领域面临着诸多挑战。内存容量的限制、功耗的急剧上升、成本的高企以及来自其他 AI 芯片的竞争，都对 GPU 的未来发展提出了严峻的考验。这些挑战需要整个产业共同努力，通过技术创新、架构优化和生态建设来应对。

展望未来，GPU 在 AI 领域仍将发挥重要作用，但其发展道路不会一帆风顺。随着 AI 应用的不断演进和新兴技术的涌现，GPU 需要不断创新和进化，才能在激烈的竞争中保持领先地位。对于刚接触 GPU 计算的你来说，深入理解 GPU 的计算原理，掌握其在 AI 场景下的应用方法，将为你在 AI 领域的发展奠定坚实的基础。

GPU 计算的发展历程告诉我们，技术的进步永无止境。在 AI 时代的浪潮中，只有不断学习、勇于创新，才能把握住技术发展的机遇，为 AI 的发展贡献自己的力量。希望本文的内容能够帮助你更好地理解 GPU 在 AI 场景下的计算原理，开启你在 GPU 计算领域的探索之旅。

**参考资料&#x20;**

\[1] CPU能不能代替GPU?那GPU又能不能代替CPU呢?\_创业者李孟[ http://m.toutiao.com/group/7544383975075332648/?upstream\_biz=doubao](http://m.toutiao.com/group/7544383975075332648/?upstream_biz=doubao)

\[2] Understanding CPUs, GPUs, NPUs, and TPUs: A Simple Guide to Processing Units[ https://guptadeepak.com/understanding-cpus-gpus-npus-and-tpus-a-simple-guide-to-processing-units/](https://guptadeepak.com/understanding-cpus-gpus-npus-and-tpus-a-simple-guide-to-processing-units/)

\[3] CPU vs GPU: Comparing Key Differences Between Processing Units[ https://www.servermania.com/kb/articles/cpu-vs-gpu](https://www.servermania.com/kb/articles/cpu-vs-gpu)

\[4] CPU, GPU, TPU & NPU: What to Use for AI Workloads (2025 Guide)[ https://www.fluence.network/blog/cpu-gpu-tpu-npu-guide/](https://www.fluence.network/blog/cpu-gpu-tpu-npu-guide/)

\[5] CPU与GPU的算力演进:从串行控制到并行革命-腾讯云开发者社区-腾讯云[ https://cloud.tencent.com/developer/article/2556778](https://cloud.tencent.com/developer/article/2556778)

\[6] 为什么GPU可以实现并行运算，而CPU不行?\_为什么GPU比CPU更适合并行计算\_ - CSDN文库[ https://wenku.csdn.net/answer/2ow6ezgvcq](https://wenku.csdn.net/answer/2ow6ezgvcq)

\[7] 一文详解服务器芯片CPU与GPU的区别-51CTO.COM[ https://server.51cto.com/article/824522.html](https://server.51cto.com/article/824522.html)

\[8] 5.6 练习 - 最爱丁珰 - 博客园[ https://www.cnblogs.com/dingxingdi/p/18767525](https://www.cnblogs.com/dingxingdi/p/18767525)

\[9] GPU并行计算快速入门(一)------原理篇-CSDN博客[ https://blog.csdn.net/m0\_59012280/article/details/150921239](https://blog.csdn.net/m0_59012280/article/details/150921239)

\[10] 转载【AI系统】GPU 工作原理前面的文章对 AI 计算体系和 AI 芯片基础进行讲解，在 AI 芯片基础中关于通用图形 - 掘金[ https://juejin.cn/post/7447114867446087734](https://juejin.cn/post/7447114867446087734)

\[11] GPU和CPU的异同---ChatGPT 5 thinking作答-CSDN博客[ https://blog.csdn.net/qq\_46215223/article/details/151334738](https://blog.csdn.net/qq_46215223/article/details/151334738)

\[12] NVIDIA's Most Significant CUDA Update in Two Decades: Chip Expert Jim Keller Claims NVIDIA Is Eroding Its Competitive Edge[ https://www.c114pro.com/chipnews/131811.html](https://www.c114pro.com/chipnews/131811.html)

\[13] 显卡为何被归类为SIMD设备?\_编程语言-CSDN问答[ https://ask.csdn.net/questions/8563806](https://ask.csdn.net/questions/8563806)

\[14] L8 - GPU Architecture and Parallel Processing | Coconote[ https://coconote.app/notes/6eeed606-98bf-41d4-9551-86aef75bb16a](https://coconote.app/notes/6eeed606-98bf-41d4-9551-86aef75bb16a)

\[15] SIMD: Engine Behind High-Performance Computing[ https://tomorrowdesk.com/info/simd](https://tomorrowdesk.com/info/simd)

\[16] Высокопропускная память[ https://ru.wikipedia.org/wiki/High\_Bandwidth\_Memory](https://ru.wikipedia.org/wiki/High_Bandwidth_Memory)

\[17] What is the difference between HBM2 and GDDR6 memory in terms of deep learning performance?[ https://massedcompute.com/faq-answers/?question=What+is+the+difference+between+HBM2+and+GDDR6+memory+in+terms+of+deep+learning+performance%3F](https://massedcompute.com/faq-answers/?question=What+is+the+difference+between+HBM2+and+GDDR6+memory+in+terms+of+deep+learning+performance%3F)

\[18] What are the key differences between HBM and GDDR6 memory in terms of latency and throughput?[ https://massedcompute.com/faq-answers/?question=What+are+the+key+differences+between+HBM+and+GDDR6+memory+in+terms+of+latency+and+throughput%3F](https://massedcompute.com/faq-answers/?question=What+are+the+key+differences+between+HBM+and+GDDR6+memory+in+terms+of+latency+and+throughput%3F)

\[19] NVIDIA GPU SM(流式多处理器)详细介绍-CSDN博客[ https://blog.csdn.net/m0\_49133355/article/details/151284025](https://blog.csdn.net/m0_49133355/article/details/151284025)

\[20] Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era[ https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

\[21] GPU Performance Background User's Guide - NVIDIA Docs[ https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)

\[22] Primers • GPU Architecture[ https://aman.ai/primers/ai/gpu-architecture/](https://aman.ai/primers/ai/gpu-architecture/)

\[23] 1. GPU Architecture¶[ https://tvm.d2l.ai/chapter\_gpu\_schedules/arch.html](https://tvm.d2l.ai/chapter_gpu_schedules/arch.html)

\[24] Streaming multiprocessor[ https://docs.modular.com/glossary/gpu/streaming-multiprocessor/](https://docs.modular.com/glossary/gpu/streaming-multiprocessor/)

\[25] 搭载 NVIDIA Blackwell 架构的新型 GeForce RTX 50 系列显卡和笔记本电脑为游戏玩家和创作者带来 AI 和神经网络渲染助力的全新玩法 | GeForce 新闻 | NVIDIA[ https://www.nvidia.cn/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/](https://www.nvidia.cn/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)

\[26] AMD 发布新一代 AMD RDNA™ 4 架构，推出 AMD Radeon™ RX 9000 系列显卡[ https://www.amd.com/zh-cn/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html](https://www.amd.com/zh-cn/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html)

\[27] AMD ROCm 7.0: Built for Developers, Advancing Open Innovation[ https://www.amd.com/en/developer/resources/technical-articles/2025/amd-rocm-7-built-for-developers-ready-for-enterprises.html](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-rocm-7-built-for-developers-ready-for-enterprises.html)

\[28] AMD Unveils Next-Generation AMD RDNA™ 4 Architecture with the Launch of AMD Radeon™ RX 9000 Series Graphics Cards[ https://www.amd.com/en/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html?ref=news.itsfoss.com](https://www.amd.com/en/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html?ref=news.itsfoss.com)

\[29] Use ROCm on Radeon GPUs[ https://rocmdocs.amd.com/projects/radeon/en/latest/index.html](https://rocmdocs.amd.com/projects/radeon/en/latest/index.html)

\[30] AMD Introduces New Radeon Graphics Cards and Ryzen Threadripper Processors at COMPUTEX 2025[ https://www.amd.com/en/newsroom/press-releases/2025-5-20-amd-introduces-new-radeon-graphics-cards-and-ryzen.html](https://www.amd.com/en/newsroom/press-releases/2025-5-20-amd-introduces-new-radeon-graphics-cards-and-ryzen.html)

\[31] 为什么搞AI要用GPU而不是CPU?[ https://c.m.163.com/news/a/KE4L5GAF05168SG2.html](https://c.m.163.com/news/a/KE4L5GAF05168SG2.html)

\[32] GPUs vs CPUs for AI: Key Differences and Why It Matters for Your Workloads[ https://www.ai-infra-link.com/gpus-vs-cpus-for-ai-key-differences-and-why-it-matters-for-your-workloads/](https://www.ai-infra-link.com/gpus-vs-cpus-for-ai-key-differences-and-why-it-matters-for-your-workloads/)

\[33] Ollama CPU vs GPU Performance: Complete Benchmark Analysis 2025[ https://markaicode.com/ollama-cpu-vs-gpu-performance-benchmark-2025/](https://markaicode.com/ollama-cpu-vs-gpu-performance-benchmark-2025/)

\[34] CPU, GPU, TPU & NPU: What to Use for AI Workloads (2025 Guide)[ https://www.fluence.network/blog/cpu-gpu-tpu-npu-guide/](https://www.fluence.network/blog/cpu-gpu-tpu-npu-guide/)

\[35] How much faster is a GPU than a CPU? A comprehensive performance breakdown[ https://www.byteplus.com/en/topic/407608?title=how-much-faster-is-a-gpu-than-a-cpu-a-comprehensive-performance-breakdown](https://www.byteplus.com/en/topic/407608?title=how-much-faster-is-a-gpu-than-a-cpu-a-comprehensive-performance-breakdown)

\[36] GPU 在机器学习中的应用优势:从技术特性到云端赋能​引言:当机器学习遇见算力革命​ 在人工智能浪潮席卷全球的今天，机器 - 掘金[ https://juejin.cn/post/7494124948855062566](https://juejin.cn/post/7494124948855062566)

\[37] GPUs for Artificial Intelligence (AI) – Intel[ https://www.intel.cn/content/www/us/en/learn/gpu-for-ai.html](https://www.intel.cn/content/www/us/en/learn/gpu-for-ai.html)

\[38] GPU Evolution: What are the Key Roles of GPUs in AI and ML?[ https://acecloud.ai/blog/the-evolution-of-gpu/?nonamp=1](https://acecloud.ai/blog/the-evolution-of-gpu/?nonamp=1)

\[39] GPU が AI に最適な理由[ https://blogs.nvidia.co.jp/blog/why-gpus-are-great-for-ai/?nv\_excludes=10934%2C10938](https://blogs.nvidia.co.jp/blog/why-gpus-are-great-for-ai/?nv_excludes=10934%2C10938)

\[40] RTX4090显卡在机器学习中的实际表现-CSDN博客[ https://blog.csdn.net/weixin\_42168902/article/details/152081358](https://blog.csdn.net/weixin_42168902/article/details/152081358)

\[41] Flops Profiler[ https://www.deepspeed.ai/tutorials/flops-profiler/](https://www.deepspeed.ai/tutorials/flops-profiler/)

\[42] Measuring Max-Achievable FLOPs – Part 2[ https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html)

\[43] Tensor 核心:高效能运算和人工智慧的多样性 | NVIDIA[ https://www.nvidia.com/zh-tw/data-center/tensor-cores/?ncid=so-twit-877903%3Fncid](https://www.nvidia.com/zh-tw/data-center/tensor-cores/?ncid=so-twit-877903%3Fncid)

\[44] 深度学习GPU最全对比，到底谁才是性价比之王?-FreeNAS中文网[ https://freenas.com.cn/jishu/1725896064h889007.html](https://freenas.com.cn/jishu/1725896064h889007.html)

\[45] CNN vs Transformer for Sequence Data[ https://mljourney.com/cnn-vs-transformer-for-sequence-data-2/](https://mljourney.com/cnn-vs-transformer-for-sequence-data-2/)

\[46] 中文NLP数据\_footballboy的技术博客\_51CTO博客[ https://blog.51cto.com/u\_12196/13199370](https://blog.51cto.com/u_12196/13199370)

\[47] Transformer结构和传统深度学习架构的区别\_transformer与cnn rnn-CSDN博客[ https://blog.csdn.net/Q2024107/article/details/150422005](https://blog.csdn.net/Q2024107/article/details/150422005)

\[48] Comparative Analysis of CNN Performances Using CIFAR-100 and MNIST Databases: GPU vs. CPU Efficiency[ https://benthamscience.com/article/148842](https://benthamscience.com/article/148842)

\[49] 一文看懂AI大模型的并行训练方式(DP、PP、TP、EP)\_腾讯新闻[ http://news.qq.com/rain/a/20251128A03JP400](http://news.qq.com/rain/a/20251128A03JP400)

\[50] Pipeline Parallelism[ https://www.deepspeed.ai/tutorials/pipeline/](https://www.deepspeed.ai/tutorials/pipeline/)

\[51] How Parallel Processing with NVIDIA GPUs Enhances Computer Vision[ https://www.xenonstack.com/blog/parallel-processing-nvidia-gpus-computer-vision](https://www.xenonstack.com/blog/parallel-processing-nvidia-gpus-computer-vision)

\[52] 一文看懂 AI 大模型的并行训练方式(DP、PP、TP、EP)[ https://it.ithome.com/archiver/0/901/261.htm](https://it.ithome.com/archiver/0/901/261.htm)

\[53] 转载:【AI系统】Tensor Core 深度剖析Tensor Core 是用于加速深度学习计算的关键技术，其主要功能是 - 掘金[ https://juejin.cn/post/7447776289434812425](https://juejin.cn/post/7447776289434812425)

\[54] What is the role of Tensor Cores in accelerating AI inference tasks?[ https://massedcompute.com/faq-answers/?question=What+is+the+role+of+Tensor+Cores+in+accelerating+AI+inference+tasks%3F](https://massedcompute.com/faq-answers/?question=What+is+the+role+of+Tensor+Cores+in+accelerating+AI+inference+tasks%3F)

\[55] GPU 加速工具RAPIDS 在数据科学的应用\_mob64ca13f8b166的技术博客\_51CTO博客[ https://blog.51cto.com/u\_16213585/14198419](https://blog.51cto.com/u_16213585/14198419)

\[56] How to optimize training time and resource consumption in AI image processing? - Tencent Cloud[ https://www.tencentcloud.com/techpedia/125282](https://www.tencentcloud.com/techpedia/125282)

\[57] 70% GPU利用率提升:Deep Lake流式处理终结AI训练数据瓶颈-CSDN博客[ https://blog.csdn.net/gitblog\_00184/article/details/152063278](https://blog.csdn.net/gitblog_00184/article/details/152063278)

\[58] Accelerating Medical Image Processing with NVIDIA DALI[ https://developer.nvidia.com/blog/accelerating-medical-image-processing-with-dali](https://developer.nvidia.com/blog/accelerating-medical-image-processing-with-dali)

\[59] Efficient Data Loading in PyTorch: Tips and Tricks for Faster Training[ https://mljourney.com/efficient-data-loading-in-pytorch-tips-and-tricks-for-faster-training/](https://mljourney.com/efficient-data-loading-in-pytorch-tips-and-tricks-for-faster-training/)

\[60] 转载【AI系统】为什么 GPU 适用于 AI为什么 GPU 适用于 AI 计算或者为什么 AI 训练需要使用 GPU，而 - 掘金[ https://juejin.cn/post/7447134125390594102](https://juejin.cn/post/7447134125390594102)

\[61] GPU-Accelerated Convolution Overview[ https://www.emergentmind.com/topics/gpu-accelerated-convolution](https://www.emergentmind.com/topics/gpu-accelerated-convolution)

\[62] Performance Evaluation of cuDNN Convolution Algorithms on NVIDIA Volta GPUs[ https://impact.ornl.gov/en/publications/performance-evaluation-of-cudnn-convolution-algorithms-on-nvidia-/](https://impact.ornl.gov/en/publications/performance-evaluation-of-cudnn-convolution-algorithms-on-nvidia-/)

\[63] Performance Analysis of GPU-based Convolutional Neural Networks[ http://howie.seas.gwu.edu/publications/GPU-CNN-ICPP16.pdf](http://howie.seas.gwu.edu/publications/GPU-CNN-ICPP16.pdf)

\[64] ML Practicum: Image Classification[ https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks?authuser=2](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks?authuser=2)

\[65] cuConv: A CUDA Implementation of Convolution for CNN Inference[ https://arxiv.org/pdf/2103.16234](https://arxiv.org/pdf/2103.16234)

\[66] 🎉7.6倍训练加速与24倍吞吐提升:两项核心技术背后的大模型推理优化全景图​在大语言模型(LLM)的推理过程中，Att - 掘金[ https://juejin.cn/post/7551214653553721379](https://juejin.cn/post/7551214653553721379)

\[67] FlashAttention-3: GPU-Optimized Attention[ https://www.emergentmind.com/topics/flashattention-3](https://www.emergentmind.com/topics/flashattention-3)

\[68] Flash-Attention Kernels in Transformer Models[ https://www.emergentmind.com/topics/flash-attention-kernels](https://www.emergentmind.com/topics/flash-attention-kernels)

\[69] torch.Tensor[ https://glaringlee.github.io/tensors.html](https://glaringlee.github.io/tensors.html)

\[70] Source code for torch.cuda[ https://pytorch.ac.cn/docs/0.4.1/\_modules/torch/cuda.html](https://pytorch.ac.cn/docs/0.4.1/_modules/torch/cuda.html)

\[71] PyTorch 张量核心学习笔记-CSDN博客[ https://blog.csdn.net/Bruce\_Liu001/article/details/151142776](https://blog.csdn.net/Bruce_Liu001/article/details/151142776)

\[72] Automatic Mixed Precision package - torch.cuda.amp[ https://glaringlee.github.io/amp.html](https://glaringlee.github.io/amp.html)

\[73] 从训练到推理:Intel Extension for PyTorch混合精度优化完整指南\_51CTO博客\_混合精度训练 tensorflow[ https://blog.51cto.com/deephub/14204323](https://blog.51cto.com/deephub/14204323)

\[74] pytorch 同步机制\_pytorch stream-CSDN博客[ https://blog.csdn.net/2303\_77224751/article/details/142426250](https://blog.csdn.net/2303_77224751/article/details/142426250)

\[75] torch.cuda¶[ https://pytorch.ac.cn/docs/0.4.1/cuda.html](https://pytorch.ac.cn/docs/0.4.1/cuda.html)

\[76] GPU training (Intermediate)[ https://lightning.ai/docs/pytorch/2.5.1/accelerators/gpu\_intermediate.html](https://lightning.ai/docs/pytorch/2.5.1/accelerators/gpu_intermediate.html)

\[77] DDP Optimizations[ https://lightning.ai/docs/pytorch/stable/advanced/ddp\_optimizations.html](https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html)

\[78] Source code for lightning.pytorch.strategies.ddp[ https://lightning.ai/docs/pytorch/2.5.0/\_modules/lightning/pytorch/strategies/ddp.html](https://lightning.ai/docs/pytorch/2.5.0/_modules/lightning/pytorch/strategies/ddp.html)

\[79] DDPFullyShardedNativeStrategy[ https://lightning.ai/docs/pytorch/1.8.4/api/pytorch\_lightning.strategies.DDPFullyShardedNativeStrategy.html](https://lightning.ai/docs/pytorch/1.8.4/api/pytorch_lightning.strategies.DDPFullyShardedNativeStrategy.html)

\[80] Speed Up Model Training[ https://lightning.ai/docs/pytorch/latest/advanced/speed.html](https://lightning.ai/docs/pytorch/latest/advanced/speed.html)

\[81] Source code for pytorch\_lightning.plugins.training\_type.ddp[ https://lightning.ai/docs/pytorch/1.4.6/\_modules/pytorch\_lightning/plugins/training\_type/ddp.html](https://lightning.ai/docs/pytorch/1.4.6/_modules/pytorch_lightning/plugins/training_type/ddp.html)

\[82] Pytorch分布式训练最佳实践\_pytorch lightning dp和ddp-CSDN博客[ https://blog.csdn.net/sjtu\_wyy/article/details/149631582](https://blog.csdn.net/sjtu_wyy/article/details/149631582)

\[83] PyTorch 分布式训练底层原理与 DDP 实战指南-腾讯云开发者社区-腾讯云[ https://cloud.tencent.cn/developer/article/2593964](https://cloud.tencent.cn/developer/article/2593964)

\[84] Distributed communication package - torch.distributed¶[ https://pytorch.ac.cn/docs/0.4.1/distributed.html](https://pytorch.ac.cn/docs/0.4.1/distributed.html)

\[85] GTC 2025:Agentic AI 引爆 10 倍算力需求，市场格局如何颠覆?\_2025 gtc agentic ai-CSDN博客[ https://blog.csdn.net/suanlix/article/details/146426468](https://blog.csdn.net/suanlix/article/details/146426468)

\[86] NVIDIA DGX and the Future of AI Desktop Computing | IDC Blog[ https://blogs.idc.com/2025/05/19/nvidia-dgx-and-the-future-of-ai-desktop-computing/](https://blogs.idc.com/2025/05/19/nvidia-dgx-and-the-future-of-ai-desktop-computing/)

\[87] Tendencias en GPUs 2025 – IA y renderizado en tiempo real[ https://www.profesionalreview.com/2025/09/13/tendencias-en-gpus-2025/amp/](https://www.profesionalreview.com/2025/09/13/tendencias-en-gpus-2025/amp/)

\[88] The AI-Cloud Nexus[ https://www.linkedin.com/pulse/ai-cloud-nexus-why-gpus-new-gold-2025-edition-lab7ai-fsvaf](https://www.linkedin.com/pulse/ai-cloud-nexus-why-gpus-new-gold-2025-edition-lab7ai-fsvaf)

\[89] 算力格局新变局:当 AI 从 GPU 走向 TPU、NPU、ASIC\_为什么越来越多gpu.tpu-CSDN博客[ https://blog.csdn.net/aifs2025/article/details/153814907](https://blog.csdn.net/aifs2025/article/details/153814907)

\[90] NVIDIA’s Huge Risk: KAIST NPUs Are Better At AI Than GPUs[ https://tgdaily.com/technology/nvidias-huge-risk-kaist-npus-are-better-at-ai-than-gpus/](https://tgdaily.com/technology/nvidias-huge-risk-kaist-npus-are-better-at-ai-than-gpus/)

\[91] Google's TPU Revolution: The \$13 Billion Challenge to Nvidia's AI Chip Dominance[ https://www.alphamatch.ai/blog/google-tpu-nvidia-ai-chip-competition-2025](https://www.alphamatch.ai/blog/google-tpu-nvidia-ai-chip-competition-2025)

\[92] TPUs vs NPUs: Complete AI Hardware Comparison Guide 2025[ https://www.ofzenandcomputing.com/tpus-vs-npus-complete-ai-hardware-comparison-guide-cy/](https://www.ofzenandcomputing.com/tpus-vs-npus-complete-ai-hardware-comparison-guide-cy/)

\[93] Custom AI Chips Challenge NVIDIA's Dominance as Tech Giants Enter[ https://www.chosun.com/english/industry-en/2025/11/28/GY3M7EV2XZCJVJGVF7XG76UQRM/](https://www.chosun.com/english/industry-en/2025/11/28/GY3M7EV2XZCJVJGVF7XG76UQRM/)

\[94] TPU Vs GPU: 7 Critical Truths Defining The 2025 AI Chip War[ https://binaryverseai.com/tpu-vs-gpu-ai-hardware-war-guide-nvidia-google/](https://binaryverseai.com/tpu-vs-gpu-ai-hardware-war-guide-nvidia-google/)

\[95] 图文详情[ https://m.kepuchina.cn/tuwendetail?id=613978](https://m.kepuchina.cn/tuwendetail?id=613978)

\[96] HOWTO: Estimating and Profiling GPU Memory Usage for Generative AI[ https://www.osc.edu/resources/getting\_started/howto/howto\_estimating\_and\_profiling\_gpu\_memory\_usage\_for\_generative\_ai](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai)

\[97] 有限GPU显存下的大语言模型训练技术综述[ https://www.fitee.zjujournals.com/zh/article/doi/10.1631/FITEE.2300710/](https://www.fitee.zjujournals.com/zh/article/doi/10.1631/FITEE.2300710/)

\[98] 多年来英伟达的 AI 服务器功耗增长了 100 倍: AI 引发的能源危机升级\_万物云联网[ http://m.toutiao.com/group/7561603907743466035/?upstream\_biz=doubao](http://m.toutiao.com/group/7561603907743466035/?upstream_biz=doubao)

\[99] The AI Power Surge: What Every CIO, CTO and Executive Needs to Know About Infrastructure Readiness[ https://www.nextdc.com/blog/whats-next-for-neo-clouds-0](https://www.nextdc.com/blog/whats-next-for-neo-clouds-0)

\[100] How NVIDIA Blackwell GPUs Are Transforming Cloud Costs and Revenue[ https://www.cloudsyntrix.com/blogs/how-nvidia-blackwell-gpus-are-transforming-cloud-costs-and-revenue/](https://www.cloudsyntrix.com/blogs/how-nvidia-blackwell-gpus-are-transforming-cloud-costs-and-revenue/)

\[101] The Costs of Deploying AI: Energy, Cooling, & Management[ https://www.exxactcorp.com/blog/hpc/the-costs-of-deploying-ai-energy-cooling-management](https://www.exxactcorp.com/blog/hpc/the-costs-of-deploying-ai-energy-cooling-management)

\[102] CPU de 600 W y GPU de 700 W, Gigabyte revela como aumentara el consumo en servidores[ https://www.profesionalreview.com/2023/05/16/cpu-600-w-gpu-de-700-w-servidores/amp/](https://www.profesionalreview.com/2023/05/16/cpu-600-w-gpu-de-700-w-servidores/amp/)

\[103] 微软大量GPU，开始吃灰[ https://c.m.163.com/news/a/KEBSR595051487C6.html](https://c.m.163.com/news/a/KEBSR595051487C6.html)

> （注：文档部分内容可能由 AI 生成）