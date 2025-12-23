

> 让小模型先跑一会，大模型来收尾，推理速度提升2-3倍不再是梦想。

在大语言模型（LLM）席卷各行各业的今天，一个严峻的挑战摆在面前：**模型推理速度严重跟不上实际应用需求**。传统的自回归解码方式要求模型逐个生成token，如同单线程作业，造成了严重的计算瓶颈。而 Speculative Decoding（推测解码）技术的出现，正在从根本上改变这一局面。

## 技术概述：什么是 Speculative Decoding？

Speculative Decoding（推测解码）是一种创新的推理加速技术，其核心思想是 **"小模型快速推测 + 大模型验证修正"**。它通过协同利用大小两个模型，在不改变目标模型输出分布的前提下，显著减少推理过程中的计算开销。

与传统自回归模型逐个生成token的方式不同，推测解码采用两阶段管道：首先使用一个更小、更快的"草稿模型"（draft model）快速生成一段候选token序列；然后由"目标模型"（target model）一次性并行验证这些候选token的正确性。

这种设计的巧妙之处在于，它充分利用了大小模型的特点：小模型计算速度快但能力有限；大模型能力强大但计算昂贵。让适合的工作交给适合的模型执行，从而达到整体效率的最优。

## 发展历程：从萌芽到成熟

推测解码的思想并非一蹴而就。其技术渊源可以追溯到更早期的研究：

**早期探索**（2022年前）：相关思想在分块解码（Blockwise Decoding）等技术中已有体现，研究者尝试通过集成额外的前馈神经网络头，实现单步生成多个token。

**关键突破**（2022年）：Leviathan等人在论文《Fast Inference from Transformers via Speculative Decoding》中正式提出了推测解码算法，并引入了**投机采样**（Speculative Sampling）这一无损加速技术，为后续研究奠定了基础。

**快速发展**（2023年至今）：随着大模型应用的爆发式增长，推测解码技术迎来了一系列改进和扩展。不仅出现了基于N-gram的轻量级替代方案，还涌现了**自推测解码**（Self-Speculative Decoding）等无需单独草稿模型的变体。

如今，推测解码已成为大模型推理加速的标准技术之一，被广泛应用于各类推理框架中，如vLLM、OpenVINO™ GenAI等。

## 核心原理：工作机制全解析

### 基本工作流程

推测解码的执行过程包含六个核心步骤：

**步骤1：初始化与输入准备**
确定目标模型（大模型）和草稿模型（小模型），给定输入序列 $x=[x_1,x_2,...,x_n]$，初始化输出序列 $y$ 为空。

**步骤2：草稿模型生成候选序列**
草稿模型基于当前输入，自回归地生成一段长度为 $k$ 的候选序列 $\hat{y} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_k]$。这里的 $k$ 通常为2-10个token。

**步骤3：目标模型并行验证**
将初始输入 $x$ 与候选序列 $\hat{y}$ 拼接，目标模型一次性并行计算每个位置的条件概率：
$$P_{\text{目标}}(\hat{y}_i | x + \hat{y}_1...\hat{y}_{i-1})$$

**步骤4：基于接受规则筛选**
对每个候选token $\hat{y}_i$，计算接受概率比：
$$r_i = \frac{P_{\text{目标}}(\hat{y}_i | ...)}{P_{\text{草稿}}(\hat{y}_i | ...)}$$

如果 $r_i \geq 1$（或预设阈值如0.9），则接受该token；否则拒绝。

**步骤5：修正并更新输出序列**
找到第一个被拒绝的token位置 $m$，接受前 $m-1$ 个token，被拒绝的token由目标模型重新生成。

**步骤6：迭代至生成结束**
重复上述过程，直到生成结束符或达到最大长度。

### 数学基础与接受规则

推测解码的数学优雅性体现在其接受规则上。当 $r_i \geq 1$ 时，表示目标模型认为候选token至少与草稿模型一样可能，因此接受是合理的。理论分析表明，当阈值设置为1时，推测解码可以**严格保证**输出分布与目标模型完全一致。

接受规则的核心是平衡加速比与准确性。更宽松的阈值（如0.8）可能带来更高的加速效果，但会轻微偏离目标模型的分布；而严格的阈值（1.0）则保证输出无损。

### 一个简单示例

考虑词表为{A,B,C}，输入为"今天天气"：
- 草稿模型生成：$\hat{y} = [\text{"很"}, \text{"好"}]$
- 目标模型验证：
  - 位置1：$P_{\text{目标}}(\text{"很"}) = 0.4$, $P_{\text{草稿}}(\text{"很"}) = 0.6$，$r_1 = 0.4/0.6 ≈ 0.67$
  - 位置2：$P_{\text{目标}}(\text{"好"}) = 0.5$, $P_{\text{草稿}}(\text{"好"}) = 0.3$，$r_2 = 0.5/0.3 ≈ 1.67$

假设第一个token被接受，第二个token肯定被接受（$r_2 > 1$），则本轮生成两个有效token，加速成功。

## 实用实现指南

### 在vLLM中使用Speculative Decoding

vLLM从0.3版本开始支持推测解码，使用极为简便。以下是一个示例命令：

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8006 \
    --model /data/models/Qwen/Qwen3-8B/ \
    --served-model-name qwen3 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.6 \
    -tp 1 \
    --max-model-len 6000 \
    --speculative_config '{"model": "[ngram]", "num_speculative_tokens": 5}'
```

此配置使用N-gram模型作为草稿模型，每次推测5个token。实测显示，在相同任务下，推理时间从22秒减少到6.5秒，加速比达到3.4倍。

vLLM也支持使用小模型作为草稿模型：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-chat-hf \
    --speculative_model mistralai/Mistral-7B-Instruct-v0.2
```

### 基于OpenVINO™的实现

OpenVINO™ GenAI提供了完整的推测解码实现方案。以下是一个Python示例：

```python
# 硬件设备配置
main_device = 'CPU'  # 也可使用'GPU'
draft_device = 'CPU'

# 调度器配置
scheduler_config = openvino_genai.SchedulerConfig()
scheduler_config.cache_size = 2
scheduler_config.num_assistant_tokens = 5

# 初始化模型
draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device)
pipe = openvino_genai.LLMPipeline(
    args.model_dir, main_device, scheduler_config=scheduler_config, draft_model=draft_model
)

# 生成配置
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 5

# 执行生成
pipe.generate(args.prompt, config, streamer)
```

关键参数说明：
- `num_assistant_tokens`：草稿模型每次迭代生成的候选token数量
- `cache_size`：缓存大小，影响内存使用和性能
- `assistant_confidence_threshold`：可选置信度阈值，用于早期接受

## 进阶扩展与最新进展

### 自推测解码(Self-Speculative Decoding)

自推测解码是一种无需独立草稿模型的变体，它通过目标模型自身的某些机制生成候选序列，主要方法包括：

**层跳过（Layer Skipping）**：在草稿生成过程中选择性跳过部分中间层，快速生成草稿token，如Draft&Verify、SWIFT等方法。

**依赖头（Dependent Heads）**：在目标模型的隐藏状态上添加轻量级预测头，直接生成后续token，如EAGLE、Hydra等方法。

### 多Token预测(Multi-Token Prediction)

多Token预测（MTP）是一种训练时技术，让模型同时预测多个未来token，与推测解码天然互补。如DeepSeek-V2中采用的MTP目标，既能提升模型性能，又能为推测解码提供更好的基础。

Medusa提出了一种参数高效方案，在预训练语言模型上微调轻量级解码头，每个头专门预测序列中特定未来位置。

### 基于检索的方法

LLMA算法利用语言模型输出与参考文档之间的重叠来加速推理。它从参考文档中检索匹配片段作为候选序列，然后进行并行验证。

Speculative RAG通过微调的专家语言模型生成完整答案草稿，并通过聚类检索到的文档生成多样化草稿，使用自一致性评分代替逐词验证。

## 应用场景与最佳实践

### 适用场景分析

推测解码在以下场景中表现优异：

**长文本生成任务**：如文章创作、代码生成等，序列长度越长，加速效果越明显。

**实时交互应用**：聊天机器人、虚拟助手等对响应速度敏感的场景。

**简单推理任务**：内容提取、文本总结等任务中，草稿模型的准确率较高。

**资源受限环境**：边缘设备等计算资源有限但需要大模型能力的场景。

### 性能优化经验

**草稿模型选择**：理想情况下，草稿模型应与目标模型**tokenizer兼容**，且参数量约为目标模型的1/10-1/3。例如，对8B目标模型，可选择0.6B-3B的草稿模型。

**推测长度调优**：推测长度 $k$ 影响加速效果，通常取3-7为宜。过小则加速效果有限，过大则接受率下降。

**内存优化**：合理设置GPU内存使用率（如0.6），避免内存溢出。

实测数据显示，在相同prompt下，Qwen3-8B使用N-gram草稿模型时，推理时间从22秒降至6.5秒（加速3.4倍）；使用Qwen3-1.5B作为草稿模型时，降至13秒（加速1.7倍）。

## 未来展望与挑战

尽管推测解码技术已显示出巨大潜力，但仍面临一些挑战和发展机遇：

**精度与效率的权衡**：如何在不牺牲输出质量的前提下进一步提升加速比，是需要持续探索的问题。

**多模态扩展**：将推测解码应用于图像生成、语音合成等多模态任务，是未来的重要方向。

**系统级优化**：如何在大规模分布式系统中高效部署推测解码，需要进一步的工程优化。

**理论框架统一**：建立跨模态的统一理论框架，更好地平衡并行性和输出质量。

随着技术的不断发展，推测解码有望成为大模型推理的标准配置，为AI应用提供更强大的实时支持。

## 总结

Speculative Decoding通过巧妙地协同利用大小模型，打破了传统自回归解码的顺序依赖瓶颈，在不改变输出质量的前提下实现了显著的推理加速。随着相关技术的成熟和生态的完善，这一技术有望进一步降低大模型的部署门槛，推动AI技术在更多实时场景中的落地应用。

对于希望优化大模型推理性能的实践者来说，掌握并应用Speculative Decoding技术，将成为提升系统竞争力的关键一环。

---

*参考文献与推荐阅读*：

1. https://blog.csdn.net/huanxingchen1/article/details/153689820
2. https://arxiv.org/abs/2211.17192
3. https://arxiv.org/pdf/2401.07851.pdf
4. https://github.com/vllm-project/vllm
5. https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text_generation/prompt_lookup_decoding_lm.py