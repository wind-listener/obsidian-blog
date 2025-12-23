> 23年5月的工作 **论文精读笔记 - Gorilla: Large Language Model Connected with Massive APIs**

# 问题

**LLMs的API调用能力没有完全开发**
- 原来一些手工的小的API，文档比较完备也可以直接放入context，实现编写API call
- 但是“Supporting a web scale collection of potentially millions of changing APIs requires rethinking our approach to how we integrate tools.” (Patil 等, 2023, p. 2)，另外，很多API功能和内容有重叠，同时又有细微的差别和限制

# 贡献
1. 提出模型   “Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls.“
2. 发布数据集  “we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs.”

# 方法
	1. self-instruct fine-tuning
	2. retrieval
下面详细介绍
## 自我指令微调

## 检索器
1. 在训练过程中，将检索到的API文档附加到用户提示后，训练模型解析第二部分的文档来回答第一部分的问题。
2. 在模型推理过程中，使用信息检索器（如BM25或GPT-Index）从API文档数据库中检索最相关的API文档。 检索到的文档与用户提示一起输入模型，帮助模型生成更准确的API调用。

**优点**：
- 使模型能够适应测试时API文档的变化。
- 减少模型在生成API调用时的幻觉错误（即生成不存在或错误的API调用）。
- 提高模型在使用API文档时的准确性和实用性。

# Function Call 能力突出的LLM：Gorilla
![[Pasted image 20240517163834.png]] 

# 数据集 APIBench
## 构造
### API Documentation
- HuggingFace, 925 models
    每个领域前20个模型
    we pick the top 20 models from each domain. We consider 7 domains in multimodal data, 8 in CV, 12 in NLP, 5 in Audio, 2 in tabular data, and 2 in reinforcement learning.
- Torch Hub，94 API calls  95 models
    
- TensoFlow Hub ，696 API calls   626 models

### Instruction Generation #不懂 
16,45 {instruction,API} pairs
Instruction Generation Guided by the self-instruct paradigm [42], we employed GPT-4 to generate synthetic instruction data. We provided three in-context examples, along with a reference API documentation, and tasked the model with generating real-world use cases that call upon the API. We specifically instructed the model to refrain from using any API names or hints when creating instructions. We constructed six examples (Instruction-API pairs) for each of the three model hubs. These 18 points, were the only hand-generated or modified data. For each of our 1,645 API datapoints, we sample 3 of 6 corresponding instruction examples to generate a total of 10 instruction-api pairs as demonstrated in Figure 3. We would like to highlight that we only need to employ GPT-4 to generate the instructions and this can be swapped with open-source alternatives such as LLaMA, Alpaca, etc.
## 评价正确性的方法：AST sub-tree matching 

we employ self-instruct to generate {instruction, API} pairs.

关键
- **API Call with Constraints 训练**：Specifically, for machine learning API calls, two common sets of constraints are: parameter size and a lower bound on accuracy. *“Invoke an image classification model that uses less than 10M parameters, but maintains an ImageNet accuracy of at least 70%”.*
- **Retriever-Aware training 加上检索**：user prompt + *"Use this API documentation for reference: <retrieved_API_doc_JSON>"*
	- 被证明有好处
	- 但是加上retriever，有时会降低性能
- **Gorilla Inference 推理**：两种推理模式——zero-shot/with retriever

验证API调用脚本
**AST tree-matching strategy** 判断数据库中的哪个API是LLM调用的








---

## 1. 引言

### 背景
- **大型语言模型（LLMs）**：在自然对话、数学推理、程序合成等任务上表现出色，但在通过API调用使用工具的能力仍未得到充分发挥。
- **问题**：现有LLMs（如GPT-4）难以生成准确的输入参数，并且容易在API调用上产生幻觉。

### 研究目标
- 提出 **Gorilla**，一个基于LLaMA微调的模型，在API调用上超越GPT-4。
- 通过结合文档检索，Gorilla能够适应测试时的文档变化，减少幻觉问题。
- 引入 **APIBench**，一个包含大量API调用的数据集，用于评估模型能力。

---

## 2. 相关工作

### 大型语言模型
- 近年来，LLMs在多个领域取得显著进展，但仍受限于固定权重和静态计算图。
- 通过工具使用可以扩大LLMs的知识库和计算任务。

### 工具使用
- 现有工作大多集中在特定工具上，而Gorilla探索广泛的API调用，覆盖多个应用领域。

### 程序合成
- 现有策略包括上下文学习、任务分解和自我调试等，Gorilla则专注于通过API调用的线性程序合成。

---

## 3. 方法论

### 数据集构建
- **APIBench**：从TorchHub、TensorHub和HuggingFace中收集API模型卡片，构建数据集。
- **数据集规模**：包含1645个API调用和16450个指令-API对。

### 模型训练
- **Gorilla**：一个微调的LLaMA-7B模型，通过自我指令生成数据集进行训练。
- **API调用约束**：训练时考虑API调用的功能性和约束条件。

### 评价指标
- 使用 **AST子树匹配** 技术评估生成的API调用的功能正确性和幻觉问题。

---

## 4. 评估

### 模型对比
- **基线模型**：GPT-4、GPT-3.5-turbo、Claude、LLaMA-7B。
- **检索器**：BM25、GPT-Index和Oracle检索器。
- **实验结果**：Gorilla在TorchHub、HuggingFace和TensorHub数据集上的API功能性准确率和幻觉错误上均优于基线模型。

### 测试时文档变化适应性
- **Gorilla** 能够有效适应API文档的变化，保持模型的实用性和准确性。

### 约束条件下的API调用
- **Gorilla** 在处理带有约束条件的API调用任务上表现优异，能在准确率和参数数量等约束条件下选择合适的API。

---

## 5. 结论
- **Gorilla** 提升了LLMs在API调用上的性能，减少了幻觉问题，并能适应测试时文档的变化，显示了LLMs在实际应用中的巨大潜力。

---

## 6. 限制与社会影响
- **数据集局限**：集中于ML领域的API，可能会导致对某些子群体的不公平预测。
- **未来工作**：发布数据集以促进社区对API的研究，推动机器学习的公平使用。

---

## 7. 致谢
- 本研究得到了UC Berkeley Sky Computing Lab及多家公司的支持。

---

**引用**
Patil, Shishir G., et al. "Gorilla: Large Language Model Connected with Massive APIs." UC Berkeley, Microsoft Research, 2023.






**笔记总结**
1. **研究动机**：提升LLMs在API调用上的准确性和适应性。
2. **核心贡献**：提出Gorilla模型和APIBench数据集，验证了检索增强训练方法的有效性。
3. **实验结果**：Gorilla在多个基准数据集上超越现有最先进的LLMs。
4. **未来展望**：通过发布数据集推动社区研究，解决机器学习中的公平性问题。