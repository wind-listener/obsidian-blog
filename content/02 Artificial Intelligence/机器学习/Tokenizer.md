---
title: "Tokenizer"
date: 2025-08-07
draft: false
---

在机器学习（尤其是自然语言处理，NLP）领域，**Tokenizer（分词器）** 是一个将文本数据转换为模型可以处理的数值形式的工具或过程。它是将自然语言处理问题转化为机器学习模型所需输入的关键步骤之一。

  

**Tokenizer 的核心作用**

  

Tokenizer 的主要作用是将**原始文本数据**转化为**模型可以理解的表示形式**。这一过程通常包括：

1. **将文本切分为基本单位**（如单词、子词或字符）。

2. **将这些单位映射为数值索引**（token IDs），供模型使用。

  

通过这种转化，机器学习模型（如 Transformer、BERT、GPT 等）可以接收数值输入并进行训练或推理。

  

**Token 的定义**

  

在 NLP 中，**Token** 是对文本的基本单位表示，可以是：

• **单词**（如 ['I', 'love', 'learning']）。

• **子词**（如 ['I', 'lov', 'ing']，子词拆分的原因见下文）。

• **字符**（如 ['I', ' ', 'l', 'o', 'v', 'e']）。

• **特殊符号**（如 [CLS], [SEP]，通常用于表示句子的开头或分隔）。

  

**Tokenizer 的主要步骤**

1. **文本标准化**：

• 清理或标准化原始文本数据，例如：

• 转换为小写（I Love You → i love you）。

• 去除多余空格或特殊符号。

• 替换缩写（can't → cannot）。

2. **分词（Tokenization）**：

• 将文本拆分为较小的单位（即 token）。不同的分词方式如下：

• **基于单词的分词**：按空格或标点分隔，如 I love NLP → ['I', 'love', 'NLP']。

• **基于子词的分词**（Subword Tokenization）：将单词拆分为更小的子单位，如 playing → ['play', 'ing']。

• **基于字符的分词**：逐字符分割，如 hello → ['h', 'e', 'l', 'l', 'o']。

3. **映射为索引（Token IDs）**：

• 将每个 token 映射到唯一的数值 ID。例如，['I', 'love', 'NLP'] 可能映射为 [12, 45, 98]。

4. **添加特殊 token**：

• 在一些模型中，需要添加特殊 token。例如：

• [CLS] 表示序列的起始。

• [SEP] 表示句子之间的分隔符。

• [PAD] 用于补齐长度不足的序列。

5. **处理序列长度**：

• 长度不足的序列会用 [PAD] 填充到固定长度。

• 长度超出限制的序列会被截断。

  

**常见分词方法**

  

**1. 基于规则的分词**

• **方法**：通过空格、标点等规则直接切分文本。

• **优点**：简单易实现。

• **缺点**：对多语言支持较差，可能导致 OOV（Out of Vocabulary，词表外词汇）问题。

  

**2. 基于词典的分词**

• **方法**：预定义一个词典，将句子中的词与词典匹配。

• **优点**：提高了分词质量。

• **缺点**：无法处理未登录词（词典中不存在的词）。

  

**3. 子词分词（Subword Tokenization）**

• **方法**：将文本切分为常见的子词单元，而不是完整的单词。

• **常见算法**：

• **Byte Pair Encoding (BPE)**：

• 将罕见单词拆分为更频繁的子词（例如，playing → ['play', 'ing']）。

• **WordPiece**：

• 类似 BPE，但更注重概率和频率，用于 BERT。

• **Unigram**：

• 基于概率建模，允许多个可能的切分，常用于 SentencePiece。

• **优点**：

• 减少 OOV 问题（词表外的词可以用子词表示）。

• 词表规模更小，便于处理多语言。

• **缺点**：

• 生成的 token 数量可能比基于单词的分词更多。

  

**4. 基于字符的分词**

• **方法**：将每个字符视为一个 token。

• **优点**：

• 完全避免 OOV 问题。

• 模型能更好地处理拼写错误或未登录词。

• **缺点**：

• 序列长度显著增加，模型计算量更大。

  

**常见分词器（Tokenizer）工具**

1. **Hugging Face Tokenizer**:

• 为流行的 Transformer 模型（如 BERT、GPT）提供高效的分词器。

• 支持多种分词算法（如 WordPiece、BPE、Unigram）。

• 示例代码：

  

from transformers import AutoTokenizer

  

# 加载 BERT 分词器

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  

# 对句子分词并编码

tokens = tokenizer("I love NLP!", padding=True, truncation=True, return_tensors="pt")

print(tokens)

  

  

2. **SentencePiece**:

• 支持子词分词（如 BPE、Unigram）。

• 特点是无需空格分词，适合多语言模型。

• 示例代码：

  

import sentencepiece as spm

  

sp = spm.SentencePieceProcessor(model_file='model.spm')

print(sp.encode('I love NLP!', out_type=str))

  

  

3. **SpaCy 和 NLTK**:

• SpaCy 和 NLTK 是经典的 NLP 工具库，支持规则分词和词性标注等功能。

• 示例代码（使用 SpaCy）：

  

import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("I love NLP!")

print([token.text for token in doc])

  

**Tokenization 在机器学习中的应用**

1. **NLP 模型的输入**：

• Tokenizer 将原始文本数据转为数值表示，供模型（如 BERT、GPT）输入。

2. **词表构建**：

• Tokenizer 会生成一个词表（Vocabulary），为文本中的每个 token 分配唯一的 ID。

3. **适配多语言场景**：

• 使用子词分词的 Tokenizer（如 BERT 和 GPT-3）可以有效处理多语言问题。

4. **处理未登录词**：

• 子词分词能够将复杂的单词拆解为更基础的子单元，解决 OOV 问题。

  

**总结**

• **Tokenizer 的概念**：将文本拆分为 Token，并映射为模型可以理解的数值表示。

• **重要性**：分词是 NLP 中预处理的关键步骤，直接影响模型的效果和性能。

• **选择合适的 Tokenizer**：根据任务需求和模型类型选择合适的分词器（如规则分词、子词分词）。

• **现代趋势**：子词分词（BPE、WordPiece、Unigram）已经成为主流，尤其在多语言和深度学习模型中表现出色。