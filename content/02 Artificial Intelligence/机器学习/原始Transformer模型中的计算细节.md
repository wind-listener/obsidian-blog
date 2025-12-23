在Transformer模型中，输入的流动过程可分为**编码器输入处理→编码器堆叠计算→解码器输入处理→解码器堆叠计算→输出预测**五个核心阶段，每个阶段涉及具体的矩阵操作和维度变换，细节如下：


### 一、编码器输入处理：嵌入（Embedding）与位置编码（Positional Encoding）
输入为源语言序列的token（如单词或子词），首先经过嵌入层和位置编码，转换为包含语义和位置信息的向量。

1. **嵌入层（Embedding Layer）**  
   - 目的：将离散token映射为连续向量。  
   - 操作：设输入token序列长度为$n$，每个token通过嵌入矩阵$W_{embed} \in \mathbb{R}^{V \times d_{model}}$（$V$为词汇表大小，$d_{model}=512$）进行映射，得到嵌入矩阵$E \in \mathbb{R}^{n \times d_{model}}$，即：  
     $E = \text{token\_indices} \cdot W_{embed}$  
   - 缩放：嵌入结果需乘以$\sqrt{d_{model}}$，即$E_{\text{scaled}} = E \times \sqrt{d_{model}}$。  

2. **位置编码（Positional Encoding）**  
   - 目的：注入token的位置信息（因模型无循环/卷积，需显式编码位置）。  
   - 操作：使用正弦余弦函数生成位置编码矩阵$PE \in \mathbb{R}^{n \times d_{model}}$，公式为：  
     $PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d_{model}}\right)$  
     $PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d_{model}}\right)$  
     （$pos$为位置索引，$i$为维度索引）。  
   - 相加：位置编码与嵌入向量逐元素相加，得到编码器输入$X_{\text{enc}} = E_{\text{scaled}} + PE$，维度仍为$n \times d_{model}$。  


### 二、编码器堆叠计算：6层编码器层的迭代处理
编码器由6个相同的“编码器层”堆叠而成（$N=6$），每层输入为上一层输出，初始输入为$X_{\text{enc}}$。每个编码器层包含两个子层：**多头自注意力**和**位置-wise前馈网络**，每个子层后均有残差连接和层归一化。

#### 2.1 编码器层第1子层：多头自注意力（Multi-Head Self-Attention）
输入为当前层的输入向量$X \in \mathbb{R}^{n \times d_{model}}$，输出为注意力加权后的向量。  

- **步骤1：线性投影（生成多头Q、K、V）**  
  对$X$分别通过$h=8$个独立的线性投影矩阵生成查询（Q）、键（K）、值（V）：  
  - 第$i$个头（$i=1,...,8$）的投影：  
    $Q_i = X \cdot W_i^Q$，其中$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$（$d_k=64$），$Q_i \in \mathbb{R}^{n \times d_k}$  
    $K_i = X \cdot W_i^K$，其中$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$（$d_k=64$）， $K_i \in \mathbb{R}^{n \times d_k}$  
    $V_i = X \cdot W_i^V$，其中$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$（$d_v=64$），$V_i \in \mathbb{R}^{n \times d_v}$

- **步骤2：缩放点积注意力（Scaled Dot-Product Attention）**  
  每个头独立计算注意力输出：  
  $head_i = \text{softmax}\left(\frac{Q_i \cdot K_i^T}{\sqrt{d_k}}\right) \cdot V_i$  
  - 细节：$Q_i \cdot K_i^T$为$n \times n$的相似度矩阵（每行对应一个位置的查询与所有键的点积）；除以$\sqrt{d_k}$避免数值过大；softmax后得到$n \times n$的注意力权重矩阵；与$V_i$相乘得到$head_i \in \mathbb{R}^{n \times d_v}$

- **步骤3：拼接（Concatenation）**  
  将8个头的输出沿最后一维拼接，得到$Concat(head_1,...,head_8) \in \mathbb{R}^{n \times (h \cdot d_v)}$（$8 \times 64=512$，与$d_{model}$一致）

- **步骤4：最终投影**  
  通过输出投影矩阵$W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}}$映射回$d_{model}$维度：  
  $Attention_{\text{output}} = Concat(head_1,...,head_8) \cdot W^O$，维度为$(8\times64)\times d_{model}=512*512$

- **残差连接与层归一化**  
  自注意力输出与输入$X$相加后进行层归一化：  
  $X_1 = \text{LayerNorm}(X + Attention_{\text{output}})$，维度仍为$n \times d_{model}$


#### 2.2 编码器层第2子层：位置-wise前馈网络（Feed-Forward Network）
输入为$X_1$，输出为经过非线性变换的向量。  
- 操作：包含两次线性变换和ReLU激活，公式为：  
  $FFN(X_1) = \max(0, X_1 \cdot W_1 + b_1) \cdot W_2 + b_2$  
  - 细节：$W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$（$d_{ff}=2048$），第一次变换后维度为$n \times 2048$；ReLU激活（$\max(0, \cdot)$）；$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，第二次变换后维度回到$n \times 512$。  

- **残差连接与层归一化**  
  前馈网络输出与$X_1$相加后归一化，得到当前编码器层的输出：  
  $X_{\text{enc\_layer}} = \text{LayerNorm}(X_1 + FFN(X_1))$，维度$n \times d_{model}$。  


#### 2.3 编码器堆叠输出
6层编码器层依次处理，最终输出编码器全局特征$Z \in \mathbb{R}^{n \times d_{model}}$（$Z$为第6层编码器层的输出）。  


### 三、解码器输入处理：与编码器类似，但需“右移”
解码器输入为目标语言序列的token（如翻译任务中的目标端输入），处理流程与编码器输入类似，但需额外“右移”（shifted right）以实现自回归生成。

1. **嵌入与位置编码**  
   - 目标token序列经嵌入层（与编码器共享嵌入矩阵$W_{embed}$）和位置编码，得到$E_{\text{dec}} \in \mathbb{R}^{m \times d_{model}}$（$m$为目标序列长度），同样满足$E_{\text{dec\_scaled}} = E_{\text{dec}} \times \sqrt{d_{model}} + PE$。  

2. **右移（Shifted Right）**  
   目标序列输入解码器前需整体右移一位（如“我爱你”→“-pad- 我 爱”），确保生成第$i$个token时仅依赖前 $i-1$个已生成token。  


### 四、解码器堆叠计算：6层解码器层的迭代处理
解码器由6个相同的“解码器层”堆叠而成，每层输入为上一层输出，初始输入为右移后的目标序列向量$X_{\text{dec}}$。每个解码器层包含三个子层：**带掩码的多头自注意力**、**编码器-解码器注意力**、**位置-wise前馈网络**，均带残差连接和层归一化。

#### 4.1 解码器层第1子层：带掩码的多头自注意力（Masked Multi-Head Self-Attention）
输入为当前层输入$Y \in \mathbb{R}^{m \times d_{model}}$，作用是确保生成时不依赖未来位置的token。  
- 操作：与编码器的多头自注意力流程完全一致（线性投影→缩放点积→拼接→投影），但在计算softmax前需加入“掩码”：  
  - 掩码：将$Q_i \cdot K_i^T$矩阵中“未来位置”（行索引<列索引）的元素设为$-\infty$，使softmax后这些位置的权重为0，即：  
    $masked\_scores = Q_i \cdot K_i^T + \text{mask}$（$\text{mask}$为下三角矩阵，上三角元素为$-\infty$）  
    $head_i = \text{softmax}(masked\_scores / \sqrt{d_k}) \cdot V_i$  

- 残差连接与层归一化：  
  $Y_1 = \text{LayerNorm}(Y + Masked\_Attention_{\text{output}})$，维度$m \times d_{model}$。  


#### 4.2 解码器层第2子层：编码器-解码器注意力（Encoder-Decoder Attention）
输入为$Y_1$，作用是让解码器关注编码器输出的源语言信息。  
- 操作：多头注意力的变体，其中：  
  - 查询（Q）来自解码器第1子层的输出$Y_1$；  
  - 键（K）和值（V）来自编码器的最终输出$Z$；  
  - 其余流程（投影→点积→拼接→投影）与多头自注意力一致，输出$Attention_{\text{enc-dec}} \in \mathbb{R}^{m \times d_{model}}$。  

- 残差连接与层归一化：  
  $Y_2 = \text{LayerNorm}(Y_1 + Attention_{\text{enc-dec}})$，维度$m \times d_{model}$。  


#### 4.3 解码器层第3子层：位置-wise前馈网络
与编码器的前馈网络完全一致，输入$Y_2$，输出：  
$FFN_{\text{output}} = \max(0, Y_2 \cdot W_1 + b_1) \cdot W_2 + b_2$（维度$m \times d_{model}$）。  

- 残差连接与层归一化：  
  $Y_{\text{dec\_layer}} = \text{LayerNorm}(Y_2 + FFN_{\text{output}})$，维度$m \times d_{model}$


#### 4.4 解码器堆叠输出
6层解码器层依次处理，最终输出$Y_{\text{final}} \in \mathbb{R}^{m \times d_{model}}$（第6层解码器层的输出）。  


### 五、输出预测：线性变换与Softmax
解码器最终输出$Y_{\text{final}}$经过线性变换和softmax，得到每个位置的token概率分布。  
- 线性变换：使用与嵌入层共享的权重矩阵$W_{embed}^T$（转置），将$d_{model}$维度映射到词汇表大小$V$：  
  $Logits = Y_{\text{final}} \cdot W_{embed}^T$，维度$m \times V$。  
- Softmax：对$Logits$的每行进行softmax，得到每个位置的token概率：  
  $P(\text{token}) = \text{softmax}(Logits)$，维度$m \times V$。  


### 总结：输入流动全链路
源语言token→嵌入+位置编码→编码器6层（自注意力+前馈）→输出$Z$；  
目标语言token（右移）→嵌入+位置编码→解码器6层（掩码自注意力+编码器-解码器注意力+前馈）→输出$Y_{\text{final}}$→线性变换+softmax→token概率。  

每一步的矩阵操作均严格遵循文档中定义的维度（$d_{model}=512$，$h=8$，$d_k=d_v=64$，$d_{ff}=2048$）和公式，确保信息流的维度一致性和语义关联性。