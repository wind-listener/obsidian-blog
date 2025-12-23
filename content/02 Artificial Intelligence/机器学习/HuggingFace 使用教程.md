

## 下载模型文件


## 下载开放数据集



要指定 `UCSC-VLAA/HQ-Edit` 数据集的下载路径为 `/workspace/ckpt_downstream/zzm/mydataset/image_edit`，你有两种主要方法：

### 方法一：使用 `cache_dir` 参数（推荐）
这是最直接的方法，在调用 `load_dataset` 时指定缓存目录：

```python
from datasets import load_dataset

# 指定自定义下载路径
custom_path = "/workspace/ckpt_downstream/zzm/mydataset/image_edit"

# 加载数据集到指定路径
dataset = load_dataset(
    "UCSC-VLAA/HQ-Edit",
    cache_dir=custom_path,  # 关键参数：指定下载路径
    # 可选：只下载特定split以节省空间
    # split="train"  
)

print(f"数据集已下载到: {custom_path}")
```

### 方法二：设置环境变量（全局设置）
如果你想为所有 Hugging Face 数据集设置默认下载路径：

```python
import os
from datasets import load_dataset

# 设置环境变量（在加载数据集前设置）
os.environ["HF_DATASETS_CACHE"] = "/workspace/ckpt_downstream/zzm/mydataset/image_edit"

# 现在加载数据集会自动使用该路径
dataset = load_dataset("UCSC-VLAA/HQ-Edit")
```

### 重要注意事项：

1. **路径权限**：
   - 确保 Python 进程有权限写入 `/workspace/ckpt_downstream/zzm/mydataset/image_edit`
   - 在 Linux 系统中可能需要：`sudo chmod -R 777 /workspace/ckpt_downstream/zzm/mydataset`

2. **磁盘空间**：
   ```python
   # 检查可用空间（Linux系统）
   import shutil
   total, used, free = shutil.disk_usage("/workspace")
   print(f"可用空间: {free // (2**30)}GB")  # 需要约200GB空间
   ```

3. **下载验证**：
   ```python
   # 验证文件是否下载到正确位置
   import os
   print("下载内容:", os.listdir(custom_path))
   
   # 检查数据集结构
   print("数据集结构:", dataset)
   ```

4. **流式加载（节省空间）**：
   如果空间不足，可以使用流式模式：
   ```python
   dataset = load_dataset(
       "UCSC-VLAA/HQ-Edit",
       cache_dir=custom_path,
       streaming=True,  # 流式模式不缓存全部数据
       split="train"
   )
   
   # 流式访问示例
   for example in dataset.take(5):
       print(example["input"])  # 显示前5个样本的输入路径
   ```

5. **路径不存在时的处理**：
   ```python
   import os
   custom_path = "/workspace/ckpt_downstream/zzm/mydataset/image_edit"
   
   # 确保目录存在
   os.makedirs(custom_path, exist_ok=True)
   
   dataset = load_dataset("UCSC-VLAA/HQ-Edit", cache_dir=custom_path)
   ```

### 完整示例代码：
```python
from datasets import load_dataset
import os

# 指定自定义路径
custom_cache = "/workspace/ckpt_downstream/zzm/mydataset/image_edit"

# 确保目录存在
os.makedirs(custom_cache, exist_ok=True)

# 下载数据集到指定路径
try:
    hq_edit = load_dataset(
        "UCSC-VLAA/HQ-Edit",
        cache_dir=custom_cache,
        # 可选：只下载训练集
        # split="train"
    )
    print("下载成功！")
    print(f"位置: {custom_cache}")
    print(f"数据集结构: {hq_edit}")
    
    # 示例：访问第一个样本
    first_example = hq_edit["train"][0]
    print(f"输入图像: {first_example['input']}")
    print(f"编辑后图像: {first_example['edited']}")

except Exception as e:
    print(f"下载失败: {str(e)}")
    print("请检查：1) 磁盘空间 2) 路径权限 3) 网络连接")
```

### 常见问题解决：
1. **权限错误**：
   - 在终端运行：`sudo chown -R $USER:$USER /workspace/ckpt_downstream/zzm`
   - 或：`sudo chmod -R 777 /workspace/ckpt_downstream/zzm/mydataset`

2. **空间不足**：
   - 使用 `split="train"` 只下载训练集
   - 或使用 `streaming=True` 流式模式

3. **中断恢复**：
   - 重新运行相同代码会自动继续下载
   - 检查 `${custom_cache}/downloads` 中的临时文件

这些方法能确保数据集下载到你指定的精确路径。下载完成后，你可以在该路径下找到所有数据集文件。




---

## 🤗 HuggingFace 实用下载指南：模型与数据集极简教程  
**适用场景**：模型微调、本地部署、离线研究  
**核心工具**：`transformers`、`datasets`、`huggingface_hub`  

---

### 一、模型文件下载（4种方法）  
#### **1. 自动下载（推荐）**  
通过 `transformers` 库自动下载并缓存模型，适合中小型模型（如BERT、GPT-2）：  
```python  
from transformers import AutoModel, AutoTokenizer  

# 指定模型ID（如bert-base-uncased）  
model = AutoModel.from_pretrained("模型ID", cache_dir="/自定义路径")  
tokenizer = AutoTokenizer.from_pretrained("模型ID", cache_dir="/自定义路径")  

# 示例：下载BERT到指定目录  
model = AutoModel.from_pretrained("bert-base-uncased", cache_dir="/workspace/models")  
```  
- **路径自定义**：通过 `cache_dir` 参数指定存储位置（如 `/workspace/models`）。  
- **自动缓存**：首次下载后文件会缓存，后续加载无需重复下载。  

#### **2. 使用 `huggingface_hub` 批量下载**  
适合大型模型（如LLaMA）或需精确控制文件的场景：  
```python  
from huggingface_hub import snapshot_download  

# 下载整个模型仓库  
snapshot_download(  
    repo_id="google/flan-t5-large",  
    local_dir="/workspace/flan-t5",  
    revision="main",  # 指定分支  
    max_workers=8,    # 多线程加速  
    token="hf_xxx"    # 私有模型需Access Token  
)  
```  
- **关键参数**：  
  - `ignore_patterns=["*.bin"]`：排除特定文件。  
  - `allow_patterns=["*.safetensors"]`：仅下载安全格式文件。  

#### **3. 手动下载（无代码环境）**  
**步骤**：  
1. 访问 https://huggingface.co/models，搜索目标模型（如 `bert-base-uncased`）。  
2. 进入 **Files and versions** 标签页，下载关键文件：  
   - 模型权重：`pytorch_model.bin` 或 `model.safetensors`  
   - 配置文件：`config.json`  
   - 分词器：`vocab.txt`、`tokenizer_config.json`。  
3. 将文件放入本地文件夹（如 `/workspace/my_model`），加载时指向该路径：  
   ```python  
   model = AutoModel.from_pretrained("/workspace/my_model")  
   ```  

#### **4. Git LFS下载（超大型模型）**  
适合权重分片的大模型（如LLaMA-70B）：  
```bash  
# 安装Git LFS  
git lfs install  
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat  
```  
- 文件将完整克隆到本地，包含所有分片文件（如 `pytorch_model-00001-of-00002.bin`）。  

---

### 二、开放数据集下载  
#### **1. 使用 `datasets` 库（主流方法）**  
```python  
from datasets import load_dataset  

# 下载数据集（如GLUE的MRPC任务）  
dataset = load_dataset("glue", "mrpc", cache_dir="/workspace/datasets")  

# 流式模式（避免全量加载）  
stream_dataset = load_dataset("wikitext", "wikitext-103-v1", streaming=True)  
for sample in stream_dataset["train"].take(5):  
    print(sample["text"])  
```  
- **缓存路径**：通过 `cache_dir` 指定存储位置。  
- **常用数据集**：  
  - 自然语言处理：`squad`（问答）、`imdb`（情感分析）  
  - 多模态：`coco`（图像描述）、`librispeech_asr`（语音识别）。  

#### **2. 命令行下载（CLI工具）**  
```bash  
# 安装huggingface_hub  
pip install huggingface_hub  

# 下载数据集到指定目录  
huggingface-cli download lavita/medical-qa-shared-task-v1-toy \  
  --repo-type dataset \  
  --local-dir /workspace/datasets/medical-qa  
```  
- 支持断点续传和文件过滤（如 `--include "*.json"`）。  

---

### 三、避坑指南 & 加速技巧  
#### **常见问题**  
1. **下载中断**：  
   - 重试时添加 `resume_download=True` 参数（代码）或 `--resume`（CLI）。  
   - 设置国内镜像加速：  
     ```python  
     import os  
     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 镜像站点  
     ```  
2. **文件不完整**：  
   - 手动下载时确保包含：`config.json` + 权重文件 + 分词器文件。  
3. **权限问题**：  
   - 私有模型需登录：运行 `huggingface-cli login` 输入Token。  

#### **性能优化**  
- **多线程下载**：设置 `max_workers=8`（根据CPU核心数调整）。  
- **量化加载**：减少显存占用（适合GPU资源不足）：  
  ```python  
  model = AutoModel.from_pretrained("模型ID", load_in_8bit=True)  
  ```  

---

### 四、最佳实践场景推荐  
| **场景**               | **推荐方式**                     |  
|------------------------|----------------------------------|  
| 快速实验小模型         | `transformers`自动下载（代码）   |  
| 部署大型模型到生产环境 | `snapshot_download` + 本地加载   |  
| 学术研究需完整数据集   | `load_dataset` + 流式模式        |  
| 无Python环境           | 网页手动下载 + 本地加载          |  

> **提示**：所有下载的文件默认存储在 `~/.cache/huggingface/`，定期清理可释放磁盘空间。  

掌握这些方法，你已能高效获取HuggingFace的90%资源！下一步可探索模型微调（`Trainer`类）或部署（`Text Generation Inference`服务）。




Transformers
[如何更改模型下载存放路径](https://blog.csdn.net/yyh2508298730/article/details/137773125)
