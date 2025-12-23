**Panda-70M: 多模态视频字幕数据集**
论文 ： https://arxiv.org/abs/2402.19479
Web ： https://snap-research.github.io/Panda-70M

**摘要**
Panda-70M 是一个大规模视频数据集，包含 70M 个视频及其字幕。该数据集通过多模态（文本视频描述、字幕和单个视频帧）输入的多种跨模态模型自动生成字幕。相比现有的数据集，Panda-70M 更精准地描述了视频中的主要对象和动作。该数据集展示了在视频字幕生成、视频与文本检索以及文本驱动的视频生成等下游任务中的优越性。

**介绍**
数据和注释的质量直接决定了下游模型的质量。相比图像-文本对，视频-文本对更难获取，手动标注视频需要更多时间且视频有时间维度，包含多个场景和动作。为了建立一个高质量的视频数据集，我们提出了一种自动化方法，利用文本视频描述、字幕和单个视频帧等多模态输入，生成高质量的字幕。

**方法**
1. **语义感知视频分割**：使用语义感知视频分割算法将长视频切割成语义一致的片段，保持片段的语义连贯性和适当的长度。
2. **跨模态教师模型生成字幕**：使用多种跨模态教师模型生成候选字幕，包括图像字幕模型和视觉问答模型。
3. **细粒度视频-文本检索模型选择最佳字幕**：在人类注释的100K视频子集上微调细粒度检索模型，以选择最佳字幕作为最终注释。
4. **学生模型学习**：训练学生模型以从教师模型中提取知识，提高视频字幕生成的效率。

**实验结果**
Panda-70M 数据集在视频字幕生成、视频与文本检索、文本驱动的视频生成等下游任务中表现优越，显著提升了大多数评价指标。具体实验包括：
1. **视频字幕生成**：在 MSR-VTT 和 MSVD 基准测试中，Panda-2M 预训练权重显著优于官方权重。
2. **视频与文本检索**：在 MSR-VTT、DiDeMo 和 MSVD 基准测试中，使用 Panda-5M 预训练的模型在零样本和微调检索任务中均表现优越。
3. **文本驱动的视频生成**：在 UCF101 和 MSR-VTT 基准测试中，Panda-2M 预训练权重在 FVD 和 CLIP 相似度指标上优于官方权重。

**结论**
Panda-70M 是一个包含高质量字幕的大规模视频数据集，展示了在多个下游任务中的应用潜力。未来的工作可以扩展更多无声视频样本，并构建长视频和密集字幕的数据集，以进一步提升下游任务的表现。


## Methodology
### Semantics-aware Video Splitting **语义感知视频分割**
算法步骤：
1. shot boundary detection
2. ImageBind
3. 针对其他情况的处理
	1. 一镜到底的长镜头（厉不厉害）
	2. 渐入、渐出的镜头切换
	3. 移除数量过多的同一类clips，保证diversity
4. 计算Max Running LPIPS——clip语意一致性的量化指标
		原理：比较关键帧之间的感知相似性
		![[Pasted image 20240624180731.png]]
		
### Captioning with Cross-Modality Teachers **跨模态教师模型生成字幕**
除了视频本身，还利用上其他信息：useful texts (e.g., video title, description, and subtitles) and images (e.g., individual video frames).

31个模型中挑选了8个高效的（计算成本低），主要是五种基本模型：
1. VideoLLaMA (video VQA)
2. VideoChat (video VQA)
3. VideoChat Text(natural language model which textualizes the video content)
4. BLIP-2 (image captioning)
5. MiniGPT-4 (image VQA).
> Details on the captioning process of each teacher model are described in *Appendix B.2*



###  Fine-grained Video-to-Text Retrieval **细粒度视频-文本检索模型选择最佳字幕**
现有的模型使用对比学习，使用的对比数据之间相关性很弱；但是从8个备caption中选一个，这个caption是高度相关的。

通过人工选择获得一个100k的subset， 去微调Unmasked Teacher  (UMT)
>We describe the details of the dataset collection and finetuning of UMT in *Appendix C.1 and C.2* respectively.

### Multimodal Student Captioning Model **学生模型学习**
每一个clip都需要运行8个caption模型和1个retrieval模型成本是很高的。

![[Pasted image 20240624183732.png]]
