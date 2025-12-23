# 任务定义

> 图像抠图（image matting）是指从一张图像中精确提取出感兴趣对象，并将其与背景分离的任务。
> 
> 这通常需要在像素级别对目标对象进行精确分割，以便实现高质量的抠图效果。
> 
> 可以分为“需要人工辅助输入”和“全自动” automatic and auxiliary-based methodologies.

## 和"图像分割（Segementation）"的异同

相似之处：

1. 目标：图像抠图和图像分割都旨在将图像中的对象从背景中分离出来，以便进一步处理或分析。
    
2. 像素级别操作：两者都涉及对图像中的像素进行操作，通常需要对每个像素进行分类或标记。
    

不同之处：

1. 粒度：图像抠图通常关注的是精确的对象边界，要求在像素级别上对前景和背景进行精确分割，以实现逼真的合成或编辑效果。而图像分割可以是像素级别或区域级别的，目标是将图像分成几个具有语义意义的部分。
    
2. 输入：**图像抠图通常需要额外的辅助输入**，如 trimap 或 coarse 分割，以帮助算法准确地分离前景和背景。而图像分割通常只需要原始图像作为输入。
    
3. 复杂性：图像抠图任务通常更加复杂，因为它**需要处理对象与背景之间的细微边界和半透明区域**。相比之下，图像分割任务可能更简单，因为它通常只需要将图像分成几个连续的区域或类别。
    

![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=NGYxZmEwZGRjMGVkODE1MTI0Y2M0NTc5ZjFjNzM0ZTBfMmZibTd6R3pQSzhWU1phdmdaeTFqMUQ4Z3hkbkY2U3FfVG9rZW46RW5FeGJFT3p3b2ZUcXV4cnlCTWN5MklrbkJnXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)

## 应用场景

图像抠图在许多领域都有广泛的应用，例如图像编辑、虚拟现实、增强现实、电影特效等。

## 一般流程

通常，图像抠图涉及到以下几个主要步骤：

1. **前景/背景分离**：确定图像中的前景目标和背景区域。
    
2. **精确分割**：对前景目标与背景进行精确分割，通常需要逐像素地确定哪些像素属于前景，哪些像素属于背景，以及哪些像素属于半透明区域（即“未知”区域）。
    
3. **alpha通道生成**：根据分割结果生成alpha通道，用于描述每个像素的透明度。
    
4. **合成**：将分割后的前景对象与新的背景进行合成，创建一个具有逼真效果的图像。
    

  

# 方法调研

## Input：

- Automatic —— 更加适用于商业应用
    
- Auxiliary-based
    
    ![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=NDJkN2U2OTEwOGQ4MGY5MzEyZWY0OGExNTlkMDRkODlfWTduNHhyOXRWTzRhVmVJMldwWFZLRlI2cWRnTFozUnJfVG9rZW46Wk1PSWJtYTdKbzJIRDl4R0QybWNHT2t2bnJnXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)
    

## Target：

- salient opaque foregrounds
    

![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=M2Y1MWE4ZTQyNDk2MWYyMjJkOWMzNmZiMzY0YzQ4YjhfdDM5aHNERGk3elpEdXluejBpQlpjTGFadkdhUFEyb2VfVG9rZW46WmpWemJEZ01Yb0h0aXB4SmZOMGMzT3Y5bjFkXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=M2YzOTZmYTA0ZTNlODkwYTkwYzRiNmIyZWQwZjE3YzhfZE9aVXc2Q3hVSXNrb1ZOZ2FsSlBHcHY2ajlLS2lvOUdfVG9rZW46QWNkSWJ0UjA3b2RNYUF4R1M0TWNBNXBhbnVjXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)

- salient transparent foregrounds
    
    ![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=NTQwZDk1MzAwZjdhMmM1ZWRjZDMzYTdlZjBmY2JlN2NfaHdzTzVLVGRJa3VDcFNONDd0MFFIbXI5UXlqQUdiVzZfVG9rZW46QXB1cWJ6ZGFPb2p6N2F4QjdHeGMzcGxGbkdQXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)
    
- non-salient foregrounds
    
    ![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=NWQzNWUwOWM0YzU3NWQzNTc3OGE3YmVmY2E0MzY4NDRfMDVWejlPSmlsNkJwT3c4UDhrTDB6RnV1WklQRjdOUGdfVG9rZW46Sjl3ZWJDTTF2b0ZsWEF4YnM1eGNTZEdSbmhoXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)
    

## Methodoloogy

- auxiliary input-based methods
    
    - a single one-stage CNN
        
    - a one-stage CNN is used with modules carefully designed
        
    - parallel two- or multistream structures
        
- automatic methods.
    
    - a one-stage structure
        
    - a sequential two-step structure (1. segmentation mask or trimap 2. Final alpha matte)
        
    - a parallel two- or multi-stream structure
        

  

  

  

# 市场调研

## 抠图AI - 商用

  

<div style="display: flex; justify-content: center;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/微信图片_20240223093036.jpg" alt="Image 1" style="width: 300px;height: 300px; margin-right: 20px;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/mmexport1708654043766.png" alt="Image 2" style="width: 300px;height: 300px; margin-left: 20px;">

</div>

## Removal AI

https://removal.ai/

  

<div style="display: flex; justify-content: center;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/微信图片_20240223093036.jpg" alt="Image 1" style="width: 300px; margin-right: 20px;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/[removal.ai]_53579081-1598-45df-9fe7-25234df28b3e-_20240223093036.png" alt="Image 2" style="width: 300px; margin-left: 20px;">

</div>

  

## briaai/RMBG-1.4

https://huggingface.co/briaai/RMBG-1.4

https://huggingface.co/spaces/briaai/BRIA-RMBG-1.4

  

## ohyesai —— 基于开源项目 rembg

https://github.com/danielgatis/rembg

!

  

使用isnet-general-use

<div style="display: flex; justify-content: center;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/微信图片_20240223093036.jpg" alt="Image 1" style="width: 300px; margin-right: 20px;">

<img src="D:/WPSSync/Markdown/ohyesai/imgs/mmexport1708666497035.png" alt="Image 2" style="width: 300px; margin-left: 20px;">

</div>

  

## 调研结论

- 商用工具的结果对边缘处理相对较好，但是都有改进空间
    
- 但是在镜片内的背景，都没有准确识别
    

  

# 自己的方案探索

## YOLO+SAM

是图像分割的解决方案，对于抠图来说并不适用

## Grounding DINO+SAM+VitMatte

流程如下：

1. 自动标定
    
    1. 使用yolov9
        
    2. 使用GroundingDINO
        
2. 获得Trimap
    

使用bbox和SAM分割，腐蚀膨胀得到Trimap

3. 获得Alpha Matte
    

使用VitMatte

4. 输出结果
    

### Grounding DINO

![](https://vxzhdxcmgd9.feishu.cn/space/api/box/stream/download/asynccode/?code=YmFhNjFkOTlkNmQ2OTIyMTEzZmQzNWQyYzkyYzEwZjhfVUJxeTkxUGxUNjdOUlNrc3FuZ2kyM2lmU0lMRkNXckNfVG9rZW46QUxZNmJrTGlYb0ZPU0J4bTdIWWNPQTR2blJoXzE3MDk3MTI2MTY6MTcwOTcxNjIxNl9WNA)