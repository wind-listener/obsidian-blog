Python读取图片最常用的有4种方式，分别适配普通处理、CV任务、可视化、深度学习等场景，核心代码简洁易用：

### 1. Pillow（最通用，推荐优先使用）
Pillow是Python图像处理标准库，支持几乎所有图片格式，接口简洁。
```python
from PIL import Image

# 读取图片（默认保留原始格式和通道顺序）
img = Image.open("zidane.jpg")

# 常用后续操作（可选）
img = img.resize((640, 640))  # 调整尺寸
img_np = np.array(img)        # 转为numpy数组（RGB格式，shape=(H, W, 3)）
img.save("resized_zidane.jpg")# 保存图片
```
- 特点：返回`PIL.Image`对象，支持裁剪、旋转、缩放等常用操作，与numpy、PyTorch兼容性好。

---

### 2. OpenCV（CV任务首选，如目标检测、分割）
OpenCV是计算机视觉专用库，读取速度快，默认返回numpy数组，适合数值计算。
```python
import cv2

# 读取图片（默认BGR通道顺序，shape=(H, W, 3)，uint8类型）
img = cv2.imread("zidane.jpg")

# 关键：转为RGB格式（适配多数可视化和模型输入要求）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 常用后续操作（可选）
img_resized = cv2.resize(img_rgb, (640, 640))  # 调整尺寸
cv2.imwrite("opencv_zidane.jpg", img)          # 保存（需用BGR格式）
```
- 特点：返回numpy数组（uint8，0-255），适合OpenCV生态的后续处理（如边缘检测、轮廓提取）。

---

### 3. matplotlib（可视化友好，适合绘图展示）
matplotlib主要用于绘图，但读取图片方便，默认返回RGB格式，可直接配合`plt.imshow()`显示。
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图片（RGB通道顺序，shape=(H, W, 3)，numpy数组）
img = mpimg.imread("zidane.jpg")

# 直接显示图片（无需转换通道）
plt.imshow(img)
plt.axis("off")  # 隐藏坐标轴
plt.show()
```
- 特点：读取和显示一体化，适合快速查看图片，或在图表中嵌入图片。

---

### 4. torchvision（深度学习适配，直接转张量）
适合PyTorch深度学习场景，读取后可直接转为模型要求的BCHW格式张量。
```python
from torchvision.io import read_image
from torchvision.transforms import Resize

# 读取图片（默认RGB格式，返回BCHW格式张量，dtype=torch.uint8）
img_tensor = read_image("zidane.jpg")  # shape=(3, H, W)

# 常用后续操作（适配模型输入）
resize = Resize((640, 640))
img_resized = resize(img_tensor)       # 调整尺寸
img_norm = img_resized.float() / 255.0 # 归一化到[0,1]
```
- 特点：无需手动转换通道和格式，直接对接PyTorch模型，适合深度学习推理/训练。

---

要不要我帮你整理一份 **带完整注释的通用图片读取工具脚本**，包含4种方式的封装、格式转换、错误处理，可直接调用读取并适配不同使用场景？