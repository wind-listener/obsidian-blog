#optical_flow #ofs #cv #光流估计

> 光流（Optical Flow）是计算机视觉中的一个重要概念，指的是**图像序列中像素随时间的运动模式**。它可以帮助我们分析视频中的运动信息，并在**目标跟踪、视频稳定、运动检测**等领域得到广泛应用。
> 本文将介绍光流的基本原理、经典算法及其应用，并提供 Python 代码示例，帮助你快速上手。

---

# 光流的基本概念

光流定义的各种表述：
-  [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. Optical flow can also be defined as the distribution of apparent velocities of movement of brightness pattern in an image. （from WiKi）
- 光流是空间运动物体在观察成像平面上的像素运动的瞬时速度，是利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性来找到上一帧跟当前帧之间存在的对应关系，从而计算出相邻帧之间物体的运动信息的一种方法。一般而言，光流是由于场景中前景目标本身的移动、相机的运动，或者两者的共同运动所产生的。
- 光流表示的是相邻两帧图像中每个像素的运动速度和运动方向。是描述图像序列中像素位置移动的一个矢量场。

简而言之，光流描述了像素点在图像帧之间的运动，它通常表示为一个二维向量场：

• **方向**：表示像素运动的方向。
• **大小**：表示运动的速度。


  

## 直观理解

假设我们在拍摄一段视频，每一帧都是一张图片。如果视频中有一个移动的物体，光流算法可以帮助我们计算**物体在前后两帧之间的运动轨迹**。
![[光流场示意图.png]]
例如，在一个交通监控视频中，光流可以用于分析汽车的运动方向和速度。

  

## 光流的基本假设

光流计算依赖于以下假设：

1. **亮度恒定（Brightness Constancy Assumption）**：

	物体在运动过程中，像素的灰度值不会发生变化，即：
	
	$$
	
	I(x, y, t) = I(x + u, y + v, t + 1)
	
	$$
	
	其中，$I(x, y, t)$ 表示时间 t 时刻的像素亮度。 $(u, v)$ 表示该像素的位移（光流矢量）。

2. **小位移假设（Small Motion Assumption）**：

	认为相邻帧之间像素的移动很小，因此可以用[[泰勒展开]]进行线性近似。

3. **空间一致性（Spatial Coherence Assumption）**：

	认为一个小区域内的像素运动是相似的，即局部区域的光流是平滑的。

#  光流的经典算法

## Lucas-Kanade 局部光流估计算法


Lucas-Kanade（LK）方法是一种，它假设一个小窗口内的所有像素具有相同的运动，并通过最小二乘法来求解光流。

  ![[Pasted image 20250310161636 2.png]]
![[Pasted image 20250310161648 1.png]]
**优点：**

• 计算简单，速度快。

• 适用于小范围运动。

  

**缺点：**

• 对于快速运动的目标不稳定。

  

##  [[Horn-Schunck 全局光流估计算法]]

Horn-Schunck 方法是一种全局光流算法，它通过引入平滑约束，使整个图像的光流场更加平滑。

**优点：**

• 适用于大范围运动。

• 计算全局一致的光流。
**缺点：**

• 计算复杂度高，容易受噪声影响。

  

##  Farneback **密集光流**算法
  

Farneback 方法是一种**密集光流**算法，它使用多项式展开来估计每个像素的运动信息。

  

**优点：**

• 计算速度较快，适用于密集光流计算。

  

**缺点：**

• 在快速运动的情况下，可能会丢失细节。


## PWC-Net
2015年工作

## [[RAFT]]

现代光流方法使用深度学习（如 **RAFT**）进行估计，具有更高的精度。

  

**优点：**

• 精度高，适用于复杂场景。

• 适用于自动驾驶、AR/VR 等高精度需求的应用。

  

**缺点：**

• 计算成本高，需要 GPU 计算。


# 光流的缺点

我们觉得ofs的缺点是：

1. 没办法判断镜头移动，我们希望只有镜头移动对应的指标值比较小
    
2. 没办法判断背景变化，比如天空从蓝变红，这种其实动态程度也很小
    
3. 就是觉得需要有一些语义信息的融入，根据语义判断动态程度的大小是最合理的
---

# 光流的应用

  

**4.1 视频稳定**

  

光流可以用于检测相机的抖动，并进行视频稳定处理。

  

**4.2 目标跟踪**

  

光流可以跟踪视频中的物体，广泛应用于智能监控、体育分析等领域。

  

**4.3 运动估计**

  

光流可以用于预测视频帧之间的运动信息，在视频压缩（H.264）中被广泛使用。

  

**4.4 行人检测**

  

光流可以帮助监控系统检测行人的运动轨迹，识别异常行为。

  

**4.5 虚拟现实（VR）和增强现实（AR）**

光流可用于 VR/AR 头部跟踪、手势识别等交互任务。

---

# Python 实现光流
**5.1 使用 Lucas-Kanade 计算光流**

以下代码使用 OpenCV 实现 Lucas-Kanade 光流：

```Python
import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('video.mp4')

# 读取第一帧并转换为灰度图
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 角点检测
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# LK光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算光流
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    
    # 画出光流轨迹
    for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    
    cv2.imshow('Optical Flow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    prev_gray = gray.copy()
    prev_pts = next_pts

cap.release()
cv2.destroyAllWindows()
```

**5.2 使用 Farneback 计算密集光流**

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Dense Optical Flow', rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = next

cap.release()
cv2.destroyAllWindows()
```

  

# 推荐博客
[稀疏光流、稠密光流和FlowNet2算法](https://yongqi.blog.csdn.net/article/details/107478747?fromshare=blogdetail&sharetype=blogdetail&sharerId=107478747&sharerefer=PC&sharesource=weixin_44938670&sharefrom=from_link)

[光流模型概述：从 PWC-Net 到 RAFT - OpenMMLab的文章 - 知乎](https://zhuanlan.zhihu.com/p/446739441)


# 总结

• 光流用于估计视频中像素的运动信息，广泛应用于目标跟踪、视频分析等领域。

• 经典光流算法包括 **Lucas-Kanade**、**Horn-Schunck** 和 **Farneback**，深度学习方法如 **RAFT** 进一步提升了精度。

• OpenCV 提供了多种光流计算方法，可用于不同的应用场景。


