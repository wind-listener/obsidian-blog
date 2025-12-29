---
title: "**简介**"
date: 2025-10-31
draft: false
---

# **简介**

OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库。它由Gary Bradski于1999年发起，旨在为计算机视觉应用提供基础算法，帮助开发者快速开发各种视觉功能。OpenCV广泛应用于图像处理、物体识别、视频分析等领域，支持多种编程语言（如C++、Python、Java等）和操作系统（如Windows、Linux、macOS等）。[OpenCV 官网](https://opencv.org)

# 常用示例

```Python
import cv2

# 图像处理示例：读取图像并转换为灰度图
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray)

# 特征检测与匹配：使用ORB检测图像中的关键点
orb = cv2.ORB_create()
keypoints = orb.detect(gray, None)
img_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
cv2.imwrite('keypoints.jpg', img_keypoints)

# 物体检测：使用Haar级联分类器进行人脸检测
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite('face_detected.jpg', img)

# 视频分析：读取并显示视频中的帧
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

# **安装 OpenCV**

安装详情请参考[OpenCV安装指南](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)

## 一般安装

```Bash
pip install opencv-python

# 对于更复杂的项目，可能需要安装包含额外功能（如视频处理和GUI功能）的完整版本
pip install opencv-contrib-python
```

## **支持 CUDA 加速**

需要从源代码重新编译 OpenCV，并确保启用了 CUDA 支持。以下是一个简要的安装步骤：

1. 安装依赖库：
    

```Bash
apt-get update
apt-get install build-essential cmake git pkg-config
apt-get install libjpeg-dev libtiff-dev libpng-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev
apt-get install libgtk-3-dev
apt-get install libatlas-base-dev gfortran
apt-get install python3-dev
apt-get install libcuda1-11-6 libnpp-dev nvidia-cuda-toolkit
```

2. 下载 OpenCV 源代码并切换到需要的版本：
    

```Bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.x
cd ../opencv_contrib
git checkout 4.x
```

3. 编译 OpenCV：
    

```Bash
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=9.0 .. # 这里的 CUDA 架构号请根据你的 GPU 修改.
make -j8
make install
ldconfig
```

对于 NVIDIA H800，它属于 **Hopper** 架构。

**Hopper 架构**的 GPU 对应的 CUDA Compute Capability 是 **9.0**

![](https://zhipu-ai.feishu.cn/space/api/box/stream/download/asynccode/?code=NzUxMzVmNTJhMGIyYTk3NjZjZjM5YTZiNTMzZjdjZjZfU1RZbWhoeEJab1BLOTdLelZFb1BhNW51dUZINUlQT29fVG9rZW46RFkzZ2I4eVFxb2JOMlJ4VEtLZGNHbTBRbkhoXzE3NjE5MDQ2ODM6MTc2MTkwODI4M19WNA)

**检查 OpenCV CUDA 支持**：

安装完成后，你可以再次检查 OpenCV 是否启用了 CUDA：

```Python
import cv2
print(cv2.getBuildInformation())
```

确认 WITH_CUDA=YES 才能确认启用了 GPU 加速。

# **参考资料**

• [OpenCV 官方文档](https://docs.opencv.org/master/)

• [Haar Cascade Classifiers - OpenCV](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)

• [特征检测与匹配 - OpenCV](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)