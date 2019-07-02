 
#### 任务描述


本关任务：编写一个产生镜像、平移后图像的数据增强程序。


#### 相关知识


为了完成本关任务，你需要掌握：1.如何产生翻转的图像，2.如何产生平移图像。

##### 如何产生翻转图像
`cv2`的`flip()`函数可以用来做图像的翻转操作。一般`cv2.flip()`需要传入两个参数`src`和`flipCode`,前者就是`cv2.imread()`后的图像，`flipCode`如果为0，则沿X轴翻转，即垂直翻转；如果传入一个正数，例如1，代表沿Y轴翻转，即水平翻转；如果传入一个负数，例如-1，代表俩个轴都翻转一次。

示例如下：
原图：
![图片正在加载](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/step2/%E5%8E%9F%E5%9B%BE.png)

```python
import cv2

img=cv2.imread("bus (1).jpg")
#X轴翻转
img_fliped_X = cv2.flip(img, 1)
#Y轴翻转
img_fliped_Y = cv2.flip(img, 0)
#都翻转
img_fliped_X_Y = cv2.flip(img, -1)

cv2.imwrite('fliped_X.png',img_fliped_X)
cv2.imwrite('fliped_Y.png',img_fliped_Y)
cv2.imwrite('fliped_X_Y.png',img_fliped_X_Y)
```
输出：  
fliped_X.png
![图片正在加载](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/step2/flipedX.png)  
fliped_Y.png
![图片正在加载](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/step2/flipedY.jpg)  
fliped_X_Y.png
![图片正在加载](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/step2/flipedXY.jpg)  

**在本次任务中，我们只需要用到水平翻转，因为结合现实情况，一般做车辆的识别，不会遇到倒着的车。**


##### 如何产生平移图像
即在X，Y轴上平移图像。使用`cv2.warpAffine()`可以完成这个操作。这个方法主要有三个参数：`src`, `M`, `dsize`。`src`是原图的矩阵，`dsize`是输出的图像应该具有的大小，本次任务仍然输出与原图一样的尺寸，`M`是变换矩阵，如果我们做X,Y轴的平移只需要知道`M=[[1,0,X],[0,1,Y]]`,其中`X`是在X轴上要平移的量，`Y`是要在Y轴上平移的量。`X`为负数是往左平移，`Y`为负数是往上平移。

示例如下：
原图同上。

```python
img=cv2.imread("bus (1).jpg")
M=np.array([[1,0,-50],[0,1,-50]],dtype=np.float32)
img_change=cv2.warpAffine(img,M,(224,224))
cv2.imwrite('translation.png',img_change)
```

输出：
translation.png
![图片正在加载](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/step2/translation.jpg)  

#### 编程要求

根据提示，在右侧编辑器补充代码，对所有图片产生所需的改变之后图片加入到数据集中。

#### 测试说明

平台会对你编写的代码进行测试：
判断你产生的图片是否和正确的输出图片一样。


---

开始你的任务吧，祝你成功！


