# url:https://www.educoder.net/shixuns/2884/challenges/8205  
#### 任务描述


本关任务：从本地读取数据集，改变所有图片为相同的、固定的尺寸后保存到本地。


#### 相关知识


为了完成本关任务，你需要掌握：
1. 如何遍历文件夹下所有文件；
1. 如何改变图片尺寸；
1. 如何将图片保存到本地。

##### 遍历文件夹下所有文件
python标准库os中的walk()方法可以遍历一个路径下的所有文件。
如果当前所处的路径如下，包含一个文件夹dataSet,和三个文件，dataSet文件夹下又有一个文件1。
![](/attachments/download/371752)
那么在当前路径下执行以下代码后
```python
for root, dirs, files in os.walk("."):
	print('***file names***')
	for name in files:
		print(os.path.join(root,name))
```
输出：
`***file names***`
`.\AlexNet.py`
`.\preprocess.py`
`.\trainSet.txt`
`***file names***`
`.\dataSet\1`

`os.path.join(root,name)`的作用是拼接两个路径，代码中就拼接了`root`和`name`。
上例可以看到`os.walk()`是一个递归的函数，递归遍历某路径下所有文件夹和文件。

#####改变图片尺寸
将一个图片从它原始的尺寸改成固定的尺寸。

示例如下：
```python
#file是一个图片的路径
image=cv2.imread(file)
print(image.shape)
dim=(224,224)
resized=cv2.resize(image,dim)
print(resized.shape)
```

输出：
`(374, 500, 3)`
`(224, 224, 3)`

意思就是原图的尺寸是374×500，现在是224×224，最后的3是图片的RGB三个通道，这里不需要深究。

##### 将图片保存到本地
将改变尺寸后的图片，保存到本地。
`resized`是刚刚改变尺寸后的图片，`path`是想存储的文件名（路径），`cv2.imwrite()`方法就可以完成保存到本地的操作。
`cv2.imwrite(path,resized)`

#### 编程要求

根据提示，在右侧编辑器补充代码，完成对原始数据集的图片大小处理。

#### 测试说明

平台会对你编写的代码进行测试：

预期根据提供的原始数据，生成改变尺寸后的新的数据文件。

---
开始你的任务吧，祝你成功！


