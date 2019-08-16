[TOC]

---

####任务描述
本关任务：在训练过程中，我们需要知道什么时候模型过拟合训练集了，或者想记录没一次训练的结果，然后绘制训练的图像。本关以下的数据处理的内容，其实可以用`TensorFlow`做，但是没必要，我个人认为能用`Numpy`等科学计算包在模型之外处理的事情，在模型之外做更方便且自定义空间大。

####相关知识
为了完成本关任务，你需要掌握：1.如何使用`Numpy`实现`softmax`；2.如何将概率分布的数组，转为`onehot`编码的矩阵；3.如何计算准确率；4.如何绘制准确率的变化图像。

#####如何使用Numpy实现softmax
上一关中，我们模型的最后一层的输出其实是没有经过任何激活处理的数值。所以需要人工完成`softmax`的操作，然后将最大值变为1，其它位置变为0。根据`softmax`的定义，使用`numpy`实现即可。

示例：

```python
def softmax(x):
	#这一行不用管
    np.seterr(divide='ignore', invalid='ignore')
    #使用np.exp()对数组中所有元素做自然指数运算
	#np.sum(axis=1)按行求和，返回的是每一行的和的列表
	#np.exp(x).T将矩阵转置
	return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

a = [[1,2,3],
     [-2,-3,2]]
print(softmax(a))
```

输出：

```
[[0.09003057 0.24472847 0.66524096]
 [0.01786798 0.00657326 0.97555875]]

```
#####如何将概率分布的数组，转为onehot编码的矩阵
具体地说，就是把一个向量的最大值变为1，其它地方的值变为0。需要学会使用`argmax()`返回最大值的索引，学会如何对一个矩阵多个位置同时赋值。

示例如下：

```python
def returnOneHot(NNOutput):
	#先建一个全为0的数组
    out = np.zeros(NNOutput.shape)
	#返回每一行最大的值的索引。
	#如果axis=0，则是每一列最大值的索引
	#如果不指定axis参数，则返回整个数组最大值的索引。
    idx = NNOutput.argmax(axis=1)
	#np.arange(NNOutput.shape[0])返回列表
	#[0,1,2,4...]即每一行的行索引
	#idx也是一个里列表，是每一行的列索引
	#这样赋值后，每一行的最大值所在的位置就是1了。
    out[np.arange(NNOutput.shape[0]), idx] = 1
    return out

a = np.array([[1,2,3],
              [-2,-3,2]])
print(returnOneHot(a))
```

输出：

```
[[0. 0. 1.]
 [0. 0. 1.]]
```

#####如何计算准确率
首先需要知道有多少个样本正确，所以需要设置一个变量`right`来记录正确的样本个数，然后需要比较预测值和真实值是否相同，需要用到`(a==b).all()`。`a`,`b`是两个`numpy.array`类型的变量。

示例如下：

```python
a = np.array([1,2,3,4])
b = np.array([1,3,5,4])
print(a==b)
print((a==b).any())
print((a==b).all())
```

输出：

```
[ True False False  True]
True
False
```

直接用`==`比较，返回的是每个位置是否相等的布尔值的矩阵，
加上`.any()`是只要存在相等的元素即真，`.all()`是必须全相等才返回真。

#####如何绘制准确率的变化图像
使用`matplotlib.pyplot`库即可。

示例：

```python
import matplotlib.pyplot as plt
#这两行代码用于处理中文字符无法显示
#如果需要中文字符，加上这两行代码。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plot 第一个参数是x坐标，第二个参数是y坐标
#label可用于指明这一条曲线的名称
x=[1,2,3,4]
y = np.square(x)
plt.plot(x, y,label = 'x**2')
plt.legend()
plt.show()
```

输出：

![](/api/attachments/373355)

####编程要求

根据提示，在右侧编辑器补充代码，训练并记录每次的准确率。

####测试说明

平台会对你编写的代码进行测试，检测你编写的函数是否正确，如果正确，就会评定为通过。


---
开始你的任务吧，祝你成功！
