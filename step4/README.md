
#### 任务描述

本关任务：使用`TensorFlow`编写一个`AlexNet`模型。参考论文：`ImageNet Classification with Deep Convolutional Neural Networks`。论文一作的名字叫`Alex`，所以网络叫`AlexNet`。

#### 相关知识

为了完成本关任务，你需要掌握：1.`AlexNet`的特点；2.`TensorFlow`的基本使用。

##### AlexNet的特点

首先介绍一下`AlexNet`的模型结构。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step4/alexnet.png)

按照原论文的说法，即———

第一卷积层用大小为`11×11×3`的 96 个卷积核（两个`GPU`各 48 个）对`224×224×3`的输入（图像大小`224×224`，3 个通道）做卷积操作，卷积核的移动步长为 4 个像素。第二个卷积层将第一个卷积层的（经过了局部响应归一化和池化）输出作为输入，并用 256 个（连个`GPU`各 128 个）大小为`5×5×48` 的卷积核做卷积操作。第三个，第四个和第五个卷积层直接连接没有局部响应归一化和池化操作。第三卷积层具有`384`个（两个`GPU`各 192 个）大小为`3×3×256`的卷积核（其实是两个`GPU`的输入各 128 个，但是在这一层，两个`GPU`间的数据进行了交换）。 第四个卷积层有 384 个（两个`GPU`各 192 个）大小为`3×3×192`的卷积核，第五个卷积层有 256 个（两个`GPU`各 128 个）大小为`3×3×192`的卷积核。全连接层各有 4096 个神经元。看图可知，作者的全连接层其实有三层，前两层是各 2048 个神经元，加起来 4096 个，第三层是直接两个`GPU`的输出一起全连接，神经元个数是标签的类别的数量，即 1000 个。

**本次实训，其实大多数人都会受限于硬件设施，所以没有必要完全模拟这个网络。关注点要放在以下几个作者认为重要的地方，然后把可以实现的地方，实践一下就好。**

按照作者自己认为重要性排序，其四个主要特点依次是：

###### 1. ReLu 激活函数

激活函数简单来说就是对一层的输出再经过一个函数变化后输入下一层。

某层输出定义为$$X$$，激活函数定义为$$f$$，下一层接收到的数据为$$f(X)$$。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step4/relu.png)

作者使用`ReLu`激活函数的原因是模型训练起来，到达同样的精度，更快。如下图，实线是使用`ReLu`激活，虚线是使用`tanh`激活，错误率下降到25%时，后者需要训练的次数是前者的 6 倍作用

![](![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step4/train-loss.png)

###### 2.使用多个 GPU 训练

作者使用了两个`GPU`训练，但是两个`GPU`在某些特定的网络层会进行数据的交流。这一方面是提高了模型的训练速度，另一方面针对没有进行数据交流的模型，精度更高（作者说`top1`错误率下降了1.7%）。这个本次实训暂且不管，因为大部分人应该还是用个人笔记本来学习的。

###### 3.提出局部响应归一化（ Local Response Normalization，LRN ）

作者对经过`ReLu`激活后的值，使用`LRN`做了归一化，如下,$$a$$是原始值，$$b$$是正则化之后的值,$$n$$即定义的局部的范围，其它参数均为常数，需要预先定义。有兴趣的可以去看论文，这里不赘述。

![](![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step4/lrn.png)

一般而言，归一化后的输出会集中在一个比较小范围，是更利于模型的训练的。

###### 4.重叠的池化（ Overlapping Pooling ）操作

一般的池化层因为没有重叠，意思是如果池化范围为`3×3`，那么每次移动也会移动三个单位，即步长为 3 。举个例子，一个`9×9`的矩阵，用`3×3`池化后就成了`3×3`的矩阵了。但是作者池化的范围是`3×3`，每次移动却只移动 2 ，也就是说`9×9`的原始矩阵这样池化后，变成了`4×4`的矩阵。这个操作使得模型的`top1`错误率下降了0.4%。另外作者发现使用重叠的池化，与使用更小的池化范围（不重叠，产生相同大小的输出）相比，更不容易过拟合。

###### 5.随机失活 Dropout

这并不是作者认为创新的地方，只是作者为了解决过拟合的问题，除了数据增强外，使用的一种方法。

随机失活在学习过程中通过将隐含层的部分权重或输出随机归零，降低节点间的相互依赖性从而实现神经网络的正则，降低其结构风险。

我个人认为一定程度上有点随机森林的味道，每次只训练一部分的特征，当不失活的时候，网络就能有更优的表现。

##### TensorFlow 的基本使用

###### 1.学会调用 tf.placeholder() 

`tf.placeholder()`顾名思义是一个占位符，有些变量是需要人为输入的，所以需要在构建网络之前，先给这些人为输入一个位置。然后在`TensorFlow`的`session`调用`run()`的时候通过`feed_dict`，喂给模型这个占位符需要的数组。

示例：

```python
import tensorflow as tf
# placeholder
input = tf.placeholder(dtype=tf.float32,shape=(1))
#定义一个矩阵相乘的操作
output = tf.multiply(input, tf.constant([2.]))
with tf.Session() as sess:
	#feed_dict将[3.]喂给input
    a = sess.run(output, feed_dict={input: [3.]})
    print(a)
```

输出：

`[6.]`

###### 2.学会调用 tf.keras.layers 里的网络层

`tf.keras.layers`中有很多已经定义好的层，使用起来更方便。

示例：

```python
import tensorflow as tf
import numpy as np
#生成50个随机数
X = p.reshape((np.random.ranf(size=50)*2*np.pi),newshape=(50,1))
#使用Sin函数构造Y
Y_np = np.sin(X)

#将X,Y转换成Tensor
Y = tf.constant(Y_np,dtype=tf.float32)
X = tf.constant(X,dtype=tf.float32)

#使用tf.keras.layers搭建Dense层
dense1 = tf.keras.layers.Dense(32)(X)
out = tf.keras.layers.Dense(1)(X)

#定义损失为均方误差
loss = tf.reduce_mean(tf.losses.mean_squared_error(Y,out))
train = tf.train.AdamOptimizer().minimize(loss)

#开始训练，并输出损失
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(64):
        sess.run(train)
        if i%8 ==0:
            cur_out,cur_loss = sess.run([out,loss])
            print(cur_loss,end=',')

```
输出：

`1.31266,1.2358295,1.1623275,1.0924236,1.0262731,0.96392715,0.9053558,0.8504775,`

可以看到损失在下降，网络正常运行。

###### 3.学会调用 tf.nn.lrn 

`nn.local_response_normalization`和``nn.lrn`都行，即调用上述的局部响应归一化。

示例：

```python
import tensorflow as tf
import numpy as np
# 构造一个X，假设5个样本，每个样本2*2的大小，3个通道
x = np.arange(60).reshape([5, 2, 2, 3])
y = tf.nn.lrn(input=x, depth_radius=2, bias=0, alpha=1, beta=1)

with tf.Session() as sess:
    print(x[0])
    a = sess.run(y)
    print(a[0])
```

输出：

```
[[[ 0  1  2]
  [ 3  4  5]]

 [[ 6  7  8]
  [ 9 10 11]]]

[[[0.         0.2        0.4       ]
  [0.06       0.08       0.09999999]]

 [[0.04026846 0.04697987 0.05369128]
  [0.02980132 0.03311258 0.03642384]]]
```

这就是经过局部响应归一化后的结果

###### 4.使用 softmax 损失
这里主要就是要知道使用`tf.nn.softmax_cross_entropy_with_logits()`,主要需要传入两个参数`logits`和`labels`，`logits`是网络的输出（没有经过激活），`labels`是正确的`onehot`类型的标签。传入该函数后，会自动对`logits`进行`softmax`归一化，使得各维的和为 1 ，可以理解为所有类的概率和为 1 ，数值最大的就是概率最大的那个类。然后这个函数会自动计算损失，具体的计算过程有兴趣的可以去了解，这里不赘述。

#### 编程要求

根据要求搭建模型。

提示———需要用到的层有：

```
tf.keras.layers.Conv2D
需要传入的参数有：
filters, kernel_size, strides, padding, activation
filters是卷积核的个数，kernel_size是卷积核的大小，strides是卷积核移动的步长，padding是是否对原矩阵边界进行扩充，activation是激活函数。

tf.nn.local_response_normalization
需要传入的参数有：
alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0
以上采用原论文里数值即可。

tf.keras.layers.MaxPooling2D
需要传入的参数有：
pool_size，strides，padding
pool_size是池化的范围，strides是池化移动的步长。

tf.keras.layers.Flatten
不需要传入参数

tf.keras.layers.Dense
需要传入的参数有：
units，activation
units是该全连接层的神经元个数。

tf.keras.layers.Dropout
需要传入的参数有：
rate
rate是随机失活的神经元个数
```

---
开始你的任务吧，祝你成功！
