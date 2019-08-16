
#### 任务描述


本关任务：编写一个具有残差模块的模型。


#### 相关知识


为了完成本关任务，你需要掌握：1.残差模块的特点；2.使用`TensorFlow`完成`Tensor`的加和操作；3.`TensorFlow`的`transpose()`函数；4.`TensorFlow`的`expand_dims()`函数；5.`TensorFlow`的`tile()`函数；6.使用`TensorFlow`的`batch_normalization`。

##### 残差模块的特点
残差的思想其实相对于前面的那些结构的特点要简单很多，还是比较好理解的。

深度`CNN`网络达到一定深度后，再一味地增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢。并且研究表明，排除数据集过小带来的模型过拟合等问题后， 过深的网络还会使分类准确度下降（相对于较浅些的网络而言），如下图所示，作者比较了56层和20层的卷积网络在`CIFAR-10`数据集上的表现，发现56层的网络在训练集和测试集上的错误率都要更高一些。

![](/api/attachments/375807)

这个其实反映的问题是并不是所有的网络都是易于优化的，实验也表明当时还没有方法可以在（较大幅度地）增加网络深度的同时不增加错误率。于是作者就提出了残差结构，用于解决这个问题。

先假设实际上需要的一个关于`X`的映射是`H(X)`，那么作者就让那些堆叠起来的非线性变化的网络层尝试拟合`H(X)-X`，将这个式子记为`F(X)`，这一项其实就是残差，然后易得`H(X)=F(X)+X`。

这样的目的是什么呢？首先明确要实现的目标是网络加深但不影响效果，换句话说希望加入的层都趋于单位矩阵，因为`X·I=X`，这样子加入的层就不影响原先的网络了，但是让一个随机初始化的矩阵优化到单位矩阵是很困难的，作者认为让`F(X)`趋于0，相对而言是更容易的。

接下来问题就是`F(X)+X`该怎么实现了，于是就有了如下结构，这个直接加到输出层的这条传播路径叫做`shortcut connections`，它相当于完成了单位矩阵的映射操作。

![](/api/attachments/375809)

然后作者就将这个结构加入到一个34层的直接堆叠的网络中，并将两者做了一个对比。

![](/api/attachments/375810)

结果表明，错误率从28.54%下降到了25.03，就是确实是成功的。

之后在论文`《Identity Mappings in Deep Residual Networks》`中，这个基本的残差结构又有了一些其它的改变，这篇论文里提到信息传播部分（`shortcut connections`部分）最好保证简单，这样利于优化。这篇论文里提出了新的一个结构如下（右），并表示有更好的效果。

![](/api/attachments/375814)

这也是我们本关需要实现的结构。

##### `Tensor`的加和操作

顾名思义，将形状一样的`Tensor`，各个数值对应求和形成新的`Tensor`。

示例如下:

```python
import tensorflow as tf

a = [[[1.,2.,3.],[3.,4.,5.]],
     [[3.,5.,4.],[2.,1.,4.]]]

b = [[[0.1,0.2,0.3],[0.5,0.6,0.1]],
     [[0.1,0.2,0.1],[0.5,0.4,0.3]]]

c = tf.add(a,b)

sess = tf.Session()
print(sess.run(c))
```

输出：

```
[[[1.1 2.2 3.3]
  [3.5 4.6 5.1]]

 [[3.1 5.2 4.1]
  [2.5 1.4 4.3]]]
```
##### `TensorFlow`的`transpose()`函数

该函数用于置换张量的维，或者说重新排列张量的维，可以结合实例来理解，可能更容易。我们将用到这个函数去实现使用一个`Weight`矩阵与不同的通道的`feature map`做积，但因为输入一般是通道是最后一维，所以，这里可以把通道提到尺寸的前面，这样就更容易做以上操作。

示例如下：

```python
a = [[[1., 2., 3.], [3., 4., 5.]],
     [[3., 5., 4.], [2., 1., 4.]]]
# 参数[2,0,1]就是置换之后的顺序
# 原来的第2维是现在的第一维，原来的第0，1维分别是现在的第1，2维。
c = tf.transpose(a, [2,0,1])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
d = sess.run(c)
print(d)
```

输出如下：

```
[[[1. 3.]
  [3. 2.]]

 [[2. 4.]
  [5. 1.]]

 [[3. 5.]
  [4. 4.]]]
```

可以看到数据的维度已经变换过了，如果觉得没有理解，可以自己动手算算，多体会一下。

##### `TensorFlow`的`expand_dims()`函数

这个函数用于在一个张量中插入一维，例如`[a,b,c]`变成`[1,a,b,c]`。我们之所以要用到这个函数，是为了生成一个`weight`矩阵后，先扩充一维，然后再扩充的这一维上堆叠这个同样的矩阵，用于与各个通道的`feature map`做积。

示例如下：

```python
x_ = tf.constant([[10.,20.],
                  [30.,40.],
                  [50.,60.]
                  ])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
g = tf.expand_dims(x_,0)
g_ = tf.expand_dims(g,0)

print('x_')
print(sess.run(x_))
print(x_.shape)
print('g')
print(sess.run(g))
print(g.shape)
print('g_')
print(sess.run(g_))
print(g_.shape)
```

输出

```
x_
[[10. 20.]
 [30. 40.]
 [50. 60.]]
(3, 2)
g
[[[10. 20.]
  [30. 40.]
  [50. 60.]]]
(1, 3, 2)
g_
[[[[10. 20.]
   [30. 40.]
   [50. 60.]]]]
(1, 1, 3, 2)
```

##### `TensorFlow`的`tile()`函数

这个函数会在一个张量的某一维上，重复你需要的次数，这个需要重复的次数就是你需要传入的参数`multiples`，他要是一个一维的张量，长度与输入张量的维数相等，每个索引处的值对应该索引对应维的重复次数。我们将需要通过此函数，将权重矩阵或者偏执项重复堆叠以方便与各个通道的`feature map`做积，并保持参数的共享。

示例如下：

```python
x_ = tf.constant([[10.,20.],
                  [30.,40.],
                  [50.,60.]
                  ])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
g = tf.expand_dims(x_,0)
print('g')
print(sess.run(g))
print(g.shape)
i = tf.tile(g,tf.constant([1,2,1]))
print(sess.run(i))
print(i.shape)
```

输出：

```
g
[[[10. 20.]
  [30. 40.]
  [50. 60.]]]
(1, 3, 2)
[[[10. 20.]
  [30. 40.]
  [50. 60.]
  [10. 20.]
  [30. 40.]
  [50. 60.]]]
(1, 6, 2)

```

可以看到我的需求是第二维重复两遍，那么输出就是这样的了。

##### `TensorFlow`的`batch_normalization`

先简单介绍一下`batch normalization`，顾名思义，应该是一种归一化手段。先简单介绍一下它的处理过程，对于每一批训练的数据，即数据的标准差为$$\sigma$$，均值为$$\mu$$，则$$x_{new}=\gamma(x-\mu)/\sigma+\beta$$。

其中$$\gamma$$和$$\beta$$就是要训练得到的数值。简单来说就是把`Z-score`标准化后的值又做了一个线性变换。

这样的变换能带来以下几个好处：1.减轻了对参数初始化的依赖；2.训练更快，可以使用更高的学习率；3.`BN`一定程度上增加了泛化能力。

至于为什么能带来这些好处，可以以后在谈，这里重点主要是应用，有兴趣的朋友可以研读下这篇论文`《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif》`。

`TensorFlow`中使用`batch_normalization`，可以使用`tf.layers.batch_normalization()`调用，调用它有一个特别需要注意的地方，就是训练的时候要传入参数`training=True`，做预测的时候要传入`training=False`，源文档中对这个参数的解释是：`Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder). Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics). NOTE: make sure to set this parameter correctly, or else your training/inference will not work properly.`。实践过程中，这个参数的值可以通过`placeholder`传入。

另外，官方`API`中特别强调了使用这个`tf.layers.batch_normalization()`，要注意`when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be executed alongside the train_op. `就是说训练的时候标准差$$\sigma$$,均值$$\mu$$，是需要更新的，因为每一次的标准差和均值是某一个`batch`的，他不代表全体样本，但是做预测的时候，是希望这个是全体样本的均值和方差，所以每训练一个`batch`都需要因为接收到更多样本更新一次这两个值。但是这种更新类型的算符不在需要训练的算符中，所以直接运行训练算符时，并不会更新。因此需要每次训练时执行这个更新操作。

官方给出的示例如下：

```python
x_norm = tf.compat.v1.layers.batch_normalization(x, training=training)

  # ...

update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = optimizer.minimize(loss)
train_op = tf.group([train_op, update_ops])
```

并且特别提到` In particular, tf.control_dependencies(tf.GraphKeys.UPDATE_OPS) should not be used`，但是事实上这么用，如下，并没有什么问题(参考自：`https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow`)：

```
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss)
...
sess.run([train_op], ...)
```

我也一直是这么用的，**右侧代码的实现，我们将使用这个方案**。

#### 编程要求

根据提示，在右侧编辑器根据提示补充代码，搭建一个简单的有残差模块的`ResNet`模型。其中已经给出的代码，完善即可，最好不要有删改。

#### 测试说明

平台会对你编写的模型的结构进行测试。如果正确，说明你已掌握模型结构，则通过本关测试！

---
开始你的任务吧，祝你成功！
