
### 任务描述


本关任务：使用`TensorFlow`编写一个`Inception`的模块。


### 相关知识


为了完成本关任务，你需要掌握：1.`Inception`的特点。

#### Inception 的特点

首先个人认为`Inception`并不是一个完整的网络的名称，而是卷积神经网络中一种特殊的模块，可以与其它的卷积网络连接。

##### PART 1

在`《Going deeper with convolutions》`这篇论文中，`Inception`被首次提出，最后设计的一个22层的网络叫做`GoogleNet`。其中有谈到一个问题：提高深度神经网络性能最直接的方式是增加它们的深度和每一层神经元个数。但是这个方案既容易造成过拟合，也浪费硬件资源。然后提出`The fundamental way of solving both issues would be by ultimately moving from fully connected to sparsely connected architectures, even inside the convolutions.`，也就是说把全连接这种稠密的方式转换成稀疏地连接，这也更符合人类大脑神经元的特点。但是呢，当前的计算框架（`computing infrastructures`）并不太适合做这种稀疏类型的结构的运算，效率会比较低。这个就是作者设计`Inception`结构的动机，作者希望将稀疏的矩阵聚类为相对密集的子矩阵（`The vast literature on sparse matrix computations suggests that clustering sparse matrices into relatively dense submatrices tends to give state of the art practical performance for sparse matrix multiplication. `），说实话这里我是没看太懂的，有兴趣的朋友可以去参考一下这句话引用的文章`《On two-dimensional sparse matrix partitioning: Models, methods, and a recipe》`。下面就直接从实践层面来讲`Inception`的结构特点。

###### 1.1增加网络宽度

作者采用不同大小的卷积核对上一层的输入做卷积，然后拼接将它们的输出拼接。另外由于池化对于当前的卷积网络的成功十分重要，所以作者建议在每一个`Inception`模块中也加入一个池化。那么最初的结构就如下所示。值得一提的是作者这里选用`1×1`、`3×3`、`5×5`的卷积核的原因并没有什么特别的原因。不同大小的卷积核可以提取不同尺度的信息，可以理解为人们在看物体时是从不同分辨率的层次去接受信息的。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/1.png)

###### 1.2使用`1×1`卷积核降维

上述网络还有一个问题，就是输入的数据的通道数如果比较大，那么使用`3×3`、`5×5`的卷积核对计算资源的需求就还是比较高的，所以作者提出使用`1×1`的卷积核在做`3×3`、`5×5`的卷积核的卷积操作前减少通道数。另外作者还提到，这里`1×1`的卷积核的其实相当于一个线性的激活函数，这也是另外一个作用。这样该模块就是如下的结构。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/2.png)

###### 1.3使用平均池化替代全连接层

作者在文中提到，把最后的全连接层（`softmax`分类器前一层）变成一个平均池化层，可以让`top-1`准确度提高0.6%，（不过最后平均池化后还是加了一层全连接）

###### 1.4中间层的损失
`GoogleNet`在22层的网络的中间的两个大的部分，分别增加了一个连接了全连接层的`softmax`分类器，下图截取了一部分以供参考。在训练过程中，损失包括这两个分类器的损失，这个目的就是为了解决梯度消失。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/midloss.png)  

##### PART 2

接着，在`《Rethinking the Inception Architecture for Computer Vision》`这篇论文在，又做了几件事情来优化这个网络。

###### 2.0关于辅助分类器

刚刚上面提到了辅助分类器，即中间层的损失，然而在这篇文章中由提出`The removal of the lower auxiliary branch did not have any adverse effect on the final quality of the network.`，即其实这个低层的辅助分类器也没啥太大的作用。不过呢，作者又说到如果这个辅助分类器分支被`batch-normalized`过，或者有`Dropout`层，那么主分类器效果会更好，这个一定程度上可以认为辅助分类器有正则项的作用。个人认为，这个也不用多管。

###### 2.1使用小的卷积核来替代大的卷积核

这个在前文（`VGGNet`）中有提到，这里就不赘述。

###### 2.2非对称卷积

这一点还是很有创新性的。使用小卷积核，例如把`5×5`变成`3×3`这样，虽然减少了参数，但其实也就`(5×5-3×3×2)/25 = 28%`，减小了28%，如果是一个`3×3`变成两个`2×2`，那也就11%。所以作者提出把`n×n`的卷积核变成`1×n`和`n×1`两个卷积核分别进行卷积操作，例如作者说`very good results can be achieved by using 1 × 7 convolutions followed by 7 × 1 convolutions.`，即把`7×7`的卷积用先`1×7`卷积，然后`7×1`卷积，效果不错。**但是，特别需要注意的是，这个操作在靠前的层效果不好，当特征图的维度在12-20之间时，效果才比较好。**

###### 2.3使用平行的池化层和卷积层

一般我们使用池化操作减小特征图（`feature map`）的尺寸，但是这样会使得表示能力变差，所以一般需要在池化之前，增加卷积核的个数，但是这样运算量就会增大，因为要对更大的特征图做卷积，并且池化的时候也是更多的维，示例图如下。

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/red1.png)

因此作者提出了一个新的方法，并行的池化层和卷积层，卷积和池化的步长都是2，保证输出维度相同，如下：

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/red2.png)

具体的细节如下：

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/red3.png)

###### 2.4标签平滑正则项

这一点主要是改变了模型的损失函数，最终`top1`和`top5`的效果都提升了0.2%。这个不适合在这里与模型的结构混为一谈，有兴趣的朋友可以自己去看论文。

#### 编程要求

根据注释提示，使用`TensorFlow`搭建一个基于`Inception`模块的网络结构。其中需要实现的各`Inception`模块对应的结构如下：

##### `Inception_traditional`:

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/Inception_traditional.png)
 
##### `Inception_AsymmetricConv`:

双向箭头意思是顺序调换

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/Inception_AsymmetricConv.png)

##### `Inception_parallelAsymmetricConv`：

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/Inception_parallelAsymmetricConv.png)

##### `reduction`：

即上文2.3部分最后一张图。

##### `InitialPart`：

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/InitialPart.png)

我训练的一些结果：

堆叠三个`Inception_traditional`:

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/loss1.png)

堆叠三个`Inception_AsymmetricConv`:

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/loss2.png)

各堆叠一个，中间有`reduction`:

![](https://github.com/Lintianqianjin/reappearance-of-some-classical-CNNs/blob/master/img/step7/loss3.png)

#### 测试说明

平台会对你编写的模型结构测试，如果你按照提示和要求规范编写了代码，你的模型结构将于参考答案模型相同，平台将认为你通过本关，认为你掌握了`Inception`的各模块的结构。


---
开始你的任务吧，祝你成功！
