#### 任务描述


本关任务：本关将结合此前各个关卡，正式训练并保存一个网络，并能够随时加载这个模型对位置数据集进行预测。


#### 相关知识


为了完成本关任务，你需要掌握：1.如何训练一个网络；2.如何保存一个网络；3.如何加载复用一个网络。

##### 如何训练一个网络
训练一个`TensorFlow`搭建的神经网络，只需要调用`tf.Session().run()`，需要`run`的算符就是在网络中定义的`train`算符；另外如果在模型的搭建过程中，存在`placeholder`，则需要通过`run()`的参数`feed_dict`传入这些`placeholder`的值，传入的参数是以字典的形式，然后字典的`key`是搭建过程中`placeholder`对应的变量名，`value`就是对应的需要传入的值。

示例如下：


```python
BNTraining = tf.placeholder(tf.bool, name='BNTraining')
    keeProb = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    batchImgInput = tf.placeholder(tf.float32, shape=(None, img_size, img_size, n_channels), name='batchImgInput')
    labels = tf.placeholder(tf.float32, shape=(None, label_size), name='Labels')
# .....（此处省略中间网络结构）
# 定义网络的损失和优化方案
loss = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense2, labels=labels), dtype=tf.float32))
train = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
	#特别重要的两行，一定不要掉了
    init = tf.global_variables_initializer()
    sess.run(init)
	#run train 算符调整网络参数，即完成一个batch的训练
	_, cur_loss=sess.run([train, loss],feed_dict={batchImgInput: X, labels: Y, keeProb: keep_prob_train, BNTraining: True})
```

##### 如何保存一个网络


调用`tf.train.Saver()`来完成网络的保存，使用方法非常简单，两行代码即可。  

示例如下：  


```python
# 与session同样的级别定义一个saver。
saver = tf.train.Saver()
# 在需要保存模型的时候使用save(),必须传入的参数是session对象和保存的路径。
saver.save(sess, "Model/ResNet")
```

这里还需要特别提醒一些东西~

###### 1.Saver()的一些参数


常用到的有———


`var_list`：定义需要保存的变量，如果`None`的话是默认保存所有可保存的变量，`API`的用词是`If None, defaults to the list of all saveable objects.`，所以不管是不是`trainable variable`都会被保存下来，并不需要设定`var_list=tf.global_variables()`，因为看到网上很多人这么写，尤其是在提到用`tf.layers.batch_normalization()`的时候，所以我提一下。如果错了的话，也请大家提醒一下我。


`max_to_keep`：保存最近的几个网络，如果你隔一段时间就保存一次，训练时间长的话，肯定就会保存特别多的网络，这是没有必要的，这个参数默认是5，只保留最近的五个。


`keep_checkpoint_every_n_hours`：从时间的角度出发，多久保存一次模型，默认是10,000小时。


还有一些参数，大家有兴趣了解可以去看官方`API`(https://tensorflow.google.cn/api_docs/python/tf/train/Saver)


###### 2.Save()的一些参数


其实主要就一个参数`global_step`，可以标记保存的是第几次被保存的。这个直接看官方示例，非常简单易懂：


```
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
```


###### 3.保存模型后输出的文件


`save()`之后，设定的文件夹下，将会出现四个文件，如下：


```
|Model
|    |--checkpoint
|    |--Model.meta
|    |--Model.data-00000-of-00001
|    |--Model.index
```
`checkpoint`文件是个文本文件，可以直接打开看，示例如下：
```
model_checkpoint_path: "ResNet"
all_model_checkpoint_paths: "ResNet"
```


它定义了模型保存的路径。


`meta`定义了模型的图结构，我们知道`TensorFLow`的模型叫做一个`Graph`，里面有很多的算符`operator`，那么这个`meta`文件就保存了这张图的结构，可以理解为模型的结构。


`index`和`data`保存了模型的所有的变量和对应的数值，方便后续加载使用。


##### 如何加载复用一个网络


如果我们只保存最优的一个网络，加载该模型只需要两行代码。


```python
saver = tf.train.import_meta_graph('Model/ResNet.meta')
saver.restore(sess,tf.train.latest_checkpoint('Model'))
```

第一步，调用`tf.train.import_meta_graph()`，加载网络结构，第二步，调用`restore()`加载变量的值。


接下来对于需要调用的变量需要先找到它们（使用函数`get_tensor_by_name()`，参数由两部分组成，冒号前是算符的名字，冒号后是算符的输出张量的序号，有的图结构中，一个算符可能会输出不止一次张量，本次实训不涉及，大家都填0就行），然后使用它，`placeholder`也是如此，先找到，然后使用`feed_dict`的形式传入你想要的值后，调用`session.run()`，示例如下——


```python
# 四个placeholder
saver = tf.train.import_meta_graph('Model/ResNet.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Model'))
# 调用tf.get_default_graph()获取当前进程的图，前面两行代码已经加载了图了。
graph = tf.get_default_graph()

batchImgInput = graph.get_tensor_by_name("batchImgInput:0")
labels = graph.get_tensor_by_name("Labels:0")
keeProb = graph.get_tensor_by_name("dropout_keep_prob:0")
BNTraining = graph.get_tensor_by_name("BNTraining:0")
# 最后输出算符
out = graph.get_tensor_by_name("model_outputs/BiasAdd:0")

# 构造feed_dict
feed_dict = {batchImgInput: X_v, labels: Y_v, keeProb: 1., BNTraining : False}

# sess.run()
output = sess.run(out, feed_dict=feed_dict)
```


这里如果在模型的搭建阶段没有好好命名的话，很有可能这个时候已经不知道算符节点的名字了，那么大家可以这样——


```python
for op in graph.get_operations():
	print(op.name)
```


这样可以打印所有算符节点的名字，然后大家可以搜索一下关键字，找一找。

#### 编程要求

根据提示，在右侧编辑器补充代码，训练得到你想要的模型，并保存到指定的路径。

#### 测试说明

平台会对你编写的代码进行测试：


**本关只允许你训练1分钟，所以请自行设置合适的batchSize和epoch，以及合适的网络结构。**，平台会计算你训练好的模型在该图片集上的准确率，**如果高于给定的阈值35%**，则说明你的模型训练时在往正确的方向上逐渐收敛，平台将判定为通过本关。


---
开始你的任务吧，祝你成功！
