# 论文：ImageNet Classification with Deep Convolutional Neural Networks
# 论文一作的名字叫Alex，所以网络叫AlexNet

import tensorflow as tf
import sys
sys.path.append('..\\step2')

from generatorCompleted import batchGenerator


# 定义超参数 开始
batchSize = 256
n_channels = 3
img_size = 224
learning_rate = 0.01
label_size = 4
keep_prob_train = 0.8
keep_prob_val = 1
# 定义超参数 结束

# 定义placeholder 开始
keeProb = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
batchImgInput = tf.placeholder(tf.float32, shape=(None, img_size, img_size, n_channels), name='batchImgInput')
labels = tf.placeholder(tf.float32, shape=(None, label_size))
# 定义placeholder 结束

# user todo: 根据提示和现有代码补充完整
#********** Begin **********#

# 第一层卷积+归一化+池化 开始
# 要求使用tf.keras.layers.Conv2D()
# 参数：96个卷积核,大小11×11, 步长4，padding设置为'valid',激活函数relu
# conv1 =

# 要求使用tf.nn.local_response_normalization()
# alpha取1e-4, beta取0.75, depth_radius取2, bias取2.0
# lrn1 = tf.nn.local_response_normalization()

# 要求使用tf.keras.layers.MaxPooling2D()
# 最大池化范围为3×3，步长为2，padding为valid
# pool1 = tf.keras.layers.MaxPooling2D()
# 第一层卷积+归一化+池化 结束


# 第二层卷积+归一化+池化 开始
# 要求使用tf.keras.layers.Conv2D()
# 参数：256个卷积核,大小5×5, 步长1，padding设置为'same',激活函数relu
# conv2 = tf.keras.layers.Conv2D()

# 要求同上
# lrn2 = tf.nn.local_response_normalization()

# 要求同上
# pool2 = tf.keras.layers.MaxPooling2D()
# 第二层卷积+归一化+池化 结束


# 定义三层直接连接的卷积 开始
# 要求堆叠三个卷积层，卷积核大小均为3×3，步长1，padding为same，激活函数用relu
# 卷积核数分别为192 192 128
# conv3 =
# conv4 =
# conv5 =
# 定义三层直接连接的卷积 结束

# 池化后变为一维 开始
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv5)
flatten = tf.keras.layers.Flatten()(pool3)
# 池化后变为一维 结束

# 第一层全连接+随机失活 开始
# 要求使用tf.keras.layers.Dense()
# 该层要求512个神经元，激活函数relu
# dense1 = tf.keras.layers.Dense()
# dropout1 = tf.nn.dropout(dense1, keeProb)
# 第一层全连接+随机失活 结束
#********** End **********#


# 第二层全连接+随机失活 开始
dense2 = tf.keras.layers.Dense(units=512, activation='relu')(dropout1)
dropout2 = tf.nn.dropout(dense2, keeProb)
# 第二层全连接+随机失活 结束

# 第三层全连接+随机失活 开始
dense3 = tf.keras.layers.Dense(units=256, activation='relu')(dropout2)
dropout3 = tf.nn.dropout(dense3, keeProb)
# 第三层全连接+随机失活 结束

# 额外加了一层全连接层 输出为类别数量 开始
dense4 = tf.keras.layers.Dense(units=label_size, activation=None)(dropout3)
# 额外加了一层全连接层 输出为类别数量 结束

# 定义损失 开始
loss = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense4, labels=labels), tf.float32))
# 定义损失 结束

# 定义训练 开始
train = tf.train.AdamOptimizer().minimize(loss)
# 定义训练 结束



# 以下不用管
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "userModelInfo/AlexNet")