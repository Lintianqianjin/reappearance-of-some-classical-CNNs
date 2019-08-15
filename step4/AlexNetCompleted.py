import tensorflow as tf
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
#----以下是答案部分 begin----#

# 定义placeholder 开始
keeProb = tf.placeholder(tf.float32, shape=())
batchImgInput = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(None, 4))

# 第一层卷积+归一化+池化 开始
conv1 = tf.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation=tf.nn.relu)(
    batchImgInput)
lrn1 = tf.nn.local_response_normalization(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
pool1 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)
# 第一层卷积+归一化+池化 结束

# 第二层卷积+归一化+池化 开始
conv2 = tf.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)(
    pool1)
lrn2 = tf.nn.local_response_normalization(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
pool2 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv2)
# 第二层卷积+归一化+池化 结束

# 定义三层直接连接的卷积 开始
conv3 = tf.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(
    pool2)
conv4 = tf.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(
    conv3)
conv5 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(
    conv4)
# 定义三层直接连接的卷积 结束

# 池化后变为一维 开始
pool3 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv5)
flatten = tf.layers.Flatten()(pool3)
# 池化后变为一维 结束

# 第一层全连接+随机失活 开始
dense1 = tf.layers.Dense(units=512, activation=tf.nn.relu)(flatten)
dropout1 = tf.nn.dropout(dense1, keeProb)
# 第一层全连接+随机失活 结束

# 第二层全连接+随机失活 开始
dense2 = tf.layers.Dense(units=512, activation=tf.nn.relu)(dropout1)
dropout2 = tf.nn.dropout(dense2, keeProb)
# 第二层全连接+随机失活 结束

# 第三层全连接+随机失活 开始
dense3 = tf.layers.Dense(units=256, activation=tf.nn.relu)(dropout2)
dropout3 = tf.nn.dropout(dense3, keeProb)
# 第三层全连接+随机失活 结束

# 额外加了一层全连接层 输出为类别数量 开始
outPuts = tf.layers.Dense(units=4, activation=None)(dropout3)
# 额外加了一层全连接层 输出为类别数量 结束

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outPuts, labels=labels))
train = tf.train.AdamOptimizer().minimize(loss)

#----以上是答案部分 end----#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename="step4/modelInfo/AlexNet",
                               graph=tf.get_default_graph())

tf.reset_default_graph()