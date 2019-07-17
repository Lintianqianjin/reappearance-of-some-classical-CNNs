# 论文：ImageNet Classification with Deep Convolutional Neural Networks
# 论文一作的名字叫Alex，所以网络叫AlexNet

import tensorflow as tf

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

# 第一层卷积+归一化+池化 开始
conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(
    batchImgInput)
lrn1 = tf.nn.local_response_normalization(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)
# 第一层卷积+归一化+池化 结束

# 第二层卷积+归一化+池化 开始
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(
    pool1)
lrn2 = tf.nn.local_response_normalization(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv2)
# 第二层卷积+归一化+池化 结束

# 定义三层直接连接的卷积 开始
conv3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    pool2)
conv4 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    conv3)
conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
    conv4)
# 定义三层直接连接的卷积 结束

# 池化后变为一维 开始
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv5)
flatten = tf.keras.layers.Flatten()(pool3)
# 池化后变为一维 结束

# 第一层全连接+随机失活 开始
dense1 = tf.keras.layers.Dense(units=512, activation='relu')(flatten)
dropout1 = tf.nn.dropout(dense1, keeProb)
# 第一层全连接+随机失活 结束

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

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "ModelInfo/AlexNet")