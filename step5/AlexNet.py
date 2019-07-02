# 论文：ImageNet Classification with Deep Convolutional Neural Networks
# 论文一作的名字叫Alex，所以网络叫AlexNet

import tensorflow as tf
import numpy as np
import sys
sys.path.append('..\\step3')

from generatorCompleted import batchGenerator

from outputsUtils import softmax,returnOneHot,computeAccuracy

# 定义超参数 开始
batchSize = 256
n_channels = 3
img_size = 224
learning_rate = 0.01
label_size = 4
keep_prob_train = 0.8
keep_prob_val = 1
# 定义超参数 结束
keeProb = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
batchImgInput = tf.placeholder(tf.float32, shape=(None, img_size, img_size, n_channels), name='batchImgInput')
labels = tf.placeholder(tf.float32, shape=(None, label_size))

conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),padding='valid', activation='relu')(batchImgInput)
lrn1 = tf.nn.local_response_normalization(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
# lrn1 = tf.keras.layers.BatchNormalization()(conv1,training = True)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)

conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),padding='same', activation='relu')(pool1)
lrn2 = tf.nn.local_response_normalization(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)
# lrn2 = tf.keras.layers.BatchNormalization()(conv2,training = True)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv2)

conv3 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu')(pool2)
conv4 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu')(conv3)
conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu')(conv4)

pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv5)
flatten = tf.keras.layers.Flatten()(pool3)

dense1 = tf.keras.layers.Dense(units=512, activation='relu')(flatten)
dropout1 = tf.nn.dropout(dense1, keeProb)

dense2 = tf.keras.layers.Dense(units=512, activation='relu')(dropout1)
dropout2 = tf.nn.dropout(dense2, keeProb)

dense3 = tf.keras.layers.Dense(units=256, activation='relu')(dropout2)
dropout3 = tf.nn.dropout(dense3, keeProb)

dense4 = tf.keras.layers.Dense(units=label_size, activation=None)(dropout3)

loss = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense4, labels=labels), tf.float32))

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
train = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    G_Train = batchGenerator(batchSize=256)
    G_Valid = batchGenerator(batchSize=80,basePath='..\\step1\\processed\\valid_224')

    acc_Train = []
    acc_Val = []


    for i in range(1024):

        X, Y = G_Train.getBatch()

        _,cur_loss = sess.run([train,loss], feed_dict={batchImgInput: X, labels: Y,keeProb:keep_prob_train})

        if i%8 == 0:
            print(i, end=': loss: ')
            print(cur_loss)

            # 验证集
            X_v, Y_v = G_Valid.getBatch()
            output_v = softmax(sess.run(dense4,feed_dict={batchImgInput: X_v, labels: Y_v,keeProb:keep_prob_val}))
            output_v = returnOneHot(output_v)
            acc_v = computeAccuracy(output_v,Y_v)
            acc_Val.append(acc_v)
            print(f'current accuracy: {acc_v}')

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot([i for i in range(1, len(acc_Val) + 1)], acc_Val, label=u'验证集准确率')
plt.legend()
# todo: 更换模型时要改名字
plt.show()
