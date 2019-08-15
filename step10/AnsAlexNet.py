import sys
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings('ignore')
sys.path.append('step3')
sys.path.append('step9')
from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy
from prevModules import (Inception_traditional, Inception_parallelAsymmetricConv,
                               Inception_AsymmetricConv,InitialPart,reduction,ResNetBlock)
#********** Begin **********#
# 定义placeholder 开始
keeProb = tf.placeholder(tf.float32, shape=(),name='dropout_keep_prob')
batchImgInput = tf.placeholder(tf.float32, shape=(None, 224, 224, 3),name='batchImgInput')
labels = tf.placeholder(tf.float32, shape=(None, 4),name='Labels')
# 定义placeholder 结束
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
conv4 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)(
    conv3)
# 定义三层直接连接的卷积 结束
# 池化后变为一维 开始
pool3 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv4)
flatten = tf.layers.Flatten()(pool3)
# 池化后变为一维 结束
# 第一层全连接+随机失活 开始
dense1 = tf.layers.Dense(units=256, activation=tf.nn.relu)(flatten)
dropout1 = tf.nn.dropout(dense1, keeProb)
# 第一层全连接+随机失活 结束
# 第三层全连接+随机失活 开始
# dense3 = tf.layers.Dense(units=256, activation=tf.nn.relu)(dropout1)
# dropout3 = tf.nn.dropout(dense3, keeProb)
# 第三层全连接+随机失活 结束
# 额外加了一层全连接层 输出为类别数量 开始
outPuts = tf.layers.Dense(units=4, activation=None,name='model_outputs')(dropout1)
# 额外加了一层全连接层 输出为类别数量 结束
# 定义损失 开始
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outPuts, labels=labels))
# 定义损失 结束
# 定义训练 开始
train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
# 定义训练 结束
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    G_Train = batchGenerator(batchSize=8)
    G_Valid = batchGenerator(batchSize=8, basePath='data/processed/valid_224')
    acc_Train = []
    max_acc = 0
    for i in range(24):
        X, Y = G_Train.getBatch()
        cur_BatchSize = X.shape[0]
        _, cur_loss = sess.run([train, loss],
                               feed_dict={batchImgInput: X, labels: Y, keeProb: 0.8})

        print(i)
        # print(i, end=': loss: ')
        # print(cur_loss)
        acc_v = 0
        # 验证集
        for i in range(10):
            X_v, Y_v = G_Valid.getBatch()
            output_v = softmax(
                sess.run(outPuts,
                         feed_dict={batchImgInput: X_v, labels: Y_v, keeProb: 1.}))
            output_v = returnOneHot(output_v)
            acc_v += computeAccuracy(output_v, Y_v)
        acc_v /= 10
        print('current accuracy: ', str(acc_v))
            # if acc_v > 0.7 and acc_v > max_acc:
            #     max_acc = acc_v
            #     saver.save(sess, "step10/Model/FinalNet")
#********** End **********#