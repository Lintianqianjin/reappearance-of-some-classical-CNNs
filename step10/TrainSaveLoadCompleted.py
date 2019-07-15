import sys

import tensorflow as tf

sys.path.append('..\\step7')
sys.path.append('..\\step3')
sys.path.append('..\\step9')
sys.path.append('..\\step8')

from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy
from InceptionCompleted import Conv, maxpool
from ResNetCompleted import multiChannelWeightLayer,ResNetBlock

if __name__ == '__main__':
    # 定义超参数 开始
    batchSize = 256
    n_channels = 3
    img_size = 224
    learning_rate = 0.01
    label_size = 4
    keep_prob_train = 0.8
    keep_prob_val = 1
    # 定义超参数 结束
    BNTraining = tf.placeholder(tf.bool, name='BNTraining')
    keeProb = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    batchImgInput = tf.placeholder(tf.float32, shape=(None, img_size, img_size, n_channels), name='batchImgInput')
    labels = tf.placeholder(tf.float32, shape=(None, label_size), name='Labels')

    conv1 = Conv(batchImgInput, num_kernels=96, kernel_width=11, kernel_height=11, stride_w=4, stride_h=4)
    pool1 = maxpool(conv1, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    conv2 = Conv(pool1, num_kernels=128, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    pool2 = maxpool(conv2, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2)

    resBlock1 = ResNetBlock(pool2, batchNormTraining=BNTraining, name='resblock_1')
    conv4 = Conv(resBlock1, num_kernels=128, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    resBlock2 = ResNetBlock(conv4, batchNormTraining=BNTraining, name='resblock_2')
    conv5 = Conv(resBlock2, num_kernels=64, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    resBlock3 = ResNetBlock(conv5, batchNormTraining=BNTraining, name='resblock_3')
    conv6 = Conv(resBlock3, num_kernels=64, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)

    pool3 = maxpool(conv6, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    flattened = tf.layers.flatten(pool3)

    dense1 = tf.layers.dense(flattened, units=256)
    dropout1 = tf.nn.dropout(dense1, keeProb)
    dense2 = tf.layers.dense(dropout1, units=4, name='model_outputs')

    loss = tf.reduce_mean(
        tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense2, labels=labels), dtype=tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        G_Train = batchGenerator(batchSize=80)
        G_Valid = batchGenerator(batchSize=80, basePath='..\\step1\\processed\\valid_224')

        acc_Train = []
        acc_Val = []
        max_acc = 0

        for i in range(128):

            X, Y = G_Train.getBatch()

            _, cur_loss = sess.run([train, loss],
                                   feed_dict={batchImgInput: X, labels: Y, keeProb: keep_prob_train, BNTraining: True})

            if i % 1 == 0:
                print(i, end=': loss: ')
                print(cur_loss)

                # 验证集
                X_v, Y_v = G_Valid.getBatch()
                output_v = softmax(
                    sess.run(dense2,
                             feed_dict={batchImgInput: X_v, labels: Y_v, keeProb: keep_prob_val, BNTraining: False}))
                output_v = returnOneHot(output_v)
                acc_v = computeAccuracy(output_v, Y_v)
                acc_Val.append(acc_v)
                print(f'current accuracy: {acc_v}')

                if acc_v > 0.7 and acc_v > max_acc:
                    max_acc = acc_v
                    saver.save(sess, "Model/FinalNet")

        print('****')
        print(max_acc)

    #
    # import matplotlib.pyplot as plt
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot([i for i in range(1, len(acc_Val) + 1)], acc_Val, label=u'验证集准确率')
    # plt.legend()
    # plt.show()
