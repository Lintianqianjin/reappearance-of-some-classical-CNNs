import sys
import tensorflow as tf

sys.path.append('step3')
sys.path.append('step7')
sys.path.append('step8')
sys.path.append('step9')

from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy
from InceptionCompleted import (Inception_traditional, Inception_parallelAsymmetricConv,
                               Inception_AsymmetricConv,InitialPart,reduction)

from ResNetCompleted import ResNetBlock



# 定义placeholder 开始
Input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='Imgs')
keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
Labels = tf.placeholder(shape=(None, 4), dtype=tf.float32, name='Labels')
# 定义placeholder 结束

# 模型初始部分
processedInitially = InitialPart(Input)
Inception_traditional_1 = Inception_traditional(processedInitially)

reduction_1 = reduction(Inception_traditional_1)

Inception_Asymmetric_1 = Inception_AsymmetricConv(reduction_1)

reduction_2 = reduction(Inception_Asymmetric_1)

Inception_parallelAsymmetric_1 = Inception_parallelAsymmetricConv(reduction_2)

featureSize = Inception_parallelAsymmetric_1.get_shape()[1].value
averagePool1 = tf.layers.average_pooling2d(Inception_parallelAsymmetric_1, pool_size=featureSize, strides=1,
                                           padding='same')

flattened = tf.layers.flatten(averagePool1)
dropout = tf.nn.dropout(flattened, keep_prob)
outputs = tf.layers.dense(dropout, units=4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=Labels))
train = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    G_Train = batchGenerator(batchSize=128)
    G_Valid = batchGenerator(batchSize=80, basePath='data/processed/valid_224')

    acc_Train = []
    acc_Val = []
    max_acc = 0

    for i in range(256):

        X, Y = G_Train.getBatch()
        cur_BatchSize = X.shape[0]
        _, cur_loss = sess.run([train, loss],
                               feed_dict={Input: X, Labels: Y, keep_prob: 0.8})

        if i % 1 == 0:
            print(i, end=': loss: ')
            print(cur_loss)

            # 验证集
            X_v, Y_v = G_Valid.getBatch()
            output_v = softmax(
                sess.run(outputs,
                         feed_dict={Input: X_v, Labels: Y_v, keep_prob: 1.}))
            output_v = returnOneHot(output_v)
            acc_v = computeAccuracy(output_v, Y_v)
            acc_Val.append(acc_v)
            print('current accuracy: ',str(acc_v))

            if acc_v > 0.7 and acc_v > max_acc:
                max_acc = acc_v
                saver.save(sess, "step10/Model/FinalNet")

