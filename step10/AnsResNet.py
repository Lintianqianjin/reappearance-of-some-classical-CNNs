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


BNTraining = tf.placeholder(tf.bool,name='BNTraining')
keeProb = tf.placeholder(tf.float32, shape=(),name='dropout_keep_prob')
batchImgInput = tf.placeholder(tf.float32, shape=(None, 224, 224, 3),name='batchImgInput')
labels = tf.placeholder(tf.float32, shape=(None, 4),name='Labels')
InputBatchSize = tf.placeholder(tf.int32,name='InputBatchSize')


conv1 = tf.layers.conv2d(batchImgInput, filters=96, kernel_size=11, strides=4,padding='same',
                 activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')
conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, strides=1,padding='same',
                 activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2, padding='same')

resBlock1 = ResNetBlock(pool2, batchNormTraining=BNTraining, batchSize=InputBatchSize)
conv3 = tf.layers.conv2d(resBlock1, filters=128, kernel_size=3,strides=1,padding='same',
                 activation=tf.nn.relu)
resBlock2 = ResNetBlock(conv3, batchNormTraining=BNTraining, batchSize=InputBatchSize)
conv4 =  tf.layers.conv2d(resBlock2, filters=64, kernel_size=3,strides=1,padding='same',
                 activation=tf.nn.relu)
resBlock3 = ResNetBlock(conv4, batchNormTraining=BNTraining, batchSize=InputBatchSize)
conv5 = tf.layers.conv2d(resBlock3, filters=64, kernel_size=3,strides=1,padding='same',
                 activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=2, padding='same')
flattened = tf.layers.flatten(pool3)

dense1 = tf.layers.dense(flattened, units=256)
dropout1 = tf.nn.dropout(dense1, keeProb)
dense2 = tf.layers.dense(dropout1, units=4,name='model_outputs')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense2, labels=labels))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    G_Train = batchGenerator(batchSize=256)
    G_Valid = batchGenerator(batchSize=80, basePath='data/processed/valid_224')

    acc_Train = []
    acc_Val = []
    max_acc = 0

    for i in range(256):

        X, Y = G_Train.getBatch()
        cur_BatchSize = X.shape[0]
        _, cur_loss = sess.run([train, loss],
                               feed_dict={batchImgInput: X, labels: Y, keeProb: 0.8, BNTraining: True,
                                          InputBatchSize: cur_BatchSize})

        if i % 1 == 0:
            print(i, end=': loss: ')
            print(cur_loss)

            # 验证集
            X_v, Y_v = G_Valid.getBatch()
            output_v = softmax(
                sess.run(dense2,
                         feed_dict={batchImgInput: X_v, labels: Y_v, keeProb: 1., BNTraining: False,
                                    InputBatchSize: 80}))
            output_v = returnOneHot(output_v)
            acc_v = computeAccuracy(output_v, Y_v)
            acc_Val.append(acc_v)
            print('current accuracy: '+str(acc_v))

            if acc_v > 0.7 and acc_v > max_acc:
                max_acc = acc_v
                saver.save(sess, "step10/Model/FinalNet")