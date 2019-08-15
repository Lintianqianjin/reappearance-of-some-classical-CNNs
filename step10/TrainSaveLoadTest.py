import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import sys
sys.path.append('step3')
sys.path.append('step9')

from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy

import TrainSaveLoadForUsers

tf.reset_default_graph()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('step10/Model/FinalNet.meta')
    saver.restore(sess, tf.train.latest_checkpoint('step10/Model'))
    graph = tf.get_default_graph()

    # for op in graph.get_operations():
    #     print(op.name)
    # mm = graph.get_tensor_by_name("batch_normalization/moving_mean:0")
    # print(sess.run(mm))
    # exit()

    G_Valid = batchGenerator(batchSize=8, basePath='data/processed/valid_224')
    X_v, Y_v = G_Valid.getBatch()

    batchImgInput = graph.get_tensor_by_name("batchImgInput:0")
    labels = graph.get_tensor_by_name("Labels:0")
    keeProb = graph.get_tensor_by_name("dropout_keep_prob:0")
    try:
        BNTraining = graph.get_tensor_by_name("BNTraining:0")
        batchSize = graph.get_tensor_by_name("InputBatchSize:0")
    except:
        BNTraining, batchSize = None, None


    acc_v = 0

    for i in range(10):
        X_v, Y_v = G_Valid.getBatch()

        if BNTraining is not None and batchSize is not None:
            feed_dict = {batchImgInput: X_v, labels: Y_v, keeProb: 1., BNTraining: False, batchSize: 8}
        else:
            feed_dict = {batchImgInput: X_v, labels: Y_v, keeProb: 1.}

        out = graph.get_tensor_by_name("model_outputs/BiasAdd:0")
        output_v = softmax(
            sess.run(out,
                     feed_dict=feed_dict))

        output_v = returnOneHot(output_v)
        acc_v += computeAccuracy(output_v, Y_v)


    acc_v /= 10

    # print(acc_v)
    if acc_v>0.35:
        print('训练时间40S内，预测准确率高于35%，恭喜通过本关测试！',end='')
    else:
        print('训练时间40S内，预测准确率低于35%，很遗憾并未通过本关测试！',end='')


