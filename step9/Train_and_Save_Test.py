import tensorflow as tf
import sys
sys.path.append('..\\step3')

from generatorCompleted import batchGenerator
from outputsUtils import softmax, returnOneHot, computeAccuracy

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Model/ResNet.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Model'))

    G_Valid = batchGenerator(batchSize=80, basePath='..\\step1\\processed\\valid_224')
    X_v, Y_v = G_Valid.getBatch()

    graph = tf.get_default_graph()

    # for op in graph.get_operations():
    #     print(op.name)
    batchImgInput = graph.get_tensor_by_name("batchImgInput:0")
    labels = graph.get_tensor_by_name("Labels:0")
    keeProb = graph.get_tensor_by_name("dropout_keep_prob:0")
    BNTraining = graph.get_tensor_by_name("BNTraining:0")
    feed_dict = {batchImgInput: X_v, labels: Y_v, keeProb: 1., BNTraining : False}

    out = graph.get_tensor_by_name("model_outputs/BiasAdd:0")

    output_v = softmax(sess.run(out, feed_dict=feed_dict))
    output_v = returnOneHot(output_v)
    acc_v = computeAccuracy(output_v, Y_v)
    print(f'current accuracy: {acc_v}')
