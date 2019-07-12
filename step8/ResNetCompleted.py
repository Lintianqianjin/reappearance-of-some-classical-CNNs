import tensorflow as tf
import sys
sys.path.append('..\\step7')
sys.path.append('..\\step3')

from generatorCompleted import batchGenerator
from outputsUtils import softmax,returnOneHot,computeAccuracy
from InceptionCompleted import Conv,maxpool

def multiChannelWeightLayer(Inputs,name):

    batchNorm = tf.layers.batch_normalization(Inputs, training=True)
    relu = tf.nn.relu(batchNorm)
    transposed = tf.transpose(relu, [0, 3, 1, 2])
    num_channels = Inputs.get_shape()[-1].value
    size = Inputs.get_shape()[1].value
    batch = Inputs.get_shape()[0].value

    weight = tf.get_variable(name=f'{name}_Weight', shape=(size, size), dtype=tf.float32, trainable=True)
    weight_expand = tf.expand_dims(weight, axis=0)
    weight_nchannels = tf.tile(weight_expand, tf.constant([num_channels, 1, 1]))
    batch_expand = tf.expand_dims(weight_nchannels, axis=0)
    weight_final = tf.tile(batch_expand, tf.constant([batch, 1, 1, 1]))

    WX = tf.matmul(transposed, weight_final)

    bias = tf.get_variable(name=f'{name}_Bias', shape=(size), dtype=tf.float32, trainable=True)
    bias_expand = tf.expand_dims(bias, axis=0)
    bias_size = tf.tile(bias_expand, tf.constant([size, 1]))
    bias_channels_expand = tf.expand_dims(bias_size, axis=0)
    bias_channels = tf.tile(bias_channels_expand, tf.constant([num_channels, 1, 1]))
    bias_batch_expand = tf.expand_dims(bias_channels, axis=0)
    bias_final = tf.tile(bias_batch_expand, tf.constant([batch, 1, 1, 1]))

    WX_PLUS_B = WX + bias_final

    outputs = tf.transpose(WX_PLUS_B, [0, 2, 3, 1])

    return outputs


def ResNetBlock(Inputs,name):
    shortcut = Inputs
    wx_1 = multiChannelWeightLayer(Inputs,name=f'{name}_firstHalf')
    res = multiChannelWeightLayer(wx_1,name=f'{name}_latterHalf')
    outputs = tf.add(shortcut,res)

    return outputs



if __name__ == '__main__':
    # 定义超参数 开始
    pass
