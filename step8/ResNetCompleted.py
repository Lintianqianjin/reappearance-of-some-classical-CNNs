import tensorflow as tf
import sys
sys.path.append('..\\step7')

from InceptionCompleted import Conv,maxpool


def multiChannelWeightLayer(Inputs, name, batchNormTraining,batchSize = None):
    '''
    对输入完成BatchNorm + relu + Wx_plus_b操作
    :param Inputs: 输入张量
    :param name: 该层的名字
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''

    batchNorm = tf.layers.batch_normalization(Inputs, training=batchNormTraining)
    relu = tf.nn.relu(batchNorm)
    transposed = tf.transpose(relu, [0, 3, 1, 2])
    num_channels = Inputs.get_shape()[-1].value
    size = Inputs.get_shape()[1].value

    weight = tf.get_variable(name=f'{name}_Weight', shape=(size, size), dtype=tf.float32, trainable=True)
    weight_expand = tf.expand_dims(weight, axis=0)
    weight_nchannels = tf.tile(weight_expand, tf.constant([num_channels, 1, 1]))
    batch_expand = tf.expand_dims(weight_nchannels, axis=0)
    weight_final = tf.tile(batch_expand, tf.concat([tf.stack([batchSize,1],axis=0),[1,1]],axis=0))

    WX = tf.matmul(transposed, weight_final)

    bias = tf.get_variable(name=f'{name}_Bias', shape=(size), dtype=tf.float32, trainable=True)
    bias_expand = tf.expand_dims(bias, axis=0)
    bias_size = tf.tile(bias_expand, tf.constant([size, 1]))
    bias_channels_expand = tf.expand_dims(bias_size, axis=0)
    bias_channels = tf.tile(bias_channels_expand, tf.constant([num_channels, 1, 1]))
    bias_batch_expand = tf.expand_dims(bias_channels, axis=0)
    bias_final = tf.tile(bias_batch_expand,tf.concat([tf.stack([batchSize,1],axis=0),[1,1]],axis=0))

    WX_PLUS_B = WX + bias_final

    outputs = tf.transpose(WX_PLUS_B, [0, 2, 3, 1])

    return outputs


def ResNetBlock(Inputs, name, batchNormTraining, bacthSize):
    '''
    堆叠两次（BatchNorm + relu + Wx_plus_b）操作，形成一个残差模块
    :param Inputs: 输入张量
    :param name: 该层的名字
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''
    shortcut = Inputs
    wx_1 = multiChannelWeightLayer(Inputs, batchNormTraining=batchNormTraining, name=f'{name}_firstHalf',batchSize=bacthSize)
    res = multiChannelWeightLayer(wx_1, batchNormTraining=batchNormTraining, name=f'{name}_latterHalf',batchSize=bacthSize)
    outputs = tf.add(shortcut, res)

    return outputs



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
    InputBatchSize = tf.placeholder(tf.int32, name='InputBatchSize')

    conv1 = Conv(batchImgInput, num_kernels=96, kernel_width=11, kernel_height=11, stride_w=4, stride_h=4)
    pool1 = maxpool(conv1, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    conv2 = Conv(pool1, num_kernels=128, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    pool2 = maxpool(conv2, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2)

    resBlock1 = ResNetBlock(pool2, batchNormTraining=BNTraining, name='resblock_1',bacthSize=InputBatchSize)
    conv4 = Conv(resBlock1, num_kernels=128, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    resBlock2 = ResNetBlock(conv4, batchNormTraining=BNTraining, name='resblock_2',bacthSize=InputBatchSize)
    conv5 = Conv(resBlock2, num_kernels=64, kernel_width=3, kernel_height=3, stride_w=1, stride_h=1)
    resBlock3 = ResNetBlock(conv5, batchNormTraining=BNTraining, name='resblock_3',bacthSize=InputBatchSize)
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

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "modelInfo/ResNet")