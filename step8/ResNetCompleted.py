import tensorflow as tf

def multiChannelWeightLayer(Inputs, batchNormTraining,batchSize):
    '''
    对输入完成BatchNorm + relu + Wx_plus_b操作
    :param Inputs: 输入张量
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''

    batchNorm = tf.layers.batch_normalization(Inputs, training=batchNormTraining)
    relu = tf.nn.relu(batchNorm)
    transposed = tf.transpose(relu, [0, 3, 1, 2])
    num_channels = Inputs.get_shape()[-1].value
    size = Inputs.get_shape()[1].value

    weight = tf.Variable(tf.truncated_normal(shape=(size, size)), dtype=tf.float32, trainable=True)
    weight_expand = tf.expand_dims(weight, axis=0)
    weight_nchannels = tf.tile(weight_expand, tf.constant([num_channels, 1, 1]))
    batch_expand = tf.expand_dims(weight_nchannels, axis=0)
    weight_final = tf.tile(batch_expand, tf.concat([tf.stack([batchSize,1],axis=0),[1,1]],axis=0))

    WX = tf.matmul(transposed, weight_final)

    bias = tf.Variable(tf.truncated_normal(shape=(size,)), dtype=tf.float32, trainable=True)
    bias_expand = tf.expand_dims(bias, axis=0)
    bias_size = tf.tile(bias_expand, tf.constant([size, 1]))
    bias_channels_expand = tf.expand_dims(bias_size, axis=0)
    bias_channels = tf.tile(bias_channels_expand, tf.constant([num_channels, 1, 1]))
    bias_batch_expand = tf.expand_dims(bias_channels, axis=0)
    bias_final = tf.tile(bias_batch_expand,tf.concat([tf.stack([batchSize,1],axis=0),[1,1]],axis=0))

    WX_PLUS_B = WX + bias_final

    outputs = tf.transpose(WX_PLUS_B, [0, 2, 3, 1])

    return outputs


def ResNetBlock(Inputs, batchNormTraining, batchSize):
    '''
    堆叠两次（BatchNorm + relu + Wx_plus_b）操作，形成一个残差模块
    :param Inputs: 输入张量
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''
    shortcut = Inputs
    wx_1 = multiChannelWeightLayer(Inputs, batchNormTraining=batchNormTraining, batchSize=batchSize)
    res = multiChannelWeightLayer(wx_1, batchNormTraining=batchNormTraining, batchSize=batchSize)
    outputs = tf.add(shortcut, res)

    return outputs





BNTraining = tf.placeholder(tf.bool)
keeProb = tf.placeholder(tf.float32, shape=())
batchImgInput = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
labels = tf.placeholder(tf.float32, shape=(None, 4))
InputBatchSize = tf.placeholder(tf.int32)


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
outputs = tf.layers.dense(dropout1, units=4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=labels))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename='step8/modelInfo/ResNet',
                               graph=tf.get_default_graph())

tf.reset_default_graph()