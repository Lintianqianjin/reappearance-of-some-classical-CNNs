import tensorflow as tf

# Inception module 1
def Inception_traditional(Inputs, nfilters_11=64, nfilters_11Before33=64,
                          nfilters_11Before55=48, nfilters_11After33Pool=32,
                          nfilters_33=96, nfilters_55=64, name=None):
    '''
    最基本的Inception模块，拼接不同感受野的卷积结果
    其实传入的参数还能更加细，这里默认所有卷积步长都是1，padding都是same
    :param Inputs: 上一层的输出，该层的输入
    :param nfilters_11: 1×1卷积层的卷积核数
    :param nfilters_11Before33: 3×3卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11Before55: 5×5卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11After33Pool: 3×3池化后的1×1卷积核的数量
    :param nfilters_33: 3×3卷积层的卷积核数
    :param nfilters_55: 5×5卷积层的卷积核数（下面的实现用俩个3×3替代了5×5，两个3×3的卷积核数都为该参数）
    :param name: 该层的名字
    :return:
    '''

    # 1×1的卷积层
    conv1 = tf.layers.conv2d(inputs=Inputs, filters=nfilters_11, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 3×3的卷积层
    conv2_1 = tf.layers.conv2d(inputs=Inputs, filters=nfilters_11Before33, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=nfilters_33, kernel_size=3, strides=1, padding='same',
                               activation=tf.nn.relu)

    # 5×5的卷积层
    conv3_1 = tf.layers.conv2d(inputs=Inputs, filters=nfilters_11Before55, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=nfilters_55, kernel_size=3, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=nfilters_55, kernel_size=3, strides=1, padding='same',
                               activation=tf.nn.relu)

    # 池化+卷积
    pool = tf.layers.average_pooling2d(inputs=Inputs, pool_size=3, strides=1, padding='same')
    conv4 = tf.layers.conv2d(inputs=pool, filters=nfilters_11After33Pool, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1, conv2_2, conv3_3, conv4], axis=-1)

    return outputs


# Inception module 2 带不对称的卷积
def Inception_AsymmetricConv(Inputs, nfilters_11=192, nfilters_11Before7=128,
                             nfilters_11Before77=128, nfilters_11After33Pool=192,
                             nfilters_7=128, nfilters_77=128, name=None):
    '''
    将n×n的卷积变成连续的1×n和n×1的两次卷积
    其实这一层的参数也不止这么多，不过大概是这么个意思
    有兴趣的朋友可以让参数更加具体地描述该模块
    步长都默认1
    :param Inputs: 输入
    :param nfilters_11: 1×1卷积层的卷积核数
    :param nfilters_11Before7: 1×7然后7×1卷积前1×1卷积核数
    :param nfilters_11Before77: 7×1，1×7然后又7×1，1×7卷积前1×1的卷积核数
    :param nfilters_11After33Pool: 3×3池化后的1×1卷积核的数量
    :param nfilters_7: 1×7然后7×1卷积的卷积核数
    :param nfilters_77: 7×1，1×7然后又7×1，1×7卷积的卷积核数
    :param name: 该层的名字
    :return:
    '''

    # 1×1的卷积层
    conv1 = tf.layers.conv2d(Inputs, filters=nfilters_11, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 1×7然后7×1的卷积层
    conv2_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before7, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(conv2_1, filters=nfilters_7, kernel_size=(1, 7), strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_3 = tf.layers.conv2d(conv2_2, filters=nfilters_7, kernel_size=(7, 1), strides=1, padding='same',
                               activation=tf.nn.relu)

    # 7×1，1×7然后又7×1，1×7的卷积层
    conv3_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before77, kernel_size=1, strides=1)
    conv3_2 = tf.layers.conv2d(conv3_1, filters=nfilters_77, kernel_size=(7, 1), strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_3 = tf.layers.conv2d(conv3_2, filters=nfilters_77, kernel_size=(1, 7), strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_4 = tf.layers.conv2d(conv3_3, filters=nfilters_77, kernel_size=(7, 1), strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_5 = tf.layers.conv2d(conv3_4, filters=nfilters_77, kernel_size=(1, 7), strides=1, padding='same',
                               activation=tf.nn.relu)

    # 池化+卷积
    pool = tf.layers.average_pooling2d(Inputs, pool_size=3, strides=1, padding='same')
    conv4 = tf.layers.conv2d(pool, filters=nfilters_11After33Pool, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1, conv2_3, conv3_5, conv4], axis=-1)

    return outputs


# Inception module 3 平行的不对称的卷积
def Inception_parallelAsymmetricConv(Inputs, nfilters_11=320, nfilters_11Before33=384,
                                     nfilters_11Before55=448, nfilters_11After33Pool=192,
                                     nfilters_33=384, nfilters_55=384, name=None):
    '''
    将1×n和n×1的两个卷积并行操作，然后拼接起来
    :param Inputs: 输入
    :param nfilters_11: 1×1卷积层的卷积核数
    :param nfilters_11Before33: 3×3卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11Before55: 5×5卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11After33Pool: 3×3池化后的1×1卷积核的数量
    :param nfilters_33: 平行的1×3和3×1方式卷积的卷积核数
    :param nfilters_55: 两个3×3构成的卷积层，但是第二个3×3会用平行的1×3和3×1方式卷积
    :param name:
    :return:
    '''

    # 1×1的卷积层
    conv1 = tf.layers.conv2d(Inputs, filters=nfilters_11, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 3×3的卷积层
    conv2_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before33, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_21 = tf.layers.conv2d(conv2_1, filters=nfilters_33, kernel_size=(1, 3), strides=1, padding='same',
                                activation=tf.nn.relu)
    conv2_22 = tf.layers.conv2d(conv2_1, filters=nfilters_33, kernel_size=(3, 1), strides=1, padding='same',
                                activation=tf.nn.relu)
    conv2_3 = tf.concat([conv2_21, conv2_22], axis=-1)

    # 两个3×3的卷积层
    conv3_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before55, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(conv3_1, filters=nfilters_55, kernel_size=3, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv3_31 = tf.layers.conv2d(conv3_2, filters=nfilters_55, kernel_size=(1, 3), strides=1, padding='same',
                                activation=tf.nn.relu)
    conv3_32 = tf.layers.conv2d(conv3_2, filters=nfilters_55, kernel_size=(3, 1), strides=1, padding='same',
                                activation=tf.nn.relu)
    conv3_4 = tf.concat([conv3_31, conv3_32], axis=-1)

    # 池化+卷积
    pool = tf.layers.average_pooling2d(Inputs, pool_size=3, strides=1, padding='same')
    conv4 = tf.layers.conv2d(pool, filters=nfilters_11After33Pool, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1, conv2_3, conv3_4, conv4], axis=-1)

    return outputs


# 池化和卷积并行的降特征图尺寸的方法
def reduction(Inputs, nfilters_11Before33=192, nfilters_11Before55=192,
              nfilters_33=320, nfilters_55=192, ):
    '''
    注意拼接前的最后一次的卷积步长要变成2了
    :param Inputs: 输入
    :param nfilters_11Before33: 3×3卷积前的1×1卷积核数量
    :param nfilters_11Before55: 两个3×3卷积前的1×1卷积核数量
    :param nfilters_33: 3×3卷积核数量
    :param nfilters_55: 两个3×3卷积的核数量
    :return:
    '''

    # 3×3卷积
    conv1_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before33, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(conv1_1, filters=nfilters_33, kernel_size=3, strides=2, padding='same',
                               activation=tf.nn.relu)

    # 两个3×3卷积
    conv2_1 = tf.layers.conv2d(Inputs, filters=nfilters_11Before55, kernel_size=1, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(conv2_1, filters=nfilters_55, kernel_size=3, strides=1, padding='same',
                               activation=tf.nn.relu)
    conv2_3 = tf.layers.conv2d(conv2_2, filters=nfilters_55, kernel_size=3, strides=2, padding='same',
                               activation=tf.nn.relu)

    # 池化
    pool = tf.layers.average_pooling2d(Inputs, pool_size=3, strides=2, padding='same')

    # 拼接
    outputs = tf.concat([conv1_2, conv2_3, pool], axis=-1)

    return outputs


# 模型初始的部分
def InitialPart(Inputs):
    '''
    论文模型中在使用Inception模块之前还是正常的一些卷积和池化
    :param Inputs: 初始img输入
    :return:
    '''
    conv1 = tf.layers.conv2d(Inputs, filters=32, kernel_size=3, strides=2, padding='same',
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=1, padding='same',
                             activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1, padding='same',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=2, padding='same')
    conv4 = tf.layers.conv2d(pool1, filters=80, kernel_size=1, strides=1, padding='same',
                             activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, filters=192, kernel_size=3, strides=1, padding='same',
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv5, pool_size=3, strides=2, padding='same')
    return pool2



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
averagePool1 = tf.layers.average_pooling2d(Inception_parallelAsymmetric_1,pool_size=featureSize,strides=1, padding='same')

flattened = tf.layers.flatten(averagePool1)
dropout = tf.nn.dropout(flattened, keep_prob)
outputs = tf.layers.dense(dropout, units=4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=Labels))
train = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename='step7/modelInfo/InceptionNet',
                               graph=tf.get_default_graph())

tf.reset_default_graph()