import tensorflow as tf

# 卷积层
def Conv(Inputs, num_kernels, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1,padding='SAME',name=None):
    '''
    :param Inputs: 上一层的输出，该层的输入
    :param kernel_height: 该层卷积的长
    :param kernel_width: 该层卷积的宽
    :param num_kernels: 该层卷积的数量
    :param stride_h: 该层卷积的移动步长纵向
    :param stride_w: 该层卷积的移动步长横向
    :param name: 该层的名字
    :return:
    '''

    # 获取输入的通道数
    num_channels = Inputs.get_shape()[-1].value

    # 设置卷积核
    kernel = tf.Variable(tf.truncated_normal(
        shape=[kernel_height, kernel_width, num_channels, num_kernels],
        dtype=tf.float32), trainable=True, name='weights')
    # 调用tensorflow的卷积层
    conv = tf.nn.conv2d(Inputs, kernel, strides=[1, stride_h, stride_w, 1], padding=padding)
    # 添加偏执项
    biases = tf.Variable(tf.constant(0.0, shape=[num_kernels], dtype=tf.float32), trainable=True, name='biases')
    Conv_plus_b = tf.nn.bias_add(conv, biases)

    # 激活
    activation = tf.nn.relu(Conv_plus_b)
    return activation

# 平均池化层
def averagePool(Inputs, kernel_h, kernel_w, stride_h=None, stride_w=None, name=None):
    '''
    :param Inputs: 上一层的输出，该层的输入
    :param kernel_h: 池化的高
    :param kernel_w: 池化的宽
    :param stride_h: 移动的步长纵向
    :param stride_w: 移动的步长横向
    :param name: 该层名字
    :return:
    '''
    if stride_h is None:
        stride_h = kernel_h

    if stride_w is None:
        stride_w = kernel_w

    return tf.nn.avg_pool(Inputs,
                          # 池化范围
                          ksize=[1, kernel_h, kernel_w, 1],
                          # 移动步长
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name=name)

# 最大池化层
def maxpool(Inputs, kernel_h, kernel_w, stride_h=None, stride_w=None, name=None):
    '''
    :param Inputs: 上一层的输出，该层的输入
    :param kernel_h: 池化的高
    :param kernel_w: 池化的宽
    :param stride_h: 移动的步长纵向
    :param stride_w: 移动的步长横向
    :param name: 该层名字
    :return:
    '''

    if stride_h is None:
        stride_h = kernel_h

    if stride_w is None:
        stride_w = kernel_w

    return tf.nn.max_pool(Inputs,
                          # 池化范围
                          ksize=[1, kernel_h, kernel_w, 1],
                          # 移动步长
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name=name)

# todo: 根据提示完成三个Inception module和一个reduction的编写
# todo: 根据提示完成main函数中model的编写

#********** Begin **********#
# Inception module 1
def Inception_traditional(Inputs,nfilters_11=64,nfilters_11Before33=64,
                          nfilters_11Before55=48,nfilters_11After33Pool=32,
                          nfilters_33=96,nfilters_55=64,name=None):
    '''
    最基本的Inception模块，拼接不同感受野的卷积结果
    其实传入的参数还能更加细，这里默认所有卷积步长都是1，padding都是same
    :param Inputs: 上一层的输出，该层的输入
    :param nfilters_11: 1×1卷积层的卷积核数
    :param nfilters_11Before33: 3×3卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11Before55: 5×5卷积层前的1×1卷积降维的卷积核数
    :param nfilters_11After33Pool: 3×3池化后的1×1卷积核的数量
    :param nfilters_33: 3×3卷积层的卷积核数
    :param nfilters_55: 5×5卷积层的卷积核数（下面的实现用两个3×3替代5×5，两个3×3的卷积核数都为该参数）
    :param name: 该层的名字
    :return:
    '''

    # 1×1的卷积层


    # 3×3的卷积层


    # 5×5（2个3×3）的卷积层


    # 池化+卷积


    # 在通道维度上拼接各输出


    # return outputs

# Inception module 2 带不对称的卷积
def Inception_AsymmetricConv(Inputs,nfilters_11=192,nfilters_11Before7=128,
                          nfilters_11Before77=128,nfilters_11After33Pool=192,
                          nfilters_7=128,nfilters_77=128,name=None):
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


    # 1×7然后7×1的卷积层


    # 7×1，1×7然后又7×1，1×7的卷积层


    # 池化+卷积


    # 在通道维度上拼接各输出


    # return outputs

# Inception module 3 平行的不对称的卷积
def Inception_parallelAsymmetricConv(Inputs,nfilters_11=320,nfilters_11Before33=384,
                          nfilters_11Before55=448,nfilters_11After33Pool=192,
                          nfilters_33=384,nfilters_55=384,name=None):
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


    # 3×3的卷积层，拼接1×3和3×1 两个并行的卷积


    # 两个3×3的卷积层，第一个3×3正常卷积，第二个拼接1×3和3×1 两个并行的卷积


    # 池化+卷积

    # 在通道维度上拼接各输出


    # return outputs

# 池化和卷积并行的降特征图尺寸的方法
def reduction(Inputs,nfilters_11Before33=192,nfilters_11Before55=192,
              nfilters_33=320,nfilters_55=192,):
    '''
    注意拼接前的最后一次的卷积步长要变成2了
    :param Inputs: 输入
    :param nfilters_11Before33: 3×3卷积前的1×1卷积核数量
    :param nfilters_11Before55: 两个3×3卷积前的1×1卷积核数量
    :param nfilters_33: 3×3卷积核数量
    :param nfilters_55: 两个3×3卷积的核数量
    :return:
    '''

    # 3×3卷积，注意步长


    # 两个3×3卷积 最后一个卷积注意步长


    # 池化，注意步长


    # 拼接

    # return outputs

#********** End **********#

# 模型初始的部分
def InitialPart(Inputs):
    '''
    论文模型中在使用Inception模块之前还是正常的一些卷积和池化
    :param Inputs: 初始img输入
    :return:
    '''
    conv1 = Conv(Inputs,num_kernels=32,kernel_width=3,kernel_height=3,stride_w=2,stride_h=2)
    conv2 = Conv(conv1,num_kernels=32,kernel_width=3,kernel_height=3,stride_w=1,stride_h=1)
    conv3 = Conv(conv2,num_kernels=64,kernel_width=3,kernel_height=3,stride_w=1,stride_h=1)
    pool1 = maxpool(conv3,kernel_h=3,kernel_w=3,stride_h=2,stride_w=2)
    conv4 = Conv(pool1,num_kernels=80,kernel_width=1,kernel_height=1,stride_w=1,stride_h=1)
    conv5 = Conv(conv4,num_kernels=192,kernel_width=3,kernel_height=3,stride_w=1,stride_h=1)
    pool2 = maxpool(conv5,kernel_h=3,kernel_w=3,stride_h=2,stride_w=2)
    return pool2

if __name__ == '__main__':

    # 定义placeholder 开始
    Input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='Imgs')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    Labels = tf.placeholder(shape=(None, 4), dtype=tf.float32, name='Labels')
    # 定义placeholder 结束

    #********** Begin **********#
    # 模型初始部分
    processedInitially = InitialPart(Input)

    # 叠三个nception_traditional

    # reduction一次

    # 叠四个Inception_AsymmetricConv

    # reduction一次

    # 叠三个Inception_parallelAsymmetricConv

    # 平均池化成一张图一个值
    # featureSize = Inception_parallelAsymmetric_3.get_shape()[1].value

    # flatten然后dropout然后全连接

    #********** End **********#

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense, labels=Labels))
    # train = tf.train.AdamOptimizer().minimize(loss)


    # 以下不要改动
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "userModelInfo/InceptionNet.ckpt")


