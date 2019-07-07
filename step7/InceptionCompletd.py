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
    with tf.name_scope(name) as scope:
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
        activation = tf.nn.relu(Conv_plus_b, name=scope)
        return activation

def averagePool(Inputs, kernel_h, kernel_w, stride_h, stride_w, name=None):
    '''
    :param Inputs: 上一层的输出，该层的输入
    :param kernel_h: 池化的高
    :param kernel_w: 池化的宽
    :param stride_h: 移动的步长纵向
    :param stride_w: 移动的步长横向
    :param name: 该层名字
    :return:
    '''

    return tf.nn.avg_pool(Inputs,
                          # 池化范围
                          ksize=[1, kernel_h, kernel_w, 1],
                          # 移动步长
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name=name)

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
    :param nfilters_55: 5×5卷积层的卷积核数（下面的实现用俩个3×3替代了5×5，两个3×3的卷积核数都为该参数）
    :param name: 该层的名字
    :return:
    '''

    # 1×1的卷积层
    conv1 = Conv(Inputs,num_kernels=nfilters_11,kernel_height=1,kernel_width=1)

    # 3×3的卷积层
    conv2 = Conv(Inputs,num_kernels=nfilters_11Before33,kernel_height=1,kernel_width=1)
    conv2 = Conv(conv2,num_kernels=nfilters_33,kernel_height=3,kernel_width=3)

    # 5×5的卷积层
    conv3 = Conv(Inputs,num_kernels=nfilters_11Before55,kernel_height=1,kernel_width=1)
    conv3 = Conv(conv3,num_kernels=nfilters_55,kernel_height=3,kernel_width=3)
    conv3 = Conv(conv3,num_kernels=nfilters_55,kernel_height=3,kernel_width=3)

    # 池化+卷积
    pool = averagePool(Inputs,kernel_h=3,kernel_w=3,stride_h=1,stride_w=1)
    conv4 = Conv(pool,num_kernels= nfilters_11After33Pool,kernel_height=1,kernel_width=1)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1,conv2,conv3,conv4],axis=-1)

    return outputs

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
    conv1 = Conv(Inputs,num_kernels=nfilters_11,kernel_height=1,kernel_width=1)

    # 1×7然后7×1的卷积层
    conv2 = Conv(Inputs,num_kernels=nfilters_11Before7,kernel_height=1,kernel_width=1)
    conv2 = Conv(conv2,num_kernels=nfilters_7,kernel_height=1,kernel_width=7)
    conv2 = Conv(conv2,num_kernels=nfilters_7,kernel_height=7,kernel_width=1)

    # 7×1，1×7然后又7×1，1×7的卷积层
    conv3 = Conv(Inputs,num_kernels=nfilters_11Before77,kernel_height=1,kernel_width=1)
    conv3 = Conv(conv3,num_kernels=nfilters_77,kernel_height=7,kernel_width=1)
    conv3 = Conv(conv3,num_kernels=nfilters_77,kernel_height=1,kernel_width=7)
    conv3 = Conv(conv3, num_kernels=nfilters_77, kernel_height=7, kernel_width=1)
    conv3 = Conv(conv3, num_kernels=nfilters_77, kernel_height=1, kernel_width=7)

    # 池化+卷积
    pool = averagePool(Inputs,kernel_h=3,kernel_w=3,stride_h=1,stride_w=1)
    conv4 = Conv(pool,num_kernels= nfilters_11After33Pool,kernel_height=1,kernel_width=1)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1,conv2,conv3,conv4],axis=-1)

    return outputs

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
    conv1 = Conv(Inputs, num_kernels=nfilters_11, kernel_height=1, kernel_width=1)

    # 3×3的卷积层
    conv2 = Conv(Inputs, num_kernels=nfilters_11Before33, kernel_height=1, kernel_width=1)
    conv2_1 = Conv(conv2, num_kernels=nfilters_33, kernel_height=1, kernel_width=3)
    conv2_2 = Conv(conv2, num_kernels=nfilters_33, kernel_height=3, kernel_width=1)
    conv2 = tf.concat([conv2_1,conv2_2],axis=-1)

    # 两个3×3的卷积层
    conv3 = Conv(Inputs, num_kernels=nfilters_11Before55, kernel_height=1, kernel_width=1)
    conv3 = Conv(conv3, num_kernels=nfilters_55, kernel_height=3, kernel_width=3)
    conv3_1 = Conv(conv3, num_kernels=nfilters_55, kernel_height=1, kernel_width=3)
    conv3_2 = Conv(conv3, num_kernels=nfilters_55, kernel_height=3, kernel_width=1)
    conv3 = tf.concat([conv3_1, conv3_2], axis=-1)

    # 池化+卷积
    pool = averagePool(Inputs, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1)
    conv4 = Conv(pool, num_kernels=nfilters_11After33Pool, kernel_height=1, kernel_width=1)

    # 在通道维度上拼接各输出
    outputs = tf.concat([conv1, conv2, conv3, conv4], axis=-1)

    return outputs

def reduction(Inputs,nfilters_11Before33=192,nfilters_11Before55=192,
              nfilters_33=320,nfilters_55=192,nfilters_33After77=192):
    '''
    同样的5×5用两个3×3代替
    :param Inputs:
    :param nfilters_11Before33:
    :param nfilters_11Before77:
    :param nfilters_33:
    :param nfilters_77:
    :param nfilters_33After77:
    :return:
    '''

