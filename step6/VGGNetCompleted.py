import tensorflow as tf

# 卷积层
def Conv(Inputs, kernel_height, kernel_width, num_kernels, stride_h, stride_w, name):
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
        conv = tf.nn.conv2d(Inputs, kernel, strides=[1, stride_h, stride_w, 1], padding='SAME')
        # 添加偏执项
        biases = tf.Variable(tf.constant(0.0, shape=[num_kernels], dtype=tf.float32), trainable=True, name='biases')
        Conv_plus_b = tf.nn.bias_add(conv, biases)

        # 激活
        activation = tf.nn.relu(Conv_plus_b, name=scope)
        return activation


# 全连接层
def Dense(Inputs, num_units, name):
    '''

    :param Inputs: 上一层的输出，该层的输入
    :param num_units: 该层的神经元数
    :param name: 该层的名字
    :return:
    '''

    # 获取输入（已经扁平化了）的长度
    n_in = Inputs.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # 设置权重矩阵
        weights = tf.Variable(tf.truncated_normal(shape=[n_in, num_units], dtype=tf.float32), trainable=True,
                              name='weights')
        # 添加偏执项
        biases = tf.Variable(tf.constant(0.0, shape=[num_units], dtype=tf.float32), trainable=True, name='biases')
        # WX+B
        Wx = tf.matmul(Inputs, weights)
        Wx_plus_b = Wx + biases
        # 激活
        activation = tf.nn.relu(Wx_plus_b, name=scope)

        return activation


# 最大池化层
def maxpool(Inputs, kernel_h, kernel_w, stride_h, stride_w, name):
    '''
    :param Inputs: 上一层的输出，该层的输入
    :param kernel_h: 池化的高
    :param kernel_w: 池化的宽
    :param stride_h: 移动的步长纵向
    :param stride_w: 移动的步长横向
    :param name: 该层名字
    :return:
    '''

    return tf.nn.max_pool(Inputs,
                          # 池化范围
                          ksize=[1, kernel_h, kernel_w, 1],
                          # 移动步长
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name=name)


if __name__ == '__main__':

    # 定义placeholder 开始
    Input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='Imgs')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    Labels = tf.placeholder(shape=(None, 4), dtype=tf.float32, name='Labels')
    # 定义placeholder 结束

    # 第一部分
    Part1_Conv1 = Conv(Input, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=64,
                       name='Part1_Conv1')
    Part1_Conv2 = Conv(Part1_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=64,
                       name='Part1_Conv2')
    Part1_pool1 = maxpool(Part1_Conv2, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part1_pool1")

    # 第二部分
    Part2_Conv1 = Conv(Part1_pool1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=128,
                       name='Part2_Conv1')
    Part2_Conv2 = Conv(Part2_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=128,
                       name='Part2_Conv2')
    Part2_pool1 = maxpool(Part2_Conv2, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part2_pool1")

    # 第三部分
    Part3_Conv1 = Conv(Part2_pool1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=256,
                       name='Part3_Conv1')
    Part3_Conv2 = Conv(Part3_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=256,
                       name='Part3_Conv2')
    Part3_Conv3 = Conv(Part3_Conv2, kernel_height=1, kernel_width=1, stride_h=1, stride_w=1, num_kernels=256,
                       name='Part3_Conv3')
    Part3_pool1 = maxpool(Part3_Conv3, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part3_pool1")

    # 第四部分
    Part4_Conv1 = Conv(Part3_pool1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
                       name='Part4_Conv1')
    Part4_Conv2 = Conv(Part4_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
                       name='Part4_Conv2')
    Part4_Conv3 = Conv(Part4_Conv2, kernel_height=1, kernel_width=1, stride_h=1, stride_w=1, num_kernels=512,
                       name='Part4_Conv3')
    Part4_pool1 = maxpool(Part4_Conv3, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part4_pool1")

    # 针对本任务，一个4分类的问题，其实网络太深不见得好，这里直接省略掉第五层（如下，不用取消注释）
    # Part5_Conv1 = Conv(Part4_pool1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv1')
    # Part5_Conv2 = Conv(Part5_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv2')
    # Part5_Conv3 = Conv(Part5_Conv2, kernel_height=1, kernel_width=1, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv3')
    # Part5_pool1 = maxpool(Part5_Conv3, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part5_pool1")

    # 全连接部分
    # flatten扁平化
    flattened = tf.layers.flatten(Part4_pool1)
    # 同样本任务不宜太宽的网络，该层要求使用512个神经元
    dense1 = Dense(flattened, num_units=512, name='dense1')
    dropout1 = tf.nn.dropout(dense1, keep_prob, name="dropout1")
    # 该层要求使用256个神经元
    dense2 = Dense(dropout1, num_units=256, name='dense2')
    dropout2 = tf.nn.dropout(dense2, keep_prob, name="dropout2")
    dense3 = Dense(dropout2, num_units=4, name='dense3')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense3, labels=Labels))
    train = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "modelInfo/VGGNet")
