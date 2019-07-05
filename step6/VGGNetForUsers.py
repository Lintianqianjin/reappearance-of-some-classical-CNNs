import tensorflow as tf

# todo:按要求补充代码，并且请自行理解后，将部分注释掉的代码取消注释

# 卷积层
def Conv(Inputs, kernel_height, kernel_width, num_kernels, stride_h, stride_w, name):
    "Inputs为上一层的输出，该层的输入；" \
    "kernel_height,kernel_width,num_kernels分别是该层卷积的长\宽\数量" \
    "stride_h, stride_w是该层卷积的移动步长" \
    "name是该层的名字"

    # 获取输入的通道数
    num_channels = Inputs.get_shape()[-1].value

    with tf.name_scope(name) as scope:

        #********** Begin **********#

        # 设置卷积核
        kernel = tf.Variable(tf.truncated_normal(
            # todo: 补充shape参数的值
            # shape=,
            dtype=tf.float32), trainable=True, name='weights')

        # 调用tensorflow的卷积层
        # todo: 补充strides的参数值
        # conv = tf.nn.conv2d(Inputs, kernel, strides=, padding='SAME')
        # 添加偏执项
        # biases = tf.Variable(tf.constant(0.0, shape=[num_kernels], dtype=tf.float32), trainable=True, name='biases')

        # Conv_plus_b = tf.nn.bias_add(conv, biases)

        # 激活
        # activation = tf.nn.relu(Conv_plus_b, name=scope)
        # return activation

        #********** End **********#

# 全连接层
def Dense(Inputs, num_units, name):
    "Inputs为上一层的输出，该层的输入；" \
    "num_units是该层的神经元数" \
    "name是该层的名字"

    # 获取输入（已经扁平化了）的长度
    n_in = Inputs.get_shape()[-1].value
    with tf.name_scope(name) as scope:

        #********** Begin **********#

        # 设置权重矩阵 todo:补充shape参数
        # weights = tf.Variable(tf.truncated_normal(shape=, dtype=tf.float32), trainable=True,
        #                       name='weights')
        # 添加偏执项
        biases = tf.Variable(tf.constant(0.0, shape=[num_units], dtype=tf.float32), trainable=True, name='biases')
        # WX+B
        # Wx = tf.matmul(Inputs, weights)
        # Wx_plus_b = Wx + biases
        # 激活
        # activation = tf.nn.relu(Wx_plus_b, name=scope)

        # return activation

        #********** End **********#

# 最大池化层
def mpool_op(Inputs, kernel_h, kernel_w, stride_h, stride_w, name):
    "Inputs为上一层的输出，该层的输入；" \
    "kernel_h, kernel_w,是池化的高/宽" \
    "stride_h, stride_w是移动的步长" \
    "name是该层的名字"

    #********** Begin **********#

    return tf.nn.max_pool(Inputs,
                          # 池化范围 todo:补充该参数值
                          # ksize=,
                          # 移动步长 todo:补充该参数值
                          # strides=,
                          padding='SAME',
                          name=name)

    #********** End **********#

if __name__ == '__main__':
    # 定义placeholder 开始
    Input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32, name='Imgs')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    Labels = tf.placeholder(shape=(None, 4), dtype=tf.float32, name='Labels')
    # 定义placeholder 结束

    # ********** Begin **********#

    # 第一部分 todo:根据VGG模型补充完整,请不要改变name参数
    # Part1_Conv1 = Conv(Input,
    #                    name='Part1_Conv1')
    # Part1_Conv2 = Conv(Part1_Conv1,
    #                    name='Part1_Conv2')
    # Part1_pool1 = mpool_op(Part1_Conv2, name="Part1_pool1")

    # 第二部分 todo:根据VGG模型补充完整,请不要改变name参数
    # Part2_Conv1 = Conv(Part1_pool1,
    #                    name='Part2_Conv1')
    # Part2_Conv2 = Conv(Part2_Conv1,
    #                    name='Part2_Conv2')
    # Part2_pool1 = mpool_op(Part2_Conv2, name="Part2_pool1")

    # 第三部分 todo:根据VGG模型补充完整,请不要改变name参数
    # Part3_Conv1 = Conv(Part2_pool1,
    #                    name='Part3_Conv1')
    # Part3_Conv2 = Conv(Part3_Conv1,
    #                    name='Part3_Conv2')
    # Part3_Conv3 = Conv(Part3_Conv2,
    #                    name='Part3_Conv3')
    # Part3_pool1 = mpool_op(Part3_Conv3,  name="Part3_pool1")

    # 第四部分 todo:根据VGG模型补充完整,请不要改变name参数
    # Part4_Conv1 = Conv(Part3_pool1,
    #                    name='Part4_Conv1')
    # Part4_Conv2 = Conv(Part4_Conv1,
    #                    name='Part4_Conv2')
    # Part4_Conv3 = Conv(Part4_Conv2,
    #                    name='Part4_Conv3')
    # Part4_pool1 = mpool_op(Part4_Conv3, name="Part4_pool1")

    # 针对本任务，一个4分类的问题，其实网络太深不见得好，这里直接省略掉第五层（如下）
    # Part5_Conv1 = Conv(Part4_pool1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv1')
    # Part5_Conv2 = Conv(Part5_Conv1, kernel_height=3, kernel_width=3, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv2')
    # Part5_Conv3 = Conv(Part5_Conv2, kernel_height=1, kernel_width=1, stride_h=1, stride_w=1, num_kernels=512,
    #                    name='Part4_Conv3')
    # Part5_pool1 = mpool_op(Part5_Conv3, kernel_h=2, kernel_w=2, stride_w=2, stride_h=2, name="Part5_pool1")

    # 全连接部分
    # flatten扁平化 todo:根据提示补充完整,请不要改变name参数
    # flattened = tf.layers.flatten(Part4_pool1)
    # 同样本任务不宜太宽的网络，该层要求使用512个神经元
    # dense1 = Dense(name='dense1')
    # dropout1 = tf.nn.dropout(dense1, keep_prob, name="dropout1")
    # 该层要求使用256个神经元
    # dense2 = Dense(name='dense2')
    # dropout2 = tf.nn.dropout(dense2, keep_prob, name="dropout2")
    # dense3 = Dense( name='dense3')

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense3, labels=Labels))
    # train = tf.train.AdamOptimizer().minimize(loss)

    # ********** End **********#

    #---以下代码不要改动---#
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "userModelInfo/VGGNet.ckpt")
