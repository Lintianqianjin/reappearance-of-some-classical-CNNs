import tensorflow as tf

# 本关你仅需要按要求搭建网络结构。

# 首先需要定义三个placeholder 分别是dropout的神经元保存比率、每个batch的图片输入、每个batch输入对应的标签

# 然后需要依次堆叠下列网络层：
# 所有卷积使用tf.keras.layers.Conv2D(),
# 所有池化使用tf.keras.layers.MaxPooling2D()
# 所有归一化使用tf.nn.local_response_normalization(), alpha取1e-4, beta取0.75, depth_radius取2, bias取2.0
# 扁平化使用tf.keras.layers.Flatten()
# 全连接使用tf.keras.layers.Dense()
# dropout使用tf.nn.dropout()

# 第一层卷积+归一化+池化
# 卷积层96个卷积核,大小11×11, 步长4，padding为'valid',激活函数relu
# 归一化
# 最大池化范围为3×3，步长为2，padding为valid

# 第二层卷积+归一化+池化
# 这里卷积层256个卷积核,大小5×5, 步长1，padding为'same',激活函数relu
# 归一化
# 池化同上

# 三层直接连接的卷积
# 要求堆叠三个卷积层，卷积核大小均为3×3，步长1，padding为same，激活函数用relu
# 卷积核数分别为192 192 128

# 池化+扁平化

# 第一层全连接+随机失活
# 该层要求512个神经元，激活函数relu

# 第二层全连接+随机失活 要求同上

# 第三层全连接+随机失活
# 该层要求256个神经元，激活函数relu

# 全连接层 输出为类别数量，不需要激活

# 损失: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2())
# 优化器: tf.train.AdamOptimizer()

#********** Begin **********#


#********** End **********#



#---以下代码不要改动---#
#---否则影响测评---#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename="step4/userModelInfo/AlexNet",
                               graph=tf.get_default_graph())