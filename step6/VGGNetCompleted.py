import tensorflow as tf

# def generateModel():
    #----以下是答案部分 begin----#

# 定义placeholder 开始
Input = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32, shape=())
Labels = tf.placeholder(shape=(None, 4), dtype=tf.float32)
# 定义placeholder 结束

# 第一部分
Part1_Conv1 = tf.layers.conv2d(inputs=Input,filters=64, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part1_Conv2 = tf.layers.conv2d(inputs=Part1_Conv1, filters=64, kernel_size=3 ,strides=1,padding='same',activation=tf.nn.relu)
Part1_pool1 = tf.layers.max_pooling2d(Part1_Conv2, pool_size=2, strides=2,padding='same')

# 第二部分
Part2_Conv1 = tf.layers.conv2d(inputs=Part1_pool1, filters=128, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part2_Conv2 = tf.layers.conv2d(inputs=Part2_Conv1, filters=128, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part2_pool1 = tf.layers.max_pooling2d(Part2_Conv2, pool_size=2, strides=2,padding='same')
#
# 第三部分
Part3_Conv1 = tf.layers.conv2d(inputs=Part2_pool1, filters=256, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part3_Conv2 = tf.layers.conv2d(inputs=Part3_Conv1, filters=256, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part3_Conv3 = tf.layers.conv2d(inputs=Part3_Conv2, filters=256, kernel_size=1,strides=1,padding='same',activation=tf.nn.relu)
Part3_pool1 = tf.layers.max_pooling2d(Part3_Conv3, pool_size=2, strides=2,padding='same')

# 第四部分
Part4_Conv1 = tf.layers.conv2d(inputs=Part3_pool1, filters=512, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part4_Conv2 = tf.layers.conv2d(inputs=Part4_Conv1, filters=512, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part4_Conv3 = tf.layers.conv2d(inputs=Part4_Conv2, filters=512, kernel_size=1,strides=1,padding='same',activation=tf.nn.relu)
Part4_pool1 = tf.layers.max_pooling2d(Part4_Conv3, pool_size=2, strides=2,padding='same')

Part5_Conv1 = tf.layers.conv2d(inputs=Part4_pool1, filters=512, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part5_Conv2 = tf.layers.conv2d(inputs=Part5_Conv1, filters=512, kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
Part5_Conv3 = tf.layers.conv2d(inputs=Part5_Conv2, filters=512, kernel_size=1,strides=1,padding='same',activation=tf.nn.relu)
Part5_pool1 = tf.layers.max_pooling2d(Part5_Conv3, pool_size=2, strides=2,padding='same')

# 全连接部分
# flatten扁平化
flattened = tf.layers.flatten(Part5_pool1)
# 同样本任务不宜太宽的网络，该层要求使用512个神经元
dense1 = tf.layers.dense(flattened, units=512,activation=tf.nn.relu)
dropout1 = tf.nn.dropout(dense1, keep_prob)
# 该层要求使用256个神经元
dense2 = tf.layers.dense(dropout1, units=256,activation=tf.nn.relu)
dropout2 = tf.nn.dropout(dense2, keep_prob)
outputs = tf.layers.dense(dropout2, units=4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=Labels))
train = tf.train.AdamOptimizer().minimize(loss)

#----以上是答案部分 end----#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.save(sess, "modelInfo/VGGNet")
    tf.train.export_meta_graph(filename="step6/modelInfo/VGGNet",
                               graph=tf.get_default_graph())
tf.reset_default_graph()
# if __name__ == '__main__':
#     generateModel()