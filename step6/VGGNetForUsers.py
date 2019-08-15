import tensorflow as tf
# 所有的卷积层：卷积核大小为3,步长为1,需要padding,激活函数是relu
# 所有的池化层：为最大池化,池化范围是2,步长也为2,需要padding
# 中间全连接层：激活函数是relu
# 两个隐藏全连接层神经元数分别是512,256
# 全连接层之间需要dropout
# dropout使用tf.nn,其它层全部使用tf.layers

# 损失: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2())
# 优化器: tf.train.AdamOptimizer()

#********** Begin **********#


#********** End **********#

#---以下代码不要改动---#
#---否则影响测评---#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename='step6/userModelInfo/VGGNet',
                               graph=tf.get_default_graph())
