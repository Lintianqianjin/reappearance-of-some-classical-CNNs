# 先定义如下函数

# multiChannelWeightLayer
# 使用所学的tf的API, transpose()/expand_dims()/tile()等

# ResNetBlock
# 堆叠两个multiChannelWeightLayer, 然后将输出加上输入

#********** Begin **********#

import tensorflow as tf

def multiChannelWeightLayer(Inputs, batchNormTraining,batchSize):
    '''
    对输入完成BatchNorm + relu + Wx_plus_b操作
    :param Inputs: 输入张量
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''



def ResNetBlock(Inputs, batchNormTraining, batchSize):
    '''
    堆叠两次（BatchNorm + relu + Wx_plus_b）操作，形成一个残差模块
    :param Inputs: 输入张量
    :param batchNormTraining: batchNorm层Training参数，在训练和预测阶段传入不同值
    :return:
    '''


# 首先定义五个placeholder
# BNTraining / keeProb / batchImgInput / labels / InputBatchSize

# 模型结构:
# (tf.layers.conv2d + tf.layers.max_pooling2d)*2 + (ResNetBlock + tf.layers.conv2d)*3
# tf.layers.max_pooling2d + tf.layers.flatten + tf.layers.dense(unit = 256) + tf.nn.dropout + tf.layers.dense

# 损失: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2())
# 优化器: tf.train.AdamOptimizer()


# 将模型保存至step8/userModelInfo文件夹，保存名称为ResNet
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename='step8/userModelInfo/ResNet',
                               graph=tf.get_default_graph())

#********** End **********#
