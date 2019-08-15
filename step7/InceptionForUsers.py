# 先定义如下函数,是Inception的一些结构模块

# Inception module 1: Inception_traditional

# Inception module 2: Inception_AsymmetricConv

# Inception module 3: Inception_parallelAsymmetricConv

# reduction

# InitialPart

#********** Begin **********#

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


# 模型初始的部分
def InitialPart(Inputs):
    '''
    论文模型中在使用Inception模块之前还是正常的一些卷积和池化
    :param Inputs: 初始img输入
    :return:
    '''


# 首先需要定义三个placeholder: Input,keep_prob,Labels

# 模型结构:
# InitialPart + Inception_traditional + reduction+Inception_AsymmetricConv + reduction+Inception_parallelAsymmetricConv
# tf.layers.average_pooling2d + tf.layers.flatten + tf.nn.dropout + tf.layers.dense

# 损失: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2())
# 优化器: tf.train.AdamOptimizer()

# 保存至step7/userModelInfo文件夹，保存名称为InceptionNet

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.export_meta_graph(filename='step7/userModelInfo/InceptionNet',
                               graph=tf.get_default_graph())

#********** End **********#
