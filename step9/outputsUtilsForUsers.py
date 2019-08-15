import numpy as np

def softmax(x):
    '''
    对输出的每一行(即对每个样本的各个标签上的值)做softmax变换
    :param x: 一般是(batchSize,num_Labels), 是模型的输出
    :return: 每一行经过了softmax处理之后的结果
    '''

    #********** Begin **********#

    #********** End **********#


def returnOneHot(Output):
    '''
    softmax的输出不是onehot型的, 将最大概率处替换为1，其他位置均置为0
    :param Output: 神经网络的输出
    :return: onehot型的输出
    '''

    #********** Begin **********#

    #********** End **********#


def computeAccuracy(pred,label):
    '''
    计算预测的正确率
    :param pred: 预测的标签
    :param label:  预测样本真实的标签
    :return: 正确率
    '''

    #********** Begin **********#

    #********** End **********#

