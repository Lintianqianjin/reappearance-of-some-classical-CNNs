import numpy as np

def softmax(x):
    np.seterr(divide='ignore', invalid='ignore')
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


def returnOneHot(Output):
    '''
    :param Output: 神经网络的输出
    :return:
    '''
    out = np.zeros(Output.shape,dtype=np.int)
    idx = Output.argmax(axis=1)
    out[np.arange(Output.shape[0]), idx] = 1
    return out

def computeAccuracy(pred,label):
    '''
    :param pred: 预测值
    :param label: 实际值
    :return:
    '''
    right = 0
    for p,l in zip(pred,label):
        if (p==l).all():
            right+=1
    return right/len(pred)