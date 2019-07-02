import numpy as np

def softmax(x):
    np.seterr(divide='ignore', invalid='ignore')
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


def returnOneHot(NNOutput):
    out = np.zeros(NNOutput.shape)
    idx = NNOutput.argmax(axis=1)
    out[np.arange(NNOutput.shape[0]), idx] = 1
    return out

def computeAccuracy(pred,label):
    right = 0
    for p,l in zip(pred,label):
        if (p==l).all():
            right+=1
    return right/len(pred)