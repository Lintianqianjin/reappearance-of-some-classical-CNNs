import numpy as np

def VGGPreprocessingBatch(batch_originImgMatrix):
    '''
    你需要对batch中的每一个img的数据作如下预处理:
    各个像素点上rgb三个通道上的值，均减去该图片上三个通道分别的均值
    例如整张img r通道均值为2, g通道均值为1, b通道均值为3
    某像素点为[5,1,0], 则处理后，该像素点为[3,0,-3]

    :param batch_originImgMatrix: 一个数组或者是一个numpy.ndarray，shape是(batchSize,imgSize,imgSize,3)
    :return: 返回处理正确后的数据，shape不变，返回类型为numpy.ndarray
    '''


    #********** Begin **********#


    #********** End **********#


