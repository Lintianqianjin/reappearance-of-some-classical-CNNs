import cv2
import os

def changeSize(BasePath='data/testSet', targetPath='data/userOutputs'):
    '''
    请使用os.walk()循环origin目录下的全部文件,使用cv2改变图片尺寸为(224,224)
    BasePath中文件夹有两个，分别是‘train’ 和 'valid'
    targetPath中文件夹有两个，分别是'train_224'和'valid_224'
    图片名可以不变
    :param BasePath:原始图片所在路径，类型为str
    :param targetPath:处理后图片保存的路径，类型为str
    :return:无
    '''

    #********** Begin **********#

    #********** End **********#


if __name__ == '__main__':
    changeSize()