import numpy as np
import os
import cv2


class batchGenerator:
    '''
    训练数据生成器,需要你完成下面两个函数
    '''

    def __init__(self, basePath='data/processed/train_224/', batchSize=256):
        '''
        数据集中有四类图片分别是'bus','family sedan','fire engine','racing car',
        每个图片的文件名形式为"XXX (id).jpg"或"XXX (id)_flipx.jpg",例如"bus (1).jpg","bus (1)_flipx.jpg"

        :param basePath:数据集路径
        :param batchSize: 每次获取的图片数量
        '''
        #********** Begin **********#


        #********** End **********#

    def getBatch(self):
        '''
        循环遍历数据集
        可以通过分割文件名获取所属类别，然后你需要将类别转为onehot类型的表示形式，例如[0,0,1,0]代表'fire engine'
        如果一次循环最后剩余样本数不到bactchSize,仅返回剩余全部样本即可

        :return: 批图片数据(batchSize,224,224,3)，与每个图片对应的标签(batchSize,4)
        '''
        #********** Begin **********#



        #********** End **********#