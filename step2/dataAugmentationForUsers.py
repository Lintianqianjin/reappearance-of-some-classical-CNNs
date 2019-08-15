import cv2
import os

def dataAugmentation(BasePath = 'data/rightOutputs/train_224'):
    '''
    只需写水平翻转，翻转后的文件的文件名命名规范为文件名末尾加上 _flipx
    示例 原文件名 “图片1.png”，翻转后保存的文件命名为 “图片1__flipx.png”
    文件保存在'data/flipUserOutputs'目录下
    :param BasePath: 待处理的图片文件夹路径
    :return:
    '''

    #********** Begin **********#


    #********** End **********#


if __name__ == '__main__':
    dataAugmentation()