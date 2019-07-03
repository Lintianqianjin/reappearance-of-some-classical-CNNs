import numpy as np

def VGGPreprocessing(originImgMatrix):
    "The only preprocessing we do is subtracting the mean RGB value, \
    computed on the training set, from each pixel.\
    原论文中对输入的RGB矩阵做了一个减去均值的预处理，该函数实现这个预处理"
    if type(originImgMatrix) is not np.ndarray:
        originImgMatrix = np.ndarray(originImgMatrix)

    # 矩阵X*Y*3
    # axis=0，代表第一维，即把X（行）消除了，所以返回的是每一列RGB的均值，形状是（Y*3）
    # axis=1, 代表第二维，即把Y（列）消除了，所以返回的是全图的RGB的均值，形状是（3，）
    # todo: 正确完成RGBMean的计算
    # originImgMatrix_RGBMean =

    # 直接减就行 todo: 减一减~
    # subtract_Img =

    # return subtract_Img

def VGGPreprocessingBatch(batch_originImgMatrix):
    for index, img in enumerate(batch_originImgMatrix):
        # todo: 调用VGGPreprocessing 处理该样本的数据,并重新复制到到该样本的索引
        pass
    return batch_originImgMatrix

