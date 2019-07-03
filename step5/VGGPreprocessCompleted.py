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
    originImgMatrix_RGBMean = np.mean(originImgMatrix, axis=(0, 1))

    # 直接减就行
    subtract_Img = originImgMatrix - originImgMatrix_RGBMean

    return subtract_Img


def VGGPreprocessingBatch(batch_originImgMatrix):
    for index, img in enumerate(batch_originImgMatrix):
        batch_originImgMatrix[index] = VGGPreprocessing(img)
    return batch_originImgMatrix


