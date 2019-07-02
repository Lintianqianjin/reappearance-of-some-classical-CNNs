import numpy as np
import keras
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.models import Sequential
import sys

sys.path.append('..\\step3')
sys.path.append('..\\step5')
from generatorCompleted import batchGenerator
from outputsUtils import returnOneHot,computeAccuracy


def VGGPreprocessing(originImgMatrix):
    "The only preprocessing we do is subtracting the mean RGB value, \
    computed on the training set, from each pixel.\
    原论文中对输入的RGB矩阵做了一个减去均值的预处理，该函数实现这个预处理"
    if type(originImgMatrix) is not np.ndarray:
        originImgMatrix = np.ndarray(originImgMatrix)

    # 矩阵X*Y*3
    # axis=0，代表第一维，即把X（行）消除了，所以返回的是每一列RGB的均值，形状是（Y*3）
    # axis=1, 代表第二维，即把Y（列）消除了，所以返回的是全图的RGB的均值，形状是（3，）
    originImgMatrix_RGBMean = np.mean(originImgMatrix,axis=(0,1))

    # 直接减就行
    subtract_Img =originImgMatrix-originImgMatrix_RGBMean

    return subtract_Img
    # example-------------------------------------------------
    # a = np.array([[[1, 2, 3], [4, 5, 6], [1, 1, 1]],
    #               [[7, 8, 9], [10, 11, 12], [2, 2, 2]],
    #               [[13, 14, 15], [16, 17, 18], [0, 3, 6]]])
    # VGGPreprocessing(a)
    # [[[-5. - 5. - 5.]
    #   [-2. - 2. - 2.]
    #   [-5. - 6. - 7.]]
    #
    #  [[1.  1.  1.]
    #     [4.  4.  4.]
    # [-4. - 5. - 6.]]
    #
    # [[7.  7.  7.]
    #  [10. 10. 10.]
    # [-6. - 4. - 2.]]]

def VGGPreprocessingBatch(batch_originImgMatrix):
    for index,img in enumerate(batch_originImgMatrix):
        batch_originImgMatrix[index] = VGGPreprocessing(img)
    return batch_originImgMatrix

def VGGNetBasedKeras(input_shape = (112, 112, 3),output_shape = 4):
    "VGG的结构其实比较简单，也没有特别需要自定义的层，直接用keras搭建会非常简单。"
    model = Sequential()
    # 池化层不指定步长默认步长是不重叠。

    # 第一部分，两层卷积（卷积核3×3，64个）接一层最大池化
    # model.add(AveragePooling2D(pool_size=(2, 2),input_shape=input_shape))
    # model.add(Conv2D(64, (11, 11), strides=(4, 4),  padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二部分，两层卷积（卷积核3×3，128个）接一层最大池化
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第三部分，两层卷积（卷积核3×3，256个）+一层卷积（卷积核1×1，256个） 接一层最大池化
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第四部分，两层卷积（卷积核3×3，512个）+一层卷积（卷积核1×1，512个） 接一层最大池化
    # model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 第五部分，两层卷积（卷积核3×3，512个）+一层卷积（卷积核1×1，512个） 接一层最大池化
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第六部分，三层全连接，有dropout
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(output_shape, activation='softmax'))
    # 原文使用sgd最优化，这里就用更优的adam来优化
    keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':

    G_train = batchGenerator(batchSize = 256,basePath='processed\\train_224')
    G_Valid = batchGenerator(batchSize=80, basePath='processed\\valid_224')
    X_V,Y_V = G_Valid.getBatch()
    # X_V = VGGPreprocessingBatch(X_V)
    acc_Val = []

    model = VGGNetBasedKeras()

    for i in range(1024):
        X,Y = G_train.getBatch()
        # X = VGGPreprocessingBatch(X)
        model.fit(X,Y,batch_size=256)

        if i%8 ==0:
            y_pred = model.predict(X_V)
            cur_acc = computeAccuracy(pred = y_pred, label = Y_V)
            acc_Val.append(cur_acc)

    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(1, len(acc_Val) + 1)], acc_Val, label=u'验证集准确率')
    plt.legend()

    plt.show()
