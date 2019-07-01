import keras
from keras import Sequential
from keras.layers import *
import sys
sys.path.append('..\\step2')

from generatorForCatDog import batchGenerator

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
# AlexNet
model = Sequential()
#第一段
model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', input_shape=(224,224,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#第二段
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#第三段
model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#第四段
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

G = batchGenerator(batchSize=64)
# G_v = batchGenerator(batchSize=80,basePath='D:\\educoderFile\\vehicle\\valid_224\\')

acc_Val=[]
for i in range(1024):
    X, Y = G.getBatch()
    model.fit(X,Y,epochs=1,batch_size=32,verbose=0)

    if i%8==0:
        X_v, Y_v = G.getBatch()
        output_v = model.predict(X_v)
        output_v = returnOneHot(output_v)
        acc_v = computeAccuracy(output_v, Y_v)
        acc_Val.append(acc_v)
        print(f'current accuracy: {acc_v}')


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot([i for i in range(1, len(acc_Val) + 1)], acc_Val, label=u'训练损失')
plt.legend()
# todo: 更换模型时要改名字
plt.show()
