import numpy as np
import cv2
import os

def dataAugmentation(BasePath = 'data/rightOutputs/train_224'):

    fileNames = os.listdir(BasePath)
    for fileName in fileNames:
        file = os.path.join(BasePath,fileName)
        namePrex = fileName.split('.')[0]
        image=cv2.imread(file)

        img_fliped_X = cv2.flip(image, 1)
        writePath = os.path.join('data/flipUserOutputs', f'{namePrex}_flipx.png')
        cv2.imwrite(writePath,img_fliped_X)


if __name__ == '__main__':
    dataAugmentation()