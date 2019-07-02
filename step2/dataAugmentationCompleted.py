import numpy as np
import cv2
import os

def dataAugmentation(BasePath = '..\\step1\\processed\\train_224'):

    fileNames = os.listdir(BasePath)
    for fileName in fileNames:
        file = os.path.join(BasePath,fileName)
        namePrex = fileName.split('.')[0]
        image=cv2.imread(file)
        dim=(224,224)

        # M = np.array([[1, 0, 0], [0, 1, -50]], dtype=np.float32)
        # img_change = cv2.warpAffine(image, M, dim)
        # writePath = os.path.join(BasePath,f'{namePrex}_up.png')
        # cv2.imwrite(writePath, img_change)
        #
        # M = np.array([[1, 0, 0], [0, 1, 50]], dtype=np.float32)
        # img_change = cv2.warpAffine(image, M, dim)
        # writePath = os.path.join(BasePath, f'{namePrex}_down.png')
        # cv2.imwrite(writePath, img_change)
        #
        # M = np.array([[1, 0, 50], [0, 1, 0]], dtype=np.float32)
        # img_change = cv2.warpAffine(image, M, dim)
        # writePath = os.path.join(BasePath, f'{namePrex}_right.png')
        # cv2.imwrite(writePath, img_change)
        #
        # M = np.array([[1, 0, -50], [0, 1, 0]], dtype=np.float32)
        # img_change = cv2.warpAffine(image, M, dim)
        # writePath = os.path.join(BasePath, f'{namePrex}_left.png')
        # cv2.imwrite(writePath, img_change)

        img_fliped_X = cv2.flip(image, 1)
        writePath = os.path.join(BasePath, f'{namePrex}_flipx.png')
        cv2.imwrite(writePath,img_fliped_X)



if __name__ == '__main__':
    dataAugmentation()