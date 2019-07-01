import cv2
import os

def changeSize(BasePath = 'origin',targetPath = 'userOutputs'):

    for root, dirs, files in os.walk(BasePath):
        for fileName in files:
            file=os.path.join(root,fileName)
            try:
                image=cv2.imread(file)
                dim=(224,224)
                resized=cv2.resize(image,dim)
                if root.endswith('train'):
                    targetSet = 'train_224'
                else:
                    targetSet = 'valid_224'

                target_path=os.path.join(targetPath,targetSet,fileName)
                cv2.imwrite(target_path,resized)

            except:
                print(file)
                print('file error')
                os.remove(file)

if __name__ == '__main__':
    changeSize()