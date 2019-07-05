import cv2
import os

# *****需要取消注释的地方请取消注释*****
def changeSize(BasePath='origin', targetPath='userOutputs'):
    # ********** Begin **********#

    # 请使用os.walk()循环origin目录下的全部文件
    # for下
        #循环阅读当前文件夹下所有文件
        # for
            # 使用os.path.join将该文件的root和filename拼接
            # file =
            # try:
                # 使用cv2.imread读取该文件
                # image =
                # dim = (224, 224)
                # 使用cv2.resize改变图片尺寸为dim
                # resized =
                # if root.endswith('train'):
                #     targetSet = 'train_224'
                # else:
                #     targetSet = 'valid_224'

                # target_path = os.path.join(targetPath, targetSet, fileName)
                # 使用cv2.imwrite将该图片写入目标文件夹


            # except:
                # print(file)
                # print('file error')
                # os.remove(file)

    # ********** End **********#


if __name__ == '__main__':
    changeSize()