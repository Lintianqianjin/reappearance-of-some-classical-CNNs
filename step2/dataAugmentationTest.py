import os
import filecmp
from dataAugmentationForUsers import dataAugmentation
from subprocess import *

dataAugmentation()
# 执行用户完成的脚本
# call('python step2/dataAugmentationForUsers.py')

correctTrainPath = 'data/flipRightOutputs'

userTrainPath = 'data/flipUserOutputs'


correctTrainFile = os.listdir(correctTrainPath)


# 比对全部水平翻转的图片是否与正确翻转的图片相同

try:
    # 从用户的输出文件中抽取样本，只需要考虑用户输出的训练集
    filelist = [flipimg for flipimg in correctTrainFile]
    if filelist:
        for imgUserTrainName in filelist:
            # 获取文件名
            imgUserTrainPath = os.path.join(userTrainPath, imgUserTrainName)

            # 找到正确文件中的对应样本的路径
            imgCorrectTrainPath = os.path.join(correctTrainPath, imgUserTrainName)


            bool = filecmp.cmp(imgCorrectTrainPath, imgUserTrainPath)

            if not bool :
                print(imgCorrectTrainPath)
                print(imgUserTrainPath)
                print('未能通过本关测试!产生的图片不相同！',end='')
                break

        else:
            print('恭喜你通过本关测试!',end='')
    else:
        print('未能通过本关测试!没有产生翻转文件',end='')
except:
    print('未能通过本关测试!读取图片出错！',end='')