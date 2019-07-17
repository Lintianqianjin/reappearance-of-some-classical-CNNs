import os
import filecmp
import random

correctTrainPath = 'processed\\train_224'

userTrainPath = 'userOutputs\\train_224'


usertTrainFile = os.listdir(userTrainPath)


# 随机训练集和验证集各抽样10个样本，都一样，肯定就没问题（因为原文件是for循环做的处理）

try:
    for i in range(10):
        # 从用户的输出文件中抽取样本，只需要考虑用户输出的训练集
        while True:
            imgUserTrainName = random.choice(usertTrainFile)
            if imgUserTrainName.endswith('_flipx.png'):
                break

        # 获取文件名
        imgUserTrainPath = os.path.join(userTrainPath, imgUserTrainName)


        # 找到正确文件中的对应样本的路径
        imgCorrectTrainPath = os.path.join(correctTrainPath, imgUserTrainName)


        bool = filecmp.cmp(imgCorrectTrainPath, imgUserTrainPath)


        if not bool :
            print('Wrong')
            break
    else:
        print('Right')
except:
    print('Wrong')