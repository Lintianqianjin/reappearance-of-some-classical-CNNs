import os
import filecmp
import random

correctTrainPath = 'processed\\train_224'
correctValidPath = 'processed\\valid_224'
userTrainPath = 'userOutputs\\train_224'
userValidPath = 'userOutputs\\valid_224'

usertTrainFile = os.listdir(userTrainPath)
usertValidFile = os.listdir(userValidPath)

# 随机训练集和验证集各抽样10个样本，都一样，肯定就没问题（因为原文件是for循环做的处理）

try:
    for i in range(10):
        # 从用户的输出文件中抽取样本，一个是用户输出的训练集，一个是用户输出的验证集
        imgUserTrainName = random.choice(usertTrainFile)
        imgUserValidName = random.choice(usertValidFile)
        # 获取对应文件名
        imgUserTrainPath = os.path.join(userTrainPath, imgUserTrainName)
        imgUserValidPath = os.path.join(userValidPath, imgUserValidName)

        # 找到正确文件中的对应样本的路径
        imgCorrectTrainPath = os.path.join(correctTrainPath,imgUserTrainName)
        imgCorrectValidPath = os.path.join(correctValidPath,imgUserValidName)

        bool_1 = filecmp.cmp(imgCorrectTrainPath, imgUserTrainPath)
        bool_2 = filecmp.cmp(imgCorrectValidPath, imgUserValidPath)

        if not (bool_1 and bool_2):
            print('Wrong')
            break
    else:
        print('Right')
except:
    print('Wrong')