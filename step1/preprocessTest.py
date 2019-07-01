import os
import filecmp
import random

correctTrainPath = 'processed\\train_224'
correctValidPath = 'processed\\valid_224'
userTrainPath = 'userOutputs\\train_224'
userValidPath = 'userOutputs\\valid_224'

correctTrainFile = os.listdir(correctTrainPath)
correctValidFile = os.listdir(correctValidPath)

# 随机训练集和验证集各抽样10个样本，都一样，肯定就没问题（因为原文件是for循环做的处理）

for i in range(10):
    imgTrainName = random.choice(correctTrainFile)
    imgValidName = random.choice(correctValidFile)

    imgCorrectTrainPath = os.path.join(correctTrainPath, imgTrainName)
    imgCorrectValidPath = os.path.join(correctValidPath, imgValidName)

    imgUserTrainPath = os.path.join(userTrainPath,imgTrainName)
    imgUserValidPath = os.path.join(userValidPath,imgValidName)

    bool_1 = filecmp.cmp(imgCorrectTrainPath, imgUserTrainPath)
    bool_2 = filecmp.cmp(imgCorrectValidPath, imgUserValidPath)

    if not (bool_1 and bool_2):
        print('Wrong')
        break
else:
    print('Right')