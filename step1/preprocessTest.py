import os
import filecmp

from preprocessForUsers import changeSize

changeSize()

correctTrainPath = 'data/rightOutputs/train_224'
correctValidPath = 'data/rightOutputs/valid_224'
userTrainPath = 'data/userOutputs/train_224'
userValidPath = 'data/userOutputs/valid_224'

usertTrainFile = os.listdir(userTrainPath)
usertValidFile = os.listdir(userValidPath)

if len(usertTrainFile) <= 1:
    print('未能通过本关测试!没有正确生成处理后图片。', end='')
    exit()

if len(usertValidFile) <= 1:
    print('未能通过本关测试!没有正确生成处理后图片。', end='')
    exit()

try:
    for imgUserTrainName in usertTrainFile:

        if imgUserTrainName != '.gitignore':

            # 从用户的输出文件中抽取样本，一个是用户输出的训练集，一个是用户输出的验证集
            # 获取对应文件名
            imgUserTrainPath = os.path.join(userTrainPath, imgUserTrainName)

            # 找到正确文件中的对应样本的路径
            imgCorrectTrainPath = os.path.join(correctTrainPath,imgUserTrainName)

            bool = filecmp.cmp(imgCorrectTrainPath, imgUserTrainPath)


            if not bool:
                print('未能通过本关测试!',end='')
                break
    else:
        for imgUserValidName in usertValidFile:
            if imgUserValidName != '.gitignore':
            # 从用户的输出文件中抽取样本，一个是用户输出的训练集，一个是用户输出的验证集
            # 获取对应文件名
                imgUserValidPath = os.path.join(userValidPath, imgUserValidName)
                imgCorrectValidPath = os.path.join(correctValidPath, imgUserValidName)
                bool = filecmp.cmp(imgCorrectValidPath, imgUserValidPath)

                if not bool:
                    print('未能通过本关测试!', end='')
                    break
        else:
            print('恭喜你通过本关测试!', end='')
            try:
                os.remove('data/userOutputs/valid_224/.gitignore')
                os.remove('data/userOutputs/train_224/.gitignore')
            except:
                pass
#
except:
    print('未能通过本关测试!',end='')