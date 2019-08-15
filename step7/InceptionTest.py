import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import warnings
warnings.filterwarnings('ignore')

import InceptionCompleted

import InceptionForUsers


rightModelPath = 'step7/modelInfo/InceptionNet'
userModelPath = 'step7/userModelInfo/InceptionNet'

#
# print(os.path.exists(rightModelPath))
# print(os.path.exists(userModelPath))
# print(os.path.getsize(rightModelPath))
# print(os.path.getsize(userModelPath))


try:

    isRight = os.path.getsize(rightModelPath) ==  os.path.getsize(userModelPath)

    if isRight:
        print('恭喜你通过本关测试!你已掌握Inception各模块结构!',end='')
    else:
        print('未能通过本关测试!模块结构有误!')
except:
    print('未能通过本关测试!模型结构文件没能正确保存!')