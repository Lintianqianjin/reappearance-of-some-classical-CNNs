import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import warnings
warnings.filterwarnings('ignore')

import AlexNetCompleted

import AlexNetForUsers

rightModelPath = 'step4/modelInfo/AlexNet'
userModelPath = 'step4/userModelInfo/AlexNet'

# print(os.path.exists(rightModelPath))
# print(os.path.exists(userModelPath))
# print(os.path.getsize(rightModelPath))
# print(os.path.getsize(userModelPath))

try:
    # isRight = IsEqual(rightModelPath, userModelPath)
    # print(isRight)
    if os.path.getsize(rightModelPath)==os.path.getsize(userModelPath):
        print('恭喜你通过本关测试!模型结构正确,你已经掌握了AlexNet的结构!',end='')
    else:
        print('模型结构有误!未能通过本关测试!')
except:
    print('模型结构文件保存有误!未能通过本关测试')